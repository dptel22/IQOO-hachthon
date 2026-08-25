"""
generate_dataset.py — Synthetic Payment Failure Dataset Generator
================================================================

Outputs: synthetic_payment_failures.jsonl  (500 lines, one JSON event per line)
         synthetic_payment_failures_train.jsonl  (400 lines)
         synthetic_payment_failures_test.jsonl   (100 lines)

Random seed: 42  (documented for reproducibility)

Label-generation rationale
-------------------------
The `recovered` label is NOT a hard boolean AND of simple rules. It is sampled
from a Bernoulli distribution whose probability is computed via a weighted-linear
combination of features passed through a sigmoid. This ensures a logistic
regression model actually has something non-trivial to learn — the interaction
effects create recovery patterns that no simple lookup table can capture perfectly.

Coefficient reference (used in architecture doc / video script):
----------------------------------------------------------------
Base weights by error category (log-odds space):
  temporary failures:   +1.8   (gateway/processing — transient, retry-friendly)
  insufficient_funds:   +0.2   (may succeed on retry if customer tops up)
  card_declined:        -0.8   (hard decline, retry rarely helps)
  auth_failed:          -1.5   (3DS / OTP — customer must act)
  expired_card:         -2.0   (terminal — card is dead)
  invalid requests:     -3.0   (will always fail — code/config bug)

Penalties:
  retry_penalty:        -0.35  per prior retry (cumulative attrition)
  amount_penalty:       -0.25  * log(amount)    (larger amounts recover less)

Interaction terms:
  1. payment_method * amount:  UPI amounts > ₹5000 recover slightly better
     (upi_amount_interaction = +0.4 if method=upi AND amount > 5000, else 0)
  2. hour_of_day * error_category: overnight (0–5am) gateway errors recover
     worse than daytime (peak_hour_interaction = +0.5 if 8<=hour<=20 AND
     category=temporary, else 0; -0.4 if hour<=5 AND category=temporary)
  3. payment_method * prior_retry_count: UPI retries above 3 recover worse
     than card retries at same count (upi_retry_interaction = -0.5 if
     method=upi AND prior_retry_count >= 3, else 0)
  4. amount * error_category: high-value temporary failures recover worse
     (high_value_temp_interaction = -0.35 if amount > 10000 AND
     category=temporary, else 0)

Label noise: ~18% of labels are randomly flipped AFTER Bernoulli sampling.
  This is deliberately large — it ensures a hard-rule model visibly
  underperforms a model that learns the true latent surface.

Error-code distribution targets:
  temporary (gateway_error, processing_failed):          40%  → 200/500
  card declines (declined, insufficient_funds):          30%  → 150/500
  auth failures (authentication_failed):                 15%  →  75/500
  invalid requests (invalid_amount, invalid_currency):   10%  →  50/500
  expired card (expired_card):                            5%  →  25/500
"""

import json
import random
import math
import uuid
from datetime import datetime, timedelta, timezone

# ── Configuration ──────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_EVENTS = 500
TRAIN_SIZE = 400
TEST_SIZE = 100
LABEL_NOISE_RATE = 0.18  # 18% label flip rate

# ── Error-code taxonomy ───────────────────────────────────────────────────────
# Maps each code to (category, human-readable reason from Razorpay docs)

ERROR_CODES = {
    # Temporary failures — 40%
    "payment_gateway_error":           ("temporary",         "The payment gateway returned an unexpected error; retrying may succeed"),
    "payment_processing_failed":      ("temporary",         "Payment processing failed due to a transient issue; retry the payment"),
    # Card declines — 15%
    "payment_declined":               ("card_declined",     "The payment was declined by the issuing bank; contact your bank"),
    # Insufficient funds — 15%
    "payment_insufficient_funds":     ("insufficient_funds","Insufficient funds in the customer's account to complete the payment"),
    # Auth failures — 15%
    "payment_authentication_failed":  ("auth_failed",       "Payment authentication failed; the customer could not complete 3DS / OTP verification"),
    # Invalid requests — 10%  (invalid_amount + invalid_currency)
    "payment_invalid_amount":         ("invalid_request",   "The payment amount is invalid or outside allowed limits"),
    "payment_invalid_currency":       ("invalid_request",   "The currency code is not supported for this payment method"),
    # Expired card — 5%
    "payment_expired_card":           ("expired_card",      "The card used for the payment has expired; update card details"),
    # Extra codes in same categories for variety (not new categories)
    "payment_failed":                 ("temporary",         "The payment failed due to an unspecified error; retrying may succeed"),
    "payment_authorization_failed":   ("card_declined",     "Authorization was denied by the issuing bank"),
    "payment_capture_failed":         ("temporary",         "Payment was authorized but capture failed; retrying capture may succeed"),
}

# Weighted distribution: (category, fraction)
CATEGORY_WEIGHTS = [
    ("temporary",        0.40),   # 200/500
    ("card_declined",    0.15),   #  75/500
    ("insufficient_funds", 0.15), #  75/500
    ("auth_failed",      0.15),   #  75/500
    ("invalid_request",  0.10),   #  50/500
    ("expired_card",     0.05),   #  25/500
]

# Codes within each category for weighted random pick
CODES_BY_CATEGORY = {
    "temporary":         ["payment_gateway_error", "payment_processing_failed",
                          "payment_failed", "payment_capture_failed"],
    "card_declined":     ["payment_declined", "payment_authorization_failed"],
    "insufficient_funds":["payment_insufficient_funds"],
    "auth_failed":       ["payment_authentication_failed"],
    "invalid_request":   ["payment_invalid_amount", "payment_invalid_currency"],
    "expired_card":      ["payment_expired_card"],
}

# ── Coefficients for latent score ─────────────────────────────────────────────

BASE_WEIGHTS = {
    "temporary":         1.8,
    "insufficient_funds": 0.2,
    "card_declined":    -0.8,
    "auth_failed":      -1.5,
    "expired_card":     -2.0,
    "invalid_request":  -3.0,
}

RETRY_PENALTY  = -0.35   # per prior retry count
AMOUNT_PENALTY = -0.25   # * log(amount)

# Payment methods for sampling
PAYMENT_METHODS = ["card", "upi", "netbanking", "wallet"]
PAYMENT_METHOD_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

# ── Sigmoid ───────────────────────────────────────────────────────────────────

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

# ── Latent score computation ──────────────────────────────────────────────────

def compute_latent_score(
    category: str,
    amount: float,
    prior_retry_count: int,
    payment_method: str,
    hour_of_day: int,
) -> float:
    """
    Compute the latent recovery score (log-odds space) using main effects
    and 4 interaction terms. Pass through sigmoid to get p_recover.
    """
    score = BASE_WEIGHTS[category]
    score += RETRY_PENALTY * prior_retry_count
    score += AMOUNT_PENALTY * math.log(amount + 1)

    # ── Interaction term 1: payment_method × amount ──────────────────────
    # UPI amounts > ₹5000 recover slightly better (common top-up pattern)
    if payment_method == "upi" and amount > 5000:
        score += 0.4

    # ── Interaction term 2: hour_of_day × error_category ─────────────────
    # Overnight gateway errors (0–5am) recover worse; daytime (8am–8pm) better
    if category == "temporary" and 0 <= hour_of_day <= 5:
        score -= 0.4
    elif category == "temporary" and 8 <= hour_of_day <= 20:
        score += 0.5

    # ── Interaction term 3: payment_method × prior_retry_count ───────────
    # UPI retries above 3 recover worse than card retries at same count
    if payment_method == "upi" and prior_retry_count >= 3:
        score -= 0.5

    # ── Interaction term 4: amount × error_category ──────────────────────
    # High-value temporary failures recover worse (bank-side holds)
    if amount > 10000 and category == "temporary":
        score -= 0.35

    return score

# ── Sampling helpers ──────────────────────────────────────────────────────────

def sample_category(rng: random.Random) -> str:
    r = rng.random()
    cumulative = 0.0
    for cat, weight in CATEGORY_WEIGHTS:
        cumulative += weight
        if r <= cumulative:
            return cat
    return CATEGORY_WEIGHTS[-1][0]

def sample_code_for_category(rng: random.Random, category: str) -> str:
    codes = CODES_BY_CATEGORY[category]
    return rng.choice(codes)

def sample_event(rng: random.Random, event_index: int) -> dict:
    """Generate one synthetic payment failure event."""
    # Timestamp: random within last 30 days
    now = datetime.now(timezone.utc)
    offset = timedelta(seconds=rng.randint(0, 30 * 24 * 3600))
    ts = now - offset
    hour_of_day = ts.hour

    # Category and error code
    category = sample_category(rng)
    error_code = sample_code_for_category(rng, category)
    error_reason = ERROR_CODES[error_code][1]

    # Amount: lognormal, clamped ₹100–₹50,000
    # Use lognormal centered so median ~₹1500, with right tail
    raw_amount = rng.lognormvariate(math.log(1500), 0.8)
    amount = max(100.0, min(50000.0, round(raw_amount, 2)))

    # Force ~10% to exceed ₹10,000 by bumping if needed
    # (lognormal with these params gives ~8-12% naturally, but enforce it)
    if event_index < int(N_EVENTS * 0.10):
        # First 10% of events get high amounts (will be shuffled later)
        amount = max(amount, rng.uniform(10001.0, 50000.0))
        amount = round(amount, 2)

    # Payment method
    payment_method = rng.choices(PAYMENT_METHODS, weights=PAYMENT_METHOD_WEIGHTS, k=1)[0]

    # Prior retry count: geometric p=0.5, capped at 5
    # Python's random doesn't have geometric directly; use negative binomial approach
    prior_retry_count = min(5, int(math.floor(math.log(rng.random()) / math.log(1 - 0.5))))

    # Latent score and label
    latent = compute_latent_score(category, amount, prior_retry_count,
                                  payment_method, hour_of_day)
    p_recover = sigmoid(latent)
    recovered = int(rng.random() < p_recover)

    # Customer ID
    cust_id = "cust_" + "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=8))

    return {
        "event_id": f"evt_{uuid.UUID(int=rng.getrandbits(128), version=4)}",
        "timestamp": ts.isoformat(),
        "amount": amount,
        "currency": "INR",
        "customer_id": cust_id,
        "payment_method": payment_method,
        "error_code": error_code,
        "error_reason": error_reason,
        "hour_of_day": hour_of_day,
        "prior_retry_count": prior_retry_count,
        "recovered": recovered,
        "metadata": {
            "category": category,
            "latent_score": round(latent, 4),
            "p_recover": round(p_recover, 4),
        },
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(RANDOM_SEED)

    # Generate all events
    events = [sample_event(rng, i) for i in range(N_EVENTS)]

    # ── Force ~10% high-value (amount > ₹10,000) ──────────────────────────
    # Shuffle first 10% indices so high amounts are spread across the dataset
    high_value_indices = list(range(int(N_EVENTS * 0.10)))
    rng.shuffle(high_value_indices)

    for idx in high_value_indices:
        if events[idx]["amount"] <= 10000:
            events[idx]["amount"] = round(rng.uniform(10001.0, 50000.0), 2)
            # Recompute latent score with new amount
            cat = events[idx]["metadata"]["category"]
            amt = events[idx]["amount"]
            prc = events[idx]["prior_retry_count"]
            pm = events[idx]["payment_method"]
            hod = events[idx]["hour_of_day"]
            new_latent = compute_latent_score(cat, amt, prc, pm, hod)
            new_p = sigmoid(new_latent)
            events[idx]["recovered"] = int(rng.random() < new_p)
            events[idx]["metadata"]["latent_score"] = round(new_latent, 4)
            events[idx]["metadata"]["p_recover"] = round(new_p, 4)

    # ── Apply label noise (18% flip rate) ──────────────────────────────────
    n_flipped = 0
    for ev in events:
        if rng.random() < LABEL_NOISE_RATE:
            ev["recovered"] = 1 - ev["recovered"]
            n_flipped += 1

    # ── Shuffle and split ──────────────────────────────────────────────────
    rng.shuffle(events)
    train_set = events[:TRAIN_SIZE]
    test_set  = events[TRAIN_SIZE:]

    # ── Write full dataset ─────────────────────────────────────────────────
    with open("synthetic_payment_failures.jsonl", "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    # ── Write train/test splits ────────────────────────────────────────────
    with open("synthetic_payment_failures_train.jsonl", "w") as f:
        for ev in train_set:
            f.write(json.dumps(ev) + "\n")

    with open("synthetic_payment_failures_test.jsonl", "w") as f:
        for ev in test_set:
            f.write(json.dumps(ev) + "\n")

    # ── Validation Gate ────────────────────────────────────────────────────
    print("=" * 65)
    print("PHASE 1 GATE — Dataset Validation")
    print("=" * 65)

    # 1. Class balance
    n_recovered = sum(e["recovered"] for e in events)
    pct_recovered = n_recovered / len(events) * 100
    print(f"\nClass balance: recovered=1 → {n_recovered}/{len(events)} ({pct_recovered:.1f}%)")
    print(f"  recovered=0 → {len(events) - n_recovered}/{len(events)} ({100 - pct_recovered:.1f}%)")
    assert 25 <= pct_recovered <= 75, f"Class balance looks off: {pct_recovered:.1f}%"
    print("  ✓ Class balance looks reasonable (not extreme)")

    # 2. High-value amount check
    n_high = sum(1 for e in events if e["amount"] > 10000)
    pct_high = n_high / len(events) * 100
    print(f"\nHigh-value (>₹10,000): {n_high}/{len(events)} ({pct_high:.1f}%)")
    assert 7 <= pct_high <= 15, f"High-value % outside 7-15% range: {pct_high:.1f}%"
    print("  ✓ ~10% target met")

    # 3. Error-code category distribution
    cat_counts = {}
    for ev in events:
        cat = ev["metadata"]["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print("\nError-code category distribution:")
    for cat, target_frac in CATEGORY_WEIGHTS:
        actual = cat_counts.get(cat, 0)
        actual_pct = actual / len(events) * 100
        target_pct = target_frac * 100
        marker = "✓" if abs(actual_pct - target_pct) < 8 else "✗"
        print(f"  {cat:20s}: {actual:3d} ({actual_pct:5.1f}%)  target ~{target_pct:.0f}%  {marker}")

    # 4. Quick correlation check — no single feature should perfectly separate classes
    print("\nSeparability check (no single feature should perfectly split classes):")
    for feat_name, get_val in [
        ("error_code_category", lambda e: e["metadata"]["category"]),
        ("payment_method",      lambda e: e["payment_method"]),
    ]:
        groups = {}
        for ev in events:
            key = get_val(ev)
            groups.setdefault(key, []).append(ev["recovered"])
        perfect = True
        for val, labels in groups.items():
            if all(l == 0 for l in labels) or all(l == 1 for l in labels):
                perfect = False
                print(f"  ⚠ {feat_name}={val}: perfectly separable (all {labels[0]})")
        if perfect:
            print(f"  ✓ {feat_name}: no perfect separation")

    # Also check amount as a numeric feature
    amounts_rec  = [e["amount"] for e in events if e["recovered"] == 1]
    amounts_nrec = [e["amount"] for e in events if e["recovered"] == 0]
    min_rec, max_rec = min(amounts_rec), max(amounts_rec)
    min_nrec, max_nrec = min(amounts_nrec), max(amounts_nrec)
    print(f"  amount ranges — recovered: [{min_rec:.0f}, {max_rec:.0f}], "
          f"not-recovered: [{min_nrec:.0f}, {max_nrec:.0f}]")
    if max_nrec < min_rec or max_rec < min_nrec:
        print("  ⚠ amount ranges are disjoint — would be trivially separable")
    else:
        print("  ✓ amount: ranges overlap — not trivially separable")

    # 5. Label noise applied
    print(f"\nLabel noise applied: {n_flipped} labels flipped ({n_flipped/len(events)*100:.1f}%)")
    print(f"  Target: ~{LABEL_NOISE_RATE*100:.0f}%")

    # 6. Split sizes
    print(f"\nSplit: train={len(train_set)}, test={len(test_set)}")
    print(f"  Written to: synthetic_payment_failures_train.jsonl / test.jsonl")

    print("\n" + "=" * 65)
    print("PHASE 1 COMPLETE — Ready for Phase 2 (Detection Module)")
    print("=" * 65)

if __name__ == "__main__":
    main()
