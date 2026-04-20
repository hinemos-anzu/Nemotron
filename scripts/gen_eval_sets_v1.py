#!/usr/bin/env python3
"""
Generate S1 evaluation foundation assets v1.
Produces: quick_gate_v1.jsonl, diagnostic_v1.jsonl, promotion_v1.jsonl,
          category_manifest_v1.csv
All sample answers are PENDING_BASELINE_RUN until S1.5 inference is executed.
"""
import json, csv, os, random
from pathlib import Path

REPO = Path(__file__).parent.parent
DATA_DIR = REPO / "data" / "eval"
for d in [DATA_DIR, REPO / "reports" / "eval", REPO / "logs" / "experiments"]:
    d.mkdir(parents=True, exist_ok=True)

# ─── problem generators ──────────────────────────────────────────────────────

def _r(seed):
    r = random.Random(seed)
    return r

def gen_numeral(seed):
    r = _r(seed)
    t = r.randint(0, 6)
    if t == 0:
        n = r.randint(10, 127)
        return f"Convert {n} (base 10) to binary (base 2). Give the result without leading zeros."
    elif t == 1:
        b = format(r.randint(1, 63), '06b')
        return f"What is the decimal value of the binary number {b}?"
    elif t == 2:
        a, m = r.randint(50, 500), r.choice([7, 11, 13, 17, 19, 23])
        return f"What is {a} mod {m}?"
    elif t == 3:
        n = r.randint(4, 12)
        return f"What is 2^{n}?"
    elif t == 4:
        a, b = r.randint(1, 9), r.choice([4, 8, 16, 25, 32])
        return f"Express {a}/{b} as a decimal. If it terminates, give the exact decimal. If repeating, write the repeating block in parentheses."
    elif t == 5:
        n = r.randint(100, 255)
        return f"Convert {n} (decimal) to hexadecimal. Use uppercase letters."
    else:
        a, b = r.choice([12, 18, 24, 36, 48, 60]), r.choice([8, 10, 16, 20, 30])
        return f"What is the GCD of {a} and {b}?"

def gen_unit_conversion(seed):
    r = _r(seed)
    t = r.randint(0, 6)
    v = round(r.uniform(1.5, 120.0), 1)
    if t == 0:
        return f"Convert {v} km/h to m/s. Round to 2 decimal places."
    elif t == 1:
        return f"Convert {round(v, 0):.0f} kg to grams."
    elif t == 2:
        temp = round(r.uniform(-30.0, 120.0), 1)
        return f"Convert {temp} °C to Kelvin."
    elif t == 3:
        return f"Convert {v} miles to kilometers. (1 mile = 1.60934 km) Round to 2 decimal places."
    elif t == 4:
        return f"Convert {round(v / 10, 2)} liters to milliliters."
    elif t == 5:
        s = r.randint(65, 3600)
        return f"Express {s} seconds as minutes and seconds (format: M min S sec)."
    else:
        return f"Convert {round(v / 10, 2)} feet to meters. (1 foot = 0.3048 m) Round to 3 decimal places."

def gen_gravity(seed):
    r = _r(seed)
    t = r.randint(0, 4)
    h = round(r.uniform(5.0, 80.0), 1)
    m = r.choice([2, 5, 10, 20, 50])
    if t == 0:
        return f"An object falls from rest at height h = {h} m. What is its speed just before impact? (g = 9.8 m/s², round to 2 decimal places)"
    elif t == 1:
        return f"An object is dropped from h = {h} m. How long does it take to reach the ground? (g = 9.8 m/s², round to 2 decimal places)"
    elif t == 2:
        return f"What is the gravitational potential energy of a {m} kg object at height {h} m? (g = 9.8 m/s²)"
    elif t == 3:
        v0 = r.choice([5, 10, 15, 20])
        return f"A {m} kg object is thrown upward at {v0} m/s. How high does it reach? (g = 9.8 m/s², round to 2 decimal places)"
    else:
        return f"A {m} kg object is dropped from {h} m. What is its kinetic energy just before impact? (g = 9.8 m/s²)"

def gen_cipher(seed):
    r = _r(seed)
    words = ["HELLO", "WORLD", "PYTHON", "CIPHER", "KAGGLE", "NVIDIA", "MODEL",
             "REASON", "AGENT", "LOGIC", "SIGNAL", "TOKEN", "LAYER", "CHAIN"]
    word = r.choice(words)
    shift = r.choice([3, 7, 11, 13, 17, 21])
    encoded = "".join(chr((ord(c) - 65 + shift) % 26 + 65) if c.isalpha() else c for c in word)
    t = r.randint(0, 3)
    if t == 0:
        return f"Decode '{encoded}' using a Caesar cipher with shift {shift}. Answer in uppercase."
    elif t == 1:
        return f"Encode '{word}' using a Caesar cipher with shift {shift}. Answer in uppercase."
    elif t == 2:
        return f"Apply ROT13 to '{word}'. What is the result? Answer in uppercase."
    else:
        return f"The string '{encoded}' was Caesar-encoded with shift {shift}. Reverse it to get the original word."

def gen_equation(seed):
    r = _r(seed)
    t = r.randint(0, 8)
    a = r.choice([1, 2, 3, 4, 5])
    b = r.choice([-7, -5, -3, -1, 1, 3, 5, 7, 9])
    c = r.choice([2, 4, 6, 8, 10, 12, 14, 16])
    A = r.choice([1, 2, 3])
    B = r.choice([-6, -5, -3, 1, 3, 5, 7])
    C = r.choice([-4, -2, 2, 4, 6])
    if t == 0:
        return f"Solve for x: {a}x + {b} = {c}"
    elif t == 1:
        rhs = a * c + b
        return f"If {a}x + {b} = {rhs}, what is x?"
    elif t == 2:
        n = r.choice([2, 4, 8, 16, 32, 64, 128])
        exp = int(round(r.uniform(1, 4)))
        return f"Find x: log₂({n}) = x"
    elif t == 3:
        return f"Simplify: ({a}x² + {b}x) / x, for x ≠ 0."
    elif t == 4:
        return f"Solve: {A}x² + {B}x + {C} = 0. List all real roots, separated by commas."
    elif t == 5:
        x1, x2 = r.choice([1, 2, 3, -1, -2]), r.choice([4, 5, 6, -3, -4])
        return f"Expand (x - {x1})(x - {x2})."
    elif t == 6:
        p1 = (r.randint(-3, 3), r.randint(-3, 3))
        p2 = (r.randint(1, 6), r.randint(1, 8))
        return f"What is the slope of the line passing through ({p1[0]}, {p1[1]}) and ({p2[0]}, {p2[1]})?"
    elif t == 7:
        return f"If f(x) = {a}x² + {b}x + {c}, evaluate f({r.choice([0, 1, 2, -1, 3])})."
    else:
        return f"Factor completely: x² + {b+a}x + {a*b}. (Assume integer roots)"

def gen_bit_manipulation(seed):
    r = _r(seed)
    t = r.randint(0, 13)
    a = r.randint(15, 200)
    b = r.randint(15, 200)
    bits4 = format(r.randint(1, 14), '04b')
    bits4b = format(r.randint(1, 14), '04b')
    shift = r.choice([1, 2])
    bit_pos = r.randint(0, 6)
    if t == 0:
        return f"What is {a} XOR {b}? Give the answer in decimal."
    elif t == 1:
        return f"What is {a} AND {b}? Give the answer in decimal."
    elif t == 2:
        return f"What is {a} OR {b}? Give the answer in decimal."
    elif t == 3:
        s = r.choice([1, 2, 3])
        return f"What is {a} left-shifted by {s} bits? Give the answer in decimal."
    elif t == 4:
        s = r.choice([1, 2, 3])
        return f"What is {a} right-shifted by {s} bits (logical)? Give the answer in decimal."
    elif t == 5:
        return f"Apply left circular rotation by {shift} bit(s) to the 4-bit binary number {bits4}. Give the result as a 4-bit binary string."
    elif t == 6:
        return f"Apply right circular rotation by {shift} bit(s) to the 4-bit binary number {bits4}. Give the result as a 4-bit binary string."
    elif t == 7:
        return f"How many 1-bits (popcount) are in the binary representation of {a}?"
    elif t == 8:
        return f"Is bit {bit_pos} set in {a}? (bit 0 is the least significant bit) Answer YES or NO."
    elif t == 9:
        return f"Set bit {r.randint(0,3)} of the 4-bit binary number {bits4}. Give the result as a 4-bit binary string."
    elif t == 10:
        return f"Clear bit {r.randint(0,3)} of the 4-bit binary number {bits4}. Give the result as a 4-bit binary string."
    elif t == 11:
        # transformation rule detection
        in_b = format(r.randint(1,14), '04b')
        rule = r.choice(['flip_all', 'shift_left', 'reverse'])
        if rule == 'flip_all':
            out_b = ''.join('1' if c == '0' else '0' for c in in_b)
            test_in = format(r.randint(1, 14), '04b')
            test_out = ''.join('1' if c == '0' else '0' for c in test_in)
        elif rule == 'shift_left':
            out_b = in_b[1:] + in_b[0]
            test_in = format(r.randint(1, 14), '04b')
            test_out = test_in[1:] + test_in[0]
        else:
            out_b = in_b[::-1]
            test_in = format(r.randint(1, 14), '04b')
            test_out = test_in[::-1]
        return f"Identify the transformation rule: {in_b} → {out_b}. Apply the same rule to {test_in}. Give the result as a 4-bit binary string."
    elif t == 12:
        return f"What is the bitwise NOT of {a} as an 8-bit unsigned integer?"
    else:
        return f"Toggle bit {r.randint(0,3)} of the 4-bit binary number {bits4}. Give the result as a 4-bit binary string."

def gen_problem(cat, seed):
    gen_fn = {
        "numeral": gen_numeral,
        "unit_conversion": gen_unit_conversion,
        "gravity": gen_gravity,
        "cipher": gen_cipher,
        "equation": gen_equation,
        "bit_manipulation": gen_bit_manipulation,
        "conversion_sensitive": gen_bit_manipulation,
        "low_logprob_suspect": gen_equation,
        "hard": gen_equation,
    }
    return gen_fn[cat](seed)

# ─── sample metadata ──────────────────────────────────────────────────────────

FAILURE_MODES = {
    "numeral":              ("FORMAT_FAILURE",              "EXTRACTION_FAILURE"),
    "unit_conversion":      ("FORMAT_FAILURE",              "EASY_TASK_REGRESSION"),
    "gravity":              ("FORMAT_FAILURE",              "EASY_TASK_REGRESSION"),
    "cipher":               ("EXTRACTION_FAILURE",          "FORMAT_FAILURE"),
    "equation":             ("OPERATOR_CONFUSION",          "FORMAT_FAILURE"),
    "bit_manipulation":     ("BIT_RULE_FAILURE",            "POSITIONAL_MISMATCH"),
    "conversion_sensitive": ("PREPOST_CONVERSION_REGRESSION","BIT_RULE_FAILURE"),
    "low_logprob_suspect":  ("LOW_LOGPROB_COLLAPSE",        "OPERATOR_CONFUSION"),
    "hard":                 ("UNKNOWN",                     "OPERATOR_CONFUSION"),
}
DIFFICULTY = {
    "numeral": "easy",  "unit_conversion": "easy", "gravity": "easy",
    "cipher": "easy",   "equation": "medium",       "bit_manipulation": "medium",
    "conversion_sensitive": "medium", "low_logprob_suspect": "hard", "hard": "hard",
}
CONV_FLAG = {
    "numeral": False, "unit_conversion": False, "gravity": False, "cipher": False,
    "equation": False, "bit_manipulation": False,
    "conversion_sensitive": True, "low_logprob_suspect": True, "hard": False,
}
INCLUSION = {
    "numeral":              "high_frequency_easy_category",
    "unit_conversion":      "high_frequency_easy_category",
    "gravity":              "high_frequency_easy_category",
    "cipher":               "high_frequency_easy_category",
    "equation":             "representative_symbolic_slice",
    "bit_manipulation":     "representative_bit_slice",
    "conversion_sensitive": "known_conversion_brittle_pattern",
    "low_logprob_suspect":  "historical_low_logprob_sample",
    "hard":                 "known_hard_unstable_case",
}

def make_sample(sid, split, cat, seed):
    fm1, fm2 = FAILURE_MODES[cat]
    return {
        "sample_id": sid,
        "split_name": split,
        "category": cat,
        "difficulty_bucket": DIFFICULTY[cat],
        "problem": gen_problem(cat, seed),
        "expected_answer": "PENDING_BASELINE_RUN",
        "failure_mode_primary": fm1,
        "failure_mode_secondary": fm2,
        "conversion_sensitive_flag": CONV_FLAG[cat],
        "inclusion_reason": INCLUSION[cat],
    }

# ─── Quick Gate v1  (75 samples) ─────────────────────────────────────────────
# 35% easy=26, 20% equation=15, 20% bit=15, 25% hard/unstable=19

QG_PLAN = [
    ("numeral", 10), ("unit_conversion", 8), ("gravity", 5), ("cipher", 3),  # 26
    ("equation", 15),
    ("bit_manipulation", 15),
    ("conversion_sensitive", 8), ("low_logprob_suspect", 6), ("hard", 5),    # 19
]

qg_samples = []
n = 0
for cat, cnt in QG_PLAN:
    for i in range(cnt):
        n += 1
        qg_samples.append(make_sample(f"QG_{n:03d}", "quick_gate_v1", cat, n * 37 + i * 13))

assert len(qg_samples) == 75, f"QG count wrong: {len(qg_samples)}"

# ─── Diagnostic v1  (150 samples) ────────────────────────────────────────────
# All 75 QG samples re-tagged + 75 new samples emphasising brittle/conversion

DG_EXTRA_PLAN = [
    ("numeral", 5), ("unit_conversion", 4), ("gravity", 3), ("cipher", 2),   # 14
    ("equation", 15),
    ("bit_manipulation", 15),
    ("conversion_sensitive", 12), ("low_logprob_suspect", 9), ("hard", 10),   # 31
]

dg_samples = []
for s in qg_samples:
    dg_samples.append({**s, "split_name": "diagnostic_v1"})

d = 0
for cat, cnt in DG_EXTRA_PLAN:
    for i in range(cnt):
        d += 1
        dg_samples.append(make_sample(f"DG_{d:03d}", "diagnostic_v1", cat, d * 41 + i * 17 + 500))

assert len(dg_samples) == 150, f"DG count wrong: {len(dg_samples)}"

# ─── Promotion v1  (400 samples) ─────────────────────────────────────────────
# easy 30%=120, equation 20%=80, bit 20%=80,
# conversion_sensitive 10%=40, low_logprob_suspect 10%=40, hard 10%=40

PR_PLAN = [
    ("numeral", 40), ("unit_conversion", 30), ("gravity", 25), ("cipher", 25),  # 120
    ("equation", 80),
    ("bit_manipulation", 80),
    ("conversion_sensitive", 40),
    ("low_logprob_suspect", 40),
    ("hard", 40),
]

pr_samples = []
p = 0
for cat, cnt in PR_PLAN:
    for i in range(cnt):
        p += 1
        pr_samples.append(make_sample(f"PR_{p:03d}", "promotion_v1", cat, p * 43 + i * 19 + 1000))

assert len(pr_samples) == 400, f"PR count wrong: {len(pr_samples)}"

# ─── Write JSONL files ────────────────────────────────────────────────────────

def write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  wrote {len(samples)} samples → {path}")

write_jsonl(DATA_DIR / "quick_gate_v1.jsonl",   qg_samples)
write_jsonl(DATA_DIR / "diagnostic_v1.jsonl",   dg_samples)
write_jsonl(DATA_DIR / "promotion_v1.jsonl",    pr_samples)

# ─── Category manifest  (all 625 unique samples) ─────────────────────────────

MANIFEST_FIELDS = [
    "sample_id", "split_name", "category", "difficulty_bucket",
    "failure_mode_primary", "failure_mode_secondary",
    "conversion_sensitive_flag", "inclusion_reason",
]

# Deduplicate: QG samples appear in DG with same sample_id → keep both rows (different split)
all_samples = qg_samples + [s for s in dg_samples if s["sample_id"].startswith("DG_")] + pr_samples

with open(DATA_DIR / "category_manifest_v1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
    w.writeheader()
    for s in all_samples:
        w.writerow({k: s[k] for k in MANIFEST_FIELDS})

print(f"  wrote {len(all_samples)} rows → {DATA_DIR / 'category_manifest_v1.csv'}")

# ─── Coverage report for stdout ───────────────────────────────────────────────

from collections import Counter

def coverage(samples, label):
    cats = Counter(s["category"] for s in samples)
    diffs = Counter(s["difficulty_bucket"] for s in samples)
    conv = sum(1 for s in samples if s["conversion_sensitive_flag"])
    print(f"\n{label} ({len(samples)} samples):")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat:<25} {cnt:>4}  ({100*cnt/len(samples):.1f}%)")
    print(f"  difficulty: {dict(diffs)}")
    print(f"  conversion_sensitive: {conv}")

coverage(qg_samples, "Quick Gate v1")
coverage(dg_samples, "Diagnostic v1")
coverage(pr_samples, "Promotion v1")
print("\nDone.")
