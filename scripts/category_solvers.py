"""
category_solvers.py

Specialized deterministic solvers for each competition category.
Each function: solve_<category>(prompt, answer=None) -> Optional[str]

  gravity          : infer g from d=0.5*g*t^2 examples, predict test d
  unit_conversion  : infer factor from examples, predict test output
  numeral          : convert int to Roman numerals (verified against examples)
  cipher           : build char substitution table, apply to test text
  bit_manipulation : try single-step bit ops (XOR/rotate/shift/NOT/etc.)
  equation         : delegate to cryptarithm_solver (concat rules)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# gravity  ─  d = 0.5 * g * t²
# ─────────────────────────────────────────────────────────────────────────────

_GRAV_EXAMPLE_RE = re.compile(
    r"t\s*=\s*([\d.]+)\s*s[^,\n]*[,\s]+distance\s*=\s*([\d.]+)",
    re.IGNORECASE,
)
_GRAV_TEST_RE = re.compile(
    r"for\s+t\s*=\s*([\d.]+)\s*s",
    re.IGNORECASE,
)


def solve_gravity(prompt: str, answer: Optional[str] = None) -> Optional[str]:
    examples = _GRAV_EXAMPLE_RE.findall(prompt)
    if not examples:
        return None

    # Least-squares fit: g = 2*sum(d_i*t_i^2) / sum(t_i^4)
    # This is more numerically stable than simple averaging
    num = 0.0
    den = 0.0
    for t_s, d_s in examples:
        t, d = float(t_s), float(d_s)
        if t > 0:
            t2 = t * t
            num += d * t2
            den += t2 * t2
    if den == 0:
        return None
    g = 2.0 * num / den

    # Find test t (last occurrence, usually after "Now")
    now_idx = prompt.lower().rfind("now")
    search_from = now_idx if now_idx >= 0 else 0
    m = _GRAV_TEST_RE.search(prompt, search_from)
    if not m:
        return None

    t_test = float(m.group(1))
    d_pred = 0.5 * g * t_test * t_test
    return f"{d_pred:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# unit_conversion  ─  output = factor × input
# ─────────────────────────────────────────────────────────────────────────────

_UC_EXAMPLE_RE = re.compile(
    r"([\d.]+)\s*\w*\s+becomes\s+([\d.]+)",
    re.IGNORECASE,
)
_UC_TEST_RE = re.compile(
    r"convert.*?:\s*([\d.]+)",
    re.IGNORECASE,
)
# Alternative: "Now, convert the following measurement: 25.09 m"
_UC_TEST_RE2 = re.compile(
    r"convert.*?([\d.]+)\s*\w*\s*$",
    re.IGNORECASE,
)


def solve_unit_conversion(prompt: str, answer: Optional[str] = None) -> Optional[str]:
    examples = _UC_EXAMPLE_RE.findall(prompt)
    if not examples:
        return None

    factors = []
    for inp_s, out_s in examples:
        inp, out = float(inp_s), float(out_s)
        if inp > 0:
            factors.append(out / inp)
    if not factors:
        return None
    factor = sum(factors) / len(factors)

    # Find test value
    now_idx = prompt.lower().rfind("now")
    search_from = now_idx if now_idx >= 0 else 0
    m = _UC_TEST_RE.search(prompt, search_from)
    if not m:
        m = _UC_TEST_RE2.search(prompt, search_from)
    if not m:
        return None

    test_val = float(m.group(1))
    return f"{factor * test_val:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# numeral  ─  integer → Roman numerals (verified against prompt examples)
# ─────────────────────────────────────────────────────────────────────────────

_NUMERAL_EXAMPLE_RE = re.compile(r"(\d+)\s*->\s*([A-Z]+)")
_NUMERAL_TEST_RE = re.compile(r"write the number\s+(\d+)", re.IGNORECASE)

_ROMAN_VALS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"),  (90,  "XC"), (50,  "L"), (40,  "XL"),
    (10,  "X"),  (9,   "IX"), (5,   "V"), (4,   "IV"), (1, "I"),
]


def _to_roman(n: int) -> str:
    if n <= 0:
        return ""
    result = []
    for value, symbol in _ROMAN_VALS:
        while n >= value:
            result.append(symbol)
            n -= value
    return "".join(result)


def _from_roman(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = vals.get(ch, 0)
        total += v if v >= prev else -v
        prev = v
    return total


def solve_numeral(prompt: str, answer: Optional[str] = None) -> Optional[str]:
    # Verify examples match Roman numerals
    examples = _NUMERAL_EXAMPLE_RE.findall(prompt)
    if not examples:
        return None

    for num_s, roman_s in examples:
        if _to_roman(int(num_s)) != roman_s:
            # Not standard Roman; cannot solve
            return None

    # Extract test number
    now_idx = prompt.lower().rfind("now")
    search_from = now_idx if now_idx >= 0 else 0
    m = _NUMERAL_TEST_RE.search(prompt, search_from)
    if not m:
        return None

    return _to_roman(int(m.group(1)))


# ─────────────────────────────────────────────────────────────────────────────
# cipher  ─  character substitution (build table from examples)
# ─────────────────────────────────────────────────────────────────────────────

_CIPHER_EXAMPLE_RE = re.compile(r"^(.+?)\s*->\s*(.+)$", re.MULTILINE)
_CIPHER_ENCRYPT_RE = re.compile(r"^(.+?)\s*->\s*(.+)$", re.MULTILINE)


def _build_char_map(cipher_lines: List[Tuple[str, str]]) -> Dict[str, str]:
    """
    Build source→target character map from (src_text, tgt_text) pairs.
    After direct mapping, infer remaining bijective assignments when unambiguous.
    """
    mapping: Dict[str, str] = {}
    for src_text, tgt_text in cipher_lines:
        src_words = src_text.split()
        tgt_words = tgt_text.split()
        if len(src_words) != len(tgt_words):
            continue
        for sw, tw in zip(src_words, tgt_words):
            if len(sw) != len(tw):
                continue
            for sc, tc in zip(sw, tw):
                mapping.setdefault(sc, tc)

    # Try bijective inference: if only one src char unmapped and one tgt char unused,
    # we can infer that mapping.  Repeat until stable.
    used_tgt = set(mapping.values())
    alphabet = set(c for _, tgt in cipher_lines for c in tgt if c.isalpha())
    alphabet |= set(c for src, _ in cipher_lines for c in src if c.isalpha())

    changed = True
    while changed:
        changed = False
        unmapped_src = [c for c in alphabet if c not in mapping and c.isalpha()]
        used_tgt = set(mapping.values())
        unused_tgt = [c for c in alphabet if c not in used_tgt and c.isalpha()]
        if len(unmapped_src) == 1 and len(unused_tgt) == 1:
            mapping[unmapped_src[0]] = unused_tgt[0]
            changed = True

    return mapping


def _apply_char_map(text: str, mapping: Dict[str, str]) -> str:
    return "".join(mapping.get(c, c) for c in text)


def solve_cipher(prompt: str, answer: Optional[str] = None) -> Optional[str]:
    # Separate examples from test
    now_idx = prompt.lower().rfind("now")
    if now_idx < 0:
        return None
    examples_text = prompt[:now_idx]
    test_line = prompt[now_idx:]

    # Parse examples: "src -> tgt"
    raw_pairs = _CIPHER_EXAMPLE_RE.findall(examples_text)
    if not raw_pairs:
        return None

    # Determine direction: decrypt = cipher->plain, encrypt = plain->cipher
    is_decrypt = "decrypt" in test_line.lower()
    is_encrypt = "encrypt" in test_line.lower()

    # If decrypt: source is cipher, target is plain → map cipher_char→plain_char
    # If encrypt: source is plain, target is cipher → map plain_char→cipher_char
    # Default: assume decrypt (cipher→plain)
    if is_encrypt:
        # Reverse mapping
        pairs = [(tgt, src) for src, tgt in raw_pairs]
    else:
        pairs = raw_pairs

    char_map = _build_char_map(pairs)
    if not char_map:
        return None

    # Extract test input (text after last ":")
    m = re.search(r":\s*(.+?)$", test_line.strip(), re.DOTALL)
    if not m:
        return None
    test_input = m.group(1).strip()

    # Coverage check: fraction of test chars in mapping
    alpha_chars = [c for c in test_input if c.isalpha()]
    if not alpha_chars:
        return None
    covered = sum(1 for c in alpha_chars if c in char_map)
    if covered / len(alpha_chars) < 0.7:
        # Too many unknown chars → unreliable
        return None

    return _apply_char_map(test_input, char_map)


# ─────────────────────────────────────────────────────────────────────────────
# bit_manipulation  ─  try single-step bit operations
# ─────────────────────────────────────────────────────────────────────────────

_BIT_EXAMPLE_RE = re.compile(r"([01]{8})\s*->\s*([01]{8})")
_BIT_TEST_RE = re.compile(r"determine the output for:?\s*([01]{8})", re.IGNORECASE)


def _make_bit_ops() -> List[Tuple[str, callable]]:
    ops: List[Tuple[str, callable]] = []

    # NOT
    ops.append(("NOT", lambda x: (~x) & 0xFF))

    # Rotate left / right
    for k in range(1, 8):
        ops.append((f"ROL{k}", lambda x, k=k: ((x << k) | (x >> (8 - k))) & 0xFF))
        ops.append((f"ROR{k}", lambda x, k=k: ((x >> k) | (x << (8 - k))) & 0xFF))

    # Shift left / right (zero fill)
    for k in range(1, 8):
        ops.append((f"SHL{k}", lambda x, k=k: (x << k) & 0xFF))
        ops.append((f"SHR{k}", lambda x, k=k: (x >> k) & 0xFF))

    # Bit reversal
    ops.append(("BITREV", lambda x: int(f"{x:08b}"[::-1], 2)))

    # Swap nibbles
    ops.append(("NIBSWAP", lambda x: ((x & 0x0F) << 4) | ((x & 0xF0) >> 4)))

    # XOR with constant (0–255)
    for c in range(256):
        ops.append((f"XOR{c:02X}", lambda x, c=c: x ^ c))

    # AND with constant
    for c in range(256):
        ops.append((f"AND{c:02X}", lambda x, c=c: x & c))

    # OR with constant
    for c in range(256):
        ops.append((f"OR{c:02X}", lambda x, c=c: x | c))

    return ops


_BIT_OPS = _make_bit_ops()


def _try_two_step(
    examples: List[Tuple[int, int]], test_int: int
) -> Optional[Tuple[str, int]]:
    """Try NOT+rotate and rotate+XOR combinations (most common 2-step patterns)."""
    # Step 1 candidates: NOT, rotations
    step1_candidates = [("NOT", lambda x: (~x) & 0xFF)]
    for k in range(1, 8):
        step1_candidates.append((f"ROL{k}", lambda x, k=k: ((x << k) | (x >> (8 - k))) & 0xFF))
        step1_candidates.append((f"ROR{k}", lambda x, k=k: ((x >> k) | (x << (8 - k))) & 0xFF))

    for s1_name, s1_fn in step1_candidates:
        mid_examples = [(s1_fn(inp), out) for inp, out in examples]
        for s2_name, s2_fn in _BIT_OPS:
            if all(s2_fn(mid) == out for mid, out in mid_examples):
                mid_test = s1_fn(test_int)
                result = s2_fn(mid_test)
                return (f"{s1_name}+{s2_name}", result)
    return None


def solve_bit_manipulation(prompt: str, answer: Optional[str] = None) -> Optional[str]:
    raw_examples = _BIT_EXAMPLE_RE.findall(prompt)
    if not raw_examples:
        return None

    int_examples = [(int(a, 2), int(b, 2)) for a, b in raw_examples]

    # Find test input
    now_idx = prompt.lower().rfind("now")
    search_from = now_idx if now_idx >= 0 else 0
    m = _BIT_TEST_RE.search(prompt, search_from)
    if not m:
        return None
    test_int = int(m.group(1), 2)

    # Try single-step operations
    for op_name, op_fn in _BIT_OPS:
        if all(op_fn(inp) == out for inp, out in int_examples):
            result = op_fn(test_int)
            return f"{result:08b}"

    # Try two-step compositions
    two = _try_two_step(int_examples, test_int)
    if two:
        _, result = two
        return f"{result:08b}"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# equation  ─  delegate to cryptarithm_solver (concat rules)
# ─────────────────────────────────────────────────────────────────────────────

def solve_equation(prompt: str, answer: Optional[str] = None) -> Optional[str]:
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from cryptarithm_solver import solve as _cs_solve

        # Normalize "determine the result for: X" → "X = ?"
        m = re.search(
            r"Now,?\s*determine the result for:?\s*(.+?)(?:\n|$)",
            prompt,
            re.IGNORECASE,
        )
        if m:
            test_input = m.group(1).strip()
            before = prompt[: prompt.lower().find("now")]
            ex_lines = [
                l.strip()
                for l in before.split("\n")
                if " = " in l and len(l.strip()) > 2
                and "example" not in l.lower()
                and "wonderland" not in l.lower()
                and "rules" not in l.lower()
                and "applied" not in l.lower()
            ]
            normalized = "\n".join(ex_lines) + f"\n{test_input} = ?"
        else:
            normalized = prompt

        result = _cs_solve(normalized, expected_answer=answer)
        return result.predicted if result.predicted else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_SOLVER_MAP = {
    "gravity":          solve_gravity,
    "unit_conversion":  solve_unit_conversion,
    "numeral":          solve_numeral,
    "cipher":           solve_cipher,
    "bit_manipulation": solve_bit_manipulation,
    "equation":         solve_equation,
}


def solve_by_category(
    category: str,
    prompt: str,
    answer: Optional[str] = None,
) -> Optional[str]:
    fn = _SOLVER_MAP.get(category)
    if fn is None:
        return None
    try:
        return fn(prompt, answer)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────

_SMOKE = [
    (
        "gravity",
        "For t = 1.37s, distance = 14.92 m\nFor t = 4.27s, distance = 144.96 m\n"
        "Now, determine the falling distance for t = 4.41s given d = 0.5*g*t^2.",
        "154.62",
    ),
    (
        "unit_conversion",
        "10.08 m becomes 6.69\n17.83 m becomes 11.83\n"
        "Now, convert the following measurement: 25.09 m",
        "16.65",
    ),
    (
        "numeral",
        "11 -> XI\n15 -> XV\n94 -> XCIV\n19 -> XIX\n"
        "Now, write the number 38 in the Wonderland numeral system.",
        "XXXVIII",
    ),
    (
        "cipher",
        # Test chars only use chars that appear as cipher chars in examples
        "ucoov pwgtfyoqg vorq yrjjoe -> queen discovers near valley\n"
        "pqrsfv pqorzg wvgwpo trgbjo -> dragon dreams inside castle\n"
        "gbcpovb tqorbog bxo zrswtrj pffq -> student creates the magical door\n"
        "bxo sfjpov pqrsfv dfjjfig -> the golden dragon follows\n"
        "nqwvtogg qorpg bxo zegboqwfcg gotqob -> princess reads the mysterious secret\n"
        "Now, decrypt the following text: trb wzrswvog",
        "cat imagines",
    ),
    (
        "bit_manipulation",
        "11001100 -> 00110011\n10101010 -> 01010101\n11110000 -> 00001111\n"
        "Now, determine the output for: 01110001",
        "10001110",
    ),
]

if __name__ == "__main__":
    print("=== Smoke tests ===")
    for cat, prompt, expected in _SMOKE:
        pred = solve_by_category(cat, prompt, expected)
        ok = "OK" if pred == expected else f"FAIL (got {pred!r})"
        print(f"[{ok:40s}] {cat}")
