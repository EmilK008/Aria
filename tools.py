"""
tools.py -- exact tools that don't need the neural net.

A ~226M model is genuinely bad at arithmetic (it guesses digits). A calculator is
exact. So we detect math questions and compute the real answer in Python, the same
way retrieval.py handles facts -- the model shouldn't guess what a computer can know.

  solve_math(message) -> "3 + 4 = 7"  (a ready-to-show string) or None
"""

import ast
import math
import operator
import re

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log10, "ln": math.log,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp,
    "factorial": math.factorial,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    """Safely evaluate a parsed arithmetic expression (no arbitrary code -- only
    the numbers, operators, and functions we whitelist above)."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError("unsupported expression")


def _fmt(value):
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"           # trim float noise
    return str(value)


# words people use for operators -> symbols
_WORDS = [
    (r"\bplus\b|\band\b", "+"), (r"\bminus\b|\bsubtract(ed)?( by)?\b", "-"),
    (r"\btimes\b|\bmultiplied by\b|\bx\b|×", "*"),
    (r"\bdivided by\b|÷|\bover\b", "/"),
    (r"\bto the power of\b|\braised to\b|\^", "**"),
    (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"),
]
# needs at least one real operation, so a bare number ("8849") isn't "math"
_HAS_OP = re.compile(r"[+\-*/%^]|sqrt|factorial|\bsquared\b|\bcubed\b|percent|\bof\b")


def solve_math(message):
    """Return a ready-to-show answer string for a math question, or None."""
    m = message.lower().strip().rstrip("?.! ").strip()
    if not re.search(r"\d", m):
        return None

    # "X% of Y" / "X percent of Y"
    pct = re.search(r"([\d.]+)\s*(?:%|percent)\s*of\s*([\d.]+)", m)
    if pct:
        val = float(pct.group(1)) / 100 * float(pct.group(2))
        return f"{pct.group(1)}% of {pct.group(2)} = {_fmt(val)}"

    if not _HAS_OP.search(m):
        return None

    expr = m
    expr = re.sub(r"square root of\s*", "sqrt", expr)
    for pat, sym in _WORDS:
        expr = re.sub(pat, sym, expr)
    # strip conversational lead-in
    expr = re.sub(r"^(what\s*(is|are|'?s)?|whats|calculate|compute|how much is|"
                  r"solve|evaluate|the answer to)\s*", "", expr).strip()
    expr = expr.replace("=", "").strip()
    # wrap bare sqrt123 -> sqrt(123)
    expr = re.sub(r"sqrt\s*([\d.]+)", r"sqrt(\1)", expr)
    # keep only characters that belong in an expression
    if not re.fullmatch(r"[\d\s+\-*/%.()a-z]*", expr):
        return None
    try:
        value = _eval(ast.parse(expr, mode="eval"))
    except Exception:
        return None
    if isinstance(value, complex) or (isinstance(value, float) and math.isinf(value)):
        return None
    # echo it back cleanly
    shown = re.sub(r"\s+", " ", expr).replace("**", "^")
    return f"{shown} = {_fmt(value)}"


# "what did you just say / what was the last answer / repeat that / remind me..." --
# the model can't reliably recall, but the app HAS the exact history, so we answer
# these deterministically. Broad on purpose (phrasing varies a lot).
_RECALL = re.compile(
    r"\b("
    r"(last|previous|prior|first)\s+(answer|question|reply|response|thing|message)|"
    r"answer to (the |my |that )?(last|previous|prior|first)|"
    r"what did (you|i)\s+(just\s+)?(say|said|answer|tell|told|ask|asked|mean)|"
    r"what('?s| is| was| were)\s+(your|the|my|that|it)?\s*(last|previous|prior|first)?\s*"
    r"(answer|question|reply|response)|"
    r"what('?s| is| was)\s+(that|it|the answer)\s+again|"
    r"repeat (that|it|yourself|the (answer|question)|what you)|"
    r"say (that|it) again|come again|one more time|run that by me|"
    r"remind me (what|of|again)|"
    r"what (did|were) we (just\s+)?(talk|talking|discuss|say|saying)|"
    r"you (just\s+)?(said|told me|mentioned|answered)|"
    r"go back to (that|what)|what was that"
    r")\b",
    re.I,
)


def recall_answer(message, history):
    """Answer recall questions ('what was the last answer', 'remind me', ...) from history."""
    if not history or not _RECALL.search(message):
        return None
    ml = message.lower()
    turn = history[0] if re.search(r"\bfirst\b", ml) else history[-1]
    which = "first" if turn is history[0] and len(history) > 1 else "last"
    # asking about the QUESTION / what the user said -> echo the user turn
    if re.search(r"\b(i (just )?(ask|asked|said)|my (last|previous|first)? ?question|"
                 r"(the |that )?question)\b", ml) and not re.search(
                 r"\b(answer|reply|response|you (say|said|tell|told|answer))\b", ml):
        return f'You asked: "{turn[0]}"'
    return f"The {which} answer was: {turn[1]}"


if __name__ == "__main__":
    for q in ["what is 3 + 4?", "12 * 8", "what's 20% of 150", "square root of 144",
              "100 / 7", "2 to the power of 10", "5 squared", "how tall is Everest",
              "hello there", "what is the capital of France"]:
        print(f"{q!r:40} -> {solve_math(q)}")
