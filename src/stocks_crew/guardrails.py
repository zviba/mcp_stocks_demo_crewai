# guardrails.py — the check that runs *after* the crew, on what it produced.
#
# The demo already has two guardrails, and both act before the model speaks:
# validate_symbol() in crew.py filters the input, and the per-agent `tools:`
# lists in agents.yaml limit what each agent may reach. This module is the third
# kind — an output guardrail — and it exists because of the note at the top of
# tasks.yaml: "call this tool, do not guess" is a *prompt*, not a guarantee.
#
# trace.py already answers "which tools were called?". What it does not answer is
# "does the report only contain numbers those tools actually returned?" That is a
# deterministic question about two strings, so it is settled here in plain Python
# rather than by asking another model. A guardrail that can hallucinate is not a
# guardrail.
#
# Two checks:
#   1. grounding  — every figure in the report traces to some tool output.
#   2. advice     — the report does not slip into investment advice, which every
#                   task brief forbids and no prompt can enforce.
#
# Both are heuristics, and the interesting classroom question is which way they
# should be wrong. These are tuned to under-report: a rounding tolerance, a scale
# tolerance for "44.1 million", and a floor under which integers are ignored. A
# guardrail that cries wolf on every run gets switched off in week two.
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

#: Numbers as they appear in prose: optional sign, thousands separators, decimals.
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

#: Dates are stripped before scanning. They are figures in the regex sense but
#: not claims about the market, and "2026-08-05" would otherwise contribute a
#: bogus 2026 — one that can never be grounded, because the same stripping
#: removes the year from the tool output that would have justified it.
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

#: The tools speak ISO; a report writes "As of August 4, 2026". Stripping only
#: the ISO form flagged that year as invented on a run where every real figure
#: checked out — the exact cry-wolf failure these tolerances exist to avoid.
#: Bare years with no month ("in 2026") are still counted; ignoring every
#: four-digit integer in a plausible year range would blind the check to real
#: figures, so the narrower rule is the one worth having.
_MONTHS = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
PROSE_DATE_RE = re.compile(
    r"\b(?:"
    rf"(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"  # August 4, 2026
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?,?\s+\d{{4}}"  # 4 August 2026
    rf"|(?:{_MONTHS})\.?\s+\d{{4}}"  # August 2026
    r")\b",
    re.IGNORECASE,
)

#: Bare integers below this are ignored: section numbers, indicator windows
#: (20/50/14), "52-week", the horizon in days. None of them is a market figure,
#: and all of them would be noise. Anything with a decimal point is checked
#: regardless of size — that is where prices, percentages and RSI live.
MIN_INTEGER_CLAIM = 1000

#: A report may restate a volume of 44,123,456 as "44.1 million", so a claim is
#: also compared against the tool's number divided by each of these.
SCALES = (1.0, 1e3, 1e6, 1e9)

#: Phrases that are advice however they are phrased. Deliberately multi-word:
#: matching a bare "buy" would fire on the mandatory disclaimer ("this is not a
#: recommendation to buy or sell"), i.e. on the very sentence that proves the
#: crew complied. A guardrail that punishes compliance is worse than none.
ADVICE_RE = re.compile(
    r"\b("
    r"price targets?"
    r"|strong (?:buy|sell)"
    r"|(?:we|i) recommend"
    r"|(?:you|investors|readers) should (?:buy|sell|hold)"
    r"|will (?:reach|hit|climb to|fall to|drop to)"
    r"|(?:expected|likely) to (?:reach|hit)"
    r")\b",
    re.IGNORECASE,
)

#: How far back to look for a negation before an advice match. Section 4 of the
#: report ("what the signals do and do not support") is *supposed* to say things
#: like "these indicators do not support a price target".
NEGATION_WINDOW = 48
NEGATION_RE = re.compile(r"\b(?:no|not|never|without|cannot|can't|avoid|refrain)\b", re.IGNORECASE)


def _to_float(literal: str) -> float:
    return float(literal.replace(",", ""))


def _decimal_places(literal: str) -> int:
    return len(literal.split(".")[1]) if "." in literal else 0


def extract_numbers(text: str) -> List[str]:
    """Every numeric literal in `text`, as written, minus dates."""
    without_dates = PROSE_DATE_RE.sub(" ", ISO_DATE_RE.sub(" ", text or ""))
    return NUMBER_RE.findall(without_dates)


def _is_claim(literal: str) -> bool:
    """Is this literal a market figure worth checking, or scaffolding?"""
    if "." in literal:
        return True
    return abs(_to_float(literal)) >= MIN_INTEGER_CLAIM


def _grounded(literal: str, evidence: Iterable[float]) -> bool:
    """Does some tool output contain this number, allowing for rounding?

    The report rounds — the tool says 189.4300003 and the writer says 189.43 —
    so an exact string match would flag every honest figure. The tolerance is one
    unit in the claim's last decimal place, which accepts both rounding and
    truncation.

    Magnitudes are compared, not signed values: the tool returns
    `"change": -1.24` and a report properly writes "down 1.24", carrying the sign
    in the word. Insisting on the sign flagged that as invented — a false
    positive on correct output, found by the tests in test_guardrails.py.

    Note what these tolerances cost. For a bare integer the tolerance is 1.0, so
    "190" passes against a tool value of 189.87; and dropping the sign means a
    report that says "up 1.24" when the stock fell is grounded as far as this
    check is concerned. Both are deliberate. This guardrail asks "where did this
    number come from?", not "is the sentence true" — the second question needs a
    different check, and it is worth asking the room how they would write it.
    """
    value = abs(_to_float(literal))
    tolerance = 10.0 ** -_decimal_places(literal)
    for number in evidence:
        for scale in SCALES:
            if abs(abs(number) / scale - value) < tolerance:
                return True
    return False


def _advice_hits(report: str) -> List[Dict[str, str]]:
    """Advice-shaped phrases, skipping the ones that are being denied."""
    hits: List[Dict[str, str]] = []
    for match in ADVICE_RE.finditer(report or ""):
        before = report[max(0, match.start() - NEGATION_WINDOW) : match.start()]
        if NEGATION_RE.search(before):
            continue
        start = max(0, match.start() - 60)
        hits.append({
            "phrase": match.group(0).lower(),
            "context": report[start : match.end() + 60].replace("\n", " ").strip(),
        })
    return hits


def check_report(report: str, evidence: Iterable[str]) -> Dict[str, Any]:
    """Check the crew's report against the tool output it was supposed to use.

    Args:
        report: The final report text.
        evidence: The *full* output of every tool call in the run. Full, not the
            240-character `result_preview` from the trace — checking against a
            truncated preview would flag row 40 of a price series as invented.
            See ToolTracer.evidence.

    Returns a dict with `passed` plus the detail behind it, which is merged into
    the run_analysis() result and rendered in the UI. It reports; it does not
    block. Whether a failed check should instead withhold the report, or hand it
    back for one retry, is the question to put to the room.
    """
    numbers: List[float] = []
    for output in evidence:
        numbers.extend(_to_float(n) for n in extract_numbers(str(output)))

    claims = [n for n in extract_numbers(report) if _is_claim(n)]
    # dict.fromkeys: de-duplicate, keep the order they appear in the report.
    ungrounded = [n for n in dict.fromkeys(claims) if not _grounded(n, numbers)]
    advice = _advice_hits(report)

    notes: List[str] = []
    if not numbers:
        notes.append(
            "No tool output to check against — the agents produced this report "
            "without calling a single tool."
        )
    if claims and not ungrounded:
        notes.append("Every figure in the report traces to a tool result.")

    return {
        "passed": not ungrounded and not advice,
        "figures_checked": len(set(claims)),
        "ungrounded_figures": ungrounded,
        "advice_phrases": advice,
        "evidence_values": len(set(numbers)),
        "notes": notes,
    }
