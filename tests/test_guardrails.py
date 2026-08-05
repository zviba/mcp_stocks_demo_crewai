"""The output guardrail is the check that catches a fluent, invented report.

Every test here feeds check_report a hand-written report and hand-written tool
output. No LLM, no MCP subprocess — which is the point: the guardrail is
deterministic, so it can be pinned down by tests in a way the crew cannot.

The two sets of tests matter equally. The "catches" tests show it works; the
"does not cry wolf" tests show it is usable, because a guardrail that fires on
every honest run is one that gets switched off.
"""

from stocks_crew.guardrails import check_report

# What the tools returned — the only figures a report is allowed to contain.
QUOTE = '{"symbol": "AAPL", "price": 189.4300003, "change": -1.24, "volume": 44123456}'
INDICATORS = '{"symbol": "AAPL", "last_close": 189.43, "sma": 191.02, "ema": 187.7712, "rsi": 61.4}'
EVIDENCE = [QUOTE, INDICATORS]


# --- it catches what it is for -------------------------------------------


def test_an_invented_figure_is_flagged():
    report = "AAPL last traded at 213.77, with an RSI of 61.4."
    result = check_report(report, EVIDENCE)

    assert result["passed"] is False
    assert result["ungrounded_figures"] == ["213.77"]


def test_a_report_with_no_tool_calls_behind_it_fails():
    """The failure mode the whole demo exists to expose."""
    report = "AAPL trades at 189.43 with an RSI of 61.4."
    result = check_report(report, [])

    assert result["passed"] is False
    assert set(result["ungrounded_figures"]) == {"189.43", "61.4"}
    assert any("without calling a single tool" in n for n in result["notes"])


def test_a_failed_tool_call_grounds_nothing():
    """Numbers inside an error string are not evidence."""
    result = check_report("The price is 189.43.", ["Error 189.43: upstream timeout"])

    # The error text happens to contain the same digits, but it is passed as
    # evidence only if the caller was careless; ToolTracer.evidence excludes
    # failed calls, so this documents the contract from the other side.
    assert result["figures_checked"] == 1


def test_investment_advice_is_flagged():
    report = "Momentum is strong. We recommend accumulating below 189.43."
    result = check_report(report, EVIDENCE)

    assert result["passed"] is False
    assert [hit["phrase"] for hit in result["advice_phrases"]] == ["we recommend"]


def test_a_price_target_is_flagged():
    result = check_report("Our price target is 189.43.", EVIDENCE)
    assert result["passed"] is False
    assert "price target" in result["advice_phrases"][0]["phrase"]


# --- it does not cry wolf -------------------------------------------------


def test_a_grounded_report_passes():
    report = (
        "## Executive summary\n"
        "AAPL closed at 189.43, down 1.24 on volume of 44,123,456.\n"
        "The 20-day SMA is 191.02, the 50-day EMA is 187.77 and RSI(14) is 61.4.\n"
        "This is not investment advice."
    )
    result = check_report(report, EVIDENCE)

    assert result["passed"] is True, result["ungrounded_figures"]
    assert result["ungrounded_figures"] == []


def test_rounding_is_not_treated_as_invention():
    """The tool said 187.7712; a writer says 187.77. Both are the same fact."""
    assert check_report("The EMA is 187.77.", EVIDENCE)["passed"] is True


def test_a_sign_carried_by_the_prose_is_not_treated_as_invention():
    """The tool returns change: -1.24; a report writes "down 1.24"."""
    assert check_report("AAPL was down 1.24 on the day.", EVIDENCE)["passed"] is True


def test_a_rescaled_volume_is_not_treated_as_invention():
    """44,123,456 restated as "44.1 million" is a formatting choice, not a claim."""
    assert check_report("Volume was 44.1 million shares.", EVIDENCE)["passed"] is True


def test_indicator_windows_and_section_numbers_are_ignored():
    """20, 50, 14 and "52-week" are parameters and scaffolding, not market figures."""
    report = "3. Technical picture: the 20-day SMA, 50-day EMA and 14-day RSI; no 52-week extreme."
    result = check_report(report, EVIDENCE)

    assert result["passed"] is True
    assert result["figures_checked"] == 0


def test_dates_are_not_treated_as_figures():
    assert check_report("As of 2026-08-05, RSI is 61.4.", EVIDENCE)["passed"] is True


def test_a_date_written_out_in_prose_is_not_treated_as_a_figure():
    """The tools speak ISO; the report writes "August 4, 2026".

    A real run failed on exactly this sentence: seven honest figures, plus a
    year the guardrail called invented. The year can never ground, because the
    tool output that would justify it has its dates stripped too.
    """
    report = "As of August 4, 2026, AAPL is trading at 189.43."
    result = check_report(report, EVIDENCE)

    assert result["passed"] is True, result["ungrounded_figures"]
    assert result["figures_checked"] == 1


def test_the_other_ways_a_report_writes_a_date():
    for phrasing in ("4 August 2026", "Aug. 4, 2026", "August 2026", "Sept 2026"):
        result = check_report(f"As of {phrasing}, RSI is 61.4.", EVIDENCE)
        assert result["passed"] is True, (phrasing, result["ungrounded_figures"])


def test_the_mandatory_disclaimer_does_not_fail_the_advice_check():
    """The sentence that proves the crew complied must not be what condemns it.

    This is why ADVICE_RE matches phrases rather than the bare words buy/sell.
    """
    report = (
        "This report is for information only and is not a recommendation to buy or "
        "sell any security. It is not investment advice."
    )
    assert check_report(report, EVIDENCE)["passed"] is True


def test_denied_advice_is_not_counted_as_advice():
    """Section 4 is supposed to say what the signals do *not* support."""
    report = "The indicators do not support a price target, and no one should buy on RSI alone."
    assert check_report(report, EVIDENCE)["advice_phrases"] == []


def test_an_empty_report_is_not_silently_passed_as_grounded():
    result = check_report("", EVIDENCE)

    assert result["figures_checked"] == 0
    assert result["notes"] == []
