from scripts.run_business_cases import _tool_sequence, classify_failure


def test_business_case_report_handles_pibench_string_tool_sequence():
    result = {"tool_calls": ["lookup_order", "record_decision"]}

    assert _tool_sequence(result) == ["lookup_order", "record_decision"]


def test_business_case_failure_classifier_prioritizes_missing_decision():
    result = {
        "status": "completed",
        "decision_error": "MISSING_DECISION",
        "outcome_results": [
            {
                "type": "tool_called_with",
                "passed": False,
                "detail": "tool_called_with(record_decision)=False",
            }
        ],
    }

    assert classify_failure(result) == "parsing"
