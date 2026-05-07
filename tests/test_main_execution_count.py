from __future__ import annotations

from types import SimpleNamespace

from main import _is_real_execution_result


def test_is_real_execution_result_accepts_successful_dict() -> None:
    assert _is_real_execution_result({"success": True, "status": "FILLED"}) is True


def test_is_real_execution_result_rejects_blocked_dict() -> None:
    assert _is_real_execution_result({"success": False, "status": "REJECTED"}) is False


def test_is_real_execution_result_accepts_successful_object() -> None:
    assert _is_real_execution_result(SimpleNamespace(success=True, status="FILLED")) is True


def test_is_real_execution_result_rejects_none() -> None:
    assert _is_real_execution_result(None) is False
