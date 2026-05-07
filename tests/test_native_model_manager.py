from core_engine.native.model_manager import _classify_model_load_error


def test_classify_model_load_error_marks_gru_keyword_mismatch_incompatible():
    exc = ValueError(
        "Unrecognized keyword arguments passed to GRU: "
        "{'batch_input_shape': [None, 60, 29], 'time_major': False}"
    )
    should_quarantine, reason = _classify_model_load_error(exc)
    assert should_quarantine is True
    assert reason == "legacy_inputlayer_batch_shape"
