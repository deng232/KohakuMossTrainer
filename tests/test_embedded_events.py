from kohakumoss_common.events import Session, current, emit, stopped


def test_session_restores_and_reports():
    rows = []
    assert current() is None
    with Session(rows.append, lambda: True) as session:
        assert current() is session
        assert stopped()
        emit("training", step=2, loss=float("nan"))
    assert current() is None
    assert rows == [{"phase": "training", "step": 2, "loss": None}]


def test_nested_session_restored_after_error():
    with Session(lambda event: None) as outer:
        try:
            with Session(lambda event: None):
                raise ValueError("failed build")
        except ValueError:
            pass
        assert current() is outer
    assert not stopped()
