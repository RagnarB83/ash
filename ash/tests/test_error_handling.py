import pytest

import ash.functions.functions_general as functions_general


@pytest.mark.parametrize("interactive", [False, True])
def test_ashexit_uses_requested_exit_code(monkeypatch, interactive):
    monkeypatch.setattr(functions_general, "is_interactive", lambda: interactive)

    with pytest.raises(SystemExit) as exc_info:
        functions_general.ashexit("expected failure", code=7)

    assert exc_info.value.code == 7
