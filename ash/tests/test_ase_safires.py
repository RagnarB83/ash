import pytest

from ash.interfaces.interface_ASE import Dynamics_ASE


def test_dynamics_ase_rejects_disabled_safires(capsys):
    with pytest.raises(SystemExit) as exc_info:
        Dynamics_ASE(safires=True)

    assert exc_info.value.code == 1
    assert "SAFIRES support is disabled in Dynamics_ASE" in capsys.readouterr().out
