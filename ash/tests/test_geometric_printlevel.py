import pytest

from ash.interfaces.interface_geometric_new import _geometric_verbose_from_printlevel


@pytest.mark.parametrize(
    ("ash_printlevel", "geometric_verbose"),
    [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3)],
)
def test_ash_printlevel_maps_to_geometric_verbose(ash_printlevel, geometric_verbose):
    assert _geometric_verbose_from_printlevel(ash_printlevel) == geometric_verbose


def test_negative_ash_printlevel_uses_quiet_geometric_verbosity():
    assert _geometric_verbose_from_printlevel(-1) == 0
