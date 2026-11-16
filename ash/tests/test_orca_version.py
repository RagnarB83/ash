from ash.interfaces.interface_ORCA import ORCAversiongrab


def test_orca_version_is_read_from_program_banner(tmp_path):
    output = tmp_path / "orca.out"
    output.write_text(
        "ORCA TEST CALCULATION\n                         Program Version 5.0.3 - RELEASE -\n",
        encoding="utf-8",
    )

    assert ORCAversiongrab(str(output)) == "5.0.3"


def test_orca_version_returns_none_when_banner_is_missing(tmp_path):
    output = tmp_path / "orca.out"
    output.write_text("ORCA output without the program banner\n", encoding="utf-8")

    assert ORCAversiongrab(str(output)) is None
