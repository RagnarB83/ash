from types import SimpleNamespace

from ash.functions.functions_general import grab_version_from_output, module_version, print_module_version


def test_module_version_uses_module_attribute(capsys):
    module = SimpleNamespace(__version__="1.2.3")

    assert module_version(module) == "1.2.3"
    assert print_module_version(module, "Example") == "1.2.3"
    assert capsys.readouterr().out.strip() == "Example version: 1.2.3"


def test_grab_version_from_output_matches_program_banner(tmp_path):
    output = tmp_path / "cp2k.out"
    output.write_text(
        "CP2K| version string: CP2K version 2025.2 (Development Version)\n",
        encoding="utf-8",
    )

    assert grab_version_from_output(
        str(output), r"CP2K\|\s+version string:\s+CP2K version\s+([^\s(]+)"
    ) == "2025.2"


def test_grab_version_from_output_returns_none_without_match(tmp_path):
    output = tmp_path / "calculation.out"
    output.write_text("Calculation completed\n", encoding="utf-8")

    assert grab_version_from_output(str(output), r"version\s+([0-9.]+)") is None


def test_grab_version_from_output_returns_none_when_file_is_missing(tmp_path):
    missing_output = tmp_path / "missing.out"

    assert grab_version_from_output(str(missing_output), r"version\s+([0-9.]+)") is None


def test_grab_version_from_output_supports_nwchem_banner(tmp_path):
    output = tmp_path / "nwchem.out"
    output.write_text(
        "Northwest Computational Chemistry Package (NWChem) 7.3.1\n",
        encoding="utf-8",
    )

    version = grab_version_from_output(
        str(output),
        r"Northwest Computational Chemistry Package \(NWChem\)\s+([0-9]+(?:\.[0-9]+)+)",
    )
    assert version == "7.3.1"


def test_grab_version_from_output_supports_gaussian_banner(tmp_path):
    output = tmp_path / "gaussian.log"
    output.write_text(
        " Gaussian 16: EM64L-G16RevC.01 10-Sep-2019\n",
        encoding="utf-8",
    )

    version = grab_version_from_output(
        str(output), r"(Gaussian\s+(?:09|16):.*?Rev[A-Z]\.\d+)"
    )
    assert version == "Gaussian 16: EM64L-G16RevC.01"


def test_grab_version_from_output_supports_dftbplus_banner(tmp_path):
    output = tmp_path / "dftb+.out"
    output.write_text("|  DFTB+ development version (commit: 5be918b8, base: 25.1)\n", encoding="utf-8")

    pattern = r"DFTB\+\s+(?:release\s+)?(?:development version\s+\(commit:[^,]+,\s*base:\s*)?([0-9]+(?:\.[0-9]+)+)"
    assert grab_version_from_output(str(output), pattern) == "25.1"

    output.write_text("|  DFTB+ release 25.1\n", encoding="utf-8")
    assert grab_version_from_output(str(output), pattern) == "25.1"


def test_grab_version_from_output_supports_cfour_banner(tmp_path):
    output = tmp_path / "cfour.out"
    output.write_text("                           Version 2.1\n", encoding="utf-8")

    assert grab_version_from_output(
        str(output), r"^\s*Version\s+([0-9]+(?:\.[0-9]+)+[A-Za-z0-9.-]*)\s*$"
    ) == "2.1"


def test_grab_version_from_output_supports_mopac_banner(tmp_path):
    output = tmp_path / "mopac.out"
    output.write_text(
        "** Site#: 123456       MOPAC                         Version 7.263L **\n",
        encoding="utf-8",
    )

    assert grab_version_from_output(
        str(output), r"\*\*\s+Site#:.+?\bVersion\s+([0-9]+(?:\.[0-9]+)+(?:[A-Za-z0-9.-]*))\s+\*\*"
    ) == "7.263L"
