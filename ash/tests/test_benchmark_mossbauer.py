from types import SimpleNamespace

import ash.modules.module_benchmarking as benchmarking


def test_run_mossbauer_benchmark_uses_delta_eq(monkeypatch, tmp_path, capsys):
    """Mossbauer property sets use ORCA Delta-EQ as their benchmark value."""
    ash_package = tmp_path / "ash"
    data_dir = ash_package / "databases" / "Benchmarking-sets" / "Mossbauer-test" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "Reference_data.txt").write_text(
        "#TESTSET_INFO Numentries: 1\n"
        "#TESTSET_INFO Unit: MHz\n"
        "#TESTSET_INFO Property: Mossbauer\n"
        "1 fe 0 0.500\n"
    )
    (data_dir / "fe.xyz").write_text("1\n0 6\nFe 0.0 0.0 0.0\n")

    monkeypatch.setattr(benchmarking.ash, "__file__", str(ash_package / "__init__.py"))

    class ORCATheory:
        def __init__(self):
            self.filename = "orca"
            self.numcores = 1
            self.propertyblock = ""

        def cleanup(self):
            pass

    theory = ORCATheory()

    def fake_singlepoint(fragment, theory, charge, mult):
        assert "nuclei = all Fe {rho,fgrad}" in theory.propertyblock
        with open(theory.filename + ".out", "w") as output:
            output.write(" RHO(0)= 1.250\n Delta-EQ 0.456 MHz\n")
        return SimpleNamespace(energy=-100.0)

    monkeypatch.setattr(benchmarking.ash, "Singlepoint", fake_singlepoint)
    monkeypatch.chdir(tmp_path)

    benchmarking.run_benchmark(set="Mossbauer-test", theory=theory, numcores=1)

    report = capsys.readouterr().out
    assert "Mossbauer Delta-EQ" in report
    assert "0.4560" in report
    assert "-0.0440" in report
