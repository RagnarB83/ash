from ash.interfaces.interface_knarr import _print_neb_ts_energy_summary
from ash.constants import hartokcal


def test_neb_ts_energy_summary_marks_images_and_optimized_ts(capsys):
    energies = {0: -10.0, 1: -9.8, 2: -9.5, 3: -9.9, -1: -9.4}

    barrier = _print_neb_ts_energy_summary(
        energies_dict=energies, images=2, free_end=False, climbing_image=2
    )

    output = capsys.readouterr().out
    assert "Image" in output
    assert output.count("frozen") == 2
    assert "climbing" in output
    assert "optimized" in output
    assert "TS        TS" not in output
    assert barrier == (energies[-1] - energies[0]) * hartokcal
