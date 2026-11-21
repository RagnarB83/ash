import pytest

from ash import Fragment
from ash.functions.functions_parallel import Job_parallel


class ElementSumTheory:
    def __init__(self, offset):
        self.theorytype = "QM"
        self.numcores = 1
        self.label = f"stub_{offset}"
        self.offset = offset

    def run(self, elems=None, **kwargs):
        atomic_numbers = {"H": 1, "F": 9}
        return self.offset + sum(atomic_numbers[element] for element in elems)


def make_fragment(elems, label):
    coords = [[float(i), 0.0, 0.0] for i in range(len(elems))]
    return Fragment(
        elems=elems,
        coords=coords,
        charge=0,
        mult=1,
        label=label,
        printlevel=0,
    )


@pytest.mark.parametrize("input_kind", ["fragments", "fragmentfiles"])
def test_job_parallel_runs_cartesian_product_of_theories_and_fragments(tmp_path, monkeypatch, input_kind):
    monkeypatch.chdir(tmp_path)
    fragments = [make_fragment(["H", "H"], "H2"), make_fragment(["H", "F"], "HF")]
    theories = [ElementSumTheory(-10.0), ElementSumTheory(-20.0)]
    if input_kind == "fragments":
        input_args = {"fragments": fragments}
    else:
        fragmentfiles = []
        for index, fragment in enumerate(fragments):
            filename = tmp_path / f"fragment{index}.ygg"
            fragment.print_system(filename=str(filename))
            fragmentfiles.append(str(filename))
        input_args = {"fragmentfiles": fragmentfiles}

    result = Job_parallel(
        **input_args,
        theories=theories,
        numcores=2,
        printlevel=0,
    )

    assert len(result.energies) == 4
    assert result.energies_dict == {
        "theory0_fragment0": -8.0,
        "theory0_fragment1": 0.0,
        "theory1_fragment0": -18.0,
        "theory1_fragment1": -10.0,
    }
    assert set(result.worker_dirnames) == set(result.energies_dict)
