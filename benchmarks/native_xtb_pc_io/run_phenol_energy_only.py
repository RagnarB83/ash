#!/usr/bin/env python3
"""Run the legacy phenol-water xTB/MM energy-only example in both I/O modes."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ash-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--xtb-dir", type=Path, default=None,
                        help="directory holding the xtb executable (default: the xtb found on PATH)")
    parser.add_argument("--ash-pes2d", type=Path, default=None,
                        help="checkout that provides the ash_pes2d package (default: import from PYTHONPATH)")
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/ash-native-io-benchmark/phenol"))
    args = parser.parse_args()
    args.ash_root = args.ash_root.resolve()
    if args.xtb_dir is None:
        xtb = shutil.which("xtb")
        if xtb is None:
            sys.exit("xtb not found on PATH; pass --xtb-dir")
        args.xtb_dir = Path(xtb).parent
    args.xtb_dir = args.xtb_dir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)
    sys.path.insert(0, str(args.ash_root))
    if args.ash_pes2d:
        sys.path.insert(0, str(args.ash_pes2d.resolve()))

    from ash import Fragment, QMMMTheory, xTBTheory
    import ash.interfaces.interface_xtb as ix
    from ash_pes2d.xtb_fast import enable

    original_writer = ix.create_xtb_pcfile_general
    original_reader = ix.xtbpcgradientgrab
    frag = Fragment(xyzfile=str(args.ash_root / "ash/tests/xyzfiles/phenoldonor-r1.8-withH2O.xyz"),
                    charge=0, mult=1)
    qmatoms = list(range(13))
    charges = [0.0] * 13 + [-0.834, 0.417, 0.417]
    result = {"source": "ash/tests/unfinished/xtb-qm_mm_test_xtb640.py",
              "fixture": "ash/tests/xyzfiles/phenoldonor-r1.8-withH2O.xyz",
              "xtb_version": "6.8.0", "method": "GFN2", "system_atoms": 16,
              "qm_atoms": 13, "point_charges": 3, "gradient_requested": False, "modes": {}}

    for mode in ("native", "fast_io"):
        ix.create_xtb_pcfile_general, ix.xtbpcgradientgrab = original_writer, original_reader
        qm = xTBTheory(xtbdir=str(args.xtb_dir), xtbmethod="GFN2", numcores=1, printlevel=0)
        qmmm = QMMMTheory(fragment=frag, qm_theory=qm, embedding="Elstat", qmatoms=qmatoms,
                          charges=charges, printlevel=0)
        if mode == "fast_io":
            enable(qm, fast_io=True)
        writer = ix.create_xtb_pcfile_general
        samples = []

        def timed_writer(*args, **kwargs):
            t0 = time.perf_counter()
            value = writer(*args, **kwargs)
            samples.append(time.perf_counter() - t0)
            return value

        ix.create_xtb_pcfile_general = timed_writer
        t0 = time.perf_counter()
        energy = qmmm.run(current_coords=frag.coords.copy(), elems=frag.elems,
                          Grad=False, charge=0, mult=1, numcores=1)
        result["modes"][mode] = {"energy_Eh": float(energy),
                                 "elapsed_s": time.perf_counter() - t0,
                                 "writer_s": samples[0]}

    result["energy_exact"] = result["modes"]["native"]["energy_Eh"] == result["modes"]["fast_io"]["energy_Eh"]
    result["delta_energy_Eh"] = (result["modes"]["fast_io"]["energy_Eh"] -
                                 result["modes"]["native"]["energy_Eh"])
    out = Path(__file__).resolve().parent / "phenol_energy_only.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
