#!/usr/bin/env bash
# Generate the inputs that three ASH examples expect but do not ship, using the examples'
# own preparation scripts. Results are copied into prepared/ next to this script.
#
#   isobutyraldehyde_water  examples/explicit-solvation/1-expl-water-openff  (solvate.py, run_mm_md.py)
#   fecl4_water             examples/explicit-solvation/2-expl-tmc-nonb      (solvate.py, run_mm_md.py)
#   na2hpo4_molcrys         examples/Molcrys-QM-MM-example/Na2HPO4-xtl-file  (molcrys step of molcrys-test.py)
#
# Only two things differ from the example scripts:
#   * "amber/tip3p_standard.xml" (openmmforcefields, not installed) -> "amber14/tip3p.xml", the
#     TIP3P file ASH's solvate_small_molecule itself uses for these Amber-scaled ligand XMLs.
#   * FeCl4: ASH's automatic re-imaging fails for a solute without bonds, so the last NPT frame
#     is re-imaged on the solute with image_on_solute.py.
#
# Requirements: ORCA (Molcrys charge iterations) and the OpenBabel Python module (solvate.py of
# the OpenFF example). The MM equilibrations are stochastic, so regenerated boxes differ in detail.
#
# Environment: PYTHON (interpreter with ASH's dependencies, default python3), ORCA_DIR (default:
# directory of the orca found on PATH), ASH_ROOT (default: the checkout containing this folder).
#
# Usage: prepare_inputs.sh WORKDIR
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASH_ROOT="${ASH_ROOT:-$(cd "$HERE/../.." && pwd)}"
PY="${PYTHON:-python3}"
command -v orca >/dev/null || [[ -n "${ORCA_DIR:-}" ]] || { echo "orca not on PATH; set ORCA_DIR" >&2; exit 1; }
ORCA_DIR="${ORCA_DIR:-$(dirname "$(command -v orca)")}"
WORK="${1:?usage: prepare_inputs.sh WORKDIR}"

export PYTHONPATH="$ASH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ORCA_DIR:$PATH"
export LD_LIBRARY_PATH="$ORCA_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$WORK"

solvated_example() {  # name example_dir solute_atoms
    local name="$1" src="$ASH_ROOT/examples/explicit-solvation/$2" natoms="$3" dir="$WORK/$1"
    mkdir -p "$dir" "$HERE/prepared/$name"
    cp "$src"/* "$dir"/
    sed -i 's#amber/tip3p_standard.xml#amber14/tip3p.xml#' "$dir"/run_mm_md.py
    ( cd "$dir" && "$PY" solvate.py > solvate.out 2>&1 && "$PY" run_mm_md.py > run_mm_md.out 2>&1 )
    # ASH writes its re-imaged last frame as equilibration_NPT_lastframe_imaged.pdb; the examples
    # call the same file equilibration_NPT_imaged.pdb.
    if [[ -f "$dir/equilibration_NPT_lastframe_imaged.pdb" ]]; then
        cp "$dir/equilibration_NPT_lastframe_imaged.pdb" "$HERE/prepared/$name/equilibration_NPT_imaged.pdb"
    else
        "$PY" "$HERE/image_on_solute.py" "$dir/equilibration_NPT_lastframe.pdb" \
            "$HERE/prepared/$name/equilibration_NPT_imaged.pdb" --solute-atoms "$natoms"
    fi
    echo "$name: prepared"
}

solvated_example isobutyraldehyde_water 1-expl-water-openff 13
solvated_example fecl4_water 2-expl-tmc-nonb 5

# Molcrys: run the example as written. Its cluster construction and ORCA charge iterations
# finish before the final QM/MM optimization, which needs the Julia interface and may stop there.
dir="$WORK/na2hpo4_molcrys"
mkdir -p "$dir" "$HERE/prepared/na2hpo4_molcrys"
cp "$ASH_ROOT"/examples/Molcrys-QM-MM-example/Na2HPO4-xtl-file/* "$dir"/
( cd "$dir" && "$PY" molcrys-test.py > molcrys.out 2>&1 ) || true
grep -q "Charges converged" "$dir/molcrys.out"
cp "$dir"/{Cluster.ygg,Cluster_forcefield.ff,Centralmainfrag} "$HERE/prepared/na2hpo4_molcrys/"
echo "na2hpo4_molcrys: prepared"
