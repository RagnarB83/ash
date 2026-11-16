"""Run ASH's bundled electric-field-gradient benchmark set."""

import os

from ash import ORCATheory
from ash.modules.module_benchmarking import run_benchmark


# Set ORCADIR if ORCA is not configured in ash_user_settings.ini or PATH.
theory = ORCATheory(
    orcadir=os.environ.get("ORCADIR"),
    orcasimpleinput="! BP86 def2-SVP TightSCF",
    numcores=1,
    filename="orca_efg",
)

run_benchmark(
    set="EFG-Bjornsson",
    theory=theory,
    numcores=1,
    keepoutputfiles=True,
)
