import copy
import time
import numpy as np
import math

import ash.modules.module_coords
from ash.modules.module_coords import Fragment, write_pdbfile
from ash.functions.functions_general import ashexit, BC, blankline, listdiff, print_time_rel,printdebug,print_line_with_mainheader,writelisttofile,print_if_level
import ash.settings_ash
from ash.modules.module_QMMM import linkatom_force_adv,linkatom_force_chainrule,linkatom_force_lever,fullindex_to_qmindex

# Additive QM/ML theory object.
# Required at init: qm_theory and qmatoms and fragment