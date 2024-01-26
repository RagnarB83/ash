from ash import *

moldenfiles=glob.glob("*molden*")

diff_cubefiles, num_el_vals, num_el_vals_pos, num_el_vals_neg = diffdens_tool(reference_orbfile="HF.molden",
    dir='.', grid=3, printlevel=2)
