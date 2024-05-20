from ash import *

moldenfiles=glob.glob("*molden*")

diffdens_tool(reference_orbfile="ICE_CI_mp2nat_tgen_1e-06.molden",
    dir='.', grid=3, printlevel=2)
