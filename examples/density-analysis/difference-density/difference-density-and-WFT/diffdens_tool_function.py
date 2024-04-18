from ash import *

#This is an easier to use function to create multiple difference densities from ORCA GBW, NAT or MOLDEN files
#than the diffdens-from-GBW-and-NAT-files.py script

diffdens_tool(reference_orbfile="HF.gbw", dir='.', grid=3, printlevel=1)
