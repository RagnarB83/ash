import os
import shutil
from ash.functions.functions_general import writestringtofile, ashexit

#Basic VMD interface


def write_VMD_script_cube(cubefile=None,VMDfilename="VMD_script.vmd", isovalue=0.7):
    try:
        vmd_path = os.path.dirname(shutil.which('vmd'))
        vmd_path = vmd_path + "/vmd"
    except:
        vmd_path="/usr/local/bin/vmd"

    if cubefile == None:
        print("write_VMD_script_cube requires a cubefile")
        ashexit()

    vmd_settings_string=f"""#!{vmd_path}
# VMD script written by ASH

# Display settings
display projection   Orthographic
display depthcue   off

#Mol
mol new {cubefile} type cube first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
mol delrep 0 top
mol representation CPK 1.000000 0.300000 12.000000 12.000000
mol color Element
mol selection {{all}}
mol material Opaque
mol addrep top
mol selupdate 0 top 0
mol colupdate 0 top 0
mol scaleminmax top 0 0.000000 0.000000
mol smoothrep top 0 0
mol drawframes top 0 {{now}}
mol representation Isosurface {isovalue} 0 0 0 1 1
mol color ColorID 0
mol selection {{all}}
mol material Transparent
mol addrep top
mol selupdate 1 top 0
mol colupdate 1 top 0
mol scaleminmax top 1 0.000000 0.000000
mol smoothrep top 1 0
mol drawframes top 1 {{now}}
mol rename top {cubefile}

set topmol [molinfo top]
mol top $topmol
unset topmol
proc vmdrestoremycolors {{}} {{
  set colorcmds {{
    {{color Display {{Background}} white}}
    {{color Element {{Al}} purple}}
  }}
  foreach colcmd $colorcmds {{
    set val [catch {{eval $colcmd}}]
  }}
}}
vmdrestoremycolors

"""
    writestringtofile(vmd_settings_string,VMDfilename)