import subprocess as sp
import os
import shutil
import platform
from ash.functions.functions_general import BC,ashexit, writestringtofile, pygrep, print_line_with_mainheader

#Interface to ezFCF (https://iopenshell.usc.edu/downloads)
#
#To acknowledge ezFCF, use the following citation: 
#S. Gozem, A. I. Krylov, "The ezSpectra suite: An easy-to-use toolkit for spectroscopy modeling", WIRES CMS, e1546 (2021) and P. Wojcik, S. Gozem, V. Mozhayskiy, and A.I. Krylov, http://iopenshell.usc.edu/downloads


def ezFCF_run(input_file="ezFCF.xml", output_file="ezFCF.out"):
    print_line_with_mainheader("ezFCF_run (ezFCF interface)")

    #Check platform
    if platform.system() == "Linux":
        binary_name = "ezFCF_linux.exe"
    elif platform.system() == "Darwin":
        binary_name = "ezFCF_mac.exe"

    #Check if ezFCF is installed
    print("Checking if ezFCF binary is available in PATH")
    print("Looking for executables named: ezFCF_linux.exe and ezFCF_mac.exe") 
    if shutil.which(binary_name) is None:
        print("ezFCF binary not found in PATH. Please install ezFCF and add the binary directory to PATH.")
    else:
        print(f"ezFCF binary {binary_name} found in PATH")

    #Input data
    #Lets create ORCA freq files

    #python3 make_xml.py bla.xml h2o.out h2o-ip.out

    #Create input file
    with open(input_file, "w") as inp:
        inp.write(xmldata)
    
    #Create masses file

    output = open(output_file, "w")
    sp.run([binary_name, input_file], stdout=output)
    output.close()