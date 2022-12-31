from ash import *

h2string="""
H 0 0 0
H 0 0 0.7
"""

h2=Fragment(coordsstring=h2string)

#List of functional keywords (strings) to loop over
functionals=['BP86', 'B3LYP', 'TPSS', 'TPSSh', 'PBE0', 'BHLYP', 'CAM-B3LYP']

#Dictionary to keep track of results
energies_dict={}

for functional in functionals:
    print("FUNCTIONAL: ", functional)
    orcadir='/opt/orca_4.2.1'
    #Appending functional keyword to the string-variable that contains the ORCA inputline
    input="! def2-SVP Grid5 Finalgrid6 tightscf slowconv " + functional
    blocks="""
    %scf
    maxiter 200
    end
    """
    #Defining/redefining ORCA theory. Does not need charge/mult keywords.
    ORCAcalc = ORCATheory(orcadir=orcadir, orcasimpleinput=input, orcablocks=blocks, numcores=1, charge=0, mult=1)
    ORCAcalc.cleanup()
    # Run single-point job
    result = Singlepoint(theory=ORCAcalc, fragment=h2)
    energies_dict[functional] = result.energy    

    #Cleaning up after each job
    ORCAcalc.cleanup()
    print("=================================")

print("Dictionary with results:", energies_dict)
print("")
#Pretty formatted printing
print("")
print(" Functional   Energy (Eh)")
print("---------------------")
for func, e in energies_dict.items():
    print("{:10} {:13.10f}".format(func,e))
