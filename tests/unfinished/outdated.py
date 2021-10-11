from ash import *

#Ash location for PyJulia
ashpath = os.path.dirname(ash.__file__)
print("ashpath:", ashpath)
#Necessary for statically linked libpython
from julia.api import Julia
jl = Julia(compiled_modules=False)
#Import Julia
from julia import Main

#Defining Julia Module
Main.include(ashpath+"/functions/functions_julia.jl")

#Calling functions

numatoms=2
atomtypes=["UFF_B", "UFF_H"]
LJpairpotentialsdict={("UFF_B", "UFF_H"): [3.6375394661670053, 0.18]}
qmatoms=[]
#Calling function pairpot3 inside module Main.Juliafunctions.pairpot3 inside Main object
sigmaij,epsij=Main.Juliafunctions.pairpot_full(numatoms,atomtypes,LJpairpotentialsdict,qmatoms)

print("sigmaij:", sigmaij)
print("epsij:", epsij)
#Calling active-region pairpot function
actatoms=qmatoms
sigmaij,epsij=Main.Juliafunctions.pairpot_active(numatoms,atomtypes,LJpairpotentialsdict,qmatoms,actatoms)

print("sigmaij:", sigmaij)
print("epsij:", epsij)
