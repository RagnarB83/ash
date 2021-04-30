#Simple Julia script that sets up the necessary dependencies for ASH.
#Needs to be run just once
using Pkg


#Julia-Python interface: https://github.com/JuliaPy/PyCall.jl
Pkg.add("PyCall")
# Hungarian assignment. https://github.com/Gnimuc/Hungarian.jl
Pkg.add("Hungarian")
#Distances package. https://github.com/JuliaStats/Distances.jl
Pkg.add("Distances")
#LoopVectorization package. https://github.com/JuliaSIMD/LoopVectorization.jl
Pkg.add("LoopVectorization")
#Tullio package. https://github.com/mcabbott/Tullio.jl
Pkg.add("Tullio")

#For Python-Julia to work properly one may have to set ENV_PYTHON below (path to python version) and then rebuild PyCall
#ENV["PYTHON"] = "/Users/bjornssonsu/anaconda3/bin/python3"
#Pkg.build("PyCall")

# precompile dependencies
#using PackageCompiler
#create_sysimage([:Glob, :Plots, :GLM, :DataFrames], 
#                 sysimage_path = "jexio_deps.so")

exit()