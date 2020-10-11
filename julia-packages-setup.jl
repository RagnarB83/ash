#Simple Julia script that sets up the necessary dependencies for ASH.
#Needs to be run just once
using Pkg

#Julia-Python interface: https://github.com/JuliaPy/PyCall.jl
Pkg.add("PyCall")
# Hungarian assignment. https://github.com/Gnimuc/Hungarian.jl
Pkg.add("Hungarian")
#Distances package. https://github.com/JuliaStats/Distances.jl
Pkg.add("Distances")


# precompile dependencies
#using PackageCompiler
#create_sysimage([:Glob, :Plots, :GLM, :DataFrames], 
#                 sysimage_path = "jexio_deps.so")

exit()