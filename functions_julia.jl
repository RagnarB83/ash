#Some Julia functions
__precompile__()

module Juliafunctions
#using PyCall

function hellofromjulia()
println("Hello from Julia")
end

function juliatest(list)
println("Inside juliatest")
println("list is : $list")
var=5.4
return var
end

#Calculate the sigmaij and epsij arrays
function pairpot3(numatoms,atomtypes,LJpydict)
    println("numatoms: $numatoms")
    println("atomtypes: $atomtypes")
    println("LJpydict : $LJpydict")
    println(typeof(LJpydict))
    #Convert Python dict to Julia dict with correct types
    LJdict_jul=convert(Dict{Tuple{String,String},Array{Float64,1}}, LJpydict)
    println("LJdict_jul : $LJdict_jul")
        println(typeof(LJdict_jul))
    sigmaij=zeros(numatoms, numatoms)
    epsij=zeros(numatoms, numatoms)
for i in 1:numatoms
    for j in 1:numatoms
        for (ljpot_types, ljpot_values) in LJdict_jul
            if atomtypes[i] == ljpot_types[1] && atomtypes[j] == ljpot_types[2]
                sigmaij[i, j] = ljpot_values[1]
                epsij[i, j] = ljpot_values[2]
            elseif atomtypes[j] == ljpot_types[1] && atomtypes[i] == ljpot_types[2]
                sigmaij[i, j] = ljpot_values[1]
                epsij[i, j] = ljpot_values[2]
            end
        end
    end
end
#println(sigmaij[1,1])
#println(epsij[1,1])
#println("--")
#println(sigmaij[500,670])
#println(epsij[500,670])

return sigmaij,epsij
end


#End of Julia module
end