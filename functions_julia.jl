#Some Julia functions
__precompile__()

module Juliafunctions
#using PyCall


#Dummy function
function juliatest(list)
println("Inside juliatest")
println("list is : $list")
var=5.4
return var
end

#TODO functions:
#Rewrite connectivity in Julia here
#Maybe some molcrys cluster-create,delete steps??


#Calculate the sigmaij and epsij arrays
#Key things for speed:
# i:numatoms, j=i+1:numatoms
# Using fast dict-lookup, simple double-if condition for qmatoms (was slowing things down a lot with all thing)
function pairpot3(numatoms,atomtypes,LJpydict,qmatoms)
    #Updating atom indices from 0 to 1 syntax
    qmatoms=[i+1 for i in qmatoms]
    #Convert Python dict to Julia dict with correct types
    LJdict_jul=convert(Dict{Tuple{String,String},Array{Float64,1}}, LJpydict)
    #println(typeof(LJdict_jul))
    sigmaij=zeros(numatoms, numatoms)
    epsij=zeros(numatoms, numatoms)
for i in 1:numatoms
    for j in i+1:numatoms
            #if all(x in qmatoms for x in (i, j))
            if i in qmatoms && j in qmatoms
                #print("Skipping i-j pair", i,j, " as these are QM atoms")
                continue
            end
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
return sigmaij,epsij
end




#End of Julia module
end