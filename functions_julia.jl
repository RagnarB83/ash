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

#Dict version
function pairpot3(numatoms,atomtypes,LJpydict,qmatoms)
    #Updating atom indices from 0 to 1 syntax
    qmatoms=[i+1 for i in qmatoms]
    #Convert Python dict to Julia dict with correct types
    LJdict_jul=convert(Dict{Tuple{String,String},Array{Float64,1}}, LJpydict)
    #println(typeof(LJdict_jul))
    sigmaij=zeros(numatoms, numatoms)
    epsij=zeros(numatoms, numatoms)
    println("starting for-loop")
for i in 1:numatoms
    for j in i+1:numatoms
        for (ljpot_types, ljpot_values) in LJdict_jul
            #if all(x in qmatoms for x in (i, j))
            if i in qmatoms && j in qmatoms
                #print("Skipping i-j pair", i,j, " as these are QM atoms")
                continue
            elseif atomtypes[i] == ljpot_types[1] && atomtypes[j] == ljpot_types[2]
                sigmaij[i, j] = ljpot_values[1]
                epsij[i, j] = ljpot_values[2]
            elseif atomtypes[j] == ljpot_types[1] && atomtypes[i] == ljpot_types[2]
                sigmaij[i, j] = ljpot_values[1]
                epsij[i, j] = ljpot_values[2]
            #Skipping if i-j pair in qmatoms list. I.e. not doing QM-QM LJ calc.
            #tuple much faster than list
            #https://stackoverflow.com/questions/46576037/my-loops-are-slow-is-that-because-of-if-statements
            end
        end
    end
end
return sigmaij,epsij
end





#End of Julia module
end