#Some Julia functions
__precompile__()

module Juliafunctions

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
function pairpot3(numatoms,atomtypes,LJpairpotentialsdict)
println("numatoms: $numatoms")
exit()
sigmaij=zeros(numatoms, numatoms)
epsij=zeros(numatoms, numatoms)
for i in 1:numatoms
    for j in 1:numatoms
        for (ljpot_types, ljpot_values) in LJpairpotentialsdict
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
end
return sigmaij,epsij


end