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
        if i in qmatoms && j in qmatoms
                continue
        else
           v = get(LJdict_jul, (atomtypes[i],atomtypes[j]), nothing)
           if v !== nothing
             sigmaij[i, j] = v[1]
             epsij[i, j] =  v[2]
           else
             v = get(LJdict_jul, (atomtypes[j],atomtypes[i]), nothing)
             if v !== nothing
               sigmaij[i, j] = v[1]
               epsij[i, j] =  v[2]
             end
           end
	    end
    end
end
return sigmaij,epsij
end

#Old version. More intuitive but slower
function pairpot2(numatoms,atomtypes,LJpydict,qmatoms)
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
             elseif haskey(LJdict_jul, (atomtypes[i],atomtypes[j]))
                 sigmaij[i, j] = LJdict_jul[(atomtypes[i], atomtypes[j])][1]
                 epsij[i, j] = LJdict_jul[(atomtypes[i], atomtypes[j])][2]
             elseif haskey(LJdict_jul, (atomtypes[j],atomtypes[i]))
                sigmaij[i, j] = LJdict_jul[(atomtypes[j], atomtypes[i])][1]
                epsij[i, j] = LJdict_jul[(atomtypes[j], atomtypes[i])][2]
             end
    end
end
return sigmaij,epsij
end




#End of Julia module
end