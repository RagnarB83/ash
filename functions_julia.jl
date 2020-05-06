#Some Julia functions
__precompile__()

module Juliafunctions
#using PyCall

function hellofromjulia()
    println("Hello from Julia")
end

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
# Avoided dict-lookup for both key-existence and value
#https://stackoverflow.com/questions/58170034/how-do-i-check-if-a-dictionary-has-a-key-in-it-in-julia
#frozenatoms option is slow
function pairpot_full(numatoms,atomtypes,LJpydict,qmatoms)
    #Updating atom indices from 0 to 1 syntax
    qmatoms=[i+1 for i in qmatoms]
    #frozenatoms=[i+1 for i in frozenatoms]
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
           #Checking if dict contains key, return value if so, otherwise nothing
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

#Modified pairpot that only does active atoms
function pairpot_active(numatoms,atomtypes,LJpydict,qmatoms,actatoms)
	println("inside pairpot_active")
    #Updating atom indices from 0 to 1 syntax
    qmatoms=[i+1 for i in qmatoms]
    actatoms=[i+1 for i in actatoms]
    #Convert Python dict to Julia dict with correct types
    LJdict_jul=convert(Dict{Tuple{String,String},Array{Float64,1}}, LJpydict)
    #println(typeof(LJdict_jul))
    sigmaij=zeros(numatoms, numatoms)
    epsij=zeros(numatoms, numatoms)
	println("-----")
	println("qmatoms : $qmatoms")
	println("actatoms: $actatoms")
	println("-----")
	println("numatoms: $numatoms")
	for i in actatoms
		for j in 1:numatoms
			#println("i is $i and j is $j")
			#println("count_i is $count_i")
			#println("atomtypes[i]", atomtypes[i])
			#println("atomtypes[j]", atomtypes[j])
			if i in qmatoms && j in qmatoms
				continue
			else
				#println("else")
			   #Checking if dict contains key, return value if so, otherwise nothing
			   v = get(LJdict_jul, (atomtypes[i],atomtypes[j]), nothing)
			   if v !== nothing
				 sigmaij[i, j] = v[1]
				 epsij[i, j] =  v[2]
				#println("here")
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

#Distance for 2D arrays of coords
function distance(x::Array{Float64, 2}, y::Array{Float64, 2})
    nx = size(x, 1)
    ny = size(y, 1)
    r=zeros(nx,ny)
        for j = 1:ny
            @fastmath for i = 1:nx
                @inbounds dx = y[j, 1] - x[i, 1]
                @inbounds dy = y[j, 2] - x[i, 2]
                @inbounds dz = y[j, 3] - x[i, 3]
                rSq = dx*dx + dy*dy + dz*dz
                @inbounds r[i, j] = sqrt(rSq)
            end
        end
    return r
end

#function get_connected_atoms_julia(coords, elems,eldict_covrad, scale,tol, atomindex):
#    eldict_covrad_jul=convert(Dict{String,Float64}, eldict_covrad)
#    connatoms = Array(Int, 0)
#    coords_ref=coords[atomindex]
#    elem_ref=elems[atomindex]

#    for (i,c) in enumerate(coords)
#        if distance(coords_ref,c) < scale*(eldict_covrad_jul[elems[i]]+eldict_covrad_jul[elem_ref]) + tol
#            push!(connatoms, i)
#TODO: remove atomindex from connatoms
#    return connatoms
#end


#Python-ish version
#function get_connected_atoms_julia_vector(coords, elems,eldict_covrad, scale,tol, atomindex)
#    eldict_covrad_jul=convert(Dict{String,Float64}, eldict_covrad)

    #Pre-compute Euclidean distance array
#    dists=distance(coords,coords)

    #Getting all thresholds as list via list comprehension.
#    el_covrad_ref=eldict_covrad[elems[atomindex]]
    # TODO: Slowest part but hard to make faster
#    thresholds=np.array([eldict_covrad[elems[i]] for i in range(len(elems))])
    #Numpy addition and multiplication done on whole array
#    thresholds=thresholds+el_covrad_ref
#    thresholds=thresholds*scale
#    thresholds=thresholds+tol
    #Old slow way
    #thresholds=np.array([threshold_conn(elems[i], elem_ref,scale,tol) for i in range(len(elems))])
    #Getting difference of distances and thresholds
#    diff=distances-thresholds

#    connatoms = []
    #Getting connatoms by finding indices of diff with negative values (i.e. where distance is smaller than threshold)
#    connatoms=np.where(diff<0)[0].tolist()
#    return connatoms
#end



#End of Julia module
end