#=
unusedJuliacode:
- Julia version: 1.4.0
- Author: bjornssonsu
- Date: 2020-07-27
=#







#Connectivity functions below using column-major order
# Tried various things:
# Column-major. no difference
# View, does not make a difference
# Pre-calculated arrays, distances (pairwise) and thresholds (rad_dist_array), too slow.
#Fully vectorized (get_connected_atoms_julia_col2), too slow


#Connectivity (fraglists) for whole fragment
function calc_connectivity_col(coords,elems,conndepth,scale, tol,eldict_covrad)
	println("Inside calc_connectivity_col (julia)")
	#0-index based atomlist
	atomlist=[0:length(elems)-1;]
	return calc_fraglist_for_atoms_col(atomlist,coords, elems, conndepth, scale, tol,eldict_covrad)
end


#Get fraglist for list of atoms (called by molcrys directly). Using 0-based indexing until get_conn_atoms
#Col-major version
# Testing to pre-calculate distances instead of doing insicde get_conn_atoms
function calc_fraglist_for_atoms_col(atomlist,coords, elems, conndepth, scale, tol,eldict_covrad)
	eldict_covrad_jul=convert(Dict{String,Float64}, eldict_covrad)
	found_atoms = Int64[]
	#List of lists
	fraglist = Array{Int64}[]
	#Using permutedims instead of transpose to get correct type
	@time coords_trans = permutedims(coords)
	#Calculating all distances
	@inbounds @time distances = pairwise(Euclidean(), coords_trans, coords_trans, dims=2)
	@time rad_dist_array=zeros(length(elems), length(elems))
	#Calculate thresholds beforehand
	for i in 1:length(elems)
		for j in 1:length(elems)
			@inbounds rad_dist_array[j,i] = scale*(eldict_covrad_jul[elems[j]]+eldict_covrad_jul[elems[i]]) + tol
		end
	end
	#println("rad_dist_array :", typeof(rad_dist_array))
	#exit()
	#exit()
	for atom in atomlist
		if atom ∉ found_atoms
			members = get_molecule_members_julia_col(coords_trans, elems, conndepth, scale, tol, eldict_covrad_jul, atom,distances,rad_dist_array)
			if members ∉ fraglist
				push!(fraglist,members)
				found_atoms = [found_atoms;members]
			end
		end
	end
	return fraglist
end

#get_molecule_members_julia now wants eldict_covrad_dict to be a Julia object from beginning
#Means we need a wrapper for Python to call directly (like calc_fraglist_for_atoms) to convert dictionary
#Does not help with speed it seems though.
function get_molecule_members_julia_col(coords, elems, loopnumber, scale, tol, eldict_covrad_jul, atomindex,distances,rad_dist_array)
    #eldict_covrad_jul=convert(Dict{String,Float64}, eldict_covrad)
	membs = Int64[]
	membs = get_connected_atoms_julia_col(coords, elems, eldict_covrad_jul, scale, tol, atomindex,distances,rad_dist_array)
	finalmembs = membs
	for i in 1:loopnumber
		# Get list of lists of connatoms for each member
		newmembers = Int64[]
		for k in membs
			new = get_connected_atoms_julia_col(coords, elems, eldict_covrad_jul, scale, tol, k, distances,rad_dist_array)
			newmembers = [newmembers;new]
		end
		# Get a unique flat list
		trimmed_flat = sort(unique(newmembers))
		# Check if new atoms not previously found
		membs = setdiff(trimmed_flat, finalmembs)
		if length(membs) == 0
			return finalmembs
		end
		finalmembs = [finalmembs;membs]
		finalmembs = sort(unique(finalmembs))
	end
	return finalmembs
end

#Here accessing Julia arrays. Switching from 0-based to 1-based indexing here
function get_connected_atoms_julia_col(coords::Array{Float64,2}, elems::Array{String,1},
    eldict_covrad_jul::Dict{String,Float64},scale::Float64,tol::Float64, atomindex::Int64, distances::Array{Float64,2}, rad_dist_array::Array{Float64,2})
    connatoms = Int64[]
    #@inbounds elem_ref=elems[atomindex+1]
    @inbounds for i=1:length(elems)
			@inbounds dist = distances[i,atomindex+1]
			#dist = euclidean(coords[i],coords[atomindex+1])
			#dist = euclidean(view(coords,i,:),view(coords,atomindex+1,:))
			#@fastmath @inbounds rad_dist = scale*(eldict_covrad_jul[elems[i]]+eldict_covrad_jul[elem_ref]) + tol
			@inbounds rad_dist = rad_dist_array[i,atomindex+1]
        	if dist < rad_dist
            @inbounds @fastmath push!(connatoms, i-1)
			#exit()
			end
	end
    return connatoms
end

#Here accessing Julia arrays. Switching from 0-based to 1-based indexing here
#Fully vectorized. Slow, not useful
function get_connected_atoms_julia_col2(coords::Array{Float64,2}, elems::Array{String,1},
    eldict_covrad_jul::Dict{String,Float64},scale::Float64,tol::Float64, atomindex::Int64)
    connatoms = Int64[]
	@inbounds dists = colwise(Euclidean(), coords, coords[:,atomindex+1])
    @inbounds elem_ref=elems[atomindex+1]
	@inbounds el_covrad_ref=eldict_covrad_jul[elems[atomindex+1]]
	@inbounds thresholds=[eldict_covrad_jul[elems[i]] for i in 1:length(elems)]
    @fastmath thresholds=thresholds .+ el_covrad_ref
    @fastmath thresholds=thresholds .* scale
    @fastmath thresholds=thresholds .+ tol
	@fastmath diff=dists-thresholds
	connatoms = [index-1 for (index,x) in enumerate(diff) if x < 0.0]
    return connatoms
end

#Distance between atom i and j in coords
function distance_col(coords::Array{Float64,2},i::Int64,j::Int64)
			@fastmath @inbounds rij_x = coords[1,i] - coords[1,j]
            @fastmath @inbounds rij_y = coords[2,i] - coords[2,j]
            @fastmath @inbounds rij_z = coords[3,i] - coords[3,j]
            @fastmath r = rij_x*rij_x+rij_y*rij_y+rij_z*rij_z
            @fastmath dist = sqrt(r)
			return dist
end

#Distance between atom i and j in coords
#Using view instead. Seems to be slower
function distance_view_col(coords::Array{Float64,2},i::Int64,j::Int64)
			@fastmath @inbounds rij_x = view(coords,1,i)[1] - view(coords,1,j)[1]
            @fastmath @inbounds rij_y = view(coords,2,i)[1] - view(coords,2,j)[1]
            @fastmath @inbounds rij_z = view(coords,3,i)[1] - view(coords,3,j)[1]
            @fastmath r = rij_x*rij_x+rij_y*rij_y+rij_z*rij_z
            @fastmath dist = sqrt(r)
			return dist
end





#Connectivity entirely via Julia
#Old: Delete
function old_calc_connectivity(coords,elems,conndepth,scale, tol,eldict_covrad)
    # Calculate connectivity by looping over all atoms
	found_atoms = Int64[]
	#List of lists
	fraglist = Array{Int64}[]
	#println(typeof(fraglist))
    #Looping over atoms
	for atom in 1:length(elems)
		if length(found_atoms) == length(elems)
			println("All atoms accounted for. Exiting...")
			return fraglist
		end
		if atom-1 ∉ found_atoms
			members = get_molecule_members_julia(coords, elems, conndepth, scale, tol, eldict_covrad, atomindex=atom-1)
			if members ∉ fraglist
				push!(fraglist,members)
				found_atoms = [found_atoms;members]
			end
		end
	end
	return fraglist
end

#Compute distances between each pair of two arrays, mimics scipy.spatial.distance.cdist
#This does col by col by first transposing
#Slow compared to distance_array due to slicing.
#Could be replaced by view but element-wise version in distance_array is probably best
function distance_array_col(XA,XB)
    XAt=permutedims(XA)
    XBt=permutedims(XB)
    distances=zeros(size(XA,1),size(XA,1))
    @inbounds for i in 1:size(XA,1)
        @inbounds for j in 1:size(XB,1)
                #This slicing is really slow
                @inbounds col_a = XAt[:,i]
                @inbounds col_b = XBt[:,j]
                 @inbounds dist = distance_two_vectors(col_a,col_b)
                 @inbounds distances[i,j] = dist
            end
        end
    return distances
end
#This does row by row. Same problem as col version
function distance_array_row(XA,XB)
    distances=zeros(size(XA,1),size(XA,1))
    for i in 1:size(XA,1)
        for j in 1:size(XB,1)
                #This slicing is really slow
                row_a=XA[i,:]
                row_b=XB[j,:]
                 @inbounds dist = distance_two_vectors(row_a,row_b)
                 distances[i,j] = dist
        end
    end
    return distances
end