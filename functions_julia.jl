#Some Julia functions
__precompile__()

module Juliafunctions
#using Profile
#using PyCall
#using Distances

#TODO:
# We are not utilizing Julia column-major much. Latter supposedly better
#Tried for connectivity, no difference
# Try maybe also for Lennard_jones? Pairpot arrays??
# Read more:
# https://julialang.org/blog/2013/09/fast-numeric/


#Connectivity (fraglists) for whole fragment
function calc_connectivity(coords,elems,conndepth,scale, tol,eldict_covrad)
	#0-index based atomlist
	atomlist=[0:length(elems)-1;]
	return calc_fraglist_for_atoms(atomlist,coords, elems, conndepth, scale, tol,eldict_covrad)
end

#Get fraglist for list of atoms (called by molcrys directly). Using 0-based indexing until get_conn_atoms
function calc_fraglist_for_atoms(atomlist,coords, elems, conndepth, scale, tol,eldict_covrad)
	eldict_covrad_jul=convert(Dict{String,Float64}, eldict_covrad)
	found_atoms = Int64[]
	#List of lists
	fraglist = Array{Int64}[]
	for atom in atomlist
		if atom ∉ found_atoms
			members = get_molecule_members_julia(coords, elems, conndepth, scale, tol, eldict_covrad_jul, atom)
			if members ∉ fraglist
				push!(fraglist,members)
				found_atoms = [found_atoms;members]
			end
		end
	end
	return fraglist
end



#Distance between atom i and j in coords
function distance(coords::Array{Float64,2},i::Int64,j::Int64)
			@fastmath @inbounds rij_x = coords[i,1] - coords[j,1]
            @fastmath @inbounds rij_y = coords[i,2] - coords[j,2]
            @fastmath @inbounds rij_z = coords[i,3] - coords[j,3]
            @fastmath r = rij_x*rij_x+rij_y*rij_y+rij_z*rij_z
            @fastmath dist = sqrt(r)
			return dist
end

#Distance between atom i and j in coords
#Using view instead. Seems to be slower
function distance_view(coords::Array{Float64,2},i::Int64,j::Int64)
			@fastmath @inbounds rij_x = view(coords,i,1)[1] - view(coords,j,1)[1]
            @fastmath @inbounds rij_y = view(coords,i,2)[1] - view(coords,j,2)[1]
            @fastmath @inbounds rij_z = view(coords,i,3)[1] - view(coords,j,3)[1]
            @fastmath r = rij_x*rij_x+rij_y*rij_y+rij_z*rij_z
            @fastmath dist = sqrt(r)
			return dist
end

#get_molecule_members_julia now wants eldict_covrad_dict to be a Julia object from beginning
#Means we need a wrapper for Python to call directly (like calc_fraglist_for_atoms) to convert dictionary
#Does not help with speed it seems though.
function get_molecule_members_julia(coords, elems, loopnumber, scale, tol, eldict_covrad_jul, atomindex)
    #eldict_covrad_jul=convert(Dict{String,Float64}, eldict_covrad)
	membs = Int64[]
	membs = get_connected_atoms_julia(coords, elems, eldict_covrad_jul, scale, tol, atomindex)
	finalmembs = membs
	for i in 1:loopnumber
		# Get list of lists of connatoms for each member
		newmembers = Int64[]
		for k in membs
			new = get_connected_atoms_julia(coords, elems, eldict_covrad_jul, scale, tol, k)
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
function get_connected_atoms_julia(coords::Array{Float64,2}, elems::Array{String,1},
    eldict_covrad_jul::Dict{String,Float64},scale::Float64,tol::Float64, atomindex::Int64)
    connatoms = Int64[]
    @inbounds elem_ref=elems[atomindex+1]
    @inbounds for i=1:length(elems)
			@inbounds dist = distance(coords,i,atomindex+1)
			#dist = euclidean(coords[i],coords[atomindex+1])
			#dist = euclidean(view(coords,i,:),view(coords,atomindex+1,:))
			@fastmath @inbounds rad_dist = scale*(eldict_covrad_jul[elems[i]]+eldict_covrad_jul[elem_ref]) + tol
        	if dist < rad_dist
            	@inbounds @fastmath push!(connatoms, i-1)
			end
	end
    return connatoms
end



#Lennard-Jones+Coulomb function.
#Tested. Faster than gcc-compiled Fortran function (LJCoulombv1.f90)
function LJcoulombchargev1a(charges, coords, epsij, sigmaij, connectivity=nothing)
    """Fast LJ + Coulomb function"""
    ang2bohr = 1.88972612546
	bohr2ang = 0.52917721067
	hartokcal = 627.50946900
	coords_b=coords*ang2bohr
    num=length(charges)
    VC=0.0
    VLJ=0.0
    gradient = zeros(size(coords_b)[1], 3)
    @inbounds for j=1:num
        @inbounds for i=j+1:num
            sigma=sigmaij[j,i]
            eps=epsij[j,i]
            rij_x = coords_b[i,1] - coords_b[j,1]
            rij_y = coords_b[i,2] - coords_b[j,2]
            rij_z = coords_b[i,3] - coords_b[j,3]
            @fastmath r = rij_x*rij_x+rij_y*rij_y+rij_z*rij_z
            @fastmath d = sqrt(r)
			@fastmath d_ang = d / ang2bohr
            @fastmath ri=1/r
            @fastmath ri3=ri*ri*ri
            @fastmath VC+=charges[i]*charges[j]/(d)
            @fastmath VLJ+=4*eps*((sigma/d_ang)^12-(sigma/d_ang)^6)
			@fastmath kC=charges[i]*charges[j]*sqrt(ri3)
            @fastmath kLJ=-1*(1/hartokcal)*bohr2ang*bohr2ang*((24*eps*((sigma/d_ang)^6-2*(sigma/d_ang)^12))*(1/(d_ang^2)))
            @fastmath Gij_x=(kLJ+kC)*rij_x
            @fastmath Gij_y=(kLJ+kC)*rij_y
            @fastmath Gij_z=(kLJ+kC)*rij_z

            gradient[j,1] +=  Gij_x
            gradient[j,2] +=  Gij_y
            gradient[j,3] +=  Gij_z
            gradient[i,1] -=  Gij_x
            gradient[i,2] -=  Gij_y
            gradient[i,3] -=  Gij_z
        end
    end
    E = VC + VLJ/hartokcal
    return E, gradient, VLJ/hartokcal, VC
end

#Lennard-Jones+Coulomb function.
#Tested. Marginally better than v1a
function LJcoulombchargev1c(charges, coords, epsij, sigmaij, connectivity=nothing)
    """LJ + Coulomb function"""
    ang2bohr = 1.88972612546
    bohr2ang = 0.52917721067
    hartokcal = 627.50946900
    coords_b=coords*ang2bohr
    num=length(charges)
    VC=0.0
    VLJ=0.0
    gradient = zeros(size(coords_b)[1], 3)
    constant=-1*(1/hartokcal)*bohr2ang*bohr2ang

    @inbounds for j in 1:num
        for i in j+1:num
            @inbounds sigma=sigmaij[j,i]
            @inbounds eps=epsij[j,i]
            @inbounds rij_x = coords_b[i,1] - coords_b[j,1]
            @inbounds rij_y = coords_b[i,2] - coords_b[j,2]
            @inbounds rij_z = coords_b[i,3] - coords_b[j,3]
            @fastmath r = rij_x*rij_x+rij_y*rij_y+rij_z*rij_z
            @fastmath d = sqrt(r)
            @fastmath d_ang = d / ang2bohr
            @fastmath ri=1/r
            @fastmath ri3=ri*ri*ri
            @inbounds @fastmath VC += charges[i] * charges[j] / (d)
            @inbounds @fastmath VLJ += 4.0 * eps * ((sigma / d_ang)^12 - (sigma / d_ang)^6 )
            @inbounds @fastmath kC=charges[i]*charges[j]*sqrt(ri3)
            @inbounds @fastmath kLJ=constant*((24*eps*((sigma/d_ang)^6-2*(sigma/d_ang)^12))*(1/(d_ang^2)))
            @fastmath k=kLJ+kC
            @fastmath Gij_x=k*rij_x
            @fastmath Gij_y=k*rij_y
            @fastmath Gij_z=k*rij_z

            gradient[j,1] +=  Gij_x
            gradient[j,2] +=  Gij_y
            gradient[j,3] +=  Gij_z
            gradient[i,1] -=  Gij_x
            gradient[i,2] -=  Gij_y
            gradient[i,3] -=  Gij_z
        end
    end
    E = VC + VLJ/hartokcal
    return E, gradient, VLJ/hartokcal, VC
end

#Lennard-Jones+Coulomb function.
#Testing view slices.
#Not faster it seems. To be deleted...
function LJcoulombchargev1d(charges, coords, epsij, sigmaij, connectivity=nothing)
    """LJ + Coulomb function"""
    ang2bohr = 1.88972612546
    bohr2ang = 0.52917721067
    hartokcal = 627.50946900
    coords_b=coords*ang2bohr
    num=length(charges)
    VC=0.0
    VLJ=0.0
    gradient = zeros(size(coords_b)[1], 3)
    constant=-1*(1/hartokcal)*bohr2ang*bohr2ang

    @inbounds for j in 1:num
        for i in j+1:num
            @inbounds sigma=view(sigmaij,j,i)[1]
            @inbounds eps=view(epsij,j,i)[1]
            @inbounds rij_x = view(coords_b,i,1)[1] - view(coords_b,j,1)[1]
            @inbounds rij_y = view(coords_b,i,2)[1] - view(coords_b,j,2)[1]
            @inbounds rij_z = view(coords_b,i,3)[1] - view(coords_b,j,3)[1]
            @fastmath r = rij_x*rij_x+rij_y*rij_y+rij_z*rij_z
            @fastmath d = sqrt(r)
            @fastmath d_ang = d / ang2bohr
            @fastmath ri=1/r
            @fastmath ri3=ri*ri*ri
            @inbounds @fastmath VC += charges[i] * charges[j] / (d)
            @inbounds @fastmath VLJ += 4.0 * eps * ((sigma / d_ang)^12 - (sigma / d_ang)^6 )
            @inbounds @fastmath kC=charges[i]*charges[j]*sqrt(ri3)
            @inbounds @fastmath kLJ=constant*((24*eps*((sigma/d_ang)^6-2*(sigma/d_ang)^12))*(1/(d_ang^2)))
            @fastmath k=kLJ+kC
            @fastmath Gij_x=k*rij_x
            @fastmath Gij_y=k*rij_y
            @fastmath Gij_z=k*rij_z

            gradient[j,1] +=  Gij_x
            gradient[j,2] +=  Gij_y
            gradient[j,3] +=  Gij_z
            gradient[i,1] -=  Gij_x
            gradient[i,2] -=  Gij_y
            gradient[i,3] -=  Gij_z
        end
    end
    E = VC + VLJ/hartokcal
    return E, gradient, VLJ/hartokcal, VC
end


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
#Fills whole symmetric array just in case,i .e. ij and ji
function pairpot_active(numatoms,atomtypes,LJpydict,qmatoms,actatoms)
    #Updating atom indices from 0 to 1 syntax
    qmatoms=[i+1 for i in qmatoms]
    actatoms=[i+1 for i in actatoms]
    #Convert Python dict to Julia dict with correct types
    LJdict_jul=convert(Dict{Tuple{String,String},Array{Float64,1}}, LJpydict)
    sigmaij=zeros(numatoms, numatoms)
    epsij=zeros(numatoms, numatoms)
	for i in 1:numatoms
		for j in actatoms
			if i in qmatoms && j in qmatoms
				continue
			else
			   #Checking if dict contains key, return value if so, otherwise nothing
			   #Todo: what if we have v be value or 0 instead of nothing. Can then skip the if statement?
			   v = get(LJdict_jul, (atomtypes[i],atomtypes[j]), nothing)
			   if v !== nothing
				 sigmaij[i, j] = v[1]
				 epsij[i, j] =  v[2]
				 sigmaij[j, i] = v[1]
				 epsij[j, i] =  v[2]
			   else
				 v = get(LJdict_jul, (atomtypes[j],atomtypes[i]), nothing)
				 if v !== nothing
				   sigmaij[i, j] = v[1]
				   epsij[i, j] =  v[2]
				   sigmaij[j, i] = v[1]
				   epsij[j, i] =  v[2]
				 end
			   end
			end
		end
	end
	return sigmaij,epsij
	end



#End of Julia module
end