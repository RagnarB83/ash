import numpy as np
import math
import shutil
import os
import glob
import copy
import subprocess as sp

#import ash
import ash.constants
import ash.modules.module_coords
import ash.dictionaries_lists
from ash.functions.functions_general import ashexit, isodd, print_line_with_mainheader,pygrep,print_pretty_table
from ash.interfaces.interface_ORCA import ORCATheory, run_orca_plot, make_molden_file_ORCA
from ash.modules.module_coords import nucchargelist,elematomnumbers
from ash.dictionaries_lists import eldict
from ash.constants import hartokcal
from ash.interfaces.interface_multiwfn import multiwfn_run


#CM5. from https://github.com/patrickmelix/CM5-calculator/blob/master/cm5calculator.py

#data from paper for element 1-118
_radii = np.array([0.32, 0.37, 1.30, 0.99, 0.84, 0.75,
          0.71, 0.64, 0.60, 0.62, 1.60, 1.40,
          1.24, 1.14, 1.09, 1.04, 1.00, 1.01,
          2.00, 1.74, 1.59, 1.48, 1.44, 1.30,
          1.29, 1.24, 1.18, 1.17, 1.22, 1.20,
          1.23, 1.20, 1.20, 1.18, 1.17, 1.16,
          2.15, 1.90, 1.76, 1.64, 1.56, 1.46,
          1.38, 1.36, 1.34, 1.30, 1.36, 1.40,
          1.42, 1.40, 1.40, 1.37, 1.36, 1.36,
          2.38, 2.06, 1.94, 1.84, 1.90, 1.88,
          1.86, 1.85, 1.83, 1.82, 1.81, 1.80,
          1.79, 1.77, 1.77, 1.78, 1.74, 1.64,
          1.58, 1.50, 1.41, 1.36, 1.32, 1.30,
          1.30, 1.32, 1.44, 1.45, 1.50, 1.42,
          1.48, 1.46, 2.42, 2.11, 2.01, 1.90,
          1.84, 1.83, 1.80, 1.80, 1.73, 1.68,
          1.68, 1.68, 1.65, 1.67, 1.73, 1.76,
          1.61, 1.57, 1.49, 1.43, 1.41, 1.34,
          1.29, 1.28, 1.21, 1.22, 1.36, 1.43,
          1.62, 1.75, 1.65, 1.57])


_Dz = np.array([0.0056, -0.1543, 0.0000, 0.0333, -0.1030, -0.0446,
      -0.1072, -0.0802, -0.0629, -0.1088, 0.0184, 0.0000,
      -0.0726, -0.0790, -0.0756, -0.0565, -0.0444, -0.0767,
       0.0130, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      -0.0512, -0.0557, -0.0533, -0.0399, -0.0313, -0.0541,
       0.0092, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      -0.0361, -0.0393, -0.0376, -0.0281, -0.0220, -0.0381,
       0.0065, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, -0.0255, -0.0277, -0.0265, -0.0198,
      -0.0155, -0.0269, 0.0046, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, -0.0179, -0.0195,
      -0.0187, -0.0140, -0.0110, -0.0189])

_alpha = 2.474
_C = 0.705
_DHC = 0.0502
_DHN = 0.1747
_DHO = 0.1671
_DCN = 0.0556
_DCO = 0.0234
_DNO = -0.0346


#Get list-of-lists of distances of coords
def distance_matrix_from_coords(coords):
    distmatrix=[]
    for i in coords:
        dist_row=[ash.modules.module_coords.distance(i,j) for j in coords]
        distmatrix.append(dist_row)
    return distmatrix



def calc_cm5(atomicNumbers, coords, hirschfeldcharges):
    coords=np.array(coords)
    atomicNumbers=np.array(atomicNumbers)
    #all matrices have the naming scheme matrix[k,k'] according to the paper
    #distances = atoms.get_all_distances(mic=True)
    distances = np.array(distance_matrix_from_coords(coords))
    #print("distances:", distances)
    #atomicNumbers = np.array(atoms.numbers)
    #print("atomicNumbers", atomicNumbers)
    Rz = _radii[atomicNumbers-1]
    RzSum = np.tile(Rz,(len(Rz),1))
    RzSum = np.add(RzSum, np.transpose(RzSum))
    Bkk = np.exp(-_alpha * (np.subtract(distances,RzSum)), out=np.zeros_like(distances), where=distances!=0)
    assert (np.diagonal(Bkk) == 0).all()

#    Dz = _Dz[atomicNumbers]
#    Tkk = np.tile(Dz,(len(Dz),1))
#    Tkk = np.subtract(Tkk, np.transpose(Tkk))
    Tkk = np.zeros(shape=Bkk.shape)
    shape = Tkk.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            numbers = [atomicNumbers[i], atomicNumbers[j]]
            if numbers[0] == numbers[1]:
                continue
            if set(numbers) == set([1,6]):
                Tkk[i,j] = _DHC
                if numbers == [6,1]:
                    Tkk[i,j] *= -1.0
            elif set(numbers) == set([1,7]):
                Tkk[i,j] = _DHN
                if numbers == [7,1]:
                    Tkk[i,j] *= -1.0
            elif set(numbers) == set([1,8]):
                Tkk[i,j] = _DHO
                if numbers == [8,1]:
                    Tkk[i,j] *= -1.0
            elif set(numbers) == set([6,7]):
                Tkk[i,j] = _DCN
                if numbers == [7,6]:
                    Tkk[i,j] *= -1.0
            elif set(numbers) == set([6,8]):
                Tkk[i,j] = _DCO
                if numbers == [8,6]:
                    Tkk[i,j] *= -1.0
            elif set(numbers) == set([7,8]):
                Tkk[i,j] = _DNO
                if numbers == [8,7]:
                    Tkk[i,j] *= -1.0
            else:
                Tkk[i,j] = _Dz[numbers[0]-1] - _Dz[numbers[1]-1]
    assert (np.diagonal(Tkk) == 0).all()
    product = np.multiply(Tkk, Bkk)
    assert (np.diagonal(product) == 0).all()
    result = np.sum(product,axis=1)
    return np.array(hirschfeldcharges) + result


#Read cubefile.
#TODO: Clean up!
def read_cube (cubefile):
    bohrang = 0.52917721067
    LargePrint = True
    #Opening orbital cube file
    try:
        filename = cubefile
        a = open(filename,"r")
        print("Reading orbital file:", filename)
        filebase=os.path.splitext(filename)[0]
    except IndexError:
        print("error")
        quit()
    #Read cube file and get all data. Square values
    count = 0
    grabpoints = False
    grab_deset_id=False #Whether to grab line with DSET_IDs or not
    vals=[]
    elems=[]
    coords=[]
    coords_ang=[]
    numatoms=0
    for line in a:
        count += 1
        if count == 2:
            description=line
        #Grabbing origin
        if count == 3:
            #Getting possibly signed numatoms
            numat_orig=int(line.split()[0])
            if numat_orig < 0:
                #If negative then we have an ID line later with DSET_IDS
                grab_deset_id=True
            numatoms=abs(int(line.split()[0]))
            orgx=float(line.split()[1])
            orgy=float(line.split()[2])
            orgz=float(line.split()[3])
            rlowx=orgx; rlowy=orgy; rlowz=orgz
        if count == 4:
            nx=int(line.split()[0])
            dx=float(line.split()[1])
        if count == 5:
            ny=int(line.split()[0])
            dy=float(line.split()[2])
        if count == 6:
            nz=int(line.split()[0])
            dz=float(line.split()[3])
        #Grabbing molecular coordinates
        if count > 6 and count <= 6+numatoms:
            elems.append(int(line.split()[0]))
            coord=[float(line.split()[2]),float(line.split()[3]),float(line.split()[4])]
            coord_ang=[bohrang*float(line.split()[2]),bohrang*float(line.split()[3]),bohrang*float(line.split()[4])]
            coords.append(coord)
            coords_ang.append(coord_ang)
        # reading gridpoints
        if grabpoints is True:
            b = line.rstrip('\n').replace('  ', ' ').replace('  ', ' ').split(' ')
            b=list(filter(None, b))
            c =[float(i) for i in b]

            if len(c) >0:
                vals.append(c)
        # when to begin reading gridpoints
        if grab_deset_id is True and count == 7+numatoms:
            DSET_IDS_1 = int(line.split()[0])
            DSET_IDS_2 = int(line.split()[1])
        if (count >= 6+numatoms and grabpoints is False and grab_deset_id is False):
            #Setting grabpoints to True for next line
            grabpoints = True
        if (count >= 7+numatoms and grabpoints is False):
            #Now setting grabpoints to True for grabbing next
            grabpoints = True
    if LargePrint is True:
        print("Number of orb/density points:", len(vals))
    finaldict={'rlowx':rlowx,'dx':dx,'nx':nx,'orgx':orgx,'rlowy':rlowy,'dy':dy,'ny':ny,'orgy':orgy,'rlowz':rlowz,'dz':dz,'nz':nz,'orgz':orgz,'elems':elems,
        'coords':coords,'coords_ang':coords_ang,'numatoms':numatoms,'filebase':filebase,'vals':vals}
    if grab_deset_id is True:
        #In case we use it later
        finaldict['DSET_IDS_1']=DSET_IDS_1
        finaldict['DSET_IDS_2']=DSET_IDS_2

    #Sum of values
    sum_vals= sum([sum(i) for i in vals])
    #Integration volume
    int_vol = dx*dy*dz
    #Integration
    num_el_val = sum_vals*int_vol

    print("Integrated volume:", num_el_val)
    if "density" in description:
        print("Number of electrons:", num_el_val)
    return  finaldict


#Write Cube dictionary into file
def write_cube(cubedict, name="Default"):
    with open(name+".cube", 'w') as file:
        file.write("Cube file generated by ASH\n")
        file.write("Density difference\n")
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(cubedict['numatoms'],cubedict['orgx'],cubedict['orgy'],cubedict['orgz']))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(cubedict['nx'],cubedict['dx'],0.0,0.0))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(cubedict['ny'],0.0,cubedict['dy'],0.0))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(cubedict['nz'],0.0,0.0,cubedict['dz']))
        for el,c in zip(cubedict['elems'],cubedict['coords']):
            file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(el,el,c[0],c[1],c[2]))
        for v in cubedict['vals']:
            if len(v) == 6:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(v[0],v[1],v[2],v[3],v[4],v[5]))
            elif len(v) == 5:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(v[0],v[1],v[2],v[3],v[4]))
            elif len(v) == 4:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(v[0],v[1],v[2],v[3]))
            elif len(v) == 3:
                file.write("   {:.6e}   {:.6e}   {:.6e}\n".format(v[0],v[1],v[2]))
            elif len(v) == 2:
                file.write("   {:.6e}   {:.6e}\n".format(v[0],v[1]))
            elif len(v) == 1:
                file.write("   {:.6e}\n".format(v[0]))
    return




#Subtract one Cube-file from another
def write_cube_diff(cubedict1,cubedict2, name="Default"):

    #Note: For now ignoring DSET_IDS_1 lines that may have been grabbed and present in dicts

    numatoms=cubedict1['numatoms']
    orgx=cubedict1['orgx']
    orgy=cubedict1['orgy']
    orgz=cubedict1['orgz']
    nx=cubedict1['nx']
    dx=cubedict1['dx']
    ny=cubedict1['ny']
    dy=cubedict1['dy']
    nz=cubedict1['nz']
    dz=cubedict1['dz']
    elems=cubedict1['elems']
    coords=cubedict1['coords']
    val1=cubedict1['vals']
    val2=cubedict2['vals']
    #name=cubedict['name']
    diff_vals=[]
    with open(name+".cube", 'w') as file:
        file.write("Cube file generated by ASH\n")
        file.write("Density difference\n")
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(numatoms,orgx,orgy,orgz))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(nx,dx,0.0,0.0))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(ny,0.0,dy,0.0))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(nz,0.0,0.0,dz))
        for el,c in zip(elems,coords):
            file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(el,el,c[0],c[1],c[2]))
        for v1,v2 in zip(val1,val2):
            diff = [i-j for i,j in zip(v1,v2)]
            diff_vals.append(diff)

            if len(v1) == 6:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(diff[0],diff[1],diff[2],diff[3],diff[4],diff[5]))
            elif len(v1) == 5:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(diff[0],diff[1],diff[2],diff[3],diff[4]))
            elif len(v1) == 4:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(diff[0],diff[1],diff[2],diff[3]))
            elif len(v1) == 3:
                file.write("   {:.6e}   {:.6e}   {:.6e}\n".format(diff[0],diff[1],diff[2]))
            elif len(v1) == 2:
                file.write("   {:.6e}   {:.6e}\n".format(diff[0],diff[1]))
            elif len(v1) == 1:
                file.write("   {:.6e}\n".format(diff[0]))

    #Sum of values
    sum_vals= sum([sum(i) for i in diff_vals])
    #Sum of positive values
    sum_pos_vals=0
    sum_neg_vals=0
    for i in diff_vals:
        for j in i:
            if j > 0:
                sum_pos_vals+=j
            elif j < 0:
                sum_neg_vals+=j

    #Integration volume
    int_vol = dx*dy*dz
    #Integration
    num_el_val = sum_vals*int_vol
    num_el_val_pos = sum_pos_vals*int_vol
    num_el_val_neg = sum_neg_vals*int_vol

    print("\nTotal Integrated volume of difference density:", num_el_val)
    print("Integrated volume of positive part of difference density:", num_el_val_pos)
    print("Integrated volume of negative part of difference density:", num_el_val_neg)

    print("Negative diff. density means increase from Cube 1 to Cube 2")
    print("Positive diff. density means decrease from Cube 1 to Cube 2")

    return num_el_val,num_el_val_pos,num_el_val_neg

#Sum of 2 Cube-files
def write_cube_sum(cubedict1,cubedict2, name="Default"):
    #Note: For now ignoring DSET_IDS_1 lines that may have been grabbed and present in dicts
    numatoms=cubedict1['numatoms']
    orgx=cubedict1['orgx']
    orgy=cubedict1['orgy']
    orgz=cubedict1['orgz']
    nx=cubedict1['nx']
    dx=cubedict1['dx']
    ny=cubedict1['ny']
    dy=cubedict1['dy']
    nz=cubedict1['nz']
    dz=cubedict1['dz']
    elems=cubedict1['elems']
    coords=cubedict1['coords']
    val1=cubedict1['vals']
    val2=cubedict2['vals']
    #name=cubedict['name']

    with open(name+".cube", 'w') as file:
        file.write("Cube file generated by ASH\n")
        file.write("Sum of cube-files\n")
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(numatoms,orgx,orgy,orgz))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(nx,dx,0.0,0.0))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(ny,0.0,dy,0.0))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(nz,0.0,0.0,dz))
        for el,c in zip(elems,coords):
            file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(el,el,c[0],c[1],c[2]))
        for v1,v2 in zip(val1,val2):
            cubesum = [i+j for i,j in zip(v1,v2)]

            if len(v1) == 6:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(cubesum[0],cubesum[1],cubesum[2],cubesum[3],cubesum[4],cubesum[5]))
            elif len(v1) == 5:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(cubesum[0],cubesum[1],cubesum[2],cubesum[3],cubesum[4]))
            elif len(v1) == 4:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(cubesum[0],cubesum[1],cubesum[2],cubesum[3]))
            elif len(v1) == 3:
                file.write("   {:.6e}   {:.6e}   {:.6e}\n".format(cubesum[0],cubesum[1],cubesum[2]))
            elif len(v1) == 2:
                file.write("   {:.6e}   {:.6e}\n".format(cubesum[0],cubesum[1]))
            elif len(v1) == 1:
                file.write("   {:.6e}\n".format(cubesum[0]))

#Product of 2 Cube-files
def write_cube_product(cubedict1,cubedict2, name="Default"):
    #Note: For now ignoring DSET_IDS_1 lines that may have been grabbed and present in dicts
    numatoms=cubedict1['numatoms']
    orgx=cubedict1['orgx']
    orgy=cubedict1['orgy']
    orgz=cubedict1['orgz']
    nx=cubedict1['nx']
    dx=cubedict1['dx']
    ny=cubedict1['ny']
    dy=cubedict1['dy']
    nz=cubedict1['nz']
    dz=cubedict1['dz']
    elems=cubedict1['elems']
    coords=cubedict1['coords']
    val1=cubedict1['vals']
    val2=cubedict2['vals']
    #name=cubedict['name']

    with open(name+".cube", 'w') as file:
        file.write("Cube file generated by ASH\n")
        file.write("Sum of cube-files\n")
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(numatoms,orgx,orgy,orgz))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(nx,dx,0.0,0.0))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(ny,0.0,dy,0.0))
        file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(nz,0.0,0.0,dz))
        for el,c in zip(elems,coords):
            file.write("{:>5}   {:9.6f}   {:9.6f}   {:9.6f}   {:9.6f}\n".format(el,el,c[0],c[1],c[2]))
        for v1,v2 in zip(val1,val2):
            cubeprod = [i*j for i,j in zip(v1,v2)]

            if len(v1) == 6:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(cubeprod[0],cubeprod[1],cubeprod[2],cubeprod[3],cubeprod[4],cubeprod[5]))
            elif len(v1) == 5:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(cubeprod[0],cubeprod[1],cubeprod[2],cubeprod[3],cubeprod[4]))
            elif len(v1) == 4:
                file.write("   {:.6e}   {:.6e}   {:.6e}   {:.6e}\n".format(cubeprod[0],cubeprod[1],cubeprod[2],cubeprod[3]))
            elif len(v1) == 3:
                file.write("   {:.6e}   {:.6e}   {:.6e}\n".format(cubeprod[0],cubeprod[1],cubeprod[2]))
            elif len(v1) == 2:
                file.write("   {:.6e}   {:.6e}\n".format(cubeprod[0],cubeprod[1]))
            elif len(v1) == 1:
                file.write("   {:.6e}\n".format(cubeprod[0]))


#Read cubefile. Grabs coords. Calculates density if MO
def create_density_from_orb (cubefile, denswrite=True, LargePrint=True):
    bohrang = 0.52917721067
    #Opening orbital cube file
    try:
        filename = cubefile
        a = open(filename,"r")
        print("Reading orbital file:", filename)
        filebase=os.path.splitext(filename)[0]
    except IndexError:
        print("error")
        quit()
    if denswrite is True:
        #Write orbital density cube file
        output = open(filebase+'-dens.cube', "w")
    #Read cube file and get all data. Square values
    count = 0
    X = False
    densvals = []
    orbvals=[]
    elems=[]
    coords=[]
    coords_ang=[]
    numatoms=0
    for line in a:
        count += 1
        words = line.split()
        numwords=len(words)
        #Grabbing origin
        if count < 3:
            if denswrite is True:
                output.write(line)
        if count == 3:
            numatoms=abs(int(line.split()[0]))
            orgx=float(line.split()[1])
            orgy=float(line.split()[2])
            orgz=float(line.split()[3])
            rlowx=orgx;rlowy=orgy;rlowz=orgz
            if denswrite is True:
                output.write(line)
        if count == 4:
            nx=int(line.split()[0])
            dx=float(line.split()[1])
            if denswrite is True:
                output.write(line)
        if count == 5:
            ny=int(line.split()[0])
            dy=float(line.split()[2])
            if denswrite is True:
                output.write(line)
        if count == 6:
            nz=int(line.split()[0])
            dz=float(line.split()[3])
            if denswrite is True:
                output.write(line)
        #Grabbing molecular coordinates
        if count > 6 and count <= 6+numatoms:
            elems.append(int(line.split()[0]))
            coord=[float(line.split()[2]),float(line.split()[3]),float(line.split()[4])]
            coord_ang=[bohrang*float(line.split()[2]),bohrang*float(line.split()[3]),bohrang*float(line.split()[4])]
            coords.append(coord)
            coords_ang.append(coord_ang)
            if denswrite is True:
                output.write(line)
        # reading gridpoints
        if X is True:
            b = line.rstrip('\n').replace('  ', ' ').replace('  ', ' ').split(' ')
            b=list(filter(None, b))
            c =[float(i) for i in b]
            #print("c is", c)
            #Squaring orbital values to get density
            csq = [q** 2 for q in c]
            dsq = [float('%.5e' % i) for i in csq]
            densvals.append(dsq)
            dbq = [float('%.5e' % i) for i in c]
            orbvals.append(dbq)
        # when to begin reading gridpoints
        if (count > 6 and numwords == 2 and X is False):
            X = True
            if denswrite is True:
                output.write(line)

    # Go through orb and dens list and print out density file
    alldensvalues=[]
    allorbvalues=[]
    for line in densvals:
        columns = ["%13s" % cell for cell in line]
        for val in columns:
            alldensvalues.append(float(val))
        if denswrite is True:
            linep=' '.join( columns)
            output.write(linep+'\n')

    for line in orbvals:
        dolumns = ["%13s" % cell for cell in line]
        for oval in dolumns:
            allorbvalues.append(float(oval))
    if denswrite is True:
        output.close()
        print("Wrote orbital density file as:", filebase+'-dens.cube')
        print("")
    sumdensvalues=sum(i for i in alldensvalues)
    if LargePrint is True:
        print("Sum of density values is:", sumdensvalues)
        print("Number of density values is", len(alldensvalues))
        print("Number of orb values is", len(allorbvalues))
    return rlowx,dx,nx,orgx,rlowy,dy,ny,orgy,rlowz,dz,nz,orgz,alldensvalues,elems,coords_ang,numatoms,filebase


def centroid_calc(rlowx,dx,nx,orgx,rlowy,dy,ny,orgy,rlowz,dz,nz,orgz,alldensvalues ):
    #########################################################
    # Calculate centroid.
    ############################################################

    #Largest x,y,z coordinates
    rhighx=rlowx+(dx*(nx-1))
    rhighy=rlowy+(dy*(ny-1))
    rhighz=rlowz+(dz*(nz-1))
    #Lowest and highest density values
    rlowv = min(float(s) for s in alldensvalues)
    rhighv = max(float(s) for s in alldensvalues)

    sumuppos=0.0
    cenxpos=0.0
    cenypos=0.0
    cenzpos=0.0
    vcount=0

    #print ("dx, dy, dz is", dx, dy, dz)
    #print("range of x:", rlowx, rhighx)
    #print("range of y:", rlowy, rhighy)
    #print("range of z:", rlowz, rhighz)

    for i in range(1,nx+1):
        if (orgx+(i-1)*dx)<rlowx or (orgx+(i-1)*dx)>rhighx:
            print("If statement. Look into. x")
            ashexit()
            continue
        for j in range(1,ny+1):
            if (orgy+(j-1)*dy)<rlowy or (orgy+(j-1)*dy)>rhighy:
                print("If statement. Look into. y")
                ashexit()
                continue
            for k in range(1,nz+1):
                if (orgz+(k-1)*dz)<rlowz or (orgz+(k-1)*dz)>rhighz:
                    print("If statement. Look into. z")
                    ashexit()
                    continue
                #print("i,j,k is", i,j,k)
                valtmp=alldensvalues[vcount]
                if valtmp<rlowv or valtmp>rhighv:
                    print("If statement. Look into. v")
                    ashexit()
                    continue
                if valtmp>0:
                    sumuppos=sumuppos+valtmp
                    #print("sumuppos is", sumuppos)
                    cenxpos=cenxpos+(orgx+(i-1)*dx)*valtmp
                    cenypos=cenypos+(orgy+(j-1)*dy)*valtmp
                    cenzpos=cenzpos+(orgz+(k-1)*dz)*valtmp
                    #print("valtmp is", valtmp)
                    #print("-----------------------")
                vcount+=1

    #Final values
    cenxpos=cenxpos/sumuppos
    cenypos=cenypos/sumuppos
    cenzpos=cenzpos/sumuppos
    return cenxpos,cenypos,cenzpos

# MO-DOS PLOT. Multiply MO energies by -1 and sort.
def modosplot(occorbs_alpha,occorbs_beta,hftyp):
    #Defining sticks as -1 times MO energy (eV)
    stk_alpha=[]
    stk_beta=[]
    for j in occorbs_alpha:
        stk_alpha.append(-1*j)
    if hftyp == "UHF":
        for k in occorbs_beta:
            stk_beta.append(-1*k)
        stk_beta.sort()
    stk_alpha.sort()
    return stk_alpha,stk_beta

#Calculate HOMO number from nuclear charge from XYZ-file and total charge
def HOMOnumbercalc(file,charge,mult):
    el=[]
    with open(file) as f:
        for count,line in enumerate(f):
            if count >1:
                el.append(line.split()[0])
    totnuccharge=0
    for e in el:
        atcharge=eldict[e]
        totnuccharge+=atcharge
    numel=totnuccharge-charge
    HOMOnum_a="unset";HOMOnum_b="unset"
    orcaoffset=-1
    if mult == 1:
        #RHF case. HOMO is numel/2 -1
        HOMOnum_a=(numel/2)+orcaoffset
        HOMOnum_b=(numel/2)+orcaoffset
    elif mult > 1:
        #UHF case.
        numunpel=mult-1
        Doubocc=(numel-numunpel)/2
        HOMOnum_a=Doubocc+numunpel+orcaoffset
        HOMOnum_b=Doubocc+orcaoffset
    return int(HOMOnum_a),int(HOMOnum_b)

#Calculate DDEC charges and derive LJ parameters from ORCA.
#Uses Chargemol program
# Uses ORCA to calculate densities of molecule and its free atoms. Uses orca_2mkl to create Molden file and molden2aim to create WFX file from Molden.
# Wfx file is read into Chargemol program for DDEC analysis which radial moments used to compute C6 parameters and radii for Lennard-Jones equation.
def DDEC_calc(elems=None, theory=None, gbwfile=None, numcores=1, DDECmodel='DDEC3', calcdir='DDEC', molecule_charge=None,
              molecule_spinmult=None, chargemolbinarydir=None):
    #Creating calcdir. Should not exist previously
    try:
        shutil.rmtree(calcdir)
    except:
        pass
    os.mkdir(calcdir)
    os.chdir(calcdir)
    #Copying GBW file to current dir and label as molecule.gbw
    shutil.copyfile('../'+gbwfile, './' + 'molecule.gbw')

    #Finding molden2aim in PATH. Present in ASH (May require compilation)
    ashpath=os.path.dirname(ash.__file__)
    molden2aim=ashpath+"/external/Molden2AIM/src/"+"molden2aim.exe"
    if os.path.isfile(molden2aim) is False:
        print("Did not find {}. Did you compile it ? ".format(molden2aim))
        print("Go into dir:", ashpath+"/external/Molden2AIM/src")
        print("Compile using gfortran or ifort:")
        print("gfortran -O3 edflib.f90 edflib-pbe0.f90 molden2aim.f90 -o molden2aim.exe")
        print("ifort -O3 edflib.f90 edflib-pbe0.f90 molden2aim.f90 -o molden2aim.exe")
        ashexit()
    else:
        print("Found molden2aim.exe: ", molden2aim)

    print("Warning: DDEC_calc requires chargemol-binary dir to be present in environment PATH variable.")

    #Finding chargemoldir from PATH in os.path
    PATH=os.environ.get('PATH').split(':')
    print("PATH: ", PATH)
    print("Searching for molden2aim and chargemol in PATH")
    for p in PATH:
        if 'chargemol' in p:
            print("Found chargemol in path line (this dir should contain the executables):", p)
            chargemolbinarydir=p

    #Checking if we can proceed
    if chargemolbinarydir is None:
        print("chargemolbinarydir is not defined.")
        print("Please provide path as argument to DDEC_calc or put the location inside the $PATH variable on your Unix/Linux OS.")
        ashexit()

    #Defining Chargemoldir (main dir) as 3-up from binary dir
    var=os.path.split(chargemolbinarydir)[0]
    var=os.path.split(var)[0]
    chargemoldir=os.path.split(var)[0]
    print("Chargemoldir (base directory): ", chargemoldir)
    print("Chargemol binary dir:", chargemolbinarydir)

    if theory is None :
        print("DDEC_calc requires theory, keyword argument")
        ashexit()
    if theory.__class__.__name__ != "ORCATheory":
        print("Only ORCA is supported as theory in DDEC_calc currently")
        ashexit()

    # What DDEC charge model to use. Jorgensen paper uses DDEC3. DDEC6 is the newer recommended chargemodel
    print("DDEC model:", DDECmodel)

    # Serial or parallel version
    if numcores == 1:
        print("Using serial version of Chargemol")
        chargemol=glob.glob(chargemolbinarydir+'/*serial*')[0]
        #chargemol=chargemolbinarydir+glob.glob('*serial*')[0]
    else:
        print("Using parallel version of Chargemol using {} cores".format(numcores))
        #chargemol=chargemolbinarydir+glob.glob('*parallel*')[0]
        chargemol=glob.glob(chargemolbinarydir+'/*parallel*')[0]
        # Parallelization of Chargemol code. 8 should be good.
        os.environ['OMP_NUM_THREADS'] = str(numcores)
    print("Using Chargemoldir executable: ", chargemol)

    #Dictionary for spin multiplicities of atoms
    spindictionary = {'H':2, 'He': 1, 'Li':2, 'Be':1, 'B':2, 'C':3, 'N':4, 'O':3, 'F':2, 'Ne':1, 'Na':2, 'Mg':1, 'Al':2, 'Si':3, 'P':4, 'S':3, 'Cl':2, 'Ar':1, 'K':2, 'Ca':1, 'Sc':2, 'Ti':3, 'V':4, 'Cr':7, 'Mn':6, 'Fe':5, 'Co':4, 'Ni':3, 'Cu':2, 'Zn':1, 'Ga':2, 'Ge':3, 'As':4, 'Se':3, 'Br':2, 'Kr':1, 'Rb':2, 'Sr':1, 'Y':2, 'Zr':3, 'Nb':6, 'Mo':7, 'Tc':6, 'Ru':5, 'Rh':4, 'Pd':1, 'Ag':2, 'Cd':1, 'In':2, 'Sn':3, 'Sb':4, 'Te':3, 'I':2, 'Xe':1, 'Cs':2, 'Ba':1, 'La':2, 'Ce':1, 'Pr':4, 'Nd':5, 'Pm':6, 'Sm':7, 'Eu':8, 'Gd':9, 'Tb':6, 'Dy':5, 'Ho':4, 'Er':3, 'Tm':2, 'Yb':1, 'Lu':2, 'Hf':3, 'Ta':4, 'W':5, 'Re':6, 'Os':5, 'Ir':4, 'Pt':3, 'Au':2, 'Hg':1, 'Tl':2, 'Pb':3, 'Bi':4, 'Po':3, 'At':2, 'Rn':1, 'Fr':2, 'Ra':1, 'Ac':2, 'Th':3, 'Pa':4, 'U':5, 'Np':6, 'Pu':7, 'Am':8, 'Cm':9, 'Bk':6, 'Cf':5, 'Es':5, 'Fm':3, 'Md':2, 'No':1, 'Lr':2, 'Rf':3, 'Db':4, 'Sg':5, 'Bh':6, 'Hs':5, 'Mt':4, 'Ds':3, 'Rg':2, 'Cn':1, 'Nh':2, 'Fl':3, 'Mc':4, 'Lv':3, 'Ts':2, 'Og':1 }

    #Dictionary to keep track of radial volumes
    voldict = {}

    uniqelems=set(elems)
    numatoms=len(elems)

    print("Lennard-Jones parameter creation from ORCA densities")
    print("")
    print("First calculating densities of free atoms")
    print("Will skip calculation if wfx file already exists")
    print("")

    # Calculate elements
    print("------------------------------------------------------------------------")
    for el in uniqelems:
        print("Doing element:", el)

        #Skipping analysis if wfx file exists
        if os.path.isfile(el+'.molden.wfx'):
            print(el+'.molden.wfx', "exists already. Skipping calculation.")
            continue
        #TODO: Revisit with ORCA5 and TRAH?
        scfextrasettingsstring="""%scf
Maxiter 500
DIISMaxIt 0
ShiftErr 0.0000
DampFac 0.8500
DampMax 0.9800
DampErr 0.0300
cnvsoscf false
cnvkdiis false
end"""

        #Creating ORCA object for  element
        ORCASPcalculation = ORCATheory(orcadir=theory.orcadir, orcasimpleinput=theory.orcasimpleinput,
                                           orcablocks=theory.orcablocks, extraline=scfextrasettingsstring)

        #Element coordinates
        Elfrag = ash.Fragment(elems=[el], coords=[[0.0,0.0,0.0]])
        print("Elfrag dict ", Elfrag.__dict__)
        ash.Singlepoint(theory=ORCASPcalculation,fragment=Elfrag, charge=0, mult=spindictionary[el])
        #Preserve outputfile and GBW file for each element
        shutil.copyfile(ORCASPcalculation.filename+'.out', './' + str(el) + '.out')
        shutil.copyfile(ORCASPcalculation.filename+'.gbw', './' + str(el) + '.gbw')

        #Create molden file from el.gbw
        sp.call([theory.orcadir+'/orca_2mkl', el, '-molden'])


        #Cleanup ORCA calc for each element
        ORCASPcalculation.cleanup()

        #Write configuration file for molden2aim
        with open("m2a.ini", 'w') as m2afile:
            string = """########################################################################
        #  In the following 8 parameters,
        #     >0:  always performs the operation without asking the user
        #     =0:  asks the user whether to perform the operation
        #     <0:  always neglect the operation without asking the user
        molden= 1           ! Generating a standard Molden file in Cart. function
        wfn= -1              ! Generating a WFN file
        wfncheck= -1         ! Checking normalization for WFN
        wfx= 1              ! Generating a WFX file (not implemented)
        wfxcheck= 1         ! Checking normalization for WFX (not implemented)
        nbo= -1              ! Generating a NBO .47 file
        nbocheck= -1         ! Checking normalization for NBO's .47
        wbo= -1              ! GWBO after the .47 file being generated

        ########################################################################
        #  Which quantum chemistry program is used to generate the MOLDEN file?
        #  1: ORCA, 2: CFOUR, 3: TURBOMOLE, 4: JAGUAR (not supported),
        #  5: ACES2, 6: MOLCAS, 7: PSI4, 8: MRCC, 9: NBO 6 (> ver. 2014),
        #  0: other programs, or read [Program] xxx from MOLDEN.
        #
        #  If non-zero value is given, [Program] xxx in MOLDEN will be ignored.
        #
        program=1

        ########################################################################
        #  For ECP: read core information from Molden file
        #<=0: if the total_occupation_number is smaller than the total_Za, ask
        #     the user whether to read core information
        # >0: always search and read core information
        rdcore=0

        ########################################################################
        #  Which orbirals will be printed in the WFN/WFX file?
        # =0: print only the orbitals with occ. number > 5.0d-8
        # <0: print only the orbitals with occ. number > 0.1 (debug only)
        # >0: print all the orbitals
        iallmo=0

        ########################################################################
        #  Used for WFX only
        # =0: print "UNKNOWN" for Energy and Virial Ratio
        # .ne. 0: print 0.0 for Energy and 2.0 for Virial Ratio
        unknown=1

        ########################################################################
        #  Print supporting information or not
        # =0: print; .ne. 0: do not print
        nosupp=0

        ########################################################################
        #  The following parameters are used only for debugging.
        clear=1            ! delete temporary files (1) or not (0)

        ########################################################################
        """
            m2afile.write(string)

        #Write settings file
        mol2aiminput=[' ',  el+'.molden.input', 'Y', 'Y', 'N', 'N', ' ', ' ']
        m2aimfile = open("mol2aim.inp", "w")
        for mline in mol2aiminput:
            m2aimfile.write(mline+'\n')
        m2aimfile.close()

        #Run molden2aim
        m2aimfile = open('mol2aim.inp')
        p = sp.Popen(molden2aim, stdin=m2aimfile, stderr=sp.STDOUT)
        p.wait()

        #Write job control file for Chargemol
        wfxfile=el+'.molden.wfx'
        jobcontfilewrite=[
        '<atomic densities directory complete path>',
        chargemoldir+'/atomic_densities/',
        '</atomic densities directory complete path>',
        '<input filename>',
        wfxfile,
        '<charge type>',
        DDECmodel,
        '</charge type>',
        '<compute BOs>',
        '.true.',
        '</compute BOs>',
        ]
        jobfile = open("job_control.txt", "w")
        for jline in jobcontfilewrite:
            jobfile.write(jline+'\n')

        jobfile.close()
        #CALLING chargemol
        sp.call(chargemol)
        print("------------------------------------------------------------------------")


    #DONE WITH ELEMENT CALCS

    print("")
    print("=============================")
    #Getting volumes from output
    for el in uniqelems:
        with open(el+'.molden.output') as momfile:
            for line in momfile:
                if ' The computed Rcubed moments of the atoms' in line:
                    elmom=next(momfile).split()[0]
                    voldict[el] = float(elmom)

        print("Element", el, "is done.")

    print("")
    print("Calculated radial volumes of free atoms (Bohrs^3):", voldict)
    print("")

    #Now doing main molecule. Skipping ORCA calculation since we have copied over GBW file
    # Create molden file
    sp.call(['orca_2mkl', "molecule", '-molden'])

    #Write input for molden2aim

    if molecule_charge==0:
        mol2aiminput=[' ',  "molecule"+'.molden.input', str(molecule_spinmult), ' ', ' ', ' ']
    else:
        #Charged system, will ask for charge
        #str(molecule_charge)
        mol2aiminput=[' ',  "molecule"+'.molden.input', 'N', '2', ' ', str(molecule_spinmult), ' ', ' ', ' ']

    m2aimfile = open("mol2aim.inp", "w")
    for mline in mol2aiminput:
        m2aimfile.write(mline+'\n')
    m2aimfile.close()

    #Run molden2aim
    print("Running Molden2Aim for molecule")
    m2aimfile = open('mol2aim.inp')
    p = sp.Popen(molden2aim, stdin=m2aimfile, stderr=sp.STDOUT)
    p.wait()

    # Write job control file for Chargemol
    wfxfile = "molecule" + '.molden.wfx'
    jobcontfilewrite = [
        '<net charge>',
        '{}'.format(str(float(molecule_charge))),
        '</net charge>',
        '<atomic densities directory complete path>',
        chargemoldir + '/atomic_densities/',
        '</atomic densities directory complete path>',
        '<input filename>',
        wfxfile,
        '<charge type>',
        DDECmodel,
        '</charge type>',
        '<compute BOs>',
        '.true.',
        '</compute BOs>',
    ]
    jobfile = open("job_control.txt", "w")
    for jline in jobcontfilewrite:
        jobfile.write(jline+'\n')

    jobfile.close()
    if os.path.isfile("molecule"+'.molden.output') is False:
        sp.call(chargemol)
    else:
        print("Skipping Chargemol step. Output file exists")


    #Grabbing radial moments from output
    molmoms=[]
    grabmoms=False
    with open("molecule"+'.molden.output') as momfile:
        for line in momfile:
            if ' The computed Rfourth moments of the atoms' in line:
                grabmoms=False
                continue
            if grabmoms==True:
                temp=line.split()
                [molmoms.append(float(i)) for i in temp]
            if ' The computed Rcubed moments of the atoms' in line:
                grabmoms=True

    #Grabbing DDEC charges from output
    if DDECmodel == 'DDEC3':
        chargefile='DDEC3_net_atomic_charges.xyz'
    elif DDECmodel == 'DDEC6':
        chargefile='DDEC6_even_tempered_net_atomic_charges.xyz'

    grabcharge=False
    ddeccharges=[]
    with open(chargefile) as chfile:
        for line in chfile:
            if grabcharge is True:
                ddeccharges.append(float(line.split()[5]))
                if int(line.split()[0]) == numatoms:
                    grabcharge=False
            if "atom number, atomic symbol, x, y, z, net_charge," in line:
                grabcharge=True

    print("")
    print("molmoms is", molmoms)
    print("voldict is", voldict)
    print("ddeccharges: ", ddeccharges)
    print("elems: ", elems)
    os.chdir('..')
    return ddeccharges, molmoms, voldict




#Tkatchenko
#alpha_q_m = Phi*Rvdw^7
#https://arxiv.org/pdf/2007.02992.pdf
def Rvdwfree(polz):
    #Fine-structure constant (2018 CODATA recommended value)
    FSC=0.0072973525693
    Phi=FSC**(4/3)
    RvdW=(polz/Phi)**(1/7)
    return RvdW


def DDEC_to_LJparameters(elems, molmoms, voldict, scale_polarH=False):

    #voldict: Vfree. Computed using MP4SDQ/augQZ and chargemol in Jorgensen paper
    # Testing: Use free atom volumes calculated at same level of theory as molecule

    #Rfree fit parameters. Jorgensen 2016 J. Chem. Theory Comput. 2016, 12, 2312−2323. H,C,N,O,F,S,Cl
    #Thes are free atomic vdW radii
    # In Jorgensen and Cole papers these are fit parameters : rfreedict = {'H':1.64, 'C':2.08, 'N':1.72, 'O':1.6, 'F':1.58, 'S':2.0, 'Cl':1.88}
    # We are instead using atomic Rvdw derived directly from atomic polarizabilities

    print("Elems:", elems)
    print("Molmoms:", molmoms)
    print("voldict:", voldict)

    #Calculating A_i, B_i, epsilon, sigma, r0 parameters
    Blist=[]
    Alist=[]
    sigmalist=[]
    epsilonlist=[]
    r0list=[]
    Radii_vdw_free=[]
    for count,el in enumerate(elems):
        print("el :", el, "count:", count)
        atmnumber=ash.modules.module_coords.elematomnumbers[el.lower()]
        print("atmnumber:", atmnumber)
        Radii_vdw_free.append(ash.dictionaries_lists.elems_C6_polz[atmnumber].Rvdw_ang)
        print("Radii_vdw_free:", Radii_vdw_free)
        volratio=molmoms[count]/voldict[el]
        print("volratio:", volratio)
        C6inkcal=ash.constants.harkcal*(ash.dictionaries_lists.elems_C6_polz[atmnumber].C6**(1/6)* ash.constants.bohr2ang)**6
        print("C6inkcal:", C6inkcal)
        B_i=C6inkcal*(volratio**2)
        print("B_i:", B_i)
        Raim_i=volratio**(1/3)*ash.dictionaries_lists.elems_C6_polz[atmnumber].Rvdw_ang
        print("Raim_i:", Raim_i)
        A_i=0.5*B_i*(2*Raim_i)**6
        print("A_i:", A_i)
        sigma=(A_i/B_i)**(1/6)
        print("sigma :", sigma)
        r0=sigma*(2**(1/6))
        print("r0:", r0)
        epsilon=(A_i/(4*sigma**12))
        print("epsilon:", epsilon)

        sigmalist.append(sigma)
        Blist.append(B_i)
        Alist.append(A_i)
        epsilonlist.append(epsilon)
        r0list.append(r0)

    print("Before corrections:")
    print("elems:", elems)
    print("Radii_vdw_free:", Radii_vdw_free)
    print("Alist is", Alist)
    print("Blist is", Blist)
    print("sigmalist is", sigmalist)
    print("epsilonlist is", epsilonlist)
    print("r0list is", r0list)

    #Accounting for polar H. This could be set to zero as in Jorgensen paper
    if scale_polarH is True:
        print("Scaling og polar H not implemented yet")
        ashexit()
        for count,el in enumerate(elems):
            if el == 'H':
                bla=""
                #Check if H connected to polar atom (O, N, S ?)
                #if 'H' connected to polar:
                    #1. Set eps,r0/sigma to 0 if so
                    #2. Add to heavy atom if so
                    #nH = 1
                    #indextofix = 11
                    #hindex = 12
                    #Blist[indextofix] = ((Blist[indextofix]) ** (1 / 2) + nH * (Blist[hindex]) ** (1 / 2)) ** 2

    return r0list, epsilonlist


#Get number of core electrons for list of elements
def num_core_electrons(elems):
    sum=0
    #formula_list = ash.modules.module_coords.molformulatolist(fragment.formula)
    for i in elems:
        cels = ash.dictionaries_lists.atom_core_electrons[i]
        sum+=cels
    return sum


#Check if electrons pairs in element list are less than numcores. Reduce numcores if so.
#Using even number of electrons
def check_cores_vs_electrons(elems,numcores,charge):
    print("Checking whether number of cores should be reduced")
    numelectrons = int(nucchargelist(elems) - charge)
    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(elems)
    print("Number of core electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    print("Number of total electrons :", numelectrons)
    print("Number of valence electrons :", valence_electrons )
    print("Number of valence electron pairs :", electronpairs )
    if electronpairs  < numcores:
        print(f"Number of electron pairs ({electronpairs}) less than number of cores ({numcores})")
        if isodd(electronpairs):
            if electronpairs > 1:
                #Changed from subtracting 1 to 3 after DLPNO-CC of NaH calculation failed (MB16-43)
                numcores=electronpairs-3
            else:
                numcores=electronpairs
        else:
            numcores=electronpairs
        print("Changing number of cores to be approximately equal to number of electron pairs")
    if numcores == 0:
        numcores=1
    print("Number of cores will be:", numcores)
    return numcores



#Approximate J-coupling spin projection functions
def Jcoupling_Yamaguchi(HSenergy,BSenergy,HS_S2,BS_S2):
    print("Yamaguchi spin projection")
    J=-1*(HSenergy-BSenergy)/(HS_S2-BS_S2)
    J_kcal=J*ash.constants.harkcal
    J_cm=J*ash.constants.hartocm
    print("J coupling constant: {} Eh".format(J))
    print("J coupling constant: {} kcal/Mol".format(J_kcal))
    print("J coupling constant: {} cm**-1".format(J_cm))
    return J
#Strong-interaction limit (bond-formation)
def Jcoupling_Bencini(HSenergy,BSenergy,smax):
    print("Bencini spin projection")
    J=-1*(HSenergy-BSenergy)/(smax*(smax+1))
    J_kcal=J*ash.constants.harkcal
    J_cm=J*ash.constants.hartocm
    print("Smax : ", smax)
    print("J coupling constant: {} Eh".format(J))
    print("J coupling constant: {} kcal/Mol".format(J_kcal))
    print("J coupling constant: {} cm**-1".format(J_cm))
    return J
#Weak-interaction limit
def Jcoupling_Noodleman(HSenergy,BSenergy,smax):
    print("Noodleman spin projection")
    J=-1*(HSenergy-BSenergy)/(smax)**2
    J_kcal=J*ash.constants.harkcal
    J_cm=J*ash.constants.hartocm
    print("Smax : ", smax)
    print("J coupling constant: {} Eh".format(J))
    print("J coupling constant: {} kcal/Mol".format(J_kcal))
    print("J coupling constant: {} cm**-1".format(J_cm))
    return J

#Select an active space from list of occupations and thresholds
def select_space_from_occupations(occlist, selection_thresholds=[1.98,0.02]):
    upper_threshold=selection_thresholds[0]
    lower_threshold=selection_thresholds[1]
    welloccorbs=[i for i in occlist if i < upper_threshold and i > lower_threshold]
    numelectrons=round(sum(welloccorbs))
    numorbitals=len(welloccorbs)
    return [numelectrons,numorbitals]

# Similar to above but returns the first and last indices of space instead of el/orbs
# # Determing active space from natorb thresholds
def select_indices_from_occupations(occlist, selection_thresholds=[1.98,0.02]):
    print("select_indices_from_occupations function")
    print("selection_thresholds:", selection_thresholds)
    nat_occs_for_thresholds=[i for i in occlist if i < selection_thresholds[0] and i > selection_thresholds[1]]
    indices_for_thresholds=[i for i,j in enumerate(occlist) if j < selection_thresholds[0] and j > selection_thresholds[1]]
    actlist = list(range(indices_for_thresholds[0],indices_for_thresholds[-1]+1))
    return actlist


# Interface to XDM postg program
#https://github.com/aoterodelaroza/postg
def xdm_run(wfxfile=None, postgdir=None,a1=None, a2=None,functional=None):

    if postgdir == None:
        # Trying to find postgdir in path
        print("postgdir keyword argument not provided to xdm_run. Trying to find postg in PATH")
        try:
            postgdir = os.path.dirname(shutil.which('postg'))
            print("Found postg in path. Setting postgdir.")
        except:
            print("Found no postg executable in path. Exiting... ")
            ashexit()

    parameterdict= {'pw86pbe' : [0.7564,1.4545], 'b3lyp' : [0.6356, 1.5119],
    'b3pw91' : [0.6002,1.4043], 'b3p86' : [1.0400, 0.3741], 'pbe0' : [0.4186,2.6791],
    'camb3lyp' : [0.3248,2.8607], 'b97-1' : [0.1998,3.5367], 'bhandhlyp' : [0.5610, 1.9894],
    'blyp' : [0.7647,0.8457],'pbe' : [0.4492,2.5517],'lcwpbe' : [1.0149, 0.6755],
    'tpss' : [0.6612, 1.5111], 'b86bpbe' : [0.7443, 1.4072]}

    if a1 == None or a2 == None:
        print("a1/a2 parameters not given. Looking up functional in table")
        print("Parameter table:", parameterdict)
        a1, a2 = parameterdict[functional.lower()]
        print(f"XDM a1: {a1}, a2: {a2}")
    with open('xdm-postg.out', 'w') as ofile:
        process = sp.run([postgdir+'/postg', str(a1), str(a2), str(wfxfile), str(functional) ], check=True,
            stdout=ofile, stderr=ofile, universal_newlines=True)

    dispgrab=False
    dispgradient=[]
    with open('xdm-postg.out', 'r') as xdmfile:
        for line in xdmfile:
            #TODO: Grab Hirshfeld charges
            #TODO: C6,C8, C10 coefficients, moments and volumes
            if 'dispersion energy' in line:
                dispenergy = float(line.split()[-1])
            if 'dispersion force constant matrix' in line:
                dispgrab=False
            if dispgrab == True:
                if '#' not in line:
                    grad_x=-1*float(line.split()[1])
                    grad_y=-1*float(line.split()[2])
                    grad_z=-1*float(line.split()[3])
                    dispgradient.append([grad_x,grad_y,grad_z])
            if 'dispersion forces' in line:
                dispgrab=True

    dispgradient=np.array(dispgradient)
    print("dispenergy:", dispenergy)
    print("dispgradient:", dispgradient)
    return dispenergy, dispgradient

#Create difference density for 2 calculations differing in either fragment or theory-level
def difference_density_ORCA(fragment_A=None, fragment_B=None, theory_A=None, theory_B=None, griddensity=80, cubefilename='difference_density'):
    print_line_with_mainheader("difference_density_ORCA")
    print("Will calculate and create a difference density for molecule")
    print("Either fragment can be different (different geometry, different charge, different spin)")
    print("Or theory can be different (different functional, different basis set)")
    print()
    print("griddensity:", griddensity)

    if fragment_A is None or fragment_B is None:
        print("You need to provide an ASH fragment for both fragment_A and fragment_B (can be the same)")
        ashexit()
    if fragment_A.charge == None or fragment_B.charge == None:
        print("You must provide charge/multiplicity information in all fragments")
        ashexit()
    if theory_A is None or theory_A.__class__.__name__ != "ORCATheory":
        print("theory_A: You must provide an ORCATheory level")
        ashexit()
    if theory_B is None or theory_B.__class__.__name__ != "ORCATheory":
        print("theory_B: You must provide an ORCATheory level")
        ashexit()

    #------------------
    #Calculation 1
    #------------------
    theory_A.filename="calc_A"
    result_calc1=ash.Singlepoint(theory=theory_A, fragment=fragment_A)
    #Run orca_plot to request electron density creation from ORCA gbw file
    run_orca_plot("calc_A.gbw", "density", gridvalue=griddensity)


    #------------------
    #Calculation 2
    #------------------
    theory_B.filename="calc_B"
    result_calc2=ash.Singlepoint(theory=theory_B, fragment=fragment_B)
    #Run orca_plot to request electron density creation from ORCA gbw file
    run_orca_plot("calc_B.gbw", "density", gridvalue=griddensity)

    #Read Cubefiles from disk
    cube_data1 = read_cube("calc_A.eldens.cube")
    cube_data2 = read_cube("calc_B.eldens.cube")

    #Write out difference density as a Cubefile
    write_cube_diff(cube_data2, cube_data1, cubefilename)
    print()
    print(f"Difference density (B - A) file was created: {cubefilename}.cube")


#Create deformation density and do NOCV analysis by providing fragment files for AB, A and B and a theory-level object.
#TODO: Limitation, ORCA can only do closed-shell case
#TODO: Switch to multiwfn for more generality
def NOCV_density_ORCA(fragment_AB=None, fragment_A=None, fragment_B=None, theory=None, griddensity=80,
                            NOCV=True, num_nocv_pairs=5, keep_all_orbital_cube_files=False,
                            make_cube_files=True):
    print_line_with_mainheader("NOCV_density_ORCA")
    print("Will calculate and create a deformation density for molecule AB for fragments A and B")
    print("griddensity:", griddensity)
    print("NOCV option:", NOCV)
    if NOCV is True:
        print("Will do NOCV analysis on AB fragment deformation density using A+B promolecular density")
    else:
        print("Full NOCV analysis not carried out")
    #Early exits
    if fragment_AB is None or fragment_A is None or fragment_B is None:
        print("You need to provide an ASH fragment")
        ashexit()
    if fragment_AB.charge is None or fragment_A.charge is None or fragment_B.charge is None:
        print("You must provide charge/multiplicity information to all fragments")
        ashexit()
    if theory == None or theory.__class__.__name__ != "ORCATheory":
        print("You must provide an ORCATheory level")
        ashexit()

    #Creating copies of theory object provided
    calc_AB = copy.copy(theory); calc_AB.filename="calcAB"
    calc_A = copy.copy(theory); calc_A.filename="calcA"
    calc_B = copy.copy(theory); calc_B.filename="calcB"

    #-------------------------
    #Calculation on A
    #------------------------
    print("-"*120)
    print("Performing ORCA calculation on fragment A")
    print("-"*120)
    #Run A SP
    result_calcA=ash.Singlepoint(theory=calc_A, fragment=fragment_A)
    #Run orca_plot to request electron density creation from ORCA gbw file
    if make_cube_files is True:
        run_orca_plot("calcA.gbw", "density", gridvalue=griddensity)

    #-------------------------
    #Calculation on B
    #------------------------
    print()
    print("-"*120)
    print("Performing ORCA calculation on fragment B")
    print("-"*120)
    #Run B SP
    result_calcB=ash.Singlepoint(theory=calc_B, fragment=fragment_B)
    #Run orca_plot to request electron density creation from ORCA gbw file
    if make_cube_files is True:
        run_orca_plot("calcB.gbw", "density", gridvalue=griddensity)


    #-----------------------------------------
    # merge A + B to get promolecular density
    #-----------------------------------------
    print()
    print("-"*120)
    print("Using orca_mergefrag to combine GBW-files for A and B into AB promolecule file: promolecule_AB.gbw")
    print("-"*120)
    p = sp.run(['orca_mergefrag', "calcA.gbw", "calcB.gbw", "promolecule_AB.gbw"], encoding='ascii')

    #NOTE: promolecule_AB.gbw here contains orbitals that have not been orthogonalize
    #Here we run a Noiter job to orthogonalize
    promolecule_AB_orthog = copy.copy(theory)
    promolecule_AB_orthog.filename="calcAB"
    promolecule_AB_orthog.orcasimpleinput+=" noiter"
    promolecule_AB_orthog.moreadfile="promolecule_AB.gbw"
    promolecule_AB_orthog.orcablocks="%scf guessmode fmatrix end"
    promolecule_AB_orthog.filename="promol"
    promolecule_AB_orthog.keep_last_output=False
    print()
    print("-"*120)
    print("Performing ORCA noiter calculation in order to orthogonalize orbitals and get file: promolecule_AB_orthog.gbw")
    print("-"*120)
    result_promol=ash.Singlepoint(theory=promolecule_AB_orthog, fragment=fragment_AB)
    #NOTE: calc_promol.gbw will contain  orthogonalized orbitals
    #Writing out electron density of orthogonalized promolecular electron density
    print()
    if make_cube_files is True:
        print("-"*120)
        print("Performing orca_plot calculation to create density Cubefile: promolecule_AB_orthogonalized.eldens.cube")
        print("-"*120)
        run_orca_plot(promolecule_AB_orthog.filename+".gbw", "density", gridvalue=80)
        os.rename(f"{promolecule_AB_orthog.filename}.eldens.cube","promolecule_AB_orthogonalized.eldens.cube")

    #----------------------------
    #Calculation on AB with NOCV
    #----------------------------
    #Run AB SP
    if NOCV is True:
        print()
        print("NOCV option on. Note that if system is open-shell then ORCA will not perform NOCV")
        calc_AB.orcablocks = calc_AB.orcablocks + """
%scf
EDA true
guessmode fmatrix
end
"""
        calc_AB.moreadfile="promolecule_AB.gbw"
    print()
    print("-"*120)
    print("Calling ORCA to perform calculation on AB")
    print("-"*120)
    result_calcAB=ash.Singlepoint(theory=calc_AB, fragment=fragment_AB)
    if make_cube_files is True:
        #Run orca_plot to request electron density creation from ORCA gbw file
        run_orca_plot("calcAB.gbw", "density", gridvalue=griddensity)

        #-----------------------------------------
        # Make deformation density as difference
        #-----------------------------------------

        #Read Cubefiles from disk
        print()
        print("-"*120)
        print("Reading Cubefiles and creating difference density (i.e. deformation density) from orthogonalized promolecular density and final density")
        print("-"*120)
        cube_data1 = read_cube("promolecule_AB_orthogonalized.eldens.cube")
        cube_data2 = read_cube(f"calcAB.eldens.cube")

        #Write out difference density as a Cubefile
        write_cube_diff(cube_data2, cube_data1, "full_deformation_density")
        print()
        print("Deformation density file was created: full_deformation_density.cube")
        print()


    #If nocv GBW file is present then NOCV was definitely carried out and we can calculate cube files of the donor-acceptor orbitals
    if os.path.isfile("calcAB.nocv.gbw") is False:
        print("No NOCV file was created by ORCA. This probably means that ORCA could not perform the NOCV calculation.")
        print("Possibly as the system is open-shell.")
        return

    #FURTHER
    print("NOCV analysis was carried out, see calcAB.out for details")
    print()
    print("-"*120)
    print("Running dummy ORCA noiter PrintMOS job using NOCV orbitals in file: calcAB.nocv.gbw ")
    print("-"*120)
    #Creating noiter ORCA output for visualization in Chemcraft
    calc_AB.orcasimpleinput+=" noiter printmos printbasis"
    calc_AB.moreadfile="calcAB.nocv.gbw"
    calc_AB.orcablocks=""
    calc_AB.filename="NOCV-noiter-visualization"
    calc_AB.keep_last_output=False
    result_calcAB_noiter=ash.Singlepoint(theory=calc_AB, fragment=fragment_AB)

    print()
    if make_cube_files is True:
        #Creating Cube files
        print("Now creating Cube files for main NOCV pairs and making orbital-pair deformation densities")
        print("Creating Cube file for NOCV total deformation density:")
        run_orca_plot("calcAB.nocv.gbw", "density", gridvalue=griddensity)
        os.rename(f"calcAB.nocv.eldens.cube", f"NOCV-total-density.cube")
        num_mos=int(pygrep("Number of basis functions                   ...", "calcAB.out")[-1])

        #Storing individual NOCV MOs and densities in separate dir (less useful)
        print("-"*120)
        print("Creating final Cube files for NOCV pair orbitals, orbital-densities and orbital-pair deformation densities")
        print("-"*120)
        try:
            os.mkdir("NOCV_orbitals_and_densities")
        except:
            pass
        for i in range(0,num_nocv_pairs):
            print("-----------------------")
            print(f"Now doing NOCV pair: {i}")
            print("-----------------------")
            print()
            print("Creating Cube file for NOCV donor MO number:", i)
            run_orca_plot("calcAB.nocv.gbw", "mo", mo_number=i, gridvalue=griddensity)
            os.rename(f"calcAB.nocv.mo{i}a.cube", f"calcAB.NOCVpair_{i}.donor_mo{i}a.cube")
            print("Creating density for orbital")
            create_density_from_orb (f"calcAB.NOCVpair_{i}.donor_mo{i}a.cube", denswrite=True, LargePrint=True)

            print("Creating Cube file for NOCV acceptor MO number:", num_mos-1-i)
            run_orca_plot("calcAB.nocv.gbw", "mo", mo_number=num_mos-1-i, gridvalue=griddensity)
            os.rename(f"calcAB.nocv.mo{num_mos-1-i}a.cube", f"calcAB.NOCVpair_{i}.acceptor_mo{num_mos-1-i}a.cube")
            print("Creating density for orbital")
            create_density_from_orb (f"calcAB.NOCVpair_{i}.acceptor_mo{num_mos-1-i}a.cube", denswrite=True, LargePrint=False)

            #Difference density for orbital pair
            donor = read_cube(f"calcAB.NOCVpair_{i}.donor_mo{i}a-dens.cube")
            acceptor = read_cube(f"calcAB.NOCVpair_{i}.acceptor_mo{num_mos-1-i}a-dens.cube")
            print(f"Making difference density file: NOCV_pair_{i}_deform_density.cube")
            write_cube_diff(acceptor,donor, name=f"NOCV_pair_{i}_deform_density")

            #Move less important stuff to dir
            os.rename(f"calcAB.NOCVpair_{i}.donor_mo{i}a.cube",f"NOCV_orbitals_and_densities/calcAB.NOCVpair_{i}.donor_mo{i}a.cube")
            os.rename(f"calcAB.NOCVpair_{i}.acceptor_mo{num_mos-1-i}a.cube",f"NOCV_orbitals_and_densities/calcAB.NOCVpair_{i}.acceptor_mo{num_mos-1-i}a.cube")
            os.rename(f"calcAB.NOCVpair_{i}.donor_mo{i}a-dens.cube",f"NOCV_orbitals_and_densities/calcAB.NOCVpair_{i}.donor_mo{i}a-dens.cube")
            os.rename(f"calcAB.NOCVpair_{i}.acceptor_mo{num_mos-1-i}a-dens.cube",f"NOCV_orbitals_and_densities/calcAB.NOCVpair_{i}.acceptor_mo{num_mos-1-i}a-dens.cube")
        #Optionally delete whole directory at end
        if keep_all_orbital_cube_files is False:
            print("keep_all_orbital_cube_files option is False")
            print("Deleting directory: NOCV_orbitals_and_densities")
            shutil.rmtree("NOCV_orbitals_and_densities")
        print()
    ###############################
    # FINAL EDA analysis printout

    deltaE_int=(result_calcAB.energy - result_calcA.energy - result_calcB.energy)*hartokcal
    deltaE_orb=float(pygrep("Delta Total Energy  (Kcal/mol) :","calcAB.out")[-1])
    deltaE_steric=deltaE_int-deltaE_orb #Elstat+Pauli. Further ecomposition not possibly at the moment

    print("="*20)
    print("Basic EDA analysis")
    print("="*20)
    print()
    print("-"*50)
    print(f"{'dE(steric)':<20s} {deltaE_steric:>14.3f} kcal/mol")
    print(f"{'dE(orb)':<20s} {deltaE_orb:>14.3f} kcal/mol")
    print(f"{'dE(int)':<20s} {deltaE_int:>14.3f} kcal/mol")
    print("-"*50)
    print("E(steric) is sum of electrostatic and Pauli repulsion")
    print("dE(orb) is the NOCV-ETS orbital-relaxation of orthogonalized promolecular system")
    print("dE(int) is the vertical total interaction energy (without geometric relaxation)")
    print()
    print()
    print("Primary NOCV/ETS orbital interactions:")
    neg_vals,pos_vals,dE_ints = grab_NOCV_interactions("calcAB.out")

    print("-"*70)
    print(f"{'Neg. eigvals (e)':20}{'Pos. eigvals (e)':20}{'dE_orb (kcal/mol)':20}")
    print("-"*70)
    for n,p,e in zip(neg_vals,pos_vals,dE_ints):
        print(f"{n:>10.3f} {p:>20.3f} {e:>20.3f}")
    print("-"*70)
    print(f"Sum of orbital interactions: {sum(dE_ints):>23.3f} kcal/mol")


def grab_NOCV_interactions(file):
    grab=False
    neg_eigenvals=[]
    pos_eigenvals=[]
    DE_k=[]
    with open(file) as f:
        for line in f:
            if 'Consistency' in line:
                grab=False
            if grab is True:
                if len(line) >2:
                    neg_eigenvals.append(float(line.split()[0]))
                    pos_eigenvals.append(float(line.split()[1]))
                    DE_k.append(float(line.split()[-1]))
            if 'negative eigen. (e)' in line:
                grab=True

    return neg_eigenvals,pos_eigenvals,DE_k

#NOCV analysis using Multiwfn
#Need to figure out how to generalize more.
#If Molden files is the best for Multiwfn then theory levels need to create those.
#TODO: Make internal theory methods for ORCATheory, xTBtheory, PySCF etc. ?? that outputs a Molden file ???
#NOTE: Benefit, multiwfn supports open-shell analysis
#NOTE: Proper ETS analysis by fockmatrix_approximation="ETS"
#NOTE: fockmatrix_approximation: regular gives approximate energies, same as Multiwfn
def NOCV_Multiwfn(fragment_AB=None, fragment_A=None, fragment_B=None, theory=None, gridlevel=2, openshell=False,
                            num_nocv_pairs=5, make_cube_files=True, numcores=1, fockmatrix_approximation="ETS"):
    print_line_with_mainheader("NOCV_Multiwfn")
    print("Will do full NOCV analysis with Multiwfn")
    print("gridlevel:", gridlevel)
    print("Numcores:", numcores)
    print()

    if fragment_AB.mult > 1 or fragment_A.mult > 1 or fragment_B.mult > 1:
        print("Multiplicity larger than 1. Setting openshell equal to True")
        openshell=True

    print("Openshell:", openshell)
    if isinstance(theory,ORCATheory) is not True:
        print("NOCV_Multiwfn currently only works with ORCATheory")
        ashexit()
    #A
    result_calcA=ash.Singlepoint(theory=theory, fragment=fragment_A)
    make_molden_file_ORCA(theory.filename+'.gbw') #TODO: Generalize
    os.rename("orca.molden.input", "A.molden.input")
    theory.cleanup()

    #B
    result_calcB=ash.Singlepoint(theory=theory, fragment=fragment_B)
    make_molden_file_ORCA(theory.filename+'.gbw')
    os.rename("orca.molden.input", "B.molden.input")
    theory.cleanup()

    #PromolAB
    original_orcablocks=theory.orcablocks #Keeping
    blockaddition="""
    %output
    Print[P_Iter_F] 1
    end
    %scf
    maxiter 1
    end
    """
    theory.orcablocks=theory.orcablocks+blockaddition
    theory.ignore_ORCA_error=True #Otherwise ORCA subprocess will fail due to maxiter=1 fail
    result_calcAB=ash.Singlepoint(theory=theory, fragment=fragment_AB)
    shutil.copy(f"{theory.filename}.out", "promol.out")
    #Get Fock matrix of promolstate I
    Fock_Pi_a, Fock_Pi_b = read_Fock_matrix_from_ORCA(f"{theory.filename}.out")
    np.savetxt("Fock_Pi_a",Fock_Pi_a)
    #exit()
    make_molden_file_ORCA(theory.filename+'.gbw')
    os.rename("orca.molden.input", "AB.molden.input")

    #AB
    theory.ignore_ORCA_error=False #Reverting
    theory.orcablocks=original_orcablocks+"%output Print[P_Iter_F] 1 end"
    result_calcAB=ash.Singlepoint(theory=theory, fragment=fragment_AB)
    #Get Fock matrix of Finalstate F
    Fock_Pf_a, Fock_Pf_b = read_Fock_matrix_from_ORCA(f"{theory.filename}.out")
    np.savetxt("Fock_Pf_a",Fock_Pf_a)
    make_molden_file_ORCA(theory.filename+'.gbw')
    os.rename("orca.molden.input", "AB.molden.input")

    #Extended transition state
    Fock_ETS_a = 0.5*(Fock_Pi_a + Fock_Pf_a)
    print("Fock_ETS_a:", Fock_ETS_a)
    if openshell is True:
        print("Fock_Pi_b:", Fock_Pi_b)
        if Fock_Pi_b is None:
            print("No beta Fock matrix found in ORCA output. Make sure UHF/UKS keywords were added")
            ashexit()
        print("Fock_Pi_b:", Fock_Pi_b)
        print("Fock_Pf_b:", Fock_Pf_b)
        Fock_ETS_b = 0.5*(Fock_Pi_b +Fock_Pf_b)
        print("Fock_ETS_b:", Fock_ETS_b)
    else:
        Fock_ETS_b=None

    #Write ETS Fock matrix in lower-triangular form for Multiwfn: F(1,1) F(2,1) F(2,2) F(3,1) F(3,2) F(3,3) ... F(nbasis,nbasis)
    if fockmatrix_approximation  == 'ETS':
        print("fockmatrix_approximation: ETS")
        fockfile="Fock_ETS"
        print("Fock_ETS_a:", Fock_ETS_a)
        print("Fock_ETS_b:", Fock_ETS_b)
        write_Fock_matrix_ORCA_format(fockfile, Fock_a=Fock_ETS_a,Fock_b=Fock_ETS_b, openshell=openshell)
    elif fockmatrix_approximation  == 'initial':
        print("fockmatrix_approximation: initial (unconverged AB Fock matrix)")
        fockfile="Fock_Pi"
        print("Fock_Pi_a:", Fock_Pi_a)
        print("Fock_Pi_b:", Fock_Pi_b)
        write_Fock_matrix_ORCA_format(fockfile, Fock_a=Fock_Pi_a,Fock_b=Fock_Pi_b, openshell=openshell)
    elif fockmatrix_approximation  == 'final':
        print("fockmatrix_approximation: final (converged AB Fock matrix)")
        fockfile="Fock_Pf"
        print("Fock_Pf_a:", Fock_Pf_a)
        print("Fock_Pf_b:", Fock_Pf_b)
        write_Fock_matrix_ORCA_format(fockfile, Fock_a=Fock_Pf_a,Fock_b=Fock_Pf_b, openshell=openshell)
    else:
        print("Unknown fockmatrix_approximation")
        ashexit()
    print("fockfile:", fockfile)
    #NOTE: Important Writing Fock matrix in ORCA format (with simple header) so that Multiwfn recognized it as such and used ORCA ordering of columns
    # Writing out as simple lower-triangular form does not work due to weird column swapping

    #Call Multiwfn
    multiwfn_run("AB.molden.input", option='nocv', grid=gridlevel,
                    fragmentfiles=["A.molden.input","B.molden.input"],
                    fockfile=fockfile, numcores=numcores, openshell=openshell)

    #OTOD: openshell
    deltaE_int=(result_calcAB.energy - result_calcA.energy - result_calcB.energy)*hartokcal
    deltaE_orb=float(pygrep(" Sum of pair energies:","NOCV.txt")[-2])
    deltaE_steric=deltaE_int-deltaE_orb #Elstat+Pauli. Further ecomposition not possibly at the moment

    print()
    print("="*20)
    print("Basic EDA analysis")
    print("="*20)
    print()
    print("-"*50)
    print(f"{'dE(steric)':<20s} {deltaE_steric:>14.3f} kcal/mol")
    print(f"{'dE(orb)':<20s} {deltaE_orb:>14.3f} kcal/mol")
    print(f"{'dE(int)':<20s} {deltaE_int:>14.3f} kcal/mol")
    print("-"*50)
    print("E(steric) is sum of electrostatic and Pauli repulsion")
    print("dE(orb) is the NOCV-ETS orbital-relaxation of orthogonalized promolecular system")
    if fockmatrix_approximation == "initial" or fockmatrix_approximation == "final":
        print("Warning: Fock matrix approximation is initial or final")
        print("Warning: dE(orb) term is approximated when calculated by Multiwfn (as the correct TS Fock matrix is not used)")
    print("dE(int) is the vertical total interaction energy (without geometric relaxation)")

    #TODO: Grab orbital-interaction stuff from NOCV.txt and print here also
    print()
    print("TODO: NOCV orbital table to come here. See NOCV.txt for now")


def read_Fock_matrix_from_ORCA(file):
    grabA=False
    grabB=False
    foundbeta=False
    i_counter=0
    Fock_matrix_a=None;Fock_matrix_b=None
    Bcounter=None; Acounter=None
    with open(file) as f:
        for line in f:
            if 'Number of basis functions                   ...' in line:
                ndim=int(line.split()[-1])
            if grabA is True:
                Acounter+=1
                if Acounter % (ndim+1) == 0:
                    col_indices=[int(i) for i in line.split()]
                if Acounter >= 1:
                    line_vals=[float(i) for i in line.split()[1:]]
                    for colindex,val in zip(col_indices,line_vals):
                        a=colindex
                        b=int(line.split()[0])
                        Fock_matrix_a[b,a] = val
                        i_counter+=1
                    if a == b == ndim-1:
                        grabA=False
            if grabB is True:
                Bcounter+=1
                if Bcounter % (ndim+1) == 0:
                    col_indices=[int(i) for i in line.split()]
                if Bcounter >= 1:
                    line_vals=[float(i) for i in line.split()[1:]]
                    for colindex,val in zip(col_indices,line_vals):
                        a=colindex
                        b=int(line.split()[0])
                        Fock_matrix_b[b,a] = val
                        i_counter+=1
                    if a == b == ndim-1:
                        grabB=False
            if 'Fock matrix for operator 0' in line:
                grabA=True
                Acounter=-1
                Fock_matrix_a=np.zeros((ndim,ndim))
            if 'Fock matrix for operator 1' in line:
                foundbeta=True
                grabB=True
                Bcounter=-1
                Fock_matrix_b=np.zeros((ndim,ndim))
    #Write
    np.savetxt("Fock_matrix_a",Fock_matrix_a)
    if foundbeta is True:
        print("Found beta Fock matrix")
        np.savetxt("Fock_matrix_b",Fock_matrix_b)
    else:
        Fock_matrix_b=None
    return Fock_matrix_a, Fock_matrix_b


def write_Fock_matrix_ORCA_format(outputfile, Fock_a=None,Fock_b=None, openshell=False):
    print("Fock_a:", Fock_a)
    print("Fock_b:", Fock_b)
    print("Writing Fock matrix alpha")
    with open(outputfile,'w') as f:
        f.write("                                 *****************\n")
        f.write("                                 * O   R   C   A *\n")
        f.write("                                 *****************\n")
        f.write(f"Fock matrix for operator 0\n")
        #f.write("\n")
        Fock_alpha = get_Fock_matrix_ORCA_format(Fock_a)
        f.write(Fock_alpha)
        #f.write("\n")
        if openshell is True:
            print("Writing Fock matrix beta")
            f.write(f"Fock matrix for operator 1\n")
            f.write("\n")
            Fock_beta = get_Fock_matrix_ORCA_format(Fock_b)
            f.write(Fock_beta)

#Get
def get_Fock_matrix_ORCA_format(Fock):
    finalstring=""
    dim=Fock.shape[0]
    orcacoldim=6
    index=0
    tempvar=""
    chunks=dim//orcacoldim
    left=dim%orcacoldim
    xvar="                  "
    col_list=[]
    if left > 0:
        chunks=chunks+1
    for chunk in range(chunks):
        if chunk == chunks-1:
            if left == 0:
                left=6
            for temp in range(index,index+left):
                col_list.append(str(temp))
        else:
            for temp in range(index,index+orcacoldim):
                col_list.append(str(temp))
        col_list_string='          '.join(col_list)
        finalstring=finalstring+f"{xvar}{col_list_string}\n"
        col_list=[]
        for i in range(0,dim):

            if chunk == chunks-1:
                for k in range(index,index+left):
                    valstring=f"{Fock[i,k]:9.6f}"
                    tempvar=f"{tempvar}  {str(valstring)}"
            else:
                for k in range(index,index+orcacoldim):
                    valstring=f"{Fock[i,k]:9.6f}"
                    tempvar=f"{tempvar}  {str(valstring)}"
            finalstring=finalstring+f"{i:>7d}    {tempvar}\n"
            tempvar=""
        index+=6
    return finalstring

# From examples/density-analysis/difference-density/difference-density-and-WFT/diffdens-from-GBW-and-NAT-files.py
#Attempt to modularize this script

def diffdens_of_cubefiles(ref_cubefile, cubefile):
    print("Inside diffdens_of_cubefiles function")
    print("ref_cubefile:", ref_cubefile)
    print("cubefile:", cubefile)
    print()
    #Labels
    reffile_base=str(os.path.splitext(ref_cubefile)[0])
    cubefile_base=str(os.path.splitext(cubefile)[0])
    #Read Cubefiles into memory
    cube_ref=read_cube(ref_cubefile)
    cube_other=read_cube(cubefile)
    #Taking diff
    diffdens_filename=f"{cubefile_base}_{reffile_base}_diff_density"
    num_el_val,num_el_val_pos,num_el_val_neg = write_cube_diff(cube_ref, cube_other, diffdens_filename)
    print("Wrote diffdens-file :", diffdens_filename+".cube")
    return diffdens_filename+".cube",num_el_val,num_el_val_pos,num_el_val_neg


#Takes input either ORCA-GBWfile, ORCA_natorbfile or Moldenfile
def create_cubefile_from_orbfile(orbfile, option='density', grid=3, delete_temp_molden_file=True, printlevel=2):
    orcafile=False
    #First checking if input is a Molden file
    if '.molden' in orbfile or 'MOLDEN' in orbfile:
        print(f"Orbfile ({orbfile}) recognized as a Molden-file")
        moldenfile=True
        mfile=orbfile
    elif '.gbw' in orbfile:
        print(f"Orbfile ({orbfile}) recognized as ORCA GBW file")
        orcafile=True
    elif '.nat' in orbfile:
        print(f"Orbfile ({orbfile}) recognized as ORCA natural-orbital file")
        orcafile=True
    elif 'mp2nat' in orbfile:
        print(f"Orbfile ({orbfile}) recognized as ORCA natural-orbital file")
        orcafile=True


    if orcafile is True:
        print("Now using orca_2mkl to convert ORCA file to Molden file")
        # Create Molden file from GBW
        mfile = make_molden_file_ORCA(orbfile, printlevel=printlevel)
    print("Now using Multiwfn to create cube file from Moldenfile")
    cubefile = multiwfn_run(mfile, option=option, grid=grid, printlevel=printlevel)
    # Rename cubefile (shortens it)
    new_cubename=str(os.path.splitext(orbfile)[0])+".cube"
    os.rename(cubefile, new_cubename)
    print("Cube file renamed:", new_cubename)
    if delete_temp_molden_file is True:
        if orcafile is True:
            print("Removing preliminary Moldenfile created from ORCA file")
            os.remove(mfile)

    return new_cubename


def diffdens_tool(option='density', reference_orbfile="HF.gbw", dir='.', grid=3, printlevel=2):
    print_line_with_mainheader("diffdens_tool")
    print()
    print("Warning: ORCA natural-orbital files need to have the ending nat or they will not be found!")
    print("Warning: Molden files need to have the ending .molden or .MOLDEN or they will not be found!")
    print()
    print("Reference orbital file:", reference_orbfile)
    print("Directory:", dir)
    print("Gridsetting:", grid)
    print("Printlevel:", printlevel)
    print(f"Option: {option} (options are: density, valence-density)")
    print()
    ##############################################################
    # Difference density generation script via ORCA orbitalfiles and Moldenf
    ##############################################################
    # Defines a reference orbitalfile (GBW,NAT,Molden) and creates a Cubefile
    # Finds other ORCA .gbw (for SCF) or nat files (natural orbitals from WFT) or Moldenfiles
    # Creates Cubefiles from these files via Multiwfn (via a moldenfile)
    #Difference-densities generated by subtraction from reference
    ##############################################################

    os.chdir(dir)

    #Reference for Difference density cubes
    #Check if reference file is in dir
    if os.path.isfile(reference_orbfile) is False:
        print(f"Reference orbital file: {reference_orbfile} not found in directory: {dir}")
        ashexit()

    #Create Molden file from reference-orbital file
    print("Creating Cube file from reference file")
    #TODO: Check if cube file already exists
    ref_orbfile_base=str(os.path.splitext(reference_orbfile)[0])
    if os.path.isfile(f"{ref_orbfile_base}.cube") is True:
        print("Cube file created from this reference file already exists. Skipping Cube generation.")
        ref_cubefile=f"{ref_orbfile_base}.cube"
    else:
        ref_cubefile = create_cubefile_from_orbfile(reference_orbfile, option=option, grid=grid, printlevel=printlevel)
    print("Reference cubefile that will be used:", ref_cubefile)

    ###################################

    #Find all GBW,NAT and Molden files in dir
    print("\nNow searching dir for .gbw, nat and Molden files")
    gbwfiles=glob.glob('*.gbw')
    natfiles=glob.glob('*nat')
    moldenfiles=glob.glob('*molden')
    print("Found gbwfiles", gbwfiles)
    print("Found natfiles:", natfiles)
    print("Found moldenfiles:", moldenfiles)
    orbfiles=gbwfiles+natfiles+moldenfiles

    print("-"*70)
    print("Total orbfiles:", orbfiles)
    print("\nNow looping over orbfiles, creating Cubefiles and taking difference with respect to reference")
    diff_files=[]
    num_el_vals=[]
    num_el_vals_pos=[]
    num_el_vals_neg=[]
    #Looping over orbital-files (GBW or NAT)
    for orbfile in orbfiles:
        orbfile_base=str(os.path.splitext(orbfile)[0])
        #Check if Cube-file exists already
        if os.path.exists(orbfile_base+".cube"):
            print("Cubefile exists already:", orbfile_base+".cube")
            print("Skipping")
        else:
            print("Creating Cubefile from Orbfile:", orbfile)
            cube_f = create_cubefile_from_orbfile(orbfile, option=option, grid=grid, printlevel=printlevel)
            print("Now calculating difference density")
            diff_file,num_el_val,num_el_val_pos,num_el_val_neg = diffdens_of_cubefiles(ref_cubefile, cube_f)
            diff_files.append(diff_file)
            num_el_vals.append(num_el_val)
            num_el_vals_pos.append(num_el_val_pos)
            num_el_vals_neg.append(num_el_val_neg)



    print("\n All done. Difference density files created:")
    if len(diff_files) > 0:
        max_file_length = max([len(i) for i in diff_files])
        print_pretty_table(list_of_objects=[diff_files,num_el_vals,num_el_vals_pos,num_el_vals_neg],
                       list_of_labels=["File","Sum all","Sum of pos. val.","Sum of neg. val."],
                       title="",  spacing=20, divider_line_length=120)

    return diff_files, num_el_vals, num_el_vals_pos, num_el_vals_neg


def DM_AO_to_MO(DM_AO, C,S):
    from functools import reduce
    print("-"*50)
    print("Converting DM from AO to MO basis")
    print("-"*50)
    #Checking if DM_AO is symmetric
    if np.allclose(DM_AO,np.transpose(DM_AO)) is True:
            print("DM_AO is symmetric.")
    else:
        print("Error: Input DM_AO is not symmetric.")
    #Print trace
    print("Trace of input DM_AO:", np.trace(DM_AO))
    print("Trace of input DM_AO*S:", np.trace(np.dot(DM_AO,S)))
    DM_MO = reduce(np.dot, (C.T, S, DM_AO, S, C))
    if np.allclose(DM_MO,np.dot(DM_MO,DM_MO)):
        print("idempotent")
    else:
        print("not idempotent")
    print("Trace of output DM_MO:", np.trace(DM_MO))
    print("-"*30)
    return DM_MO

def DM_MO_to_AO(DM_MO, C):
    print("-"*50)
    print("Converting DM from MO to AO basis")
    print("-"*50)
    from functools import reduce
    print("Converting DM from MO to AO basis")
    #Checking if DM_MO is symmetric
    if np.allclose(DM_MO,DM_MO.T) is True:
            print("DM_MO is symmetric.")
    else:
        print("Error: Input DM_MO is not symmetric.")
    #Print trace
    print("Trace of input DM_MO:", np.trace(DM_MO))
    DM_AO = reduce(np.dot, (C, DM_MO, C.T ))
    print("-"*50)
    return DM_AO

#Diagonalize density matrix in AO basis
def diagonalize_DM_AO(D, S):
    print("Diagonalizing density matrix")
    import scipy.linalg
    from functools import reduce
    # Diagonalize the DM in AO basis
    print("Trace of input DM_AO:", np.trace(D))
    print("Trace of input DM_AO*S:", np.trace(np.dot(D,S)))
    A = reduce(np.dot, (S, D, S))
    w, v = scipy.linalg.eigh(A, b=S)
    # Flip NOONs (and NOs) since they're in increasing order
    natocc = np.flip(w)
    #print("NOs before flip:", v)
    natorb = np.flip(v, axis=1)
    print("Natural orbital coefficients:", natorb)
    print("Natural occupations:", natocc)
    return natorb, natocc

#Diagonalize density matrix in MO basis
def diagonalize_DM(D):
    print("Diagonalizing density matrix directly")
    import scipy.linalg
    #Diagonalize
    w, v = scipy.linalg.eigh(D)
    # Flip NOONs (and NOs) since they're in increasing order
    natocc = np.flip(w)
    natorb = np.flip(v, axis=1)
    print("Natural orbital coefficients:", natorb)
    print("Natural occupations:", natocc)
    return natorb, natocc

#Change nat orbitals from MO basis to AO basis by multiplying with MO coeffs
def convert_NOs_from_MO_to_AO(NOs,C):
    return np.dot(C,NOs)


#Function that creates Molden file from ASH fragment and MO coefficients, occupations and basis set
#https://www.theochem.ru.nl/molden/molden_format.html
#https://github.com/psi4/psi4/issues/504
#https://github.com/psi4/psi4/issues/60
#https://github.com/psi4/psi4/blob/master/psi4/driver/p4util/writer.py
def make_molden_file(fragment, AO_basis, MO_coeffs, MO_energies=None, MO_occs=None, AO_order=None, label="ASH_orbs", spherical_MOs=True):

    print_line_with_mainheader("make_molden_file")

    print()
    print("Will make Molden file from ASH fragment, input MO coefficients and occupations")
    print("ORCA-formatting of Molden file will be used")
    print("Optional input: MO energies and MO occupations")

    print("WARNING: NORMALIZATION may not be entirely correct")
    print("WARNING: ORDER has only been checked for s,p,d and f")

    if AO_order is None:
        print("Error: no AO_order given.")
        ashexit()
    else:
         print("AO_order_object given. Will use this to order AOs")
         print(AO_order)

    ############
    #Geometry
    ############
    header="""[Molden Format]
[Title]
Molden file created by ASH (using orca format)

[Atoms] Angs
"""
    coords_string=""
    for i,(el,c) in enumerate(zip(fragment.elems,fragment.coords)):
        cline = f"{el:2s}   {i+1}   {elematomnumbers[el.lower()]}          {c[0]:11.6f}         {c[1]:11.6f}         {c[2]:11.6f}"
        coords_string+=cline+'\n'

    ###########
    #Basis set
    ###########
    #TODO. Check
    shell_to_angmom = {"s":0,"p":1,"d":2,"f":3}
    gtostring="""[GTO]
"""
    #Looping over each atom in AO_basis object
    print("AO_basis:", AO_basis)
    #exit()
    for i,atom in enumerate(AO_basis):
        gtostring+=f"  {i+1} 0\n"
        bfs = atom["Basis"]
        for bf in bfs:
            coeffs=bf["Coefficients"]
            exponents=bf["Exponents"]
            shell=bf["Shell"]
            angmom = shell_to_angmom[shell]
            extra=1.0
            gtostring+=f"{shell}   {len(coeffs)} {extra}\n"
            for exp,coeff in zip(exponents,coeffs):
                N = normalization_ORCA(angmom,exp)
                coeff_used = coeff*N
                gtostring+=f"       {exp} {coeff_used}\n"
        gtostring+="\n"

    #Applies to ORCA
    if spherical_MOs:
        gtostring+="[5D]\n"
        gtostring+="[7F]\n"
        gtostring+="[9G]\n"

    #############
    #MO-stuff
    #############
    #EnergyUnit=MO_object["EnergyUnit"]
    #OrbitalLabels=MO_object["OrbitalLabels"]

    num_mos = len(MO_coeffs)
    print("There are", num_mos, "MOs")

    #Setting dummy values for energies and occupations if not provided
    if MO_energies is None:
        MO_energies = [0.0 for i in range(num_mos)]
    if MO_occs is None:
        MO_occs = [1.0 for i in range(num_mos)]

    mostring="""[MO]
"""
    #Loop over MOs
    #print("mo_coeffs:",MO_coeffs)
    #print("mo_coeffs:",MO_coeffs[0])
    #exit()
    print("Warning: transposing MO_coeffs for convenience")
    MO_coeffs=np.transpose(MO_coeffs)

    for i,(mo_coeffs,mo_en,mo_occ) in enumerate(zip(MO_coeffs,MO_energies,MO_occs)):
        moheader=f""" Sym=     1a
 Ene= {mo_en}
 Spin= Alpha
 Occup= {mo_occ}\n"""
        mostring+=moheader
        #Reorder according to AO_order_object
        #print("mostring:",mostring)
        factor=-1 #Sign change
        mo_coeffs_reordered =  reorder_AOs_in_MO_ORCA_to_Molden(mo_coeffs,AO_order)
        for i,mo_coeff in enumerate(mo_coeffs_reordered):
            mo_coeff=mo_coeff*factor
            mostring+=f" {i+1}      {mo_coeff:15.12f}\n"

    #Combine and write out
    with open(f"{label}.molden", "w") as mfile:
        mfile.write(header)
        mfile.write(coords_string)
        mfile.write(gtostring)
        mfile.write(mostring)

    print(f"Created Molden file: {label}.molden")
    return f"{label}.molden"

#Function that does the ORCA BF normalization
def normalization_ORCA(L,exp):
    bla ={0:[3,3,1],1:[7,5,1],2:[11,7,9],3:[15,9,225],4:[19,11,11025],5:[23,13,893025]}
    nvals=bla[L]
    n1=nvals[0];n2=nvals[1];nf=nvals[2]
    if L == 2:
        renorm_orca=math.sqrt(3)*math.sqrt(math.sqrt(2**n1*exp**n2/(math.pi**3*nf)))
    elif L == 3:
        renorm_orca=math.sqrt(15)*math.sqrt(math.sqrt(2**n1*exp**n2/(math.pi**3*nf)))
    else:
        renorm_orca=math.sqrt(math.sqrt(2**n1*exp**n2/(math.pi**3*nf)))
    return renorm_orca

#From ORCA order to Molden order
def reorder_AOs_in_MO_ORCA_to_Molden(coeffs,order):
    new_coeffs=np.zeros(len(coeffs))
    new_order = np.empty(len(order), dtype=object)

    for i,(c,o) in enumerate(zip(coeffs,order)):
        #print("i:",i)
        #print("c:",c)
        #print("o:",o)
        #exit()
        if "pz" in o:
            new_coeffs[i+2] = c
            new_order[i+2] = o
        elif "px" in o:
            new_coeffs[i-1] = c
            new_order[i-1] = o
        elif "py" in o:
            new_coeffs[i-1] = c
            new_order[i-1] = o
        elif "dz2" in o:
            new_coeffs[i] = c
            new_order[i] = o
        elif "dyz" in o:
            new_coeffs[i] = c
            new_order[i] = o
        elif "dxz" in o:
            new_coeffs[i] = c
            new_order[i] = o
        elif "dxy" in o:
            new_coeffs[i] = c
            new_order[i] = o
        elif "dx2y2" in o:
            new_coeffs[i] = c
            new_order[i] = o
        elif "f0" in o:
            new_coeffs[i] = c
            new_order[i] = o
        elif "f+1" in o:
            new_coeffs[i] = c
            new_order[i] = o
        elif "f-1" in o:
            new_coeffs[i] = c
            new_order[i] = o
        elif "f+2" in o:
            new_coeffs[i] = c
            new_order[i] = o
        elif "f-2" in o:
            new_coeffs[i] = c
            new_order[i] = o
        elif "f+3" in o:
            new_coeffs[i] = c
            new_order[i] = o
        elif "f-3" in o:
            new_coeffs[i] = c
            new_order[i] = o
        else:
            #s and others
            new_coeffs[i] = c
            new_order[i] = o
        #print("new_coeffs:",new_coeffs)
        #print("new_order:",new_order)
    return  new_coeffs

#Basic reading of molden_file
#Currently only reads atoms and coordinates
def read_molden_file(moldenfile):
    molden_properties_dict={}
    grab_atoms=False
    elems=[]
    coords=[]
    coord_scaling=1.0
    with open(moldenfile) as f:
        for line in f:
            if grab_atoms:
                if len(line.split()) == 6:
                    el = line.split()[0]
                    coord_x = float(line.split()[3])*coord_scaling
                    coord_y = float(line.split()[4])*coord_scaling
                    coord_z = float(line.split()[5])*coord_scaling
                    elems.append(el)
                    coords.append([coord_x,coord_y,coord_z])

            if '[Atoms]' in line:
                if 'AU' in line:
                    coord_scaling=0.529177
                else:
                    coord_scaling=1.0
                grab_atoms=True
    molden_properties_dict["elems"]=elems
    molden_properties_dict["coords"]=np.array(coords)

    return molden_properties_dict


#Polyradical character metrics head-Gordon
#HG-1
def poly_rad_index_nu(occupations):
    n_u=0.0
    for on in occupations:
        n_u+=min(on,2-on)
    print(f"HG-1 Number of effective unpaired electrons: {n_u:7.4f}")
    return n_u
#HG-2
def poly_rad_index_nu_nl(occupations):
    n_u_nl=0.0
    for on in occupations:
        n_u_nl+=on**2*((2-on)**2)
    print(f"HG-2 Number of effective unpaired electrons: {n_u_nl:7.4f}")
    return n_u_nl

#Original by Takatsuka and Staroverov and Davidson
#Can overestimate number of unpaired electrons
def poly_rad_index_n_d(occupations):
    n=0.0
    for on in occupations:
        n+=on*(2-on)
    print(f"Takatsuka Number of effective unpaired electrons: {n:7.4f}")
    return n




#General pcgradient function, inspired by pyscf
#NOT READY
def general_pointcharge_gradient(qm_coords, qm_charges,mm_coords,mm_charges,dm, mol):
    print("not ready")
    ashexit()
    if dm.shape[0] == 2:
        dmf = dm[0] + dm[1] #unrestricted
    else:
        dmf=dm
    # The interaction between QM atoms and MM particles
    # \sum_K d/dR (1/|r_K-R|) = \sum_K (r_K-R)/|r_K-R|^3
    dr = qm_coords[:,None,:] - mm_coords
    r = np.linalg.norm(dr, axis=2)
    g = np.einsum('r,R,rRx,rR->Rx', qm_charges, mm_charges, dr, r**-3)
    # The interaction between electron density and MM particles
    # d/dR <i| (1/|r-R|) |j> = <i| d/dR (1/|r-R|) |j> = <i| -d/dr (1/|r-R|) |j>
    #   = <d/dr i| (1/|r-R|) |j> + <i| (1/|r-R|) |d/dr j>
    for i, q in enumerate(mm_charges):
        with mol.with_rinv_origin(mm_coords[i]):
            v = mol.intor('int1e_iprinv')
        f =(np.einsum('ij,xji->x', dmf, v) +
            np.einsum('ij,xij->x', dmf, v.conj())) * -q
        g[i] += f
        #
        #TODO: Check with_rinv_orgin thingng

    return g

#Get electron correlation energy as a function of occupation numbers, sigma and the chosen distribution
def get_ec_entropy(occ, sigma, method='fermi', alpha=0.6):
    from scipy.special import erfinv
    f = occ/2.0
    f = f[(f>0) & (f<1)]
    mask=f>0.5
    f[mask] = 1.0-f[mask]
    if method=='fermi':
        fc = f*np.log(f) + (1-f)*np.log(1-f)
    elif method == 'gaussian':
        fc = -np.exp(-(erfinv(1-f*2))**2)/2.0/np.sqrt(np.pi)
    elif method == 'linear':
        fc = -f+np.sqrt(2)*f**(3.0/2.0)*2.0/3.0
    else:
        raise ValueError('Not support', method)
    Ec = 2.0*sigma*fc.sum()
    return Ec

#Calculate entropy from occupation array
#Assuming array of natural occupations (from 2.0 to 0.0)
def get_entropy(occupations):
    #Dividing by 2
    occ_2 = np.array(occupations)/2
    #Removing 0 and 1
    occ_2 = occ_2[(occ_2>0) & (occ_2<1)]
    S=0.0
    for o in occ_2:
        S+=(o*math.log(o)+(1-o)*math.log(1-o))
    return S




def yoshimine_sort(a,b,c,d):
    if a > b:
        ab = a*(a+1)/2 + b
    else:
        ab = b*(b+1)/2 + a
    if c > d:
        cd = c*(c+1)/2 + d
    else:
        cd = d*(d+1)/2 + c
    if ab > cd:
        abcd = ab*(ab+1)/2 + cd
    else:
        abcd = cd*(cd+1)/2 + ab
    return math.floor(abcd)



#General function to write FCIDUMP style integral file from Numpy arrays
# Support for different headers
# TODO: unrestricted case
# TODO: symmetry
# Confirmed to work for MRCC
def ASH_write_integralfile(two_el_integrals=None, one_el_integrals=None, nuc_repulsion_energy=None, header_format="MRCC",
                            num_corr_el=None, filename=None, int_threshold=1e-16, scf_type="RHF", mult=None):

    print("\nASH_write_integralfile")
    print()
    if two_el_integrals is None or one_el_integrals is None or nuc_repulsion_energy is None or num_corr_el is None:
        print("Error: two_el_integrals, one_el_integrals, num_corr_el or nuc_repulsion_energy not provided")
        ashexit()
    if mult is None:
        print("Please provide the spin multiplicity using the mult keyword")
        ashexit()

    print(f"Header format: {header_format} (options: FCIDUMP, MRCC)")
    print("filename:", filename)
    print("SCF_type:", scf_type)
    if scf_type == 'RHF' or scf_type == "ROHF":
        pass
    elif scf_type == 'UHF':
        print("Error: UHF not yet implemented")
        ashexit()
    basis_dim = one_el_integrals[0].size

    # Header
    if header_format == "FCIDUMP":
        #NORB: number of basis functions
        #NELEC: number of correlated electrons
        #MS2: TODO
        #isym: 
        #orbsym
        isym=1
        orbsymstring=','.join(str(1) for i in range(0,basis_dim))
        ms2=mult-1 # unpaired electrons
        uhf_option_string = ""
        if scf_type == "UHF":
            uhf_option_string = "UHF=.TRUE.,"
        header=f"""&FCI NORB={basis_dim}, NELEC={num_corr_el}, MS2={ms2},
ORBSYM={orbsymstring},
ISYM={isym},{uhf_option_string}
&END
"""
        if filename is None:
            filename="FCIDUMP"
            print("FCIDUMP option:, filename set to:", filename)
    elif header_format == "MRCC":
        # Note: assuming no symmetry setting 1 as irrep for each orbital
        header = f"""    {basis_dim}    {num_corr_el}
    {'  '.join('1' for i in range(basis_dim))}
    150000
    """
        filename="fort.55"
        print("MRCC option:, filename set to:", filename)

    print("Integral threshold:", int_threshold)
    num_integrals = two_el_integrals.shape[0]**4
    print("num_integrals:", num_integrals)

    # Integral dict
    from collections import OrderedDict
    int_1el_dict=OrderedDict()

    # 1-electron integrals (using 0 as dummy 3rd and 4th index)
    for m in range(0,basis_dim):
        for n in range(m,basis_dim):
            int_value=one_el_integrals[m,n]
            int_1el_dict[(m,n)] = [int_value,[m+1,n+1,0,0]]

    # 2-electron integrals
    npair = basis_dim*(basis_dim+1)//2

    # Open file
    f = open(filename, 'w')

    # Write header
    f.write(header)

    # Set up 2-electron integrals
    two_el_integral_string=""

    print("two_el_integrals.ndim:", two_el_integrals.ndim)
    print("two_el_integrals.size:", two_el_integrals.size)

    # Tested with pyscf : eri = ao2mo.full(theory.mol, theory.mf.mo_coeff, verbose=0)
    if two_el_integrals.ndim == 2:
        print("ndim 2, assuming 4-fold symmetry")
        xint_2el_dict=OrderedDict()
        # 4-fold symmetry
        assert (two_el_integrals.size == npair**2)
        ij = 0
        for i in range(basis_dim):
            for j in range(0, i+1):
                kl = 0
                for k in range(0, basis_dim):
                    for l in range(0, k+1):
                        if abs(two_el_integrals[ij,kl]) > int_threshold:
                            xint_2el_dict[(i+1, j+1, k+1, l+1)] = two_el_integrals[ij,kl]
                        kl += 1
                ij += 1
        # Creating string for 2-el integrals
        for k,v in xint_2el_dict.items():
            two_el_integral_string+=f"{v:>29.20E}{k[0]:>5}{k[1]:>5}{k[2]:>5}{k[3]:>5}\n"

    elif two_el_integrals.ndim == 4:
        print("ndim 4")
        int_2el_dict=OrderedDict() #yos_value : [int_value,[i,j,k,l ]]  Note, switching to 1-based indexing here
        # 2-electron integrals
        for i in range(0,basis_dim):
            for j in range(0,basis_dim):
                for k in range(0,basis_dim):
                    for l in range(0,basis_dim):
                        yos_val = yoshimine_sort(i,j,k,l)
                        if yos_val not in int_2el_dict:
                            int_value=two_el_integrals[i,j,k,l]
                            if abs(int_value) > int_threshold:
                                int_2el_dict[yos_val] = [int_value,[i+1,j+1,k+1,l+1]]
        # Creating string
        for k,v in int_2el_dict.items():
            two_el_integral_string+=f"{v[0]:>29.20E}{v[1][0]:>5}{v[1][1]:>5}{v[1][2]:>5}{v[1][3]:>5}\n"

    # Writing 2-electron integrals to file
    f.write(two_el_integral_string)

    # Writing 1-el integrals
    one_el_string=""
    for k,v in int_1el_dict.items():
        one_el_string+=f"{v[0]:>29.20E}{v[1][0]:>5}{v[1][1]:>5}{v[1][2]:>5}{v[1][3]:>5}\n"
    f.write(one_el_string)

    # Nuclear repulsion energy as last line
    f.write(f"{nuc_repulsion_energy:>29.20E}{0:>5}{0:>5}{0:>5}{0:>5}\n")

    f.close()

#Function to check occupations 
def check_occupations(occ):
    occ = list(occ)
    length = len(occ)
    print("\ncheck_occupations function")
    print("Checking occupations array:", occ)
    print("Length of occupations array:", length)

    #RHF
    if (occ.count(2.0) + occ.count(0.0)) == length:
        two_count = occ.count(2.0)
        num_el=two_count*2
        print("Occupation array consists only of 2.0 and 0.0 values")
        print("This is presumably a closed-shell RHF WF")
        print("Number of electrons:", num_el)
        label="RHF"
    #Fractional
    elif any(num not in [2.0,1.0,0.0] for num in occ):
        print("Occupation array contains fractional values")
        num_el=sum(occ)
        print("This is some kind of fractional-occupation WF")
        print("Could be CASSCF, WF NOs, UNO-transformation, smeared DFT etc.")
        print("Number of electrons:", num_el)
        label="FRACT"
    #ROHF
    elif occ.count(2.0) > 0 and occ.count(1.0) > 0:
        two_count = occ.count(2.0)
        one_count = occ.count(1.0)
        num_el=two_count*2+one_count
        print("Found 1.0 and 2.0 occupations")
        print("This is presumably an open-shell ROHF WF")
        print("Number of electrons:", num_el)
        label="ROHF"
    #UHF
    elif occ.count(2.0) == 0 and occ.count(1.0) > 0:
        print("Found no 2.0 occupations but some 1.0 occupations")
        one_count = occ.count(1.0)
        num_el=one_count
        print("This is presumably an open-shell UHF WF")
        print("Number of electrons:", num_el)
        label="UHF"
    else:
        print("unclear case")
        label="Unknown"

    return label