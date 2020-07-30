import numpy as np
from functions_coords import *
from functions_general import *
import settings_molcrys


# Frag class
class Frag:
    instances = []
    def __init__(self, name, formulastring, charge, mult):
        self.Name = name
        self.Formula = formulastring
        self.Atoms = molformulatolist(formulastring)
        self.Elements = uniq(self.Atoms)
        self.Numatoms = len(self.Atoms)
        self.Nuccharge = nucchargelist(self.Atoms)
        self.mass =totmasslist(self.Atoms)
        self.Charge = charge
        self.Mult = mult
        self.fraglist= []
        Frag.instances.append(self.Name)
    def change_name(self, new_name): # note that the first argument is self
        self.name = new_name # access the class attribute with the self keyword
    def add_fraglist(self, atomlist): # note that the first argument is self
        self.fraglist.append(atomlist) # access the class attribute with the self keyword



#######################
# USER INPUT          #
#######################

#Define fragments: Descriptive name, list of elements, charge, mult
mainfrag = Frag("Fecomplex","Fe2O4N3C8S2Cl4",-1,6)
counterfrag1 = Frag("NE4+counterion","NC8H20",1,1)

frags=[mainfrag,counterfrag1]

#Define global system settings (currently scale and tol keywords for connecitity)
settings_molcrys.init() #initialize

#Changing Scale and Tol:
settings_molcrys.scale=1.0
settings_molcrys.tol=0.1
#Observations. S-C bonds in adt can be 1.86 but default threshold 1.81. Changing tol to 0.1

print("Fragments defined:")
print("Mainfrag:", mainfrag.__dict__)
print("Counterfrag1:", counterfrag1.__dict__)
#Code to keep track of instances. Not sure if we want to use
#https://stackoverflow.com/questions/12101958/how-to-keep-track-of-class-instances
#foo_vars = {id(instance): instance for instance in Frag.instances}
#print(foo_vars)



#mainfrag.change_name("sdfsdfds")
#print("Mainfrag name:", mainfrag.name)
CIF_file="2Fe-adt-N4H4del-plus4extraH.cif"

#######################
# END OF USER INPUT   #
#######################

#MAIN PROGRAM

#Read CIF-file
print("Read CIF file:", CIF_file)
blankline()
cell_length,cell_angles,atomlabels,elems,asymmcoords,symmops=read_ciffile(CIF_file)
print("Cell parameters: {} {} {} {} {} {}".format(cell_length[0],cell_length[1], cell_length[2] , cell_angles[0], cell_angles[1], cell_angles[2]))
print("Number of fractional coordinates in asymmetric unit:", len(asymmcoords))

#Get  cell vectors.
cell_vectors=cellparamtovectors(cell_length,cell_angles)


#Create system coordinates for whole cell
print("Filling up unitcell using symmetry operations")
fullcellcoords,elems=fill_unitcell(cell_length,cell_angles,atomlabels,elems,asymmcoords,symmops)
print("Number of fractional coordinates in whole cell:", len(fullcellcoords))
print("Number of asymmetric units in whole cell:", len(fullcellcoords)/len(asymmcoords))
#print_coordinates(elems, np.array(fullcellcoords), title="Fractional coordinates")
#print_coords_all(fullcellcoords,elems)
blankline()

#Write fractional coordinate XTL file of fullcell coordinates (for visualization in VESTA)
write_xtl(cell_length,cell_angles,elems*4,fullcellcoords,"filledcell.xtl")

#Get orthogonal coordinates of cell
orthogcoords=fract_to_orthogonal(cell_vectors,fullcellcoords)

#print_coordinates(elems, orthogcoords, title="Orthogonal coordinates")
#Converting orthogcoords to numpy array for better performance
orthogcoords=np.asarray(orthogcoords)
#print_coords_all(orthogcoords,elems)

#Write XYZ-file with orthogonal coordinates for cell
write_xyzfile(elems,orthogcoords,"cell_orthog")


#members=get_molecule_members_fixed(orthogcoords,elems, 82,scale,tol)
#print(len(members))
#print_coords_for_atoms(orthogcoords,elems,members)
#print("----")
#members2=get_molecule_members_loop(orthogcoords,elems, 82,3,scale,tol)
#print(len(members2))
#exit()

import time

#secondsA = time.time()
#for i in range(len(elems)):
    #Using loopnumber of 3 to search through memberlist. To be checked if sufficient
#    members=get_molecule_members_fixed(orthogcoords,elems, i,scale,tol)
#secondsB = time.time()
#print("Excution time for fixed version (seconds):", secondsB-secondsA)
#print("")

#for i in range(len(elems)):
    #Using loopnumber of 3 to search through memberlist. To be checked if sufficient
 #   members=get_molecule_members_loop(orthogcoords,elems, i,4,scale,tol)
    #print("Atom no.", i)
    #print("Members :", members)
    #print("Length:", len(members))
    #if len(members) > 24:
    #    print_coords_for_atoms(orthogcoords,elems,members)
    #print("---------------")


#Using basic connectivity + information about fragments (nuccharge or mass), divide unitcell into fragments
systemlist=list(range(0,len(elems)))
systemlist_orig=systemlist
print("systemlist:", systemlist)
for i in range(len(elems)):
    members = get_molecule_members_loop(orthogcoords, elems, i,4,settings_molcrys.scale,settings_molcrys.tol)
    if len(members) == mainfrag.Numatoms:
        if members not in mainfrag.fraglist:
            mainfrag.add_fraglist(members)
        #print(members)
        for m in members:
            try:
                systemlist.remove(m)
            except ValueError:
                continue
    if len(members) == counterfrag1.Numatoms:
        if members not in counterfrag1.fraglist:
            counterfrag1.add_fraglist(members)
        for m in members:
            try:
                systemlist.remove(m)
            except ValueError:
                continue
print("")
print(mainfrag.fraglist)
print("Mainfrag has {} fraglists".format(len(mainfrag.fraglist)))
print("")
print(counterfrag1.fraglist)
print("Counterfrag1 has {} fraglists".format(len(counterfrag1.fraglist)))
exit()
print("systemlist:", systemlist)
print(len(systemlist))

exit()
#Reorder coordinates based on Hungarian algorithm
#Take from molcrys-Chemshell

# CHEMSHELL
#Create MM cluster here already or later

#Grab mainfrag, counterfrag1 etc. coordinates. Perform ORCA calculations, population analysis
#Grab atomic charges. Also calculate LJ parameters or something

#Assign pointcharges to each atom of MM cluster.

#SC-QM/MM PC loop of mainfrag

#OPT of mainfrag:   Interface to Py-Chemshell (should be easy)  or maybe DL-FIND directly
# Updating of coordinates???

#Calculate Hessian. Easy via Py-Chemshell. Maybe also easy via Dl-FIND

