from yggdrasill import *

import numpy as np
import sys
sys.path.insert(0,'./knarr')

from KNARRatom.utilities import InitializeAtomObject, InitializePathObject
from KNARRjobs.path import DoPathInterpolation
from KNARRio.io import ReadTraj
from KNARRjobs.neb import DoNEB
import KNARRatom.atom

#New interface to Knarr so that we don't have to change
#Assumes Knarr directory inside Yggdrasill for now

def Knarr_pathgenerator(nebsettings,path_parameters,react,prod):
    #sett = KNARRsettings.settings_types["PATH"]
    sett = nebsettings
    #pathgen = FetchSettings(sett, settings_list, settings_line_ind, full_input)
    #path_parameters
    type_of_method_string = path_parameters["METHOD"].upper()
    if type_of_method_string == "SINGLE":
        prod_is_needed = False
    elif type_of_method_string == "DOUBLE":
        prod_is_needed = True
    else:
        raise TypeError("Either choose single or double ended path generation")

    nim = path_parameters["NIMAGES"]
    path = InitializePathObject(nim, react)

    if prod_is_needed:
        # Get product
        #prod = InitializeAtomObject(name="product", input_config=main_control["CONFIG_FILE2"],
        #                            pbc=main_control["PBC"])

        # Check product
        if react.GetNDim() != prod.GetNDim():
            raise RuntimeError("Reactant / product do not match")
        if react.GetSymbols() != prod.GetSymbols():
            raise RuntimeError("Reactant / product do not match")
        path.SetConfig2(prod.GetCoords())

        # check insertion
        insertion = path_parameters["INSERT_CONFIG"]
        if insertion is not None:
            insertion = InitializeAtomObject(name="insertion", input_config=path_parameters["INSERT_CONFIG"],
                                             pbc=main_control["PBC"])
            if insertion.GetSymbols() != react.GetSymbols():
                raise ValueError("Insertion does not match reactant / product")
            path.SetInsertionConfig(insertion.GetCoords())
    else:
        prod = None

    DoPathInterpolation(path, path_parameters)

#Define manual dicts here

path_parameters = {"METHOD": "DOUBLE", "INTERPOLATION": "IDPP", "NIMAGES": 6,
              "INSERT_CONFIG": None, "IDPP_MAX_ITER": 2000,
              "IDPP_SPRINGCONST": 10.0, "IDPP_TIME_STEP": 0.01,
              "IDPP_MAX_MOVE": 0.1, "IDPP_MAX_F": 0.01, "IDPP_RMS_F": 0.005}

neb_settings = {"PATH": "neb.xyz",
              "CLIMBING": True,
              "TANGENT": "IMPROVED",
              "SPRINGTYPE": "DISTANCE",
              "PERP_SPRINGTYPE": "",
              "ENERGY_WEIGHTED": True, "SPRINGCONST": 1.0, "SPRINGCONST2": 10.0,
              "MIN_RMSD": True, "REMOVE_EXTERNAL_FORCE": True,
              "FREE_END": False, "FREE_END_TYPE": 'PERP', "FREE_END_ENERGY": 0.0,
              "FREE_END_ENERGY2": 0.0, "FREE_END_KAPPA": 0.0,
              "CONV_TYPE": "ALL", "TOL_SCALE": 10, "TOL_MAX_FCI": 0.026, "TOL_RMS_FCI": 0.013,
              "TOL_MAX_F": 0.026, "TOL_RMS_F": 0.013, "TOL_TURN_ON_CI": 1.0,
              "ZOOM": False,
              "TOL_TURN_ON_ZOOM": 0.5,
              "REPARAM": 0, "TOL_REPARAM": 0.0,
              "INTERPOLATION_TYPE": "LINEAR",
              "AUTO_ZOOM": True, "ZOOM_OFFSET": 1, "ZOOM_ALPHA": 0.5,
              "RESTART_OPT_ON_CI": True,
              "LBFGS_REPARAM_ON_RESTART": False
              }

optimizer = {"OPTIM_METHOD": "LBFGS", "MAX_ITER": 1000, "TOL_MAX_FORCE": 0.01,
             "TOL_RMS_FORCE": 0.005, "TIME_STEP": 0.01, "MAX_MOVE": 0.1, "RESTART_ON_SCALING": True,
             "LBFGS_MEMORY": 20,
             "LBFGS_DAMP": 1.0,
             "FD_STEP": 0.001,
             "LINESEARCH": None}

#TEST


#react = InitializeAtomObject(name="Reactant", input_config=input_config,
#                             pbc=main_control["PBC"], twodee=main_control["TWO_DEE_SYSTEM"])
#import knarr



react="""
  C   -1.74466226888029     -0.43764022970111      0.00340875852252
  Cl  0.08751046227979     -0.43663776654125     -0.00572350984613
  H   -2.08316380850831     -1.23082242056354      0.67167470171506
  H   -2.08839756963152     -0.61873908546936     -1.01572923119891
  H   -2.08452478644406      0.53630031360180      0.35837366325833
  F   -4.40536202881561     -0.44759081132654      0.07190561754913
"""
prod="""
  C   -2.70156642333543     -0.43658441857463      0.03323212020336
  Cl  0.24983987904870     -0.43390510382220     -0.01987082558543
  H   -2.32511980738684     -1.22504253630063      0.69082755791150
  H   -2.35113324790501     -0.61615683301537     -0.98679738541797
  H   -2.33164382536765      0.53239758053095      0.37898769832545
  F   -4.06258048905377     -0.43793231981811      0.05403279356309
"""
Reactant=Fragment(coordsstring=react)
Product=Fragment(coordsstring=prod)

#elems=['H','F']
#coords=[[0.0,0.0,0.0],[0.0,0.0,1.0]]
#lems2=['H','F']
#coords2=[[0.1,0.2,0.3],[1.0,1.0,2.0]]
#numatoms=2

#Symbols list for Knarr
Knarr_symbols=[y for y in Reactant.elems for i in range(3)]

#Coords list for Knar
def coords_to_Knarr(coords):
    coords_xyz=[]
    for i in coords:
        coords_xyz.append([i[0]]);coords_xyz.append([i[1]]);coords_xyz.append([i[2]])
    coords_xyz_np=np.array(coords_xyz)
    #print(coords_xyz_np)
    return coords_xyz_np

numatoms=Reactant.numatoms
constr = np.zeros(shape=(numatoms*3, 1))

#Create KNARR Atom objects
react = KNARRatom.atom.Atom(coords=coords_to_Knarr(Reactant.coords), symbols=Knarr_symbols, ndim=numatoms*3, ndof=numatoms*3, constraints=constr, pbc=False)
prod = KNARRatom.atom.Atom(coords=coords_to_Knarr(Product.coords), symbols=Knarr_symbols, ndim=numatoms*3, ndof=numatoms*3, constraints=constr, pbc=False)
print(react)
print(react.__dict__)
print(react.coords)
print(react.GetCoords())

#Generate path via
Knarr_pathgenerator(neb_settings,path_parameters,react,prod)
rp, ndim, nim, symb = ReadTraj("knarr_path.xyz")
path = InitializePathObject(nim, react)
path.SetCoords(rp)

#Wrapper around Yggdrasill object
class KnarrCalculator:
    def __init__(self,theory,fragment1,fragment2,runmode='serial'):
        self.forcecalls=0
        self.theory=theory
        #Yggdrasill fragments for reactant and product
        #Used for element list and keep track of full system if QM/MM
        self.fragment1=fragment1
        self.fragment2=fragment2
        self.runmode=runmode
    #def self.Compute(path, list_to_compute=range(1, path.GetNim() - 1)):
    def Compute(self,path, list_to_compute=[]):
        #
        counter=0
        F = np.zeros(shape=(path.GetNDimIm() * path.GetNim(), 1))
        E = np.zeros(shape=(path.GetNim(), 1))
        print("F:", F)
        print("E:", E)
        numatoms=int(path.ndofIm/3)
        print("Compute called")
        print("path:", path)
        print("list_to_compute:", list_to_compute)

        #for i in range(path.GetNim()):
        if self.runmode=='serial':
            for image_number in list_to_compute:
                print("image_number:", image_number)
                image_coords_1d = path.GetCoords()[image_number * path.ndimIm : (image_number + 1) * path.ndimIm]
                image_coords=np.reshape(image_coords_1d, (numatoms, 3))
                print(image_coords)


                # Request Engrad calc
                En_image, Grad_image = self.theory.run(current_coords=image_coords, elems=self.fragment1.elems, Grad=True)
                counter += 1
                print("En_image: ", En_image)
                #Energies array for all images
                E[image_number]=En_image
                #Forces array for all images
                #Todo: Check units
                F[image_number] = -1 * np.reshape(Grad_image,(path.ndofIm,1))
                print(F[image_number])
        elif self.runmode=='parallel':
            print("not yet done")
            exit()

        #
        print("F: ", F)
        print("E:", E)

        #Note: F and E is list/arrays of Forces and and Energy for all images in list provided
        path.SetForces(F)
        path.SetEnergy(E)

        #FC=Forcecalls
        path.AddFC(counter)

    def AddFC(self, x=1):
        self.forcecalls += x
        return

#Temp theory
xTBcalc = xTBTheory(charge=0, mult=1, xtbmethod='GFN1', runmode='inputfile')


calculator=KnarrCalculator(xTBcalc,fragment1=Reactant,fragment2=Product)

print("path", path)
print("path", path.__dict__)
print(path.energy)
print(path.GetEnergy())
print("-------")
DoNEB(path, calculator, neb_settings, optimizer)