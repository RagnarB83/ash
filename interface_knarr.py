#Non-intrusive interface to Knarr
#Assumes that Knarr directory exists inside Yggdrasill (for now at least)

from yggdrasill import *
import numpy as np
import sys
import os

#This makes Knarr part of python path
#Recommended way?
yggpath = os.path.dirname(yggdrasill.__file__)
sys.path.insert(0,yggpath+'/knarr')

from KNARRio.system_print import PrintHeader, PrintDivider, PrintCredit
from KNARRatom.utilities import InitializeAtomObject, InitializePathObject
from KNARRjobs.path import DoPathInterpolation
from KNARRio.io import ReadTraj
from KNARRjobs.neb import DoNEB
import KNARRatom.atom

#LOG of Knarr-code modifications
#1. Various python2 print-statements to print-functions changes
#2. Various additions of int() in order to get integer of division products (Python2/3 change)
#3. Made variable  calculator.ISCION = True . Bad idea?

#Knarr settings for path-generation, NEB and optimizer
#These will be the reasonable defaults that can be overridden by special keywords in Yggdrasill NEB object
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

#Path generator
def Knarr_pathgenerator(nebsettings,path_parameters,react,prod):
    sett = nebsettings
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

#Convert coordinates list to Knarr-type array
def coords_to_Knarr(coords):
    coords_xyz=[]
    for i in coords:
        coords_xyz.append([i[0]]);coords_xyz.append([i[1]]);coords_xyz.append([i[2]])
    coords_xyz_np=np.array(coords_xyz)
    return coords_xyz_np

#Wrapper around Yggdrasill object
class KnarrCalculator:
    def __init__(self,theory,fragment1,fragment2,runmode='serial',printlevel=None, ActiveRegion=False, actatoms=None,
                 full_fragment_reactant=None, full_fragment_product=None ):
        self.printlevel=printlevel
        self.forcecalls=0
        self.iterations=0
        self.theory=theory
        #Yggdrasill fragments for reactant and product
        #Used for element list and keep track of full system if QM/MM
        self.fragment1=fragment1
        self.fragment2=fragment2
        #Full Yggdrasill fragments for reactant and product. Inactive part of reactant will be used for all images
        self.full_fragment_reactant=full_fragment_reactant
        self.full_fragment_product=full_fragment_product
        self.runmode=runmode
        self.ISCION=False
        self.ActiveRegion=ActiveRegion
        self.actatoms=actatoms
    def Compute(self,path, list_to_compute=None):
        full_coords_images_list=[]
        #
        self.iterations+=1
        #print("self.iterations:", self.iterations)
        counter=0
        F = np.zeros(shape=(path.GetNDimIm() * path.GetNim(), 1))
        E = np.zeros(shape=(path.GetNim(), 1))
        numatoms=int(path.ndofIm/3)
        if self.runmode=='serial':
            for image_number in list_to_compute:
                image_coords_1d = path.GetCoords()[image_number * path.ndimIm : (image_number + 1) * path.ndimIm]
                image_coords=np.reshape(image_coords_1d, (numatoms, 3))
                # Request Engrad calc
                #Todo: Reduce printlevel for QM-theory here. Means that printlevel needs to be uniform accross all theories
                #Todo: Use self.printlevel so that it can adjust from outside
                blankline()

                if self.ActiveRegion == True:
                    currcoords=image_coords
                    # Defining full_coords as original coords temporarily
                    full_coords = self.full_fragment_reactant.coords
                    # Replacing act-region coordinates with coords from currcoords
                    for i, c in enumerate(full_coords):
                        if i in self.actatoms:
                            # Silly. Pop-ing first coord from currcoords until done
                            curr_c, currcoords = currcoords[0], currcoords[1:]
                            full_coords[i] = curr_c
                    full_current_image_coords = full_coords
                    #EnGrad calculation on full system
                    En_image, Grad_image_full = self.theory.run(current_coords=full_current_image_coords, elems=self.full_fragment_reactant.elems, Grad=True)

                    #Trim Full gradient down to only act-atoms gradient
                    Grad_image = np.array([Grad_image_full[i] for i in self.actatoms])
                    #List of all image-geometries (full coords)
                    full_coords_images_list.append(full_current_image_coords)

                else:
                    En_image, Grad_image = self.theory.run(current_coords=image_coords, elems=self.fragment1.elems, Grad=True)



                counter += 1
                #Energies array for all images
                E[image_number]=En_image
                #Forces array for all images
                #Todo: Check units
                F[image_number* path.ndimIm : (image_number + 1) * path.ndimIm] = -1 * np.reshape(Grad_image,(int(path.ndofIm),1))
        elif self.runmode=='parallel':
            print("not yet done")
            print("")
            exit()
        path.SetForces(F)
        path.SetEnergy(E)
        #Forcecalls
        path.AddFC(counter)
        blankline()

        #Write out full MEP path in each NEB iteration.
        if self.ActiveRegion is True:
            self.write_Full_MEP_Path(path, list_to_compute, full_coords_images_list, E)

        #print("self.ISCION:", self.ISCION)
        if self.iterations > 3 :
            if self.ISCION is True:
                print('%4ls  %4s  %9ls %5ls %6ls %9ls %9ls %9ls %6ls' % ('it', 'dS', 'Energy', 'HEI', 'RMSF', 'MaxF', 'RMSF_CI', 'MaxF_CI', 'step'))
            else:
                print(' %4ls %4s  %9ls %5ls %7ls %9ls %8ls' % ('it', 'dS', 'Energy', 'HEI', 'RMSF', 'MaxF', 'step'))
    #def AddFC(self, x=1):
    #    self.forcecalls += x
    #    return
    def write_Full_MEP_Path(self, path, list_to_compute, full_coords_images_list, E):
        #Write out MEP for full coords in each iteration. Knarr writes out Active Part.
        if self.ActiveRegion is True:
            with open("knarr_MEP_FULL.xyz", "a") as trajfile:
                #Todo: This will fail if free_end=True
                #Writing reactant image
                trajfile.write(str(self.full_fragment_reactant.numatoms) + "\n")
                trajfile.write("Image 0. Energy: {} \n".format(path.GetEnergy()[0]))
                for el, cor in zip(self.full_fragment_reactant.elems, self.full_fragment_reactant):
                    trajfile.write(el + "  " + str(cor[0]) + " " + str(cor[1]) + " " + str(cor[2]) + "\n")
                #All active images in this NEB iteration:
                for imageid,fc in list_to_compute,full_coords_images_list:
                    trajfile.write(str(self.full_fragment_reactant.numatoms) + "\n")
                    trajfile.write("Image {}. Energy: {} \n".format(imageid, E[imageid]))
                    for el, cor in zip(self.full_fragment_reactant.elems, fc):
                        trajfile.write(el + "  " + str(cor[0]) + " " + str(cor[1]) + " " + str(cor[2]) + "\n")
                #Writing product image
                trajfile.write(str(self.full_fragment_product.numatoms) + "\n")
                trajfile.write("Image X Energy: {} \n".format(path.GetEnergy()[-1]))
                for el, cor in zip(self.full_fragment_product.elems, self.full_fragment_product):
                    trajfile.write(el + "  " + str(cor[0]) + " " + str(cor[1]) + " " + str(cor[2]) + "\n")


#Yggdrasill NEB function. Calls Knarr
def NEB(reactant=None, product=None, theory=None, images=None, interpolation=None, CI=None, free_end=None,
        conv_type=None, tol_scale=None, tol_max_fci=None, tol_rms_fci=None, tol_max_f=None, tol_rms_f=None,
        tol_turn_on_ci=None, ActiveRegion=False, actatoms=None, runmode='serial'):

    if reactant==None or product==None or theory==None:
        print("You need to provide reactant and product fragment and a theory to NEB")
        exit()

    print("Launching Knarr program")
    blankline()
    PrintDivider()
    PrintDivider()
    PrintHeader()
    PrintCredit()
    PrintDivider()
    PrintDivider()
    numatoms = reactant.numatoms

    #Override some default settings if requested
    #Default is; NEB-CI, IDPP interpolation, 6 images
    if images is not None:
        path_parameters["NIMAGES"]=images
    if interpolation is not None:
        path_parameters["INTERPOLATION"]=interpolation
    if CI is not None:
        if CI is False:
            neb_settings["CLIMBING"]=False
    if free_end is not None:
        neb_settings["FREE_END"] = True
    if conv_type is not None:
        neb_settings["CONV_TYPE"] = conv_type
    if tol_scale is not None:
        neb_settings["TOL_SCALE"] = tol_scale
    if tol_max_fci is not None:
        neb_settings["TOL_MAX_FCI"] = tol_max_fci
    if tol_rms_fci is not None:
        neb_settings["TOL_RMS_FCI"] = tol_rms_fci
    if tol_max_f is not None:
        neb_settings["TOL_MAX_F"] = tol_max_f
    if tol_rms_f is not None:
        neb_settings["TOL_RMS_F"] = tol_rms_f
    if tol_turn_on_ci is not None:
        neb_settings["TOL_TURN_ON_CI"] = tol_turn_on_ci

    print("Active Knarr settings:")
    blankline()

    print("Interpolation path parameters:\n", path_parameters)
    blankline()
    print("NEB parameters:\n", neb_settings)
    blankline()
    print("Optimizer parameters:\n", optimizer)


    #Zero-valued constraints list. We probably won't use constraints for now
    constr = np.zeros(shape=(numatoms * 3, 1))

    #ActiveRegion feature
    if ActiveRegion==True:
        print("Active Region option Active. Passing only active-region coordinates to Knarr.")
        print("Number of active atoms:", len(actatoms))
        R_actcoords, R_actelems = reactant.get_coords_for_atoms(actatoms)
        P_actcoords, P_actelems = product.get_coords_for_atoms(actatoms)
        new_reactant = Fragment(coords=R_actcoords, elems=R_actelems)
        new_product = Fragment(coords=P_actcoords, elems=P_actelems)
        #Create Knarr calculator from Yggdrasill theory.
        calculator = KnarrCalculator(theory, fragment1=new_reactant, fragment2=new_product, runmode=runmode
                                     ActiveRegion=True, full_fragment_reactant=reactant, full_fragment_product=product )

        # Symbols list for Knarr
        Knarr_symbols = [y for y in new_reactant.elems for i in range(3)]

        # Create KNARR Atom objects. Used in path generation
        react = KNARRatom.atom.Atom(coords=coords_to_Knarr(new_reactant.coords), symbols=Knarr_symbols, ndim=numatoms * 3,
                                    ndof=numatoms * 3, constraints=constr, pbc=False)
        prod = KNARRatom.atom.Atom(coords=coords_to_Knarr(new_product.coords), symbols=Knarr_symbols, ndim=numatoms * 3,
                                   ndof=numatoms * 3, constraints=constr, pbc=False)
    else:
        #Create Knarr calculator from Yggdrasill theory
        calculator = KnarrCalculator(theory, fragment1=reactant, fragment2=product, ActiveRegion=False, runmode=runmode)

        # Symbols list for Knarr
        Knarr_symbols = [y for y in reactant.elems for i in range(3)]

        # Create KNARR Atom objects. Used in path generation
        react = KNARRatom.atom.Atom(coords=coords_to_Knarr(reactant.coords), symbols=Knarr_symbols, ndim=numatoms * 3,
                                    ndof=numatoms * 3, constraints=constr, pbc=False)
        prod = KNARRatom.atom.Atom(coords=coords_to_Knarr(product.coords), symbols=Knarr_symbols, ndim=numatoms * 3,
                                   ndof=numatoms * 3, constraints=constr, pbc=False)


    # Generate path via Knarr_pathgenerator
    Knarr_pathgenerator(neb_settings, path_parameters, react, prod)

    #Reading initial path from XYZ file. Hardcoded as knarr_path.xyz
    rp, ndim, nim, symb = ReadTraj("knarr_path.xyz")
    path = InitializePathObject(nim, react)
    path.SetCoords(rp)

    #Now starting NEB from path object, using neb_settings and optimizer settings
    DoNEB(path, calculator, neb_settings, optimizer)

    #TODO: Get NEB-trajectory for full system
    #Option 1: Read knarr_MEP.xyz, add full part to everything and write out knarr_MEP_FULL.xyz
    #Option 2: Write out knarr_MEP_FULL.xyz in each NEB iteration inside Compute??

    #TODO: Get Yggdrasill fragment with saddlepoint-geometry. Full system

    print('KNARR successfully terminated')
    blankline()
    print("Please consider citing the following paper if you found the NEB module (from Knarr) useful: To be added")