# ASH - A GENERAL COMPCHEM AND QM/MM ENVIRONMENT
from constants import *
from elstructure_functions import *
import os
import glob
from functions_solv import *
import functions_coords
from functions_coords import *
from functions_ORCA import *
from functions_Psi4 import *
from functions_general import *
from functions_freq import *
import settings_ash
from functions_MM import *
from functions_optimization import *
from interface_geometric import geomeTRICOptimizer
import shutil
import subprocess as sp
import elstructure_functions
from workflows import *
from benchmarking import *

debugflag=False

import sys
import inspect


#Julia dependency
#Current behaviour: We try to import, if not possible then we continue
load_julia = True
if load_julia is True:
    try:
        print("Import PyJulia interface")
        from julia.api import Julia
        from julia import Main
        #Hungarian package needs to be installed
        try:
            from julia import Hungarian
        except:
            print("Problem loading Julia packages: Hungarian")
        ashpath = os.path.dirname(ash.__file__)
        #Various Julia functions
        print("Loading Julia functions")
        Main.include(ashpath + "/functions_julia.jl")
    except:
        print("Problem importing Pyjulia")
        print("Make sure Julia is installed, PyJulia within Python, Pycall within Julia, Julia packages have been installed and you are using python-jl")
        print("Python routines will be used instead when possible")



#Debug print. Behaves like print but reads global debug var first
def printdebug(string,var=''):
    global debugflag
    if debugflag is True:
        print(BC.OKRED,string,var,BC.END)


def print_ash_header():
    programversion = 0.1

    #Getting commit version number from file VERSION (updated by yggpull) inside module dir
    try:
        with open(os.path.dirname(ash.__file__)+"/VERSION") as f:
            git_commit_number = int(f.readline())
    except:
        git_commit_number="Unknown"

    #http://asciiflow.com
    #https://textik.com/#91d6380098664f89
    #https://www.gridsagegames.com/rexpaint/

    ascii_banner="""
   ▄████████    ▄████████    ▄█    █▄    
  ███    ███   ███    ███   ███    ███   
  ███    ███   ███    █▀    ███    ███   
  ███    ███   ███         ▄███▄▄▄▄███▄▄ 
▀███████████ ▀███████████ ▀▀███▀▀▀▀███▀  
  ███    ███          ███   ███    ███   
  ███    ███    ▄█    ███   ███    ███   
  ███    █▀   ▄████████▀    ███    █▀    
                                         
    """

    ascii_banner_center = """
                            ▄████████    ▄████████    ▄█    █▄    
                           ███    ███   ███    ███   ███    ███   
                           ███    ███   ███    █▀    ███    ███   
                           ███    ███   ███         ▄███▄▄▄▄███▄▄ 
                         ▀███████████ ▀███████████ ▀▀███▀▀▀▀███▀  
                           ███    ███          ███   ███    ███   
                           ███    ███    ▄█    ███   ███    ███   
                           ███    █▀   ▄████████▀    ███    █▀    
                                         
    """
    ascii_tree = """                                                                               
                  [      ▓          ▒     ╒                  ▓                  
                  ╙█     ▓▓         ▓  -µ ╙▄        ¬       ▓▀             
               ▌∩   ▀█▓▌▄▄▓▓▓▄,  ▀▄▓▌    ▓  ▀█      ▌   █ ╓▓Γ ▄▓▀¬              
              ╓Γ╘     % ╙▀▀▀▀▀▓▓▄  ▓▓   █▓,╒  ▀▓   ▓▌ Æ▀▓▓▓ ▄▓▓¬ .     ▌        
              ▓  ^█    ▌  ▄    ▀▓▓ ▓▓m ▐▓ ▓▓  ▐▌▓▌▓▀   Å▓▓▓▓▀     ▐   ▐▓    ▄   
            ╘▓▌ \  ▀▓▄▐▓ ▐▓     ▀▓▓▓▓m ▓▌▄▓▓█ ▓┘ ▓▓ ▄█▀▀▓▓▄   ▐▓  ▐▄  j▓   ▓    
          █  ╙▓  ╙▀▄▄▀▓▓µ ▓   ▄▄▀▀▓▓▓ ▓▓▓Σ▄▓▓▓Γ  ▓▓▓▀ █▓▀^▓  ▄▓   ▓   ▐▓  █     
        ▌  ▀▀█,▓▓  ▄╙▀▓▓▓▄▓▌▄▓▌   ▓▓  ▓▓▓▀▓▓▓▓▓  ▓▓▓▓▓▀ ╞ ▓▄▓▀ ╓─ ▓▄ ,▓█▀▀ .▀   
        ▐▄    ▀▓▓  ▓   ▓▓▓▓▓Γ╙▓▄  ▓▓▓▓▓▌▓▓▀▓Γ ▓▓▄▓▓▓▀  ▄ ▓▓▀  ▀   ▐▓▄▓▀   ▄▓    
   ╓┴▀███▓▓▓▄   ▓▓▓    █▓▓▓ ║ ¬▓▓▓▓▓▓▓▓▀  ▓▌  ▐▓▓▓▓  ▄▓▓▓▀  ▄▓██▀▀  ▓▓▄▓█▀Γ     
           ╙▀▓▓▓▓▓▓     ▓▓▓  █▄  ▀▓▓▓▓▓  j▓▄  ▓▓▓▓▓▓▓▓▀   █▓▓▀   ▄▓▓▓▀▐         
          ,▄µ  ▀▓▓▓▓▓▄  ▓▓▓   ▓▓▄▓▓▀▓▓▓▓▓▓▓  ▄▓▓▓▓▓▓▓▓▓▓▓▓▀  ▄▓▓▀▀¬             
    ▄▓█▀▀▀▀▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▄  ²▓▓▓    ▀▓▓▓▓ ▄▓▓▓▓▓▓▀▀▀¬   ▄▓▓▀¬ ▄Φ` ▄`       ▓  
  Σ▀Σ  ▄██▀▀Σ     ¬▀▓▓▓▓▓▓▓▓  j▓▓▌     ▐▓▓▓▓▓▓▓▓▀       .▓▓▀  ▄▓  ▄▓Γ   ▄▄▄▀▀   
     ╒▓               ▀▓▓▓▓▓▄,▓▓▓▓▓▓▓▓▄ ▓▓▓▓▓▓▓▓▓▄,     ▓▓▓▄▄▓▓▓█▀▀  ▓▓▀¬       
     ╙⌐       ▄         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▐▀▓▓▓▓▓▓▓▓▓▓▓▓▀▀▓▓▓▓▓▓▓▀          
          ▄█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▀▀¬  ▐▀▓▓▓▓▓▓▓▓▓                      ▐▀▀▄         
       ▄▀▀▐▓▓▀▐  ¬*▐▀▀▀▀             ▓▓▓▓▓▓▓                                 
     ╒Γ   ▐▓                         ▓▓▓▓▓▓▓                                    
     Γ    ╫                          ▓▓▓▓▓▓▓                                    
                                     ▓▓▓▓▓▓▓                                    
                                    ▓▓▓▓▓▓▓▓▄                                   
                                   ▓▓▓▓▓▓▓▓▓▓▄                                  
                                ,▓▓▓▓▓▓▓▓▓▓▓▓▓▄                                 
                          ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▄▄▌▌▄▄▄                       
                ,▄æ∞▄▄▄█▓▀▀▐¬▐▀▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▄ ▐▀▓▓▓▓▓▀▀▀▀Σ╙           
         ╙Φ¥¥▀▀▀▀   ╒▓   ▄▓█▀▀▓▓▓▓▓▓▓▓▀  ▓▓▓▓▓▓ ▀▓▓▄▄▐▀▀▀▀▀▐▐▀██                
                    ▓▌▄▓▓▀  ▓▓▓▓▓▀▀▓▓     ▓▓ ▓▓▓   ▐▀▀█▓▓▌¥4▀▀▀▀▓▓▄             
               `ΓΦ▀▓▀▐   ▄▄▓▓ ▓▓  ▐▌▓     ▓▓▓ ▓▀▀▓▄     ,▓▓      ▀▀▀▀▀          
                 ▄▀,  ▄█▀▀ ▓¬ ▓τ  ▓ ▐▓   ▓  ▓  ¥ ▓        ▀▌                    
               ^    ▓▀    █  ▓▀ ▄█¬ ▐▓  ▌   Γ   ▐▓τ*       ▀█▄                  
                   ▓   /Γ ƒ▀ⁿ  █    ▓   Γ      ╓▀▐⌐          ¬▀²                
                   ▓      \    ▀    ╙µ  ⌡                                       
                                     └                                                                                                                      
    """
    print(BC.OKGREEN,"----------------------------------------------------------------------------------",BC.END)
    print(BC.OKGREEN,"----------------------------------------------------------------------------------",BC.END)
    #print(BC.OKBLUE,ascii_banner3,BC.END)
    #print(BC.OKBLUE,ascii_banner2,BC.END)
    print(BC.OKGREEN,ascii_banner_center,BC.END)
    print(BC.OKGREEN,ascii_tree,BC.END)
    print(BC.WARNING,BC.BOLD,"ASH version", programversion,BC.END)
    print(BC.WARNING, "Git commit version: ", git_commit_number, BC.END)
    print(BC.WARNING,"A COMPCHEM AND QM/MM ENVIRONMENT", BC.END)
    print(BC.OKGREEN,"----------------------------------------------------------------------------------",BC.END)
    print(BC.OKGREEN,"----------------------------------------------------------------------------------",BC.END)

#Functions to run each displacement in parallel NumFreq run. Have to be here.
#Simple QM object
def displacement_QMrun(arglist):
    #print("arglist:", arglist)
    geo = arglist[0]
    elems = arglist[1]
    #Numcores can be used. We can launch ORCA-OpenMPI in parallel it seems. Only makes sense if we have may more cores available than displacements
    numcores = arglist[2]
    theory = arglist[3]
    label = arglist[4]
    dispdir=label.replace(' ','')
    os.mkdir(dispdir)
    os.chdir(dispdir)
    #Todo: Copy previous GBW file in here if ORCA, xtbrestart if xtb, etc.
    print("Running displacement: {}".format(label))
    energy, gradient = theory.run(current_coords=geo, elems=elems, Grad=True, nprocs=numcores)
    print("Energy: ", energy)
    os.chdir('..')
    #Delete dir?
    #os.remove(dispdir)
    return [label, energy, gradient]


#Function to run each displacement in parallel NumFreq run
#Version where geo is read from file to avoid large memory pickle inside pool.map
def displacement_QMMMrun(arglist):
    #global QMMM_xtb
    print("arglist:", arglist)

    #print("locals", locals())
    #print("globals", globals())
    print("-----------")
    #import gc
    #print(gc.get_objects())

    filelabel=arglist[0]
    #elems = arglist[1]
    #Numcores can be used. We can launch ORCA-OpenMPI in parallel it seems. Only makes sense if we have may more cores available than displacements
    numcores = arglist[1]
    label = arglist[2]
    fragment= arglist[3]
    qm_theory = arglist[4]
    mm_theory = arglist[5]
    actatoms = arglist[6]
    qmatoms = arglist[7]
    embedding = arglist[8]
    charges = arglist[9]
    printlevel = arglist[10]
    frozenatoms = arglist[11]

    dispdir=label.replace(' ','')
    os.mkdir(dispdir)
    os.chdir(dispdir)
    shutil.move('../'+filelabel+'.xyz','./'+filelabel+'.xyz')
    # Read XYZ-file from file
    elems,coords = read_xyzfile(filelabel+'.xyz')

    #Todo: Copy previous GBW file in here if ORCA, xtbrestart if xtb, etc.
    print("Running displacement: {}".format(label))

    #If QMMMTheory init keywords are changed this needs to be updated
    qmmmobject = QMMMTheory(fragment=fragment, qm_theory=qm_theory, mm_theory=mm_theory, actatoms=actatoms,
                          qmatoms=qmatoms, embedding=embedding, charges=charges, printlevel=printlevel,
                            nprocs=numcores, frozenatoms=frozenatoms)

    energy, gradient = qmmmobject.run(current_coords=coords, elems=elems, Grad=True, nprocs=numcores)
    print("Energy: ", energy)
    os.chdir('..')
    #Delete dir?
    #os.remove(dispdir)
    return [label, energy, gradient]

#Single-point energy function
# *args is used by parallel version

def Singlepoint(fragment=None, theory=None, Grad=False):
    print("")
    '''
    The Singlepoint function carries out a single-point energy calculation
    :param fragment:
    :type fragment: ASH object of class Fragment
    :param theory:
    :type theory: ASH theory object
    :param Grad: whether to do Gradient or not.
    :type Grad: Boolean.
    '''
    if fragment is None or theory is None:
        print(BC.FAIL,"Singlepoint requires a fragment and a theory object",BC.END)
        exit(1)
    coords=fragment.coords
    elems=fragment.elems
    # Run a single-point energy job
    if Grad ==True:
        print(BC.WARNING,"Doing single-point Energy+Gradient job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
        # An Energy+Gradient calculation where we change the number of cores to 12
        energy,gradient= theory.run(current_coords=coords, elems=elems, Grad=True)
        print("Energy: ", energy)
        return energy,gradient
    else:
        print(BC.WARNING,"Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)

        energy = theory.run(current_coords=coords, elems=elems)
        print("Energy: ", energy)

        #Now adding total energy to fragment
        fragment.energy=energy

        return energy


#Stripped down version of Singlepoint function for Singlepoint_parallel
def Single_par(list):
    theory=list[0]
    fragment=list[1]
    #Making label flexible
    #label=''.join([str(i) for i in list[2]])
    label=list[2]
    #Creating separate inputfilename using label
    theory.inputfilename=''.join([str(i) for i in list[2]])
    
    if label is None:
        print("No label provided to fragment or theory objects. This is required to distinguish between calculations ")
        print("Exiting...")
        exit(1)


    coords = fragment.coords
    elems = fragment.elems
    print(BC.WARNING,"Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
    print("\n\nProcess ID {} is running calculation with label: {} \n\n".format(mp.current_process(),label))

    energy = theory.run(current_coords=coords, elems=elems)
    print("Energy: ", energy)
    # Now adding total energy to fragment
    fragment.energy = energy
    return (label,energy)

#PARALLEL Single-point energy function
#will run over fragments, over theories or both
def Singlepoint_parallel(fragments=None, theories=None, numcores=None):
    print("")
    '''
    The Singlepoint_parallel function carries out multiple single-point calculations in a parallel fashion
    :param fragments:
    :type list: list of ASH objects of class Fragment
    :param theories:
    :type list: list of ASH theory objects
    :param Grad: whether to do Gradient or not.
    :type Grad: Boolean.
    '''
    if fragments is None or theories is None or numcores is None:
        print(BC.FAIL,"Singlepoint_parallel requires a fragment and a theory object and a numcores values",BC.END)
        exit(1)

    blankline()
    print("Singlepoint_parallel function")
    print("Number of CPU cores available: ", numcores)
    print("Number of fragments:", len(fragments))
    print("Number of theories:", len(theories))
    print("Running single-point calculations in parallel")

    pool = mp.Pool(numcores)
    # Singlepoint(fragment=None, theory=None, Grad=False)
    #Case: 1 theory, multiple fragments
    if len(theories) == 1:
        print("Case: Multiple fragments but one theory")
        theory = theories[0]
        results = pool.map(Single_par, [[theory,fragment, fragment.label] for fragment in fragments])
        pool.close()
        print("Calculations are done")
    # Case: Multiple theories, 1 fragment
    elif len(fragments) == 1:
        print("Case: Multiple theories but one fragment")
        fragment = fragments[0]
        results = pool.map(Single_par, [[theory,fragment, theory.label] for theory in theories])
        pool.close()
        print("Calculations are done")
    else:
        print("Multiple theories and multiple fragments provided.")
        print("This is not supported. Exiting...")
        #fragment = fragments[0]
        #results = pool.map(Single_par, [[theory,fragment,label] for theory,fragment in zip(theories,fragments)])
        #pool.close()


    #Convert list of tuples into dict
    energy_dict = {result[0]: result[1] for result in results}
    print("energy_dict:", energy_dict)

    return energy_dict


#Analytical frequencies function
#Only works for ORCAtheory at the moment
def AnFreq(fragment=None, theory=None, numcores=1, temp=298.15, pressure=1.0):
    print(BC.WARNING, BC.BOLD, "------------ANALYTICAL FREQUENCIES-------------", BC.END)
    if theory.__class__.__name__ == "ORCATheory":
        print("Requesting analytical Hessian calculation from ORCATheory")
        print("")
        #Do single-point ORCA Anfreq job
        energy = theory.run(current_coords=fragment.coords, elems=fragment.elems, Hessian=True, nprocs=numcores)
        #Grab Hessian
        #Hessian = Hessgrab(theory.inputfilename+".hess")
        #TODO: diagonalize it ourselves. Need to finish projection
        
        # For now, we grab frequencies from ORCA Hessian file
        frequencies = ORCAfrequenciesgrab(theory.inputfilename+".hess")
        
        hessatoms=list(range(0,fragment.numatoms))
        Thermochemistry = thermochemcalc(frequencies,hessatoms, fragment, theory.mult, temp=temp,pressure=pressure)
        
        print(BC.WARNING, BC.BOLD, "------------ANALYTICAL FREQUENCIES END-------------", BC.END)
        return Thermochemistry
        
    else:
        print("Analytical frequencies not available for theory. Exiting.")
        exit()

#Numerical frequencies function
def NumFreq(fragment=None, theory=None, npoint=1, displacement=0.0005, hessatoms=None, numcores=1, runmode='serial', temp=298.15, pressure=1.0):
    
    print(BC.WARNING, BC.BOLD, "------------NUMERICAL FREQUENCIES-------------", BC.END)
    shutil.rmtree('Numfreq_dir', ignore_errors=True)
    os.mkdir('Numfreq_dir')
    os.chdir('Numfreq_dir')
    print("Creating separate directory for displacement calculations: Numfreq_dir ")
    if fragment is None or theory is None:
        print("NumFreq requires a fragment and a theory object")

    coords=fragment.coords
    elems=fragment.elems
    numatoms=len(elems)
    #Hessatoms list is allatoms (if not defined), otherwise the atoms provided and thus a partial Hessian is calculated.
    allatoms=list(range(0,numatoms))
    if hessatoms is None:
        hessatoms=allatoms

    #Making sure hessatoms list is sorted
    hessatoms.sort()

    displacement_bohr = displacement * constants.ang2bohr

    print("Starting Numerical Frequencies job for fragment")
    print("System size:", numatoms)
    print("Hessian atoms:", hessatoms)
    if hessatoms != allatoms:
        print("This is a partial Hessian.")
    if npoint ==  1:
        print("One-point formula used (forward difference)")
    elif npoint == 2:
        print("Two-point formula used (central difference)")
    else:
        print("Unknown npoint option. npoint should be set to 1 (one-point) or 2 (two-point formula).")
        exit()
    if runmode=="serial":
        print("Numfreq running in serial mode")
    elif runmode=="parallel":
        print("Numfreq running in parallel mode")
    blankline()
    print("Displacement: {:5.4f} Å ({:5.4f} Bohr)".format(displacement,displacement_bohr))
    blankline()
    print("Starting geometry:")
    #Converting to numpy array
    #TODO: get rid list->np-array conversion
    current_coords_array=np.array(coords)

    print("Printing hessatoms geometry...")
    print_coords_for_atoms(coords,elems,hessatoms)
    blankline()

    #Looping over each atom and each coordinate to create displaced geometries
    #Only displacing atom if in hessatoms list. i.e. possible partial Hessian
    list_of_displaced_geos=[]
    list_of_displacements=[]
    for atom_index in range(0,len(current_coords_array)):
        if atom_index in hessatoms:
            for coord_index in range(0,3):
                val=current_coords_array[atom_index,coord_index]
                #Displacing in + direction
                current_coords_array[atom_index,coord_index]=val+displacement
                y = current_coords_array.copy()
                list_of_displaced_geos.append(y)
                list_of_displacements.append([atom_index, coord_index, '+'])
                if npoint == 2:
                    #Displacing  - direction
                    current_coords_array[atom_index,coord_index]=val-displacement
                    y = current_coords_array.copy()
                    list_of_displaced_geos.append(y)
                    list_of_displacements.append([atom_index, coord_index, '-'])
                #Displacing back
                current_coords_array[atom_index, coord_index] = val

    # Original geo added here if onepoint
    if npoint == 1:
        list_of_displaced_geos.append(current_coords_array)
        list_of_displacements.append('Originalgeo')

    print("List of displacements:", list_of_displacements)

    #Creating displacement labels
    list_of_labels=[]
    for disp in list_of_displacements:
        if disp == 'Originalgeo':
            calclabel = 'Originalgeo'
        else:
            atom_disp = disp[0]
            if disp[1] == 0:
                crd = 'x'
            elif disp[1] == 1:
                crd = 'y'
            elif disp[1] == 2:
                crd = 'z'
            drection = disp[2]
            # displacement_jobname='Numfreq-Disp-'+'Atom'+str(atom_disp)+crd+drection
            #print("Displacing Atom: {} Coordinate: {} Direction: {}".format(atom_disp, crd, drection))
            calclabel = 'Atom: {} Coord: {} Direction: {}'.format(atom_disp, crd, drection)
        list_of_labels.append(calclabel)

    assert len(list_of_labels) == len(list_of_displaced_geos), "something is wrong"

    #Write all geometries to disk as XYZ-files
    list_of_filelabels=[]
    for label, dispgeo in zip(list_of_labels,list_of_displaced_geos):
        filelabel=label.replace(' ','').replace(':','')
        list_of_filelabels.append(filelabel)
        write_xyzfile(elems=elems, coords=dispgeo,name=filelabel)

    #RUNNING displacements
    displacement_grad_dictionary = {}
    if runmode == 'serial':
        #Looping over geometries and running.
        #   key: AtomNCoordPDirectionm   where N=atomnumber, P=x,y,z and direction m: + or -
        #   value: gradient
        for label, geo in zip(list_of_labels,list_of_displaced_geos):
            if label == 'Originalgeo':
                calclabel = 'Originalgeo'
                print("Doing original geometry calc.")
            else:
                calclabel=label
                #displacement_jobname='Numfreq-Disp-'+'Atom'+str(atom_disp)+crd+drection
                print("Displacing {}".format(calclabel))
            energy, gradient = theory.run(current_coords=geo, elems=elems, Grad=True, nprocs=numcores)
            #Adding gradient to dictionary for AtomNCoordPDirectionm
            displacement_grad_dictionary[calclabel] = gradient
    elif runmode == 'parallel':
        import pickle4reducer
        import multiprocessing as mp
        ctx = mp.get_context()
        ctx.reducer = pickle4reducer.Pickle4Reducer()

        pool = mp.Pool(numcores)
        blankline()
        print("Running snapshots in parallel using multiprocessing.Pool")
        print("Number of CPU cores: ", numcores)
        print("Number of displacements:", len(list_of_displaced_geos))

        #NumcoresQM can be larger value (e.g. ORCA-parallelization). ORCA seems to run fine with OpenMPI without complaints.
        #However, this only makes sense to use if way more CPUs available than displacements.
        #Unlikely situation, so hardcoding to 1 for now.
        numcoresQM=1
        print("Setting nprocs for theory object to: ", numcoresQM)
        #results = pool.map(displacement_run, [[geo, elems, numcoresQM, theory, label] for geo,label in zip(list_of_displaced_geos,list_of_labels)])
        #results = pool.map(displacement_run2, [[filelabel, numcoresQM, theory, label] for label,filelabel in zip(list_of_labels,list_of_filelabels)])

        #Reducing size of theory object
        print("size of theory:", get_size(theory))
        print("size of theory.coords:", get_size(theory.coords))
        print("size of coords:", get_size(coords))
        theory.coords=[]
        theory.elems=[]
        theory.connectivity=[]
        print(theory)
        #print(theory.__dict__)
        print("size of theory after del:", get_size(theory))

        #QMMM_xtb = QMMMTheory(fragment=Saddlepoint, qm_theory=xtbcalc, mm_theory=MMpart, actatoms=Centralmainfrag,
        #                      qmatoms=Centralmainfrag, embedding='Elstat', nprocs=numcores)

        #results = pool.map(displacement_run2, [[filelabel, numcoresQM, label] for label,filelabel in zip(list_of_labels,list_of_filelabels)])

        #Because passing QMMMTheory is too big for pickle inside mp.Pool we create a new QMMMTheory object inside displacement funciont.
        #This means we need the components of theory object. Here distinguishing between QMMMTheory and other theory (QM theory)
        #Still seems to be too messy

        #https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
        if theory.__class__.__name__ == "QMMMTheory":
            try:
                import ray
            except:
                print("Parallel QM/MM Numerical Frequencies require the ray library.")
                print("Please install ray : pip install ray")
                exit(1)
            print("Numfreq with QMMMTheory")
            ray.init(num_cpus = numcores)
            #going to make QMMMTheory object a shared object that all workers can access


            theory.mm_theory.calculate_LJ_pairpotentials(qmatoms=theory.qmatoms, actatoms=theory.actatoms)
            print("theory.mm_theory sigmaij", theory.mm_theory.sigmaij)
            theory_shared = ray.put(theory)

            @ray.remote
            def dispfunction_ray(label, filelabel, numcoresQM, theory_shared):
                print("inside dispfunction")
                print("label:", label)
                print("filelabel:", filelabel)
                print("theory_shared:", theory_shared)
                # print("theory_shared.qmatoms: ", theory_shared.qmatoms )
                print("xx")
                # Numcores can be used. We can launch ORCA-OpenMPI in parallel it seems.
                # Only makes sense if we have may more cores available than displacements
                print("a")
                elems, coords = read_xyzfile(filelabel + '.xyz')
                print("b")
                dispdir = label.replace(' ', '')
                os.mkdir(dispdir)
                os.chdir(dispdir)
                print("d")
                # shutil.move('../' + filelabel + '.xyz', './' + filelabel + '.xyz')
                # Read XYZ-file from file
                print("e")

                print("f")
                # Todo: Copy previous GBW file in here if ORCA, xtbrestart if xtb, etc.
                print("Running displacement: {}".format(label))
                energy, gradient = theory_shared.run(current_coords=coords, elems=elems, Grad=True, nprocs=numcoresQM)
                print("Energy: ", energy)
                os.chdir('..')
                # Delete dir?
                # os.remove(dispdir)
                return [label, energy, gradient]


            result_ids = [dispfunction_ray.remote(label,filelabel,numcoresQM,theory_shared) for label,filelabel in
                          zip(list_of_labels,list_of_filelabels)]

            #result_ids = [f.remote(df_id) for _ in range(4)]

            #results = pool.map(displacement_QMrun, [[geo, elems, numcoresQM, theory, label] for geo, label in
            #                                        zip(list_of_displaced_geos, list_of_labels)])

            results = ray.get(result_ids)

            print(results)
            #results = pool.map(displacement_QMMMrun, [[filelabel, numcoresQM, label, theory.fragment, theory.qm_theory, theory.mm_theory,
            #                                        theory.actatoms, theory.qmatoms, theory.embedding, theory.charges, theory.printlevel,
            #                                        theory.frozenatoms] for label,filelabel in zip(list_of_labels,list_of_filelabels)])
        #Passing QM theory directly
        else:
            results = pool.map(displacement_QMrun, [[geo, elems, numcoresQM, theory, label] for geo,label in zip(list_of_displaced_geos,list_of_labels)])
        pool.close()

        #Gathering results in dictionary
        for result in results:
            print("result:", result)
            calclabel=result[0]
            energy=result[1]
            gradient=result[2]
            displacement_grad_dictionary[calclabel] = gradient

    print("Displacement calculations done.")
    #print("displacement_grad_dictionary:", displacement_grad_dictionary)
    #If partial Hessian remove non-hessatoms part of gradient:
    #Get partial matrix by deleting atoms not present in list.
    if npoint == 1:
        original_grad=get_partial_matrix(allatoms, hessatoms, displacement_grad_dictionary['Originalgeo'])
        original_grad_1d = np.ravel(original_grad)
    #Initialize Hessian
    hesslength=3*len(hessatoms)
    hessian=np.zeros((hesslength,hesslength))


    #Onepoint-formula Hessian
    if npoint == 1:
        #Starting index for Hessian array
        index=0
        #Getting displacements as keys from dictionary and sort
        dispkeys = list(displacement_grad_dictionary.keys())
        #Sort seems to sort it correctly w.r.t. atomnumber,x,y,z and +/-
        dispkeys.sort()
        #print("dispkeys:", dispkeys)
        #for displacement, grad in displacement_grad_dictionary.items():
        for dispkey in dispkeys:
            grad=displacement_grad_dictionary[dispkey]
            #Skipping original geo
            if dispkey != 'Originalgeo':
                #Getting grad as numpy matrix and converting to 1d
                # If partial Hessian remove non-hessatoms part of gradient:
                grad = get_partial_matrix(allatoms, hessatoms, grad)
                grad_1d = np.ravel(grad)
                Hessrow=(grad_1d - original_grad_1d)/displacement_bohr
                hessian[index,:]=Hessrow
                index+=1
    #Twopoint-formula Hessian. pos and negative directions come in order
    elif npoint == 2:
        count=0; hessindex=0
        #Getting displacements as keys from dictionary and sort
        dispkeys = list(displacement_grad_dictionary.keys())
        #Sort seems to sort it correctly w.r.t. atomnumber,x,y,z and +/-
        dispkeys.sort()
        #print("dispkeys:", dispkeys)
        #for file in freqinputfiles:
        #for displacement, grad in testdict.items():
        for dispkey in dispkeys:
            if dispkey != 'Originalgeo':
                count+=1
                if count == 1:
                    grad_pos=displacement_grad_dictionary[dispkey]
                    #print("pos I hope")
                    #print("dispkey:", dispkey)
                    # If partial Hessian remove non-hessatoms part of gradient:
                    grad_pos = get_partial_matrix(allatoms, hessatoms, grad_pos)
                    grad_pos_1d = np.ravel(grad_pos)
                elif count == 2:
                    grad_neg=displacement_grad_dictionary[dispkey]
                    #print("neg I hope")
                    #print("dispkey:", dispkey)
                    #Getting grad as numpy matrix and converting to 1d
                    # If partial Hessian remove non-hessatoms part of gradient:
                    grad_neg = get_partial_matrix(allatoms, hessatoms, grad_neg)
                    grad_neg_1d = np.ravel(grad_neg)
                    Hessrow=(grad_pos_1d - grad_neg_1d)/(2*displacement_bohr)
                    hessian[hessindex,:]=Hessrow
                    grad_pos_1d=0
                    grad_neg_1d=0
                    count=0
                    hessindex+=1
                else:
                    print("Something bad happened")
                    exit()
    blankline()

    #Symmetrize Hessian by taking average of matrix and transpose
    symm_hessian=(hessian+hessian.transpose())/2
    hessian=symm_hessian


    #Project out Translation+Rotational modes
    #TODO

    #Diagonalize mass-weighted Hessian
    # Get partial matrix by deleting atoms not present in list.
    hesselems = get_partial_list(allatoms, hessatoms, elems)
    hessmasses = get_partial_list(allatoms, hessatoms, fragment.list_of_masses)
    hesscoords = [fragment.coords[i] for i in hessatoms]
    print("Elements:", hesselems)
    print("Masses used:", hessmasses)
    #Todo: Note. elems is redefined here. Not ideal
    frequencies, nmodes, numatoms, elems, evectors, atomlist, masses = diagonalizeHessian(hessian,hessmasses,hesselems)
    #frequencies=diagonalizeHessian(hessian,hessmasses,hesselems)[0]

    #Print out normal mode output. Like in Chemshell or ORCA
    blankline()
    print("Normal modes:")
    #TODO: Eigenvectors print here.
    #TODO: or perhaps elemental normal mode composition factors
    print("Eigenvectors to be  be printed here")
    blankline()
    #Print out Freq output. Maybe print normal mode compositions here instead???
    printfreqs(frequencies,len(hessatoms))



    #Print out thermochemistry
    if theory.__class__.__name__ == "QMMMTheory":
        Thermochemistry = thermochemcalc(frequencies,hessatoms, fragment, theory.qm_theory.mult, temp=temp,pressure=pressure)
    else:
        Thermochemistry = thermochemcalc(frequencies,hessatoms, fragment, theory.mult, temp=temp,pressure=pressure)


    #Write Hessian to file
    with open("Hessian", 'w') as hfile:
        hfile.write(str(hesslength)+' '+str(hesslength)+'\n')
        for row in hessian:
            rowline=' '.join(map(str, row))
            hfile.write(str(rowline)+'\n')
        blankline()
        print("Wrote Hessian to file: Hessian")
    #Write ORCA-style Hessian file. Hardcoded filename here. Change?
    #Note: Passing hesscords here instead of coords. Change?
    write_ORCA_Hessfile(hessian, hesscoords, hesselems, hessmasses, hessatoms, "orcahessfile.hess")
    print("Wrote ORCA-style Hessian file: orcahessfile.hess")

    #Create dummy-ORCA file with frequencies and normal modes
    printdummyORCAfile(hesselems, hesscoords, frequencies, evectors, nmodes, "orcahessfile.hess")
    print("Wrote dummy ORCA outputfile with frequencies and normal modes: orcahessfile.hess_dummy.out")
    print("Can be used for visualization")

    #TODO: https://pages.mtu.edu/~msgocken/ma5630spring2003/lectures/diff/diff/node6.html
    blankline()
    print(BC.WARNING, BC.BOLD, "------------NUMERICAL FREQUENCIES END-------------", BC.END)

    #Thermochemistry object. Should contain frequencies, zero-point energy, enthalpycorr, gibbscorr, etc.
    
    #Return to ..
    os.chdir('..')
    return Thermochemistry

#Molecular dynamics class
class MolecularDynamics:
    def __init__(self, fragment, theory, ensemble, temperature):
        self.fragment=fragment
        self.theory=theory
        self.ensemble=ensemble
        self.temperature=temperature
    def run(self):
        print("Molecular dynamics is not ready yet")
        exit()

def print_time_rel(timestampA,modulename=''):
    secsA=time.time()-timestampA
    minsA=secsA/60
    print("-------------------------------------------------------------------")
    print("Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes.".format(modulename, secsA, minsA ))
    print("-------------------------------------------------------------------")

def print_time_rel_and_tot(timestampA,timestampB, modulename=''):
    secsA=time.time()-timestampA
    minsA=secsA/60
    #hoursA=minsA/60
    secsB=time.time()-timestampB
    minsB=secsB/60
    #hoursB=minsB/60
    print("-------------------------------------------------------------------")
    print("Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes.".format(modulename, secsA, minsA ))
    print("Total Walltime: {:3.1f} seconds, {:3.1f} minutes.".format(secsB, minsB ))
    print("-------------------------------------------------------------------")

def print_time_rel_and_tot_color(timestampA,timestampB, modulename=''):
    secsA=time.time()-timestampA
    minsA=secsA/60
    #hoursA=minsA/60
    secsB=time.time()-timestampB
    minsB=secsB/60
    #hoursB=minsB/60
    print(BC.WARNING,"-------------------------------------------------------------------", BC.END)
    print(BC.WARNING,"Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes.".format(modulename, secsA, minsA ), BC.END)
    print(BC.WARNING,"Total Walltime: {:3.1f} seconds, {:3.1f} minutes.".format(secsB, minsB ), BC.END)
    print(BC.WARNING,"-------------------------------------------------------------------", BC.END)

#Theory classes

#Dummy MM theory template

#class Theory:
#    def __init__(self, printlevel=None):
#        #Atom types
#    pass
#    def run(self):
#        pass

# Dummy QM theory template
#class Theory:
#    def __init__(self, charge, mult, printlevel=None):
#    def run(self):
#        pass

# Dummy QM theory template



# Theory object that always gives zero energy and zero gradient. Useful for setting constraints
class ZeroTheory:
    def __init__(self, fragment=None, charge=None, mult=None, printlevel=None, nprocs=1, label=None):
        self.nprocs=nprocs
        self.charge=charge
        self.mult=mult
        self.printlevel=printlevel
        self.label=label
        self.fragment=fragment
        pass
    def run(self, current_coords=None, elems=None, Grad=False, PC=False, nprocs=None ):
        self.energy = 0.0
        #Numpy object
        self.gradient = np.zeros((len(elems), 3))
        return self.energy,self.gradient



# Different MM theories

#Todo: Also think whether we want do OpenMM simulations in case we have to make another object maybe
#Amberfiles:
# Needs just amberprmtopfile (new-style Amber7 format). Not inpcrd file, read into ASH fragment instead

#CHARMMfiles:
#Need psffile, a CHARMM topologyfile (charmmtopfile) and a CHARMM parameter file (charmmprmfile)

#GROMACSfiles:
# Need gromacstopfile and grofile (contains periodic information along with coordinates) and gromacstopdir location (topology)

#Dependencies:
#OpenMM. Install via conda: conda install -c omnia openmm
#


class OpenMMTheory:
    def __init__(self, pdbfile=None, platform='CPU', active_atoms=None, frozen_atoms=None,
                 CHARMMfiles=False, psffile=None, charmmtopfile=None, charmmprmfile=None,
                 GROMACSfiles=False, gromacstopfile=None, grofile=None, gromacstopdir=None,
                 Amberfiles=False, amberprmtopfile=None, printlevel=2, do_energy_composition=True,
                 xmlfile=None, periodic=False, periodic_cell_dimensions=None, customnonbondedforce=False):
        
        timeA = time.time()
        # OPEN MM load
        try:
            import simtk.openmm.app
            import simtk.unit
            #import simtk.openmm
        except ImportError:
            raise ImportError(
                "OpenMM requires installing the OpenMM package. Try: conda install -c omnia openmm  \
                Also see http://docs.openmm.org/latest/userguide/application.html")

        print(BC.WARNING, BC.BOLD, "------------Defining OpenMM object-------------", BC.END)
        #Printlevel
        self.printlevel=printlevel

        #Parallelization
        #Control by setting env variable: $OPENMM_CPU_THREADS in shell before running.
        #Don't think it's possible to change variable inside Python environment
        print("OpenMM will use {} threads".format(os.environ["OPENMM_CPU_THREADS"]))
        
        #Whether to do energy composition of MM energy or not. Takes time. Can be turned off for MD runs
        self.do_energy_composition=do_energy_composition
        #Initializing
        self.coords=[]
        self.charges=[]
        self.Periodic = periodic

            
        #OpenMM things
        self.openmm=simtk.openmm
        self.simulationclass=simtk.openmm.app.simulation.Simulation
        self.langevinintegrator=simtk.openmm.LangevinIntegrator
        self.platform_choice=platform
        self.unit=simtk.unit
        self.Vec3=simtk.openmm.Vec3


        #TODO: Should we keep this? Probably not. Coordinates would be handled by ASH.
        #PDB_ygg_frag = Fragment(pdbfile=pdbfile, conncalc=False)
        #self.coords=PDB_ygg_frag.coords
        print_time_rel(timeA, modulename="prep")
        timeA = time.time()

        self.Forcefield=None
        #What type of forcefield files to read. Reads in different way.
        # #Always creates object we call self.forcefield that contains topology attribute
        if CHARMMfiles is True:
            self.Forcefield='CHARMM'
            print("Reading CHARMM files")
            # Load CHARMM PSF files. Both CHARMM-style and XPLOR allowed I believe. Todo: Check
            self.psffile=psffile
            self.psf = simtk.openmm.app.CharmmPsfFile(psffile)
            self.params = simtk.openmm.app.CharmmParameterSet(charmmtopfile, charmmprmfile)
            # self.pdb = simtk.openmm.app.PDBFile(pdbfile) probably not reading coordinates here
            #self.forcefield = self.psf
            self.topology = self.psf.topology
            
            #Setting active and frozen variables once topology is in place
            self.set_active_and_frozen_regions(active_atoms=active_atoms, frozen_atoms=frozen_atoms)
            
            # Create an OpenMM system by calling createSystem on psf
            
            #Periodic
            if self.Periodic is True:
                print("System is periodic")
                self.periodic_cell_dimensions = periodic_cell_dimensions
                self.a = periodic_cell_dimensions[0] * self.unit.angstroms
                self.b = periodic_cell_dimensions[1] * self.unit.angstroms
                self.c = periodic_cell_dimensions[2] * self.unit.angstroms
                
                #Parameters here are based on OpenMM DHFR example
                self.psf.setBox(self.a, self.b, self.c)
                
                self.system = self.psf.createSystem(self.params, nonbondedMethod=simtk.openmm.app.PME,
                                                nonbondedCutoff=12 * self.unit.angstroms, switchDistance=10*self.unit.angstroms)
                
                #TODO: Customnonbonded force option here
                
                print("self.system.getForces() :", self.system.getForces())
                for i,force in enumerate(self.system.getForces()):
                    if isinstance(force, simtk.openmm.CustomNonbondedForce):
                        print('CustomNonbondedForce: %s' % force.getUseSwitchingFunction())
                        print('LRC? %s' % force.getUseLongRangeCorrection())
                        force.setUseLongRangeCorrection(False)
                    elif isinstance(force, simtk.openmm.NonbondedForce):
                        print('NonbondedForce: %s' % force.getUseSwitchingFunction())
                        print('LRC? %s' % force.getUseDispersionCorrection())
                        force.setUseDispersionCorrection(False)
                        force.setPMEParameters(1.0/0.34, periodic_cell_dimensions[3], periodic_cell_dimensions[4], periodic_cell_dimensions[5]) 
                        self.nonbonded_force=force
                        # NOTE: These are hard-coded!
                        
                #Set charges in OpenMMobject by taking from Force
                print("Setting charges")
                self.getatomcharges(self.nonbonded_force)
                        
                
            #Non-Periodic
            else:
                print("System is non-periodic")
                #For frozen systems we use Customforce in order to specify interaction groups
                #if len(self.frozen_atoms) > 0:
                    
                    #Two possible ways.
                    #https://github.com/openmm/openmm/issues/2698
                    #1. Use CustomNonbondedForce  with interaction groups. Could be slow
                    #2. CustomNonbondedForce but with scaling
                
                
                #https://ahy3nz.github.io/posts/2019/30/openmm2/
                #http://www.maccallumlab.org/news/2015/1/23/testing
                
                #Comes close to NonbondedForce results (after exclusions) but still not correct
                #The issue is most likely that the 1-4 LJ interactions should not be excluded but rather scaled.
                #See https://github.com/openmm/openmm/issues/1200
                #https://github.com/openmm/openmm/issues/1696
                #How to do:
                #1. Keep nonbonded force for only those interactions and maybe also electrostatics?
                #Mimic this??: https://github.com/openmm/openmm/blob/master/devtools/forcefield-scripts/processCharmmForceField.py
                #Or do it via Parmed? Better supported for future??
                #2. Go through the 1-4 interactions and not exclude but scale somehow manually. But maybe we can't do that in CustomNonbonded Force?
                #Presumably not but maybe can add a special force object just for 1-4 interactions. We
                def create_cnb(original_nbforce):
                    """Creates a CustomNonbondedForce object that mimics the original nonbonded force
                    and also a Custombondforce to handle 14 exceptions
                    """
                    #Next, create a CustomNonbondedForce with LJ and Coulomb terms
                    ONE_4PI_EPS0 = 138.935456
                    #ONE_4PI_EPS0=1.0
                    #TODO: Not sure whether sqrt should be present or not in epsilon???
                    energy_expression  = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod/r;"
                    #sqrt ??
                    energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
                    energy_expression += "sigma = 0.5*(sigma1+sigma2);"
                    energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
                    energy_expression += "chargeprod = charge1*charge2;"
                    custom_nonbonded_force = simtk.openmm.CustomNonbondedForce(energy_expression)
                    custom_nonbonded_force.addPerParticleParameter('charge')
                    custom_nonbonded_force.addPerParticleParameter('sigma')
                    custom_nonbonded_force.addPerParticleParameter('epsilon')
                    # Configure force
                    custom_nonbonded_force.setNonbondedMethod(simtk.openmm.CustomNonbondedForce.NoCutoff)
                    #custom_nonbonded_force.setCutoffDistance(9999999999)
                    custom_nonbonded_force.setUseLongRangeCorrection(False)
                    #custom_nonbonded_force.setUseSwitchingFunction(True)
                    #custom_nonbonded_force.setSwitchingDistance(99999)
                    print('adding particles to custom force')
                    for index in range(self.system.getNumParticles()):
                        [charge, sigma, epsilon] = original_nbforce.getParticleParameters(index)
                        custom_nonbonded_force.addParticle([charge, sigma, epsilon])
                    #For CustomNonbondedForce we need (unlike NonbondedForce) to create exclusions that correspond to the automatic exceptions in NonbondedForce
                    #These are interactions that are skipped for bonded atoms
                    numexceptions = original_nbforce.getNumExceptions()
                    print("numexceptions in original_nbforce: ", numexceptions)
                    
                    #Turn exceptions from NonbondedForce into exclusions in CustombondedForce
                    # except 1-4 which are not zeroed but are scaled. These are added to Custombondforce
                    exceptions_14=[]
                    numexclusions=0
                    for i in range(0,numexceptions):
                        #print("i:", i)
                        #Get exception parameters (indices)
                        p1,p2,charge,sigma,epsilon = original_nbforce.getExceptionParameters(i)
                        #print("p1,p2,charge,sigma,epsilon:", p1,p2,charge,sigma,epsilon)
                        #If 0.0 then these are CHARMM 1-2 and 1-3 interactions set to zero
                        if charge._value==0.0 and epsilon._value==0.0:
                            #print("Charge and epsilons are 0.0. Add proper exclusion")
                            #Set corresponding exclusion in customnonbforce
                            custom_nonbonded_force.addExclusion(p1,p2)
                            numexclusions+=1
                        else:
                            #print("This is not an exclusion but a scaled interaction as it is is non-zero. Need to keep")
                            exceptions_14.append([p1,p2,charge,sigma,epsilon])
                            #[798, 801, Quantity(value=-0.0684, unit=elementary charge**2), Quantity(value=0.2708332103146632, unit=nanometer), Quantity(value=0.2672524882578271, unit=kilojoule/mole)]
                    
                    print("len exceptions_14", len(exceptions_14))
                    #print("exceptions_14:", exceptions_14)
                    print("numexclusions:", numexclusions)
                    
                    
                    #Creating custombondforce to handle these special exceptions
                    #Now defining pair parameters
                    #https://github.com/openmm/openmm/issues/2698
                    energy_expression  = "(4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod/r);"
                    energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)  # already in OpenMM units
                    custom_bond_force = self.openmm.CustomBondForce(energy_expression)
                    custom_bond_force.addPerBondParameter('chargeprod')
                    custom_bond_force.addPerBondParameter('sigma')
                    custom_bond_force.addPerBondParameter('epsilon')
                    
                    for exception in exceptions_14:
                        idx=exception[0];jdx=exception[1];c=exception[2];sig=exception[3];eps=exception[4]
                        custom_bond_force.addBond(idx, jdx, [c, sig, eps])
                    
                    print('Number of defined 14 bonds in custom_bond_force:', custom_bond_force.getNumBonds())
                    
                    
                    return custom_nonbonded_force,custom_bond_force

                #TODO: Look into: https://github.com/ParmEd/ParmEd/blob/7e411fd03c7db6977e450c2461e065004adab471/parmed/structure.py#L2554
                    
                    #myCustomNBForce= simtk.openmm.CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")
                    #myCustomNBForce.setNonbondedMethod(simtk.openmm.app.NoCutoff)
                    #myCustomNBForce.setCutoffDistance(1000*simtk.openmm.unit.angstroms)
                    #Frozen-Act interaction
                    #myCustomNBForce.addInteractionGroup(self.frozen_atoms,self.active_atoms)
                    #Act-Act interaction
                    #myCustomNBForce.addInteractionGroup(self.active_atoms,self.active_atoms)
                

                self.system = self.psf.createSystem(self.params, nonbondedMethod=simtk.openmm.app.NoCutoff,
                                                    nonbondedCutoff=1000 * simtk.openmm.unit.angstroms)
                print("system created")
                print("Number of forces:", self.system.getNumForces())
                print(self.system.getForces())
                print("")                
                print("")
                #print("original forces: ", forces)
                # Get charges from OpenMM object into self.charges
                #self.getatomcharges(forces['NonbondedForce'])
                self.getatomcharges(self.system.getForces()[6])
                

                #CASE CUSTOMNONBONDED FORCE
                if customnonbondedforce is True:

                    #Create CustomNonbonded force
                    for i,force in enumerate(self.system.getForces()):
                        if isinstance(force, self.openmm.NonbondedForce):
                            custom_nonbonded_force,custom_bond_force = create_cnb(self.system.getForces()[i])
                    print("1custom_nonbonded_force:", custom_nonbonded_force)
                    print("num exclusions in customnonb:", custom_nonbonded_force.getNumExclusions())
                    print("num 14 exceptions in custom_bond_force:", custom_bond_force.getNumBonds())
                    
                    #TODO: Deal with frozen regions. NOT YET DONE
                    #Frozen-Act interaction
                    #custom_nonbonded_force.addInteractionGroup(self.frozen_atoms,self.active_atoms)
                    #Act-Act interaction
                    #custom_nonbonded_force.addInteractionGroup(self.active_atoms,self.active_atoms)
                    #print("2custom_nonbonded_force:", custom_nonbonded_force)
                
                    #Pointing self.nonbonded_force to CustomNonBondedForce instead of Nonbonded force
                    self.nonbonded_force = custom_nonbonded_force
                    print("self.nonbonded_force:", self.nonbonded_force)
                    self.custom_bondforce = custom_bond_force
                    
                    #Update system with new forces and delete old force
                    self.system.addForce(self.nonbonded_force) 
                    self.system.addForce(self.custom_bondforce) 
                    
                    #Remove oldNonbondedForce
                    for i,force in enumerate(self.system.getForces()):
                        if isinstance(force, self.openmm.NonbondedForce):
                            self.system.removeForce(i)

                else:
                    #Regular Nonbonded force
                    self.nonbonded_force=self.system.getForce(6)
                            

                print("")
                print("Number of forces:", self.system.getNumForces())
                print(self.system.getForces())
        elif GROMACSfiles is True:
            print("Warning: Gromacs-file interface not tested")
            #Reading grofile, not for coordinates but for periodic vectors
            gro = simtk.openmm.app.GromacsGroFile(grofile)
            self.grotop = simtk.openmm.app.GromacsTopFile(gromacstopfile, periodicBoxVectors=gro.getPeriodicBoxVectors(),
                                 includeDir=gromacstopdir)
            #self.forcefield=self.grotop
            self.topology = self.grotop.topology
            # Create an OpenMM system by calling createSystem on grotop
            self.system = self.grotop.createSystem(nonbondedMethod=simtk.openmm.app.NoCutoff,
                                                nonbondedCutoff=1 * simtk.openmm.unit.nanometer)
            
            forces = {self.system.getForce(index).__class__.__name__: self.system.getForce(index) for index in range(self.system.getNumForces())}
            self.nonbonded_force = forces['NonbondedForce']
        elif Amberfiles is True:
            self.Forcefield='Amber'
            print("Warning: Amber-file interface not tested")
            #Note: Only new-style Amber7 prmtop files work
            self.prmtop = simtk.openmm.app.AmberPrmtopFile(amberprmtopfile)
            #inpcrd = simtk.openmm.app.AmberInpcrdFile(inpcrdfile)  probably not reading coordinates here
            #self.forcefield = self.prmtop
            self.topology = self.prmtop.topology
            # Create an OpenMM system by calling createSystem on prmtop
            self.system = self.prmtop.createSystem(nonbondedMethod=simtk.openmm.app.NoCutoff,
                                                nonbondedCutoff=1 * simtk.openmm.unit.nanometer)
            
            forces = {self.system.getForce(index).__class__.__name__: self.system.getForce(index) for index in range(self.system.getNumForces())}
            self.nonbonded_force = forces['NonbondedForce']
        else:
            print("Reading OpenMM XML forcefield file and PDB file")
            #This would be regular OpenMM Forcefield definition requiring XML file
            #Topology from PDBfile annoyingly enough
            pdb = simtk.openmm.app.PDBFile(pdbfile)
            self.topology = pdb.topology
            #Todo: support multiple xml file here
            #forcefield = simtk.openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
            self.forcefield = simtk.openmm.app.ForceField(xmlfile)
            self.system = self.forcefield.createSystem(nonbondedMethod=simtk.openmm.app.NoCutoff,
                                                nonbondedCutoff=1 * simtk.openmm.unit.nanometer)
            
            forces = {self.system.getForce(index).__class__.__name__: self.system.getForce(index) for index in range(self.system.getNumForces())}
            self.nonbonded_force = forces['NonbondedForce']







        print_time_rel(timeA, modulename="system create")
        timeA = time.time()
        #constraints=simtk.openmm.app.HBonds, AllBonds, HAngles


        # Remove Frozen-Frozen interactions
        #Todo: Will be requested by QMMM object so unnecessary unless during pure MM??
        #if frozen_atoms is not None:
        #    print("Removing Frozen-Frozen interactions")
        #    self.addexceptions(frozen_atoms)


        #Modify particle masses in system object. For freezing atoms
        #for i in self.frozen_atoms:
        #    self.system.setParticleMass(i, 0 * simtk.openmm.unit.dalton)
        #print_time_rel(timeA, modulename="frozen atom setup")
        #timeA = time.time()

        #Modifying constraints after frozen-atom setting
        #print("Constraints:", self.system.getNumConstraints())

        #Finding defined constraints that involved frozen atoms. add to remove list
        #removelist=[]
        #for i in range(0,self.system.getNumConstraints()):
        #    constraint=self.system.getConstraintParameters(i)
        #    if constraint[0] in self.frozen_atoms or constraint[1] in self.frozen_atoms:
        #        #self.system.removeConstraint(i)
        #        removelist.append(i)

        #print("removelist:", removelist)
        #print("length removelist", len(removelist))
        #Remove constraints
        #removelist.reverse()
        #for r in removelist:
        #    self.system.removeConstraint(r)

        #print("Constraints:", self.system.getNumConstraints())
        #print_time_rel(timeA, modulename="constraint fix")
        timeA = time.time()
        
        self.forcegroupify()
        print_time_rel(timeA, modulename="forcegroupify")
        timeA = time.time()
        
        #Dummy integrator
        self.integrator = self.langevinintegrator(300 * self.unit.kelvin,  # Temperature of heat bath
                                        1 / self.unit.picosecond,  # Friction coefficient
                                        0.002 * self.unit.picoseconds)  # Time step
        self.platform = simtk.openmm.Platform.getPlatformByName(self.platform_choice)

        #Defined first here. 
        #NOTE: If self.system is modified then we have to remake self.simulation
        #self.simulation = simtk.openmm.app.simulation.Simulation(self.topology, self.system, self.integrator,self.platform)
        self.simulation = self.simulationclass(self.topology, self.system, self.integrator,self.platform)


        print_time_rel(timeA, modulename="simulation setup")
        timeA = time.time()
        



    def set_active_and_frozen_regions(self, active_atoms=None, frozen_atoms=None):
        #FROZEN AND ACTIVE ATOMS
        self.numatoms=int(self.psf.topology.getNumAtoms())
        print("self.numatoms:", self.numatoms)
        self.allatoms=list(range(0,self.numatoms))
        if active_atoms is None and frozen_atoms is None:
            print("All {} atoms active, no atoms frozen".format(len(self.allatoms)))
            self.frozen_atoms = []
        elif active_atoms is not None and frozen_atoms is None:
            self.active_atoms=active_atoms
            self.frozen_atoms=listdiff(self.allatoms,self.active_atoms)
            print("{} active atoms, {} frozen atoms".format(len(self.active_atoms),len(self.frozen_atoms)))
            #listdiff
        elif frozen_atoms is not None and active_atoms is None:
            self.frozen_atoms = frozen_atoms
            self.active_atoms = listdiff(self.allatoms, self.frozen_atoms)
            print("{} active atoms, {} frozen atoms".format(len(self.active_atoms),len(self.frozen_atoms)))
        else:
            print("active_atoms and frozen_atoms can not be both defined")
            exit(1)
    #This removes interactions between particles (e.g. QM-QM or frozen-frozen pairs)
    # list of atom indices for which we will remove all pairs

    #Todo: Way too slow to do for all frozen atoms but works well for qmatoms list size
    # Alternative: Remove force interaction and then add in the interaction of active atoms to frozen atoms
    # should be reasonably fast
    # https://github.com/openmm/openmm/issues/2124
    #https://github.com/openmm/openmm/issues/1696
    def addexceptions(self,atomlist):
        import itertools
        print("Add exceptions/exclusions. Removing i-j interactions for list :", len(atomlist), "atoms")
        timeA=time.time()
        #Has duplicates
        #[self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i in atomlist for j in atomlist]
        #https://stackoverflow.com/questions/942543/operation-on-every-pair-of-element-in-a-list
        #[self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i,j in itertools.combinations(atomlist, r=2)]
        numexceptions=0
        if isinstance(self.nonbonded_force, self.openmm.NonbondedForce):
            print("Case Nonbondedforce. Adding Exception for ij pair")
            for i in atomlist:
                for j in atomlist:
                    #print("i,j : ", i,j)
                    self.nonbonded_force.addException(i,j,0, 0, 0, replace=True)
                    numexceptions+=1
        elif isinstance(self.nonbonded_force, self.openmm.CustomNonbondedForce):
            print("Case CustomNonbondedforce. Adding Exclusion for ij pair")
            for i in atomlist:
                for j in atomlist:
                    #print("i,j : ", i,j)
                    self.nonbonded_force.addExclusion(i,j)
                    numexceptions+=1
        print("Number of exceptions/exclusions added: ", numexceptions)
        
        #Seems like updateParametersInContext does not reliably work here so we have to remake the simulation instead
        #Might be bug (https://github.com/openmm/openmm/issues/2709). Revisit
        #self.nonbonded_force.updateParametersInContext(self.simulation.context)
        self.integrator = self.langevinintegrator(300 * self.unit.kelvin,  # Temperature of heat bath
                                        1 / self.unit.picosecond,  # Friction coefficient
                                        0.002 * self.unit.picoseconds)  # Time step
        self.simulation = self.simulationclass(self.topology, self.system, self.integrator,self.platform)
        
        print_time_rel(timeA, modulename="add exception")
    #Run: coords or framents can be given (usually coords). qmatoms in order to avoid QM-QM interactions (TODO)
    #Probably best to do QM-QM exclusions etc. in a separate function though as we want run to be as simple as possible
    #qmatoms list provided for generality of MM objects. Not used here for now
    
    
    #Functions for energy compositions
    def forcegroupify(self):
        self.forcegroups = {}
        print("inside forcegroupify")
        print("self.system.getForces() ", self.system.getForces())
        print("Number of forces:", self.system.getNumForces())
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            force.setForceGroup(i)
            self.forcegroups[force] = i
        print("self.forcegroups :", self.forcegroups)
        #exit()
    def getEnergyDecomposition(self,context, forcegroups):
        energies = {}
        for f, i in forcegroups.items():
            energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
        return energies
    
    def printEnergyDecomposition(self):
        timeA=time.time()
        #Energy composition
        #TODO: Calling this is expensive as the energy has to be recalculated.
        # Only do for cases: a) single-point b) First energy-step in optimization and last energy-step
        # OpenMM energy components
        openmm_energy = dict()
        energycomp = self.getEnergyDecomposition(self.simulation.context, self.forcegroups)
        print("energycomp: ", energycomp)
        print("len energycomp", len(energycomp))
        print("openmm_energy: ", openmm_energy)
        print("")
        bondterm_set=False
        extrafcount=0
        #This currently assumes CHARMM36 components, More to be added
        for comp in energycomp.items():
            #print("comp: ", comp)
            if 'HarmonicBondForce' in str(type(comp[0])):
                #Not sure if this works in general.
                if bondterm_set is False:
                    openmm_energy['Bond'] = comp[1]
                    bondterm_set=True
                else:
                    openmm_energy['Urey-Bradley'] = comp[1]
            elif 'HarmonicAngleForce' in str(type(comp[0])):
                openmm_energy['Angle'] = comp[1]
            elif 'PeriodicTorsionForce' in str(type(comp[0])):
                #print("Here")
                openmm_energy['Dihedrals'] = comp[1]
            elif 'CustomTorsionForce' in str(type(comp[0])):
                openmm_energy['Impropers'] = comp[1]
            elif 'CMAPTorsionForce' in str(type(comp[0])):
                openmm_energy['CMAP'] = comp[1]
            elif 'NonbondedForce' in str(type(comp[0])):
                openmm_energy['Nonbonded'] = comp[1]
            elif 'CMMotionRemover' in str(type(comp[0])):
                openmm_energy['CMM'] = comp[1]
            elif 'CustomBondForce' in str(type(comp[0])):
                openmm_energy['14-LJ'] = comp[1]
            else:
                extrafcount+=1
                openmm_energy['Otherforce'+str(extrafcount)] = comp[1]
                
        
        print_time_rel(timeA, modulename="energy composition")
        timeA = time.time()
        
        #The force terms to print in the ordered table.
        # Deprecated. Better to print everything.
        #Missing terms in force_terms will be printed separately
        #if self.Forcefield == 'CHARMM':
        #    force_terms = ['Bond', 'Angle', 'Urey-Bradley', 'Dihedrals', 'Impropers', 'CMAP', 'Nonbonded', '14-LJ']
        #else:
        #    #Modify...
        #    force_terms = ['Bond', 'Angle', 'Urey-Bradley', 'Dihedrals', 'Impropers', 'CMAP', 'Nonbonded']

        #Sum all force-terms
        sumofallcomponents=0.0
        for val in openmm_energy.values():
            sumofallcomponents+=val._value
        
        #Print energy table       
        print('%-20s | %-15s | %-15s' % ('Component', 'kJ/mol', 'kcal/mol'))
        print('-'*56)
        #TODO: Figure out better sorting of terms
        for name in sorted(openmm_energy):
            print('%-20s | %15.2f | %15.2f' % (name, openmm_energy[name] / self.unit.kilojoules_per_mole, openmm_energy[name] / self.unit.kilocalorie_per_mole))
        print('-'*56)
        print('%-20s | %15.2f | %15.2f' % ('Sumcomponents', sumofallcomponents, sumofallcomponents / 4.184))
        print("")
        print('%-20s | %15.2f | %15.2f' % ('Total', self.energy * constants.hartokj , self.energy * constants.harkcal))
        
        print("")
        print("")
        print_time_rel(timeA, modulename="print table")
        
        
        timeA = time.time()
    
    def run(self, current_coords=None, elems=None, Grad=True, fragment=None, qmatoms=None):
        timeA = time.time()
        print(BC.OKBLUE, BC.BOLD, "------------RUNNING OPENMM INTERFACE-------------", BC.END)
        #If no coords given to run then a single-point job probably (not part of Optimizer or MD which would supply coords).
        #Then try if fragment object was supplied.
        #Otherwise internal coords if they exist
        print("if stuff")
        if current_coords is None:
            if fragment is None:
                if len(self.coords) != 0:
                    print("Using internal coordinates (from OpenMM object)")
                    current_coords=self.coords
                else:
                    print("Found no coordinates!")
                    exit(1)
            else:
                current_coords=fragment.coords

        print_time_rel(timeA, modulename="if stuff")
        timeA = time.time()
        #Making sure coords is np array and not list-of-lists
        print("doing coords array")
        current_coords=np.array(current_coords)
        print_time_rel(timeA, modulename="coords array")
        timeA = time.time()
        ##  unit conversion for energy
        #eqcgmx = 2625.5002
        ## unit conversion for force
        #TODO: Check this.
        #fqcgmx = -49614.75258920567
        #fqcgmx = -49621.9
        #Convert from kj/(nm *mol) = kJ/(10*Ang*mol)
        #factor=2625.5002/(10*1.88972612546)
        #factor=-138.93548724479302
        #Correct:
        factor=-49614.752589207

        #pos = [Vec3(coords[:,0]/10,coords[:,1]/10,coords[:,2]/10)] * u.nanometer
        #Todo: Check speed on this
        print("Updating coordinates")
        timeA = time.time()
        pos = [self.Vec3(current_coords[i, 0] / 10, current_coords[i, 1] / 10, current_coords[i, 2] / 10) for i in range(len(current_coords))] * self.unit.nanometer
        self.simulation.context.setPositions(pos)
        print_time_rel(timeA, modulename="context pos")
        timeA = time.time()
        print("Calculating MM state")
        state = self.simulation.context.getState(getEnergy=True, getForces=True)
        print_time_rel(timeA, modulename="state")
        timeA = time.time()
        self.energy = state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole) / constants.hartokj
        self.gradient = np.array(state.getForces(asNumpy=True)/factor)

        print("OpenMM Energy:", self.energy, "Eh")
        print("OpenMM Energy:", self.energy*constants.harkcal, "kcal/mol")
        
        #Do energy components or not. Can be turned off for e.g. MM MD simulation
        if self.do_energy_composition is True:
            self.printEnergyDecomposition()
        
        print("self.energy : ", self.energy, "Eh")
        print("Energy:", self.energy*constants.harkcal, "kcal/mol")
        #print("self.gradient:", self.gradient)

        print(BC.OKBLUE, BC.BOLD, "------------ENDING OPENMM INTERFACE-------------", BC.END)
        return self.energy, self.gradient
    
    #Get list of charges from chosen force object (usually original nonbonded force object)
    def getatomcharges(self,force):
        chargelist = []
        for i in range( force.getNumParticles() ):
            charge = force.getParticleParameters( i )[0]
            if isinstance(charge, self.unit.Quantity):
                charge = charge / self.unit.elementary_charge
                chargelist.append(charge)
        self.charges=chargelist
        return chargelist
    #Updating charges in OpenMM object. Used to set QM charges to 0 for example
    #Taking list of atom-indices and list of charges (usually zero) and setting new charge
    def update_charges(self,atomlist,atomcharges):
        print("Updating charges in OpenMM object.")
        assert len(atomlist) == len(atomcharges)
        newcharges=[]
        #print("atomlist:", atomlist)
        for atomindex,newcharge in zip(atomlist,atomcharges):
            #Updating big chargelist of OpenMM object.
            #TODO: Is this actually used?
            self.charges[atomindex]=newcharge
            #print("atomindex: ", atomindex)
            #print("newcharge: ",newcharge)
            oldcharge, sigma, epsilon = self.nonbonded_force.getParticleParameters(atomindex)
            #Different depending on type of NonbondedForce
            if isinstance(self.nonbonded_force, self.openmm.CustomNonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, [newcharge,sigma,epsilon])
                #bla1,bla2,bla3 = self.nonbonded_force.getParticleParameters(i)
                #print("bla1,bla2,bla3", bla1,bla2,bla3)
            elif isinstance(self.nonbonded_force, self.openmm.NonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, newcharge,sigma,epsilon)
                #bla1,bla2,bla3 = self.nonbonded_force.getParticleParameters(atomindex)
                #print("bla1,bla2,bla3", bla1,bla2,bla3)

        #Instead of recreating simulation we can just update like this:
        print("Updating simulation object for modified Nonbonded force")
        self.nonbonded_force.updateParametersInContext(self.simulation.context)
        
    def modify_bonded_forces(self,atomlist):
        print("Modifying bonded forces")
        print("")
        #This is typically used by QM/MM object to set bonded forces to zero for qmatoms (atomlist) 
        #Mimicking: https://github.com/openmm/openmm/issues/2792
        
        numharmbondterms_removed=0
        numharmangleterms_removed=0
        numpertorsionterms_removed=0
        numcustomtorsionterms_removed=0
        numcmaptorsionterms_removed=0
        numcmmotionterms_removed=0
        numcustombondterms_removed=0
        
        for force in self.system.getForces():
            if isinstance(force, self.openmm.HarmonicBondForce):
                print("HarmonicBonded force")
                print("There are {} HarmonicBond terms defined.".format(force.getNumBonds()))
                print("")
                #REVISIT: Neglecting QM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                #CURRENT BEHAVIOUR. Keeping QM1-MM1 interaction but removing all QM-QM interactions
                for i in range(force.getNumBonds()):
                    #print("i:", i)
                    p1, p2, length, k = force.getBondParameters(i)
                    #print("p1: {} p2: {} length: {} k: {}".format(p1,p2,length,k))
                    exclude = (p1 in atomlist and p2 in atomlist)
                    #print("exclude:", exclude)
                    if exclude is True:
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before p1: {} p2: {} length: {} k: {}".format(p1,p2,length,k))
                        force.setBondParameters(i, p1, p2, length, 0)
                        numharmbondterms_removed+=1
                        #p1, p2, length, k = force.getBondParameters(i)
                        #print("After p1: {} p2: {} length: {} k: {}".format(p1,p2,length,k))
                        #exit()
                print("Updating force")
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.HarmonicAngleForce):
                print("HarmonicAngle force")
                print("There are {} HarmonicAngle terms defined.".format(force.getNumAngles()))
                for i in range(force.getNumAngles()):
                    p1, p2, p3, angle, k = force.getAngleParameters(i)
                    #Are angle-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3]]
                    #Excluding if 2 or 3 QM atoms. i.e. a QM2-QM1-MM1 or QM3-QM2-QM1 term
                    if presence.count(True) >= 2:
                        #print("presence.count(True):", presence.count(True))
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before p1: {} p2: {} p3: {} angle: {} k: {}".format(p1,p2,p3,angle,k))
                        force.setAngleParameters(i, p1, p2, p3, angle, 0)
                        numharmangleterms_removed+=1
                        #p1, p2, p3, angle, k = force.getAngleParameters(i)
                        #print("After p1: {} p2: {} p3: {} angle: {} k: {}".format(p1,p2,p3,angle,k))
                print("Updating force")
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.PeriodicTorsionForce):
                print("PeriodicTorsionForce force")
                print("There are {} PeriodicTorsionForce terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                    #Are torsion-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3,p4]]
                    #Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    #print("Before p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1,p2,p3,p4,periodicity, phase,k))
                    if presence.count(True) >= 3:
                        print("Found torsion in QM-region")
                        #print("presence.count(True):", presence.count(True))
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1,p2,p3,p4,periodicity, phase,k))
                        force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0)
                        numpertorsionterms_removed+=1
                        p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                        #print("After p1: {} p2: {} p3: {} p4: {} periodicity: {} phase: {} k: {}".format(p1,p2,p3,p4,periodicity, phase,k))
                print("Updating force")
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CustomTorsionForce):
                print("CustomTorsionForce force")
                print("There are {} CustomTorsionForce terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                    #Are torsion-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3,p4]]
                    #Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    #print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                    #print("pars:", pars)
                    if presence.count(True) >= 3:
                        print("Found torsion in QM-region")
                        #print("presence.count(True):", presence.count(True))
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                        force.setTorsionParameters(i, p1, p2, p3, p4, (0.0,0.0))
                        numcustomtorsionterms_removed+=1
                        #p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                        #print("After p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                print("Updating force")
                force.updateParametersInContext(self.simulation.context)
            elif isinstance(force, self.openmm.CMAPTorsionForce):
                print("CMAPTorsionForce force")
                print("There are {} CMAP terms defined.".format(force.getNumTorsions()))
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, a,b,c,d,e = force.getTorsionParameters(i)
                    #Are torsion-atoms in atomlist? 
                    presence=[i in atomlist for i in [p1,p2,p3,p4]]
                    #Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    #print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                    if presence.count(True) >= 3:
                        print("Found torsion in QM-region")
                        #print("presence.count(True):", presence.count(True))
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                        force.setTorsionParameters(i, p1, p2, p3, p4, (0.0,0.0))
                        numcustomtorsionterms_removed+=1
                        #p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                        #print("After p1: {} p2: {} p3: {} p4: {} pars {}".format(p1,p2,p3,p4,pars))
                print("Updating force")
                force.updateParametersInContext(self.simulation.context)
            
            elif isinstance(force, self.openmm.CustomBondForce):
                print("CustomBondForce")
                print("There are {} force terms defined.".format(force.getNumBonds()))
                #Neglecting QM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                for i in range(force.getNumBonds()):
                    #print("i:", i)
                    p1, p2, vars = force.getBondParameters(i)
                    #print("p1: {} p2: {}".format(p1,p2))
                    charge=vars[0];sigma=vars[1];epsilon=vars[2]
                    #print("charge: {} sigma: {} epsilon: {}".format(charge,sigma,epsilon))
                    exclude = (p1 in atomlist or p2 in atomlist)
                    #print("exclude:", exclude)
                    if exclude is True:
                        #print("exclude True")
                        #print("atomlist:", atomlist)
                        #print("i:", i)
                        #print("Before")
                        #print("p1: {} p2: {}")
                        #print("charge: {} sigma: {} epsilon: {}".format(charge,sigma,epsilon))
                        force.setBondParameters(i, p1, p2, [0.0,0.0,0.0])
                        numcustombondterms_removed+=1
                        #p1, p2, vars = force.getBondParameters(i)
                        #charge=vars[0];sigma=vars[1];epsilon=vars[2]
                        #print("p1: {} p2: {}")
                        #print("charge: {} sigma: {} epsilon: {}".format(charge,sigma,epsilon))
                        #exit()
                print("Updating force")
                force.updateParametersInContext(self.simulation.context)
            
            elif isinstance(force, self.openmm.CMMotionRemover):
                print("CMMotionRemover ")
                print("nothing to be done")
            elif isinstance(force, self.openmm.CustomNonbondedForce):
                print("CustomNonbondedForce force")
                print("nothing to be done")
            elif isinstance(force, self.openmm.NonbondedForce):
                print("NonbondedForce force")
                print("nothing to be done")
            else:
                print("Other force: ", force)
                print("nothing to be done")

        print("")
        print("Number of bonded terms removed:", )
        print("Harmonic Bond terms:", numharmbondterms_removed)
        print("Harmonic Angle terms:", numharmangleterms_removed)
        print("Periodic Torsion terms:", numpertorsionterms_removed)
        print("Custom Torsion terms:", numcustomtorsionterms_removed)
        print("CMAP Torsion terms:", numcmaptorsionterms_removed)
        print("CustomBond terms", numcustombondterms_removed)
        print("")

# Simple nonbonded MM theory. Charges and LJ-potentials
class NonBondedTheory:
    def __init__(self, atomtypes=None, forcefield=None, charges = None, LJcombrule='geometric',
                 codeversion='julia', printlevel=2):

        #Printlevel
        self.printlevel=printlevel

        #Atom types
        self.atomtypes=atomtypes
        #Read MM forcefield.
        self.forcefield=forcefield

        #
        self.numatoms = len(self.atomtypes)
        self.LJcombrule=LJcombrule

        #Todo: Delete
        # If qmatoms list passed to Nonbonded theory then we are doing QM/MM
        #self.qmatoms=qmatoms
        #print("Defining Nonbonded Theory")
        #print("qmatoms:", self.qmatoms)

        self.codeversion=codeversion

        #These are charges for whole system including QM.
        self.atom_charges = charges
        #Possibly have self.mm_charges here also??


        #Initializing sigmaij and epsij arrays. Will be filled by calculate_LJ_pairpotentials
        self.sigmaij=np.zeros((self.numatoms, self.numatoms))
        self.epsij=np.zeros((self.numatoms, self.numatoms))
        self.pairarrays_assigned = False

    #Todo: Need to make active-region version of pyarray version here.
    def calculate_LJ_pairpotentials(self, qmatoms=None, actatoms=None, frozenatoms=None):

        #actatoms
        if actatoms is None:
            actatoms=[]
        if frozenatoms is None:
            frozenatoms=[]

        #Deleted combination_rule argument. Now using variable assigned to object

        combination_rule=self.LJcombrule

        #If qmatoms passed list passed then QM/MM and QM-QM pairs will be ignored from pairlist
        if self.printlevel >= 2:
            print("Inside calculate_LJ_pairpotentials")
        #Todo: Figure out if we can find out if qmatoms without being passed
        if qmatoms is None or qmatoms == []:
            qmatoms = []
            print("WARNING: qmatoms list is empty.")
            print("This is fine if this is a pure MM job.")
            print("If QM/MM job, then qmatoms list should be passed to NonBonded theory.")


        import math
        if self.printlevel >= 2:
            print("Defining Lennard-Jones pair potentials")

        #List to store pairpotentials
        self.LJpairpotentials=[]
        #New: multi-key dict instead using tuple
        self.LJpairpotdict={}
        if combination_rule == 'geometric':
            if self.printlevel >= 2:
                print("Using geometric mean for LJ pair potentials")
        elif combination_rule == 'arithmetic':
            if self.printlevel >= 2:
                print("Using geometric mean for LJ pair potentials")
        elif combination_rule == 'mixed_geoepsilon':
            if self.printlevel >= 2:
                print("Using mixed rule for LJ pair potentials")
                print("Using arithmetic rule for r/sigma")
                print("Using geometric rule for epsilon")
        elif combination_rule == 'mixed_geosigma':
            if self.printlevel >= 2:
                print("Using mixed rule for LJ pair potentials")
                print("Using geometric rule for r/sigma")
                print("Using arithmetic rule for epsilon")
        else:
            print("Unknown combination rule. Exiting")
            exit()

        #A large system has many atomtypes. Creating list of unique atomtypes to simplify loop
        CheckpointTime = time.time()
        self.uniqatomtypes = np.unique(self.atomtypes).tolist()
        DoAll=True
        for count_i, at_i in enumerate(self.uniqatomtypes):
            #print("count_i:", count_i)
            for count_j,at_j in enumerate(self.uniqatomtypes):
                #if count_i < count_j:
                if DoAll==True:
                    #print("at_i {} and at_j {}".format(at_i,at_j))
                    #Todo: if atom type not in dict we get a KeyError here.
                    # Todo: Add exception or add zero-entry to dict ??
                    if len(self.forcefield[at_i].LJparameters) == 0:
                        continue
                    if len(self.forcefield[at_j].LJparameters) == 0:
                        continue
                    if self.printlevel >= 3:
                        print("LJ sigma_i {} for atomtype {}:".format(self.forcefield[at_i].LJparameters[0], at_i))
                        print("LJ sigma_j {} for atomtype {}:".format(self.forcefield[at_j].LJparameters[0], at_j))
                        print("LJ eps_i {} for atomtype {}:".format(self.forcefield[at_i].LJparameters[1], at_i))
                        print("LJ eps_j {} for atomtype {}:".format(self.forcefield[at_j].LJparameters[1], at_j))
                        blankline()
                    if combination_rule=='geometric':
                        sigma=math.sqrt(self.forcefield[at_i].LJparameters[0]*self.forcefield[at_j].LJparameters[0])
                        epsilon=math.sqrt(self.forcefield[at_i].LJparameters[1]*self.forcefield[at_j].LJparameters[1])
                        if self.printlevel >=3:
                            print("LJ sigma_ij : {} for atomtype-pair: {} {}".format(sigma,at_i, at_j))
                            print("LJ epsilon_ij : {} for atomtype-pair: {} {}".format(epsilon,at_i, at_j))
                            blankline()
                    elif combination_rule=='arithmetic':
                        if self.printlevel >=3:
                            print("Using arithmetic mean for LJ pair potentials")
                            print("NOTE: to be confirmed")
                        sigma=0.5*(self.forcefield[at_i].LJparameters[0]+self.forcefield[at_j].LJparameters[0])
                        epsilon=0.5-(self.forcefield[at_i].LJparameters[1]+self.forcefield[at_j].LJparameters[1])
                    elif combination_rule=='mixed_geosigma':
                        if self.printlevel >=3:
                            print("Using geometric mean for LJ sigma parameters")
                            print("Using arithmetic mean for LJ epsilon parameters")
                            print("NOTE: to be confirmed")
                        sigma=math.sqrt(self.forcefield[at_i].LJparameters[0]*self.forcefield[at_j].LJparameters[0])
                        epsilon=0.5-(self.forcefield[at_i].LJparameters[1]+self.forcefield[at_j].LJparameters[1])
                    elif combination_rule=='mixed_geoepsilon':
                        if self.printlevel >=3:
                            print("Using arithmetic mean for LJ sigma parameters")
                            print("Using geometric mean for LJ epsilon parameters")
                            print("NOTE: to be confirmed")
                        sigma=0.5*(self.forcefield[at_i].LJparameters[0]+self.forcefield[at_j].LJparameters[0])
                        epsilon=math.sqrt(self.forcefield[at_i].LJparameters[1]*self.forcefield[at_j].LJparameters[1])
                    self.LJpairpotentials.append([at_i, at_j, sigma, epsilon])
                    #Dict using two keys (actually a tuple of two keys)
                    self.LJpairpotdict[(at_i,at_j)] = [sigma, epsilon]
                    #print(self.LJpairpotentials)
        #Takes not time so disabling time-printing
        #print_time_rel(CheckpointTime, modulename="pairpotentials")
        #Remove redundant pair potentials
        CheckpointTime = time.time()
        for acount, pairpot_a in enumerate(self.LJpairpotentials):
            for bcount, pairpot_b in enumerate(self.LJpairpotentials):
                if acount < bcount:
                    if set(pairpot_a) == set(pairpot_b):
                        del self.LJpairpotentials[bcount]
        if self.printlevel >= 2:
            #print("Final LJ pair potentials (sigma_ij, epsilon_ij):\n", self.LJpairpotentials)
            print("New: LJ pair potentials as dict:")
            print("self.LJpairpotdict:", self.LJpairpotdict)

        #Create numatomxnumatom array of eps and sigma
        blankline()
        if self.printlevel >= 2:
            print("Creating epsij and sigmaij arrays ({},{})".format(self.numatoms,self.numatoms))
            print("Will skip QM-QM ij pairs for qmatoms: ", qmatoms)
            print("Will skip frozen-frozen ij pairs")
            print("len actatoms:", len(actatoms))
            print("len frozenatoms:", len(frozenatoms))
        beginTime = time.time()

        CheckpointTime = time.time()
        # See speed-tests at /home/bjornsson/pairpot-test

        if self.codeversion=="julia":
            if self.printlevel >= 2:
                print("Using PyJulia for fast sigmaij and epsij array creation")
            # Necessary for statically linked libpython
            try:
                from julia.api import Julia
                from julia import Main
            except:
                print("Problem importing Pyjulia (import julia)")
                print("Make sure Julia is installed and PyJulia module available")
                print("Also, are you using python-jl ?")
                print("Alternatively, use codeversion='py' argument to NonBondedTheory to use slower Python version for array creation")
                exit(9)


            # Do pairpot array for whole system
            if len(actatoms) == 0:
                print("Calculating pairpotential array for whole system")
                self.sigmaij, self.epsij = Main.Juliafunctions.pairpot_full(self.numatoms, self.atomtypes, self.LJpairpotdict,qmatoms)
            else:
            #    #or only for active region
                print("Calculating pairpotential array for active region only")
                #pairpot_active(numatoms,atomtypes,LJpydict,qmatoms,actatoms)
                print("self.numatoms", self.numatoms)
                print("self.atomtypes", self.atomtypes)
                print("self.LJpairpotdict", self.LJpairpotdict)
                print("qmatoms", qmatoms)
                print("actatoms", actatoms)
                
                self.sigmaij, self.epsij = Main.Juliafunctions.pairpot_active(self.numatoms, self.atomtypes, self.LJpairpotdict, qmatoms, actatoms)
        # New for-loop for creating sigmaij and epsij arrays. Uses dict-lookup instead
        elif self.codeversion=="py":
            if self.printlevel >= 2:
                print("Using Python version for array creation")
                print("Does not yet skip frozen-frozen atoms...to be fixed")
                #Todo: add frozen-frozen atoms skip
            #Update: Only doing half of array
            for i in range(self.numatoms):
                for j in range(i+1, self.numatoms):
                    #Skipping if i-j pair in qmatoms list. I.e. not doing QM-QM LJ calc.
                    #if all(x in qmatoms for x in (i, j)) == True:
                    #
                    if i in qmatoms and j in qmatoms:
                        #print("Skipping i-j pair", i,j, " as these are QM atoms")
                        continue
                    elif (self.atomtypes[i], self.atomtypes[j]) in self.LJpairpotdict:
                        self.sigmaij[i, j] = self.LJpairpotdict[(self.atomtypes[i], self.atomtypes[j])][0]
                        self.epsij[i, j] = self.LJpairpotdict[(self.atomtypes[i], self.atomtypes[j])][1]
                    elif (atomtypes[j], atomtypes[i]) in self.LJpairpotdict:
                        self.sigmaij[i, j] = self.LJpairpotdict[(self.atomtypes[j], self.atomtypes[i])][0]
                        self.epsij[i, j] = self.LJpairpotdict[(self.atomtypes[j], self.atomtypes[i])][1]
        else:
            print("unknown codeversion")
            exit()

        if self.printlevel >= 2:
            #print("self.sigmaij ({}) : {}".format(len(self.sigmaij), self.sigmaij))
            #print("self.epsij ({}) : {}".format(len(self.epsij), self.epsij))
            print("sigmaij size: {}".format(len(self.sigmaij)))
            print("epsij size: {}".format(len(self.epsij)))
        print_time_rel(CheckpointTime, modulename="pairpot arrays")
        self.pairarrays_assigned = True

    def update_charges(self,atomlist,charges):
        print("Updating charges.")
        assert len(atomlist) == len(charges)
        for atom,charge in zip(atomlist,charges):
            self.atom_charges[atom] = charge
        #print("Charges are now:", charges)
        print("Sum of charges:", sum(charges))

    # current_coords is now used for full_coords, charges for full coords
    def run(self, current_coords=None, elems=None, charges=None, connectivity=None,
            Coulomb=True, Grad=True, qmatoms=None, actatoms=None, frozenatoms=None):

        if current_coords is None:
            print("No current_coords argument. Exiting...")
            exit()
        CheckpointTime = time.time()
        #If qmatoms list provided to run (probably by QM/MM object) then we are doing QM/MM
        #QM-QM pairs will be skipped in LJ

        #Testing if arrays assigned or not. If not calling calculate_LJ_pairpotentials
        #Passing qmatoms over so pairs can be skipped
        #This sets self.sigmaij and self.epsij and also self.LJpairpotentials
        #Todo: if actatoms have been defined this will be skipped in pairlist creation
        #if frozenatoms passed frozen-frozen interactions will be skipped
        if self.pairarrays_assigned is False:
            print("Calling LJ pairpot calc")
            self.calculate_LJ_pairpotentials(qmatoms=qmatoms,actatoms=actatoms)
        else:
            print("LJ pairpot arrays exist...")

        if len(self.LJpairpotentials) > 0:
            LJ=True
        #If charges not provided to run function. Use object charges
        if charges == None:
            charges=self.atom_charges

        #If coords not provided to run function. Use object coords
        #HMM. I guess we are not keeping coords as part of MMtheory?
        #if len(full_cords)==0:
        #    full_coords=

        if self.printlevel >= 2:
            print(BC.OKBLUE, BC.BOLD, "------------RUNNING NONBONDED MM CODE-------------", BC.END)
            print("Calculating MM energy and gradient")
        #initializing
        self.Coulombchargeenergy=0
        self.LJenergy=0
        self.MMGradient=[]
        self.Coulombchargegradient=[]
        self.LJgradient=[]

        #Slow Python version
        if self.codeversion=='py':
            if self.printlevel >= 2:
                print("Using slow Python MM code")
            #Sending full coords and charges over. QM charges are set to 0.
            if Coulomb==True:
                self.Coulombchargeenergy, self.Coulombchargegradient  = coulombcharge(charges, current_coords)
                if self.printlevel >= 2:
                    print("Coulomb Energy (au):", self.Coulombchargeenergy)
                    print("Coulomb Energy (kcal/mol):", self.Coulombchargeenergy * constants.harkcal)
                    print("")
                    #print("self.Coulombchargegradient:", self.Coulombchargegradient)
                blankline()
            # NOTE: Lennard-Jones should  calculate both MM-MM and QM-MM LJ interactions. Full coords necessary.
            if LJ==True:
                #LennardJones(coords, epsij, sigmaij, connectivity=[], qmatoms=[])
                #self.LJenergy,self.LJgradient = LennardJones(full_coords,self.atomtypes, self.LJpairpotentials, connectivity=connectivity)
                self.LJenergy,self.LJgradient = LennardJones(current_coords,self.epsij,self.sigmaij)
                #print("Lennard-Jones Energy (au):", self.LJenergy)
                #print("Lennard-Jones Energy (kcal/mol):", self.LJenergy*constants.harkcal)
            self.MMEnergy = self.Coulombchargeenergy+self.LJenergy
            if Grad==True:
                self.MMGradient = self.Coulombchargegradient+self.LJgradient
        #Combined Coulomb+LJ Python version. Slow
        elif self.codeversion=='py_comb':
            print("not active")
            exit()
            self.MMenergy, self.MMgradient = LJCoulpy(current_coords, self.atomtypes, charges, self.LJpairpotentials,
                                                          connectivity=connectivity)
        elif self.codeversion=='f2py':
            if self.printlevel >= 2:
                print("Using Fortran F2Py MM code")
            try:
                #print(os.environ.get("LD_LIBRARY_PATH"))
                import LJCoulombv1
            except:
                print("Fortran library LJCoulombv1 not found! Make sure you have run the installation script.")
            self.MMEnergy, self.MMGradient, self.LJenergy, self.Coulombchargeenergy =\
                LJCoulomb(current_coords, self.epsij, self.sigmaij, charges, connectivity=connectivity)
        elif self.codeversion=='f2pyv2':
            if self.printlevel >= 2:
                print("Using fast Fortran F2Py MM code v2")
            try:
                import LJCoulombv2
                print(LJCoulombv2.__doc__)
                print("----------")
            except:
                print("Fortran library LJCoulombv2 not found! Make sure you have run the installation script.")
            self.MMEnergy, self.MMGradient, self.LJenergy, self.Coulombchargeenergy =\
                LJCoulombv2(current_coords, self.epsij, self.sigmaij, charges, connectivity=connectivity)
        elif self.codeversion=='julia':
            if self.printlevel >= 2:
                print("Using fast Julia version, v1")
            # Necessary for statically linked libpython
            try:
                from julia.api import Julia
                from julia import Main
            except:
                print("Problem importing Pyjulia (import julia)")
                print("Make sure Julia is installed and PyJulia module available")
                print("Also, are you using python-jl ?")
                print("Alternatively, use codeversion='py' argument to NonBondedTheory to use slower Python version for array creation")
                exit(9)
            print_time_rel(CheckpointTime, modulename="from run to just before calling ")
            self.MMEnergy, self.MMGradient, self.LJenergy, self.Coulombchargeenergy =\
                Main.Juliafunctions.LJcoulombchargev1c(charges, current_coords, self.epsij, self.sigmaij, connectivity)
            print_time_rel(CheckpointTime, modulename="from run to done julia")
        else:
            print("Unknown version of MM code")
            exit(1)

        if self.printlevel >= 2:
            print("Lennard-Jones Energy (au):", self.LJenergy)
            print("Lennard-Jones Energy (kcal/mol):", self.LJenergy * constants.harkcal)
            print("Coulomb Energy (au):", self.Coulombchargeenergy)
            print("Coulomb Energy (kcal/mol):", self.Coulombchargeenergy * constants.harkcal)
            print("MM Energy:", self.MMEnergy)
        if self.printlevel >= 3:
            print("self.MMGradient:", self.MMGradient)

        if self.printlevel >= 2:
            print(BC.OKBLUE, BC.BOLD, "------------ENDING NONBONDED MM CODE-------------", BC.END)
        return self.MMEnergy, self.MMGradient

#Polarizable Embedding theory object.
#Required at init: qm_theory and qmatoms, X, Y
#Currently only Polarizable Embedding (PE). Only available for Psi4, PySCF and Dalton.
#Peatoms: polarizable atoms. MMatoms: nonpolarizable atoms (e.g. TIP3P)
class PolEmbedTheory:
    def __init__(self, fragment=None, qm_theory=None, qmatoms=None, peatoms=None, mmatoms=None, pot_create=True,
                 potfilename='System', pot_option=None, pyframe=False, PElabel_pyframe='MM', daltondir=None, pdbfile=None):
        print(BC.WARNING,BC.BOLD,"------------Defining PolEmbedTheory object-------------", BC.END)
        self.pot_create=pot_create
        self.pyframe=pyframe
        self.pot_option=pot_option
        self.PElabel_pyframe = PElabel_pyframe
        self.potfilename = potfilename
        #Theory level definitions
        allowed_qmtheories=['Psi4Theory', 'PySCFTheory', 'DaltonTheory']
        self.qm_theory=qm_theory
        self.qm_theory_name = self.qm_theory.__class__.__name__
        if self.qm_theory_name in allowed_qmtheories:
            print(BC.OKGREEN, "QM-theory:", self.qm_theory_name, "is supported in Polarizable Embedding", BC.END)
        else:
            print(BC.FAIL, "QM-theory:", self.qm_theory_name, "is  NOT supported in Polarizable Embedding", BC.END)

        if self.pot_option=='LoProp':
            if daltondir is None:
                print("LoProp option chosen. This requires daltondir variable")
                exit()


        if pdbfile is not None:
            print("PDB file provided, will use residue information")

        # Region definitions
        if qmatoms is None:
            self.qmatoms = []
        else:
            self.qmatoms=qmatoms
        if peatoms is None:
            self.peatoms = []
        else:
            self.peatoms=peatoms
        if mmatoms is None:
            print("WARNING...mmatoms list is empty...")
            self.mmatoms = []
        else:
            self.mmatoms=mmatoms

        #If fragment object has been defined
        if fragment is not None:
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
            self.connectivity=fragment.connectivity

            self.allatoms = list(range(0, len(self.elems)))
            self.qmcoords=[self.coords[i] for i in self.qmatoms]
            self.qmelems=[self.elems[i] for i in self.qmatoms]
            self.pecoords=[self.coords[i] for i in self.peatoms]
            self.peelems=[self.elems[i] for i in self.peatoms]
            self.mmcoords=[self.coords[i] for i in self.mmatoms]
            self.mmelems=[self.elems[i] for i in self.mmatoms]

            #print("List of all atoms:", self.allatoms)
            print("System size:", len(self.allatoms))
            print("QM region size:", len(self.qmatoms))
            print("PE region size", len(self.peatoms))
            print("MM region size", len(self.mmatoms))
            blankline()

            #Creating list of QM, PE, MM labels used by reading residues in PDB-file
            #Also making residlist necessary
            #TODO: This needs to be rewritten, only applies to water-solvent
            self.hybridatomlabels=[]
            self.residlabels=[]
            count=2
            rescount=0
            for i in self.allatoms:
                if i in self.qmatoms:
                    print("i : {} in qmatoms".format(i))
                    self.hybridatomlabels.append('QM')
                    self.residlabels.append(1)
                elif i in self.peatoms:
                    print("i : {} in peatoms".format(i))
                    self.hybridatomlabels.append(self.PElabel_pyframe)
                    self.residlabels.append(count)
                    rescount+=1
                elif i in self.mmatoms:
                    #print("i : {} in mmatoms".format(i))
                    self.hybridatomlabels.append('WAT')
                    self.residlabels.append(count)
                    rescount+=1
                if rescount==3:
                    count+=1
                    rescount=0

        print("self.hybridatomlabels:", self.hybridatomlabels)
        print("self.residlabels:", self.residlabels)
        #Create Potential file here. Usually true.
        if self.pot_create==True:
            print("Potfile Creation is on!")
            if self.pyframe==True:
                print("Using PyFrame")
                try:
                    import pyframe
                    print("PyFrame found")
                except:
                    print("Pyframe not found. Install pyframe via pip (https://pypi.org/project/PyFraME):")
                    print("pip install pyframe")
                    exit(9)
                #Create dummy pdb-file if PDB-file not provided
                if pdbfile is None:
                    write_pdbfile_dummy(self.elems, self.coords, self.potfilename, self.hybridatomlabels, self.residlabels)
                    file=self.potfilename+'.pdb'
                #Pyframe
                if self.pot_option=='SEP':
                    print("Pot option: SEP")
                    system = pyframe.MolecularSystem(input_file=file)
                    solventPol = system.get_fragments_by_name(names=[self.PElabel_pyframe])
                    solventNonPol = system.get_fragments_by_name(names=['WAT'])
                    system.add_region(name='solventpol', fragments=solventPol, use_standard_potentials=True,
                          standard_potential_model='SEP')
                    system.add_region(name='solventnonpol', fragments=solventNonPol, use_standard_potentials=True,
                          standard_potential_model='TIP3P')
                    project = pyframe.Project()
                    project.create_embedding_potential(system)
                    project.write_potential(system)
                    self.potfile=self.potfilename+'.pot'
                    print("Created potfile: ", self.potfile)
                elif self.pot_option=='TIP3P':
                    #Not sure if we use this much or at all. Needs to be checked.
                    print("Pot option: TIP3P")
                    system = pyframe.MolecularSystem(input_file=file)
                    solvent = system.get_fragments_by_name(names=['WAT'])
                    system.add_region(name='solvent', fragments=solvent, use_standard_potentials=True,
                          standard_potential_model='TIP3P')
                    project = pyframe.Project()
                    project.create_embedding_potential(system)
                    project.write_potential(system)
                    self.potfile=self.potfilename+'.pot'
                    print("Created potfile: ", self.potfile)
                #RB. TEST. Protein system using standard potentials
                elif self.pot_option=='Protein-SEP':
                    file=pdbfile
                    print("Pot option: Protein-SEP")
                    exit()
                    system = pyframe.MolecularSystem(input_file=file)
                    Polregion = system.get_fragments_by_name(names=[self.PElabel_pyframe])
                    NonPolregion = system.get_fragments_by_name(names=['WAT'])
                    system.add_region(name='solventpol', fragments=solventPol, use_standard_potentials=True,
                          standard_potential_model='SEP')
                    system.add_region(name='solventnonpol', fragments=solventNonPol, use_standard_potentials=True,
                          standard_potential_model='TIP3P')
                    project = pyframe.Project()
                    project.create_embedding_potential(system)
                    project.write_potential(system)
                    self.potfile=self.potfilename+'.pot'
                    print("Created potfile: ", self.potfile)

                elif self.pot_option=='LoProp':
                    print("Pot option: LoProp")
                    print("Note: dalton and loprop binaries need to be in shell PATH before running.")
                    #os.environ['PATH'] = daltondir + ':'+os.environ['PATH']
                    #print("Current PATH is:", os.environ['PATH'])
                    #TODO: Create pot file from scratch. Requires LoProp and Dalton I guess
                    system = pyframe.MolecularSystem(input_file=file)
                    core = system.get_fragments_by_name(names=['QM'])
                    system.set_core_region(fragments=core, program='Dalton', basis='pcset-1')
                    # solvent = system.get_fragments_by_distance(reference=core, distance=4.0)
                    solvent = system.get_fragments_by_name(names=[self.PElabel_pyframe])
                    system.add_region(name='solvent', fragments=solvent, use_mfcc=True, use_multipoles=True, 
                                      multipole_order=2, multipole_model='LoProp', multipole_method='DFT', multipole_xcfun='PBE0',
                                      multipole_basis='loprop-6-31+G*', use_polarizabilities=True, polarizability_model='LoProp',
                                      polarizability_method='DFT', polarizability_xcfun='PBE0', polarizability_basis='loprop-6-31+G*')
                    project = pyframe.Project()
                    print("Creating embedding potential")
                    project.create_embedding_potential(system)
                    project.write_core(system)
                    project.write_potential(system)
                    self.potfile=self.potfilename+'.pot'
                    print("Created potfile (via Dalton and LoProp): ", self.potfile)
                else:
                    print("Invalid option")
                    exit()
                #Copying pyframe-created potfile from dir:
                shutil.copyfile(self.potfilename+'/' + self.potfilename+'.pot', './'+self.potfilename+'.pot')

            #Todo: Manual potential file creation. Maybe only if pyframe is buggy
            else:
                print("Manual potential file creation (instead of Pyframe)")
                print("Not ready yet!")
                if self.pot_option == 'SEP':
                    numatomsolvent = 3
                    Ocharge = -0.67444000
                    Hcharge = 0.33722000
                    Opolz = 5.73935000
                    Hpolz = 2.30839000
                    numpeatoms=len(self.peatoms)
                    with open('System' + '.pot', 'w') as potfile:
                        potfile.write('! Generated by Pot-Gen-RB\n')
                        potfile.write('@COORDINATES\n')
                        potfile.write(str(numpeatoms) + '\n')
                        potfile.write('AA\n')
                        for i in range(0, numpeatoms):
                            atom = self.peatoms[i]
                            c = self.pecoords[i]
                            potfile.write(
                                atom + '   ' + str(c[0]) + '   ' + str(c[1]) + '   ' + str(c[2]) + '   ' + str(
                                    i + 1) + '\n')
                        potfile.write('@MULTIPOLES\n')
                        # Assuming simple pointcharge here. To be extended
                        potfile.write('ORDER 0\n')
                        potfile.write(str(numpeatoms) + '\n')
                        for i in range(0, numpeatoms):
                            atom = self.peatoms[i]
                            if atom == 'O':
                                SPcharge = Ocharge
                            elif atom == 'H':
                                SPcharge = Hcharge
                            potfile.write(str(i + 1) + '   ' + str(SPcharge) + '\n')
                        potfile.write('@POLARIZABILITIES\n')
                        potfile.write('ORDER 1 1\n')
                        potfile.write(str(numpeatoms) + '\n')
                        for i in range(0, numpeatoms):
                            atom = self.peatoms[i]
                            if atom == 'O':
                                SPpolz = Opolz
                            elif atom == 'H':
                                SPpolz = Hpolz
                            potfile.write(str(i + 1) + '    ' + str(SPpolz) + '   0.0000000' + '   0.0000000    ' + str(
                                SPpolz) + '   0.0000000    ' + str(SPpolz) + '\n')
                        potfile.write('EXCLISTS\n')
                        potfile.write(str(numpeatoms) + ' 3\n')
                        for j in range(1, numpeatoms, numatomsolvent):
                            potfile.write(str(j) + ' ' + str(j + 1) + ' ' + str(j + 2) + '\n')
                            potfile.write(str(j + 1) + ' ' + str(j) + ' ' + str(j + 2) + '\n')
                            potfile.write(str(j + 2) + ' ' + str(j) + ' ' + str(j + 1) + '\n')

                else:
                    print("Other pot options not yet available")
                    exit()
        else:
            print("Pot creation is off for this object. Assuming potfile has been provided")
            self.potfile=potfilename+'.pot'

    def run(self, current_coords=None, elems=None, Grad=False, nprocs=1, potfile=None, restart=False):
        print(BC.WARNING, BC.BOLD, "------------RUNNING PolEmbedTheory MODULE-------------", BC.END)
        if restart==True:
            print("Restart Option On!")
        else:
            print("Restart Option Off!")
        print("QM Module:", self.qm_theory_name)

        #Check if potfile provide to run (rare use). If not, use object file
        if potfile is not None:
            self.potfile=potfile

        print("Using potfile:", self.potfile)

        #If no coords provided to run (from Optimizer or NumFreq or MD) then use coords associated with object.
        #if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords=self.coords

        #Updating QM coords and MM coords.
        #TODO: Should we use different name for updated QMcoords and MMcoords here??
        self.qmcoords=[current_coords[i] for i in self.qmatoms]

        if self.qm_theory_name == "Psi4Theory":
            #Calling Psi4 theory, providing current QM and MM coordinates.
            #Currently doing SP case only without Grad

            self.QMEnergy = self.qm_theory.run(current_coords=self.qmcoords, qm_elems=self.qmelems, Grad=False,
                                               nprocs=nprocs, pe=True, potfile=self.potfile, restart=restart)
        elif self.qm_theory_name == "DaltonTheory":
            print("self.potfile:", self.potfile)
            self.QMEnergy = self.qm_theory.run(current_coords=self.qmcoords, qm_elems=self.qmelems, Grad=False,
                                               nprocs=nprocs, pe=True, potfile=self.potfile, restart=restart)
        elif self.qm_theory_name == "PySCFTheory":
            self.QMEnergy = self.qm_theory.run(current_coords=self.qmcoords, qm_elems=self.qmelems, Grad=False,
                                               nprocs=nprocs, pe=True, potfile=self.potfile, restart=restart)

        elif self.qm_theory_name == "ORCATheory":
            print("not available for ORCATheory")
            exit()

        elif self.qm_theory_name == "NWChemTheory":
            print("not available for NWChemTheory")
            exit()
        else:
            print("invalid QM theory")
            exit()

        #Todo: self.MM_Energy from PolEmbed calc?
        self.MMEnergy=0
        #Final QM/MM Energy
        self.PolEmbedEnergy = self.QMEnergy+self.MMEnergy
        self.energy=self.PolEmbedEnergy
        blankline()
        print("{:<20} {:>20.12f}".format("QM energy: ",self.QMEnergy))
        print("{:<20} {:>20.12f}".format("MM energy: ", self.MMEnergy))
        print("{:<20} {:>20.12f}".format("PolEmbed energy: ", self.PolEmbedEnergy))
        blankline()
        return self.PolEmbedEnergy


#QM/MM theory object.
#Required at init: qm_theory and qmatoms. Fragment not. Can come later
#TODO NOTE: If we add init arguments, remember to update Numfreq QMMM option as it depends on the keywords
class QMMMTheory:
    def __init__(self, qm_theory=None, qmatoms=None, fragment=None, mm_theory=None , charges=None,
                 embedding="Elstat", printlevel=2, nprocs=1, actatoms=None, frozenatoms=None):

        print(BC.WARNING,BC.BOLD,"------------Defining QM/MM object-------------", BC.END)

        #Linkatoms False by default. Later checked.
        self.linkatoms=False

        #If fragment object has been defined
        #This probably needs to be always true
        if fragment is not None:
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
            self.connectivity=fragment.connectivity

            # Region definitions
            self.allatoms=list(range(0,len(self.elems)))
            print("All atoms in fragment:", len(self.allatoms))
            #Sorting qmatoms list
            self.qmatoms = sorted(qmatoms)
            self.mmatoms=listdiff(self.allatoms,self.qmatoms)

            # FROZEN AND ACTIVE ATOMS
            if actatoms is None and frozenatoms is None:
                print("Actatoms/frozenatoms list not passed to QM/MM object. Will do all frozen interactions in MM (expensive).")
                print("All {} atoms active, no atoms frozen".format(len(self.allatoms)))
                self.actatoms=self.allatoms
                self.frozenatoms=[]
            elif actatoms is not None and frozenatoms is None:
                print("Actatoms list passed to QM/MM object. Will skip all frozen interactions in MM.")
                #Sorting actatoms list
                self.actatoms = sorted(actatoms)
                self.frozenatoms = listdiff(self.allatoms, self.actatoms)
                print("{} active atoms, {} frozen atoms".format(len(self.actatoms), len(self.frozenatoms)))
            elif frozenatoms is not None and actatoms is None:
                print("Frozenatoms list passed to QM/MM object. Will skip all frozen interactions in MM.")
                self.frozenatoms = sorted(frozenatoms)
                self.actatoms = listdiff(self.allatoms, self.frozenatoms)
                print("{} active atoms, {} frozen atoms".format(len(self.actatoms), len(self.frozenatoms)))
            else:
                print("active_atoms and frozen_atoms can not be both defined")
                exit(1)
            
            #print("List of all atoms:", self.allatoms)
            print("QM region ({} atoms): {}".format(len(self.qmatoms),self.qmatoms))
            print("MM region ({} atoms)".format(len(self.mmatoms)))
            #print("MM region", self.mmatoms)
            blankline()

            #List of QM and MM labels
            self.hybridatomlabels=[]
            for i in self.allatoms:
                if i in self.qmatoms:
                    self.hybridatomlabels.append('QM')
                elif i in self.mmatoms:
                    self.hybridatomlabels.append('MM')
        else:
            print("Fragment has not been defined for QM/MM. Exiting")
            exit(1)

        #Flag to check whether QMCharges have been zeroed in self.charges_qmregionzeroed list
        self.QMChargesZeroed=False

        #Theory level definitions
        self.printlevel=printlevel
        self.qm_theory=qm_theory
        self.qm_theory_name = self.qm_theory.__class__.__name__
        
        #Setting QM/MM qmatoms in QMtheory also (used for Spin-flipping currently)
        self.qm_theory.qmatoms=self.qmatoms
        
        self.mm_theory=mm_theory
        self.mm_theory_name = self.mm_theory.__class__.__name__
        if self.mm_theory_name == "str":
            self.mm_theory_name="None"
        print("QM-theory:", self.qm_theory_name)
        print("MM-theory:", self.mm_theory_name)
        
        #Setting nprocs of object.
        #This will be when calling QMtheory and probably MMtheory
        
        #nproc-setting in QMMMTheory takes precedent
        if nprocs != 1:
            self.nprocs=nprocs
        #If QMtheory nprocs was set (and QMMMTHeory not)
        elif self.qm_theory.nprocs != 1:
            self.nprocs=self.qm_theory.nprocs
        #Default 1 proc
        else:
            self.nprocs=1
        print("QM/MM object selected to use {} cores".format(self.nprocs))

        #Embedding type: mechanical, electrostatic etc.
        self.embedding=embedding
        print("Embedding:", self.embedding)

        #if atomcharges are not passed to QMMMTheory object, get them from MMtheory (that should have been defined then)
        if charges is None:
            print("No atomcharges list passed to QMMMTheory object")
            self.charges=[]
            if self.mm_theory_name == "OpenMMTheory":
                print("Getting system charges from OpenMM object")
                #Todo: Call getatomcharges directly or should that have been called from within openmm object at init ?
                #self.charges = mm_theory.getatomcharges()
                self.charges = mm_theory.charges
            elif self.mm_theory_name == "NonBondedTheory":
                print("Getting system charges from NonBondedTheory object")
                #Todo: normalize charges vs atom_charges
                self.charges=mm_theory.atom_charges
                        
            else:
                print("Unrecognized MM theory for QMMMTheory")
                exit(1)
        else:
            self.charges=charges
        
        if len(self.charges) == 0:
            print("No charges present in QM/MM object. Exiting...")
            exit()
        
        
        #CHARGES DEFINED FOR OBJECT:
        #Self.charges are original charges that are defined above (on input, from OpenMM or from NonBondedTheory)
        #self.charges_qmregionzeroed is self.charges but with 0-value for QM-atoms
        #self.pointcharges are pointcharges that the QM-code will see (dipole-charges, no zero-valued charges etc)
        #Length of self.charges: system size
        #Length of self.charges_qmregionzeroed: system size
        #Length of self.pointcharges: unknown. does not contain zero-valued charges (e.g. QM-atoms etc.), contains dipole-charges 
        
        #self.charges_qmregionzeroed will have QM-charges zeroed (but not removed)
        self.charges_qmregionzeroed = []
        
        #Self.pointcharges are pointcharges that the QM-program will see (but not the MM program)
        # They have QM-atoms zeroed, zero-charges removed, dipole-charges added etc.
        #Defined later
        self.pointcharges = []

        #If MM THEORY (not just pointcharges)
        if mm_theory is not None:
            #Add possible exception for QM-QM atoms here.
            #Maybe easier to just just set charges to 0. LJ for QM-QM still needs to be done by MM code
            if self.mm_theory_name == "OpenMMTheory":
                print("Now adding exceptions for frozen atoms")
                if len(self.frozenatoms) > 0:
                    print("Here adding exceptions for OpenMM")
                    print("Frozen-atom exceptions currently inactive...")
                    #print("Num frozen atoms: ", len(self.frozenatoms))
                    #Disabling for now, since so bloody slow. Need to speed up
                    #mm_theory.addexceptions(self.frozenatoms)


            #Check if we need linkatoms by getting boundary atoms dict:
            blankline()
            self.boundaryatoms = get_boundary_atoms(self.qmatoms, self.coords, self.elems, settings_ash.scale, settings_ash.tol)
            
            if len(self.boundaryatoms) >0:
                print("Found covalent QM-MM boundary. Linkatoms option set to True")
                print("Boundaryatoms (QM:MM pairs):", self.boundaryatoms)
                self.linkatoms=True
                
                #Get MM boundary information. Stored as self.MMboundarydict
                self.get_MMboundary()
            else:
                print("No covalent QM-MM boundary. Linkatoms option set to False")
                self.linkatoms=False
            

            if self.embedding=="Elstat":
                
                #Remove bonded interactions in MM part. Only in OpenMM. Assuming they were never defined in NonbondedTHeory
                
                if self.mm_theory_name == "OpenMMTheory":
                    print("Removing bonded terms for QM-region in MMtheory")
                    self.mm_theory.modify_bonded_forces(self.qmatoms)

                    #NOTE: Temporary. Exceptions for nonbonded QM atoms. Will ignore QM-QM Coulomb and LJ interactions. Coulomb interactions are also set to zero elsewhere.
                    print("Removing nonbonded terms for QM-region in MMtheory")
                    self.mm_theory.addexceptions(self.qmatoms)
                
                #Change charges
                # Keeping self.charges as originally defined.
                #Setting QM charges to 0 since electrostatic embedding
                #and Charge-shift QM-MM boundary
                
                #Zero QM charges
                #TODO: DO here or inside run instead?? Needed for MM code.
                self.ZeroQMCharges() #Modifies self.charges_qmregionzeroed
                print("length of self.charges_qmregionzeroed :", len(self.charges_qmregionzeroed))
                
                # Todo: make sure this works for OpenMM and for NonBondedTheory
                # Updating charges in MM object. Using charges that have been zeroed for QM (no other modifications)
                #Updated...
                self.mm_theory.update_charges(self.qmatoms,[0.0 for i in self.qmatoms])
                
                
                print("Charges of QM atoms set to 0 (since Electrostatic Embedding):")
                if self.printlevel > 3:
                    for i in self.allatoms:
                        if i in self.qmatoms:
                            print("QM atom {} ({}) charge: {}".format(i, self.elems[i], self.charges_qmregionzeroed[i]))
                        else:
                            print("MM atom {} ({}) charge: {}".format(i, self.elems[i], self.charges_qmregionzeroed[i]))
                blankline()
        else:
            #Case: No actual MM theory but we still want to zero charges for QM elstate embedding calculation
            #TODO: Remove option for no MM theory or keep this ??
            self.ZeroQMCharges() #Modifies self.charges_qmregionzeroed
            print("length of self.charges_qmregionzeroed :", len(self.charges_qmregionzeroed))

    #From QM1:MM1 boundary dict, get MM1:MMx boundary dict (atoms connected to MM1)
    def get_MMboundary(self):
        # if boundarydict is not empty we need to zero MM1 charge and distribute charge from MM1 atom to MM2,MM3,MM4
        #Creating dictionary for each MM1 atom and its connected atoms: MM2-4
        self.MMboundarydict={}
        for (QM1atom,MM1atom) in self.boundaryatoms.items():
            connatoms = get_connected_atoms(self.coords, self.elems, settings_ash.scale, settings_ash.tol, MM1atom)
            #Deleting QM-atom from connatoms list
            connatoms.remove(QM1atom)
            self.MMboundarydict[MM1atom] = connatoms
        print("")
        print("MM boundary (MM1:MMx pairs):", self.MMboundarydict)
                
    # Set QMcharges to Zero and shift charges at boundary
    #TODO: Add both L2 scheme (delete whole charge-group of M1) and charge-shifting scheme (shift charges to Mx atoms and add dipoles for each Mx atom)
    
    def ZeroQMCharges(self):
        print("Setting QM charges to Zero")
        #Looping over charges and setting QM atoms to zero
        #1. Copy charges to charges_qmregionzeroed
        self.charges_qmregionzeroed=copy.copy(self.charges)
        #2. change charge for QM-atom
        for i, c in enumerate(self.charges_qmregionzeroed):
            #Setting QMatom charge to 0
            if i in self.qmatoms:
                self.charges_qmregionzeroed[i] = 0.0
        #3. Flag that this has been done
        self.QMChargesZeroed = True
    def ShiftMMCharges(self):
        print("Shifting MM charges at QM-MM boundary.")
        print("len self.charges_qmregionzeroed: ", len(self.charges_qmregionzeroed))
        print("len self.charges: ", len(self.charges))
        
        #Create self.pointcharges list
        self.pointcharges=copy.copy(self.charges_qmregionzeroed)
        
        #Looping over charges and setting QM/MM1 atoms to zero and shifting charge to neighbouring atoms
        for i, c in enumerate(self.pointcharges):

            #If index corresponds to MMatom at boundary, set charge to 0 (charge-shifting
            if i in self.MMboundarydict.keys():
                MM1charge = self.charges[i]
                #print("MM1atom charge: ", MM1charge)
                self.pointcharges[i] = 0.0
                #MM1 charge fraction to be divided onto the other MM atoms
                MM1charge_fract = MM1charge / len(self.MMboundarydict[i])
                #print("MM1charge_fract :", MM1charge_fract)

                #TODO: Should charges be updated for MM program also ?
                #Putting the fractional charge on each MM2 atom
                for MMx in self.MMboundarydict[i]:
                    #print("MMx : ", MMx)
                    #print("Old charge : ", self.charges_qmregionzeroed[MMx])
                    self.pointcharges[MMx] += MM1charge_fract
                    #print("New charge : ", self.charges_qmregionzeroed[MMx])
                    #exit()
                
    #Create dipole charge (twice) for each MM2 atom that gets fraction of MM1 charge
    def get_dipole_charge(self,delq,direction,mm1index,mm2index):
        #Distance between MM1 and MM2
        MM_distance = distance_between_atoms(fragment=self.fragment, atom1=mm1index, atom2=mm2index)
        #Coordinates
        mm1coords=np.array(self.fragment.coords[mm1index])
        mm2coords=np.array(self.fragment.coords[mm2index])
        
        SHIFT=0.15
        #Normalize vector
        def vnorm(p1):
            r = math.sqrt((p1[0]*p1[0])+(p1[1]*p1[1])+(p1[2]*p1[2]))
            v1=np.array([p1[0] / r, p1[1] / r, p1[2] /r])
            return v1
        diffvector=mm2coords-mm1coords
        normdiffvector=vnorm(diffvector)
        
        #Dipole
        d = delq*2.5
        #Charge (abs value)
        q0 = 0.5 * d / SHIFT
        #print("q0 : ", q0)
        #Actual shift
        #print("direction : ", direction)
        shift = direction * SHIFT * ( MM_distance / 2.5 )
        #print("shift : ", shift)
        #Position
        #print("normdiffvector :", normdiffvector)
        #print(normdiffvector*shift)
        pos = mm2coords+np.array((shift*normdiffvector))
        #print("pos :", pos)
        #Returning charge with sign based on direction and position
        #Return coords as regular list
        return -q0*direction,list(pos)
    def SetDipoleCharges(self):
        print("Adding extra charges to preserve dipole moment for charge-shifting")
        #Adding 2 dipole pointcharges for each MM2 atom
        self.dipole_charges = []
        self.dipole_coords = []
        #print("self.MMboundarydict : ", self.MMboundarydict)
        for MM1,MMx in self.MMboundarydict.items():
            #print("MM1 :", MM1)
            #print("MMx : ", MMx)
            #Getting original MM1 charge (before set to 0)
            MM1charge = self.charges[MM1]
            #print("MM1atom charge: ", MM1charge)
            MM1charge_fract=MM1charge/len(MMx)
            #print("MM1charge_fract:", MM1charge_fract)
            
            for MM in MMx:
                #print("MM :", MM)
                q_d1, pos_d1 = self.get_dipole_charge(MM1charge_fract,1,MM1,MM)
                #print("q_d1: ", q_d1)
                #print("pos_d1: ", pos_d1)
                q_d2, pos_d2 = self.get_dipole_charge(MM1charge_fract,-1,MM1,MM)
                #print("q_d2: ", q_d2)
                #print("pos_d2: ", pos_d2)
                self.dipole_charges.append(q_d1)
                self.dipole_charges.append(q_d2)
                self.dipole_coords.append(pos_d1)
                self.dipole_coords.append(pos_d2)
    
    def run(self, current_coords=None, elems=None, Grad=False, nprocs=1):
        CheckpointTime = time.time()
        if self.printlevel >= 2:
            print(BC.WARNING, BC.BOLD, "------------RUNNING QM/MM MODULE-------------", BC.END)
            print("QM Module:", self.qm_theory_name)
            print("MM Module:", self.mm_theory_name)

        #If no coords provided to run (from Optimizer or NumFreq or MD) then use coords associated with object.
        #if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords=self.coords

        if self.embedding=="Elstat":
            PC=True
        else:
            PC=False
        
        #If nprocs was set when calling .run then using, otherwise use self.nprocs
        if nprocs==1:
            nprocs=self.nprocs
        
        if self.printlevel >= 2:
            print("Running QM/MM object with {} cores available".format(nprocs))
        #Updating QM coords and MM coords.
        
        #TODO: Should we use different name for updated QMcoords and MMcoords here??
        self.qmcoords=[current_coords[i] for i in self.qmatoms]
        self.mmcoords=[current_coords[i] for i in self.mmatoms]
        
        self.qmelems=[self.elems[i] for i in self.qmatoms]
        self.mmelems=[self.elems[i] for i in self.mmatoms]
        
        
        
        #LINKATOMS
        #1. Get linkatoms coordinates
        if self.linkatoms==True:
            linkatoms_dict = get_linkatom_positions(self.boundaryatoms,self.qmatoms, current_coords, self.elems)
            print("linkatoms_dict:", linkatoms_dict)
            #2. Add linkatom coordinates to qmcoords???
            print("Adding linkatom positions to QM coords")
            

            linkatoms_indices=[]
            
            #Sort by QM atoms:
            print("linkatoms_dict.keys :", linkatoms_dict.keys())
            for pair in sorted(linkatoms_dict.keys()):
                print("Pair :", pair)
                self.qmcoords.append(linkatoms_dict[pair])
                #print("self.qmcoords :", self.qmcoords)
                #print(len(self.qmcoords))
                #exit()
                #Linkatom indices for book-keeping
                linkatoms_indices.append(len(self.qmcoords)-1)
                print("linkatoms_indices: ", linkatoms_indices)
                
            #TODO: Modify qm_elems list. Use self.qmelems or separate qmelems ?
            #TODO: Should we do this at object creation instead?
            current_qmelems=self.qmelems + ['H']*len(linkatoms_dict)
            print("")
            #print("current_qmelems :", current_qmelems)
            print(len(current_qmelems))
            
            #Charge-shifting + Dipole thing
            print("Doing charge-shifting...")
            #print("Before: self.pointcharges are: ", self.pointcharges)
            #Do Charge-shifting. MM1 charge distributed to MM2 atoms
            
            self.ShiftMMCharges() # Creates self.pointcharges
            #print("After: self.pointcharges are: ", self.pointcharges)
            print("len self.pointcharges: ", len(self.pointcharges))
            
            #TODO: Code alternative to Charge-shifting: L2 scheme which deletes whole charge-group that MM1 belongs to
            
            # Defining pointcharges as only containing MM atoms
            self.pointcharges=[self.pointcharges[i] for i in self.mmatoms]
            #print("After: self.pointcharges are: ", self.pointcharges)
            print("len self.pointcharges: ", len(self.pointcharges))
            #Set 
            self.SetDipoleCharges() #Creates self.dipole_charges and self.dipole_coords

            #Adding dipole charge coords to MM coords (given to QM code) and defining pointchargecoords
            print("Adding {} dipole charges to PC environment".format(len(self.dipole_charges)))
            self.pointchargecoords=self.mmcoords+self.dipole_coords
            
            #Adding dipole charges to MM charges list (given to QM code)
            #TODO: Rename as pcharges list so as not to confuse with what MM code sees??
            self.pointcharges=self.pointcharges+self.dipole_charges
            print("len self.pointcharges after dipole addition: ", len(self.pointcharges))
            print(len(self.pointcharges))
            print(len(self.pointchargecoords))
        else:
            #If no linkatoms then use original self.qmelems
            current_qmelems = self.qmelems
            #If no linkatoms then self.pointcharges are just original charges with QM-region zeroed
            print("self.mmatoms:", self.mmatoms)
            print("self.charges_qmregionzeroed: ", self.charges_qmregionzeroed)
            self.pointcharges=[self.charges_qmregionzeroed[i] for i in self.mmatoms]
            #If no linkatoms MM coordinates are the same
            self.pointchargecoords=self.mmcoords
       
        #TODO: Now we have updated MM-coordinates (if doing linkatoms, wtih dipolecharges etc) and updated mm-charges (more, due to dipolecharges if linkatoms)
        # We also have MMcharges that have been set to zero due to QM/mm
        # Choice: should we now delete charges that are zero or not. chemshell does
        #TODO: do here or have QM-theory do it. probably best to do here (otherwise we have to write multiple QM interface routines)
        

        #Removing zero-valued charges
        #NOTE: Problem, if we remove zero-charges we lose our indexing as the charges removed could be anywhere
        # NOTE: Test: Let's not remove them.
        print("Number of charges :", len(self.pointcharges))
        #print("Removing zero-valued charges")
        #self.pointcharges, self.pointchargecoords = remove_zero_charges(self.pointcharges, self.pointchargecoords)
        print("Number of charges :", len(self.pointcharges))
        print("Number of charge coordinates :", len(self.pointchargecoords))
        print_time_rel(CheckpointTime, modulename='QM/MM run prep')
        
        #If no qmatoms then do MM-only
        if len(self.qmatoms) == 0:
            print("No qmatoms list provided. Setting QMtheory to None")
            self.qm_theory_name="None"
            self.QMenergy=0.0
        
        
        
        if self.qm_theory_name=="ORCATheory":
            #Calling ORCA theory, providing current QM and MM coordinates.
            if Grad==True:
                if PC==True:
                    self.QMenergy, self.QMgradient, self.PCgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                                                         current_MM_coords=self.pointchargecoords,
                                                                                         MMcharges=self.pointcharges,
                                                                                         qm_elems=current_qmelems,
                                                                                         Grad=True, PC=True, nprocs=nprocs)
                else:
                    self.QMenergy, self.QMgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.pointchargecoords, MMcharges=self.pointcharges,
                                                      qm_elems=current_qmelems, Grad=True, PC=False, nprocs=nprocs)
            else:
                self.QMenergy = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.pointchargecoords, MMcharges=self.pointcharges,
                                                      qm_elems=current_qmelems, Grad=False, PC=PC, nprocs=nprocs)
        elif self.qm_theory_name == "Psi4Theory":
            #Calling Psi4 theory, providing current QM and MM coordinates.
            if Grad==True:
                if PC==True:
                    print(BC.WARNING, "Pointcharge gradient for Psi4 is not implemented.",BC.END)
                    print(BC.WARNING, "Warning: Only calculating QM-region contribution, skipping electrostatic-embedding gradient on pointcharges", BC.END)
                    print(BC.WARNING, "Only makes sense if MM region is frozen! ", BC.END)
                    self.QMenergy, self.QMgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                                                         current_MM_coords=self.pointchargecoords,
                                                                                         MMcharges=self.pointcharges,
                                                                                         qm_elems=current_qmelems,
                                                                                         Grad=True, PC=True, nprocs=nprocs)
                    #Creating zero-gradient array
                    self.PCgradient = np.zeros((len(self.mmatoms), 3))
                else:
                    print("grad. mech embedding. not ready")
                    exit()
                    self.QMenergy, self.QMgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.pointchargecoords, MMcharges=self.pointcharges,
                                                      qm_elems=current_qmelems, Grad=True, PC=False, nprocs=nprocs)
            else:
                print("grad false.")
                if PC == True:
                    print("PC embed true. not ready")
                    self.QMenergy = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.pointchargecoords, MMcharges=self.pointcharges,
                                                      qm_elems=current_qmelems, Grad=False, PC=PC, nprocs=nprocs)
                else:
                    print("mech true", not ready)
                    exit()


        elif self.qm_theory_name == "xTBTheory":
            #Calling xTB theory, providing current QM and MM coordinates.
            if Grad==True:
                if PC==True:
                    self.QMenergy, self.QMgradient, self.PCgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                                                         current_MM_coords=self.pointchargecoords,
                                                                                         MMcharges=self.pointcharges,
                                                                                         qm_elems=current_qmelems,
                                                                                         Grad=True, PC=True, nprocs=nprocs)
                else:
                    self.QMenergy, self.QMgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.pointchargecoords, MMcharges=self.pointcharges,
                                                      qm_elems=current_qmelems, Grad=True, PC=False, nprocs=nprocs)
            else:
                self.QMenergy = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.pointchargecoords, MMcharges=self.pointcharges,
                                                      qm_elems=current_qmelems, Grad=False, PC=PC, nprocs=nprocs)


        elif self.qm_theory_name == "DaltonTheory":
            print("not yet implemented")
            exit(1)
        elif self.qm_theory_name == "NWChemtheory":
            print("not yet implemented")
            exit(1)
        elif self.qm_theory_name == "None":
            print("No QMtheory. Skipping QM calc")
            self.QMenergy=0.0;self.linkatoms=False;self.PCgradient=np.array([0.0, 0.0, 0.0])
            self.QMgradient=np.array([0.0, 0.0, 0.0])
        elif self.qm_theory_name == "ZeroTheory":
            self.QMenergy=0.0;self.linkatoms=False;self.PCgradient=np.array([0.0, 0.0, 0.0])
            self.QMgradient=np.array([0.0, 0.0, 0.0])
        else:
            print("invalid QM theory")
            exit(1)
        print_time_rel(CheckpointTime, modulename='QM step')
        CheckpointTime = time.time()



        # MM THEORY
        if self.mm_theory_name == "NonBondedTheory":
            if self.printlevel >= 2:
                print("Running MM theory as part of QM/MM.")
                print("Using MM on full system. Charges for QM region  have to be set to zero ")
                printdebug("Charges for full system is: ", self.charges)
                print("Passing QM atoms to MMtheory run so that QM-QM pairs are skipped in pairlist")
                print("Passing active atoms to MMtheory run so that frozen pairs are skipped in pairlist")
            assert len(current_coords) == len(self.charges_qmregionzeroed)
                
            # NOTE: charges_qmregionzeroed for full system but with QM-charges zeroed (no other modifications)
            #NOTE: Using original system coords here (not with linkatoms, dipole etc.). Also not with deleted zero-charge coordinates. 
            #charges list for full system, can be zeroed but we still want the LJ interaction
                
            self.MMenergy, self.MMgradient= self.mm_theory.run(current_coords=current_coords,
                                                               charges=self.charges_qmregionzeroed, connectivity=self.connectivity,
                                                               qmatoms=self.qmatoms, actatoms=self.actatoms)

        elif self.mm_theory_name == "OpenMMTheory":
            if self.printlevel >= 2:
                print("Running OpenMM theory as part of QM/MM.")
            if self.QMChargesZeroed==True:
                if self.printlevel >= 2:
                    print("Using MM on full system. Charges for QM region {} have been set to zero ".format(self.qmatoms))
            else:
                print("QMCharges have not been zeroed")
                exit(1)
            printdebug("Charges for full system is: ", self.charges)
            #Todo: Need to make sure OpenMM skips QM-QM Lj interaction => Exclude
            #Todo: Need to have OpenMM skip frozen region interaction for speed  => => Exclude
            self.MMenergy, self.MMgradient= self.mm_theory.run(current_coords=current_coords, qmatoms=self.qmatoms)
        else:
            self.MMenergy=0
        print_time_rel(CheckpointTime, modulename='MM step')
        CheckpointTime = time.time()
        #Final QM/MM Energy
        self.QM_MM_energy= self.QMenergy+self.MMenergy
        blankline()
        if self.printlevel >= 2:
            print("{:<20} {:>20.12f}".format("QM energy: ",self.QMenergy))
            print("{:<20} {:>20.12f}".format("MM energy: ", self.MMenergy))
            print("{:<20} {:>20.12f}".format("QM/MM energy: ", self.QM_MM_energy))
        blankline()

        #Final QM/MM gradient. Combine QM gradient, MM gradient and PC-gradient (elstat MM gradient from QM code).
        #First combining QM and PC gradient to one.
        if Grad == True:
            
            #TODO: Deal with linkatom gradient here.
            # Add contribution to QM1 and MM1 contribution???
            
            if self.linkatoms==True:
                #This projects the linkatom force onto the respective QM atom and MM atom
                def linkatom_force_fix(Qcoord, Mcoord, Lcoord, Qgrad,Mgrad,Lgrad):
                    #QM1-L and QM1-MM1 distances
                    QLdistance=distance(Qcoord,Lcoord)
                    #print("QLdistance:", QLdistance)
                    MQdistance=distance(Mcoord,Qcoord)
                    #print("MQdistance:", MQdistance)
                    #B and C: a 3x3 arrays
                    B=np.zeros([3,3])
                    C=np.zeros([3,3])
                    for i in range(0,2):
                        for j in range(0,2):
                            B[i,j]=-1*QLdistance*(Mcoord[i]-Qcoord[i])*(Mcoord[j]-Qcoord[j]) / (MQdistance*MQdistance*MQdistance)
                    for i in range(0,2):
                        B[i,i] = B[i,i] + QLdistance / MQdistance
                    for i in range(0,2):
                        for j in range(0,2):
                            C[i,j]= -1 * B[i,j]
                    for i in range(0,2):
                        C[i,i] = C[i,i] + 1.0                
                
                    #QM atom gradient
                    #print("Qgrad:", Qgrad)
                    #print("Lgrad:", Lgrad)
                    #print("C: ", C)
                    #print("B:", B)
                    #Multiply grad by C-diagonal
                    Qgrad[0] = Qgrad[0]*C[0][0]
                    Qgrad[1] = Qgrad[1]*C[1][1]
                    Qgrad[2] = Qgrad[2]*C[2][2]
                    
                    #print("Qgrad:", Qgrad)
                    #MM atom gradient
                    #print("Mgrad:", Mgrad)
                    Mgrad[0] = Mgrad[0]*B[0][0]
                    Mgrad[1] = Mgrad[1]*B[1][1]
                    Mgrad[2] = Mgrad[2]*B[2][2]                    
                    #print("Mgrad:", Mgrad)
                    
                    return Qgrad,Mgrad
                
                def fullindex_to_qmindex(fullindex,qmatoms):
                    qmindex=qmatoms.index(fullindex)
                    return qmindex
                
                #print("here")
                #print("linkatoms_dict: ", linkatoms_dict)
                #print("linkatoms_indices: ", linkatoms_indices)
                num_linkatoms=len(linkatoms_indices)
                for pair in sorted(linkatoms_dict.keys()):
                    #print("pair: ", pair)
                    linkatomindex=linkatoms_indices.pop(0)
                    #print("linkatomindex:", linkatomindex)
                    Lgrad=self.QMgradient[linkatomindex]
                    #print("Lgrad:",Lgrad)
                    Lcoord=linkatoms_dict[pair]
                    #print("Lcoord:", Lcoord)
                    fullatomindex_qm=pair[0]
                    #print("fullatomindex_qm:", fullatomindex_qm)
                    #print("self.qmatoms:", self.qmatoms)
                    qmatomindex=fullindex_to_qmindex(fullatomindex_qm,self.qmatoms)
                    #print("qmatomindex:", qmatomindex)
                    fullatomindex_mm=pair[1]
                    #print("fullatomindex_mm:", fullatomindex_mm)
                    Qcoord=self.qmcoords[qmatomindex]
                    #print("Qcoords: ", Qcoord)
                    #print("type self.QMGradient", type(self.QMgradient))
                    Qgrad=self.QMgradient[qmatomindex]
                    #print("Qgrad:", Qgrad)
                    
                    #print("length of self.MMgradient:", len(self.MMgradient))
                    Mcoord=current_coords[fullatomindex_mm]
                    #print("Mcoord:", Mcoord)
                    #print("type self.MMgradient", type(self.MMgradient))
                    #print("fullatomindex_mm:", fullatomindex_mm)
                    Mgrad=self.MMgradient[fullatomindex_mm]
                    print("Mgrad: ", Mgrad)
                    Qgrad,Mgrad= linkatom_force_fix(Qcoord, Mcoord, Lcoord, Qgrad,Mgrad,Lgrad)
                    #print("Qgrad: ", Qgrad)
                    #print("Mgrad: ", Mgrad)
                    self.QMgradient[qmatomindex]=Qgrad
                    self.MMgradient[fullatomindex_mm]=Mgrad
                #Fix QMgradient by removing linkatom contributions (bottom)
                #Redundant?
                #print("self.QMgradient:", self.QMgradient)
                self.QMgradient=self.QMgradient[0:-num_linkatoms] #remove linkatoms
                #print("self.QMgradient:", self.QMgradient)
            
                # QM_PC_gradient is system size. Combining QM part (after linkatom-contribution removed) and PC part into 
                #print("self.allatoms:", len(self.allatoms))
                #print("len(self.QMgradient) :", len(self.QMgradient))
                #print("len(self.MMgradient) :", len(self.MMgradient))
                #print("length self.PCgradient ", len(self.PCgradient))
            
                assert len(self.allatoms) == len(self.MMgradient)
                assert len(self.QMgradient) + len(self.PCgradient) - len(self.dipole_charges)  == len(self.MMgradient)
            
            self.QM_PC_gradient = np.zeros((len(self.MMgradient), 3))
            qmcount=0;pccount=0
            for i in self.allatoms:
                if i in self.qmatoms:
                    #QM-gradient. Linkatom gradients are skipped
                    self.QM_PC_gradient[i]=self.QMgradient[qmcount]
                    qmcount+=1
                else:
                    #Pointcharge-gradient. Dipole-charge gradients are skipped (never reached)
                    self.QM_PC_gradient[i] = self.PCgradient[pccount]
                    pccount += 1
            
            #print("qmcount:", qmcount)
            #print("pccount:", pccount)
            #print("self.QM_PC_gradient len ", len(self.QM_PC_gradient))
            assert qmcount == len(self.qmatoms)
            assert pccount == len(self.mmatoms)
            
            #Now assemble final QM/MM gradient
            assert len(self.QM_PC_gradient) == len(self.MMgradient)
            self.QM_MM_gradient=self.QM_PC_gradient+self.MMgradient
            #print_time_rel(CheckpointTime, modulename='QM/MM gradient combine')
            if self.printlevel >=3:
                print("QM gradient (au/Bohr):")
                print_coords_all(self.QMgradient, self.qmelems, self.qmatoms)
                blankline()
                print("PC gradient (au/Bohr):")
                print_coords_all(self.PCgradient, self.mmelems, self.mmatoms)
                blankline()
                print("QM+PC gradient (au/Bohr):")
                print_coords_all(self.QM_PC_gradient, self.elems, self.allatoms)
                blankline()
                print("MM gradient (au/Bohr):")
                print_coords_all(self.MMgradient, self.elems, self.allatoms)
                blankline()
                print("Total QM/MM gradient (au/Bohr):")
                print("")
                print_coords_all(self.QM_MM_gradient, self.elems,self.allatoms)
            if self.printlevel >= 2:
                print(BC.WARNING,BC.BOLD,"------------ENDING QM/MM MODULE-------------",BC.END)
            return self.QM_MM_energy, self.QM_MM_gradient
        else:
            return self.QM_MM_energy



class DaltonTheory:
    def __init__(self, daltondir=None, fragment=None, charge=None, mult=None, printlevel=2, nprocs=1, pe=False, potfile='',
                 label=None, method=None, response=None, dalton_input=None, basis_name=None,basis_dir=None):
        if daltondir is None:
            print("No daltondir argument passed to DaltonTheory. Attempting to find daltondir variable inside settings_ash")
            self.daltondir=settings_ash.daltondir
        else:
            self.daltondir = daltondir

        if basis_name is not None:
            print("here")
            self.basis_name=basis_name
        else:
            print("Please provide basis_name to DaltonTheory object")
            exit()

        #Directory where basis sets are. If not defined, ASH will assume basis directory is one dir up
        self.basis_dir=basis_dir

        #Used to write name in MOLECULE.INP. Not necessary?
        self.moleculename="None"

        #Dalton input as a multi-line string
        self.dalton_input=dalton_input

        #Label to distinguish different Dalton objects
        self.label=label
        #Printlevel
        self.printlevel=printlevel
        #Setting nprocs of object
        self.nprocs=nprocs
        
        #Setting energy to 0.0 for now
        self.energy=0.0
        
        self.pe=pe
        
        #Optional linking of coords to theory object, not necessary. TODO: Delete
        if fragment != None:
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        #print("frag elems", self.fragment.elems)
        if charge!=None:
            self.charge=int(charge)
        if mult!=None:
            self.mult=int(mult)
        print("Note: Dalton assumes mult=1 for even electrons and mult=2 for odd electrons.")
        if self.printlevel >=2:
            print("")
            print("Creating DaltonTheory object")
            print("Dalton dir:", self.daltondir)
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old Dalton files")
        list=[]
        #list.append(self.inputfilename + '.gbw')
        for file in list:
            try:
                os.remove(file)
            except:
                pass
        # os.remove(self.inputfilename + '.out')
        try:
            for tmpfile in glob.glob("self.inputfilename*tmp"):
                os.remove(tmpfile)
        except:
            pass
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, qm_elems=None, Grad=False, nprocs=None, pe=None, potfile='', restart=False ):
        
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING DALTON INTERFACE-------------", BC.END)
        if pe is None:
            pe=self.pe
        
        print("pe: ", pe)
        #Coords provided to run or else taken from initialization.
        #if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords=self.coords

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems=self.elems
            else:
                qm_elems = elems

        if nprocs==None:
            nprocs=self.nprocs
        print("Running Dalton object with {} cores available".format(nprocs))

        #DALTON.INP
        print("Creating inputfile: DALTON.INP")
        with open("DALTON.INP",'w') as dalfile:
            for substring in self.dalton_input.split('\n'):
                dalfile.write(substring+'\n')
                if 'DALTON' in substring:
                    if pe is True:
                        dalfile.write(".PEQM\n")
                
        #Write the ugly MOLECULE.INP
        uniq_elems=set(qm_elems)
        
        with open("MOLECULE.INP",'w') as molfile:
            molfile.write('BASIS\n')
            molfile.write('{}\n'.format(self.basis_name))
            molfile.write('{}\n'.format(self.moleculename))
            molfile.write('------------------------\n')
            molfile.write('AtomTypes={} NoSymmetry Angstrom Charge={}\n'.format(len(uniq_elems),self.charge))
            for uniqel in uniq_elems:
                nuccharge=float(elemstonuccharges([uniqel])[0])
                num_elem=qm_elems.count(uniqel)
                molfile.write('Charge={} Atoms={}\n'.format(nuccharge,num_elem))
                for el,coord in zip(qm_elems,current_coords):
                    if el == uniqel:
                        molfile.write('{}    {} {} {}\n'.format(el,coord[0],coord[1],coord[2]))
        
        #POTENTIAL FILE
        #Renaming potfile as POTENTIAL.INP
        os.rename(potfile,'POTENTIAL.INP')
        print("Rename potential file {} as POTENTIAL.INP".format(potfile))
        
        print("Charge: {}  Mult: {}".format(self.charge, self.mult))
        
        print("Launching Dalton")
        print("daltondir:", self.daltondir)
        if self.basis_dir is None:
            print("No Dalton basis_dir provided. Attempting to set BASDIR via daltondir:")
            os.environ['BASDIR'] = self.daltondir+"/../basis"
            print("BASDIR:", os.environ['BASDIR'])
        else:
            print("Using basis_dir:", self.basis_dir)
            os.environ['BASDIR'] = self.basis_dir
        def run_dalton_serial(daltondir):
            with open("DALTON.OUT", 'w') as ofile:
                
                process = sp.run([daltondir + '/dalton.x'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
            
        run_dalton_serial(self.daltondir)
        
        #Grab final energy
        #TODO: Support more than DFT energies
        #ALSO: Grab excitation energies etc
        with open("DALTON.OUT") as outfile:
            for line in outfile:
                if 'Final DFT energy:' in line:
                    self.energy=float(line.split()[-1])
        
        if self.energy==0.0:
            print("Problem. Energy not found in output: DALTON.OUT")
            exit()
        return self.energy

#ORCA Theory object. Fragment object is optional. Only used for single-points.

class ORCATheory:
    def __init__(self, orcadir=None, fragment=None, charge=None, mult=None, orcasimpleinput='', printlevel=2, extrabasisatoms=None, extrabasis=None,
                 orcablocks='', extraline='', brokensym=None, HSmult=None, atomstoflip=None, nprocs=1, label=None, moreadfile=None):

        if orcadir is None:
            print("No orcadir argument passed to ORCATheory. Attempting to find orcadir variable inside settings_ash")
            self.orcadir=settings_ash.orcadir
        else:
            self.orcadir = orcadir

        #Label to distinguish different ORCA objects
        self.label=label

        #Create inputfile with generic name
        self.inputfilename="orca-input"

        #MOREAD-file
        self.moreadfile=moreadfile

        #Using orcadir to set LD_LIBRARY_PATH
        old = os.environ.get("LD_LIBRARY_PATH")
        if old:
            os.environ["LD_LIBRARY_PATH"] = self.orcadir + ":" + old
        else:
            os.environ["LD_LIBRARY_PATH"] = self.orcadir
        #os.environ['LD_LIBRARY_PATH'] = orcadir + ':$LD_LIBRARY_PATH'

        #Printlevel
        self.printlevel=printlevel

        #Setting nprocs of object
        self.nprocs=nprocs


        if fragment != None:
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        #print("frag elems", self.fragment.elems)
        if charge!=None:
            self.charge=int(charge)
        else:
            self.charge=None
        if mult!=None:
            self.mult=int(mult)
        else:
            self.charge=None
        self.orcasimpleinput=orcasimpleinput
        self.orcablocks=orcablocks
        self.extraline=extraline
        #BROKEN SYM OPTIONS
        self.brokensym=brokensym
        self.HSmult=HSmult
        if type(atomstoflip) is int:
            print(BC.FAIL,"Error: atomstoflip should be list of integers (e.g. [0] or [2,3,5]), not a single integer.", BC.END)
            exit(1)
        if atomstoflip != None:
            self.atomstoflip=atomstoflip
        else:
            self.atomstoflip=[]
        #Extrabasis
        if extrabasisatoms != None:
            self.extrabasisatoms=extrabasisatoms
            self.extrabasis=extrabasis
        else:
            self.extrabasisatoms=[]
            self.extrabasis=""
        
        
        # self.qmatoms need to be set for Flipspin to work for QM/MM job.
        #Overwritten by QMMMtheory, used in Flip-spin
        self.qmatoms=[]
            
        if self.printlevel >=2:
            print("")
            print("Creating ORCA object")
            print("ORCA dir:", self.orcadir)
            #if molcrys then there is not charge and mult available
            #print("Charge: {} Mult: {}".format(self.charge,self.mult))
            print(self.orcasimpleinput)
            print(self.orcablocks)
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old ORCA files")
        list=[]
        list.append(self.inputfilename + '.gbw')
        list.append(self.inputfilename + '.ges')
        list.append(self.inputfilename + '.prop')
        list.append(self.inputfilename + '.uco')
        list.append(self.inputfilename + '_property.txt')
        list.append(self.inputfilename + '.inp')
        list.append(self.inputfilename + '.out')
        list.append(self.inputfilename + '.engrad')
        for file in list:
            try:
                os.remove(file)
            except:
                pass
        # os.remove(self.inputfilename + '.out')
        try:
            for tmpfile in glob.glob("self.inputfilename*tmp"):
                os.remove(tmpfile)
        except:
            pass
    
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, Hessian=False, PC=False, nprocs=None ):
        print(BC.OKBLUE,BC.BOLD, "------------RUNNING ORCA INTERFACE-------------", BC.END)
        #Coords provided to run or else taken from initialization.
        #if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords=self.coords

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems=self.elems
            else:
                qm_elems = elems

        #If QM/MM then extrabasisatoms and atomstoflip have to be updated
        if len(self.qmatoms) != 0:
            #extrabasisatomindices if QM/MM
            print("self.qmatoms :", self.qmatoms)
            qmatoms_extrabasis=[self.qmatoms.index(i) for i in self.extrabasisatoms]
            #new QM-region indices for atomstoflip if QM/MM
            qmatomstoflip=[self.qmatoms.index(i) for i in self.atomstoflip]
        else:
            qmatomstoflip=self.atomstoflip
            qmatoms_extrabasis=self.extrabasisatoms
        
        if nprocs==None:
            nprocs=self.nprocs
        print("Running ORCA object with {} cores available".format(nprocs))


        print("Creating inputfile:", self.inputfilename+'.inp')
        print("ORCA input:")
        print(self.orcasimpleinput)
        print(self.extraline)
        print(self.orcablocks)
        print("Charge: {}  Mult: {}".format(self.charge, self.mult))
        #Printing extra options chosen:
        if self.brokensym==True:
            print("Brokensymmetry SpinFlipping on! HSmult: {}.".format(self.HSmult))
            for flipatom,qmflipatom in zip(self.atomstoflip,qmatomstoflip):
                print("Flipping atom: {} QMregionindex: {} Element: {}".format(flipatom, qmflipatom, qm_elems[qmflipatom]))
        if self.extrabasis != "":
            print("Using extra basis ({}) on QM-region indices : {}".format(self.extrabasis,qmatoms_extrabasis))

        if PC==True:
            print("Pointcharge embedding is on!")
            create_orca_pcfile(self.inputfilename, current_MM_coords, MMcharges)
            if self.brokensym == True:
                create_orca_input_pc(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        self.charge, self.mult, extraline=self.extraline, HSmult=self.HSmult, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                     atomstoflip=qmatomstoflip, extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis)
            else:
                create_orca_input_pc(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        self.charge, self.mult, extraline=self.extraline, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                        extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis)
        else:
            if self.brokensym == True:
                create_orca_input_plain(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        self.charge,self.mult, extraline=self.extraline, HSmult=self.HSmult, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                     atomstoflip=qmatomstoflip, extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis)
            else:
                create_orca_input_plain(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        self.charge,self.mult, extraline=self.extraline, Grad=Grad, Hessian=Hessian, moreadfile=self.moreadfile,
                                        extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis)

        #Run inputfile using ORCA parallelization. Take nprocs argument.
        #print(BC.OKGREEN, "------------Running ORCA calculation-------------", BC.END)
        print(BC.OKGREEN, "ORCA Calculation started.", BC.END)
        # Doing gradient or not. Disabling, doing above instead.
        #if Grad == True:
        #    run_orca_SP_ORCApar(self.orcadir, self.inputfilename + '.inp', nprocs=nprocs, Grad=True)
        #else:
        run_orca_SP_ORCApar(self.orcadir, self.inputfilename + '.inp', nprocs=nprocs)
        print(BC.OKGREEN, "ORCA Calculation done.", BC.END)

        #Now that we have possibly run a BS-DFT calculation, turning Brokensym off for future calcs (opt, restart, etc.)
        # using this theory object
        #TODO: Possibly use different flag for this???
        self.brokensym=False

        #Check if finished. Grab energy and gradient
        outfile=self.inputfilename+'.out'
        engradfile=self.inputfilename+'.engrad'
        pcgradfile=self.inputfilename+'.pcgrad'
        if checkORCAfinished(outfile) == True:
            self.energy=ORCAfinalenergygrab(outfile)

            if Grad == True:
                self.grad=ORCAgradientgrab(engradfile)
                if PC == True:
                    #Grab pointcharge gradient. i.e. gradient on MM atoms from QM-MM elstat interaction.
                    self.pcgrad=ORCApcgradientgrab(pcgradfile)
                    print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                    return self.energy, self.grad, self.pcgrad
                else:
                    print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                    return self.energy, self.grad

            else:
                print("Single-point ORCA energy:", self.energy)
                print(BC.OKBLUE,BC.BOLD,"------------ENDING ORCA-INTERFACE-------------", BC.END)
                return self.energy
        else:
            print(BC.FAIL,"Problem with ORCA run", BC.END)
            print(BC.OKBLUE,BC.BOLD, "------------ENDING ORCA-INTERFACE-------------", BC.END)
            exit(1)



#Psi4 Theory object. Fragment object is optional. Only used for single-points.
#PSI4 runmode:
#   : library means that ASH will load Psi4 libraries and run psi4 directly
#   : inputfile means that ASH will create Psi4 inputfile and run a separate psi4 executable
#psi4dir only necessary for inputfile-based userinterface. Todo: Possibly unnexessary
#printsetting is by default set to 'File. Change to something else for stdout print
# PE: Polarizable embedding (CPPE). Pass pe_modulesettings dict as well
class Psi4Theory:
    def __init__(self, fragment=None, charge=None, mult=None, printsetting='False', psi4settings=None, psi4method=None,
                 runmode='library', psi4dir=None, pe=False, potfile='', outputname='psi4output.dat', label='psi4input',
                 psi4memory=3000, nprocs=1, printlevel=2):

        #Printlevel
        self.printlevel=printlevel


        self.nprocs=nprocs
        self.psi4memory=psi4memory
        self.label=label
        self.outputname=outputname
        self.printsetting=printsetting
        self.runmode=runmode
        #CPPE Polarizable Embedding options
        self.pe=pe
        #Potfile from user or passed on via QM/MM Theory object ?
        self.potfile=potfile
        #Determining runmode
        if self.runmode != 'library':
            print("Defining Psi4 object with runmode=psithon")
            if psi4dir is not None:
                print("Path to Psi4 provided:", psi4dir)
                self.psi4path=psi4dir
            else:
                self.psi4path=shutil.which('psi4')
                if self.psi4path==None:
                    print("Found no psi4 in path. Add Psi4 to Shell environment or provide psi4dir variable")
                    exit()
                else:
                    print("Found psi4 in path:", self.psi4path)


        if fragment is not None:
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        #print("frag elems", self.fragment.elems)
        if charge is not None:
            self.charge=int(charge)
        if mult is not None:
            self.mult=int(mult)
        self.psi4settings=psi4settings

        #DFT-specific. Remove? Marked for deletion
        #self.psi4functional=psi4functional

        #All valid Psi4 methods that can be arguments in energy() function
        self.psi4method=psi4method

    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old Psi4 files")
        try:
            os.remove('timer.dat')
            os.remove('psi4output.dat')
        except:
            pass
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, nprocs=None, pe=False, potfile='', restart=False ):

        if nprocs==None:
            nprocs=self.nprocs

        print(BC.OKBLUE,BC.BOLD, "------------RUNNING PSI4 INTERFACE-------------", BC.END)

        #If pe and potfile given as run argument
        if pe is not False:
            self.pe=pe
        if potfile != '':
            self.potfile=potfile

        #Coords provided to run or else taken from initialization.
        #if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords=self.coords

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems=self.elems
            else:
                qm_elems = elems

        #PSI4 runmode:
        #   : library means that ASH will load Psi4 libraries and run psi4 directly
        #   : inputfile means that ASH will create Psi4 inputfile and run a separate psi4 executable

        if self.runmode=='library':
            print("Psi4 Runmode: Library")
            try:
                import psi4
            except:
                print(BC.FAIL,"Problem importing psi4. Make sure psi4 has been installed as part of same Python as ASH", BC.END)
                print(BC.WARNING,"If problematic, switch to inputfile based Psi4 interface instead.", BC.END)
                exit(9)
            #Changing namespace may prevent crashes due to multiple jobs running at same time
            if self.label=='label':
                psi4.core.IO.set_default_namespace("psi4job_ygg")
            else:
                psi4.core.IO.set_default_namespace(self.label)

            #Printing to stdout or not:
            if self.printsetting:
                print("Printsetting = True. Printing output to stdout...")
            else:
                print("Printsetting = False. Printing output to file: {}) ".format(self.outputname))
                psi4.core.set_output_file(self.outputname, False)

            #Psi4 scratch dir
            print("Setting Psi4 scratchdir to ", os.getcwd())
            psi4_io = psi4.core.IOManager.shared_object()
            psi4_io.set_default_path(os.getcwd())

            #Creating Psi4 molecule object using lists and manual information
            psi4molfrag = psi4.core.Molecule.from_arrays(
                elez=elemstonuccharges(qm_elems),
                fix_com=True,
                fix_orientation=True,
                fix_symmetry='c1',
                molecular_charge=self.charge,
                molecular_multiplicity=self.mult,
                geom=current_coords)
            psi4.activate(psi4molfrag)

            #Adding MM charges as pointcharges if PC=True
            #Might be easier to use PE and potfile ??
            if PC==True:
                #Chargefield = psi4.QMMM()
                Chargefield = psi4.core.ExternalPotential()
                #Mmcoords seems to be in Angstrom
                for mmcharge,mmcoord in zip(MMcharges,current_MM_coords):
                    Chargefield.addCharge(mmcharge, mmcoord[0], mmcoord[1], mmcoord[2])
                psi4.core.set_global_option("EXTERN", True)
                psi4.core.EXTERN = Chargefield

            #Setting inputvariables
            print("Psi4 memory (MB): ", self.psi4memory)

            psi4.set_memory(str(self.psi4memory)+' MB')

            #Changing charge and multiplicity
            #psi4molfrag.set_molecular_charge(self.charge)
            #psi4molfrag.set_multiplicity(self.mult)

            #Setting RKS or UKS reference
            #For now, RKS always if mult 1 Todo: Make more flexible
            if self.mult == 1:
                self.psi4settings['reference'] = 'RHF'
            else:
                self.psi4settings['reference'] = 'UHF'

            #Controlling orbital read-in guess.
            if restart==True:
                self.psi4settings['guess'] = 'read'
                #Renameing orbital file
                PID = str(os.getpid())
                print("Restart Option On!")
                print("Renaming lastrestart.180 to {}".format(os.path.splitext( self.outputname)[0] + '.default.' + PID + '.180.npy'))
                os.rename('lastrestart.180', os.path.splitext( self.outputname)[0] + '.default.' + PID + '.180.npy')
            else:
                self.psi4settings['guess'] = 'sad'

            #Reading dict object with basic settings and passing to Psi4
            psi4.set_options(self.psi4settings)
            print("Psi4 settings:", self.psi4settings)

            #Reading module options dict and passing to Psi4
            #TODO: Make one for SCF, CC, PCM etc.
            #psi4.set_module_options(modulename, moduledict)

            #Reading PE module options if PE=True
            if self.pe==True:
                print(BC.OKGREEN,"Polarizable Embedding Option On! Using CPPE module inside Psi4", BC.END)
                print(BC.WARNING, "Potfile: ", self.potfile, BC.END)
                try:
                    if os.path.exists(self.potfile):
                        pass
                    else:
                        print(BC.FAIL, "Potfile: ", self.potfile, "does not exist!", BC.END)
                        exit()
                except:
                    exit()
                psi4.set_module_options('pe', {'potfile' : self.potfile})
                self.psi4settings['pe'] = 'true'

            #Controlling OpenMP parallelization. Controlled here, not via OMP_NUM_THREADS etc.
            psi4.set_num_threads(nprocs)

            #Namespace issue overlap integrals requires this when running with multiprocessing:
            # http://forum.psicode.org/t/wfn-form-h-errors/1304/2
            #psi4.core.clean()

            #Running energy or energy+gradient. Currently hardcoded to SCF-DFT jobs

            #TODO: Support pointcharges and PE embedding in Grad job?
            if Grad==True:
                print("Running gradient with Psi4 method:", self.psi4method)
                #grad=psi4.gradient('scf', dft_functional=self.psi4functional)
                grad=psi4.gradient(self.psi4method)
                self.gradient=np.array(grad)
                self.energy = psi4.variable("CURRENT ENERGY")
            else:
                #This might be unnecessary as I think all DFT functionals work as keyword to energy function. Hence psi4method works for all
                #self.energy = psi4.energy('scf', dft_functional=self.psi4functional)
                print("Running energy with Psi4 method:", self.psi4method)
                self.energy = psi4.energy(self.psi4method)
            #Keep restart file 180 as lastrestart.180
            PID = str(os.getpid())
            try:
                print("Renaming {} to lastrestart.180".format(os.path.splitext(self.outputname)[0]+'.default.'+PID+'.180.npy'))
                os.rename(os.path.splitext(self.outputname)[0]+'.default.'+PID+'.180.npy', 'lastrestart.180')
            except:
                pass

            #TODO: write in error handling here

            print(BC.OKBLUE, BC.BOLD, "------------ENDING PSI4-INTERFACE-------------", BC.END)

            if Grad == True:
                print("Single-point PSI4 energy:", self.energy)
                return self.energy, self.gradient
            else:
                print("Single-point PSI4 energy:", self.energy)
                return self.energy

        #Psithon INPUT-FILE BASED INTERFACE. Creates Psi4 inputfiles and runs Psithon as subprocessses
        elif self.runmode=='psithon':
            print("Psi4 Runmode: Psithon")
            print("Current directory:", os.getcwd())
            #Psi4 scratch dir
            #print("Setting Psi4 scratchdir to ", os.getcwd())
            #Possible option: Set scratch env-variable as subprocess??? TODO:
            #export PSI_SCRATCH=/path/to/existing/writable/local-not-network/directory/for/scratch/files
            #Better :
            #psi4_io.set_default_path('/scratch/user')
            #Setting inputvariables

            print("Psi4 Memory:", self.psi4memory)

            #Printing Psi4settings
            print("Psi4 method:", self.psi4method)
            print("Psi4 settings:", self.psi4settings)

            #Printing PE options and checking for ptfile
            if self.pe==True:
                print(BC.OKGREEN,"Polarizable Embedding Option On! Using CPPE module inside Psi4", BC.END)
                print(BC.WARNING, "Potfile: ", self.potfile, BC.END)
                try:
                    if os.path.exists(self.potfile):
                        pass
                    else:
                        print(BC.FAIL, "Potfile: ", self.potfile, "does not exist!", BC.END)
                        exit()
                except:
                    exit()

            #Write inputfile
            with open(self.label+'.inp', 'w') as inputfile:
                inputfile.write('psi4_io.set_default_path(\'{}\')\n'.format(os.getcwd()))
                inputfile.write('memory {} MB\n'.format(self.psi4memory))
                inputfile.write('molecule molfrag {\n')
                inputfile.write(str(self.charge)+' '+str(self.mult)+'\n')
                for el,c in zip(qm_elems, current_coords):
                    inputfile.write(el+' '+str(c[0])+' '+str(c[1])+' '+str(c[2])+'\n')
                inputfile.write('symmetry c1\n')
                inputfile.write('no_reorient\n')
                inputfile.write('no_com\n')
                inputfile.write('}\n')
                inputfile.write('\n')

                # Adding MM charges as pointcharges if PC=True
                # Might be easier to use PE and potfile ??
                if PC == True:
                    inputfile.write('Chrgfield = QMMM()\n')
                    # Mmcoords in Angstrom
                    for mmcharge, mmcoord in zip(MMcharges, current_MM_coords):
                        inputfile.write('Chrgfield.extern.addCharge({}, {}, {}, {})\n'.format(mmcharge, mmcoord[0], mmcoord[1], mmcoord[2]))
                    inputfile.write('psi4.set_global_option_python(\'EXTERN\', Chrgfield.extern)\n')
                inputfile.write('\n')
                #Adding Psi4 settings
                inputfile.write('set {\n')
                for key,val in self.psi4settings.items():
                    inputfile.write(key+' '+val+'\n')
                #Setting RKS or UKS reference. For now, RKS always if mult 1 Todo: Make more flexible
                if self.mult == 1:
                    self.psi4settings['reference'] = 'RHF'
                else:
                    inputfile.write('reference UHF \n')
                #Orbital guess
                if restart == True:
                    inputfile.write('guess read \n')
                else:
                    inputfile.write('guess sad \n')
                #PE
                if self.pe == True:
                    inputfile.write('pe true \n')
                #end
                inputfile.write('}\n')

                if self.pe==True:
                    inputfile.write('set pe { \n')
                    inputfile.write(' potfile {} \n'.format(self.potfile))
                    inputfile.write('}\n')

                #Writing job directive
                inputfile.write('\n')

                if restart==True:
                    #function add .npy extension to lastrestart.180
                    inputfile.write('wfn = core.Wavefunction.from_file(\'{}\')\n'.format('lastrestart.180'))
                    inputfile.write('newfile = wfn.get_scratch_filename(180)\n')
                    inputfile.write('wfn.to_file(newfile)\n')
                    inputfile.write('\n')

                #RUNNING
                if Grad==True:
                    #inputfile.write('scf_energy, wfn = gradient(\'scf\', dft_functional=\'{}\', return_wfn=True)\n'.format(self.psi4functional))
                    inputfile.write("energy, wfn = gradient(\'{}\', return_wfn=True)\n".format(self.psi4method))
                    inputfile.write("print(\"FINAL TOTAL ENERGY :\", wfn.energy())")
                else:
                    #inputfile.write('scf_energy, wfn = energy(\'scf\', dft_functional=\'{}\', return_wfn=True)\n'.format(self.psi4functional))
                    inputfile.write('energy, wfn = energy(\'{}\', return_wfn=True)\n'.format(self.psi4method))
                    inputfile.write("print(\"FINAL TOTAL ENERGY :\", energy)")
                    inputfile.write('\n')

            print("Running inputfile:", self.label+'.inp')
            #Running inputfile
            with open(self.label + '.out', 'w') as ofile:
                #Psi4 -m option for saving 180 file
                print("nprocs:", nprocs)
                process = sp.run(['psi4', '-m', '-i', self.label + '.inp', '-o', self.label + '.out', '-n', '{}'.format(str(nprocs)) ], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

            #Keep restart file 180 as lastrestart.180
            try:
                restartfile=glob.glob(self.label+'*180.npy')[0]
                print("restartfile:", restartfile)
                print("Psi4 Done. Renaming {} to lastrestart.180.npy".format(restartfile))
                os.rename(restartfile, 'lastrestart.180.npy')
            except:
                pass

            #Delete big WF files. Todo: move to cleanup function?
            wffiles=glob.glob('*.34')
            for wffile in wffiles:
                os.remove(wffile)

            #Grab energy and possibly gradient
            self.energy, self.gradient = grabPsi4EandG(self.label + '.out', len(qm_elems), Grad)

            #TODO: write in error handling here

            print(BC.OKBLUE, BC.BOLD, "------------ENDING PSI4-INTERFACE-------------", BC.END)

            if Grad == True:
                print("Single-point PSI4 energy:", self.energy)
                return self.energy, self.gradient
            else:
                print("Single-point PSI4 energy:", self.energy)
                return self.energy
        else:
            print("Unknown Psi4 runmode")
            exit()



#PySCF Theory object. Fragment object is optional. Only used for single-points.
#PySCF runmode: Library only
# PE: Polarizable embedding (CPPE). Not completely active in PySCF 1.7.1. Bugfix required I think
class PySCFTheory:
    def __init__(self, fragment='', charge='', mult='', printsetting='False', printlevel=2, pyscfbasis='', pyscffunctional='',
                 pe=False, potfile='', outputname='pyscf.out', pyscfmemory=3100, nprocs=1):


        #Printlevel
        self.printlevel=printlevel

        self.nprocs=nprocs

        self.pyscfmemory=pyscfmemory
        self.outputname=outputname
        self.printsetting=printsetting
        #CPPE Polarizable Embedding options
        self.pe=pe
        #Potfile from user or passed on via QM/MM Theory object ?
        self.potfile=potfile


        if fragment != '':
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        #print("frag elems", self.fragment.elems)
        if charge!='':
            self.charge=int(charge)
        if mult!='':
            self.mult=int(mult)
        self.pyscfbasis=pyscfbasis
        self.pyscffunctional=pyscffunctional
    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old PySCF files")
        try:
            os.remove('timer.dat')
            os.remove('pyscfoutput.dat')
        except:
            pass
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, nprocs=None, pe=False, potfile=None, restart=False ):

        if nprocs==None:
            nprocs=self.nprocs



        print(BC.OKBLUE,BC.BOLD, "------------RUNNING PYSCF INTERFACE-------------", BC.END)

        #If pe and potfile given as run argument
        if pe is not False:
            self.pe=pe
        if potfile is not None:
            self.potfile=potfile

        #Coords provided to run or else taken from initialization.
        #if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords=self.coords

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems=self.elems
            else:
                qm_elems = elems


        try:
            import pyscf
        except:
            print(BC.FAIL, "Problem importing pyscf. Make sure pyscf has been installed: pip install pyscf", BC.END)
            exit(9)
        #PySCF scratch dir. Todo: Need to adapt
        #print("Setting PySCF scratchdir to ", os.getcwd())

        from pyscf import gto
        from pyscf import scf
        from pyscf import lib
        from pyscf.dft import xcfun
        if self.pe==True:
            import pyscf.solvent as solvent
            from pyscf.solvent import pol_embed
            import cppe

        #Defining mol object
        mol = gto.Mole()
        #Not very verbose system printing
        mol.verbose = 3
        coords_string=create_coords_string(qm_elems,current_coords)
        mol.atom = coords_string
        mol.symmetry = 1
        mol.charge = self.charge
        mol.spin = self.mult-1
        #PYSCF basis object: https://sunqm.github.io/pyscf/tutorial.html
        #Object can be string ('def2-SVP') or a dict with element-specific keys and values
        mol.basis=self.pyscfbasis
        #Memory settings
        mol.max_memory = self.pyscfmemory
        #BUILD mol object
        mol.build()
        if self.pe==True:
            print(BC.OKGREEN, "Polarizable Embedding Option On! Using CPPE module inside PySCF", BC.END)
            print(BC.WARNING, "Potfile: ", self.potfile, BC.END)
            try:
                if os.path.exists(self.potfile):
                    pass
                else:
                    print(BC.FAIL, "Potfile: ", self.potfile, "does not exist!", BC.END)
                    exit()
            except:
                exit()

            # TODO: Adapt to RKS vs. UKS etc.
            mf = solvent.PE(scf.RKS(mol), self.potfile)
        else:

            if PC is True:
                # QM/MM pointcharge embedding
                #mf = mm_charge(dft.RKS(mol), [(0.5, 0.6, 0.8)], MMcharges)
                mf = mm_charge(dft.RKS(mol), current_MM_coords, MMcharges)

            else:
                #TODO: Adapt to RKS vs. UKS etc.
                mf = scf.RKS(mol)
                #Verbose printing. TODO: put somewhere else
            mf.verbose=4


        #Printing settings.
        if self.printsetting==True:
            print("Printsetting = True. Printing output to stdout...")
            #np.set_printoptions(linewidth=500) TODO: not sure
        else:
            print("Printsetting = False. Printing to:", self.outputname )
            mf.stdout = open(self.outputname, 'w')


        #TODO: Restart settings for PySCF

        #Controlling OpenMP parallelization.
        lib.num_threads(nprocs)

        #Setting functional
        mf.xc = self.pyscffunctional
        #TODO: libxc vs. xcfun interface control here
        #mf._numint.libxc = xcfun

        mf.conv_tol = 1e-8
        #Control printing here. TOdo: make variable
        mf.verbose = 4



        #RUN ENERGY job. mf object should have been wrapped by PE or PC here
        result = mf.run()
        self.energy = result.e_tot
        print("SCF energy components:", result.scf_summary)
        
        #if self.pe==True:
        #    print(mf._pol_embed.cppe_state.summary_string)

        #Grab energy and gradient
        if Grad==True:
            if PC is True:
                print("THIS IS NOT CONFIRMED TO WORK!!!!!!!!!!!!")
                print("Units need to be checked.")
                hfg = mm_charge_grad(grad.dft.RKS(mf), current_MM_coords, MMcharges)
                #                grad = mf.nuc_grad_method()
                self.gradient = hfg.kernel()
            else:
                grad = mf.nuc_grad_method()
                self.gradient = grad.kernel()


        #TODO: write in error handling here
        print(BC.OKBLUE, BC.BOLD, "------------ENDING PYSCF INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point PySCF energy:", self.energy)
            return self.energy, self.gradient
        else:
            print("Single-point PySCF energy:", self.energy)
            return self.energy


#CFour Theory object. Fragment object is optional. Used??
class CFourTheory:
    def __init__(self, fragment=None, charge=None, mult=None, printlevel=2, cfourbasis=None, cfourmethod=None,
                cfourmemory=3100, nprocs=1):

        #Printlevel
        self.printlevel=printlevel

        self.charge=charge
        self.mult=mult
        self.cfourbasis=cfourbasis
        self.cfourmethod=cfourmethod
        self.cfourmemory=cfourmemory
        self.nprocs=nprocs

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, nprocs=None, restart=False):

        if nprocs == None:
            nprocs = self.nprocs

        print(BC.OKBLUE, BC.BOLD, "------------RUNNING CFOUR INTERFACE-------------", BC.END)


        # Coords provided to run or else taken from initialization.
        # if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords = self.coords

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems = self.elems
            else:
                qm_elems = elems


        def write_cfour_input(method,basis,reference,charge,mult,frozencore,memory):
            with open("ZMAT", 'w') as inpfile:
                inpfile.write('ASH-created inputfile\n')
                for el,c in zip(elems,qm_elems):
                    inpfile.write('{} {} {} {}\n'.format(el,c[0],c[1],c[2]))
                inpfile.write('\n')
                inpfile.write('*CFOUR(CALC={},BASIS={},COORD=CARTESIAN,REF={},CHARGE={},MULT={},FROZEN_CORE={},GEO_MAXCYC=1,MEM_UNIT=MB,MEMORY={})\n'.format(
                    method,basis,reference,charge,mult,frozencore,memory))

        def run_cfour(cfourdir):
            fdg="dsgfs"


        #Grab energy and gradient
        #TODO: No qm/MM yet. need to check if possible in CFour
        if Grad==True:

            write_cfour_input(self.method,self.basis,self.reference,self.charge,self.mult,self.frozen_core,self.memory)
            run_cfour(self.cfourdir)
            self.energy = 0.0
            #self.gradient = X
        else:
            write_cfour_input(self.method,self.basis,self.reference,self.charge,self.mult,self.frozen_core,self.memory)
            run_cfour(self.cfourdir)
            self.energy = 0.0

        #TODO: write in error handling here
        print(BC.OKBLUE, BC.BOLD, "------------ENDING CFOUR INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point CFour energy:", self.energy)
            return self.energy, self.gradient
        else:
            print("Single-point CFour energy:", self.energy)
            return self.energy

#MRCC Theory object. Fragment object is optional. Used??
class MRCCTheory:
    def __init__(self, fragment=None, charge=None, mult=None, printlevel=2, cfourbasis=None, cfourmethod=None,
                mrccmemory=3100, nprocs=1):

        #Printlevel
        self.printlevel=printlevel

        self.charge=charge
        self.mult=mult
        self.cfourbasis=cfourbasis
        self.cfourmethod=cfourmethod
        self.mrccmemory=mrccmemory
        self.nprocs=nprocs

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, nprocs=None, restart=False):

        if nprocs == None:
            nprocs = self.nprocs

        print(BC.OKBLUE, BC.BOLD, "------------RUNNING MRCC INTERFACE-------------", BC.END)


        # Coords provided to run or else taken from initialization.
        # if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords = self.coords

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems = self.elems
            else:
                qm_elems = elems


        def write_cfour_input(method,basis,reference,charge,mult,frozencore,memory):
            with open("ZMAT", 'w') as inpfile:
                inpfile.write('ASH-created inputfile\n')
                for el,c in zip(elems,qm_elems):
                    inpfile.write('{} {} {} {}\n'.format(el,c[0],c[1],c[2]))
                inpfile.write('\n')
                inpfile.write('*CFOUR(CALC={},BASIS={},COORD=CARTESIAN,REF={},CHARGE={},MULT={},FROZEN_CORE={},GEO_MAXCYC=1,MEM_UNIT=MB,MEMORY={})\n'.format(
                    method,basis,reference,charge,mult,frozencore,memory))

        def run_cfour(cfourdir):
            fdg="dsgfs"


            #Grab energy and gradient
            #TODO: No qm/MM yet. need to check if possible in MRCC
            if Grad==True:

                write_cfour_input(self.method,self.basis,self.reference,self.charge,self.mult,self.frozen_core,self.memory)
                run_cfour(self.cfourdir)
                self.energy = 0.0
                #self.gradient = X
            else:
                write_cfour_input(self.method,self.basis,self.reference,self.charge,self.mult,self.frozen_core,self.memory)
                run_cfour(self.cfourdir)
                self.energy = 0.0

            #TODO: write in error handling here
            print(BC.OKBLUE, BC.BOLD, "------------ENDING MRCC INTERFACE-------------", BC.END)
            if Grad == True:
                print("Single-point MRCC energy:", self.energy)
                return self.energy, self.gradient
            else:
                print("Single-point MRCC energy:", self.energy)
                return self.energy


# Fragment class
class Fragment:
    def __init__(self, coordsstring=None, fragfile=None, xyzfile=None, pdbfile=None, chemshellfile=None, coords=None, elems=None, connectivity=None,
                 atomcharges=None, atomtypes=None, conncalc=True, scale=None, tol=None, printlevel=2, charge=None,
                 mult=None, label=None, readchargemult=False):
        #Label for fragment (string). Useful for distinguishing different fragments
        self.label=label

        #Printlevel. Default: 2 (slightly verbose)
        self.printlevel=printlevel

        #New. Charge and mult attribute of fragment. Useful for workflows
        self.charge = charge
        self.mult = mult

        if self.printlevel >= 2:
            print("New ASH fragment object")
        self.energy = None
        self.elems=[]
        self.coords=[]
        self.connectivity=[]
        self.atomcharges = []
        self.atomtypes = []
        self.Centralmainfrag = []
        self.formula = None
        if atomcharges is not None:
            self.atomcharges=atomcharges
        if atomtypes is not None:
            self.atomtypes=atomtypes

        # Something perhaps only used by molcrys but defined here. Needed for print_system
        # Todo: revisit this
        self.fragmenttype_labels=[]
        #Here either providing coords, elems as lists. Possibly reading connectivity also
        if coords is not None:
            #self.add_coords(coords,elems,conn=conncalc)
            #Adding coords as list of lists. Possible conversion from numpy array below.
            self.coords=[list(i) for i in coords]
            self.elems=elems
            self.update_attributes()
            #If connectivity passed
            if connectivity is not None:
                conncalc=False
                self.connectivity=connectivity
            #If connectivity requested (default for new frags)
            if conncalc==True:
                self.calc_connectivity(scale=scale, tol=tol)
        #If coordsstring given, read elems and coords from it
        elif coordsstring is not None:
            self.add_coords_from_string(coordsstring, scale=scale, tol=tol)
        #If xyzfile argument, run read_xyzfile
        elif xyzfile is not None:
            self.read_xyzfile(xyzfile, readchargemult=readchargemult,conncalc=conncalc)
        elif pdbfile is not None:
            self.read_pdbfile(pdbfile, conncalc=conncalc)
        elif chemshellfile is not None:
            self.read_chemshellfile(chemshellfile, conncalc=conncalc)
        elif fragfile is not None:
            self.read_fragment_from_file(fragfile)
    def update_attributes(self):
        self.nuccharge = nucchargelist(self.elems)
        self.numatoms = len(self.coords)
        self.atomlist = list(range(0, self.numatoms))
        #Unnecessary alias ? Todo: Delete
        self.allatoms = self.atomlist
        self.mass = totmasslist(self.elems)
        self.list_of_masses = list_of_masses(self.elems)
        #Elemental formula
        self.formula = elemlisttoformula(self.elems)
        #Pretty formula without 1
        self.prettyformula = self.formula.replace('1','')

        if self.printlevel >= 2:
            print("Fragment numatoms: {} Formula: {}  Label: {}".format(self.numatoms,self.prettyformula,self.label))

    #Add coordinates from geometry string. Will replace.
    #Todo: Needs more work as elems and coords may be lists or numpy arrays
    def add_coords_from_string(self, coordsstring, scale=None, tol=None):
        if self.printlevel >= 2:
            print("Getting coordinates from string:", coordsstring)
        if len(self.coords)>0:
            if self.printlevel >= 2:
                print("Fragment already contains coordinates")
                print("Adding extra coordinates")
        coordslist=coordsstring.split('\n')
        for count, line in enumerate(coordslist):
            if len(line)> 1:
                self.elems.append(line.split()[0])
                self.coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        self.update_attributes()
        self.calc_connectivity(scale=scale, tol=tol)
    #Replace coordinates by providing elems and coords lists. Optional: recalculate connectivity
    def replace_coords(self, elems, coords, conn=False, scale=None, tol=None):
        if self.printlevel >= 2:
            print("Replacing coordinates in fragment.")
        
        self.elems=elems
        # Adding coords as list of lists. Possible conversion from numpy array below.
        self.coords = [list(i) for i in coords]
        self.update_attributes()
        if conn==True:
            self.calc_connectivity(scale=scale, tol=tol)
    def delete_coords(self):
        self.coords=[]
        self.elems=[]
        self.connectivity=[]
    def add_coords(self, elems,coords,conn=True, scale=None, tol=None):
        if self.printlevel >= 2:
            print("Adding coordinates to fragment.")
        if len(self.coords)>0:
            if self.printlevel >= 2:
                print("Fragment already contains coordinates")
                print("Adding extra coordinates")
        print(elems)
        print(type(elems))
        self.elems = self.elems+list(elems)
        self.coords = self.coords+coords
        self.update_attributes()
        if conn==True:
            self.calc_connectivity(scale=scale, tol=tol)
    def print_coords(self):
        if self.printlevel >= 2:
            print("Defined coordinates (Å):")
        print_coords_all(self.coords,self.elems)
    #Read Amber coordinate file?
    def read_amberinpcrdfile(self,filename,conncalc=False):
        #Todo: finish
        pass
    #Read GROMACS coordinates file
    def read_grofile(self,filename,conncalc=False):
        #Todo: finish
        pass
    #Read CHARMM? coordinate file?
    def read_charmmfile(self,filename,conncalc=False):
        #Todo: finish
        pass
    def read_chemshellfile(self,filename,conncalc=False, scale=None, tol=None):
        #Read Chemshell fragment file (.c ending)
        if self.printlevel >= 2:
            print("Reading coordinates from Chemshell file \"{}\" into fragment".format(filename))
        try:
            elems, coords = read_fragfile_xyz(filename)
        except FileNotFoundError:
            print("File {} not found".format(filename))
            exit()
        self.coords = coords
        self.elems = elems

        self.update_attributes()
        if conncalc is True:
            self.calc_connectivity(scale=scale, tol=tol)
        else:
            # Read connectivity list
            print("reading conn from file")
            print("this is not ready")
        #exit()

    #Read PDB file
    def read_pdbfile(self,filename,conncalc=True, scale=None, tol=None):
        if self.printlevel >= 2:
            print("Reading coordinates from PDBfile \"{}\" into fragment".format(filename))
        residuelist=[]
        #If elemcolumn found
        elemcol=[]
        #Not atomtype but atomname
        atom_name=[]
        atomindex=[]
        residname=[]

        #TODO: Check. Are there different PDB formats?
        #used this: https://cupnet.net/pdb-format/
        try:
            with open(filename) as f:
                for line in f:
                    if 'ATOM' in line:
                        atomindex.append(float(line[6:11].replace(' ','')))
                        atom_name.append(line[12:16].replace(' ',''))
                        residname.append(line[17:20].replace(' ',''))
                        residuelist.append(line[22:26].replace(' ',''))
                        coords_x=float(line[30:38].replace(' ',''))
                        coords_y=float(line[38:46].replace(' ',''))
                        coords_z=float(line[46:54].replace(' ',''))
                        self.coords.append([coords_x,coords_y,coords_z])
                        elem=line[76:78].replace(' ','')
                        if len(elem) != 0:
                            if len(elem)==2:
                                #Making sure second elem letter is lowercase
                                elemcol.append(elem[0]+elem[1].lower())
                            else:
                                elemcol.append(elem)    
                        #self.coords.append([float(line.split()[6]), float(line.split()[7]), float(line.split()[8])])
                        #elemcol.append(line.split()[-1])
                        #residuelist.append(line.split()[3])
                        #atom_name.append(line.split()[3])
                    if 'HETATM' in line:
                        print("HETATM line in file found. Please rename to ATOM")
                        exit()
        except FileNotFoundError:
            print("File {} does not exist!".format(filename))
            exit()
        if len(elemcol) != len(self.coords):
            print("len coords", len(self.coords))
            print("len elemcol", len(elemcol))            
            print("did not find same number of elements as coordinates")
            print("Need to define elements in some other way")
            exit()
        else:
            self.elems=elemcol
        self.update_attributes()
        if conncalc is True:
            self.calc_connectivity(scale=scale, tol=tol)
    #Read XYZ file
    def read_xyzfile(self,filename, scale=None, tol=None, readchargemult=False,conncalc=True):
        if self.printlevel >= 2:
            print("Reading coordinates from XYZfile {} into fragment".format(filename))
        with open(filename) as f:
            for count,line in enumerate(f):
                if count == 0:
                    self.numatoms=int(line.split()[0])
                elif count == 1:
                    if readchargemult is True:
                        self.charge=int(line.split()[0])
                        self.mult=int(line.split()[1])
                elif count > 1:
                    if len(line) > 3:
                        self.elems.append(line.split()[0])
                        self.coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        if self.numatoms != len(self.coords):
            print("Number of atoms in header not equal to number of coordinate-lines. Check XYZ file!")
            exit()
        self.update_attributes()
        if conncalc is True:
            self.calc_connectivity(scale=scale, tol=tol)
    def set_energy(self,energy):
        self.energy=float(energy)
    # Get coordinates for specific atoms (from list of atom indices)
    def get_coords_for_atoms(self, atoms):
        subcoords=[self.coords[i] for i in atoms]
        subelems=[self.elems[i] for i in atoms]
        return subcoords,subelems
    #Calculate connectivity (list of lists) of coords
    def calc_connectivity(self, conndepth=99, scale=None, tol=None, codeversion='julia' ):
        #Using py version if molecule is small. Otherwise Julia by default
        if len(self.coords) < 100:
            codeversion='py'
        elif len(self.coords) > 10000:
            if self.printlevel >= 2:
                print("Atom number > 10K. Connectivity calculation could take a while")

        
        if scale == None:
            try:
                scale = settings_ash.scale
                tol = settings_ash.tol
                if self.printlevel >= 2:
                    print("Using global scale and tol parameters from settings_ash. Scale: {} Tol: {} ".format(scale, tol ))

            except:
                scale = 1.0
                tol = 0.1
                if self.printlevel >= 2:
                    print("Exception: Using hard-coded scale and tol parameters. Scale: {} Tol: {} ".format(scale, tol ))
        else:
            if self.printlevel >= 2:
                print("Using scale: {} and tol: {} ".format(scale, tol))

        #Setting scale and tol as part of object for future usage (e.g. QM/MM link atoms)
        self.scale = scale
        self.tol = tol

        # Calculate connectivity by looping over all atoms
        timestampA=time.time()
        
        
        if codeversion=='py':
            print("Calculating connectivity of fragment using py")
            timestampB = time.time()
            fraglist = calc_conn_py(self.coords, self.elems, conndepth, scale, tol)
            print_time_rel(timestampB, modulename='calc connectivity py')
        elif codeversion=='julia':
            print("Calculating connectivity of fragment using julia")
            # Import Julia
            try:
                from julia.api import Julia
                from julia import Main
                timestampB = time.time()
                fraglist_temp = Main.Juliafunctions.calc_connectivity(self.coords, self.elems, conndepth, scale, tol,
                                                                      eldict_covrad)
                fraglist = []
                # Converting from numpy to list of lists
                for sublist in fraglist_temp:
                    fraglist.append(list(sublist))
                print_time_rel(timestampB, modulename='calc connectivity julia')
            except:
                print(BC.FAIL,"Problem importing Pyjulia (import julia)", BC.END)
                print("Make sure Julia is installed and PyJulia module available, and that you are using python-jl")
                print(BC.FAIL,"Using Python version instead (slow for large systems)", BC.END)
                fraglist = calc_conn_py(self.coords, self.elems, conndepth, scale, tol)



        if self.printlevel >= 2:
            pass
            #print_time_rel(timestampA, modulename='calc connectivity full')
        #flat_fraglist = [item for sublist in fraglist for item in sublist]
        self.connectivity=fraglist
        #Calculate number of atoms in connectivity list of lists
        conn_number_sum=0
        for l in self.connectivity:
            conn_number_sum+=len(l)
        if self.numatoms != conn_number_sum:
            print(BC.FAIL,"Connectivity problem", BC.END)
            exit()
        self.connected_atoms_number=conn_number_sum

    def update_atomcharges(self, charges):
        self.atomcharges = charges
    def update_atomtypes(self, types):
        self.atomtypes = types
    #Adding fragment-type info (used by molcrys, identifies whether atom is mainfrag, counterfrag1 etc.)
    #Old slow version below. To be deleted
    def old_add_fragment_type_info(self,fragmentobjects):
        # Create list of fragment-type label-list
        self.fragmenttype_labels = []
        for i in self.atomlist:
            for count,fobject in enumerate(fragmentobjects):
                if i in fobject.flat_clusterfraglist:
                    self.fragmenttype_labels.append(count)
    #Adding fragment-type info (used by molcrys, identifies whether atom is mainfrag, counterfrag1 etc.)
    #This one is fast
    def add_fragment_type_info(self,fragmentobjects):
        # Create list of fragment-type label-list
        combined_flat_clusterfraglist = []
        combined_flat_labels = []
        #Going through objects, getting flat atomlists for each object and combine (combined_flat_clusterfraglist)
        #Also create list of labels (using fragindex) for each atom
        for fragindex,frago in enumerate(fragmentobjects):
            combined_flat_clusterfraglist.extend(frago.flat_clusterfraglist)
            combined_flat_labels.extend([fragindex]*len(frago.flat_clusterfraglist))
        #Getting indices required to sort atomindices in ascending order
        sortindices = np.argsort(combined_flat_clusterfraglist)
        #labellist contains unsorted list of labels
        #Now ordering the labels according to the sort indices
        self.fragmenttype_labels =  [combined_flat_labels[i] for i in sortindices]

    #Molcrys option:
    def add_centralfraginfo(self,list):
        self.Centralmainfrag = list
    def write_xyzfile(self,xyzfilename="Fragment-xyzfile.xyz"):
        #Energy written to XYZ title-line if present. Otherwise: None
        with open(xyzfilename, 'w') as ofile:
            ofile.write(str(len(self.elems)) + '\n')
            if self.energy is None:
                ofile.write("Energy: None" + '\n')
            else:
                ofile.write("Energy: {:14.8f}".format(self.energy) + '\n')
            for el, c in zip(self.elems, self.coords):
                line = "{:4} {:14.8f} {:14.8f} {:14.8f}".format(el, c[0], c[1], c[2])
                ofile.write(line + '\n')
        if self.printlevel >= 2:
            print("Wrote XYZ file:", xyzfilename)
    #Print system-fragment information to file. Default name of file: "fragment.ygg
    def print_system(self,filename='fragment.ygg'):
        if self.printlevel >= 2:
            print("Printing fragment to disk:", filename)

        #Setting atomcharges, fragmenttype_labels and atomtypes to dummy lists if empty
        if len(self.atomcharges)==0:
            self.atomcharges=[0.0 for i in range(0,self.numatoms)]
        if len(self.fragmenttype_labels)==0:
            self.fragmenttype_labels=[0 for i in range(0,self.numatoms)]
        if len(self.atomtypes)==0:
            self.atomtypes=['None' for i in range(0,self.numatoms)]

        with open(filename, 'w') as outfile:
            outfile.write("Fragment: \n")
            outfile.write("Num atoms: {}\n".format(self.numatoms))
            outfile.write("Formula: {}\n".format(self.formula))
            outfile.write("Energy: {}\n".format(self.energy))
            outfile.write("\n")
            outfile.write(" Index    Atom         x                  y                  z               charge        fragment-type        atom-type\n")
            outfile.write("---------------------------------------------------------------------------------------------------------------------------------\n")
            for at, el, coord, charge, label, atomtype in zip(self.atomlist, self.elems,self.coords,self.atomcharges, self.fragmenttype_labels, self.atomtypes):
                line="{:>6} {:>6}  {:17.11f}  {:17.11f}  {:17.11f}  {:14.8f} {:12d} {:>21}\n".format(at, el,coord[0], coord[1], coord[2], charge, label, atomtype)
                outfile.write(line)
            outfile.write(
                "===========================================================================================================================================\n")
            #outfile.write("elems: {}\n".format(self.elems))
            #outfile.write("coords: {}\n".format(self.coords))
            #outfile.write("list of masses: {}\n".format(self.list_of_masses))
            outfile.write("atomcharges: {}\n".format(self.atomcharges))
            outfile.write("Sum of atomcharges: {}\n".format(sum(self.atomcharges)))
            outfile.write("atomtypes: {}\n".format(self.atomtypes))
            outfile.write("connectivity: {}\n".format(self.connectivity))
            outfile.write("Centralmainfrag: {}\n".format(self.Centralmainfrag))

    #Reading fragment from file. File created from Fragment.print_system
    def read_fragment_from_file(self, fragfile):
        if self.printlevel >= 2:
            print("Reading ASH fragment from file:", fragfile)
        coordgrab=False
        coords=[]
        elems=[]
        atomcharges=[]
        atomtypes=[]
        fragment_type_labels=[]
        connectivity=[]
        #Only used by molcrys:
        Centralmainfrag = []
        with open(fragfile) as file:
            for n, line in enumerate(file):
                if 'Num atoms:' in line:
                    numatoms=int(line.split()[-1])
                if coordgrab==True:
                    #If end of coords section
                    if '===============' in line:
                        coordgrab=False
                        continue
                    elems.append(line.split()[1])
                    coords.append([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                    atomcharges.append(float(line.split()[5]))
                    fragment_type_labels.append(int(line.split()[6]))
                    atomtypes.append(line.split()[7])

                if '--------------------------' in line:
                    coordgrab=True
                if 'Centralmainfrag' in line:
                    if '[]' not in line:
                        l = line.lstrip('Centralmainfrag:')
                        l = l.replace('\n','')
                        l = l.replace(' ','')
                        l = l.replace('[','')
                        l = l.replace(']','')
                        Centralmainfrag = [int(i) for i in l.split(',')]
                #Incredibly ugly but oh well
                if 'connectivity:' in line:
                    l=line.lstrip('connectivity:')
                    l=l.replace(" ", "")
                    for x in l.split(']'):
                        if len(x) < 1:
                            break
                        y=x.strip(',[')
                        y=y.strip('[')
                        y=y.strip(']')
                        list=[int(i) for i in y.split(',')]
                        connectivity.append(list)
        self.elems=elems
        self.coords=coords
        self.atomcharges=atomcharges
        self.atomtypes=atomtypes
        self.update_attributes()
        self.connectivity=connectivity
        self.Centralmainfrag = Centralmainfrag



# https://github.com/grimme-lab/xtb/blob/master/python/xtb/interface.py
#Now supports 2 runmodes: 'library' (fast Python C-API) or 'inputfile'
#
class xTBTheory:
    def __init__(self, xtbdir=None, fragment=None, charge=None, mult=None, xtbmethod=None, runmode='inputfile', nprocs=1, printlevel=2):

        #Printlevel
        self.printlevel=printlevel

        if xtbmethod is None:
            print("xTBTheory requires xtbmethod keyword to be set")
            exit(1)

        self.nprocs=nprocs
        if fragment != None:
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        self.charge=charge
        self.mult=mult
        self.xtbmethod=xtbmethod
        self.runmode=runmode
        if self.runmode=='library':
            print("Using library-based xTB interface")
            print("Loading library...")
            os.environ["OMP_NUM_THREADS"] = str(nprocs)
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            # Load xtB library and ctypes datatypes that run uses
            try:
                #import xtb_interface_library
                import interface_xtb
                self.xtbobject = interface_xtb.XTBLibrary()
            except:
                print("Problem importing xTB library. Check that the library dir (containing libxtb.so) is available in LD_LIBRARY_PATH.")
                print("e.g. export LD_LIBRARY_PATH=/path/to/xtb_6.2.3/lib64:$LD_LIBRARY_PATH")
                print("Or that the MKL library is available and loaded")
                exit(9)
            from ctypes import c_int, c_double
            #Needed for complete interface?:
            # from ctypes import Structure, c_int, c_double, c_bool, c_char_p, c_char, POINTER, cdll, CDLL
            self.c_int = c_int
            self.c_double = c_double
        else:
            if xtbdir == None:
                # Trying to find xtbdir in path
                print("xtbdir argument not provided to xTBTheory object. Trying to find xtb in path")
                try:
                    self.xtbdir = os.path.dirname(shutil.which('xtb'))
                    print("Found xtb in path. Setting xtbdir.")
                except:
                    print("Found no xtb executable in path. Exiting... ")
            else:
                self.xtbdir = xtbdir
    #Cleanup after run.
    def cleanup(self):
        if self.printlevel >= 2:
            print("Cleaning up old xTB files")
        try:
            os.remove('xtb-inpfile.xyz')
            os.remove('xtb-inpfile.out')
            os.remove('gradient')
            os.remove('charges')
            os.remove('energy')
            os.remove('xtbrestart')
        except:
            pass
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
                elems=None, Grad=False, PC=False, nprocs=None):
        if MMcharges is None:
            MMcharges=[]

        if nprocs is None:
            nprocs=self.nprocs

        if self.printlevel >= 2:
            print("------------STARTING XTB INTERFACE-------------")
        #Coords provided to run or else taken from initialization.
        #if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords=self.coords

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems=self.elems
            else:
                qm_elems = elems


        #Parallellization
        #Todo: this has not been confirmed to work
        #Needs to be done before library-import??
        os.environ["OMP_NUM_THREADS"] = str(nprocs)
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        if self.runmode=='inputfile':
            if self.printlevel >=2:
                print("Using inputfile-based xTB interface")
            #TODO: Add restart function so that xtbrestart is not always deleted
            #Create XYZfile with generic name for xTB to run
            inputfilename="xtb-inpfile"
            if self.printlevel >= 2:
                print("Creating inputfile:", inputfilename+'.xyz')
            num_qmatoms=len(current_coords)
            num_mmatoms=len(MMcharges)
            self.cleanup()
            #Todo: xtbrestart possibly. needs to be optional
            functions_coords.write_xyzfile(qm_elems, current_coords, inputfilename,printlevel=self.printlevel)

            #Run inputfile. Take nprocs argument.
            if self.printlevel >= 2:
                print("------------Running xTB-------------")
                print("...")
            if Grad==True:
                if PC==True:
                    create_xtb_pcfile_general(current_MM_coords, MMcharges)
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, inputfilename + '.xyz', self.charge, self.mult, Grad=True)
                else:
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, inputfilename + '.xyz', self.charge, self.mult,
                                  Grad=True)
            else:
                if PC==True:
                    create_xtb_pcfile_general(current_MM_coords, MMcharges)
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, inputfilename + '.xyz', self.charge, self.mult)
                else:
                    run_xtb_SP_serial(self.xtbdir, self.xtbmethod, inputfilename + '.xyz', self.charge, self.mult)

            if self.printlevel >= 2:
                print("------------xTB calculation done-----")
            #Check if finished. Grab energy
            if Grad==True:
                self.energy,self.grad=xtbgradientgrab(num_qmatoms)
                if PC==True:
                    # Grab pointcharge gradient. i.e. gradient on MM atoms from QM-MM elstat interaction.
                    self.pcgrad = xtbpcgradientgrab(num_mmatoms)
                    if self.printlevel >= 2:
                        print("xtb energy :", self.energy)
                        print("------------ENDING XTB-INTERFACE-------------")

                    return self.energy, self.grad, self.pcgrad
                else:
                    if self.printlevel >= 2:
                        print("xtb energy :", self.energy)
                        print("------------ENDING XTB-INTERFACE-------------")
                    return self.energy, self.grad
            else:
                outfile=inputfilename+'.out'
                self.energy=xtbfinalenergygrab(outfile)
                if self.printlevel >= 2:
                    print("xtb energy :", self.energy)
                    print("------------ENDING XTB-INTERFACE-------------")
                return self.energy
        elif self.runmode=='library':

            if PC==True:
                print("Pointcharge-embedding on but xtb-runmode is library!")
                print("The xtb library-interface is not yet ready for QM/MM calculations")
                print("Use runmode='inputfile' for now")
                exit(1)


            #Hard-coded options. Todo: revisit
            options = {
                "print_level": 1,
                "parallel": 0,
                "accuracy": 1.0,
                "electronic_temperature": 300.0,
                "gradient": True,
                "restart": False,
                "ccm": True,
                "max_iterations": 30,
                "solvent": "none",
            }

            #Using the xtbobject previously defined
            num_qmatoms=len(current_coords)
            #num_mmatoms=len(MMcharges)
            nuc_charges=np.array(elemstonuccharges(qm_elems), dtype=self.c_int)

            #Converting coords to numpy-array and then to Bohr.
            current_coords_bohr=np.array(current_coords)*constants.ang2bohr
            positions=np.array(current_coords_bohr, dtype=self.c_double)
            args = (num_qmatoms, nuc_charges, positions, options, 0.0, 0, "-")
            print("------------Running xTB-------------")
            if self.xtbmethod=='GFN1':
                results = self.xtbobject.GFN1Calculation(*args)
            elif self.xtbmethod=='GFN2':
                results = self.xtbobject.GFN2Calculation(*args)
            else:
                print("Unknown xtbmethod.")
                exit()
            print("------------xTB calculation done-------------")
            if Grad==True:
                self.energy = float(results['energy'])
                self.grad = results['gradient']
                print("xtb energy:", self.energy)
                #print("self.grad:", self.grad)
                print("------------ENDING XTB-INTERFACE-------------")
                return self.energy, self.grad
            else:
                self.energy = float(results['energy'])
                print("xtb energy:", self.energy)
                print("------------ENDING XTB-INTERFACE-------------")
                return self.energy
        else:
            print("Unknown option to xTB interface")
            exit()


#Called from run_QMMM_SP_in_parallel. Runs
def run_QM_MM_SP(list):
    orcadir=list[0]
    current_coords=list[1]
    theory=list[2]
    #label=list[3]
    #Create new dir (name of label provided
    #Cd dir
    theory.run(Grad=True)

def run_QMMM_SP_in_parallel(orcadir, list_of__geos, list_of_labels, QMMMtheory, numcores):
    import multiprocessing as mp
    blankline()
    print("Number of CPU cores: ", numcores)
    print("Number of geos:", len(list_of__geos))
    print("Running snapshots in parallel")
    pool = mp.Pool(numcores)
    results = pool.map(run_QM_MM_SP, [[orcadir,geo, QMMMtheory ] for geo in list_of__geos])
    pool.close()
    print("Calculations are done")

#def run_displacements_in_parallel(list_of__geos, list_of_labels, QMMMtheory, numcores):
#    import multiprocessing as mp
#    blankline()
#    print("Number of CPU cores: ", numcores)
#    print("Number of geos:", len(list_of__geos))
#    print("Running displacements in parallel")
#    pool = mp.Pool(numcores)
#    results = pool.map(theory.run, [[geo, QMMMtheory ] for geo in list_of__geos])
#    pool.close()
#    print("Calculations are done")


#MMAtomobject used to store LJ parameter and possibly charge for MM atom with atomtype, e.g. OT
class AtomMMobject:
    def __init__(self, atomcharge=None, LJparameters=None):
        sf="dsf"
        self.atomcharge = atomcharge
        self.LJparameters = LJparameters
    def add_charge(self, atomcharge=None):
        self.atomcharge = atomcharge
    def add_LJparameters(self, LJparameters=None):
        self.LJparameters=LJparameters

#Makes more sense to store this here. Simplifies ASH inputfile import.
def MMforcefield_read(file):
    print("Reading forcefield file:", file)
    MM_forcefield = {}
    atomtypes=[]
    with open(file) as f:
        for line in f:
            if '#' not in line:
                if 'combination_rule' in line:
                    combrule=line.split()[-1]
                    print("Found combination rule defintion in forcefield file:", combrule)
                    MM_forcefield["combination_rule"]=combrule
                if 'charge' in line:
                    print("Found charge definition in forcefield file:", ' '.join(line.split()[:]))
                    atomtype=line.split()[1]
                    if atomtype not in MM_forcefield.keys():
                        MM_forcefield[atomtype]=AtomMMobject()
                    charge=float(line.split()[2])
                    MM_forcefield[atomtype].add_charge(atomcharge=charge)
                    # TODO: Charges defined are currently not used I think
                if 'LennardJones_i_sigma' in line:
                    print("Found LJ single-atom sigma definition in forcefield file:", ' '.join(line.split()[:]))
                    atomtype=line.split()[1]
                    if atomtype not in MM_forcefield.keys():
                        MM_forcefield[atomtype] = AtomMMobject()
                    sigma_i=float(line.split()[2])
                    eps_i=float(line.split()[3])
                    MM_forcefield[atomtype].add_LJparameters(LJparameters=[sigma_i,eps_i])
                if 'LennardJones_i_R0' in line:
                    print("Found LJ single-atom R0 definition in forcefield file:", ' '.join(line.split()[:]))
                    atomtype=line.split()[1]
                    R0tosigma=0.5**(1/6)
                    if atomtype not in MM_forcefield.keys():
                        MM_forcefield[atomtype] = AtomMMobject()
                    sigma_i=float(line.split()[2])*R0tosigma
                    eps_i=float(line.split()[3])
                    MM_forcefield[atomtype].add_LJparameters(LJparameters=[sigma_i,eps_i])
                if 'LennardJones_ij' in line:
                    print("Found LJ pair definition in forcefield file")
                    atomtype_i=line.split()[1]
                    atomtype_j=line.split()[2]
                    sigma_ij=float(line.split()[3])
                    eps_ij=float(line.split()[4])
                    print("This is incomplete. Exiting")
                    exit()
                    # TODO: Need to finish this. Should replace LennardJonespairpotentials later
    return MM_forcefield

#Better place for this?
def ReactionEnergy(stoichiometry=None, list_of_fragments=None, list_of_energies=None, unit='kcal/mol', label=None, reference=None):
    conversionfactor = { 'kcal/mol' : 627.50946900, 'kcalpermol' : 627.50946900, 'kJ/mol' : 2625.499638, 'kJpermol' : 2625.499638, 
                        'eV' : 27.211386245988, 'cm-1' : 219474.6313702 }
    if label is None:
        label=''
    #print(BC.OKBLUE,BC.BOLD, "ReactionEnergy function. Unit:", unit, BC.END)
    reactant_energy=0.0 #hartree
    product_energy=0.0 #hartree
    if stoichiometry is None:
        print("stoichiometry list is required")
        exit(1)

    #List of energies option
    if list_of_energies is not None:
        #print("List of total energies provided (Eh units assumed).")
        for i,stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy=reactant_energy+list_of_energies[i]*abs(stoich)
            if stoich > 0:
                product_energy=product_energy+list_of_energies[i]*abs(stoich)
        reaction_energy=(product_energy-reactant_energy)*conversionfactor[unit]
        if reference is None:
            error=None
            print(BC.BOLD, "Reaction_energy({}): {} {}".format(label,BC.OKGREEN,reaction_energy, unit), BC.END)
        else:
            error=reaction_energy-reference
            print(BC.BOLD, "Reaction_energy({}): {} {} {} (Error: {}) {}".format(label,BC.OKGREEN,reaction_energy, unit, error, BC.END))
    else:
        print("No list of total energies provided. Using internal energy of each fragment instead.")
        print("")
        for i,stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy=reactant_energy+list_of_fragments[i].energy*abs(stoich)
            if stoich > 0:
                product_energy=product_energy+list_of_fragments[i].energy*abs(stoich)
        reaction_energy=(product_energy-reactant_energy)*conversionfactor[unit]
        if reference is None:
            error=None
            print(BC.BOLD, "Reaction_energy({}): {} {}".format(label,BC.OKGREEN,reaction_energy, unit), BC.END)
        else:
            error=reaction_energy-reference
            print(BC.BOLD, "Reaction_energy({}): {} {} {} (Error: {})".format(label,BC.OKGREEN,reaction_energy, unit, error, BC.END))
    return reaction_energy, error


