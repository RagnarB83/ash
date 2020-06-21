# ASH - A GENERAL COMPCHEM AND QM/MM ENVIRONMENT

#TODO: This is really too much import!!!! Reduce
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

debugflag=False

import sys
import inspect



#Useful function to measure size of object:
#https://goshippo.com/blog/measure-real-size-any-python-object/
#https://github.com/bosswissam/pysize/blob/master/pysize.py
def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_size(v, seen) for v in obj.values()))
        size += sum((get_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum((get_size(i, seen) for i in obj))

    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(get_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return size


#Debug print. Behaves like print bug reads global debug var first
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



    print(BC.OKGREEN,"----------------------------------------------------------------------------------",BC.END)
    print(BC.OKGREEN,"----------------------------------------------------------------------------------",BC.END)
    #print(BC.OKBLUE,ascii_banner3,BC.END)
    #print(BC.OKBLUE,ascii_banner2,BC.END)
    print(BC.OKGREEN,ascii_banner,BC.END)
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
#Should be used by user instead of run.
def Singlepoint(fragment=None, theory=None, Grad=False):
    if fragment is None or theory is None:
        print("Singlepoint requires a fragment and a theory object")

    coords=fragment.coords
    elems=fragment.elems

    # Run a single-point energy job
    if Grad ==True:
        print("Doing single-point Energy+Gradient job")
        # An Energy+Gradient calculation where we change the number of cores to 12
        energy,gradient= theory.run(current_coords=coords, elems=elems, Grad=True)
        print("Energy: ", energy)
        return energy,gradient
    else:
        print("Doing single-point Energy job")
        energy = theory.run(current_coords=coords, elems=elems)
        print("Energy: ", energy)

        #Now adding total energy to fragment
        fragment.energy=energy

        return energy


#Numerical frequencies function
def NumFreq(fragment=None, theory=None, npoint=1, displacement=0.0005, hessatoms=None, numcores=1, runmode='serial'):
    print(BC.WARNING, BC.BOLD, "------------NUMERICAL FREQUENCIES-------------", BC.END)
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
    print("x elems:", elems)
    hesselems = get_partial_list(allatoms, hessatoms, elems)
    hessmasses = get_partial_list(allatoms, hessatoms, fragment.list_of_masses)
    hesscoords = [fragment.coords[i] for i in hessatoms]
    print("Elements:", hesselems)
    print("Masses used:", hessmasses)
    print("hesscoords:", hesscoords)
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
        thermochemcalc(frequencies,hessatoms, fragment, theory.qm_theory.mult, temp=298.18,pressure=1)
    else:
        thermochemcalc(frequencies,hessatoms, fragment, theory.mult, temp=298.18,pressure=1)


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
#        #Atom types
#    pass
#    def run(self):
#        pass



# Different MM theories


#Todo: Also think whether we want do OpenMM simulations in case we have to make another object maybe
#Amberfiles:
# Needs just amberprmtopfile (new-style Amber7 format). Not inpcrd file, read into ASH fragment instead

#CHARMMfiles:
#Need psffile, a CHARMM topologyfile (charmmtopfile) and a CHARMM parameter file (charmmprmfile)

#GROMACSfiles:
# Need gromacstopfile and grofile (contains periodic information along with coordinates) and gromacstopdir location (topology)

class OpenMMTheory:
    def __init__(self, pdbfile=None, platform='CPU', active_atoms=None, frozen_atoms=None,
                 CHARMMfiles=False, psffile=None, charmmtopfile=None, charmmprmfile=None,
                 GROMACSfiles=False, gromacstopfile=None, grofile=None, gromacstopdir=None,
                 Amberfiles=False, amberprmtopfile=None, printlevel=2, nprocs=1,
                 xmlfile=None):

        print("Setting OpenMM CPU Threads to: ", nprocs)
        print("TODO: confirm parallelization")
        os.environ["OPENMM_CPU_THREADS"] = str(nprocs)

        #Printlevel
        self.printlevel=printlevel

        print(BC.WARNING, BC.BOLD, "------------Defining OpenMM object-------------", BC.END)
        timeA = time.time()
        self.coords=[]
        self.charges=[]
        self.platform_choice=platform

        # OPEN MM load
        try:
            import simtk.openmm.app
            import simtk.unit
            #import simtk.openmm
        except ImportError:
            raise ImportError(
                "OpenMM requires installing the OpenMM package. Try: conda install -c omnia openmm  \
                Also see http://docs.openmm.org/latest/userguide/application.html")

        self.unit=simtk.unit
        self.Vec3=simtk.openmm.Vec3



        #TODO: Should we keep this? Probably not. Coordinates would be handled by ASH.
        #PDB_ygg_frag = Fragment(pdbfile=pdbfile, conncalc=False)
        #self.coords=PDB_ygg_frag.coords
        print_time_rel(timeA, modulename="prep")
        timeA = time.time()


        #What type of forcefield files to read. Reads in different way.
        # #Always creates object we call self.forcefield that contains topology attribute
        if CHARMMfiles is True:
            print("Reading CHARMM files")
            # Load CHARMM PSF files. Both CHARMM-style and XPLOR allowed I believe. Todo: Check
            self.psf = simtk.openmm.app.CharmmPsfFile(psffile)
            self.params = simtk.openmm.app.CharmmParameterSet(charmmtopfile, charmmprmfile)
            # self.pdb = simtk.openmm.app.PDBFile(pdbfile) probably not reading coordinates here
            #self.forcefield = self.psf
            self.topology = self.psf.topology
            # Create an OpenMM system by calling createSystem on psf
            self.system = self.psf.createSystem(self.params, nonbondedMethod=simtk.openmm.app.NoCutoff,
                                                nonbondedCutoff=1 * simtk.openmm.unit.nanometer)
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
        elif Amberfiles is True:
            print("Warning: Amber-file interface not tested")
            #Note: Only new-style Amber7 prmtop files work
            self.prmtop = simtk.openmm.app.AmberPrmtopFile(amberprmtopfile)
            #inpcrd = simtk.openmm.app.AmberInpcrdFile(inpcrdfile)  probably not reading coordinates here
            #self.forcefield = self.prmtop
            self.topology = self.prmtop.topology
            # Create an OpenMM system by calling createSystem on prmtop
            self.system = self.prmtop.createSystem(nonbondedMethod=simtk.openmm.app.NoCutoff,
                                                nonbondedCutoff=1 * simtk.openmm.unit.nanometer)
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

        #Define force object here, for possible modification of charges (QM/MM)
        forces = {self.system.getForce(index).__class__.__name__:
                      self.system.getForce(index) for index in range(self.system.getNumForces())}
        self.nonbonded_force = forces['NonbondedForce']


        # Get charges from OpenMM object into self.charges
        self.getatomcharges()


        print_time_rel(timeA, modulename="system create")
        timeA = time.time()
        #constraints=simtk.openmm.app.HBonds, AllBonds, HAngles

        #FROZEN AND ACTIVE ATOMS
        self.numatoms=int(self.psf.topology.getNumAtoms())
        print("self.numatoms:", self.numatoms)
        self.allatoms=list(range(0,self.numatoms))
        if active_atoms is None and frozen_atoms is None:
            print("All {} atoms active, no atoms frozen".format(len(self.allatoms)))
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

        # Remove Frozen-Frozen interactions
        #Todo: Will be requested by QMMM object so unnecessary unless during pure MM??
        #if frozen_atoms is not None:
        #    print("Removing Frozen-Frozen interactions")
        #    self.addexceptions(frozen_atoms)


        #Modify particle masses in system object. For freezing atoms
        for i in self.frozen_atoms:
            self.system.setParticleMass(i, 0 * simtk.openmm.unit.dalton)
        print_time_rel(timeA, modulename="frozen atom setup")
        timeA = time.time()

        #Modifying constraints after frozen-atom setting
        print("Constraints:", self.system.getNumConstraints())

        #Finding defined constraints that involved frozen atoms. add to remove list
        removelist=[]
        for i in range(0,self.system.getNumConstraints()):
            constraint=self.system.getConstraintParameters(i)
            if constraint[0] in self.frozen_atoms or constraint[1] in self.frozen_atoms:
                #self.system.removeConstraint(i)
                removelist.append(i)

        #print("removelist:", removelist)
        print("length removelist", len(removelist))
        #Remove constraints
        removelist.reverse()
        for r in removelist:
            self.system.removeConstraint(r)

        print("Constraints:", self.system.getNumConstraints())
        print_time_rel(timeA, modulename="constraint fix")
        timeA = time.time()
        #Dummy integrator
        self.integrator = simtk.openmm.LangevinIntegrator(300 * simtk.openmm.unit.kelvin,  # Temperature of head bath
                                        1 / simtk.openmm.unit.picosecond,  # Friction coefficient
                                        0.002 * simtk.openmm.unit.picoseconds)  # Time step
        print("self.platform_choice:", self.platform_choice)
        self.platform = simtk.openmm.Platform.getPlatformByName(self.platform_choice)


        self.simulation = simtk.openmm.app.simulation.Simulation(self.topology, self.system, self.integrator,
                                                                 self.platform)


        print_time_rel(timeA, modulename="simulation setup")
        timeA = time.time()

    #This removes interactions between particles (e.g. QM-QM or frozen-frozen pairs)
    # list of atom indices for which we will remove all pairs

    #Todo: Way too slow to do.
    # Alternative: Remove force interaction and then add in the interaction of active atoms to frozen atoms
    # should be reasonably fast
    # https://github.com/openmm/openmm/issues/2124
    #https://github.com/openmm/openmm/issues/1696
    def addexceptions(self,atomlist):
        import itertools
        print("Removing i-j interactions for list :", len(atomlist), "atoms")
        timeA=time.time()
        #Has duplicates
        #[self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i in atomlist for j in atomlist]
        #https://stackoverflow.com/questions/942543/operation-on-every-pair-of-element-in-a-list
        [self.nonbonded_force.addException(i,j,0, 0, 0, replace=True) for i,j in itertools.combinations(atomlist, r=2)]

        #for i in atomlist:
        #    for j in atomlist:
        #        self.nonbonded_force.addException(i,j,0, 0, 0, replace=True)
        print_time_rel(timeA, modulename="add exception")
    #Run: coords or framents can be given (usually coords). qmatoms in order to avoid QM-QM interactions (TODO)
    #Probably best to do QM-QM exclusions etc. in a separate function though as we want run to be as simple as possible
    def run(self, coords=None, fragment=None):
        timeA = time.time()
        print(BC.OKBLUE, BC.BOLD, "------------RUNNING OPENMM INTERFACE-------------", BC.END)
        #If no coords given to run then a single-point job probably (not part of Optimizer or MD which would supply coords).
        #Then try if fragment object was supplied.
        #Otherwise internal coords if they exist
        print("if stuff")
        if coords is None:
            if fragment is None:
                if len(self.coords) != 0:
                    print("Using internal coordinates (from OpenMM object)")
                    coords=self.coords
                else:
                    print("Found no coordinates!")
                    exit(1)
            else:
                coords=fragment.coords

        print_time_rel(timeA, modulename="if stuff")
        timeA = time.time()
        #Making sure coords is np array and not list-of-lists
        print("doing coords array")
        coords=np.array(coords)
        print_time_rel(timeA, modulename="coords array")
        timeA = time.time()
        ##  unit conversion for energy
        eqcgmx = 2625.5002
        ## unit conversion for force
        fqcgmx = -49621.9
        #pos = [Vec3(a / 10, b / 10, c / 10)] * u.nanometer


        #pos = [Vec3(coords[:,0]/10,coords[:,1]/10,coords[:,2]/10)] * u.nanometer
        #Todo: Check speed on this
        print("doing pos")
        pos = [self.Vec3(coords[i, 0] / 10, coords[i, 1] / 10, coords[i, 2] / 10) for i in range(len(coords))] * self.unit.nanometer
        print_time_rel(timeA, modulename="pos create")
        timeA = time.time()
        self.simulation.context.setPositions(pos)
        print_time_rel(timeA, modulename="context pos")
        timeA = time.time()
        print("doing state")
        state = self.simulation.context.getState(getEnergy=True, getForces=True)
        print_time_rel(timeA, modulename="state")
        timeA = time.time()
        print("doing energy")
        self.energy = state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole) / eqcgmx
        print_time_rel(timeA, modulename="energy")
        timeA = time.time()
        print("doing gradient")
        self.gradient = state.getForces(asNumpy=True).flatten() / fqcgmx
        print_time_rel(timeA, modulename="gradient")
        timeA = time.time()

        #Todo: Check units
        print("self.energy:", self.energy)
        print("self.gradient:", self.gradient)

        print(BC.OKBLUE, BC.BOLD, "------------ENDING OPENMM INTERFACE-------------", BC.END)
        return self.energy, self.gradient
    def getatomcharges(self):
        chargelist = []
        for i in range( self.nonbonded_force.getNumParticles() ):
            charge = self.nonbonded_force.getParticleParameters( i )[0]
            if isinstance(charge, self.unit.Quantity):
                charge = charge / self.unit.elementary_charge
                chargelist.append(charge)
        self.charges=chargelist
        return chargelist
    def update_charges(self,charges):
        print("Updating charges in OpenMM object.")
        #Check that force-particles and charges are same number
        print("self.nonbonded_force.getNumParticles():", self.nonbonded_force.getNumParticles())
        print(len(charges))
        print(self.nonbonded_force.getNumParticles())
        assert self.nonbonded_force.getNumParticles() == len(charges)
        newcharges=[]
        for i,newcharge in enumerate(charges):
            oldcharge, sigma, epsilon = self.nonbonded_force.getParticleParameters(i)
            self.nonbonded_force.setParticleParameters(i, newcharge, sigma, epsilon)
            newcharges.append(newcharge)
        self.charges = newcharges
        #print("OpenMMobject charges are now:", self.charges)

# Simple nonbonded MM theory. Charges and LJ-potentials
class NonBondedTheory:
    def __init__(self, atomtypes=None, forcefield=None, charges = None, LJcombrule='geometric',
                 codeversion='f2py', pairarrayversion='julia', printlevel=2):

        #Printlevel
        self.printlevel=printlevel

        #Atom types
        self.atomtypes=atomtypes
        #Read MM forcefield.
        self.forcefield=forcefield

        #
        self.numatoms = len(self.atomtypes)
        self.LJcombrule=LJcombrule

        self.pairarrayversion=pairarrayversion

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
        if self.printlevel >= 3:
            print("Final LJ pair potentials (sigma_ij, epsilon_ij):\n", self.LJpairpotentials)
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

        if self.pairarrayversion=="julia":
            if self.printlevel >= 2:
                print("Using PyJulia for fast sigmaij and epsij array creation")
            ashpath = os.path.dirname(ash.__file__)

            # Necessary for statically linked libpython
            #IF not doing python-jl
            #from julia.api import Julia
            #jl = Julia(compiled_modules=False)
            #Possibly disables deprecated warning
            #Julia(depwarn=True)
            # Import Julia
            try:
                from julia.api import Julia
                #Does not work
                #jl = Julia(depwarn=False)
                from julia import Main
            except:
                print("Problem importing Pyjulia (import julia)")
                print("Make sure Julia is installed and PyJulia module available")
                print("Also, are you using python-jl ?")
                print("Alternatively, use pairarrayversion='py' argument to NonBondedTheory to use slower Python version for array creation")
                exit(9)
            # Defining Julia Module
            Main.include(ashpath + "/functions_julia.jl")


            # Do pairpot array for whole system
            if len(actatoms) == 0:
                print("Calculating pairpotential array for whole system")
                self.sigmaij, self.epsij = Main.Juliafunctions.pairpot_full(self.numatoms, self.atomtypes, self.LJpairpotdict,qmatoms)
            else:
            #    #or only for active region
                print("Calculating pairpotential array for active region only")
                self.sigmaij, self.epsij = Main.Juliafunctions.pairpot_active(self.numatoms, self.atomtypes, self.LJpairpotdict, qmatoms, actatoms)
        # New for-loop for creating sigmaij and epsij arrays. Uses dict-lookup instead
        elif self.pairarrayversion=="py":
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
            print("unknown pairarrayversion")
            exit()

        if self.printlevel >= 2:
            print("self.sigmaij ({}) : {}".format(len(self.sigmaij), self.sigmaij))
            print("self.epsij ({}) : {}".format(len(self.epsij), self.epsij))
        print_time_rel(CheckpointTime, modulename="pairpot arrays")

    def update_charges(self,charges):
        print("Updating charges.")
        self.atom_charges = charges
        #print("Charges are now:", charges)
        print("Sum of charges:", sum(charges))
    #Provide specific coordinates (MM region) and charges (MM region) upon run
    def run(self, full_coords=None, mm_coords=None, charges=None, connectivity=None,
            Coulomb=True, Grad=True, qmatoms=None, actatoms=None, frozenatoms=None):

        #If qmatoms list provided to run (probably by QM/MM object) then we are doing QM/MM
        #QM-QM pairs will be skipped in LJ

        #Testing if self.sigmaij array has been assigned or not. If not calling calculate_LJ_pairpotentials
        #Passing qmatoms over so pairs can be skipped
        #This sets self.sigmaij and self.epsij and also self.LJpairpotentials
        #Todo: if actatoms have been defined this will be skipped in pairlist creation
        #if frozenatoms passed frozen-frozen interactions will be skipped
        if np.count_nonzero(self.sigmaij) == 0:
            self.calculate_LJ_pairpotentials(qmatoms=qmatoms,actatoms=actatoms)

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


        #Slow Python version or fast Fortran version
        if self.codeversion=='py':
            if self.printlevel >= 2:
                print("Using slow Python MM code")
            #Sending full coords and charges over. QM charges are set to 0.
            if Coulomb==True:
                self.Coulombchargeenergy, self.Coulombchargegradient  = coulombcharge(charges, full_coords)
                if self.printlevel >= 2:
                    print("Coulomb Energy (au):", self.Coulombchargeenergy)
                    print("Coulomb Energy (kcal/mol):", self.Coulombchargeenergy * constants.harkcal)
                    print("")
                    print("self.Coulombchargegradient:", self.Coulombchargegradient)
                blankline()
            # NOTE: Lennard-Jones should  calculate both MM-MM and QM-MM LJ interactions. Full coords necessary.
            if LJ==True:
                #LennardJones(coords, epsij, sigmaij, connectivity=[], qmatoms=[])
                #self.LJenergy,self.LJgradient = LennardJones(full_coords,self.atomtypes, self.LJpairpotentials, connectivity=connectivity)
                self.LJenergy,self.LJgradient = LennardJones(full_coords,self.epsij,self.sigmaij)
                #print("Lennard-Jones Energy (au):", self.LJenergy)
                #print("Lennard-Jones Energy (kcal/mol):", self.LJenergy*constants.harkcal)
            self.MMEnergy = self.Coulombchargeenergy+self.LJenergy

            if Grad==True:
                self.MMGradient = self.Coulombchargegradient+self.LJgradient
        #Combined Coulomb+LJ Python version. Slow
        elif self.codeversion=='py_comb':
            print("not active")
            exit()
            self.MMenergy, self.MMgradient = LJCoulpy(full_coords, self.atomtypes, charges, self.LJpairpotentials,
                                                          connectivity=connectivity)
        elif self.codeversion=='f2py':
            if self.printlevel >= 2:
                print("Using fast Fortran F2Py MM code")
            try:
                #print(os.environ.get("LD_LIBRARY_PATH"))
                import LJCoulombv1
                #print(LJCoulombv1.__doc__)
                #print("----------")
            except:
                print("Fortran library LJCoulombv1 not found! Make sure you have run the installation script.")
            self.MMEnergy, self.MMGradient, self.LJenergy, self.Coulombchargeenergy =\
                LJCoulomb(full_coords, self.epsij, self.sigmaij, charges, connectivity=connectivity)
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
                LJCoulombv2(full_coords, self.epsij, self.sigmaij, charges, connectivity=connectivity)

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
                 potfilename='System', pot_option='', pyframe=False, PElabel_pyframe='MM'):
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
            print("QM region size:", len(self.qmatoms))
            print("PE region size", len(self.peatoms))
            print("MM region size", len(self.mmatoms))
            blankline()

            #Creating list of QM, PE, MM labels used by reading residues in PDB-file
            #Also making residlist necessary
            self.hybridatomlabels=[]
            self.residlabels=[]
            count=2
            rescount=0
            for i in self.allatoms:
                if i in self.qmatoms:
                    self.hybridatomlabels.append('QM')
                    self.residlabels.append(1)
                elif i in self.peatoms:
                    self.hybridatomlabels.append(self.PElabel_pyframe)
                    self.residlabels.append(count)
                    rescount+=1
                elif i in self.mmatoms:
                    self.hybridatomlabels.append('WAT')
                    self.residlabels.append(count)
                    rescount+=1
                if rescount==3:
                    count+=1
                    rescount=0

        #print("self.hybridatomlabels:", self.hybridatomlabels)



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

                else:
                    #TODO: Create pot file from scratch. Requires LoProp and Dalton I guess
                    print("Only pot_option SEP or TIP3P possible right now")
                    exit()
                    system = pyframe.MolecularSystem(input_file=file)
                    core = system.get_fragments_by_name(names=['QM'])
                    system.set_core_region(fragments=core, program='Dalton', basis='pcset-1')
                    # solvent = system.get_fragments_by_distance(reference=core, distance=4.0)
                    solvent = system.get_fragments_by_name(names=[self.PElabel_pyframe])
                    system.add_region(name='solvent', fragments=solvent, use_standard_potentials=True,
                          standard_potential_model=self.pot_option)
                    project = pyframe.Project()
                    project.create_embedding_potential(system)
                    project.write_core(system)
                    project.write_potential(system)
                    self.potfile=self.potfilename+'.pot'
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
            print("Pot creation is off for this object.")


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

        elif self.qm_theory_name == "PySCFTheory":
            print("not yet implemented with PolEmbed")
            exit()
        elif self.qm_theory_name == "DaltonTheory":
            print("not yet implemented")
            exit()
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
    def __init__(self, qm_theory="", qmatoms="", fragment='', mm_theory="" , charges=None,
                 embedding="Elstat", printlevel=2, nprocs=None, actatoms=None, frozenatoms=None):

        print(BC.WARNING,BC.BOLD,"------------Defining QM/MM object-------------", BC.END)

        #Setting nprocs of object
        if nprocs==None:
            self.nprocs=1
        else:
            self.nprocs=nprocs

        #If fragment object has been defined
        #This probably needs to be always true
        if fragment != '':
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
            self.connectivity=fragment.connectivity

            # Region definitions
            self.allatoms=list(range(0,len(self.elems)))
            self.qmatoms = qmatoms
            self.mmatoms=listdiff(self.allatoms,self.qmatoms)

            # FROZEN AND ACTIVE ATOMS
            if actatoms is None and frozenatoms is None:
                print("Actatoms/frozenatoms list not passed to QM/MM object. Will do all frozen interactions in MM (expensive).")
                print("All {} atoms active, no atoms frozen".format(len(self.allatoms)))
                self.actatoms=self.allatoms
                self.frozenatoms=[]
            elif actatoms is not None and frozenatoms is None:
                print("Actatoms list passed to QM/MM object. Will skip all frozen interactions in MM.")
                self.actatoms = actatoms
                self.frozenatoms = listdiff(self.allatoms, self.actatoms)
                print("{} active atoms, {} frozen atoms".format(len(self.actatoms), len(self.frozenatoms)))
            elif frozenatoms is not None and actatoms is None:
                print("Frozenatoms list passed to QM/MM object. Will skip all frozen interactions in MM.")
                self.frozenatoms = frozenatoms
                self.actatoms = listdiff(self.allatoms, self.frozenatoms)
                print("{} active atoms, {} frozen atoms".format(len(self.actatoms), len(self.frozenatoms)))
            else:
                print("active_atoms and frozen_atoms can not be both defined")
                exit(1)

            #Coords and elems lists.
            #Coords may change by run command
            self.qmcoords=[self.coords[i] for i in self.qmatoms]
            self.qmelems=[self.elems[i] for i in self.qmatoms]
            self.mmcoords=[self.coords[i] for i in self.mmatoms]
            self.mmelems=[self.elems[i] for i in self.mmatoms]
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

        self.QMChargesZeroed=False

        #Theory level definitions
        self.printlevel=printlevel
        self.qm_theory=qm_theory
        self.qm_theory_name = self.qm_theory.__class__.__name__
        self.mm_theory=mm_theory
        self.mm_theory_name = self.mm_theory.__class__.__name__
        if self.mm_theory_name == "str":
            self.mm_theory_name="None"
        print("QM-theory:", self.qm_theory_name)
        print("MM-theory:", self.mm_theory_name)
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


        #If MM THEORY (not just pointcharges)
        if mm_theory != "":
            #Add possible exception for QM-QM atoms here.
            #Maybe easier to just just set charges to 0. LJ for QM-QM still needs to be done by MM code
            if self.mm_theory_name == "OpenMMTheory":
                print("Now adding exceptions for frozen atoms")
                if len(self.frozenatoms) > 0:
                    mm_theory.addexceptions(self.frozenatoms)

            linkatom=False
            if linkatom==True:
                print("Adding link atoms...")
                #Link atoms. In an additive scheme we would always have link atoms, regardless of mechanical/electrostatic coupling
                #Charge-shifting would be part of Elstat below

                #Protocol:
                #1. Recognize QM-MM boundary. Connectivity?? Get QM1 and MM1 coords pairs
                #2. For QM-region coords, add linkatom at MM1 position initially. Then adjust distance
                #3. Modify charges of MM atoms according to Chemshell scheme. Update both OpenMM and self.charges


            if self.embedding=="Elstat":
                #Setting QM charges to 0 since electrostatic embedding
                self.ZeroQMCharges()
                print("Charges of QM atoms set to 0 (since Electrostatic Embedding):")
                if self.printlevel > 2:
                    for i in self.allatoms:
                        if i in qmatoms:
                            print("QM atom {} ({}) charge: {}".format(i, self.elems[i], self.charges[i]))
                        else:
                            print("MM atom {} ({}) charge: {}".format(i, self.elems[i], self.charges[i]))
                blankline()

        #QM and MM charges are defined even though an MMtheory may not be present
        # Charges defined for regions
        self.qmcharges = [self.charges[i] for i in self.qmatoms]
        print("QM-region charges:", self.qmcharges)
        self.mmcharges=[self.charges[i] for i in self.mmatoms]
        #print("self.mmcharges:", self.mmcharges)

    # Set QMcharges to Zero
    def ZeroQMCharges(self):
        newcharges = []
        for i, c in enumerate(self.charges):
            if i in self.mmatoms:
                newcharges.append(c)
            else:
                newcharges.append(0.0)
        # Todo: use self.charges or use newcharges. Since done temporarily??
        self.charges = newcharges
        # Todo: make sure this works for OpenMM and for NonBOndedTheory
        # Updating charges in MM object
        self.mm_theory.update_charges(self.charges)
        self.QMChargesZeroed = True
    def run(self, current_coords=None, elems=None, Grad=False, nprocs=None):
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

        if nprocs==None:
            nprocs=self.nprocs
        if self.printlevel >= 2:
            print("Running QM/MM object with {} cores available".format(nprocs))
        #Updating QM coords and MM coords.
        #TODO: Should we use different name for updated QMcoords and MMcoords here??
        self.qmcoords=[current_coords[i] for i in self.qmatoms]
        self.mmcoords=[current_coords[i] for i in self.mmatoms]
        if self.qm_theory_name=="ORCATheory":
            #Calling ORCA theory, providing current QM and MM coordinates.
            if Grad==True:
                if PC==True:
                    self.QMEnergy, self.QMgradient, self.PCgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                                                         current_MM_coords=self.mmcoords,
                                                                                         MMcharges=self.mmcharges,
                                                                                         qm_elems=self.qmelems, mm_elems=self.mmelems,
                                                                                         Grad=True, PC=True, nprocs=nprocs)
                else:
                    self.QMEnergy, self.QMgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.mmcoords, MMcharges=self.mmcharges,
                                                      qm_elems=self.qmelems, mm_elems=self.mmelems, Grad=True, PC=False, nprocs=nprocs)
            else:
                self.QMEnergy = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.mmcoords, MMcharges=self.mmcharges,
                                                      qm_elems=self.qmelems, mm_elems=self.mmelems, Grad=False, PC=PC, nprocs=nprocs)
        elif self.qm_theory_name == "Psi4Theory":
            #Calling Psi4 theory, providing current QM and MM coordinates.
            if Grad==True:
                print("Grad true")

                if PC==True:
                    print("Pointcharge gradient for Psi4 is not implemented.")
                    print(BC.WARNING, "Warning: Only calculating QM-region contribution, skipping electrstatic-embedding gradient on pointcharges", BC.END)
                    print(BC.WARNING, "Only makes sense if MM region is frozen! ", BC.END)
                    self.QMEnergy, self.QMgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                                                         current_MM_coords=self.mmcoords,
                                                                                         MMcharges=self.mmcharges,
                                                                                         qm_elems=self.qmelems, mm_elems=self.mmelems,
                                                                                         Grad=True, PC=True, nprocs=nprocs)
                    #Creating zero-gradient array
                    self.PCgradient = np.zeros((len(self.mmatoms), 3))
                else:
                    print("grad. mech embedding. not ready")
                    exit()
                    self.QMEnergy, self.QMgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.mmcoords, MMcharges=self.mmcharges,
                                                      qm_elems=self.qmelems, mm_elems=self.mmelems, Grad=True, PC=False, nprocs=nprocs)
            else:
                print("grad false.")
                if PC == True:
                    print("PC embed true. not ready")
                    self.QMEnergy = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.mmcoords, MMcharges=self.mmcharges,
                                                      qm_elems=self.qmelems, mm_elems=self.mmelems, Grad=False, PC=PC, nprocs=nprocs)
                else:
                    print("mech true", not ready)
                    exit()


        elif self.qm_theory_name == "xTBTheory":
            #Calling xTB theory, providing current QM and MM coordinates.
            if Grad==True:
                if PC==True:
                    self.QMEnergy, self.QMgradient, self.PCgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                                                         current_MM_coords=self.mmcoords,
                                                                                         MMcharges=self.mmcharges,
                                                                                         qm_elems=self.qmelems, mm_elems=self.mmelems,
                                                                                         Grad=True, PC=True, nprocs=nprocs)
                else:
                    self.QMEnergy, self.QMgradient = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.mmcoords, MMcharges=self.mmcharges,
                                                      qm_elems=self.qmelems, mm_elems=self.mmelems, Grad=True, PC=False, nprocs=nprocs)
            else:
                self.QMEnergy = self.qm_theory.run(current_coords=self.qmcoords,
                                                      current_MM_coords=self.mmcoords, MMcharges=self.mmcharges,
                                                      qm_elems=self.qmelems, mm_elems=self.mmelems, Grad=False, PC=PC, nprocs=nprocs)


        elif self.qm_theory_name == "DaltonTheory":
            print("not yet implemented")
            exit(1)
        elif self.qm_theory_name == "NWChemtheory":
            print("not yet implemented")
            exit(1)
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
            self.MMEnergy, self.MMGradient= self.mm_theory.run(full_coords=current_coords, mm_coords=self.mmcoords,
                                                               charges=self.charges, connectivity=self.connectivity,
                                                               qmatoms=self.qmatoms, actatoms=self.actatoms)
            #self.MMEnergy=self.mm_theory.MMEnergy
            #if Grad==True:
            #    self.MMGrad = self.mm_theory.MMGrad
            #    print("self.MMGrad:", self.MMGrad)
        elif self.mm_theory_name == "OpenMMTheory":
            if self.printlevel >= 2:
                print("Running OpenMM theory as part of QM/MM.")
            if self.QMChargesZeroed==True:
                if self.printlevel >= 2:
                    print("Using MM on full system. Charges for QM region {} have to be set to zero ".format(self.qmatoms))
            else:
                print("QMCharges have not been zeroed")
                exit(1)
            printdebug("Charges for full system is: ", self.charges)
            #Todo: Need to make sure OpenMM skips QM-QM Lj interaction => Exclude
            #Todo: Need to have OpenMM skip frozen region interaction for speed  => => Exclude
            self.MMEnergy, self.MMGradient= self.mm_theory.run(coords=current_coords, qmatoms=self.qmatoms)
        else:
            self.MMEnergy=0
        print_time_rel(CheckpointTime, modulename='MM step')
        CheckpointTime = time.time()
        #Final QM/MM Energy
        self.QM_MM_Energy= self.QMEnergy+self.MMEnergy
        blankline()
        if self.printlevel >= 2:
            print("{:<20} {:>20.12f}".format("QM energy: ",self.QMEnergy))
            print("{:<20} {:>20.12f}".format("MM energy: ", self.MMEnergy))
            print("{:<20} {:>20.12f}".format("QM/MM energy: ", self.QM_MM_Energy))
        blankline()

        #Final QM/MM gradient. Combine QM gradient, MM gradient and PC-gradient (elstat MM gradient from QM code).
        #First combining QM and PC gradient to one.
        if Grad == True:
            self.QM_PC_Gradient = np.zeros((len(self.allatoms), 3))
            qmcount=0;pccount=0
            for i in self.allatoms:
                if i in self.qmatoms:
                    self.QM_PC_Gradient[i]=self.QMgradient[qmcount]
                    qmcount+=1
                else:
                    self.QM_PC_Gradient[i] = self.PCgradient[pccount]
                    pccount += 1
            #Now assemble final QM/MM gradient
            self.QM_MM_Gradient=self.QM_PC_Gradient+self.MMGradient
            #print_time_rel(CheckpointTime, modulename='QM/MM gradient combine')
            if self.printlevel >=3:
                print("QM gradient (au/Bohr):")
                print_coords_all(self.QMgradient, self.qmelems, self.qmatoms)
                blankline()
                print("PC gradient (au/Bohr):")
                print_coords_all(self.PCgradient, self.mmelems, self.mmatoms)
                blankline()
                print("QM+PC gradient (au/Bohr):")
                print_coords_all(self.QM_PC_Gradient, self.elems, self.allatoms)
                blankline()
                print("MM gradient (au/Bohr):")
                print_coords_all(self.MMGradient, self.elems, self.allatoms)
                blankline()
                print("Total QM/MM gradient (au/Bohr):")
                print_coords_all(self.QM_MM_Gradient, self.elems,self.allatoms)
            if self.printlevel >= 2:
                print(BC.WARNING,BC.BOLD,"------------ENDING QM/MM MODULE-------------",BC.END)
            return self.QM_MM_Energy, self.QM_MM_Gradient
        else:
            return self.QM_MM_Energy



#ORCA Theory object. Fragment object is optional. Only used for single-points.
class ORCATheory:
    def __init__(self, orcadir, fragment=None, charge='', mult='', orcasimpleinput='', printlevel=2,
                 orcablocks='', extraline='', brokensym=None, HSmult=None, atomstoflip=None, nprocs=1):

        #Create inputfile with generic name
        self.inputfilename="orca-input"

        #Using orcadir to set LD_LIBRARY_PATH
        old = os.environ.get("LD_LIBRARY_PATH")
        if old:
            os.environ["LD_LIBRARY_PATH"] = orcadir + ":" + old
        else:
            os.environ["LD_LIBRARY_PATH"] = orcadir
        #os.environ['LD_LIBRARY_PATH'] = orcadir + ':$LD_LIBRARY_PATH'

        #Printlevel
        self.printlevel=printlevel

        #Setting nprocs of object
        self.nprocs=nprocs

        self.orcadir = orcadir
        if fragment != None:
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        #print("frag elems", self.fragment.elems)
        if charge!='':
            self.charge=int(charge)
        if mult!='':
            self.mult=int(mult)
        self.orcasimpleinput=orcasimpleinput
        self.orcablocks=orcablocks
        self.extraline=extraline
        self.brokensym=brokensym
        self.HSmult=HSmult
        if type(atomstoflip) is int:
            print(BC.FAIL,"Error: atomstoflip should be list of integers (e.g. [0] or [2,3,5]), not a single integer.", BC.END)
            exit(1)
        self.atomstoflip=atomstoflip
        if self.printlevel >=2:
            print("Creating ORCA object")
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
            mm_elems=None, elems=None, Grad=False, PC=False, nprocs=None ):
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

        if nprocs==None:
            nprocs=self.nprocs
        print("Running ORCA object with {} cores available".format(nprocs))


        print("Creating inputfile:", self.inputfilename+'.inp')
        print("ORCA input:")
        print(self.orcasimpleinput)
        print(self.extraline)
        print(self.orcablocks)
        print("Charge: {}  Mult: {}".format(self.charge, self.mult))
        if PC==True:
            print("Pointcharge embedding is on!")
            create_orca_pcfile(self.inputfilename, mm_elems, current_MM_coords, MMcharges)
            if self.brokensym==True:
                print("Brokensymmetry SpinFlipping on! HSmult: {}.".format(self.HSmult))
                for flipatom in self.atomstoflip:
                    print("Flipping atom: {} {}".format(flipatom, qm_elems[flipatom]))
                create_orca_input_pc(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        self.charge, self.mult, extraline=self.extraline, HSmult=self.HSmult, Grad=Grad,
                                     atomstoflip=self.atomstoflip)
            else:
                create_orca_input_pc(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        self.charge, self.mult, extraline=self.extraline, Grad=Grad)
        else:
            if self.brokensym == True:
                print("Brokensymmetry SpinFlipping on! HSmult: {}.".format(self.HSmult))
                for flipatom in self.atomstoflip:
                    print("Flipping atom: {} {}".format(flipatom, qm_elems[flipatom]))
                create_orca_input_plain(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        self.charge,self.mult, extraline=self.extraline, HSmult=self.HSmult, Grad=Grad,
                                     atomstoflip=self.atomstoflip)
            else:
                create_orca_input_plain(self.inputfilename, qm_elems, current_coords, self.orcasimpleinput,self.orcablocks,
                                        self.charge,self.mult, extraline=self.extraline, Grad=Grad)

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
    def __init__(self, fragment=None, charge=None, mult=None, printsetting='False', psi4settings=None, psi4method=None, psi4functional=None,
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
        #DFT-specific. Remove?
        self.psi4functional=psi4functional
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
            mm_elems=None, elems=None, Grad=False, PC=False, nprocs=None, pe=False, potfile='', restart=False ):

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
                self.psi4settings['reference'] = 'RKS'
            else:
                self.psi4settings['reference'] = 'UKS'

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
                if self.psi4functional is not None:
                    grad=psi4.gradient('scf', dft_functional=self.psi4functional)
                    self.gradient=np.array(grad)
                    self.energy = psi4.variable("CURRENT ENERGY")
                else:
                    print("Running gradient with Psi4 method:", self.psi4method)
                    grad=psi4.gradient(self.psi4method)
                    self.gradient=np.array(grad)
                    self.energy = psi4.variable("CURRENT ENERGY")
            else:
                #This might be unnecessary as I think all DFT functionals work as keyword to energy function. Hence psi4method works for all
                if self.psi4functional is not None:
                    self.energy = psi4.energy('scf', dft_functional=self.psi4functional)
                else:
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
                    inputfile.write('Chrgfield = QMMM()')
                    # Mmcoords in Angstrom
                    for mmcharge, mmcoord in zip(MMcharges, current_MM_coords):
                        inputfile.write('Chrgfield.extern.addCharge({}, {}, {}, {})'.format(mmcharge, mmcoord[0], mmcoord[1], mmcoord[2]))
                    inputfile.write('psi4.set_global_option_python(\'EXTERN\', Chrgfield.extern)')

                #Adding Psi4 settings
                inputfile.write('set {\n')
                for key,val in self.psi4settings.items():
                    inputfile.write(key+' '+val+'\n')
                #Setting RKS or UKS reference. For now, RKS always if mult 1 Todo: Make more flexible
                if self.mult == 1:
                    self.psi4settings['reference'] = 'RKS'
                else:
                    inputfile.write('reference UKS \n')
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

                if Grad==True:
                    inputfile.write('scf_energy, wfn = gradient(\'scf\', dft_functional=\'{}\', return_wfn=True)\n'.format(self.psi4functional))
                else:
                    inputfile.write('scf_energy, wfn = energy(\'scf\', dft_functional=\'{}\', return_wfn=True)\n'.format(self.psi4functional))
                    inputfile.write('\n')

            print("Running inputfile:", self.label+'.inp')
            #Running inputfile
            with open(self.label + '.out', 'w') as ofile:
                #Psi4 -m option for saving 180 file
                process = sp.run(['psi4', '-m', '-i', self.label + '.inp', '-o', self.label + '.out', '-n', str(nprocs) ],
                                 check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

            #Keep restart file 180 as lastrestart.180
            try:
                restartfile=glob.glob(self.label+'*180.npy')[0]
                print("restartfile:", restartfile)
                print("SCF Done. Renaming {} to lastrestart.180.npy".format(restartfile))
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
            mm_elems=None, elems=None, Grad=False, PC=False, nprocs=None, pe=False, potfile='', restart=False ):

        if nprocs==None:
            nprocs=self.nprocs



        print(BC.OKBLUE,BC.BOLD, "------------RUNNING PYSCF INTERFACE-------------", BC.END)

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
            pe_options = cppe.PeOptions()
            pe_options.do_diis = True
            pe_options.potfile = self.potfile
            pe = pol_embed.PolEmbed(mol, pe_options)
            # TODO: Adapt to RKS vs. UKS etc.
            mf = solvent.PE(scf.RKS(mol), pe)
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

        #RUN ENERGY job
        self.energy = mf.kernel()

        if self.pe==True:
            print(mf._pol_embed.cppe_state.summary_string)

        #Grab energy and gradient
        if Grad==True:
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




# Fragment class
class Fragment:
    def __init__(self, coordsstring=None, fragfile=None, xyzfile=None, pdbfile=None, coords=None, elems=None, connectivity=None,
                 atomcharges=None, atomtypes=None, conncalc=True, scale=None, tol=None):
        print("Defining new ASH fragment object")
        self.energy = None
        self.elems=[]
        self.coords=[]
        self.connectivity=[]
        self.atomcharges = []
        self.atomtypes = []
        self.Centralmainfrag = []
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
            self.coords=coords
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
            self.read_xyzfile(xyzfile)
        elif pdbfile is not None:
            self.read_pdbfile(pdbfile, conncalc=conncalc)
        elif fragfile is not None:
            self.read_fragment_from_file(fragfile)
    def update_attributes(self):
        self.nuccharge = nucchargelist(self.elems)
        self.numatoms = len(self.coords)
        print("Fragment numatoms:", self.numatoms)
        self.atomlist = list(range(0, self.numatoms))
        #Unnecessary alias ? Todo: Delete
        self.allatoms = self.atomlist
        self.mass = totmasslist(self.elems)
        self.list_of_masses = list_of_masses(self.elems)
    #Add coordinates from geometry string. Will replace.
    #Todo: Needs more work as elems and coords may be lists or numpy arrays
    def add_coords_from_string(self, coordsstring, scale=None, tol=None):
        print("Getting coordinates from string:", coordsstring)
        if len(self.coords)>0:
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
        print("Replacing coordinates in fragment.")
        self.elems=elems
        self.coords=coords
        self.update_attributes()
        if conn==True:
            self.calc_connectivity(scale=scale, tol=tol)
    def delete_coords(self):
        self.coords=[]
        self.elems=[]
        self.connectivity=[]
    def add_coords(self, elems,coords,conn=True, scale=None, tol=None):
        print("Adding coordinates to fragment.")
        if len(self.coords)>0:
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
    #Read PDB file
    def read_pdbfile(self,filename,conncalc=True, scale=None, tol=None):

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
                        elemcol.append(elem)
                    #self.coords.append([float(line.split()[6]), float(line.split()[7]), float(line.split()[8])])
                    #elemcol.append(line.split()[-1])
                    #residuelist.append(line.split()[3])
                    #atom_name.append(line.split()[3])
                if 'HETATM' in line:
                    print("HETATM line in file found. Please rename to ATOM")
                    exit()
        if len(elemcol) != len(self.coords):
            print("did not find same number of elements as coordinates")
            print("Need to define elements in some other way")
            exit()
        else:
            self.elems=elemcol
        self.update_attributes()
        if conncalc is True:
            self.calc_connectivity(scale=scale, tol=tol)
    #Read XYZ file
    def read_xyzfile(self,filename, scale=None, tol=None):
        print("Reading coordinates from XYZfile {} into fragment".format(filename))
        with open(filename) as f:
            for count,line in enumerate(f):
                if count == 0:
                    self.numatoms=int(line.split()[0])
                if count > 1:
                    if len(line) > 3:
                        self.elems.append(line.split()[0])
                        self.coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        if self.numatoms != len(self.coords):
            print("Number of atoms in header not equal to number of coordinate-lines. Check XYZ file!")
            exit()
        self.update_attributes()
        self.calc_connectivity(scale=scale, tol=tol)
    def set_energy(self,energy):
        self.energy=float(energy)
    # Get coordinates for specific atoms (from list of atom indices)
    def get_coords_for_atoms(self, atoms):
        subcoords=[self.coords[i] for i in atoms]
        subelems=[self.elems[i] for i in atoms]
        return subcoords,subelems
    #Calculate connectivity (list of lists) of coords
    def calc_connectivity(self, conndepth=99, scale=None, tol=None ):
        print("Calculating connectivity of fragment...")

        if len(self.coords) > 10000:
            print("Atom number > 10K. Connectivity calculation could take a while")

        if scale == None:
            try:
                scale = settings_ash.scale
                tol = settings_ash.tol
                print("Using global scale and tol parameters from settings_ash. Scale: {} Tol: {} ".format(scale, tol ))

            except:
                scale = 1
                tol = 0.1
                print("Exception: Using hard-coded scale and tol parameters. Scale: {} Tol: {} ".format(scale, tol ))
        else:
            print("Using scale: {} and tol: {} ".format(scale, tol))
        # Calculate connectivity by looping over all atoms
        found_atoms = []
        fraglist = []
        count = 0
        #Todo: replace by Fortran code? Pretty slow for 10K atoms
        timestampA=time.time()
        for atom in range(0, len(self.elems)):
            if atom not in found_atoms:
                count += 1
                members = get_molecule_members_loop_np2(self.coords, self.elems, conndepth, scale,
                                                        tol, atomindex=atom)
                if members not in fraglist:
                    fraglist.append(members)
                    found_atoms += members
        print_time_rel(timestampA, modulename='calc connectivity1')
        #flat_fraglist = [item for sublist in fraglist for item in sublist]
        self.connectivity=fraglist
        #Calculate number of atoms in connectivity list of lists
        conn_number_sum=0
        for l in self.connectivity:
            conn_number_sum+=len(l)
        if self.numatoms != conn_number_sum:
            print("Connectivity problem")
            exit()
        self.connected_atoms_number=conn_number_sum

    def update_atomcharges(self, charges):
        self.atomcharges = charges
    def update_atomtypes(self, types):
        self.atomtypes = types
    #Adding fragment-type info (used by molcrys, identifies whether atom is mainfrag, counterfrag1 etc.)
    def add_fragment_type_info(self,fragmentobjects):
        # Create list of fragment-type label-list
        self.fragmenttype_labels = []
        for i in self.atomlist:
            for count,fobject in enumerate(fragmentobjects):
                if i in fobject.flat_clusterfraglist:
                    self.fragmenttype_labels.append(count)
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
        print("Wrote XYZ file:", xyzfilename)
    #Print system-fragment information to file. Default name of file: "fragment.ygg
    def print_system(self,filename='fragment.ygg'):
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
            outfile.write("Energy: {}\n".format(self.energy))
            outfile.write("\n")
            outfile.write(" Index    Atom         x                  y                  z               charge        fragment-type        atom-type\n")
            outfile.write("----------------------------------------------------------------------------------------------------------------------\n")
            for at, el, coord, charge, label, atomtype in zip(self.atomlist, self.elems,self.coords,self.atomcharges, self.fragmenttype_labels, self.atomtypes):
                line="{:>6} {:>6}  {:17.11f}  {:17.11f}  {:17.11f}  {:14.8f} {:12d} {:>21}\n".format(at, el,coord[0], coord[1], coord[2], charge, label, atomtype)
                outfile.write(line)
            outfile.write(
                "======================================================================================================================\n")
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
                    l = line.lstrip('Centralmainfrag:')
                    l = l.strip('[')
                    l = l.strip(']')
                    Centralmainfrag = [i for i in l.split(',')]
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
            print("xTBTheory requires xtbnmethod keyword to be set")
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
                mm_elems=None, elems=None, Grad=False, PC=False, nprocs=None):
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
def ReactionEnergy(stoichiometry=None, list_of_fragments=None, list_of_energies=None, unit='kcalpermol'):
    conversionfactor = { 'kcalpermol' : 627.50946900, 'kJpermol' : 2625.499638, 'eV' : 27.211386245988, 'cm-1' : 219474.6313702 }
    print("")
    print(BC.OKBLUE,BC.BOLD, "ReactionEnergy function. Unit:", unit, BC.END)
    print("")
    reactant_energy=0.0 #hartree
    product_energy=0.0 #hartree
    if stoichiometry is None:
        print("stoichiometry list is required")
        exit(1)

    #List of energies option
    if list_of_energies is not None:
        print("List of total energies provided (Eh units assumed).")
        print("")
        for i,stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy=reactant_energy+list_of_energies[i]*abs(stoich)
            if stoich > 0:
                product_energy=product_energy+list_of_energies[i]*abs(stoich)
        reaction_energy=(product_energy-reactant_energy)*conversionfactor[unit]
        print(BC.OKGREEN,BC.BOLD, "Reaction_energy:", reaction_energy, unit, BC.END)
    else:
        print("No list of total energies provided. Using internal energy of each fragment instead.")
        print("")
        for i,stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy=reactant_energy+list_of_fragments[i].energy*abs(stoich)
            if stoich > 0:
                product_energy=product_energy+list_of_fragments[i].energy*abs(stoich)
        reaction_energy=(product_energy-reactant_energy)*conversionfactor[unit]
        print(BC.OKGREEN,BC.BOLD, "Reaction_energy:", reaction_energy, unit, BC.END)
    return reaction_energy