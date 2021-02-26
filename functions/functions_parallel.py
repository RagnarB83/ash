import copy
import multiprocessing as mp
import os

import ash
from functions_general import BC,blankline
#Various calculation-functions run in parallel


#Stripped down version of Singlepoint function for Singlepoint_parallel
#TODO: This function is ORCA-centric. Needs to be generalized 
def Single_par(list):
    #Creating new copy of theory to prevent Brokensym feature from being deactivated by each run
    #NOTE: Alternatively we can add an if-statement inside orca.run
    theory=copy.deepcopy(list[0])
    fragment=list[1]
    #Making label flexible. Can be tuple but inputfilename is converted to string below
    label=list[2]
    print("label:", label)
    if label is None:
        print("No label provided to fragment or theory objects. This is required to distinguish between calculations ")
        print("Exiting...")
        exit(1)

    #Using label (could be tuple) to create a labelstring which is used to name inputfiles

    if type(label) == tuple: 
        labelstring=str(label[0])+'_'+str(label[1])
    else:
        labelstring=label

    print("labelstring:", labelstring)
    #Creating separate inputfilename using label
    #Removing . in inputfilename as ORCA can get confused
    if theory.__class__.__name__ == "ORCATheory":
        theory.filename=''.join([str(i) for i in labelstring].replace('.','_'))
    #TODO: filename changes for other codes ?

    coords = fragment.coords
    elems = fragment.elems
    #Creating new dir and running calculation inside
    os.mkdir(labelstring)
    os.chdir(labelstring)
    print(BC.WARNING,"Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
    print("\n\nProcess ID {} is running calculation with label: {} \n\n".format(mp.current_process(),label))

    energy = theory.run(current_coords=coords, elems=elems, label=label)
    os.chdir('..')
    print("Energy: ", energy)
    # Now adding total energy to fragment
    fragment.energy = energy
    return (label,energy)


def bla(blux):
    print("here")
    print(blux)

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
        print("Making copy of theory object")
        theory = theories[0]
        #NOTE: Python 3.8 and higher use spawn in MacOS. Leads to ash import problems
        #NOTE: Unix/Linux uses fork which seems better behaved
        results = pool.map(Single_par, [[theory,fragment, fragment.label] for fragment in fragments])
        
        pool.close()
        print("Calculations are done")
        print("results:", results)
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


#Functions to run each displacement in parallel NumFreq run.
#Simple QM object
def displacement_QMrun(arglist):
    #print("arglist:", arglist)
    geo = arglist[0]
    elems = arglist[1]
    #We can launch ORCA-OpenMPI in parallel it seems. Only makes sense if we have may more cores available than displacements
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
    elems,coords = module_coords.read_xyzfile(filelabel+'.xyz')

    #Todo: Copy previous GBW file in here if ORCA, xtbrestart if xtb, etc.
    print("Running displacement: {}".format(label))

    #If QMMMTheory init keywords are changed this needs to be updated
    qmmmobject = ash.QMMMTheory(fragment=fragment, qm_theory=qm_theory, mm_theory=mm_theory, actatoms=actatoms,
                          qmatoms=qmatoms, embedding=embedding, charges=charges, printlevel=printlevel,
                            nprocs=numcores, frozenatoms=frozenatoms)

    energy, gradient = qmmmobject.run(current_coords=coords, elems=elems, Grad=True, nprocs=numcores)
    print("Energy: ", energy)
    os.chdir('..')
    #Delete dir?
    #os.remove(dispdir)
    return [label, energy, gradient]




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
