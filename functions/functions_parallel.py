import copy
import multiprocessing as mp
import os
import sys
import time

import ash
from functions_general import BC,blankline,print_line_with_mainheader,print_line_with_subheader1

#Various calculation-functions run in parallel


#Stripped down version of Singlepoint function for Singlepoint_parallel
#TODO: This function may still be a bit ORCA-centric. Needs to be generalized 
def Single_par_improved(fragment=None, fragment_file=None, theory=None, label=None, mofilesdir=None, event=None):

    #Creating new copy of theory to prevent Brokensym feature from being deactivated by each run
    #NOTE: Alternatively we can add an if-statement inside orca.run
    theory=copy.deepcopy(theory)

    print("Fragment:", fragment)
    print("fragment_file:", fragment_file)

    #Making label flexible. Can be tuple but inputfilename is converted to string below
    print("label: {} (type {})".format(label,type(label)))
    if label == None:
        print("No label provided to fragment or theory objects. This is required to distinguish between calculations ")
        print("Exiting.")
        #event.set()
        #print("event is_set: ", event.is_set())
        raise Exception("Labelproblem")

    #Using label (could be tuple) to create a labelstring which is used to name worker directories
    # Tuple-label (1 or 2 element) used by calc_surface functions.
    # Otherwise normally string
    if type(label) == tuple:
        if len(label) ==2:
            labelstring=str(str(label[0])+'_'+str(label[1])).replace('.','_')
        else:
            labelstring=str(str(label[0])).replace('.','_')
        print("Labelstring:", labelstring)
        #RC1_0.9-RC2_170.0.xyz
        #orca_RC1_0.9RC2_170.0.gbw
        #TODO: what if tuple is only a single number???

        if mofilesdir != None:
            print("Mofilesdir option.")
            if len(label) == 2:
                moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])+'-'+'RC2_'+str(label[1])
            else:
                moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])

    #Label is not tuple. Not coming from calc_surface funcitons
    elif type(label) == float or type(label) == int:
        print("Label is float or int")
        #
        #Label is float or int. 
        if mofilesdir != None:
            print("Mofilesdir option.")
            moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])
    else:
        print("Here. label.", label)
        #Label is not tuple. String or single number
        labelstring=str(label).replace('.','_')

    #Creating separate inputfilename using label
    #Removing . in inputfilename as ORCA can get confused
    if theory.__class__.__name__ == "ORCATheory":
        #theory.filename=''.join([str(i) for i in labelstring])
        #NOTE: Not sure if we really need to use labelstring for input since inside separate directoreies
        #Disabling for now
        #theory.filename=labelstring
        if mofilesdir != None:
            theory.moreadfile=moreadfile_path+'.gbw'
            print("Setting moreadfile to:", theory.moreadfile)
    elif theory.__class__.__name__ == "MRCCTheory":
        if mofilesdir != None:
            print("Case MRCC MOREADfile parallel")
            print("moreadfile_path:", moreadfile_path)
        print("not finished. exiting")
        raise Exception()
    else:
        if mofilesdir != None:
            print("moreadfile option not ready for this Theory. exiting")
            raise Exception()

    #Creating new dir and running calculation inside
    os.mkdir('Pooljob_'+labelstring)
    os.chdir('Pooljob_'+labelstring)
    print(BC.WARNING,"Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
    print("\n\nProcess ID {} is running calculation with label: {} \n\n".format(mp.current_process(),label))

    energy = theory.run(current_coords=fragment.coords, elems=fragment.elems, label=label)
    os.chdir('..')
    print("Energy: ", energy)
    # Now adding total energy to fragment
    fragment.energy = energy
    return (label,energy)





#Stripped down version of Singlepoint function for Singlepoint_parallel
#TODO: This function may still be a bit ORCA-centric. Needs to be generalized 
def Single_par(listx):
    print("listx:", listx)
    #Multiprocessing event
    event = listx[4]

    #Creating new copy of theory to prevent Brokensym feature from being deactivated by each run
    #NOTE: Alternatively we can add an if-statement inside orca.run
    theory=copy.deepcopy(listx[0])

    #listx[1] is either an ASH Fragment or Fragment file string (avoids massive pickle object)
    if listx[1].__class__.__name__ == "Fragment":
        print("Here", listx[1].__class__.__name__)
        fragment=listx[1]
    elif listx[1].__class__.__name__ == "str":
        if "ygg" in listx[1]:
            print("Found string assumed to be ASH fragmentfile")
            fragment=ash.Fragment(fragfile=listx[1])
    else:
        print("Unknown object passed")
        kill_all_mp_processes()
        exit()
    print("Fragment:", fragment)

    #Making label flexible. Can be tuple but inputfilename is converted to string below
    label=listx[2]
    mofilesdir=listx[3]
    print("label: {} (type {})".format(label,type(label)))
    if label == None:
        print("No label provided to fragment or theory objects. This is required to distinguish between calculations ")
        print("Exiting...")
        print("event:", event)
        print("event is_set: ", event.is_set())
        event.set()
        print("after event")
        print("event is_set: ", event.is_set())
        exit()
        #sys.exit()
        #kill_all_mp_processes()
        #exit()

    #Using label (could be tuple) to create a labelstring which is used to name worker directories
    # Tuple-label (1 or 2 element) used by calc_surface functions.
    # Otherwise normally string
    if type(label) == tuple:
        if len(label) ==2:
            labelstring=str(str(label[0])+'_'+str(label[1])).replace('.','_')
        else:
            labelstring=str(str(label[0])).replace('.','_')
        print("Labelstring:", labelstring)
        #RC1_0.9-RC2_170.0.xyz
        #orca_RC1_0.9RC2_170.0.gbw
        #TODO: what if tuple is only a single number???

        if mofilesdir != None:
            print("Mofilesdir option.")
            if len(label) == 2:
                moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])+'-'+'RC2_'+str(label[1])
            else:
                moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])

    #Label is not tuple. Not coming from calc_surface funcitons
    elif type(label) == float or type(label) == int:
        print("Label is float or int")
        #
        #Label is float or int. 
        if mofilesdir != None:
            print("Mofilesdir option.")
            moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])
    else:
        print("Here. label.", label)
        #Label is not tuple. String or single number
        labelstring=str(label).replace('.','_')

    #Creating separate inputfilename using label
    #Removing . in inputfilename as ORCA can get confused
    if theory.__class__.__name__ == "ORCATheory":
        #theory.filename=''.join([str(i) for i in labelstring])
        #NOTE: Not sure if we really need to use labelstring for input since inside separate directoreies
        #Disabling for now
        #theory.filename=labelstring
        if mofilesdir != None:
            theory.moreadfile=moreadfile_path+'.gbw'
            print("Setting moreadfile to:", theory.moreadfile)
    elif theory.__class__.__name__ == "MRCCTheory":
        if mofilesdir != None:
            print("Case MRCC MOREADfile parallel")
            print("moreadfile_path:", moreadfile_path)
        print("not finished. exiting")
        kill_all_mp_processes()
        exit()
    else:
        if mofilesdir != None:
            print("moreadfile option not ready for this Theory. exiting")
            kill_all_mp_processes()
            exit()

    #Creating new dir and running calculation inside
    os.mkdir('Pooljob_'+labelstring)
    os.chdir('Pooljob_'+labelstring)
    print(BC.WARNING,"Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
    print("\n\nProcess ID {} is running calculation with label: {} \n\n".format(mp.current_process(),label))

    energy = theory.run(current_coords=fragment.coords, elems=fragment.elems, label=label)
    os.chdir('..')
    print("Energy: ", energy)
    # Now adding total energy to fragment
    fragment.energy = energy
    return (label,energy)


#PARALLEL Single-point energy function
#will run over fragments or fragmentfiles, over theories or both
#mofilesdir. Directory containing MO-files (GBW files for ORCA). Usef for multiple fragment option
def Singlepoint_parallel(fragments=None, fragmentfiles=None, theories=None, numcores=None, mofilesdir=None):
    print("")
    '''
    The Singlepoint_parallel function carries out multiple single-point calculations in a parallel fashion
    :param fragments:
    :type list: list of ASH objects of class Fragment
    :type list: list of ASH fragmentfiles (strings)
    :param theories:
    :type list: list of ASH theory objects
    :param Grad: whether to do Gradient or not.
    :type Grad: Boolean.
    '''

    if fragments == None and fragmentfiles == None:
        print(BC.FAIL,"Singlepoint_parallel requires a list of ASH fragments or a list of fragmentfilenames",BC.END)
        exit(1)
    if theories == None or numcores == None :
        print("theories:", theories)
        print("numcores:", numcores)
        print(BC.FAIL,"Singlepoint_parallel requires a theory object and a numcores value",BC.END)
        exit(1)

    blankline()
    print_line_with_subheader1("Singlepoint_parallel function")
    print("Number of CPU cores available: ", numcores)
    if fragments != None:
        print("Number of fragments:", len(fragments))
    else:
        fragments=[]
    if fragmentfiles != None:
        print("Number of fragmentfiles:", len(fragmentfiles))
    else:
        fragmentfiles=[]
    print("Number of theories:", len(theories))
    print("Running single-point calculations in parallel")
    print("Mofilesdir:", mofilesdir)
    print(BC.WARNING, "Warning: Output from Singlepoint_parallel will be erratic due to simultaneous output from multiple workers", BC.END)
    pool = mp.Pool(numcores)
    manager = mp.Manager()
    event = manager.Event()
    print("event is_set: ", event.is_set())

    #Function to handle exception of child processes
    def Terminate_Pool_processes(message):
        print("Terminating Pool processes due to exception")
        print("Exception message:", message)
        print("Setting event")
        event.set()
        print("XXXXX")
        #pool.close()
        #print("a")
        #pool.terminate()
        #print("b")
        #sys.exit()
        #event.set()

    # Singlepoint(fragment=None, theory=None, Grad=False)
    #Case: 1 theory, multiple fragments
    if len(theories) == 1:
        theory = theories[0]
        print("Case: Multiple fragments but one theory")
        print("")
        print("Launching multiprocessing pool.map:")

        #Change theory nprocs to 1 since we are running ASH in parallel
        #NOTE: Alternative to exit here instead ??
        if theory.nprocs != 1:
            print(BC.WARNING,"Theory nprocs set to:", theory.nprocs, BC.END)
            print(BC.WARNING,"Since ASH is running in parallel we will now turn off Theory Parallelization",BC.END)
            theory.nprocs=1

        #NOTE: Python 3.8 and higher use spawn in MacOS. Leads to ash import problems
        #NOTE: Unix/Linux uses fork which seems better behaved



        #Passing list of fragments
        if len(fragments) > 0:
            print("Launching multiprocessing and passing list of ASH fragments")
            print("fragments:", fragments)
            #results = pool.map(Single_par, [[theory,fragment, fragment.label, mofilesdir, event] for fragment in fragments], error_callback=blax)
            for fragment in fragments:
                print("fragment:", fragment)
                results = pool.apply_async(Single_par_improved, kwds=dict(theory=theory,fragment=fragment,label=fragment.label,mofilesdir=mofilesdir,event=event), error_callback=Terminate_Pool_processes)
            
        #Passing list of fragment files
        elif len(fragmentfiles) > 0:
            print("Launching multiprocessing and passing list of ASH fragmentfiles")
            results = pool.map(Single_par, [[theory,fragmentfile, fragmentfile, mofilesdir, event] for fragmentfile in fragmentfiles])

        print("Calculations are done")
        print("results:", results)
    # Case: Multiple theories, 1 fragment
    elif len(fragments) == 1:
        print("Case: Multiple theories but one fragment")
        fragment = fragments[0]
        results = pool.map(Single_par, [[theory,fragment, theory.label, event] for theory in theories])

        print("Calculations are done")
    elif len(fragmentfiles) == 1:
        print("Case: Multiple theories but one fragmentfile")
        fragmentfile = fragmentfiles[0]
        results = pool.map(Single_par, [[theory,fragmentfile, theory.label,event] for theory in theories])
        print("Calculations are done")  
    else:
        print("Multiple theories and multiple fragments provided.")
        print("This is not supported. Exiting...")
        exit(1)


    print("xy2")
    while True:
        print("Pool multiprocessing underway....")
        time.sleep(3)
        if event.is_set():
            print("Event has been set! Now termininating Pool processes")
            pool.terminate()
            break
    print("YYYX3")
    pool.close()
    pool.join()
    print("results:", results)
    energy_dict = {result[0]: result[1] for result in results}
    print("energy_dict:", energy_dict)

    exit()







    print("here")
    exit()
    print("Closing Pool")
    pool.close()
    #Setting event to True since all is done
    event.set()
    #print("event is_set: ", event.is_set())

    #Terminate Pool if event flag was set to True (either above or by error in Single_par)
    event.wait()
    print("Pool terminate")
    

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
