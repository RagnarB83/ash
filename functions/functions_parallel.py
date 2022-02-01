import copy
import multiprocessing as mp
import os
import sys
import time

import ash
from functions.functions_general import ashexit, BC,blankline,print_line_with_mainheader,print_line_with_subheader1
from modules.module_coords import check_charge_mult
#Various calculation-functions run in parallel


#Stripped down version of Singlepoint functffragment_fileion for Singlepoint_parallel.
#NOTE: Version intended for apply_async
#TODO: This function may still be a bit ORCA-centric. Needs to be generalized 
def Single_par(fragment=None, fragmentfile=None, theory=None, label=None, mofilesdir=None, event=None, charge=None, mult=None):

    #Creating new copy of theory to prevent Brokensym feature from being deactivated by each run
    #NOTE: Alternatively we can add an if-statement inside orca.run
    theory=copy.deepcopy(theory)

    print("Fragment:", fragment)
    print("fragmentfile:", fragmentfile)
    if fragmentfile != None:
        print("Reading fragmentfile from disk")
        fragment=ash.Fragment(fragfile=fragmentfile)

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


    #Check charge/mult.
    charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Single_par")

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

    energy = theory.run(current_coords=fragment.coords, elems=fragment.elems, label=label, charge=charge, mult=mult)

    #Some theories like CC_CBS_Theory may return both energy and energy componentsdict as a tuple
    #TODO: avoid this nasty fix
    if type(energy) is tuple:
        componentsdict=energy[1]
        energy=energy[0]

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
        ashexit()
    if theories == None or numcores == None :
        print("theories:", theories)
        print("numcores:", numcores)
        print(BC.FAIL,"Singlepoint_parallel requires a theory object and a numcores value",BC.END)
        ashexit()

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
    print("Launching multiprocessing and passing list of ASH fragments")
    pool = mp.Pool(numcores)
    manager = mp.Manager()
    event = manager.Event()

    #Function to handle exception of child processes
    def Terminate_Pool_processes(message):
        print(BC.FAIL,"Terminating Pool processes due to exception", BC.END)
        print(BC.FAIL,"Exception message:", message, BC.END)
        pool.terminate()
        event.set()
        ashexit()
        #print("Setting event")
        #event.set()
        #print("XXXXX")
        #pool.close()
        #print("a")
        #pool.terminate()
        #print("b")
        #sys.exit()
        #event.set()


    #Case: 1 theory, multiple fragments
    results=[]
    if len(theories) == 1:
        theory = theories[0]
        print("Case: Multiple fragments but one theory")
        print("")
        print("Launching multiprocessing pool.map:")

        #Change theory numcores to 1 since we are running ASH in parallel
        #NOTE: Alternative to exit here instead ??
        if theory.numcores != 1:
            print(BC.WARNING,"Theory numcores set to:", theory.numcores, BC.END)
            print(BC.WARNING,"Since ASH is running in parallel we will now turn off Theory Parallelization",BC.END)
            theory.numcores=1

        #NOTE: Python 3.8 and higher use spawn in MacOS. Leads to ash import problems
        #NOTE: Unix/Linux uses fork which seems better behaved

        #Passing list of fragments
        if len(fragments) > 0:
            print("fragments:", fragments)
            for fragment in fragments:
                print("fragment:", fragment)
                results.append(pool.apply_async(Single_par, kwds=dict(theory=theory,fragment=fragment,label=fragment.label,mofilesdir=mofilesdir,event=event), error_callback=Terminate_Pool_processes))
        #Passing list of fragment files
        elif len(fragmentfiles) > 0:
            print("Launching multiprocessing and passing list of ASH fragmentfiles")
            for fragmentfile in fragmentfiles:
                print("fragmentfile:", fragmentfile)
                results.append(pool.apply_async(Single_par, kwds=dict(theory=theory,fragmentfile=fragmentfile,label=fragmentfile,mofilesdir=mofilesdir,event=event), error_callback=Terminate_Pool_processes))
    # Case: Multiple theories, 1 fragment
    elif len(fragments) == 1:
        print("Case: Multiple theories but one fragment")
        fragment = fragments[0]
        #results = pool.map(Single_par, [[theory,fragment, theory.label, event] for theory in theories])
        for theory in theories:
            print("theory:", theory)
            results.append(pool.apply_async(Single_par, kwds=dict(theory=theory,fragment=fragment,label=fragment.label,mofilesdir=mofilesdir,event=event), error_callback=Terminate_Pool_processes))
    # Case: Multiple theories, 1 fragmentfile
    elif len(fragmentfiles) == 1:
        print("Case: Multiple theories but one fragmentfile")
        fragmentfile = fragmentfiles[0]
        for theory in theories:
            print("theory:", theory)
            results.append(pool.apply_async(Single_par, kwds=dict(theory=theory,fragmentfile=fragmentfile,label=fragmentfile,mofilesdir=mofilesdir,event=event), error_callback=Terminate_Pool_processes))
    else:
        print("Multiple theories and multiple fragments provided.")
        print("This is not supported. Exiting...")
        ashexit()

    pool.close()
    pool.join()
    event.set()

    #While loop that is only terminated if processes finished or exception occurred
    while True:
        print("Pool multiprocessing underway....")
        time.sleep(3)
        if event.is_set():
            print("Event has been set! Now terminating Pool processes")
            pool.terminate()
            break

    #Going through each result-object and adding to energy_dict if ready
    #This prevents hanging for ApplyResult.get() if Pool did not finish correctly
    energy_dict={}
    for i,r in enumerate(results):
        print("Result {} ready: {}".format(i, r.ready()))
        if r.ready() == True:
            energy_dict[r.get()[0]] = r.get()[1]

    #Dict comprehension to get results from list of Pool-ApplyResult objects
    #energy_dict = {result.get()[0]: result.get()[1] for result in results}
    #print("energy_dict:", energy_dict)

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
    energy, gradient = theory.run(current_coords=geo, elems=elems, Grad=True, numcores=numcores)
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
                            numcores=numcores, frozenatoms=frozenatoms)

    energy, gradient = qmmmobject.run(current_coords=coords, elems=elems, Grad=True, numcores=numcores)
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




# #Stripped down version of Singlepoint function for Singlepoint_parallel
# #NOTE: Old version that used Pool.map instead of apply_async. TO BE DELETED
# def Single_par_old(listx):
#     print("listx:", listx)
#     #Multiprocessing event
#     event = listx[4]

#     #Creating new copy of theory to prevent Brokensym feature from being deactivated by each run
#     #NOTE: Alternatively we can add an if-statement inside orca.run
#     theory=copy.deepcopy(listx[0])

#     #listx[1] is either an ASH Fragment or Fragment file string (avoids massive pickle object)
#     if listx[1].__class__.__name__ == "Fragment":
#         print("Here", listx[1].__class__.__name__)
#         fragment=listx[1]
#     elif listx[1].__class__.__name__ == "str":
#         if "ygg" in listx[1]:
#             print("Found string assumed to be ASH fragmentfile")
#             fragment=ash.Fragment(fragfile=listx[1])
#     else:
#         print("Unknown object passed")
#         kill_all_mp_processes()
#         ashexit()
#     print("Fragment:", fragment)

#     #Making label flexible. Can be tuple but inputfilename is converted to string below
#     label=listx[2]
#     mofilesdir=listx[3]
#     print("label: {} (type {})".format(label,type(label)))
#     if label == None:
#         print("No label provided to fragment or theory objects. This is required to distinguish between calculations ")
#         print("Exiting...")
#         print("event:", event)
#         print("event is_set: ", event.is_set())
#         event.set()
#         print("after event")
#         print("event is_set: ", event.is_set())
#         ashexit()
#         #sys.exit()
#         #kill_all_mp_processes()
#         #ashexit()

#     #Using label (could be tuple) to create a labelstring which is used to name worker directories
#     # Tuple-label (1 or 2 element) used by calc_surface functions.
#     # Otherwise normally string
#     if type(label) == tuple:
#         if len(label) ==2:
#             labelstring=str(str(label[0])+'_'+str(label[1])).replace('.','_')
#         else:
#             labelstring=str(str(label[0])).replace('.','_')
#         print("Labelstring:", labelstring)
#         #RC1_0.9-RC2_170.0.xyz
#         #orca_RC1_0.9RC2_170.0.gbw
#         #TODO: what if tuple is only a single number???

#         if mofilesdir != None:
#             print("Mofilesdir option.")
#             if len(label) == 2:
#                 moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])+'-'+'RC2_'+str(label[1])
#             else:
#                 moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])

#     #Label is not tuple. Not coming from calc_surface funcitons
#     elif type(label) == float or type(label) == int:
#         print("Label is float or int")
#         #
#         #Label is float or int. 
#         if mofilesdir != None:
#             print("Mofilesdir option.")
#             moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])
#     else:
#         print("Here. label.", label)
#         #Label is not tuple. String or single number
#         labelstring=str(label).replace('.','_')

#     #Creating separate inputfilename using label
#     #Removing . in inputfilename as ORCA can get confused
#     if theory.__class__.__name__ == "ORCATheory":
#         #theory.filename=''.join([str(i) for i in labelstring])
#         #NOTE: Not sure if we really need to use labelstring for input since inside separate directoreies
#         #Disabling for now
#         #theory.filename=labelstring
#         if mofilesdir != None:
#             theory.moreadfile=moreadfile_path+'.gbw'
#             print("Setting moreadfile to:", theory.moreadfile)
#     elif theory.__class__.__name__ == "MRCCTheory":
#         if mofilesdir != None:
#             print("Case MRCC MOREADfile parallel")
#             print("moreadfile_path:", moreadfile_path)
#         print("not finished. exiting")
#         kill_all_mp_processes()
#         ashexit()
#     else:
#         if mofilesdir != None:
#             print("moreadfile option not ready for this Theory. exiting")
#             kill_all_mp_processes()
#             ashexit()

#     #Creating new dir and running calculation inside
#     os.mkdir('Pooljob_'+labelstring)
#     os.chdir('Pooljob_'+labelstring)
#     print(BC.WARNING,"Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
#     print("\n\nProcess ID {} is running calculation with label: {} \n\n".format(mp.current_process(),label))

#     energy = theory.run(current_coords=fragment.coords, elems=fragment.elems, label=label)
#     os.chdir('..')
#     print("Energy: ", energy)
#     # Now adding total energy to fragment
#     fragment.energy = energy
#     return (label,energy)
