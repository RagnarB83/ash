import copy
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
import subprocess as sp
import os
import sys
import time
import shutil

#import ash
from ash.functions.functions_general import ashexit, BC,blankline,print_line_with_mainheader,print_line_with_subheader1
from ash.modules.module_coords import check_charge_mult, Fragment, read_xyzfile
from ash.modules.module_results import ASH_Results
from ash.modules.module_QMMM import QMMMTheory

###############################################
#CHECKS FOR OPENMPI
#NOTE: Perhaps to be moved to other location
###############################################
def check_OpenMPI():
    #Find mpirun and take path
    try:
        openmpibindir = os.path.dirname(shutil.which('mpirun'))
    except:
        print(BC.FAIL,"No mpirun found in PATH. Make sure to add OpenMPI to PATH in your environment/jobscript", BC.END)
        ashexit()
    print("OpenMPI binary directory found:", openmpibindir)
    #Test that mpirun is executable and grab OpenMPI version number for printout
    test_OpenMPI()
    return

def test_OpenMPI():
    print("Testing that mpirun is executable...", end="")
    p = sp.Popen(["mpirun", "-V"], stdout = sp.PIPE)
    out, err = p.communicate()
    mpiversion=out.split()[3].decode()
    print(BC.OKGREEN,"yes",BC.END)
    print("OpenMPI version:", mpiversion)


#########################################
#PARALLEL Single-point energy function. 
#########################################
#Used for standalone SP calculations, also used by NumFreq and NEB

#will run over fragments or fragmentfiles, over theories or both
#mofilesdir. Directory containing MO-files (GBW files for ORCA). Usef for multiple fragment option
#NOTE: Experimental copytheory option
#NOTE: Added threadpool option. Not sure if useful
def Singlepoint_parallel(fragments=None, fragmentfiles=None, theories=None, numcores=None, mofilesdir=None, 
                         allow_theory_parallelization=False, Grad=False, printlevel=2, copytheory=False,
                         threadpool=False):
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
    print()
    if printlevel >= 2:
        print_line_with_subheader1("Singlepoint_parallel function")
        print("Number of CPU cores available: ", numcores)
        if isinstance(theory, QMMMTheory):
            print("Warning: Singlepoint_parallel using QMMMTheory with OpenMMTheory MM is experimental")
            print("Specifically there are issues with platform='CPU'.")
            print("Try platform='Reference' instead or GPU options OpenCL or CUDA if possible")
    if printlevel >= 2:
        print("Number of theories:", len(theories))
        print("Running single-point calculations in parallel")
        print("Mofilesdir:", mofilesdir)
        print(BC.WARNING, "Warning: Output from Singlepoint_parallel will be erratic due to simultaneous output from multiple workers", BC.END)
    
    #Early exits
    if fragments == None and fragmentfiles == None:
        print(BC.FAIL,"Singlepoint_parallel requires a list of ASH fragments or a list of fragmentfilenames",BC.END)
        ashexit()
    if theories == None or numcores == None :
        print("theories:", theories)
        print("numcores:", numcores)
        print(BC.FAIL,"Singlepoint_parallel requires a theory object and a numcores value",BC.END)
        ashexit()
    #Fragment objects passed or name of fragmentfiles
    if fragments != None:
        if printlevel >= 2:
            print("Number of fragments:", len(fragments))
    else:
        fragments=[]
    if fragmentfiles != None:
        if printlevel >= 2:
            print("Number of fragmentfiles:", len(fragmentfiles))
    else:
        fragmentfiles=[]

    
    ###############################
    # Multiprocessing Pool setup
    ###############################
    #NOTE: Python 3.8 and higher use spawn in MacOS (ash import problems). Unix/Linux uses fork
    #Option to use Threadpool instead (probably not useful)    
    if threadpool is True:
        pool=ThreadPool(numcores)
    else:
        pool = Pool(numcores)
    
    #Manager
    manager = mp.Manager()
    event = manager.Event()

    #Function to handle exception of child processes
    def Terminate_Pool_processes(message):
        print(BC.FAIL,"Terminating Pool processes due to exception", BC.END)
        print(BC.FAIL,"Exception message:", message, BC.END)
        pool.terminate()
        event.set()
        ashexit()


    ##############################################################
    # Calling Pool for different fragment vs. theory scenarios
    ###############################################################
    #Case: 1 theory, multiple fragments
    results=[]
    if len(theories) == 1:
        theory = theories[0]
        if printlevel >= 2:
            print("Case: Multiple fragments but one theory")
            print("")
            print("Launching multiprocessing pool.apply_async:")

            print(BC.WARNING,"Singlepoint_parallel numcores set to:", numcores, BC.END)
            print(BC.WARNING,f"ASH will run {numcores} jobs simultaneously", BC.END)

        #Whether to allow theory parallelization or not
        if theory.numcores != 1:
            if printlevel >= 2:
                print(BC.WARNING,"WARNING: Theory numcores set to:", theory.numcores, BC.END)
            if allow_theory_parallelization is True:
                totnumcores=numcores*theory.numcores
                if printlevel >= 2:
                    print(BC.WARNING,"allow_theory_parallelization is True.", BC.END)
                    print(BC.WARNING,f"Each job can use {theory.numcores} CPU cores, thus up to {totnumcores} CPU cores can be running simultaneously. Make sure that that's how many slots are available.", BC.END)
            else:
                if printlevel >= 2:
                    print(BC.WARNING,"allow_theory_parallelization is False. Now turning off theory.parallelization (setting theory numcores to 1)", BC.END)
                    print(BC.WARNING,"This can be overriden by: Singlepoint_parallel(allow_theory_parallelization=True)\n", BC.END)
                theory.numcores=1


        #Passing list of fragments
        if len(fragments) > 0:
            if printlevel >= 2:
                print("fragments:", fragments)
            for fragment in fragments:
                if printlevel >= 2:
                    print("fragment:", fragment)
                results.append(pool.apply_async(Single_par, kwds=dict(theory=theory,fragment=fragment,label=fragment.label,mofilesdir=mofilesdir,event=event, Grad=Grad, printlevel=printlevel, copytheory=copytheory), 
                    error_callback=Terminate_Pool_processes))
        #Passing list of fragment files
        elif len(fragmentfiles) > 0:
            if printlevel >= 2:
                print("Launching multiprocessing and passing list of ASH fragmentfiles")
            for fragmentfile in fragmentfiles:
                if printlevel >= 2:
                    print("fragmentfile:", fragmentfile)
                results.append(pool.apply_async(Single_par, kwds=dict(theory=theory,fragmentfile=fragmentfile,label=fragmentfile,mofilesdir=mofilesdir,event=event, Grad=Grad, printlevel=printlevel, copytheory=copytheory), 
                    error_callback=Terminate_Pool_processes))
    # Case: Multiple theories, 1 fragment
    elif len(fragments) == 1:
        if printlevel >= 2:
            print("Case: Multiple theories but one fragment")
        fragment = fragments[0]
        #results = pool.map(Single_par, [[theory,fragment, theory.label, event] for theory in theories])
        for theory in theories:
            if printlevel >= 2:
                print("theory:", theory)
            results.append(pool.apply_async(Single_par, kwds=dict(theory=theory,fragment=fragment,label=fragment.label,mofilesdir=mofilesdir,event=event, Grad=Grad, printlevel=printlevel, copytheory=copytheory), 
                error_callback=Terminate_Pool_processes))
    # Case: Multiple theories, 1 fragmentfile
    elif len(fragmentfiles) == 1:
        if printlevel >= 2:
            print("Case: Multiple theories but one fragmentfile")
        fragmentfile = fragmentfiles[0]
        for theory in theories:
            if printlevel >= 2:
                print("theory:", theory)
            results.append(pool.apply_async(Single_par, kwds=dict(theory=theory,fragmentfile=fragmentfile,label=fragmentfile,mofilesdir=mofilesdir,event=event, Grad=Grad, printlevel=printlevel, copytheory=copytheory), 
                error_callback=Terminate_Pool_processes))
    else:
        print("Multiple theories and multiple fragments provided.")
        print("This is not supported. Exiting...")
        ashexit()

    pool.close()
    pool.join()
    event.set()

    #While loop that is only terminated if processes finished or exception occurred
    while True:
        if printlevel >= 2:
            print("Pool multiprocessing underway....")
        time.sleep(3)
        if event.is_set():
            if printlevel >= 2:
                print("Event has been set! Now terminating Pool processes")
            pool.terminate()
            break

    ##############################################################
    # END OF POOL
    ###############################################################


    ###########
    # RESULTS
    ###########
    #Going through each result-object and adding to energy_dict if ready
    #This prevents hanging for ApplyResult.get() if Pool did not finish correctly
    energy_dict={}

    result = ASH_Results(label="Singlepoint_parallel", energies=[], gradients=[])
    if Grad == True:
        gradient_dict={}
        for i,r in enumerate(results):
            if r.ready() == True:
                energy_dict[r.get()[0]] = r.get()[1]
                gradient_dict[r.get()[0]] = r.get()[2]
                result.energies.append(r.get()[1])
                result.gradients.append(r.get()[2])
        #return energy_dict,gradient_dict
        result.gradients_dict = gradient_dict
    else:
        for i,r in enumerate(results):
            #print("Result {} ready: {}".format(i, r.ready()))
            if r.ready() == True:
                energy_dict[r.get()[0]] = r.get()[1]
                result.energies.append(r.get()[1])
        #return energy_dict

    #Adding dictionary also
    result.energies_dict = energy_dict

    return result


#Version of Singlepoint used by Singlepoint_parallel.
#NOTE: Needs to be simplified
#NOTE: Version intended for apply_async
#TODO: This function contains 2 many QM-code specifics. Needs to be generalized (QM-specifics moved to QMtheory class)
def Single_par(fragment=None, fragmentfile=None, theory=None, label=None, mofilesdir=None, event=None, charge=None, mult=None, Grad=False, printlevel=2, copytheory=False):

    #Check charge/mult.
    charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Single_par", theory=theory, printlevel=printlevel)

    #BASIC PRINTING
    if printlevel >= 2:
        print("Fragment:", fragment)
        print("fragmentfile:", fragmentfile)
        print("Theory:", theory)

    #Creating new copy of theory to avoid deactivation of certain first-run features
    # Example: Brokensym feature in ORCATheory
    #NOTE: Alternatively add if-statement inside orca.run
    #NOTE: This is not compatible with Dualtheory
    if copytheory == True:
        #print("copytheory True")
        theory=copy.deepcopy(theory)
    else:
        #print("copytheory False")
        pass

    #Optional fragment-creation from disk
    if fragmentfile != None:
        if printlevel >= 2:
            print("Reading fragmentfile from disk")
        fragment=Fragment(fragfile=fragmentfile)

    ###############################
    # Labels distinguishing jobs 
    ###############################
    #Making label flexible. Can be tuple but inputfilename is converted to string below
    if printlevel >= 2:
        print("label: {} (type {})".format(label,type(label)))
    if label == None:
        print("No label provided to fragment or theory objects. This is required to distinguish between calculations ")
        print("Exiting.")
        raise Exception("Labelproblem")

    #Using label (could be tuple) to create a labelstring which is used to name worker directories
    # Tuple-label (1 or 2 element) used by calc_surface functions.
    # Otherwise normally string
    #TODO: Needs to be generalized.  Remove RC1, RC2 strings
    if type(label) == tuple:
        if len(label) ==2:
            labelstring=str(str(label[0])+'_'+str(label[1])).replace('.','_')
        else:
            labelstring=str(str(label[0])).replace('.','_')
        if printlevel >= 2:
            print("Labelstring:", labelstring)
        #RC1_0.9-RC2_170.0.xyz
        #orca_RC1_0.9RC2_170.0.gbw
        #TODO: what if tuple is only a single number???

        if mofilesdir != None:
            if printlevel >= 2:
                print("Mofilesdir option.")
            if len(label) == 2:
                moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])+'-'+'RC2_'+str(label[1])
            else:
                moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])

    #Label is not tuple. Not coming from calc_surface functions
    elif type(label) == float or type(label) == int:
        if printlevel >= 2:
            print("Label is float or int")
        #
        #Label is float or int. 
        if mofilesdir != None:
            if printlevel >= 2:
                print("Mofilesdir option.")
            moreadfile_path=mofilesdir+'/'+theory.filename+'_'+'RC1_'+str(label[0])
    else:
        #Label is not tuple. String or single number
        labelstring=str(label).replace('.','_')

    ###############################

    #Creating separate inputfilename using label
    #Removing . in inputfilename as ORCA can get confused
    #TODO: Need to revisit all of this, ideally remove
    if theory.__class__.__name__ == "ORCATheory":
        #theory.filename=''.join([str(i) for i in labelstring])
        #NOTE: Not sure if we really need to use labelstring for input since inside separate directoreies
        #Disabling for now
        #theory.filename=labelstring
        if mofilesdir != None:
            theory.moreadfile=moreadfile_path+'.gbw'
            if printlevel >= 2:
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

    ####################################
    # Handling Directory
    ####################################
    #Creating new dir and running calculation inside
    try:
        os.mkdir('Pooljob_'+labelstring)
    except:
        if printlevel >= 2:
            print("Dir exists. continuing")
        pass
    os.chdir('Pooljob_'+labelstring)
    if printlevel >= 2:
        print(BC.WARNING,"Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
        print("\n\nProcess ID {} is running calculation with label: {} \n\n".format(mp.current_process(),label))

    #####################
    #RUN WORKER JOB
    #####################
    if Grad == True:
        energy,gradient = theory.run(current_coords=fragment.coords, elems=fragment.elems, label=label, charge=charge, mult=mult, Grad=Grad)
    else:
        energy = theory.run(current_coords=fragment.coords, elems=fragment.elems, label=label, charge=charge, mult=mult)
    #####################

    if printlevel >= 2:
        print("Energy: ", energy)
    
    # Now adding total energy to fragment.
    #NOTE: Add to theory also?
    fragment.energy = energy

    #Exiting workerdir
    os.chdir('..')

    #Return label and energy or label, energy and gradient
    if Grad == True:
        return (label,energy,gradient)
    else:
        return (label,energy)














#Functions to run each displacement in parallel NumFreq run.
#Simple QM object
# def displacement_QMrun(arglist):
#     #print("arglist:", arglist)
#     geo = arglist[0]
#     elems = arglist[1]
#     #We can launch ORCA-OpenMPI in parallel it seems. Only makes sense if we have may more cores available than displacements
#     numcores = arglist[2]
#     theory = arglist[3]
#     label = arglist[4]
#     charge = arglist[5]
#     mult = arglist[6]
#     dispdir=label.replace(' ','')
#     os.mkdir(dispdir)
#     os.chdir(dispdir)
#     #Todo: Copy previous GBW file in here if ORCA, xtbrestart if xtb, etc.
#     print("Running displacement: {}".format(label))
#     energy, gradient = theory.run(current_coords=geo, elems=elems, Grad=True, numcores=numcores, charge=charge, mult=mult)
#     print("Energy: ", energy)
#     os.chdir('..')
#     #Delete dir?
#     #os.remove(dispdir)
#     return [label, energy, gradient]


#Function to run each displacement in parallel NumFreq run
#Version where geo is read from file to avoid large memory pickle inside pool.map
# def displacement_QMMMrun(arglist):
#     #global QMMM_xtb
#     print("arglist:", arglist)

#     #print("locals", locals())
#     #print("globals", globals())
#     print("-----------")
#     #import gc
#     #print(gc.get_objects())

#     filelabel=arglist[0]
#     #elems = arglist[1]
#     #Numcores can be used. We can launch ORCA-OpenMPI in parallel it seems. Only makes sense if we have may more cores available than displacements
#     numcores = arglist[1]
#     label = arglist[2]
#     fragment= arglist[3]
#     qm_theory = arglist[4]
#     mm_theory = arglist[5]
#     actatoms = arglist[6]
#     qmatoms = arglist[7]
#     embedding = arglist[8]
#     charges = arglist[9]
#     printlevel = arglist[10]
#     frozenatoms = arglist[11]

#     dispdir=label.replace(' ','')
#     os.mkdir(dispdir)
#     os.chdir(dispdir)
#     shutil.move('../'+filelabel+'.xyz','./'+filelabel+'.xyz')
#     # Read XYZ-file from file
#     elems,coords = read_xyzfile(filelabel+'.xyz')

#     #Todo: Copy previous GBW file in here if ORCA, xtbrestart if xtb, etc.
#     print("Running displacement: {}".format(label))

#     #If QMMMTheory init keywords are changed this needs to be updated
#     qmmmobject = QMMMTheory(fragment=fragment, qm_theory=qm_theory, mm_theory=mm_theory, actatoms=actatoms,
#                           qmatoms=qmatoms, embedding=embedding, charges=charges, printlevel=printlevel,
#                             numcores=numcores, frozenatoms=frozenatoms)

#     energy, gradient = qmmmobject.run(current_coords=coords, elems=elems, Grad=True, numcores=numcores)
#     print("Energy: ", energy)
#     os.chdir('..')
#     #Delete dir?
#     #os.remove(dispdir)
#     return [label, energy, gradient]




#Called from run_QMMM_SP_in_parallel. Runs
# def run_QM_MM_SP(list):
#     orcadir=list[0]
#     current_coords=list[1]
#     theory=list[2]
#     #label=list[3]
#     #Create new dir (name of label provided
#     #Cd dir
#     theory.run(Grad=True)

# def run_QMMM_SP_in_parallel(orcadir, list_of__geos, list_of_labels, QMMMtheory, numcores, threadpool=False):
#     import multiprocessing as mp
#     blankline()
#     print("Number of CPU cores: ", numcores)
#     print("Number of geos:", len(list_of__geos))
#     print("Running snapshots in parallel")
#     if threadpool is True:
#         pool=ThreadPool(numcores)
#     else:
#         pool = Pool(numcores)
#     results = pool.map(run_QM_MM_SP, [[orcadir,geo, QMMMtheory ] for geo in list_of__geos])
#     pool.close()
#     print("Calculations are done")

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
