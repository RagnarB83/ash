import copy
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
#NOTE: Can now either use built-in multiprocessing library or more reliable fork multiprocess.
#The latter uses dill serialization and should be more reliable
def Singlepoint_parallel(fragments=None, fragmentfiles=None, theories=None, numcores=None, mofilesdir=None, 
                         allow_theory_parallelization=False, Grad=False, printlevel=2, copytheory=False,
                         version='multiprocessing'):
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
        if isinstance(theories[0], QMMMTheory):
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
    if version == 'multiprocessing':
        print("Singlepoint_parallel: Using version: multiprocessing")
        import multiprocessing as mp
        from multiprocessing.pool import Pool
    #Active fork of multiprocessing that uses dill instead of pickle etc. https://github.com/uqfoundation/multiprocess
    elif version == 'multiprocess':
        print("Singlepoint_parallel: Using version: multiprocess")
        try:
            import multiprocess as mp
            from multiprocess.pool import Pool
        except ImportError:
            print("This requires the multiprocess library to be installed")
            print("Please install using pip: pip install multiprocess")
            ashexit()
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
    #import multiprocess as mp
    #from multiprocess.pool import Pool
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

