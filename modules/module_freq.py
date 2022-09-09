import numpy as np
import math
import shutil
import os
import copy
import time

from ash.functions.functions_general import ashexit, listdiff, clean_number,blankline,BC,print_time_rel, print_line_with_mainheader,isodd
import ash.modules.module_coords
from ash.modules.module_coords import check_charge_mult, check_multiplicity
import ash.interfaces.interface_ORCA
import ash.constants

#Analytical frequencies function
#Only works for ORCAtheory at the moment
def AnFreq(fragment=None, theory=None, charge=None, mult=None, numcores=1, temp=298.15, pressure=1.0):
    module_init_time=time.time()
    print(BC.WARNING, BC.BOLD, "------------ANALYTICAL FREQUENCIES-------------", BC.END)
    if theory.__class__.__name__ == "ORCATheory":
        print("Requesting analytical Hessian calculation from ORCATheory")
        print("")
        #Check charge/mult
        charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "AnFreq", theory=theory)
        #Do single-point ORCA Anfreq job
        energy = theory.run(current_coords=fragment.coords, elems=fragment.elems, charge=charge, mult=mult, Hessian=True, numcores=numcores)
        #Grab Hessian
        hessian = ash.interfaces.interface_ORCA.Hessgrab(theory.filename+".hess")
        #Add Hessian to fragment
        fragment.hessian=hessian
        
        #TODO: diagonalize it ourselves. Need to finish projection
        
        # For now, we grab frequencies from ORCA Hessian file
        frequencies = ash.interfaces.interface_ORCA.ORCAfrequenciesgrab(theory.filename+".hess")
        
        hessatoms=list(range(0,fragment.numatoms))
        freqoutputdict = thermochemcalc(frequencies,hessatoms, fragment, mult, temp=temp,pressure=pressure)

        #freqoutputdict object. Should contain frequencies, zero-point energy, enthalpycorr, gibbscorr, etc.
        freqoutputdict['hessian'] = hessian
        
        #TODO: To add once we diagonalize 
        #freqoutputdict['evectors'] = evectors
        #freqoutputdict['nmodes'] = nmodes
        #freqoutputdict['hessatoms'] = hessatoms

        print(BC.WARNING, BC.BOLD, "------------ANALYTICAL FREQUENCIES END-------------", BC.END)
        print_time_rel(module_init_time, modulename='AnFreq', moduleindex=1)
        return freqoutputdict
        
    else:
        print("Analytical frequencies not available for theory. Exiting.")
        ashexit()


#Numerical frequencies function
#ORCA uses 0.005 Bohr = 0.0026458861 Ang, CHemshell uses 0.01 Bohr = 0.00529 Ang
def NumFreq(fragment=None, theory=None, charge=None, mult=None, npoint=2, displacement=0.005, hessatoms=None, numcores=1, runmode='serial', 
        temp=298.15, pressure=1.0, hessatoms_masses=None, printlevel=1):
    module_init_time=time.time()
    print(BC.WARNING, BC.BOLD, "------------NUMERICAL FREQUENCIES-------------", BC.END)

    #Basic check
    if fragment is None or theory is None:
        print("NumFreq requires a fragment and a theory object")
        ashexit()

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "NumFreq", theory=theory)

    #Creating directory
    shutil.rmtree('Numfreq_dir', ignore_errors=True)
    os.mkdir('Numfreq_dir')
    os.chdir('Numfreq_dir')
    print("Creating separate directory for displacement calculations: Numfreq_dir ")
    
    #ORCA-specific: Copy old GBW file from .. dir 
    try:
        if theory.theorytype == "QM":
            if isinstance(theory,ash.interfaces.interface_ORCA.ORCATheory):
                print("Copying GBW file into Numfreq_dir")
                shutil.copy("../"+theory.filename+'.gbw', './'+theory.filename+'.gbw')
        elif theory.theorytype == "QM/MM":
            if isinstance(theory.qm_theory,ash.interfaces.interface_ORCA.ORCATheory):
                print("Copying GBW file into Numfreq_dir")
                shutil.copy('../'+theory.qm_theory.filename+'.gbw', './'+theory.qm_theory.filename+'.gbw')
    except:
        pass
    coords=fragment.coords
    elems=copy.deepcopy(fragment.elems)
    numatoms=len(elems)
    #Hessatoms list is allatoms (if hessatoms list not provided)
    #If hessatoms provided we do a partial Hessian
    allatoms=list(range(0,numatoms))
    if hessatoms is None:
        hessatoms=allatoms
    #Making sure hessatoms list is sorted
    hessatoms.sort()
    #Optional hessatoms_masses list
    if hessatoms_masses != None:
        if len(hessatoms_masses) != len(hessatoms):
            print(BC.FAIL,"Error: Number of provided masses (hessatoms_masses keyword) is not equal to number of Hessian-atoms.")
            print("Check input masses!",BC.END)
            ashexit()
    
    displacement_bohr = displacement * ash.constants.ang2bohr

    print("Starting Numerical Frequencies job for fragment")
    print("System size:", numatoms)
    print("Hessian atoms:", hessatoms)
    if hessatoms != allatoms:
        print("This is a partial Hessian job.")
        if len(hessatoms) == 0:
            print("hessatoms list is empty. Exiting.")
            ashexit()
    if npoint ==  1:
        print("One-point formula used (forward difference)")
    elif npoint == 2:
        print("Two-point formula used (central difference)")
    else:
        print("Unknown npoint option. npoint should be set to 1 (one-point) or 2 (two-point formula).")
        ashexit()
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
    ash.modules.module_coords.print_coords_for_atoms(coords,elems,hessatoms)
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
                list_of_displacements.append((atom_index, coord_index, '+'))
                if npoint == 2:
                    #Displacing  - direction
                    current_coords_array[atom_index,coord_index]=val-displacement
                    y = current_coords_array.copy()
                    list_of_displaced_geos.append(y)
                    list_of_displacements.append((atom_index, coord_index, '-'))
                #Displacing back
                current_coords_array[atom_index, coord_index] = val

    # Original geo added here if onepoint
    if npoint == 1:
        list_of_displaced_geos.append(current_coords_array)
        list_of_displacements.append('Originalgeo')

    if printlevel > 1:
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
            #calclabel2 = 'Atom: {} Coord: {} Direction: {}'.format(atom_disp, crd, drection)
            calclabel="Atom: {} Coord: {} Direction: {}".format(str(atom_disp),str(crd),str(drection))
        list_of_labels.append(calclabel)
        
    assert len(list_of_labels) == len(list_of_displaced_geos), "something is wrong"

    #Create ASH fragment and Write all geometries to disk as XYZ-files
    list_of_filelabels=[]
    all_disp_fragments=[]
    for label, dispgeo,disp in zip(list_of_labels,list_of_displaced_geos,list_of_displacements):
        filelabel=label.replace(' ','').replace(':','')
        list_of_filelabels.append(filelabel)
        ash.modules.module_coords.write_xyzfile(elems=elems, coords=dispgeo,name=filelabel, printlevel=printlevel)

        #Creating ASH fragments with label
        frag=ash.Fragment(coords=dispgeo, elems=elems,label=disp, printlevel=printlevel, charge=charge, mult=mult)
        all_disp_fragments.append(frag)

    #RUNNING displacements
    displacement_grad_dictionary = {}
    #TODO: Have serial use all_disp_fragments instead to be consistent with parallel
    if runmode == 'serial':
        print("Runmode: serial")
        #Looping over geometries and running.
        #   key: AtomNCoordPDirectionm   where N=atomnumber, P=x,y,z and direction m: + or -
        #   value: gradient
        for numdisp,(disp,label, geo) in enumerate(zip(list_of_displacements,list_of_labels,list_of_displaced_geos)):
            if label == 'Originalgeo':
                calclabel = 'Originalgeo'
                print("Doing original geometry calc.")
            else:
                calclabel=label
                #for index,(el,coord) in enumerate(zip(elems,coords))
                #displacement_jobname='Numfreq-Disp-'+'Atom'+str(atom_disp)+crd+drection
                print("Running displacement: {} / {}".format(numdisp+1,len(list_of_labels)))
                print(calclabel)
                #print("Displacing Atom:{} Coord:{} Direction:{}".format(disp[0],disp[1],disp[2]))
            theory.printlevel=printlevel
            energy, gradient = theory.run(current_coords=geo, elems=elems, Grad=True, numcores=numcores, charge=charge, mult=mult)
            #Keep QM outputfile for each displacement
            if theory.theorytype == "QM":
                try:
                    shutil.copy(theory.filename+'.out', theory.filename+'disp_'+str(numdisp)+'.out')
                except:
                    pass
            elif theory.theorytype == "QM/MM":
                try:
                    shutil.copy(theory.qm_theory.filename+'.out', theory.qm_theory.filename+'disp_'+str(numdisp)+'.out')
                except:
                    pass
            else:
                print("Warning. Unknown theorytype")

            #Adding gradient to dictionary for AtomNCoordPDirectionm
            displacement_grad_dictionary[disp] = gradient
    elif runmode == 'parallel':

        if isinstance(theory,ash.QMMMTheory):
            print("Numfreq in runmode='parallel' is not working with QM/MM at the moment.")
            ashexit()
            #NOTE: Because OpenMM can not be pickled. Possibly try enabling for MMTheory = NonbondedTheory
        
        print(f"Starting Numfreq calculations in parallel mode (numcores={numcores}) using Singlepoint_parallel")

        #Launching multiple ASH E+Grad calculations in parallel on list of ASH fragments: all_image_fragments
        en_dict,gradient_dict = ash.Singlepoint_parallel(fragments=all_disp_fragments, theories=[theory], numcores=numcores, 
            allow_theory_parallelization=True, Grad=True, printlevel=printlevel)

        #Gradient_dict is already correctly formatted
        displacement_grad_dictionary = gradient_dict

    #OLD code below. To be deleted probably. Attempts to avoid pickling problem
    elif runmode == 'parallel2':

        import multiprocessing as mp
        #import pickle4reducer
        #ctx = mp.get_context()
        #ctx.reducer = pickle4reducer.Pickle4Reducer()

        pool = mp.Pool(numcores)
        blankline()
        print("Running snapshots in parallel using multiprocessing.Pool")
        print("Number of CPU cores: ", numcores)
        print("Number of displacements:", len(list_of_displaced_geos))

        #NumcoresQM can be larger value (e.g. ORCA-parallelization). ORCA seems to run fine with OpenMPI without complaints.
        #However, this only makes sense to use if way more CPUs available than displacements.
        #Unlikely situation, so hardcoding to 1 for now.
        numcoresQM=1
        print("Setting numcores for theory object to: ", numcoresQM)
        #results = pool.map(displacement_run, [[geo, elems, numcoresQM, theory, label] for geo,label in zip(list_of_displaced_geos,list_of_labels)])
        #results = pool.map(displacement_run2, [[filelabel, numcoresQM, theory, label] for label,filelabel in zip(list_of_labels,list_of_filelabels)])

        #Reducing size of theory object
        #print("size of theory:", get_size(theory))
        #print("size of theory.coords:", get_size(theory.coords))
        #print("size of coords:", get_size(coords))
        theory.coords=[]
        theory.elems=[]
        theory.connectivity=[]
        print(theory)
        #print(theory.__dict__)
        #print("size of theory after del:", get_size(theory))

        #QMMM_xtb = QMMMTheory(fragment=Saddlepoint, qm_theory=xtbcalc, mm_theory=MMpart, actatoms=Centralmainfrag,
        #                      qmatoms=Centralmainfrag, embedding='Elstat', numcores=numcores)

        #results = pool.map(displacement_run2, [[filelabel, numcoresQM, label] for label,filelabel in zip(list_of_labels,list_of_filelabels)])

        #Because passing QMMMTheory is too big for pickle inside mp.Pool we create a new QMMMTheory object inside displacement funciont.
        #This means we need the components of theory object. Here distinguishing between QMMMTheory and other theory (QM theory)
        #Still seems to be too messy

        #https://towardsdatascience.com/10x-faster-parallel-python-without-python-multiprocessing-e5017c93cce1
        if theory.__class__.__name__ == "QMMMTheory":
            print("Numfreq with QMMMTheory")
            ray_library=True
            
            if ray_library == True:
                print("Ray parallelization is active")
                try:
                    import ray
                    ray.init(num_cpus = numcores)
                except:
                    print("Parallel QM/MM Numerical Frequencies require the ray library.")
                    print("Please install ray : pip install ray")
                    ashexit()
                
                if theory.mm_theory == "NonBondedTheory":
                    #Do pairpotentials before we begin if NonBondedTheoyr
                    theory.mm_theory.calculate_LJ_pairpotentials(qmatoms=theory.qmatoms, actatoms=theory.actatoms)
                    print("theory.mm_theory sigmaij", theory.mm_theory.sigmaij)
                #going to make QMMMTheory object a shared object that all workers can access
                theory_shared = ray.put(theory)
                @ray.remote
                def dispfunction_ray(label, filelabel, numcoresQM, theory_shared):
                    print("inside dispfunction")
                    print("label:", label)
                    print("filelabel:", filelabel)
                    print("theory_shared:", theory_shared)
                    # Numcores can be used. We can launch ORCA-OpenMPI in parallel it seems.
                    # Only makes sense if we have may more cores available than displacements
                    elems, coords = ash.modules.module_coords.read_xyzfile(filelabel + '.xyz')
                    dispdir = label.replace(' ', '')
                    os.mkdir(dispdir)
                    os.chdir(dispdir)
                    # Todo: Copy previous GBW file in here if ORCA, xtbrestart if xtb, etc.
                    print("Running displacement: {}".format(label))
                    energy, gradient = theory_shared.run(current_coords=coords, elems=elems, Grad=True, numcores=numcoresQM, charge=charge, mult=mult)
                    os.chdir('..')
                    # Delete dir?
                    # os.remove(dispdir)
                    return [label, energy, gradient]
                result_ids = [dispfunction_ray.remote(label,filelabel,numcoresQM,theory_shared) for label,filelabel in
                            zip(list_of_labels,list_of_filelabels)]
                results = ray.get(result_ids)
            else:
                results = pool.map(ash.functions.functions_parallel.displacement_QMMMrun, [[filelabel, numcoresQM, label, theory.fragment, theory.qm_theory, theory.mm_theory,
                                                        theory.actatoms, theory.qmatoms, theory.embedding, theory.charges, theory.printlevel,
                                                        theory.frozenatoms] for label,filelabel in zip(list_of_labels,list_of_filelabels)])
        #Passing QM theory directly
        else:
            results = pool.map(ash.functions.functions_parallel.displacement_QMrun, [[geo, elems, numcoresQM, theory, label, charge, mult] for geo,label in zip(list_of_displaced_geos,list_of_labels)])
        pool.close()

        #Gathering results in dictionary
        for result in results:
            print("result:", result)
            calclabel=result[0]
            energy=result[1]
            gradient=result[2]
            displacement_grad_dictionary[calclabel] = gradient
    else:
        print("Unknown runmode.")
        ashexit()
    
    ############################################
    print("Displacement calculations done.")
    if len(displacement_grad_dictionary) == 0:
        print("Missing gradients for displacement.")
        print("Something went wrong in Numfreq displacement calculations.")
        ashexit()
    print("Length of displacement_grad_dictionary", len(displacement_grad_dictionary))
    #Initialize empty Hessian
    hesslength=3*len(hessatoms)
    hessian=np.zeros((hesslength,hesslength))



    #Onepoint-formula Hessian
    if npoint == 1:
        print("Assembling the one-point Hessian")
        #First, grab original geometry gradient
        #If partial Hessian remove non-hessatoms part of gradient:
        #Get partial matrix by deleting atoms not present in list.
        original_grad=get_partial_matrix(displacement_grad_dictionary['Originalgeo'],hessatoms)
        #original_grad=get_partial_matrix(allatoms, hessatoms, displacement_grad_dictionary['Originalgeo'])
        print("Debugging 1b")
        original_grad_1d = np.ravel(original_grad)
        print("Debugging 1c")
        #Starting index for Hessian array
        hessindex=0
        #Loop over Hessian atoms and grab each gradient component. Calculate Hessian component and add to matrix
        #for atomindex in range(0,len(hessatoms)):
        for atomindex in hessatoms:
            print("Debugging 2x")
            #Iterate over x,y,z components
            for crd in [0,1,2]:
                #Looking up each gradient for atomindex, crd-component(x=0,y=1 or z=2) and '+' 
                timeA = time.time()
                grad_pos=displacement_grad_dictionary[(atomindex,crd,'+')]
                print_time_rel(timeA, modulename="grad_pos make", currprintlevel=1, currthreshold=1)
                timeA = time.time()
                 #Getting grad as numpy matrix and converting to 1d
                # If partial Hessian remove non-hessatoms part of gradient:
                #grad_pos = get_partial_matrix(allatoms, hessatoms, grad_pos)
                grad_pos = get_partial_matrix(grad_pos,hessatoms)
                print_time_rel(timeA, modulename="get_partial_matrix", currprintlevel=1, currthreshold=1)
                timeA = time.time()
                grad_pos_1d = np.ravel(grad_pos)
                print_time_rel(timeA, modulename="np ravel", currprintlevel=1, currthreshold=1)
                timeA = time.time()
                Hessrow=(grad_pos_1d - original_grad_1d)/displacement_bohr
                print_time_rel(timeA, modulename="Hessrow subtraction", currprintlevel=1, currthreshold=1)
                timeA = time.time()
                hessian[hessindex,:]=Hessrow
                grad_pos_1d=0
                hessindex+=1
        print("Debugging 3")
    #Twopoint-formula Hessian. pos and negative directions come in order
    elif npoint == 2:
        print("Assembling the two-point Hessian")

        hessindex=0
        #Loop over Hessian atoms and grab each gradient component. Calculate Hessian component and add to matrix
        #for atomindex in range(0,len(hessatoms)):
        for atomindex in hessatoms:
            #Iterate over x,y,z components
            for crd in [0,1,2]:
                #Looking up each gradient for atomindex, crd-component(x=0,y=1 or z=2) and '+' 
                grad_pos=displacement_grad_dictionary[(atomindex,crd,'+')]
                #Looking up each gradient for atomindex, crd-component(x=0,y=1 or z=2) and '-' 
                grad_neg=displacement_grad_dictionary[(atomindex,crd,'-')]
                 #Getting grad as numpy matrix and converting to 1d
                # If partial Hessian remove non-hessatoms part of gradient:
                #grad_pos = get_partial_matrix(allatoms, hessatoms, grad_pos)
                grad_pos = get_partial_matrix(grad_pos, hessatoms)
                grad_pos_1d = np.ravel(grad_pos)
                #grad_neg = get_partial_matrix(allatoms, hessatoms, grad_neg)
                grad_neg = get_partial_matrix(grad_neg, hessatoms)
                grad_neg_1d = np.ravel(grad_neg)
                Hessrow=(grad_pos_1d - grad_neg_1d)/(2*displacement_bohr)
                hessian[hessindex,:]=Hessrow
                grad_pos_1d=0
                grad_neg_1d=0
                hessindex+=1
    print("Debugging 4")
    print()

    #Symmetrize Hessian by taking average of matrix and transpose
    symm_hessian=(hessian+hessian.transpose())/2
    print("Debugging 5")
    hessian=symm_hessian


    #Project out Translation+Rotational modes
    #TODO
    #NOTE: Only if no frozen atoms


    #Diagonalize mass-weighted Hessian
    # Get partial matrix by deleting atoms not present in list.
    print("Debugging 6")
    hesselems = ash.modules.module_coords.get_partial_list(allatoms, hessatoms, elems)
    print("Debugging 7")
    #Use input masses if given, otherwise take from frament
    if hessatoms_masses == None:
        hessmasses = ash.modules.module_coords.get_partial_list(allatoms, hessatoms, fragment.list_of_masses)
    else:
        hessmasses=hessatoms_masses
    print("Debugging 8")
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
    print("Eigenvectors to be printed here")
    blankline()
    #Print out Freq output. Maybe print normal mode compositions here instead???
    printfreqs(frequencies,len(hessatoms))

    print("\n\n")
    print("Normal mode composition factors by element")
    printfreqs_and_nm_elem_comps(frequencies,fragment,evectors,hessatoms=hessatoms)

    print("\nNow doing thermochemistry")

    #Get and print out thermochemistry
    if theory.__class__.__name__ == "QMMMTheory":
        freqoutputdict = thermochemcalc(frequencies,hessatoms, fragment, mult, temp=temp,pressure=pressure)
    else:
        freqoutputdict = thermochemcalc(frequencies,hessatoms, fragment, mult, temp=temp,pressure=pressure)

    #Write Hessian to file
    write_hessian(hessian,hessfile="Hessian")

    #Write ORCA-style Hessian file. Hardcoded filename here. Change?
    #Note: Passing hesscords here instead of coords. Change?
    ash.interfaces.interface_ORCA.write_ORCA_Hessfile(hessian, hesscoords, hesselems, hessmasses, hessatoms, "orcahessfile.hess")
    print("Wrote ORCA-style Hessian file: orcahessfile.hess")

    #Create dummy-ORCA file with frequencies and normal modes
    printdummyORCAfile(hesselems, hesscoords, frequencies, evectors, nmodes, "orcahessfile.hess")
    print("Wrote dummy ORCA outputfile with frequencies and normal modes: orcahessfile.hess_dummy.out")
    print("Can be used for visualization")

    blankline()
    print(BC.WARNING, BC.BOLD, "------------NUMERICAL FREQUENCIES END-------------", BC.END)

    #freqoutputdict object. Should contain frequencies, zero-point energy, enthalpycorr, gibbscorr, etc.
    freqoutputdict['hessian'] = hessian
    freqoutputdict['evectors'] = evectors
    freqoutputdict['nmodes'] = nmodes
    freqoutputdict['hessatoms'] = hessatoms

    #Add things to fragment
    fragment.hessian=hessian #Hessian


    #Return to ..
    os.chdir('..')
    print_time_rel(module_init_time, modulename='NumFreq', moduleindex=1)
    return freqoutputdict




#HESSIAN-related functions below


#Get partial matrix by deleting rows not present in list of indices.
#Deletes numpy rows, stupid and slow, to be deleted
def old_get_partial_matrix(allatoms,hessatoms,matrix):
    nonhessatoms=listdiff(allatoms,hessatoms)
    nonhessatoms.reverse()
    for at in nonhessatoms:
        matrix=np.delete(matrix, at, 0)
    return matrix

#Get partial matrix properly
def get_partial_matrix(matrix,hessatoms):
    return np.take(matrix,hessatoms, axis=0)


# Open-source project in Fortran:
#https://github.com/zorkzou/UniMoVib
#Calculates Hessian etc. Has IR and Raman intensity
#Todo: Add IR/Raman intensity support


#Taken from Hess-tool on 21st Dec 2019. Modified to read in Hessian array instead of ORCA-hessfile
def diagonalizeHessian(hessian, masses, elems):
    print("Diagonalizing Hessian")
    # Grab masses, elements and numatoms from Hessianfile
    #masses, elems, numatoms = masselemgrab(hessfile)
    numatoms=len(elems)
    atomlist = []
    for i, j in enumerate(elems):
        atomlist.append(str(j) + '-' + str(i))
    # Massweight Hessian
    mwhessian, massmatrix = massweight(hessian, masses, numatoms)

    # Diagonalize mass-weighted Hessian
    evalues, evectors = np.linalg.eigh(mwhessian)
    evectors = np.transpose(evectors)

    # Calculate frequencies from eigenvalues
    vfreqs = calcfreq(evalues)

    # Unweight eigenvectors to get normal modes
    nmodes = np.dot(evectors, massmatrix)
    return vfreqs,nmodes,numatoms,elems,evectors,atomlist,masses


# Massweight Hessian
def massweight(matrix,masses,numatoms):
    mass_mat = np.zeros( (3*numatoms,3*numatoms), dtype = float )
    molwt = [ masses[int(i)] for i in range(numatoms) for j in range(3) ]
    for i in range(len(molwt)):
        mass_mat[i,i] = molwt[i] ** -0.5
    mwhessian = np.dot((np.dot(mass_mat,matrix)),mass_mat)
    return mwhessian,mass_mat

# Calculate frequencies from eigenvalus
def calcfreq(evalues):
    hartree2j = ash.constants.hartree2j
    bohr2m = ash.constants.bohr2m
    amu2kg = ash.constants.amu2kg
    c = ash.constants.c
    pi = ash.constants.pi
    evalues_si = [val*hartree2j/bohr2m/bohr2m/amu2kg for val in evalues]
    vfreq_hz = [1/(2*pi)*np.sqrt(np.complex_(val)) for val in evalues_si]
    vfreq = [val/c for val in vfreq_hz]
    return vfreq


#NOTE: THIS IS NOT CORRECT
#TODO: Need to identify SP mode
#FOR SADDLEPOINT, the SP mode will be the largest imaginary mode, hence mode 0.
def printfreqs(vfreq,numatoms):
    if numatoms == 2:
        TRmodenum=5
    else:
        TRmodenum=6
    line = "{:>4}{:>14}".format("Mode", "Freq(cm**-1)")
    print(line)
    for mode in range(0,3*numatoms):
        realpart=vfreq[mode].real
        imagpart=vfreq[mode].imag
        if realpart == 0.0:
            vib=imagpart
            line = "{:>3d}   {:>9.4f}i".format(mode, vib)
        elif imagpart == 0.0:
            vib=clean_number(vfreq[mode])
            line = "{:>3d}   {:>9.4f}".format(mode, vib)
        else:
            print("vfreq[mode]:", vfreq[mode])
            print("realpart:", realpart)
            print("imagpart:", imagpart)
            print("This should not have happened")
            ashexit()
        if mode < TRmodenum:
            line=line+" (TR mode)"
        print(line)
        #print("vib:", vib)
        #print("type of vib", type(vib))



#
def thermochemcalc(vfreq,atoms,fragment, multiplicity, temp=298.15,pressure=1.0):
    module_init_time=time.time()
    """[summary]

    Args:
        vfreq ([list]): list of vibrational frequencies in cm**-1
        atoms ([type]): number of active atoms (contributing to Hessian) 
        fragment ([type]): ASH fragment object
        multiplicity ([type]): spin multiplicity
        temp (float, optional): [description]. Defaults to 298.15.
        pressure (float, optional): [description]. Defaults to 1.0.

    Returns:
        dictionary with thermochemistry properties
    """
    blankline()
    print("Thermochemistry via rigid-rotor harmonic oscillator approximation")
    if len(atoms) == 1:
        print("System is an atom.")
        moltype="atom"
    elif len(atoms) == 2:
        print("System is 2-atomic and thus linear")
        moltype="linear"
        TRmodenum=5
    else:
        #TODO: Need to detect linearity properly
        print("System size N > 2, assumed to be nonlinear")
        moltype="nonlinear"
        TRmodenum=6
    
    coords=fragment.coords
    elems=fragment.elems

    masses=fragment.list_of_masses
    totalmass=sum(masses)
    
    ###################
    #VIBRATIONAL PART
    ###################
    if moltype != "atom":
        freqs=[]
        vibtemps=[]
        for mode in range(0, 3 * len(atoms)):
            if mode < TRmodenum:
                continue
                #print("skipping TR mode with freq:", clean_number(vfreq[mode]) )
            else:
                vib = clean_number(vfreq[mode])
                if np.iscomplex(vib):
                    print("Mode {} with frequency {} is imaginary. Skipping in thermochemistry".format(mode,vib))
                elif vib < 0:
                    print("Mode {} with frequency {} is negative. Skipping in thermochemistry".format(mode,vib))
                else:
                    freqs.append(float(vib))
                    freq_Hz=vib*ash.constants.c
                    vibtemp=(ash.constants.h_planck_hartreeseconds * freq_Hz) / ash.constants.R_gasconst
                    vibtemps.append(vibtemp)

        #Zero-point vibrational energy
        zpve=sum([i*ash.constants.halfhcfactor for i in freqs])

        #Thermal vibrational energy
        sumb=0.0
        for v in vibtemps:
            #print(v*(0.5+(1/(np.exp((v/temp) - 1)))))
            sumb=sumb+v*(0.5+(1/(np.exp((v/temp) - 1))))
        E_vib=sumb*ash.constants.R_gasconst
        vibenergycorr=E_vib-zpve

        #Vibrational entropy via RRHO.
        #TODO: Add Grimme QRRHO and others
        #TODO: add functions for other contributions
        TS_vib = S_vib(freqs,temp)
    else:
        zpve=0.0
        E_vib=0.0
        freqs=[]
        vibenergycorr=0.0
        TS_vib=0.0

    ###################
    #ROTATIONAL PART
    ###################
    if moltype != "atom":
        # Moments of inertia (amu A^2 ), eigenvalues
        center = get_center(elems,coords)
        rinertia = list(inertia(elems,coords,center))
        print("Moments of inertia (amu Å^2):", rinertia)
        #Changing units to m and kg
        I=np.array(rinertia)*ash.constants.amu2kg*ash.constants.ang2m**2

        #Rotational temperatures
        #k_b_JK or R_JK
        rot_temps_x=ash.constants.h_planck**2 / (8*math.pi**2 * ash.constants.k_b_JK * I[0])
        rot_temps_y=ash.constants.h_planck**2 / (8*math.pi**2 * ash.constants.k_b_JK * I[1])
        rot_temps_z=ash.constants.h_planck**2 / (8*math.pi**2 * ash.constants.k_b_JK * I[2])
        print("Rotational temperatures: {}, {}, {} K".format(rot_temps_x,rot_temps_y,rot_temps_z))
        #Rotational constants
        rotconstants = calc_rotational_constants(fragment, printlevel=1)
        
        #Rotational energy and entropy
        if moltype == "atom":
            q_r=1.0
            S_rot=0.0
            E_rot=0.0
        elif moltype == "linear":
            #Symmetry number
            sigma_r=1.0
            q_r=(1/sigma_r)*(temp/(rot_temps_x))
            S_rot=ash.constants.R_gasconst*(math.log(q_r)+1.0)
            E_rot=ash.constants.R_gasconst*temp
        else:
            #Nonlinear case
            #Symmetry number hardcoded. TODO: properly
            sigma_r=2.0
            q_r=(math.pi**(1/2) / sigma_r ) * (temp**(3/2)) / ((rot_temps_x*rot_temps_y*rot_temps_z)**(1/2))
            S_rot=ash.constants.R_gasconst*(math.log(q_r)+1.5)
            E_rot=1.5*ash.constants.R_gasconst*temp
        TS_rot=temp*S_rot
    else:
        E_rot=0.0
        TS_rot=0.0

    ###################
    #TRANSLATIONAL PART
    ###################
    E_trans=1.5*ash.constants.R_gasconst*temp
    
    #R gas constant in kcal/molK
    R_kcalpermolK=1.987E-3
    #Conversion factor for formula.
    #TODO: cleanup
    factor=0.025607868
    #Translation partition function and T*S_trans. Using kcal/mol
    qtrans=(factor*temp**2.5*totalmass**1.5)/pressure
    S_trans=R_kcalpermolK*(math.log(qtrans)+2.5)
    
    TS_trans=temp*S_trans/ash.constants.harkcal #Energy term converted to Eh

    #######################
    #Electronic entropy
    #######################
    q_el=multiplicity
    S_el=ash.constants.R_gasconst*math.log(q_el)
    TS_el=temp*S_el

    #######################
    # Thermodynamic corrections
    #######################
    E_tot = E_vib + E_trans + E_rot
    Hcorr = E_vib + E_trans + E_rot + ash.constants.R_gasconst*temp
    TS_tot = TS_el + TS_trans + TS_rot + TS_vib
    Gcorr = Hcorr - TS_tot


    #######################
    #PRINTING
    #######################
    print("")
    print("Thermochemistry")
    print("--------------------")
    print("Temperature:", temp, "K")
    print("Pressure:", pressure, "atm")
    #print("Total atomlist:", fragment.atomlist)
    print("Hessian atomlist:", atoms)
    #print("Masses:", masses)
    print("Total mass:", totalmass)
    print("")

    if moltype != "atom":
        print("Moments of inertia:", rinertia)
        print("Rotational constants (cm-1):", rotconstants)

    print("")
    #Thermal corrections
    print("Energy corrections:")
    print("Zero-point vibrational energy:", zpve)
    print("{} {} {} {} {}".format("Translational energy (", temp, "K) :", E_trans, "Eh"))
    print("{} {} {} {} {}".format("Rotational energy (", temp, "K) :", E_rot, "Eh"))
    print("{} {} {} {} {}".format("Total vibrational energy (", temp, "K) :", E_vib, "Eh"))
    print("{} {} {} {} {}".format("Vibrational energy correction (", temp, "K) :", vibenergycorr, "Eh"))
    print("")
    print("Entropy terms (TS):")
    print("{} {} {} {} {}".format("Translational entropy (TS_trans) (", temp, "K) :", TS_trans, "Eh"))
    print("{} {} {} {} {}".format("Rotational entropy (TS_rot) (", temp, "K) :", TS_rot, "Eh"))
    print("{} {} {} {} {}".format("Vibrational entropy (TS_vib) (", temp, "K) :", TS_vib, "Eh"))
    print("{} {} {} {} {}".format("Electronic entropy (TS_el) (", temp, "K) :", TS_el, "Eh"))
    print("")
    if moltype != "atom":
        print("Note: symmetry number : {} used for rotational entropy".format(sigma_r))
        print("")
    print("Thermodynamic terms:")
    print("{} {} {} {} {}".format("Enthalpy correction (Hcorr) (", temp, "K) :", Hcorr, "Eh"))
    print("{} {} {} {} {}".format("Entropy correction (TS_tot) (", temp, "K) :", TS_tot, "Eh"))
    print("{} {} {} {} {}".format("Gibbs free energy correction (Gcorr) (", temp, "K) :", Gcorr, "Eh"))
    print("")
    
    #Dict with properties
    thermochemcalc = {}
    thermochemcalc['frequencies'] = freqs
    thermochemcalc['ZPVE'] = zpve
    thermochemcalc['E_trans'] = E_trans
    thermochemcalc['E_rot'] = E_rot
    thermochemcalc['E_vib'] = E_vib
    thermochemcalc['E_tot'] = E_tot
    thermochemcalc['TS_trans'] = TS_trans
    thermochemcalc['TS_rot'] = TS_rot
    thermochemcalc['TS_vib'] = TS_vib
    thermochemcalc['TS_el'] = TS_el
    thermochemcalc['vibenergycorr'] = vibenergycorr
    thermochemcalc['Hcorr'] = Hcorr
    thermochemcalc['Gcorr'] = Gcorr
    thermochemcalc['TS_tot'] = TS_tot
    
    print_time_rel(module_init_time, modulename='thermochemcalc', moduleindex=4)
    return thermochemcalc

#From Hess-tool.py: Copied 13 May 2020
#Print dummy ORCA outputfile using coordinates and normal modes. Used for visualization of modes in Chemcraft
#TODO: Needs fix for TS mode
def printdummyORCAfile(elems,coords,vfreq,evectors,nmodes,hessfile):
    orca_header = """                                 *****************
                                 * O   R   C   A *
                                 *****************

           --- An Ab Initio, DFT and Semiempirical electronic structure package ---

                       *****************************
                       * Geometry Optimization Run *
                       *****************************

         *************************************************************
         *                GEOMETRY OPTIMIZATION CYCLE   1            *
         *************************************************************
---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------"""
    #Very simple check for 2-atom linear molecule
    #Todo: Need to add support for n-atom linear molecule (HCN e.g.)
    if len(elems) == 2:
        TRmodenum=5
    else:
        TRmodenum=6

    outfile = open(hessfile+'_dummy.out', 'w')
    outfile.write(orca_header+'\n')
    for el,coord in zip(elems,coords):
        x=coord[0];y=coord[1];z=coord[2]
        line = "  {0:2s} {1:11.6f} {2:12.6f} {3:13.6f}".format(el, x, y, z)
        #print(line)
        #print('  S     51.226907   65.512868  106.021030')
        #exit()
        outfile.write(line+'\n')
    outfile.write('\n')
    outfile.write('-----------------------\n')
    outfile.write('VIBRATIONAL FREQUENCIES\n')
    outfile.write('-----------------------\n')
    outfile.write('\n')
    outfile.write('Scaling factor for frequencies =  1.000000000 (Found in file - NOT applied to frequencies read from HESS file)\n')
    outfile.write('\n')
    numatoms=(len(elems))
    complexflag=False
    for mode in range(3*numatoms):
        smode = str(mode) + ':'
        #if mode < TRmodenum:
        #    freq=0.00
        #else:
        freq=clean_number(vfreq[mode])
        if np.iscomplex(freq):
            imagfreq=-1*abs(freq)
            complexflag=True
        else:
            complexflag=False
        if complexflag==True:
            line= "{0:>5s}{1:13.2f} cm**-1 ***imaginary mode***".format(smode, imagfreq)
        else:
            line= "{0:>5s}{1:13.2f} cm**-1".format(smode, freq)
        outfile.write(line+'\n')



    normalmodeheader="""------------
NORMAL MODES
------------

These modes are the cartesian displacements weighted by the diagonal matrix
M(i,i)=1/sqrt(m[i]) where m[i] is the mass of the displaced atom
Thus, these vectors are normalized but *not* orthogonal"""

#TODO: Finish write normal mode output in ORCA format from nmodes so that Chemcraft can read it

    outfile.write('\n')
    outfile.write('\n')
    outfile.write(normalmodeheader)
    outfile.write('\n')
    outfile.write('\n')

    orcahesscoldim = 6
    hessdim=3*numatoms
    hessrow = []
    index = 0
    line = ""
    chunkheader = ""

    chunks = hessdim // orcahesscoldim
    left = hessdim % orcahesscoldim
    #print("Chunks:", chunks)
    #print("left:", left)
    if left > 0:
        chunks = chunks + 1
    #print("Chunks:", chunks)

    #print("evectors", evectors)
    #print("")
    #print("nmodes", nmodes)
    #print("len(nmodes)", len(nmodes))
    #TODO: Should we be using the eigenvectors instead, i.e. the mass-weighted normalmodes.
    #Seems to be what ORCA is using?

    #Transpose of nmodes for convenience
    #nmodes_tp=np.transpose(nmodes)
    #print(nmodes_tp)
    #print("")

    #print("Beginning for loop")
    for chunk in range(chunks):
        #print("chunk is", chunk)
        if chunk == chunks - 1:
            #print("a last chunk is", chunk)
            # If last chunk and cleft is exactly 0 then all 5 columns should be done
            if left == 0:
                left = 6
            # print("index is", index)
            # print("left is", left)
            for temp in range(index, index + left):
                chunkheader = chunkheader + "          " + str(temp)
            #print(chunkheader)
        else:
            for temp in range(index, index + orcahesscoldim):
                chunkheader = chunkheader + "          " + str(temp)
            #print(chunkheader)
        outfile.write("        "+str(chunkheader) + "    \n")
        for i in range(0, hessdim):
            firstcolumnindex=6*chunk
            j=firstcolumnindex
            #print("firstcolumnindex j is:", firstcolumnindex)
            #print("i is", i)
            #print("nmodes[i]", nmodes[i])
            #print("hessdim:", hessdim)
            #If chunk = 0 then we are dealing with TR modes in first 6 columns
            if chunk == 0:
                val1 = 0.0; val2 = 0.0;val3 = 0.0; val4 = 0.0; val5 = 0.0;val6 = 0.0
            else :
                #TODO: Here defning values to print based on values in nmodes matrix. TO be confiremd that this is correct. TODO.
                if hessdim - j == 1:
                    val1 = nmodes[j][i]
                elif hessdim - j == 2:
                    val1 = nmodes[j][j]; val2 = nmodes[j+1][i]
                elif hessdim - j == 3:
                    val1 = nmodes[j][i]; val2 = nmodes[j+1][i];val3 = nmodes[j+2][i]
                elif hessdim - j == 4:
                    val1 = nmodes[j][i]; val2 = nmodes[j+1][i];val3 = nmodes[j+2][i];val4 = nmodes[j+3][i]
                elif hessdim - j == 5:
                    val1 = nmodes[j][i]; val2 = nmodes[j+1][i];val3 = nmodes[j+2][i];val4 = nmodes[j+3][i];val5 = nmodes[j+4][i]
                elif hessdim - j >= 6:
                    val1 = nmodes[j][i]; val2 = nmodes[j+1][i];val3 = nmodes[j+2][i];val4 = nmodes[j+3][i];val5 = nmodes[j+4][i];val6 = nmodes[j+5][i]
                else:
                    print("problem")
                    print("hessdim - j : ", hessdim - j)
                    ashexit()

            if chunk == chunks - 1:
                #print("b last chunk is", chunk)
                # print("index is", index)
                # print("index+orcahesscoldim-left is", index+orcahesscoldim-left)
                for k in range(index, index + left):
                    #print("i is", i, "and k is", k)
                    #print("left:", left)
                    if left == 6:
                        line = "{:>6d} {:>14.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(i, val1, val2, val3, val4, val5, val6)
                    elif left == 5:
                        line = "{:>6d} {:>14.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(i, val1, val2, val3, val4, val5)
                    elif left == 5:
                        line = "{:>6d} {:>14.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(i, val1, val2, val3, val4)
                    elif left == 3:
                        line = "{:>6d} {:>14.6f} {:>10.6f} {:>10.6f}".format(i, val1, val2, val3)
                    elif left ==2:
                        line = "{:>6d}} {:>14.6f} {:>10.6f}".format(i, val1, val2)
                    elif left == 1:
                        line = "{:>6d} {:>14.6f}".format(i, val1)
            else:
                for k in range(index, index + orcahesscoldim):
                    #print("i is", i, "and k is", k)
                    line = "{:>6d} {:>14.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(i, val1, val2, val3, val4, val5, val6)
            #print(line)
            #outfile.write("    " + str(i) + "   " + str(line) + "\n")
            outfile.write(" " + str(line) + "\n")
            line = "";
            chunkheader = ""
        index += 6

    irtable="""

-----------
IR SPECTRUM
-----------

 Mode   freq       eps      Int      T**2         TX        TY        TZ
DUMMY NUMBERS BELOW
----------------------------------------------------------------------------

 """
    outfile.write(irtable)
    for i in range(6,3*numatoms):
        d=str(i)+":"
        outfile.write(f"{d:>4s}   1606.67   0.009763   49.34  0.001896  ( 0.000000 -0.000000 -0.043546)\n")

    outfile.close()
    print("Created dummy ORCA outputfile: ", hessfile+'_dummy.out')
    
    

#Center of mass, adapted from https://code.google.com/p/pistol/source/browse/trunk/Pistol/Thermo.py?r=4
def get_center(elems,coords):
    "compute center of mass"
    xcom,ycom,zcom = 0,0,0
    totmass = 0
    for el,coord in zip(elems,coords):
        mass = ash.modules.module_coords.atommasses[ash.modules.module_coords.elematomnumbers[el.lower()]-1]
        xcom += float(mass)*float(coord[0])
        ycom += float(mass)*float(coord[1])
        zcom += float(mass)*float(coord[2])
        totmass += float(mass)
    xcom = xcom/totmass
    ycom = ycom/totmass
    zcom = zcom/totmass
    return xcom,ycom,zcom

def inertia(elems,coords,center):
    xcom=center[0]
    ycom=center[1]
    zcom=center[2]
    Ixx = 0.
    Iyy = 0.
    Izz = 0.
    Ixy = 0.
    Ixz = 0.
    Iyz = 0.

    for index,(el,coord) in enumerate(zip(elems,coords)):
        mass = ash.modules.module_coords.atommasses[ash.modules.module_coords.elematomnumbers[el.lower()]-1]
        x = coord[0] - xcom
        y = coord[1] - ycom
        z = coord[2] - zcom

        Ixx += mass * (y**2. + z**2.)
        Iyy += mass * (x**2. + z**2.)
        Izz += mass * (x**2. + y**2.)
        Ixy += mass * x * y
        Ixz += mass * x * z
        Iyz += mass * y * z

    I_ = np.matrix([[ Ixx, -Ixy, -Ixz], [-Ixy,  Iyy, -Iyz], [-Ixz, -Iyz,  Izz]])
    I = np.linalg.eigvals(I_)
    return I
    
def calc_rotational_constants(frag, printlevel=2):
    coords=frag.coords
    elems=frag.elems
    center = get_center(elems,coords)
    rinertia = list(inertia(elems,coords,center))

    #Converting from moments of inertia in amu A^2 to rotational constants in Ghz.
    #COnversion factor from http://openmopac.net/manual/thermochemistry.html
    rot_constants=[]
    for inertval in rinertia:
        #Only calculating constant if moment of inertia value not zero
        if inertval != 0.0:
            rot_ghz=5.053791E5/(inertval*1000)
            rot_constants.append(rot_ghz)
    
    rot_constants_cm = [i*ash.constants.GHztocm for i in rot_constants]
    if printlevel >= 2:
        print("Moments of inertia (amu A^2 ):", rinertia)
        print("Rotational constants (GHz):", rot_constants)
        print("Rotational constants (cm-1):", rot_constants_cm)
        print("Note: If moment of inertia is zero then rotational constant is infinite and not printed ")

    return rot_constants_cm


def calc_model_Hessian_ORCA(fragment,model='Almloef'):

    #Run ORCA dummy job to get Almloef/Lindh/Schlegel Hessian
    orcasimple="! hf"
    extraline="!noiter opt"
    orcablocks="""
    %geom
    maxiter 1
    inhess {}
    end
""".format(model)
    orcadummycalc=ash.interfaces.interface_ORCA.ORCATheory(orcasimpleinput=orcasimple,orcablocks=orcablocks, extraline=extraline)
    ash.Singlepoint(theory=orcadummycalc, fragment=fragment, charge=fragment.charge, mult=fragment.mult)
    #Read orca-input.opt containing Hessian under hessian_approx
    hesstake=False
    j=0
    #Different from orca.hess apparently
    orcacoldim=6
    shiftpar=0
    lastchunk=False
    grabsize=False
    with open(orcadummycalc.filename+'.opt') as optfile:
        for line in optfile:
            if '$bmatrix' in line:
                hesstake=False
                continue
            if hesstake==True and len(line.split()) == 2 and grabsize==True:
                grabsize=False
                hessdim=int(line.split()[0])

                hessarray2d=np.zeros((hessdim, hessdim))
            if hesstake==True and len(line.split()) == 6:
                continue
                #Headerline
            if hesstake==True and lastchunk==True:
                if len(line.split()) == hessdim - shiftpar +1:
                    for i in range(0,hessdim - shiftpar):
                        hessarray2d[j,i+shiftpar]=line.split()[i+1]
                    j+=1
            if hesstake==True and len(line.split()) == 7:
                # Hessianline
                for i in range(0, orcacoldim):
                    hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                j += 1
                if j == hessdim:
                    shiftpar += orcacoldim
                    j = 0
                    if hessdim - shiftpar < orcacoldim:
                        lastchunk = True
            if '$hessian_approx' in line:
                hesstake = True
                grabsize = True
    #fragment.hessian=hessarray2d

    return np.array(hessarray2d)


#Function to approximate large Hessian from smaller subsystem Hessian
#fragment is the large fragment
#atomindices refer to what atoms in the large fragment the small partial Hessian was generated for
#NOTE: Capping atom option is now disabled. Best made into a separate function
    #Capping atom Hessian indices are skipped
    #if capping_atoms != None:
    #    capping_atom_hessian_indices=[3*i+j for i in capping_atoms for j in [0,1,2]]
    #else:
    #    capping_atom_hessian_indices=[]
def approximate_full_Hessian_from_smaller(fragment,hessian_small,small_atomindices,large_atomindices=None,restHessian='Almloef'):

    write_hessian(hessian_small,hessfile="smallhessian")

    #If hessatoms provided then that is the size of the actual large Hessian
    if large_atomindices is not None:
        print("small_atomindices:", small_atomindices)
        print("large_atomindices:", large_atomindices)
        hess_size=len(large_atomindices)*3
        #Initializing full Hessian using hessatoms size
        fullhessian=np.zeros((hess_size,hess_size))

        #Check that atomindices (for small) are all part of hessatom
        if all(item in large_atomindices for item in small_atomindices) is False:
            print(f"small_atomindices: {small_atomindices} are not all present in large_atomindices: {large_atomindices}")
            print("This does not make sense. Exiting")
            ashexit()
        #If large Hessian is a partial Hessian of the full systemthen we need to change small Hessian atomindices
        correct_small_atomindices=[large_atomindices.index(i) for i in small_atomindices]
        print("correct_small_atomindices:", correct_small_atomindices)
        #Create new fragment from large_atomindices
        subcoords, subelems = fragment.get_coords_for_atoms(large_atomindices)
        usedfragment = ash.Fragment(elems=subelems,coords=subcoords, printlevel=0, charge=fragment.charge, mult=fragment.mult)
        
        if check_multiplicity(subelems,usedfragment.charge,usedfragment.mult, exit=False) == False:
            print("Bad multiplicity. Using dummy")
            #Dummy charge/mult
            usedfragment.charge=0
            if isodd(usedfragment.nuccharge): 
                usedfragment.mult=2
            else:
                usedfragment.mult=1
    else:
        #Size of Hessian as big as fragment
        hess_size=fragment.numatoms*3
        #If Hessian is for full fragment then we use the input atomindices directly
        correct_small_atomindices=small_atomindices
        usedfragment=fragment
    
    print("Initializing full size Hessian of dimension:", hess_size)
    fullhessian=np.zeros((hess_size,hess_size))
    print("Initial fullhessian:", fullhessian)
    print("Size:", fullhessian.size)
    write_hessian(fullhessian,hessfile="initialfullhessian")
    #Making sure hessian_small is np array
    hessian_small = np.array(hessian_small)
    #Fill up hessian_large with model approximation from ORCA
    if restHessian == 'Almloef' or restHessian == 'Lindh' or restHessian == 'Schlegel' or restHessian == 'Swart':
        print("restHessian:", restHessian)
        fullhessian = calc_model_Hessian_ORCA(usedfragment,model=restHessian)
    #GFN1-xTB restHessian
    elif restHessian == 'xtb':
        xtb = ash.xTBTheory(xtbmethod='GFN1')
        fullhessian = xtb.Hessian(fragment=usedfragment, charge=usedfragment.charge, mult=usedfragment.mult)
    #Or with unit matrix
    elif restHessian == 'unit' or restHessian == 'identity':
        print("restHessian is unit/identity")
        fullhessian = np.identity(hess_size)
    #Keep matrix at zero
    elif restHessian==None or restHessian=='Zero':
        print("RestHessian is zero.")
    else:
        print("RestHessian is zero.")
    print("Intermediate fullhessian:", fullhessian)
    print("Size:", fullhessian.size)
    write_hessian(fullhessian,hessfile="intermedfullhessian")
    #Large Hessian indices
    athessindices = [3*i+j for i in correct_small_atomindices for j in [0,1,2]]
    #Looping over and assigning small Hessian values to large
    for s_i,i in enumerate(athessindices):
        for s_j, j in enumerate(athessindices):
            fullhessian[i,j] = hessian_small[s_i,s_j]
    print("Final fullhessian:", fullhessian)
    print("Size:", fullhessian.size)
    write_hessian(fullhessian,hessfile="Finalfullhessian")
    return fullhessian


#Change isotopes of Hessian. Read-in hessian array or hessfile
#TODO: generalize. Input isotope-pair: 'H': 1.0, 'D' : '2.0' or something
def isotope_change_Hessian(hessfile=None, hessian=None, elems=None, masses=None,
                           isotope_change="deuterium"):
    if hessfile==None and hessian==None or hessfile!=None and hessian!=None:
        print("Please provide either hessfile (ORCA-style) or hessian keyword")
        return
    if hessfile != None:
        hessfile =sys.argv[1]
        print("Reading hessfile", hessfile)
        hessian, elems, coords, masses = read_ORCA_Hessian(hessfile)
    else:
        print("Hessian provided")
        if elems == None or masses == None:
            print("Please also provide elems and masses lists")
    print("elems:", elems)
    print("masses:", masses)

    #Modify masses
    print("isotope_change:", isotope_change)
    if isotope_change == deuterium:
        modmass=2.01410177811
        masses_mod = [m if el !="H" else modmass for m,el in zip(masses,elems)]
    else:
        print("unknown isotope_change")
        ashexit()
    print("masses_mod:", masses_mod)

    #Regular mass-weighted Hessian
    vfreqs1,nmodes1,numatoms1,elems1,evectors1,atomlist1,masses1 = diagonalizeHessian(hessian, masses, elems)

    #Mass- substituted
    vfreqs2,nmodes2,numatoms2,elems2,evectors2,atomlist2,masses2 = diagonalizeHessian(hessian, masses_mod, elems)

    print("masses:", masses)
    print("masses_mod:", masses)
    ###############
    print("vfreqs1:", vfreqs1)
    print("vfreqs2:", vfreqs2)
    vfreqs1_x = [float(i) for i in vfreqs1]
    zpe_freq_list1 = [0.5*i for i in vfreqs1_x]
    ZPE_1= sum(zpe_freq_list1)/349.7550112241469 #cm-1 to kcal/mol
    ############
    vfreqs2_x = [float(i) for i in vfreqs2]
    zpe_freq_list2 = [0.5*i for i in vfreqs2_x]
    ZPE_2= sum(zpe_freq_list2)/349.7550112241469 #cm-1 to kcal/mol

    #Print ZPVE in kcal/mol
    print("ZPE_1 (kcal/mol):", ZPE_1)
    print("ZPE_2 (kcal/mol):", ZPE_2)
    
    #What else?







#####################################
# NORMALMODE COMPOSITION ANALYSIS
#####################################

#Get normal mode composition factors for mode j and atom a
def normalmodecomp(evectors,j,a):

    #square elements of mode j
    esq_j=[i ** 2 for i in evectors[j]]
    #Squared elements of atom a in mode j
    esq_ja=[]
    esq_ja.append(esq_j[a*3+0]);esq_ja.append(esq_j[a*3+1]);esq_ja.append(esq_j[a*3+2])
    return sum(esq_ja)

#Get all normal mode composition factors for atom a
def normalmodecomp_for_atom(evectors,atom):
    factors = []
    for j in range(0,len(evectors)):
        factor = normalmodecomp(evectors,j,atom)
        factors.append(factor)

    print(factors)
    return factors

# Get normal mode composition factors for all atoms for a specific mode only
def normalmodecomp_all(mode,fragment,evectors, hessatoms=None):

    if hessatoms == None:
        numatoms=fragment.numatoms
    else:
        numatoms=len(hessatoms)
    normcomplist=[]
    #vib=clean_number(vfreq[mode])
    for n in range(0, numatoms):
        normcomp=normalmodecomp(evectors,mode,n)
        normcomplist.append(normcomp)
    normcompstring=['{:.6f}'.format(x) for x in normcomplist]
    #line = "{:>3d}   {:>9.4f}        {}".format(mode, vib, '   '.join(normcompstring))
    #if silent is False:
    #    print(line)

    #Returning normcomplist, a list of atomic contributions for each atom
    return normcomplist

#Function to print frequencies and also elemental normal mode composition
def printfreqs_and_nm_elem_comps(vfreq,fragment,evectors,hessatoms=None):
    #print("printfreqs_and_nm_elem_comps:")
    numatoms=len(hessatoms)
    if numatoms == 2:
        TRmodenum=5
    else:
        TRmodenum=6
    line = "{:<6}{:<16}{:<18}".format("Mode", "Freq(cm**-1)", "Elemental composition factors")
    print(line)
    for mode in range(0,3*numatoms):
        #Get elemental normalmode comps
        normmodecompelemsdict = normalmodecomp_permode_by_elems(mode,fragment,vfreq,evectors, hessatoms=hessatoms)
        normmodecompelemsdict_list=[f'{k}: {v:.2f}' for k,v in normmodecompelemsdict.items()]
        normmodecompelemsdict_string='   '.join(normmodecompelemsdict_list)
        #print("normmodecompelemsdict_list:", normmodecompelemsdict_list)
        #print("normmodecompelemsdict:", normmodecompelemsdict)
        realpart=vfreq[mode].real
        imagpart=vfreq[mode].imag
        if mode < TRmodenum:
            line=line+" (TR mode)"
        if realpart == 0.0:
            vib=imagpart
            line = "{:>3d}   {:>9.4f}i    {}".format(mode, vib, normmodecompelemsdict_string)
        elif imagpart == 0.0:
            vib=clean_number(vfreq[mode])
            line = "{:>3d}   {:>9.4f}     {}".format(mode, vib, normmodecompelemsdict_string)
        else:
            print("vfreq[mode]:", vfreq[mode])
            print("realpart:", realpart)
            print("imagpart:", imagpart)
            print("This should not have happened")
            ashexit()

        print(line)



def normalmodecomp_permode_by_elems(mode,fragment,vfreq,evectors, silent=False, hessatoms=None):
    #print("normalmodecomp_permode_by_elems------------")
    #print("mode:", mode)
    #print(vfreq)
    #print(evectors)
    #print("hessatoms:", hessatoms)
    normcomplist = normalmodecomp_all(mode,fragment,evectors, hessatoms=hessatoms)
    #print("normcomplist:", normcomplist)
    elementnormcomplist=[]

    # Sum components together

    if hessatoms != None:
        hesselems=[fragment.elems[i] for i in hessatoms]
    else:
        hesselems=fragment.elems

    uniqelems=[]
    for i in hesselems:
        if i not in uniqelems:
            uniqelems.append(i)
    #Dict to store results
    normmodecompelemsdict={}
    for u in uniqelems:
        elcompsum=0.0
        elindices=[i for i, j in enumerate(hesselems) if j == u]
        for h in elindices:
            elcompsum=float(elcompsum+float(normcomplist[h]))
        elementnormcomplist.append(elcompsum)
        normmodecompelemsdict[u] = elcompsum
    #print(elementnormcomplist)
    #allcomps.append(elementnormcomplist)
    #print("allcomps:", allcomps)
    #elementnormcomplist=['{:.6f}'.format(x) for x in elementnormcomplist]
    #print("elementnormcomplist:", elementnormcomplist)
    #exit()
    #line = "{:>3d}   {:>9.4f}        {}".format(mode, vib, '   '.join(elementnormcomplist))
    #print(line)
    return normmodecompelemsdict

#Get atoms that contribute most to specific mode of Hessian
#Example: get atoms (atom indices) most involved in imaginary mode of transition state
#TODO: Support partial Hessian
def get_dominant_atoms_in_mode(mode,fragment=None, threshold=0.3, hessatoms=None):
    
    print_line_with_mainheader("get_dominant_atoms_in_mode")
    print("Threshold:", threshold)
    #Get hessian from fragment
    hessian=fragment.hessian

    #allatoms=list(range(0,fragment.numatoms))
    #numatoms=fragment.numatoms
    #Partial Hessian or no
    #if hessatoms != None:
    #    hessmasses = ash.modules.module_coords.get_partial_list(allatoms, hessatoms, fragment.list_of_masses)

        # Get partial matrix by deleting atoms not present in list.
    #    hesselems = ash.modules.module_coords.get_partial_list(allatoms, hessatoms, fragment.elems)
    #else:
    hessmasses=fragment.list_of_masses
    hesselems=fragment.elems

    #Diagonalize Hessian
    frequencies, nmodes, numatoms, elems, evectors, atomlist, masses = diagonalizeHessian(hessian,hessmasses,hesselems)

    #Get full list of atom contributions to mode
    normcomplist_for_mode = normalmodecomp_all(mode,fragment,evectors, hessatoms=hessatoms)

    dominant_atoms=[normcomplist_for_mode.index(i) for i in normcomplist_for_mode if i > threshold]
    print(f"Dominant atoms in mode {mode}: {dominant_atoms}\n")
    return dominant_atoms


#TODO: Rewrite and make more modular
# Function to print normal mode composition factors for all atoms, element-groups, specific atom groups or specific atoms
def printnormalmodecompositions(option,TRmodenum,vfreq,numatoms,elems,evectors,atomlist):
    # Normalmodecomposition factors for mode j and atom a
    freqs=[]
    # If one set of normal atom compositions (1 atom or 1 group)
    comps=[]
    # If multiple (case: all or elements)
    allcomps=[]
    # Change TRmodenum to 5 if diatomic molecule since linear case
    if numatoms==2:
        TRmodenum=5

    if option=="all":
        # Case: All atoms
        line = "{:>4}{:>14}      {:}".format("Mode", "Freq(cm**-1)", '       '.join(atomlist))
        print(line)
        for mode in range(0,3*numatoms):
            normcomplist=[]
            if mode < TRmodenum:
                line = "{:>3d}   {:>9.4f}".format(mode,0.000)
                print(line)
            else:
                vib=clean_number(vfreq[mode])
                freqs.append(float(vib))
                for n in range(0, numatoms):
                    normcomp=normalmodecomp(evectors,mode,n)
                    normcomplist.append(normcomp)
                allcomps.append(normcomplist)
                normcomplist=['{:.6f}'.format(x) for x in normcomplist]
                line = "{:>3d}   {:>9.4f}        {}".format(mode, vib, '   '.join(normcomplist))
                print(line)
    elif option=="elements":
        # Case: By elements
        uniqelems=[]
        for i in elems:
            if i not in uniqelems:
                uniqelems.append(i)
        line = "{:>4}{:>14}      {:45}".format("Mode", "Freq(cm**-1)", '         '.join(uniqelems))
        print(line)
        for mode in range(0,3*numatoms):
            normcomplist=[]
            if mode < TRmodenum:
                line = "{:>3d}   {:>9.4f}".format(mode,0.000)
                print(line)
            else:
                vib=clean_number(vfreq[mode])
                freqs.append(float(vib))
                for n in range(0,numatoms):
                    normcomp=normalmodecomp(evectors,mode,n)
                    normcomplist.append(normcomp)
                elementnormcomplist=[]
                # Sum components together
                for u in uniqelems:
                    elcompsum=0.0
                    elindices=[i for i, j in enumerate(elems) if j == u]
                    for h in elindices:
                        elcompsum=float(elcompsum+float(normcomplist[h]))
                    elementnormcomplist.append(elcompsum)
                # print(elementnormcomplist)
                allcomps.append(elementnormcomplist)
                elementnormcomplist=['{:.6f}'.format(x) for x in elementnormcomplist]
                line = "{:>3d}   {:>9.4f}        {}".format(mode, vib, '   '.join(elementnormcomplist))
                print(line)
    elif isint(option)==True:
        # Case: Specific atom
        atom=int(option)
        if atom > numatoms-1:
            print(bcolors.FAIL, "Atom index does not exist. Note: Numbering starts from 0", bcolors.ENDC)
            ashexit()
        line = "{:>4}{:>14}      {:45}".format("Mode", "Freq(cm**-1)", atomlist[atom])
        print(line)
        for mode in range(0,3*numatoms):
            normcomplist=[]
            if mode < TRmodenum:
                line = "{:>3d}   {:>9.4f}".format(mode,0.000)
                print(line)
            else:
                vib=clean_number(vfreq[mode])
                freqs.append(float(vib))
                for n in range(0, numatoms):
                    normcomp=normalmodecomp(evectors,mode,n)
                    normcomplist.append(normcomp)
                comps.append(normcomplist[atom])
                normcomplist=['{:.6f}'.format(x) for x in normcomplist]
                line = "{:>3d}   {:>9.4f}        {}".format(mode, vib, normcomplist[atom])
                print(line)
    elif len(option.split(",")) > 1:
        # Case: Chemical group defined as list of atoms
        selatoms = option.split(",")
        selatoms=[int(i) for i in selatoms]
        grouplist=[]
        for at in selatoms:
            if at > numatoms-1:
                print(bcolors.FAIL,"Atom index does not exist. Note: Numbering starts from 0",bcolors.ENDC)
                ashexit()
            grouplist.append(atomlist[at])
        simpgrouplist='_'.join(grouplist)
        grouplist=', '.join(grouplist)
        line = "{}   {}    {}".format("Mode", "Freq(cm**-1)", "Group("+grouplist+")")
        print(line)
        for mode in range(0,3*numatoms):
            normcomplist=[]
            if mode < TRmodenum:
                line = "{:>3d}   {:>9.4f}".format(mode,0.000)
                print(line)
            else:
                vib=clean_number(vfreq[mode])
                freqs.append(float(vib))
                for n in range(0,numatoms):
                    normcomp=normalmodecomp(evectors,mode,n)
                    normcomplist.append(normcomp)
                # normcomplist=['{:.6f}'.format(x) for x in normcomplist]
                groupnormcomplist=[]
                for q in selatoms:
                    groupnormcomplist.append(normcomplist[q])
                comps.append(sum(groupnormcomplist))
                sumgroupnormcomplist='{:.6f}'.format(sum(groupnormcomplist))
                line = "{:>3d}   {:9.4f}        {}".format(mode, vib, sumgroupnormcomplist)
                print(line)
    else:
        print("Something went wrong")

    return allcomps,comps,freqs






# Write normal mode as XYZ-trajectory (with only Hessatoms or Allatoms shown)
# Read in normalmode vectors from diagonalized mass-weighted Hessian after unweighting.
# Print out XYZ-trajectory of mode
#NOTE: Now using freqdict (what Numfreq/Anfreq returns) and grabbing all info from there
#NOTE: Store this in fragment instead??
def write_normalmode(modenumber,fragment=None, freqdict=None):
    print_line_with_mainheader("write_normalmode")
    print("Printing mode:", modenumber)
    if freqdict == None:
        print("freqdict keyword needs to be set and point to a valid Numfreq/Anfreq frequency dictionary")
        ashexit()
    nmodes=freqdict['nmodes']
    hessatoms=freqdict['hessatoms']
    if modenumber >= len(nmodes):
        print("Modenumber is larger than number of normal modes. Exiting. (Note: We count from 0.)")
        return
    else:
        # Modenumber: number mode (starting from 0)
        modechosen=nmodes[modenumber]

    # hessatoms: list of atoms involved in Hessian. All atoms unless hessatoms list provided.
    if hessatoms == None:
        hessatoms=list(range(0,len(elems)))

    #Creating dictionary of displaced atoms and chosen mode coordinates
    modedict = {}
    # Convert ndarray to list for convenience
    modechosen=modechosen.tolist()
    for fo in range(0,len(hessatoms)):
        modedict[hessatoms[fo]] = [modechosen.pop(0),modechosen.pop(0),modechosen.pop(0)]

    #Opening two Modefiles (hessatom-coordinates) and fullatom coordinates
    f = open('Mode'+str(modenumber)+'.xyz','w')
    f_full = open('Mode'+str(modenumber)+'_full.xyz','w')
    # Displacement array
    dx = np.array([0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.9,0.8,0.7,0.6,0.4,0.3,0.2,0.1,0.0])

    #Looping over displacement
    for k in range(0,len(dx)):
        hessatomindex=0
        f.write('%i\n\n' % len(hessatoms))
        f_full.write('%i\n\n' % fragment.numatoms)
        #Looping over coordinates
        for j,w in zip(range(0,fragment.numatoms),fragment.coords):
            #Writing hessatom displacement to both files
            if j in hessatoms:
                f.write("{} {:12.8f} {:12.8f} {:12.8f}  \n".format(fragment.elems[j], dx[k]*modedict[hessatomindex][0]+w[0], dx[k]*modedict[hessatomindex][1]+w[1], dx[k]*modedict[hessatomindex][2]+w[2]))
                f_full.write("{} {:12.8f} {:12.8f} {:12.8f}  \n".format(fragment.elems[j], dx[k]*modedict[hessatomindex][0]+w[0], dx[k]*modedict[hessatomindex][1]+w[1], dx[k]*modedict[hessatomindex][2]+w[2]))
            #Writing non-hessatom displacement to full file only
            else:
                f_full.write("{} {:12.8f} {:12.8f} {:12.8f}  \n".format(fragment.elems[j], w[0], w[1], +w[2]))
            hessatomindex+=1
    f.close()
    print("All done. Files Mode{}.xyz and Mode{}_full.xyz have been created!".format(modenumber,modenumber))


# Compare the similarity of normal modes by cosine similarity (normalized dot product of normal mode vectors).
#Useful for isotope-substitutions. From Hess-tool.
def comparenormalmodes(hessianA,hessianB,massesA,massesB):
    numatoms=len(massesA)
    # Massweight Hessians
    mwhessianA, massmatrixA = massweight(hessianA, massesA, numatoms)
    mwhessianB, massmatrixB = massweight(hessianB, massesB, numatoms)

    # Diagonalize mass-weighted Hessian
    evaluesA, evectorsA = la.eigh(mwhessianA)
    evaluesB, evectorsB = la.eigh(mwhessianB)
    evectorsA = np.transpose(evectorsA)
    evectorsB = np.transpose(evectorsB)

    # Calculate frequencies from eigenvalues
    vfreqA = calcfreq(evaluesA)
    vfreqB = calcfreq(evaluesB)

    print("")
    # Unweight eigenvectors to get normal modes
    nmodesA = np.dot(evectorsA, massmatrixA)
    nmodesB = np.dot(evectorsB, massmatrixB)
    line = "{:>4}".format("Mode  Freq-A(cm**-1)  Freq-B(cm**-1)    Cosine-similarity")
    print(line)
    for mode in range(0, 3 * numatoms):
        if mode < TRmodenum:
            line = "{:>3d}   {:>9.4f}       {:>9.4f}".format(mode, 0.000, 0.000)
            print(line)
        else:
            vibA = clean_number(vfreqA[mode])
            vibB = clean_number(vfreqB[mode])
            cos_sim = np.dot(nmodesA[mode], nmodesB[mode]) / (
                        np.linalg.norm(nmodesA[mode]) * np.linalg.norm(nmodesB[mode]))
            if abs(cos_sim) < 0.9:
                line = "{:>3d}   {:>9.4f}       {:>9.4f}          {:.3f} {}".format(mode, vibA, vibB, cos_sim, "<------")
            else:
                line = "{:>3d}   {:>9.4f}       {:>9.4f}          {:.3f}".format(mode, vibA, vibB, cos_sim)
            print(line)

#TODO: cleanup and use in thermochemcalc
def S_vib(freqs,T):
    vibtemps = [(f*ash.constants.c*ash.constants.h_planck_hartreeseconds)/ash.constants.R_gasconst for f in freqs]
    #Vibrational entropy via RRHO.
    S_vib=0.0
    for vibtemp in vibtemps:
        S_vib+=ash.constants.R_gasconst*(vibtemp/T)/(math.exp(vibtemp/T) - 1) - ash.constants.R_gasconst*math.log(1-math.exp(-1*vibtemp/T))
    TS_vib=S_vib*T
    #print(f"TS_vib: {TS_vib} Eh")
    #print(f"TS_vib: {TS_vib*ash.constants.hartokcal} kcal/mol")
    return TS_vib

#Write Hessian to file
def write_hessian(hessian,hessfile="Hessian"):
    with open(hessfile, 'w') as hfile:
        #RB note: Skipping header to be compatible with geometric format
        #hfile.write(str(hesslength)+' '+str(hesslength)+'\n')
        for row in hessian:
            rowline=' '.join(map(str, row))
            hfile.write(str(rowline)+'\n')
        blankline()
        print(f"Wrote Hessian to file: {hessfile}")

#Read tangent file
def read_tangent(tangentfile):
    tang=[]
    with open(tangentfile) as f:
        for line in f:
            if len(line) > 10:
                x=float(line.split()[1]);y=float(line.split()[2]);z=float(line.split()[3])
                tang.append([x,y,z])
    return np.array(tang)


#Calculate Hessian in various ways.

#Calculate Hessian (GFN1) for a fragment.
def calc_hessian_xtb(fragment=None, runmode='serial', actatoms=None, numcores=1, use_xtb_feature=True,
    charge=None, mult=None):

    print_line_with_mainheader("calc_hessian_xtb")

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, "QM", fragment, "calc_hessian_xtb")

    if actatoms != None:
        print("Creating subfragment")
        #Keep original fragment
        origfragment=copy.copy(fragment)
        #Create new fragment from actatoms
        subcoords, subelems = fragment.get_coords_for_atoms(actatoms)
        fragment = ash.Fragment(elems=subelems,coords=subcoords, printlevel=0)
    print("Will now calculate xTB Hessian")
    #Creating xtb theory object
    xtb = ash.xTBTheory(xtbmethod='GFN1', numcores=numcores)
    
    #Get Hessian from xTB directly (avoiding ASH NumFreq)
    if use_xtb_feature == True:
        print("xTB program will calculate Hessian")
        hessian = xtb.Hessian(fragment=fragment, charge=charge, mult=mult)
        write_hessian(hessian,hessfile="Hessian_from_xtb")
        hessianfile="Hessian_from_xtb"
    #ASH NumFreq. Not sure how much we will use this one
    else:
        freqdict = ash.NumFreq(theory=xtb, fragment=fragment, printlevel=0, runmode=runmode, numcores=1)
        hessianfile="Hessian_from_xtb"
        shutil.copyfile("Numfreq_dir/Hessian",hessianfile)
    #Returning name of Hessian-file
    return hessianfile