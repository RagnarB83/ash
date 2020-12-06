import numpy as np
from functions_coords import *
from functions_general import *
from functions_ORCA import *
from functions_optimization import *
from functions_molcrys import *
from ash import *
from elstructure_functions import *
import time
import shutil
origtime=time.time()
currtime=time.time()


def molcrys(cif_file=None, xtl_file=None, xyz_file=None, cell_length=None, cell_angles=None, fragmentobjects=[], theory=None, numcores=1, chargemodel='',
            clusterradius=None, shortrangemodel='UFF_modH', LJHparameters=[0.0,0.0], auto_connectivity=False, simple_supercell=False, shiftasymmunit=False, cluster_type='sphere', 
            supercell_expansion=[3,3,3]):

    banner="""
    THE
╔╦╗╔═╗╦  ╔═╗╦═╗╦ ╦╔═╗
║║║║ ║║  ║  ╠╦╝╚╦╝╚═╗
╩ ╩╚═╝╩═╝╚═╝╩╚═ ╩ ╚═╝
    MODULE
    """
    #ash header now done in settings_ash.init()
    #print_ash_header()
    print(banner)

    #TODO: After more testing auto_connectivity by default to True
    print("Auto_connectivity setting (auto_connectivity keyword) is set to: ", auto_connectivity)
    print("Do auto_connectivity=False to turn off.")
    print("Do auto_connectivity=True to turn on.")

    print("Fragment object defined:")
    for fragment in fragmentobjects:
        print("Fragment:", fragment.__dict__)

    origtime = time.time()
    currtime = time.time()
    
    
    #INPUT-OPTION: CIF, XTL, XYZ
    if cif_file is not None:
        blankline()
        #Read CIF-file
        print("Reading CIF file:", cif_file)
        blankline()
        cell_length,cell_angles,atomlabels,elems,asymmcoords,symmops,cellunits=read_ciffile(cif_file)
        
        #Shifting fractional coordinates to make sure we don't have atoms on edges
        if shiftasymmunit is True:
            print("Shifting asymmetric unit")
            print("not ready")
            exit()
            shift=[-0.3,-0.3,-0.3]
            asymmcoords=shift_fractcoords(asymmcoords,shift)

        print("asymmcoords:", asymmcoords)
        print("asymmcoords length", len(asymmcoords))
        #Checking if cellunits is None or integer. If none then "_cell_formula_units" not in CIF-file and then unitcell should already be filled
        if cellunits is None:
            print("Unitcell is full (based on lack of cell_formula_units_Z line in CIF-file). Not applying symmetry operations")
            fullcellcoords=asymmcoords
        else:
            # Create system coordinates for whole cell from asymmetric unit
            print("Filling up unitcell using symmetry operations")
            fullcellcoords, elems = fill_unitcell(cell_length, cell_angles, atomlabels, elems, asymmcoords, symmops)
            print("Full-cell coords", len(fullcellcoords))
            print("fullcellcoords", fullcellcoords)
            print("elems:", elems)
            numasymmunits = len(fullcellcoords) / len(asymmcoords)
            print("Number of fractional coordinates in asymmetric unit:", len(asymmcoords))
            print("Number of asymmetric units in whole cell:", int(numasymmunits))

        #OLD, To delete
        #cell_vectors=cellparamtovectors(cell_length,cell_angles)
        print("Number of fractional coordinates in whole cell:", len(fullcellcoords))
        #print_coordinates(elems, np.array(fullcellcoords), title="Fractional coordinates")
        #print_coords_all(fullcellcoords,elems)
        
        #Calculating cell vectors.
        # Transposed cell vectors used here (otherwise nonsense for non-orthorhombic cells)
        cell_vectors=np.transpose(cellbasis(cell_angles,cell_length))
        print("cell_vectors:", cell_vectors)
        #Get orthogonal coordinates of cell
        orthogcoords=fract_to_orthogonal(cell_vectors,fullcellcoords)      
        blankline()



    elif xtl_file is not None:
        blankline()
        #Read XTL-file. Assuming full-cell coordinates presently
        #TODO: Does XTL file also support asymmetric units with symm information in header?
        print("Reading XTL file:", xtl_file)
        blankline()
        cell_length,cell_angles,elems,fullcellcoords=read_xtlfile(xtl_file)
        
        #cell_vectors=cellparamtovectors(cell_length,cell_angles)

        print("Number of fractional coordinates in whole cell:", len(fullcellcoords))

        #Calculating cell vectors.
        # Transposed cell vectors used here (otherwise nonsense for non-orthorhombic cells)
        cell_vectors=np.transpose(cellbasis(cell_angles,cell_length))
        print("cell_vectors:", cell_vectors)
        #Get orthogonal coordinates of cell
        orthogcoords=fract_to_orthogonal(cell_vectors,fullcellcoords)      
        blankline()

        
    elif xyz_file is not None:
        print("WARNING. This option is not well tested. XYZ-file must contain all coordinates of cell.")
        blankline()
        #Read XYZ-file. Assuming file contains full-cell real-space coordinates 
        print("Reading XYZ file (assuming real-space coordinates in Angstrom):", xyz_file)
        elems,orthogcoords=read_xyzfile(xyz_file)
        print("Read {} atoms from XYZ-files".format(len(orthogcoords)))
        
        #Need to read cell_lengths and cell_angles also
        if cell_length is None or cell_angles is None:
            print("cell_length/cell_angles is not defined. This is needed for XYZ-file option.")
            exit()
        blankline()
        
        #Calculating cell vectors.
        # Transposed cell vectors used here (otherwise nonsense for non-orthorhombic cells)
        cell_vectors=np.transpose(cellbasis(cell_angles,cell_length))
        print("cell_vectors:", cell_vectors)
        
    else:
        print("Neither CIF-file, XTL-file or XYZ-file passed to molcrys. Exiting...")
        exit(1)



    #Write fractional coordinate XTL file of fullcell coordinates (for visualization in VESTA)
    write_xtl(cell_length,cell_angles,elems,fullcellcoords,"complete_unitcell.xtl")



    print("Cell parameters: {} {} {} {} {} {}".format(cell_length[0],cell_length[1], cell_length[2] , cell_angles[0], cell_angles[1], cell_angles[2]))

    
    #Used by cell_extend_frag_withcenter and frag_define
    #fract_to_orthogonal uses original so it is transposed back
    
    #Converting orthogcoords to numpy array for better performance
    orthogcoords=np.asarray(orthogcoords)
    print("orthogcoords:", orthogcoords)

    write_xyzfile(elems,orthogcoords,"cell_orthog-original")
    #Change origin to centroid of coords
    orthogcoords=change_origin_to_centroid(orthogcoords)
    write_xyzfile(elems,orthogcoords,"cell_orthog-changedORIGIN")
    
    
    #Make simpler super-cell for cases where molecule is not in cell
    #TODO: Not sure if need this. 
    #if simple_supercell is True:
    #    print("Simple supercell is True")
    #    #exit()return extended, new_elems
    #    #cellextpars=[2,2,2]
    #    print("before extension len coords", len(orthogcoords))
    #    print(orthogcoords)
    #    print("")
    #    supercell_coords, supercell_elems = cell_extend_frag_withcenter(cell_vectors, orthogcoords,elems)
    #    print("supercell_coords ({}):".format(len(supercell_coords),supercell_coords))
    #    print("supercell_elems ({}) :".format(len(supercell_elems),supercell_elems))
    #    #supercell_orthogcoords=fract_to_orthogonal([cell_vectors[0],cell_vectors[1],cell_vectors[2]*1],supercell_coords)
    #    write_xyzfile(supercell_elems,supercell_coords,"supercell_coords")
    #    
    #    #DEFINING supercell as cell from now on 
    #    orthogcoords=supercell_coords
    #    elems=supercell_elems





    print("")
    #print_coordinates(elems, orthogcoords, title="Orthogonal coordinates")
    #print_coords_all(orthogcoords,elems)

    #Define fragments of unitcell. Updates mainfrag, counterfrag1 etc. object information

    #Loop through different tol settings
    if auto_connectivity == True:
        #Sticking to 1.0 here until we find case where modification is needed.
        #If so we can implement double-loop over both scale and tol parameters
        chosenscale=1.0
        test_tolerances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        print(BC.WARNING,"Automatic connectivity determination:", BC.END)
        print("Using Scale : ", chosenscale)
        print("Will loop through tolerances:", test_tolerances)
        for chosentol in test_tolerances:
            print("Current Scale", chosenscale)
            print("Current Tol: ", chosentol)
            checkflag = frag_define(orthogcoords,elems,cell_vectors,fragments=fragmentobjects, cell_angles=cell_angles, cell_length=cell_length,
                        scale=chosenscale, tol=chosentol)
            if checkflag == 0:

                print(BC.OKMAGENTA, "A miracle occurred! Fragment assignment succeeded!", BC.END)
                print("Final connectivity parameters are: Scale: {} and Tol: {}".format(chosenscale, chosentol))
                print("Setting global scale and tol parameters")
                #Should be safest option I think. To be revisited
                settings_ash.tol=chosenscale
                settings_ash.tol=chosentol

                print("")
                break
            else:
                print(BC.FAIL,"Fragment assignment failed.", BC.WARNING,"Trying next Tol parameter.", BC.END)
                #Setting found fragmentlists as empty. Otherwise trouble.
                for fragobject in fragmentobjects:
                    fragobject.fraglist=[]
                    
        # If all test_tolerances failed.
        if checkflag == 1:
            print("Automatic connectivity failed. Make sure that the fragment definitions are correct, "
                  "that the cell is not missing atoms or that it contains extra atoms")
            exit(1)
    else:
        chosenscale=settings_ash.scale
        chosentol=settings_ash.tol
        print("Determining connectivity using Scale: {} and Tol: {}".format(chosenscale,chosentol))

        #Using the global ASH settings (may have been modified by user)
        checkflag = frag_define(orthogcoords,elems,cell_vectors,fragments=fragmentobjects, cell_angles=cell_angles, cell_length=cell_length,
                    scale=chosenscale, tol=chosentol)
        if checkflag == 0:
            print(BC.OKMAGENTA, "A miracle occurred! Fragment assignment succeeded!", BC.END)
        else:
            exit(1)


    print_time_rel_and_tot(currtime, origtime, modulename='frag_define')
    currtime=time.time()



    # CREATE SPHERE OR SUPERCELL!!
    #Reorder coordinates of cell based on Hungarian algorithm
    #TODO: Reorder so that all atoms in fragment have same internal order.
    #TODO: Also reorder so that unit cell is easy to understand.
    #TODO: First all mainfrags, then counterfrags ??
    if cluster_type == 'sphere':
        #Create MM cluster here already or later
        cluster_coords,cluster_elems=create_MMcluster(orthogcoords,elems,cell_vectors,clusterradius)
        print_time_rel_and_tot(currtime, origtime, modulename='create_MMcluster')
        currtime=time.time()

        cluster_coords,cluster_elems=remove_partial_fragments(cluster_coords,cluster_elems,clusterradius,fragmentobjects, scale=chosenscale, tol=chosentol)
        print_time_rel_and_tot(currtime, origtime, modulename='remove_partial_fragments')
        currtime=time.time()
        write_xyzfile(cluster_elems,cluster_coords,"cluster_coords")

        if len(cluster_coords) == 0:
            print(BC.FAIL,"After removing all partial fragments, the Cluster fragment is empty. Something went wrong. Exiting.", BC.END)
            exit(1)
    elif cluster_type == 'supercell':
        #Super-cell instead of sphere. Advantage: If unitcell is well-behaved then we will have no dipole problem when we cut a sphere.
        #Disadvantage, we could have partial fragment at boundary. 
        #TODO: Check for partial fragments at boundary and clean up?? Adapt remove_partial_fragments ???
        print(BC.WARNING,"Warning. cluster_type = supercell is untested ",BC.END)
        cluster_coords,cluster_elems = cell_extend_frag(cell_vectors, orthogcoords,elems,supercell_expansion)
        
        #TODO: Clean up partial fragments at boundary
        
    else:
        print("unknown cluster_type")
        exit()




    #Create ASH fragment object
    blankline()
    print("Creating new Cluster fragment:")
    Cluster=Fragment(elems=cluster_elems, coords=cluster_coords, scale=chosenscale, tol=chosentol)
    
    
    print_time_rel_and_tot(currtime, origtime, modulename='create Cluster fragment')
    currtime=time.time()
    Cluster.print_system("Cluster-first.ygg")
    Cluster.write_xyzfile(xyzfilename="Cluster-first.xyz")
    print("Cluster size: ", Cluster.numatoms, "atoms")
    print_time_rel_and_tot(currtime, origtime, modulename='print Cluster system')
    currtime=time.time()
    # Going through found frags and identify mainfrags and counterfrags
    for frag in Cluster.connectivity:
        el_list = [cluster_elems[i] for i in frag]
        ncharge = nucchargelist(el_list)
        for fragmentobject in fragmentobjects:
            if ncharge == fragmentobject.Nuccharge:
                fragmentobject.add_clusterfraglist(frag)

    printdebug(fragmentobjects[0].clusterfraglist)
    print_time_rel_and_tot(currtime, origtime, modulename='fragment identification')
    currtime=time.time()
    #TODO: Reorder cluster with reflections also

    #Reorder fraglists in each fragmenttype via Hungarian algorithm.
    # Ordered fraglists can then easily be used in pointchargeupdating
    for fragmentobject in fragmentobjects:
        print("Reordering fragment object: ", fragmentobject.Name)
        reordercluster(Cluster,fragmentobject)
        printdebug(fragmentobject.clusterfraglist)
        fragmentobject.print_infofile(str(fragmentobject.Name)+'.info')
    print_time_rel_and_tot(currtime, origtime, modulename='reorder fraglists')
    currtime=time.time()
    #TODO: after removing partial fragments and getting connectivity etc. Would be good to make MM cluster neutral

    #Add fragmentobject-info to Cluster fragment
    #Old slow code:
    #Cluster.old_add_fragment_type_info(fragmentobjects)
    Cluster.add_fragment_type_info(fragmentobjects)

    print_time_rel_and_tot(currtime, origtime, modulename='Cluster fragment type info')
    currtime=time.time()
    #Cluster is now almost complete, only charges missing. Print info to file
    print(Cluster.print_system("Cluster-info-nocharges.ygg"))
    print_time_rel_and_tot(currtime, origtime, modulename='Cluster print system')
    currtime=time.time()
    # Create dirs to keep track of various files before QM calculations begin
    try:
        os.mkdir('SPloop-files')
    except:
        shutil.rmtree('SPloop-files')
        os.mkdir('SPloop-files')


    # Calculate atom charges for each gas fragment. Updates atomcharges list inside Cluster fragment
    if theory.__class__.__name__ == "ORCATheory":


        if theory.brokensym == True:

            if chargemodel =="IAO":
                print("Note IAO charges on broken-symmetry solution are probably not sensible")

            gasfragcalc_ORCA(fragmentobjects, Cluster, chargemodel, theory.orcadir, theory.orcasimpleinput,
                             theory.orcablocks, numcores, brokensym=True, HSmult=theory.HSmult,
                             atomstoflip=theory.atomstoflip)
        else:
            gasfragcalc_ORCA(fragmentobjects, Cluster, chargemodel, theory.orcadir, theory.orcasimpleinput,
                             theory.orcablocks, numcores)


    elif theory.__class__.__name__ == "xTBTheory":
        pass
        gasfragcalc_xTB(fragmentobjects,Cluster,chargemodel,theory.xtbdir,theory.xtbmethod,numcores)
    else:
        print("Unsupported theory for charge-calculations in MolCrys. Options are: ORCATheory or xTBTheory")
        exit(1)
    print_time_rel_and_tot(currtime, origtime, modulename="gasfragcalc")
    currtime=time.time()
    print("Atom charge assignments in Cluster done!")
    blankline()
    #print("Cluster:", Cluster.__dict__)

    #Cluster is now complete. Print info to file
    Cluster.print_system("Cluster-info_afterGas.ygg")
    for fragmentobject in fragmentobjects:
        fragmentobject.print_infofile(str(fragmentobject.Name)+'-info_afterGas.ygg')

    sum_atomcharges_cluster=sum(Cluster.atomcharges)
    print("sum_atomcharges_cluster:", sum_atomcharges_cluster)

    with open("system-atomcharges", 'w') as acharges:
        for charge in Cluster.atomcharges:
            acharges.write("{} ".format(charge))



    #SC-QM/MM PC loop of mainfrag for cluster
    #Hard-coded (for now) maxiterations and charge-convergence thresholds
    SPLoopMaxIter=10
    RMSD_SP_threshold=0.001
    blankline()
    #TODO: Make ORCA GBW-read more transparent during SP iterations or in general

    # Charge-model info to add to inputfile
    chargemodelline = chargemodel_select(chargemodel)
    # Creating QMtheory object without fragment information.
    # fragmentobjects[0] is always mainfrag
    if theory.__class__.__name__ == "ORCATheory":

        if theory.brokensym == True:
            #Adding UKS keyword if not already present for case AF-coupled BS-singlet to prevent RKS/RHF.
            if 'UKS' not in theory.orcasimpleinput:
                theory.orcasimpleinput=theory.orcasimpleinput+' UKS'
        QMtheory = ORCATheory(orcadir=theory.orcadir, charge=fragmentobjects[0].Charge, mult=fragmentobjects[0].Mult,
                              orcasimpleinput=theory.orcasimpleinput,
                              orcablocks=theory.orcablocks, extraline=chargemodelline)
        #COPY LAST mainfrag orbitals here: called lastorbitals.gbw from gasfragcalc (mainfrag)
        #Necessary to avoid broken-sym SpinFlip but should be good in general
        shutil.copyfile('lastorbitals.gbw', 'orca-input.gbw')

    elif theory.__class__.__name__ == "xTBTheory":
        QMtheory = xTBTheory(xtbdir=theory.xtbdir, charge=fragmentobjects[0].Charge, mult=fragmentobjects[0].Mult, xtbmethod=theory.xtbmethod)

    print("QMtheory:", QMtheory)
    print(QMtheory.__dict__)

    # Defining QM region. Should be the mainfrag at approx origin
    Centralmainfrag = fragmentobjects[0].clusterfraglist[0]
    print("Centralmainfrag:", Centralmainfrag)

    #Writing Centralmainfrag to disk as Centralmainfrag
    with open ("Centralmainfrag", 'w') as file:
        for i in Centralmainfrag:
            file.write(str(i)+' ')

    #Writing Centralmainfrag to disk as qmatoms
    with open ("qmatoms", 'w') as file:
        for i in Centralmainfrag:
            file.write(str(i)+' ')



    blankline()

    print_time_rel_and_tot(currtime, origtime, modulename="stuff before SP-loop")
    currtime=time.time()
    #SP-LOOP FOR MAINFRAG
    for SPLoopNum in range(0,SPLoopMaxIter):
        blankline()
        print("This is Charge-Iteration Loop number", SPLoopNum)
        atomcharges=[]
        print_coords_for_atoms(Cluster.coords,Cluster.elems,Centralmainfrag)
        print("")
        # Run ORCA QM/MM calculation with charge-model info
        QMMM_SP_calculation = QMMMTheory(fragment=Cluster, qm_theory=QMtheory, qmatoms=Centralmainfrag,
                                             charges=Cluster.atomcharges, embedding='Elstat')
        QMMM_SP_calculation.run(nprocs=numcores)


        #Grab atomic charges for fragment.
        if theory.__class__.__name__ == "ORCATheory":

            if chargemodel == 'DDEC3' or chargemodel == 'DDEC6':
                print("Need to think more about what happens here for DDEC")
                print("Molecule should be polarized by environment but atoms should not.")
                # Calling DDEC_calc (calls chargemol)
                #Here providing only QMTheory object to DDEC_calc as this will be used for atomic calculations (not molecule)
                elemlist_mainfrag = [Cluster.elems[i] for i in Centralmainfrag]
                print("elemlist_mainfrag: ", elemlist_mainfrag)
                atomcharges, molmoms, voldict = DDEC_calc(elems=elemlist_mainfrag, theory=QMtheory,
                                                          ncores=numcores, DDECmodel=chargemodel,
                                                          molecule_spinmult=fragmentobjects[0].Mult,
                                                          calcdir="DDEC_fragment_SPloop" + str(SPLoopNum), gbwfile="orca-input.gbw")
            else:
                atomcharges = grabatomcharges_ORCA(chargemodel, QMtheory.inputfilename + '.out')
            # Keep backup of ORCA outputfile in dir SPloop-files
            shutil.copyfile('orca-input.out', './SPloop-files/mainfrag-SPloop' + str(SPLoopNum) + '.out')
            shutil.copyfile('orca-input.pc', './SPloop-files/mainfrag-SPloop' + str(SPLoopNum) + '.pc')
            blankline()
        elif theory.__class__.__name__ == "xTBTheory":
            atomcharges = grabatomcharges_xTB()

        print("Elements:", [Cluster.elems[i] for i in Centralmainfrag ])
        print("atomcharges (SPloop {}) : {}".format(SPLoopNum,atomcharges))


        #Adding mainfrag charges to mainfrag--object
        fragmentobjects[0].add_charges(atomcharges)

        # Assign pointcharges to each atom of MM cluster.
        #pointchargeupdate calls: Cluster.update_atomcharges(chargelist)
        pointchargeupdate(Cluster, fragmentobjects[0], atomcharges)
        Cluster.print_system("Cluster-info_afterSP{}.ygg".format(SPLoopNum))
        fragmentobjects[0].print_infofile('mainfrag-info_afterSP{}.ygg'.format(SPLoopNum))
        blankline()
        #print("Current charges:", fragmentobjects[0].all_atomcharges[-1])
        #print("Previous charges:", fragmentobjects[0].all_atomcharges[-2])
        RMSD_charges=rmsd_list(fragmentobjects[0].all_atomcharges[-1],fragmentobjects[0].all_atomcharges[-2])
        print(BC.OKBLUE,"RMSD of charges: {:6.3f} in SP iteration {:6}:".format(RMSD_charges, SPLoopNum),BC.END)
        if RMSD_charges < RMSD_SP_threshold:
            print("RMSD: {} is less than threshold: {}".format(RMSD_charges,RMSD_SP_threshold))
            print(BC.OKMAGENTA,"Charges converged in SP iteration {}! SP LOOP over!".format(SPLoopNum),BC.END)
            break
        print(BC.WARNING,"Not converged in iteration {}. Continuing SP loop".format(SPLoopNum),BC.END)

    print(BC.OKMAGENTA,"Molcrys Charge-Iteration done!",BC.END)
    print("")
    print_time_rel_and_tot(currtime, origtime, modulename="SP iteration done")
    currtime=time.time()
    
    #Now that charges are converged (for mainfrag and counterfrags ???).
    #Now derive LJ parameters ?? Important for DDEC-LJ derivation
    #Defining atomtypes in Cluster fragment for LJ interaction
    if shortrangemodel=='UFF':
        print("Using UFF forcefield for all elements")
        for fragmentobject in fragmentobjects:
            #fragmentobject.Elements
            for el in fragmentobject.Elements:
                print("UFF parameter for {} :".format(el, UFFdict[el]))

        #Using UFF_ prefix before element
        atomtypelist=['UFF_'+i for i in Cluster.elems]
        atomtypelist_uniq = np.unique(atomtypelist).tolist()
        #Create ASH forcefield file by looking up UFF parameters
        with open('Cluster_forcefield.ff', 'w') as forcefile:
            forcefile.write('#UFF Lennard-Jones parameters \n')
            for atomtype in atomtypelist_uniq:
                #Getting just element-par for UFFdict lookup
                atomtype_el=atomtype.replace('UFF_','')
                forcefile.write('LennardJones_i_R0 {}  {:12.6f}   {:12.6f}\n'.format(atomtype, UFFdict[atomtype_el][0],UFFdict[atomtype_el][1]))
    #Modified UFF forcefield with 0 parameter on H atom (avoids repulsion)
    elif shortrangemodel=='UFF_modH':
        print("Using UFF forcefield with modified H-parameter")
        print("H parameters :", LJHparameters)
        print("")
        UFFdict_Hzero=copy.deepcopy(UFFdict)
        UFFdict_Hzero['H'] = [LJHparameters[0], LJHparameters[1]]
        
        #print("UFF parameters:", UFFdict)
        for fragmentobject in fragmentobjects:
            #fragmentobject.Elements
            for el in fragmentobject.Elements:
                print("UFF parameter for {} :".format(el, UFFdict_Hzero[el]))

        #Using UFF_ prefix before element
        atomtypelist=['UFF_'+i for i in Cluster.elems]
        atomtypelist_uniq = np.unique(atomtypelist).tolist()
        #Create ASH forcefield file by looking up UFF parameters
        with open('Cluster_forcefield.ff', 'w') as forcefile:
            forcefile.write('#UFF Lennard-Jones parameters \n')
            for atomtype in atomtypelist_uniq:
                #Getting just element-par for UFFdict lookup
                atomtype_el=atomtype.replace('UFF_','')
                forcefile.write('LennardJones_i_R0 {}  {:12.6f}   {:12.6f}\n'.format(atomtype, UFFdict_Hzero[atomtype_el][0],UFFdict_Hzero[atomtype_el][1]))
    elif shortrangemodel=='DDEC3' or shortrangemodel=='DDEC6':
        print("Deriving DDEC Lennard-Jones parameters")
        print("DDEC model :", shortrangemodel)

        #Getting R0 and epsilon for mainfrag
        #fragmentobjects[0].r0list, fragmentobjects[0].epsilonlist = DDEC_to_LJparameters(elems, molmoms, voldict)

        #Getting R0 and epsilon for counterfrags
        for fragmentobject in fragmentobjects:
            print("fragmentobject Atoms:", fragmentobject.Atoms)
            print("fragmentobject molmoms:", fragmentobject.molmoms)
            print("fragmentobject voldict:", fragmentobject.voldict)
            
            #Getting R0 and epsilon for mainfrag
            fragmentobject.r0list, fragmentobject.epsilonlist = DDEC_to_LJparameters(fragmentobject.Atoms, fragmentobject.molmoms, fragmentobject.voldict)

            print("fragmentobject dict", fragmentobject.__dict__)
            exit()

        #Create atomtypelist to be added to Cluster object
        atomtypelist = ""

        print("Using {}-derived forcefield for all elements".format(shortrangemodel))
        #for fragmentobject in fragmentobjects:
        #    #fragmentobject.Elements
        #    for el in fragmentobject.Elements:
        #        print("UFF parameter for {} :".format(el, UFFdict[el]))


        
        
        #atomtypelist=['UFF_'+i for i in Cluster.elems]
        #atomtypelist_uniq = np.unique(atomtypelist).tolist()
        #Create ASH forcefield file by looking up UFF parameters
        with open('Cluster_forcefield.ff', 'w') as forcefile:
            forcefile.write('#{} Lennard-Jones parameters \n'.format(shortrangemodel))
            for atomtype in atomtypelist_uniq:
                #Getting just element-par for UFFdict lookup
                atomtype_el=atomtype.replace('UFF_','')
                forcefile.write('LennardJones_i_R0 {}  {:12.6f}   {:12.6f}\n'.format(atomtype, UFFdict[atomtype_el][0],UFFdict[atomtype_el][1]))













    elif shortrangemodel=='manual':
        print("shortrangemodel option: manual")
        print("Using atomtypes for Cluster: MAN_X  where X is an element, e.g. MAN_O, MAN_C, MAN_H")
        print("Will assume presence of ASH forcefield file called: Cluster_forcefield.ff")
        print("Should contain Lennard-Jones entries for atomtypes MAN_X.")
        print("File needs to be copied to scratch for geometry optimization job.")
        #Using MAN prefix before element
        atomtypelist=['MAN_'+i for i in Cluster.elems]

    else:
        print("Undefined shortrangemodel")
        exit()
    print_time_rel_and_tot(currtime, origtime, modulename="LJ stuff done")
    currtime=time.time()
    Cluster.update_atomtypes(atomtypelist)
    print_time_rel_and_tot(currtime, origtime, modulename="update atomtypes")
    currtime=time.time()

    #Adding Centralmainfrag to Cluster
    Cluster.add_centralfraginfo(Centralmainfrag)

    #Printing out Cluster fragment file
    Cluster.print_system('Cluster.ygg')

    #Cleanup
    #QMMM_SP_calculation.qm_theory.cleanup()
    print_time_rel_and_tot(currtime, origtime, modulename="final stuff")
    currtime=time.time()

    return Cluster


