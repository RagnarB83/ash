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


def molcrys(cif_file=None, xtl_file=None, fragmentobjects=[], theory=None, numcores=None, chargemodel='',
            clusterradius=None, shortrangemodel='UFF_modH', auto_connectivity=False):

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
    if cif_file is not None:
        blankline()
        #Read CIF-file
        print("Reading CIF file:", cif_file)
        blankline()
        cell_length,cell_angles,atomlabels,elems,asymmcoords,symmops,cellunits=read_ciffile(cif_file)

        #Checking if cellunits is None or integer. If none then "_cell_formula_units" not in CIF-file and then unitcell should already be filled
        if cellunits is None:
            print("Unitcell is full (based on lack of cell_formula_units_Z line in CIF-file). Not applying symmetry operations")
            fullcellcoords=asymmcoords
        else:
            # Create system coordinates for whole cell from asymmetric unit
            print("Filling up unitcell using symmetry operations")
            fullcellcoords, elems = fill_unitcell(cell_length, cell_angles, atomlabels, elems, asymmcoords, symmops)
            numasymmunits = len(fullcellcoords) / len(asymmcoords)
            print("Number of fractional coordinates in asymmetric unit:", len(asymmcoords))
            print("Number of asymmetric units in whole cell:", int(numasymmunits))
    elif xtl_file is not None:
        blankline()
        #Read XTL-file. Assuming full-cell coordinates presently
        #TODO: Does XTL file also support asymmetric units with symm information in header?
        print("Reading XTL file:", xtl_file)
        blankline()
        cell_length,cell_angles,elems,fullcellcoords=read_xtlfile(xtl_file)
    else:
        print("Neither CIF-file or XTL-file passed to molcrys. Exiting...")
        exit(1)

    print("Cell parameters: {} {} {} {} {} {}".format(cell_length[0],cell_length[1], cell_length[2] , cell_angles[0], cell_angles[1], cell_angles[2]))

    #Calculating cell vectors.
    # Transposed cell vectors used here (otherwise nonsense for non-orthorhombic cells)
    cell_vectors=np.transpose(cellbasis(cell_angles,cell_length))
    print("cell_vectors:", cell_vectors)
    #Used by cell_extend_frag_withcenter and frag_define
    #fract_to_orthogonal uses original so it is transposed back

    #cell_vectors=cellparamtovectors(cell_length,cell_angles)
    print("Number of fractional coordinates in whole cell:", len(fullcellcoords))
    #print_coordinates(elems, np.array(fullcellcoords), title="Fractional coordinates")
    #print_coords_all(fullcellcoords,elems)
    blankline()

    #Write fractional coordinate XTL file of fullcell coordinates (for visualization in VESTA)
    write_xtl(cell_length,cell_angles,elems,fullcellcoords,"complete_unitcell.xtl")

    #Get orthogonal coordinates of cell
    orthogcoords=fract_to_orthogonal(cell_vectors,fullcellcoords)
    #Converting orthogcoords to numpy array for better performance
    orthogcoords=np.asarray(orthogcoords)
    write_xyzfile(elems,orthogcoords,"cell_orthog-original")
    #Change origin to centroid of coords
    orthogcoords=change_origin_to_centroid(orthogcoords)
    write_xyzfile(elems,orthogcoords,"cell_orthog-changedORIGIN")
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

    #Reorder coordinates of cell based on Hungarian algorithm
    #TODO: Reorder so that all atoms in fragment have same internal order.
    #TODO: Also reorder so that unit cell is easy to understand.
    #TODO: First all mainfrags, then counterfrags ??

    #Create MM cluster here already or later
    cluster_coords,cluster_elems=create_MMcluster(orthogcoords,elems,cell_vectors,clusterradius)
    print_time_rel_and_tot(currtime, origtime, modulename='create_MMcluster')
    currtime=time.time()

    #Removing partial fragments present in cluster
    #import cProfile
    #cProfile.run('remove_partial_fragments(cluster_coords,cluster_elems,sphereradius,fragmentobjects)')

    cluster_coords,cluster_elems=remove_partial_fragments(cluster_coords,cluster_elems,clusterradius,fragmentobjects, scale=chosenscale, tol=chosentol)
    print_time_rel_and_tot(currtime, origtime, modulename='remove_partial_fragments')
    currtime=time.time()
    write_xyzfile(cluster_elems,cluster_coords,"cluster_coords")

    if len(cluster_coords) == 0:
        print(BC.FAIL,"After removing all partial fragments, the Cluster fragment is empty. Something went wrong. Exiting.", BC.END)
        exit(1)


    #Create ASH fragment object
    blankline()
    print("Creating new Cluster fragment:")
    Cluster=Fragment(elems=cluster_elems, coords=cluster_coords, scale=chosenscale, tol=chosentol)

    Cluster.print_system("Cluster-first.ygg")
    print("Cluster size: ", Cluster.numatoms, "atoms")


    print_time_rel_and_tot(currtime, origtime, modulename='create Cluster fragment')
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
    Cluster.add_fragment_type_info(fragmentobjects)
    #Cluster is now almost complete, only charges missing. Print info to file
    print(Cluster.print_system("Cluster-info-nocharges.ygg"))
    print_time_rel_and_tot(currtime, origtime, modulename='Cluster fragtime')
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
        print("Using UFF forcefield with modified H-parameter (zero values for H element)")
        #print("UFF parameters:", UFFdict)
        for fragmentobject in fragmentobjects:
            #fragmentobject.Elements
            for el in fragmentobject.Elements:
                print("UFF parameter for {} :".format(el, UFF_modH_dict[el]))

        #Using UFF_ prefix before element
        atomtypelist=['UFF_'+i for i in Cluster.elems]
        atomtypelist_uniq = np.unique(atomtypelist).tolist()
        #Create ASH forcefield file by looking up UFF parameters
        with open('Cluster_forcefield.ff', 'w') as forcefile:
            forcefile.write('#UFF Lennard-Jones parameters \n')
            for atomtype in atomtypelist_uniq:
                #Getting just element-par for UFFdict lookup
                atomtype_el=atomtype.replace('UFF_','')
                forcefile.write('LennardJones_i_R0 {}  {:12.6f}   {:12.6f}\n'.format(atomtype, UFF_modH_dict[atomtype_el][0],UFF_modH_dict[atomtype_el][1]))
    elif shortrangemodel=='DDEC3' or shortrangemodel=='DDEC6':
        print("Deriving DDEC Lennard-Jones parameters")
        print("DDEC model :", shortrangemodel)

        # for fragindex,fragmentobject in enumerate(fragmentobjects):
        #    sfd=""

        # atomcharges, molmoms, voldict
        DDEC_to_LJparameters(elems, molmoms, voldict)

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

    Cluster.update_atomtypes(atomtypelist)


    #Adding Centralmainfrag to Cluster
    Cluster.add_centralfraginfo(Centralmainfrag)

    #Printing out Cluster fragment file
    Cluster.print_system('Cluster.ygg')

    #Cleanup
    #QMMM_SP_calculation.qm_theory.cleanup()


    return Cluster


