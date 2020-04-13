import numpy as np
from functions_coords import *
from functions_general import *
from functions_ORCA import *
from functions_optimization import *
import settings_molcrys
from functions_molcrys import *
from yggdrasill import *
import time
import shutil
origtime=time.time()
currtime=time.time()


#TODO: Introduce jobtype. Either SPEmbeddingloop or Optloop.
#Or maybe have molcrys only be SPEmbedding and in inputfile we do molcrys job and then opt-job.
#Could do SPembedding again after Opt

def molcrys(cif_file=None, xtl_file=None, fragmentobjects=[], theory=None, numcores=None, chargemodel='',
            clusterradius=None, shortrangemodel='UFF'):

    banner="""
╔╦╗╔═╗╦  ╔═╗╦═╗╦ ╦╔═╗
║║║║ ║║  ║  ╠╦╝╚╦╝╚═╗
╩ ╩╚═╝╩═╝╚═╝╩╚═ ╩ ╚═╝
    
    """
    print(banner)
    #Here assuming theory can only be ORCA for now
    orcadir=theory.orcadir
    orcablocks=theory.orcablocks
    orcasimpleinput=theory.orcasimpleinput

    print("Fragments defined:")
    for fragment in fragmentobjects:
        print("Fragment:", fragment.__dict__)

    origtime = time.time()
    currtime = time.time()

    if cif_file is not None:
        blankline()
        #Read CIF-file
        print("Reading CIF file:", cif_file)
        blankline()
        cell_length,cell_angles,atomlabels,elems,asymmcoords,symmops=read_ciffile(cif_file)

        # Create system coordinates for whole cell from asymmetric unit
        print("Filling up unitcell using symmetry operations")
        fullcellcoords, elems = fill_unitcell(cell_length, cell_angles, atomlabels, elems, asymmcoords, symmops)
        numasymmunits = len(fullcellcoords) / len(asymmcoords)

        print("Number of fractional coordinates in asymmetric unit:", len(asymmcoords))
        print("Number of asymmetric units in whole cell:", numasymmunits)
    elif xtl_file is not None:
        blankline()
        #Read XTL-file. Assuming full-cell coordinates present.
        #TODO: Does XTL file also support asymmetric units with symm information in header?
        print("Reading XTL file:", xtl_file)
        blankline()
        cell_length,cell_angles,elems,fullcellcoords=read_xtlfile(xtl_file)
    else:
        print("Neither CIF-file or XTL-file passed to molcrys. Exiting...")
        exit()

    print("Cell parameters: {} {} {} {} {} {}".format(cell_length[0],cell_length[1], cell_length[2] , cell_angles[0], cell_angles[1], cell_angles[2]))
    #Calling new cell vectors function instead of old
    cell_vectors=cellbasis(cell_angles,cell_length)
    print("cell_vectors:", cell_vectors)

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

    #print_coordinates(elems, orthogcoords, title="Orthogonal coordinates")
    #print_coords_all(orthogcoords,elems)

    #Define fragments of unitcell. Updates mainfrag, counterfrag1 etc. object information
    frag_define(orthogcoords,elems,cell_vectors,fragments=fragmentobjects, cell_angles=cell_angles, cell_length=cell_length)
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

    cluster_coords,cluster_elems=remove_partial_fragments(cluster_coords,cluster_elems,clusterradius,fragmentobjects)
    print_time_rel_and_tot(currtime, origtime, modulename='remove_partial_fragments')
    currtime=time.time()
    write_xyzfile(cluster_elems,cluster_coords,"cluster_coords")

    #Create Yggdrasill fragment object
    blankline()
    print("Creating new Cluster fragment:")
    Cluster=Fragment(elems=cluster_elems, coords=cluster_coords)
    Cluster.calc_connectivity(scale=settings_molcrys.scale, tol=settings_molcrys.tol)
    print_time_rel_and_tot(currtime, origtime, modulename='Cluster.calc_connectivity')
    currtime=time.time()

    # Going through found frags and identify mainfrags and counterfrags
    for frag in Cluster.connectivity:
        el_list = [cluster_elems[i] for i in frag]
        ncharge = nucchargelist(el_list)
        for fragmentobject in fragmentobjects:
            if ncharge == fragmentobject.Nuccharge:
                fragmentobject.add_clusterfraglist(frag)

    printdebug(fragmentobjects[0].clusterfraglist)
    #TODO: Reorder cluster with reflections also

    #Reorder fraglists in each fragmenttype via Hungarian algorithm.
    # Ordered fraglists can then easily be used in pointchargeupdating
    for fragmentobject in fragmentobjects:
        reordercluster(Cluster,fragmentobject)
        printdebug(fragmentobject.clusterfraglist)
        fragmentobject.print_infofile(str(fragmentobject.Name)+'.info')
    #TODO: after removing partial fragments and getting connectivity etc. Would be good to make MM cluster neutral

    #Add fragmentobject-info to Cluster fragment
    Cluster.add_fragment_type_info(fragmentobjects)
    #Cluster is now almost complete, only charges missing. Print info to file
    print(Cluster.print_system("Cluster-info-nocharges.ygg"))

    # Create dirs to keep track of various files before ORCA calculations begin
    try:
        os.mkdir('SPloop-files')
    except:
        shutil.rmtree('SPloop-files')
        os.mkdir('SPloop-files')


    # Calculate atom charges for each gas fragment. Updates atomcharges list inside Cluster fragment
    gasfragcalc(fragmentobjects,Cluster,chargemodel,orcadir,orcasimpleinput,orcablocks,numcores)

    print_time_rel_and_tot_color(currtime, origtime, modulename='gasfragcalc')
    currtime=time.time()
    print("Atom charge assignments in Cluster done!")
    blankline()
    print("Cluster:", Cluster.__dict__)

    #Cluster is now complete. Print info to file
    Cluster.print_system("Cluster-info_afterGas.ygg")
    for fragmentobject in fragmentobjects:
        fragmentobject.print_infofile(str(fragmentobject.Name)+'-info_afterGas.ygg')

    sum_atomcharges_cluster=sum(Cluster.atomcharges)
    print("sum_atomcharges_cluster:", sum_atomcharges_cluster)

    with open("system-atomcharges", 'w') as acharges:
        for charge in Cluster.atomcharges:
            acharges.write("{} ".format(charge))

    #Defining atomtypes in Cluster fragment for LJ interaction
    if shortrangemodel=='UFF':
        print("Using UFF forcefield for all elements")
        #Using UFF_ prefix before element
        atomtypelist=['UFF_'+i for i in Cluster.elems]
        atomtypelist_uniq = np.unique(atomtypelist).tolist()
        #Create Yggdrasill forcefield file by looking up UFF parameters
        with open('Cluster_forcefield.ff', 'w') as forcefile:
            forcefile.write('#UFF Lennard-Jones parameters \n')
            for atomtype in atomtypelist_uniq:
                #Getting just element-par for UFFdict lookup
                atomtype_el=atomtype.replace('UFF_','')
                forcefile.write('LennardJones_i_R0 {}  {:12.6f}   {:12.6f}\n'.format(atomtype, UFFdict[atomtype_el][0],UFFdict[atomtype_el][1]))
    #Modified UFF forcefield with 0 parameter on H atom (avoids repulsion)
    elif shortrangemodel=='UFF_modH':
        print("Using UFF forcefield with modified H-parameter (0 values for H)")
        #Using UFF_ prefix before element
        atomtypelist=['UFF_'+i for i in Cluster.elems]
        atomtypelist_uniq = np.unique(atomtypelist).tolist()
        #Create Yggdrasill forcefield file by looking up UFF parameters
        with open('Cluster_forcefield.ff', 'w') as forcefile:
            forcefile.write('#UFF Lennard-Jones parameters \n')
            for atomtype in atomtypelist_uniq:
                #Getting just element-par for UFFdict lookup
                atomtype_el=atomtype.replace('UFF_','')
                forcefile.write('LennardJones_i_R0 {}  {:12.6f}   {:12.6f}\n'.format(atomtype, UFF_modH_dict[atomtype_el][0],UFF_modH_dict[atomtype_el][1]))
    else:
        print("Undefined shortrangemodel")
        exit()

    Cluster.update_atomtypes(atomtypelist)


    #SC-QM/MM PC loop of mainfrag for cluster
    #Hard-coded (for now) maxiterations and charge-convergence thresholds
    SPLoopMaxIter=10
    RMSD_SP_threshold=0.001
    blankline()
    #TODO: Make ORCA GBW-read more transparent during SP iterations or in general

    # Charge-model info to add to inputfile
    chargemodelline = chargemodel_select(chargemodel)
    # Creating ORCA theory object without fragment information.
    # fragmentobjects[0] is always mainfrag
    ORCAQMtheory = ORCATheory(orcadir=orcadir, charge=fragmentobjects[0].Charge, mult=fragmentobjects[0].Mult,
                              orcasimpleinput=orcasimpleinput,
                              orcablocks=orcablocks, extraline=chargemodelline)
    print("ORCAQMtheory:", ORCAQMtheory)
    print(ORCAQMtheory.__dict__)

    # Defining QM region. Should be the mainfrag at approx origin
    Centralmainfrag = fragmentobjects[0].clusterfraglist[0]
    print("Centralmainfrag:", Centralmainfrag)



    #SP-LOOP FOR MAINFRAG
    for SPLoopNum in range(0,SPLoopMaxIter):
        print("This is Charge-Iteration Loop number", SPLoopNum)
        atomcharges=[]
        print_coords_for_atoms(Cluster.coords,Cluster.elems,Centralmainfrag)

        # Run ORCA QM/MM calculation with charge-model info
        QMMM_SP_ORCAcalculation = QMMMTheory(fragment=Cluster, qm_theory=ORCAQMtheory, qmatoms=Centralmainfrag,
                                             atomcharges=Cluster.atomcharges, embedding='Elstat')
        QMMM_SP_ORCAcalculation.run(nprocs=numcores)


        #Grab atomic charges for fragment.

        atomcharges = grabatomcharges(chargemodel, ORCAQMtheory.inputfilename + '.out')
        print("Elements:", [Cluster.elems[i] for i in Centralmainfrag ])
        print("atomcharges (SPloop {}) : {}".format(SPLoopNum,atomcharges))

        # Keep backup of ORCA outputfile in dir SPloop-files
        shutil.copyfile('orca-input.out', './SPloop-files/mainfrag-SPloop'+str(SPLoopNum)+'.out')
        shutil.copyfile('orca-input.pc', './SPloop-files/mainfrag-SPloop'+str(SPLoopNum)+'.pc')
        blankline()
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
        print("RMSD of charges: {:6.3f} in SP iteration {:6}:".format(RMSD_charges, SPLoopNum))
        if RMSD_charges < RMSD_SP_threshold:
            print("RMSD less than threshold: {}".format(RMSD_charges,RMSD_SP_threshold))
            print("Charges converged in SP iteration {}! SP LOOP over!".format(SPLoopNum))
            break
        print("Not converged in iteration {}. Continuing SP loop".format(SPLoopNum))

    print("Molcrys Charge-Iteration done!")
    #Printing out Cluster fragment file
    Cluster.print_system('Cluster.ygg')
    print("XXX")
    print("Cluster.atomtypes", Cluster.atomtypes)
    return Cluster

    #print("Now Doing Optimization")
    #OptLoopMaxIter=10
    #for OptLoopNum in range(0,OptLoopMaxIter):
    #    frozenlist=listdiff(Cluster.allatoms,Centralmainfrag)
    #    geomeTRICOptimizer(theory=QMMM_SP_ORCAcalculation, fragment=Cluster, frozenatoms=frozenlist,
    #                   coordsystem='tric', maxiter=70)
#
#        exit()
    #OPT of mainfrag:   Interface to Py-Chemshell (should be easy)  or maybe DL-FIND directly
    # Updating of coordinates???
    #Calculate Hessian. Easy via Py-Chemshell. Maybe also easy via Dl-FIND
