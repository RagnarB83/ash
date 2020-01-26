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

def molcrys(cif_file='', fragmentobjects=[], theory=None, numcores=None, chargemodel='', clusterradius=None):

    banner="""
╔╦╗╔═╗╦  ╔═╗╦═╗╦ ╦╔═╗
║║║║ ║║  ║  ╠╦╝╚╦╝╚═╗
╩ ╩╚═╝╩═╝╚═╝╩╚═ ╩ ╚═╝
    
    """
    print(banner)
    banner3="""
███╗   ███╗ ██████╗ ██╗      ██████╗██████╗ ██╗   ██╗███████╗
████╗ ████║██╔═══██╗██║     ██╔════╝██╔══██╗╚██╗ ██╔╝██╔════╝
██╔████╔██║██║   ██║██║     ██║     ██████╔╝ ╚████╔╝ ███████╗
██║╚██╔╝██║██║   ██║██║     ██║     ██╔══██╗  ╚██╔╝  ╚════██║
██║ ╚═╝ ██║╚██████╔╝███████╗╚██████╗██║  ██║   ██║   ███████║
╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝
    """
    #print(banner3)
    #Here assuming theory can only be ORCA for now
    orcadir=theory.orcadir
    orcablocks=theory.orcablocks
    orcasimpleinput=theory.orcasimpleinput

    print("Fragments defined:")
    print("Mainfrag:", fragmentobjects[0].__dict__)
    print("Counterfrag1:", fragmentobjects[1].__dict__)

    origtime = time.time()
    currtime = time.time()
    #Read CIF-file
    print("Read CIF file:", cif_file)
    blankline()
    cell_length,cell_angles,atomlabels,elems,asymmcoords,symmops=read_ciffile(cif_file)
    print("Cell parameters: {} {} {} {} {} {}".format(cell_length[0],cell_length[1], cell_length[2] , cell_angles[0], cell_angles[1], cell_angles[2]))
    print("Number of fractional coordinates in asymmetric unit:", len(asymmcoords))

    #Calling new cell vectors function instead of old
    cell_vectors=cellbasis(cell_angles,cell_length)
    #cell_vectors=cellparamtovectors(cell_length,cell_angles)
    print("cell_vectors:", cell_vectors)

    #Create system coordinates for whole cell
    print("Filling up unitcell using symmetry operations")
    fullcellcoords,elems=fill_unitcell(cell_length,cell_angles,atomlabels,elems,asymmcoords,symmops)
    numasymmunits=len(fullcellcoords)/len(asymmcoords)
    print("Number of fractional coordinates in whole cell:", len(fullcellcoords))
    print("Number of asymmetric units in whole cell:", numasymmunits)
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
    frag_define(orthogcoords,elems,cell_vectors,fragments=fragmentobjects)
    print_time_rel_and_tot(currtime, origtime)
    currtime=time.time()

    #Reorder coordinates of cell based on Hungarian algorithm
    #TODO: Reorder so that all atoms in fragment have same internal order.
    #TODO: Also reorder so that unit cell is easy to understand.
    #TODO: First all mainfrags, then counterfrags ??

    #Create MM cluster here already or later
    cluster_coords,cluster_elems=create_MMcluster(orthogcoords,elems,cell_vectors,clusterradius)
    print_time_rel_and_tot(currtime, origtime)
    currtime=time.time()

    #Removing partial fragments present in cluster
    #import cProfile
    #cProfile.run('remove_partial_fragments(cluster_coords,cluster_elems,sphereradius,fragmentobjects)')

    cluster_coords,cluster_elems=remove_partial_fragments(cluster_coords,cluster_elems,clusterradius,fragmentobjects)
    print_time_rel_and_tot(currtime, origtime)
    currtime=time.time()
    write_xyzfile(cluster_elems,cluster_coords,"cluster_coords")

    #Create Yggdrasill fragment object
    blankline()
    print("Creating new Cluster fragment:")
    Cluster=Fragment(elems=cluster_elems, coords=cluster_coords)
    Cluster.calc_connectivity(scale=settings_molcrys.scale, tol=settings_molcrys.tol)
    print_time_rel_and_tot(currtime, origtime)
    currtime=time.time()

    print("Cluster conn:", Cluster.connectivity)
    # Going through found frags and identify mainfrags and counterfrags
    for frag in Cluster.connectivity:
        el_list = [cluster_elems[i] for i in frag]
        ncharge = nucchargelist(el_list)
        for fragmentobject in fragmentobjects:
            if ncharge == fragmentobject.Nuccharge:
                fragmentobject.add_clusterfraglist(frag)

    print(mainfrag.clusterfraglist)
    #TODO: Reorder cluster with reflections also

    #Reorder fraglists in each fragmenttype via Hungarian algorithm.
    # Ordered fraglists can then easily be used in pointchargeupdating
    reordercluster(Cluster,mainfrag)
    reordercluster(Cluster,counterfrag1)
    #TODO: after removing partial fragments and getting connectivity etc. Would be good to make MM cluster neutral
    print(mainfrag.clusterfraglist)
    print(counterfrag1.clusterfraglist)

    mainfrag.print_infofile('mainfrag-info.txt')
    counterfrag1.print_infofile('counterfrag1-info.txt')

    #Add fragmentobject-info to Cluster fragment
    Cluster.add_fragment_type_info(fragmentobjects)

    #Cluster is now almost complete, only charges missing. Print info to file
    print(Cluster.print_system("Cluster-info-nocharges.txt"))

    # Create dirs to keep track of various files before ORCA calculations begin
    try:
        os.mkdir('SPloop-files')
    except:
        shutil.rmtree('SPloop-files')
        os.mkdir('SPloop-files')


    # Calculate atom charges for each gas fragment. Updates atomcharges list inside Cluster fragment
    gasfragcalc(fragmentobjects,Cluster,chargemodel,orcadir,orcasimpleinput,orcablocks,numcores)

    print_time_rel_and_tot_color(currtime, origtime)
    currtime=time.time()
    print("Atom charge assignments in Cluster done!")
    blankline()
    print("Cluster:", Cluster.__dict__)

    #Cluster is now complete. Print info to file
    Cluster.print_system("Cluster-info_afterGas.txt")
    mainfrag.print_infofile('mainfrag-info_afterGas.txt')

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
    #SP-LOOP FOR MAINFRAG
    for SPLoopNum in range(0,SPLoopMaxIter):
        print("This is Charge-Iteration Loop number", SPLoopNum)
        atomcharges=[]
        #Creating ORCA theory object without fragment information.
        #fragmentobjects[0] is always mainfrag
        # Charge-model info to add to inputfile
        chargemodelline = chargemodel_select(chargemodel)
        ORCAQMtheory = ORCATheory(orcadir=orcadir, charge=fragmentobjects[0].Charge, mult=fragmentobjects[0].Mult,
                              orcasimpleinput=orcasimpleinput,
                              orcablocks=orcablocks, extraline=chargemodelline)

        print("ORCAQMtheory:", ORCAQMtheory)
        print(ORCAQMtheory.__dict__)
        #Defining QM region. Should be the mainfrag at approx origin
        Centralmainfrag=fragmentobjects[0].clusterfraglist[0]
        print("Centralmainfrag:", Centralmainfrag)
        print_coords_for_atoms(Cluster.coords,Cluster.elems,Centralmainfrag)

        bla=Cluster.get_coords_for_atoms(Centralmainfrag)
        print("bla:", bla)
        print("z Cluster.atomcharges:", Cluster.atomcharges)
        QMMM_SP_ORCAcalculation = QMMMTheory(fragment=Cluster, qm_theory=ORCAQMtheory, qmatoms=Centralmainfrag,
                                     atomcharges=Cluster.atomcharges, embedding='Elstat')

        # Run ORCA calculation with charge-model info
        QMMM_SP_ORCAcalculation.run(nprocs=1)

        #Grab atomic charges for fragment.
        atomcharges = grabatomcharges(chargemodel, ORCAQMtheory.inputfilename + '.out')
        print("atomcharges:", atomcharges)

        # Keep backup of ORCA outputfile in dir SPloop-files
        shutil.copyfile('orca-input.out', './SPloop-files/mainfrag-SPloop'+str(SPLoopNum)+'.out')

        blankline()
        #Adding mainfrag charges to mainfrag--object
        fragmentobjects[0].add_charges(atomcharges)
        # Assign pointcharges to each atom of MM cluster.
        pointchargeupdate(Cluster, fragmentobjects[0], atomcharges)
        Cluster.print_system("Cluster-info_afterSP{}.txt".format(SPLoopNum))
        mainfrag.print_infofile('mainfrag-info_afterSP{}.txt'.format(SPLoopNum))
        blankline()
        print("Current charges:", fragmentobjects[0].all_atomcharges[-1])
        print("Previous charges:", fragmentobjects[0].all_atomcharges[-2])
        RMSD_charges=rmsd_list(fragmentobjects[0].all_atomcharges[-1],fragmentobjects[0].all_atomcharges[-2])
        print("RMSD of charges: {:6.3f} in SP iteration {:6}:".format(RMSD_charges, SPLoopNum))
        if RMSD_charges < RMSD_SP_threshold:
            print("RMSD less than threshold: {}".format(RMSD_charges,RMSD_SP_threshold))
            print("Charges converged in SP iteration {}! SP LOOP over!".format(SPLoopNum))
            break
        print("Not converged in iteration {}. Continuing SP loop".format(SPLoopNum))

    print("Now Doing Optimization")
    OptLoopMaxIter=10
    for OptLoopNum in range(0,OptLoopMaxIter):
        frozenlist=listdiff(Cluster.allatoms,Centralmainfrag)
        geomeTRICOptimizer(theory=QMMM_SP_ORCAcalculation, fragment=Cluster, frozenatoms=frozenlist,
                       coordsystem='tric', maxiter=70)

        exit()

    #OPT of mainfrag:   Interface to Py-Chemshell (should be easy)  or maybe DL-FIND directly
    # Updating of coordinates???

    #Calculate Hessian. Easy via Py-Chemshell. Maybe also easy via Dl-FIND
