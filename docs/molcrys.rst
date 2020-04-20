=================================================
MolCrys: Automatic QM/MM for Molecular Crystals
=================================================
The automatic molecular crystal QM/MM method in Yggdrasill is based on the work described
in articles by Bjornsson et al.[1,2]

**The basic (automatic) protocol is:**


| 1. Read CIF-file (or Vesta XTL-file) containing fractional coordinates of the cell.
| 2. Apply symmetry operations to get coordinates for whole unit cell
| 3. Identify the molecular fragments present in cell via connectivity and match with user-input
| 4. Extend the unit cell and cut out a spherical cluster with user-defined MM radius (typically 30-50 Å). Only whole molecules included.
| 5. Define atomic charges of the molecular fragments from QM calculations.
| 6. Define Lennard-Jones parameters of the molecular fragments.
| 7. Iterate the atomic charges of the main molecular fragment in the center of the cluster until self-concistency.
| 8. Optional: Perform QM/MM geometry optimization of the central fragment or other job types.
| 9. Optional: Extend the QM-region around the central fragment for improved accuracy
| 10. Optional: Perform numerical frequency calculations or molecular dynamics simulations
| 11. Optional: Molecular property calculation in the solid-state



| Critical features of the implementation:
| - Handles CIF-files with inconsistent atom ordering by automatic fragment reordering.
| - Improved accuracy via QM-region expansion
| - Numerical frequencies (to be tested)

| Limitations:
| - CIF file can not contain extra atoms such as multiple thermal populations. Also missing H-atoms have to be added beforehand.
| - Polymeric systems or pure solids (e.g. metallic) can not be described. Only system with natural fragmentation.

| Features to be implemented:
| - Automatic derivation of Lennard-Jones parameters
| - Beyond Lennard-Jones
| - Molecular dynamics

| 1. Modelling Molecular Crystals by QM/MM: Self-Consistent Electrostatic Embedding for Geometry Optimizations and Molecular Property Calculations in the Solid,  R. Bjornsson and M. Bühl,  J. Chem. Theory Comput., 2012, 8, 498-508.
| 2. R. Bjornsson, manuscript in preparation

###############################################
Example: QM/MM Cluster setup from CIF-file
###############################################
Here we show how to use code for an example Na[H2PO4] crystal. This molecular crystal contains 2 fragment-types:
Na\ :sup:`+` \ and H\ :sub:`2`\PO\ :sub:`4`:sup:`-` \

In the Python script, one has to initially *import Yggdrasill*, the molcrys module and call the *settings_yggdrasill.init()* function.

The script then actually just calls one function, called **molcrys** at the bottom of the script:

.. code-block:: python

    Cluster = molcrys(cif_file=cif_file, fragmentobjects=fragmentobjects, theory=ORCAcalc,
        numcores=numcores, clusterradius=sphereradius, chargemodel=chargemodel, shortrangemodel=shortrangemodel)


This is the only function of this script but as we can see, there are a number of arguments to be provided.
It is usually more convenient to define first the necessary variables in multiple lines above this command.
In the full script, seen below, a number of variables are defined, following standard Python syntax.
The variables are then passed as arguments to the  **molcrys** function at the bottom of the script.

.. code-block:: python

    from yggdrasill import *
    from molcrys import *
    settings_yggdrasill.init()
    #######################
    # MOLCRYS INPUT          #
    #######################
    cif_file="nah2po4_choudhary1981.cif"
    sphereradius=35

    #Number of cores available for either ORCA parallelization or multiprocessing
    numcores=12

    #Charge-iteration QMinput
    orcadir='/opt/orca_4.2.1'
    orcasimpleinput="! BP86 def2-SVP def2/J Grid5 Finalgrid6 tightscf"
    orcablocks="%scf maxiter 200 end"
    #Defining QM theory without fragment, charge or mult
    ORCAcalc = ORCATheory(orcadir=orcadir, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks, nprocs=numcores)

    #Chargemodel options: CHELPG, Hirshfeld, CM5, NPA, Mulliken
    chargemodel='Hirshfeld'
    #Shortrange model. Usually Lennard-Jones. Options: UFF_all, UFF_modH
    shortrangemodel='UFF_modH'

    #Define fragment types in crystal: Descriptive name, formula, charge and mult
    mainfrag = Fragmenttype("Phosphate","PO4H2", charge=-1,mult=1)
    counterfrag1 = Fragmenttype("Sodium","Na", charge=1,mult=1)
    #Define list of fragmentobjects. Passed on to molcrys
    fragmentobjects=[mainfrag,counterfrag1]

    #Define global system settings (currently scale and tol keywords for connectivity)
    settings_yggdrasill.scale=1.0
    settings_yggdrasill.tol=0.3
    #settings_molcrys.tol=0.0001
    # Modified radii to assist with connectivity.
    #Setting radius of Na to almost 0. Na will then not bond
    eldict_covrad['Na']=0.0001
    #eldict_covrad['H']=0.15
    print(eldict_covrad)


    #Calling molcrys function and define Cluster object
    Cluster = molcrys(cif_file=cif_file, fragmentobjects=fragmentobjects, theory=ORCAcalc,
            numcores=numcores, clusterradius=sphereradius, chargemodel=chargemodel, shortrangemodel=shortrangemodel)


We point to the CIF file that should be read and define a sphereradius. We also define the number of cores available
(should later match that defined in the jobscript), that both ORCA and Yggdrasill may use in their parallelization.
Next, an ORCA theory object is defined where we set the path to ORCA and define the structure of the inputfile used
when running ORCA calculations.


The chargemodel and shortrangemodel variables are used to define keywords that **molcrys** will recognize.
The chargemodel defines how to derive the pointcharges for the MM cluster for the QM-MM electrostatic interaction. Available chargemodels are: CHELPG, Hirshfeld, CM5, NPA, Mulliken

The shortrangemodel defines the short-range interactions between QM and MM atoms (other than the electrostatic).
Currently, only the UFF Lennard-Jones model is available that uses element-specific parameters (from the Universal Forcefield, UFF) to set up Lennard-Jones potentials between
all atoms. The "UFF_modH" keyword is currently recommended that uses available parameters for all elements except the LJ
parameters for H are set to zero to avoid artificial repulsion for acidic H-atoms.

Next, we have to define the fragments present in the crystal. In the future, this may become more automated.
Thus, we define a fragment, called *mainfrag*, that is our primary interest. Here, this is the H\ :sub:`2`\PO\ :sub:`4`:sup:`-` \
anion, while the counterion Na\ :sup:`+` \ ion is of less interest, here labelled *counterfrag1*.
This distinction between fragments means that the *mainfrag* will be at the center of the cluster.
It also means that the charge-iterations are only performed for *mainfrag*.
For each molecular fragment, we define an object of class Fragmenttype with a name e.g. "Phosphate",
elemental formula, e.g. "PO4H2", and define the charge and multiplicity of that fragment.
The elemental formula is crucial as from the formula the nuclear charges is calculated which is used to identify these
fragments in the molecular crystal. Once the fragments are defined we group them together in the following order as a list
called fragmentobjects:     fragmentobjects=[mainfrag,counterfrag1]

Finally, the script shows how the connectivity can be modified in order for the fragment identification to succeed.
The fragment identification works by finding what atoms are connected according to the formula:

(AtomA,AtomB-distance) < scale*(AtomA-covalent-radius+AtomB-covalent-radius) + tol

Thus, if the distance between atoms A and B is less than the sum of the elemental covalent radii
(which can be scaled by a parameter scale or shifted by a parameter tol) then the atoms are connected.
Using default parameters of the element radii (Alvarez 2008), the default scaling of 1.0 and a tolerance of 0.1
(global scale and tol parameters are defined in settings_yggdrasill file) works in many cases.
For the NaH\ :sub:`2` \PO\ :sub:`4` \ crystal, however, that features strong hydrogen-bonding and the ionic Na\ :sup:`+` \ fragment, however, we have to make some modifications.
In the script above, we thus have to set the tol parameter to 0.3 and change the radius of the Na\ :sup:`+` \ ion to a small value.
The covalent radii of the elements are stored in a global Python dictionary, eldict_covrad which can be easily modified as shown
and its contents printed. In the future, the radius of the Na may by default be set to a small number.

Unlike the other variables, the *settings_yggdrasill.scale*, *settings_yggdrasill.tol* and *eldict_covrad* are global variables
that **molcrys** and **Yggdrasill** will have access to.

The other variables defined in the script have to be passed as keyword argument values to the respective keyword of
the **molcrys** function:

.. code-block:: python

    Cluster = molcrys(cif_file=cif_file, fragmentobjects=fragmentobjects, theory=ORCAcalc,
        numcores=numcores, clusterradius=sphereradius, chargemodel=chargemodel, shortrangemodel=shortrangemodel)

These are currently the only arguments that can be provided to the **molcrys** function, with the exception that
instead of a *cif_file* argument, an *xtl_file* argument can alternatively be provided where the name of the XTL-file should
be passed on instead. An XTL-file can be created by the Vesta software (http://jp-minerals.org/vesta/en/).

The purpose of the molcrys function is primarily to create an Yggdrasill cluster-fragment, here called Cluster. The Cluster fragment
will contain the coordinates of the spherical MM cluster with charges from the self-consistent QM procedure and atom-types
defined via the shortrange model procedure chosen. The Cluster fragment is both present in memory and also written to disk as:
Cluster.ygg A forcefield file is also created, Cluster_forcefield.ff, that contains
the Lennard-Jones parameters defined for the atomtypes that have been chosen for every atom in the Cluster fragment.

Typically running the **molcrys** function takes only a few minutes, depending on the size of the molecular fragments
and the size of the Cluster radius but usually it is easiest to submit this to the cluster to run the QM calculations in parallel.
If the connectivity requires modifications, however, then first running through the script directly may be easier.

The Cluster fragment file, Cluster.ygg, can be used directly in a single-point property job (see later).
If using the ORCA interface, the last orca-input.inp and orca-input.pc files created by **molcrys**
can also directly be used to run a single-point electrostatically-embedded property calculation with ORCA
(note: not a geometry optimization) as they contain the QM-coordinates of the central fragment (orca-input.inp) and .
the MM coordinates and self-consistent pointcharges (orca-input.pc).

#########################################
MOLCRYS Geometry optimization
#########################################
To run a QM/MM geometry optimization, this can be done separately by preparing a regular Yggdrasill QM/MM inputfile and read in
the Cluster fragment file and the forcefield file, Cluster_forcefield.ff.
It is often more convenient to continue with a QM/MM geometry optimization in the same script, after the **molcrys** function.
In that case, the code below can simply be appended to the previous script.

.. code-block:: python

    #Once molcrys is done we have a Cluster object (named Cluster) in memory and also printed to disk as Cluster.ygg
    # We can then do optimization right here using that Cluster object.
    #Alternatively or for restart purposes we can read Cluster object into a separate QM/MM Opt job.
    print("Now Doing Optimization")
    # Defining Centralmainfrag (list of atoms) for optimization
    #Centralmainfrag=fragmentobjects[0].clusterfraglist[0]
    Centralmainfrag=Cluster.connectivity[0]
    #Can also be done manually
    #Centralmainfrag=[0, 1, 5, 8, 9, 12, 14]
    print("Centralmainfrag:", Centralmainfrag)

    charge=fragmentobjects[0].Charge
    mult=fragmentobjects[0].Mult
    #
    Cluster_FF=MMforcefield_read('Cluster_forcefield.ff')

    #Defining, QM, MM and QM/MM theory levels for Optimization
    #If same theory as used in molcrys, then orcadir, orcasimpleinput and orcablocks can be commented out/deleted.
    orcadir='/opt/orca_4.2.1'
    orcasimpleinput="! BP86 def2-SVP def2/J Grid5 Finalgrid6 tightscf"
    orcablocks="%scf maxiter 200 end"
    ORCAQMpart = ORCATheory(orcadir=orcadir, charge=charge, mult=mult, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)
    MMpart = NonBondedTheory(charges = Cluster.atomcharges, atomtypes=Cluster.atomtypes, forcefield=Cluster_FF, LJcombrule='geometric')
    QMMM_object = QMMMTheory(fragment=Cluster, qm_theory=ORCAQMpart, mm_theory=MMpart,
        qmatoms=Centralmainfrag, atomcharges=Cluster.atomcharges, embedding='Elstat', nprocs=numcores)


    geomeTRICOptimizer(theory=QMMM_object, fragment=Cluster, coordsystem='tric', maxiter=170, ActiveRegion=True, actatoms=Centralmainfrag )



We define a variable Centralmainfrag as the list of atoms that should be both described at the QM level (will be passed to qmatoms keyword argument)
and should be optimized in a geometry optimization (will be passed to actatoms of optimizer ). This list may also be a larger QM-cluster, e.g. multiple H2PO4 units or with Na+ included.

The charge and multiplicity of the molecule is then defined and a forcefield object is defined by reading in the 'Cluster_forcefield.ff'
forcefield file, previously created by the **molcrys** function.

Next we have to define a QM/MM object by combining a QM-theory object (here of class ORCATheory) and an MM theory object (of class NonBondedTheory).
See QM/MM theory page for more information on this.

Finally we call the optimizer program, here the geomeTRICoptimizer:

.. code-block:: python

    geomeTRICOptimizer(theory=QMMM_object, fragment=Cluster, coordsystem='tric', maxiter=170, ActiveRegion=True, actatoms=Centralmainfrag )


We provide a theory argument to the optimizer (our QM/MM object), the Cluster fragment, we specify the coordinate
system (here the TRIC internal coordinates are used), max no. of iterations may be provided and finally we specify that we have an active region
and that only the atoms provided to the actatoms keyword argument should be optimized. Note that MM atoms can not be optimized when
doing nonbonded QM/MM like we are doing here.

If the optimization converges, a new fragment containing the optimized geometry is provided, called "Fragment-optimized.ygg".
Note: Only the geometry of the central fragment (or whatever qmatoms/actoms was set to) is optimized. The other atoms
are still at the original positions as determined from the crystal structure.
The optimization trajectory is also available as a multi-structure XYZ file, as either "geometric_OPTtraj_Full.xyz"
(Full system) or "geometric_OPTtraj.xyz" (Act-region only).



**Note:**

If the optimization is done separately, the code above would have to be manually changed in a few places.
First the Cluster fragment would be read in:

.. code-block:: python

    Cluster=Fragment(fragfile='Cluster.ygg')


One would then manually define variables charge, mult (of the main fragment) as *fragmentobjects* would not be available.


#########################################
MOLCRYS EXPANDED QM region calculation
#########################################

For either a QM/MM geometry optimization or a QM/MM single-point property calculation (see below), the QM-region does
not have to be a single fragment. If the qmatoms list and the actatoms list (for optimizations) is modified, then a larger
QM cluster can be calculated instead in the QM/MM calculation. This should generally result in a more accurate calculation
as the QM-MM boundary effect can be reduced.

The qmatoms and actatoms lists (i.e. the values provided to qmatoms and actatoms keyword arguments to QM/MM object or
geomeTRICOptimizer function can be modified manually, e.g. by visually inspecting an XYZ-file version of the Cluster and
provide the correct list of atom indices (Note: Yggdrasill counts from zero).

More conveniently, the QMregionfragexpand function can be used to find nearby atoms for an initial list of atoms.

.. code-block:: python

    Centralmainfrag=Cluster.connectivity[0]
    expanded_central_region = QMregionfragexpand(fragment=Cluster,initial_atoms=Centralmainfrag, radius=3)

In the code example above, a new variable called "expanded_central_region" is defined that contains a new list of atoms containing
whole fragments that are 3 Å away from the central mainfrag.
This expanded_central_region list can then be fed to qmatoms and actaoms keyword arguments in either a QM/MM optimization
job or a single-point property job.
The radius variable would have to be tweaked and the result inspected to get appropriately sized and shaped QM-clusters.
**Note:** The charge and multiplicity keywords probably need to be changed for the new QM-cluster calculations.




#########################################
MOLCRYS Property calculation
#########################################

A QM/MM molecular/spectroscopic property calculations can be carried either using Yggdrasill or using the QM program directly.
If using ORCA, the appropriate property keywords can be added to orcasimpleinput or orcablocks variables that will be passed onto ORCA.

A single-point QM/MM calculation can be performed by defining a QM/MM object as done before and then simply use the object's
internal run function (run performs a single-point energy calculation). Make sure to specify the desired Cluster object: e.g. the original Cluster
from the CIF-file or the Cluster file from the QM/MM optimization (contains optimized coordinates for the central fragment).

.. code-block:: python

    from yggdrasill import *
    settings_yggdrasill.init()

    #Read in Cluster fragment
    Cluster=Fragment(fragfile='Cluster.ygg')

    # Defining Centralmainfrag (list of atoms) for optimization
    Centralmainfrag=Cluster.connectivity[0]
    #Can also be done manually
    #Centralmainfrag=[0, 1, 5, 8, 9, 12, 14]
    print("Centralmainfrag:", Centralmainfrag)

    #Can also be done done manually if fragmentobjects not available, e.g. charge=-1, mult=1
    charge=-1
    mult=1

    #Reading in force-field file
    Cluster_FF=MMforcefield_read('Cluster_forcefield.ff')

    #Defining, QM, MM and QM/MM theory levels for Optimization
    #ORCAlines: If same theory as used in molcrys, then orcadir, orcasimpleinput and orcablocks can be commented out/deleted.
    numcores=12
    orcadir='/opt/orca_4.2.1'
    orcasimpleinput="! PBE0 def2-SVP def2/J Grid5 Finalgrid6 tightscf NMR"
    orcablocks="
    %scf maxiter 200 end
    %eprnmr
    Nuclei = all B { shift }
    Nuclei = all C { shift }
    end
    "
    ORCAQMpart = ORCATheory(orcadir=orcadir, charge=charge, mult=mult, orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)
    MMpart = NonBondedTheory(charges = Cluster.atomcharges, atomtypes=Cluster.atomtypes, forcefield=Cluster_FF, LJcombrule='geometric')
    QMMM_object = QMMMTheory(fragment=Cluster, qm_theory=ORCAQMpart, mm_theory=MMpart,
        qmatoms=Centralmainfrag, atomcharges=Cluster.atomcharges, embedding='Elstat', nprocs=numcores)

    QMMM_object.run()


Alternatively (somtimes easier), the last ORCA inputfile (orca-input.pc) and pointcharge file (orca-input.pc) from either **molcrys**
or the optimization can be used to run a single-point property job. If the inputfile came from the optimization job then it contains
optimized QM coordinates and the pointcharge-file should contain the self-consistently determined pointcharges for the full cluster.
Thus a simple modification to the inputfile would be required to run a property job using all functionality available in ORCA.


#########################################
MOLCRYS Numerical frequencies
#########################################

Not yet ready

#########################################
Molecular Dynamics
#########################################

Not yet ready


#################################################
Fragment identification/Connectivity issues
##############################################

If there are difficulties in obtaining the correct fragment identification from the CIF file, first check that the CIF file is correct:

| - Are there atoms missing? e.g. hydrogens? These would have to be added to the CIF file.
| - Are there multiple thermal populations of some residues? These would have to be deleted from the CIF file
| - Do the total atoms in the unit cell add up to the expected number of atoms based on the fragments present?

If the atoms in the unitcell are correct then the problem is more likely to do with the default connectivity parameters
not being general enough for the system.
Start by playing around with the tol parameter, try values between 0 to 0.5
The scaling parameter can also be used, though often less useful.
Often, modifying the covalent radius of an element (see above example for Na+) works well.

