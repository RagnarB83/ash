==========================
QM Interfaces
==========================

QM interfaces currently supported in Yggdrasill are:
ORCA, PySCF, Psi4 and xTB.
To use the interfaces you define a QMtheory object of the appropriate Class.
The QM-interface Classes are:
ORCATheory, PySCFTheory, Psi4Theory, xTBTheory

When defining a QMtheory object you are creating an instance of one of the QMTheory classes.
When defining the object, a few keyword arguments are required, that differs between classes.
It is not necessary to define a fragment (Yggdrasill fragment) as part of the QMTheory (e.g. not done for QM/MM),
but is necessary for a single-point energy QM job. For a geometry optimization the fragment does not have to be part
of the QMTheory object (instead passed to the optimizer object/function).

###########################
ORCATheory
###########################
Coming: ORCATheory Class definition and arguments

The ORCA interface is quite flexible. It currently requires the path to the ORCA installation to be passed on as a keyword
argumentwhen creating object. The charge, multiplicity are also necessary keyword arguments (integers).
As are orcasimpleinput and orcablocks keyword arguments (accepts single or mult-line strings).


.. code-block:: python

    #Create fragment object from XYZ-file
    HF_frag=Fragment(xyzfile='hf.xyz')
    #ORCA
    orcadir='/opt/orca_4.2.1'
    input="! BP86 def2-SVP Grid5 Finalgrid6 tightscf"
    blocks="%scf maxiter 200 end"
    ORCAcalc = ORCATheory(orcadir=orcadir, fragment=HF_frag, charge=0, mult=1,
                                orcasimpleinput=input, orcablocks=blocks)

    #Run a single-point energy job
    ORCAcalc.run()
    #An Energy+Gradient calculation running on 8 cores
    ORCASP.run(Grad=True, nprocs=8)



Here a fragment (here called HF_frag) is defined (from XYZ file) and passed to the ORCAtheory object (called ORCAcalc).
Additionally, orcadir, input, and blocks string variables are defined and passed onto the ORCA object via keywords, as
are charge and spin multiplicity.
The object can then be run via the object function (ORCAcalc.run) which is the equivalent of a single-point energy job.
By using Grad=True keyword argument ORCAcalc.run, a gradient is also requested and by setting the nprocs=8 argument,
an 8-core run ORCA calculation is requested (handled via OpenMPI, requires OpenMPI variables to be set outside Python).
