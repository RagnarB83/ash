==========================
Job Types
==========================

###########################
Single-point calculation
###########################
A single-point calculation is the most basic job to perform.
After creating an Yggdrasill fragment, you create a Theory object, e.g. a QMTheory from: :doc:`QM-interfaces` an
MMTheory (TODO) or a QM/MMTheory (see XX).
The ORCATheory class (see:  :doc:`orca-interface`) is the most common theory to use.
Below, the ORCASP object is created from the ORCATheory class, providing a fragment object and various ORCA-interface
variables to it.

For a single-point calculation only then simply runs the Theory object via executing the internal run function of the
object.

.. code-block:: python

    from yggdrasill import *
    import sys
    settings_yggdrasill.init() #initialize

    HF_frag=Fragment(xyzfiles="hf.xyz")
    #ORCA
    orcadir='/opt/orca_4.2.1'
    orcasimpleinput="! BP86 def2-SVP Grid5 Finalgrid6 tightscf"
    orcablocks="%scf maxiter 200 end"
    ORCASP = ORCATheory(orcadir=orcadir, fragment=HF_frag, charge=0, mult=1,
                        orcasimpleinput=orcasimpleinput, orcablocks=orcablocks)

    #Simple Energy SP calc
    ORCASP.run()

The flexible input-nature of the ORCA interface here allows one to use any method/basis/property inside ORCA for the
single-point job.

It is also possible to request a gradient calculation :

.. code-block:: python

    #An Energy+Gradient calculation
    ORCASP.run(Grad=True)

While Yggdrasill will print out basic information about the run at runtime (e.g. the energy) the energy or gradient
(if requested) is also stored inside the object and can be accessed:

.. code-block:: python

    print(ORCASP.energy)
    print(ORCASP.grad)

By default the files created by the Theory interface are not cleaned up. To have ORCA (in this example) clean up
temporary files (e.g. so they don't interfere with a future job), one can use the cleanup function.


.. code-block:: python

    #Clean up
    ORCASP.cleanup()



###########################
Geometry optimization
###########################

###########################
Numerical frequencies
###########################


###########################
Molecular Dynamics
###########################

