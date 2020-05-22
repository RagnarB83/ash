#Dynamics in ASH



#OPENMM dynamics. Using either OpenMM ASH object or QM/MM ASH object (contains OpenMM object)

def openMMdynamics(theory=None, fragment=None, steps=100, reportsteps=1000):
    if theory.__class__.__name__ == "QMMMTheory":
        try:
            theory.mm_theory.__class__.__name__ == "OpenMMTheory":
        except:
            print("MM theory of QM/MM theory object is not OpenMMtheory object")
            exit(1)


        #Set up CustomNonbonded force here??
        #Or in OpenMM object?

        #For QM/MM


        force=theory.mm_theory.nonbonded_force

        CustomNonbondedForce * force = new CustomNonbondedForce( "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
                                                                 " sigma=0.5*(sigma1+sigma2);"
                                                                 " epsilon=sqrt(epsilon1*epsilon2)");


        #Get positions from ASH fragment
        coords=fragment.coords
        positions = [theory.mm_theory.Vec3(coords[i, 0] / 10, coords[i, 1] / 10, coords[i, 2] / 10) for i in
               range(len(coords))] * theory.unit.nanometer
        theory.mm_theory.simulation.context.setPositions(positions)
        # Set up the reporters to report energies every 1000 steps.
        theory.mm_theory.simulation.reporters.append(PDBReporter('output.pdb', reportsteps))
        theory.mm_theory.simulation.reporters.append(StateDataReporter(stdout, reportsteps, step=True,
                                                      potentialEnergy=True, temperature=True))

        # run simulation
        theory.simulation.step(steps)


    elif theory.__class__.__name__ == "OpenMMTheory":

        #Get positions from ASH fragment
        coords=fragment.coords
        positions = [theory.Vec3(coords[i, 0] / 10, coords[i, 1] / 10, coords[i, 2] / 10) for i in
               range(len(coords))] * theory.unit.nanometer
        theory.simulation.context.setPositions(positions)

        # Set up the reporters to report energies every 1000 steps.
        theory.simulation.reporters.append(PDBReporter('output.pdb', reportsteps))
        theory.simulation.reporters.append(StateDataReporter(stdout, reportsteps, step=True,
                                                      potentialEnergy=True, temperature=True))

        # run simulation
        theory.simulation.step(steps)



