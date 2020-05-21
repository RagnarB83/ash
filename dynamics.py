#Dynamics in ASH



#OPENMM dynamics. Using either OpenMM object or QM/MM object (contains OpenMM object)

def openMMdynamics(theory=None, fragment=None):
    if theory.__class__.__name__ == "QMMMTheory":
        pass
    elif theory.__class__.__name__ == "OpenMMTheory":
        pass


