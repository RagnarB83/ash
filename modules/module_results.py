from dataclasses import dataclass
import numpy as np
from ash.modules.module_coords import Fragment

#Dataclasses https://realpython.com/python-data-classes/

#Results dataclass that ASH job-functions return
@dataclass
class ASH_Results:
    label: str = ''
    #Single-job: Energy and gradient
    energy: int = None
    gradient: np.array = None
    reaction_energy: float = None
    #Energy contributions if e.g. ORCA_CC_CBS_Theory
    energy_contributions: dict = None

    #Multi-energy job: Lists of energies and gradients
    energies: list = None
    reaction_energies: list = None
    gradients: list = None
    energies_dict: dict = None
    gradients_dict: dict = None
    #Geometry informaiton
    geometry: np.array = None
    initial_geometry: np.array = None
    charge: int = None
    mult: int = None
    #Frequency information
    hessian: np.array = None
    frequencies: np.array = None
    normal_modes: np.array = None
    vib_eigenvectors: np.array = None
    thermochemistry: dict = None
    #Surface-scan job
    surfacepoints: dict = None
    #NEB-type job
    reactant_geometry: np.array = None
    product_geometry: np.array = None
    saddlepoint_geometry: np.array = None
    saddlepoint_fragment: Fragment = None
    MEP_energies_dict: dict = None
    barrier_energy: float = None

#Example: r2 = Results(label="SPjob", energy=900.1, geometry=[[24.3,43.4,433.43]])


#-----------------------------------------------------------------------------------------------------

#OLD:
#https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute


#Class that behaves as a dict but attributes can also be found like:
#results = Results(energy=1.343, gradient=[3.0, 43., 43.0])
#results["energy"] and results.energy
#class Results(dict):
#    def __init__(self, *args, **kwargs):
#        super(Results, self).__init__(*args, **kwargs)
#        self.__dict__ = self
