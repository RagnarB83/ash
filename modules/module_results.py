from dataclasses import dataclass
import numpy as np

#Dataclasses https://realpython.com/python-data-classes/

#Results dataclass that ASH job-functions return
@dataclass
class ASH_Results:
    label: str = ''
    #Energy and gradient
    energy: int = None
    gradient: np.array = None
    #Geometry
    geometry: np.array = None
    initial_geometry: np.array = None
    #Frequency
    hessian: np.array = None
    normal_modes: np.array = None
    thermochemistry: np.array = None
    #NEB
    reactant_geometry: np.array = None
    product_geometry: np.array = None
    saddlepoint_geometry: np.array = None

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
