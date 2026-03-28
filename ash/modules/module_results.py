from dataclasses import dataclass
import numpy as np
from ash.modules.module_coords import Fragment
from ash.functions.functions_general import print_if_level

# Dataclasses https://realpython.com/python-data-classes/

# Results dataclass that ASH job-functions return
@dataclass
class ASH_Results:
    label: str = None
    # Single-job: Energy and gradient
    energy: float = None
    qm_energy: float = None
    mm_energy: float = None
    qmmm_energy: float = None
    gradient: np.array = None
    reaction_energy: float = None
    # Energy contributions if e.g. ORCA_CC_CBS_Theory
    energy_contributions: dict = None

    # Multi-energy job: Lists of energies and gradients
    energies: list = None
    reaction_energies: list = None
    relative_energies: list = None
    labels: list = None
    gradients: list = None
    energies_dict: dict = None
    gradients_dict: dict = None
    # parallel Multi-energy job
    # Name of worker directories that could be accessed later
    worker_dirnames: dict = None
    # Geometry informaiton
    geometry: np.array = None
    initial_geometry: np.array = None
    charge: int = None
    mult: int = None
    # Possible unsorted information.
    properties: dict = None
    # Frequency information
    hessian: np.array = None
    frequencies: list = None
    freq_masses: list = None
    freq_elems: list = None
    freq_coords: np.array = None
    freq_atoms: list = None
    freq_TRmodenum: int = None
    freq_projection: bool = None
    freq_scaling_factor: float = None
    #freq_displacement_dipole: dict = None
    #freq_displacement_polarizability: dict = None
    freq_dipole_derivs: np.array = None
    freq_polarizability_derivs: np.array = None
    freq_Raman: bool = None
    normal_modes: np.array = None
    Raman_activities: np.array = None
    IR_intensities: np.array = None
    depolarization_ratios: np.array = None
    vib_eigenvectors: np.array = None
    thermochemistry: dict = None
    displacement_dipole_dictionary: dict = None
    displacement_polarizability_dictionary: dict = None
    # Surface-scan job
    surfacepoints: dict = None
    # NEB-type job
    reactant_geometry: np.array = None
    product_geometry: np.array = None
    saddlepoint_geometry: np.array = None
    saddlepoint_fragment: Fragment = None
    MEP_energies_dict: dict = None
    barrier_energy: float = None

    # Print only defined attributes
    def print_defined(self, printlevel=2,):
        print_if_level("\nPrinting defined attributes of ASH_Results dataclass", printlevel,2)
        for k,v in self.__dict__.items():
            if v is not None:
                print(f"{k}: {v}")
    def write_to_disk(self,filename="ASH.result", printlevel=2):
        import json
        print_if_level("\nWriting to disk defined attributes of ASH_Results dataclass", printlevel,2)
        f = open(filename,'w')

        newdict={}
        # Looping over attributes, converting ndarrays to lists and skipping ASH objects
        for k,v in self.__dict__.items():
            # Deal with np array
            if isinstance(v,np.ndarray):
                # Check for nans in array
                if np.any(np.isnan(v)):
                    print_if_level(f"Warning: nan in array {k}", printlevel,2)
                    print_if_level(f"Skipping writing to disk", printlevel,2)
                    #exit()
                else:
                    newv= v.tolist()
                    newdict[k]=newv
            # Dealing with cases of lists of np arrays (e.g. pol derivs)
            elif isinstance(v,list):
                # If list is empty, just add it
                if len(v)==0:
                    newdict[k]=v
                elif isinstance(v[0],np.ndarray):
                    newv=[i.tolist() for i in v]
                    newdict[k]=newv
                else:
                    newdict[k]=v
            elif isinstance(v,Fragment):
                print_if_level(f"Warning: Fragment object is not included in ASH.result on disk", printlevel,2)
            else:
                newdict[k]=v

        print_if_level("Results object data:", printlevel,2)
        for k,v in newdict.items():
            if type(v) is list or type(v) is np.ndarray:
                if len(v) < 20:
                    print_if_level(f"{k} : {len(v)}", printlevel,2)
                else:
                    print_if_level(f"{k} : too long to print", printlevel,2)
            else:
                if v is not None:
                    print_if_level(f"{k} : {v}", printlevel,2)
                #print(f"{k} : {v}")
        # Dump new dict
        try:
            f.write(json.dumps(newdict, allow_nan=True))
        except TypeError as e:
            print_if_level(f"Error writing ASH_Results to disk: {e}", printlevel,2)
            print_if_level("Skipping writing to disk", printlevel,2)
            return
        f.close()

# Read ASH-Results data from disk
def read_results_from_file(filename="ASH.result", printlevel=2):
    import json

    print_if_level("Reading ASH_Results data from file:", filename)
    data = json.load(open(filename))
    print_if_level("Data read from file:", printlevel,2)
    for k,v in data.items():
        print_if_level(f"{k} : {v}", printlevel,2)

    r = ASH_Results(**data)
    return r
