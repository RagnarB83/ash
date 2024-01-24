import numpy as np
#Dynamics in ASH

#Molecular dynamics
#MD simulation object, created and used by MolecularDynamics function but not intended to be created by user
class Simulation:
    def __init__(self,ensemble=None,timestep=None,numsteps=None,set_temperature=None,thermostat=None,tau=None):
        #Defined upon creation
        self.ensemble=ensemble
        self.timestep=timestep
        self.numsteps=numsteps
        self.tau=tau
        self.thermostat=thermostat
        #Provide or get from fragment
        #self.elems=elems
        #masslist, define here or get masses from fragment
        self.masslist=[]
        #Create from masslist ?
        self.masses_array=[[]]
        self.numatoms=len(self.elems)
        
        #Run variables, updated by internal run
        self.currentstep=0
        self.set_temperature=0.0
        self.current_coords=np.zeros((self.numatoms,3))
        self.current_velocities=np.zeros((self.numatoms,3))
        self.current_accel=np.zeros((self.numatoms,3))
        self.current_accel_t_plus_dt=np.zeros((self.numatoms,3))
        self.current_temperature=0.0
        self.potenergy=0.0
        self.kinenergy=0.0
        self.totenergy=0.0
    def run(self):
        #Use or not??
        fdsf="sdf"
        
        
#Simple MD function
def MolecularDynamics(fragment=None, theory=None, ensemble="NVE", timestep=1, numsteps=100, temperature=298.15, tau=500,
    thermostat="berendsen", write_xyz_frequency=1, write_log_frequency=1, write_userfunction_frequency=100, MDtrajname="trajectory.xyz", 
    initial_velocities="zero", debug=False):
    print("Running MOLECULAR DYNAMICS module")

    #Create simulation object that contains user-set parameters and also contains current simulation parameters
    simobject=Simulation(ensemble=ensemble,timestep=timestep,numsteps=numsteps,set_temperature=298.15,thermostat=thermostat,tau=tau)

    #Constants
    #Defining multiples of dt for speed
    dt=timestep
    dt_2=0.5*dt
    dt_4=0.5*dt_2
    dt_8=0.5*dt_4
    hartoeV=27.211386245988
    bohr2ang = 0.52917721067
    ang2bohr = 1.88972612546

 
    ####################
    #INITIALIZATION
    ####################


