import numpy as np
#Interface to Plumed


#PLUMED_ASH class

class plumed_ASH():
    def __init__(self, path_to_plumed_kernel=None, bias_type="1D_MTD", fragment=None, colvar_type=None, colvar_indices=None,
               temperature=300.0, hills_file="HILLS", colvar_file="COLVAR", height=None, sigma=None, biasfactor=None, timestep=None,
               stride_num=10, pace_num=500):
        
        if timestep==None:
            print("timestep= needs to be provided to plumed object")
            exit()

        
        if path_to_plumed_kernel == None:
            print("plumed_MD requires path_to_plumed_kernel argument to be set")
            print("Should point to: /path/to/libplumedKernel.so")
            exit()
        try:
            import plumed
        except:
            print("Found no plumed library. Install via: pip install plumed")
            exit()
        self.plumed=plumed
        
        #Store masses
        self.masses=np.array(fragment.list_of_masses,dtype=np.float64)
        
        if colvar_type=="distance" or colvar_type=="bondlength":
            self.colvar_type="DISTANCE"
        elif colvar_type=="torsion" or colvar_type=="dihedral":
            self.colvar_type="TORSION"
        elif colvar_type=="angle":
            self.colvar_type="ANGLE"
        elif colvar_type=="rmsd":
            self.colvar_type="RMSD"
        else:
            print("Specify colvar_type argumentt.")
            print("Options: distance, angle, torsion, rmsd")
            exit()
        #Change 0 to 1 basedindexing and converting to text-string
        self.colvar_indices_string=','.join(map(str, [i+1 for i in colvar_indices]))


        #os.environ["PLUMED_KERNEL"]=path_to_plumed_library
        #p=plumed.Plumed()
        self.plumedobj=self.plumed.Plumed(kernel=path_to_plumed_kernel)
        
        
        #Basic settings
        self.plumedobj.cmd("setMDEngine","python")
        #Timestep needs to be set
        self.plumedobj.cmd("setTimestep", timestep)
        print("timestep:", timestep)
        #Not sure about KbT
        #self.plumedobj.cmd("setKbT", 1.)
        self.plumedobj.cmd("setNatoms",fragment.numatoms)
        self.plumedobj.cmd("setLogFile","test.log")
        
        #Initialize object
        self.plumedobj.cmd("init")



        #Units: length set to Angstrom and time to ps, energy in hartree
        self.plumedobj.cmd("readInputLine","UNITS LENGTH=A TIME=ps ENERGY=Ha")
        
        
        if bias_type == "1D_MTD":
            #height=1.2
            #sigma=0.35
            #biasfactor=6.0
            #1D metadynamics
            self.plumedobj.cmd("readInputLine","d: {} ATOMS={}".format(self.colvar_type, self.colvar_indices_string))
            #p.cmd("readInputLine","RESTRAINT ARG=d AT=0 KAPPA=1")
            self.plumedobj.cmd("readInputLine","METAD LABEL=MTD ARG=d PACE={} HEIGHT={} SIGMA={} FILE={} BIASFACTOR={} TEMP={}".format(pace_num, 
                height, sigma, hills_file, biasfactor, temperature))
            #p.cmd("WALKERS_N=SET_WALKERNUM")
            #p.cmd("WALKERS_ID=SET_WALKERID")
            #p.cmd("WALKERS_DIR=../")
            #p.cmd("WALKERS_RSTRIDE=10")
            #self.plumedobj.cmd("readInputLine","... METAD")
            self.plumedobj.cmd("readInputLine","PRINT STRIDE={} ARG=d,MTD.bias FILE={}".format(stride_num, colvar_file))
        else:
            print("bias_type not implemented")
            exit()
    def run(self, coords=None, forces=None, step=None):
        #box=array.array('d',[10,0,0,0,10,0,0,0,10])
        #virial=array.array('d',[0,0,0,0,0,0,0,0,0])
        #masses=array.array('d',[1,1])
        #NOTE: Only set masses, charges etc. once ?
        #masses=np.array(fragment.list_of_masses,dtype=np.float64)
        #charges=array.array('d',[0,0])
        #forces=array.array('d',[0,0,0,0,0,0])
        #positions=array.array('d',[0,0,0,1,2,3])
        print("plumed run")
        print("coords:", coords)
        print("forces:", forces)
        print("step:", step)
        self.plumedobj.cmd("setStep",step)
        #Setting masses. Must be done after Step
        self.plumedobj.cmd("setMasses", np.array(self.masses))

        #self.plumedobj.cmd("setBox",box )
        
        #self.plumedobj.cmd("setCharges", charges )
        print("here")
        box=np.zeros(9)
        virial=np.zeros(9)
        self.plumedobj.cmd("setBox", box )
        self.plumedobj.cmd("setVirial", virial )

        self.plumedobj.cmd("setPositions", coords )
        self.plumedobj.cmd("setForces", forces )
        
        
        print("Running calc")
        self.plumedobj.cmd("calc")
        print("Calc done")
        bias = np.zeros((1),dtype=np.float64)
        self.plumedobj.cmd("getBias", bias )
        #Initialize bias array
        
        print("forces are now:", forces)
        print("bias:", bias)
        print("virial:", virial)
        print("coords", coords)
        
        energy=999.9999
        #NOTE: Return bias or modified forces?
        
        return energy,forces