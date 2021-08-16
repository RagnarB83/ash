import numpy as np
#Interface to Plumed


#PLUMED_ASH class

class plumed_ASH():
    def __init__(self, path_to_plumed_kernel=None, bias_type="1D_MTD", fragment=None, theory=None, colvar_type=None, colvar_indices=None,
               temperature=300.0, hills_file="HILLS", colvar_file="COLVAR", height=None, sigma=None, biasfactor=None,
               stride_num=10, pace_num=500):
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
        #Atom indices to text-string
        self.colvar_indices_string=','.join(map(str, colvar_indices))


        #os.environ["PLUMED_KERNEL"]=path_to_plumed_library
        #p=plumed.Plumed()
        self.plumedobj=self.plumed.Plumed(kernel=path_to_plumed_kernel)
        
        
        #Basic settings
        self.plumedobj.cmd("setMDEngine","python")
        #Needed?
        #self.plumedobj.cmd("setTimestep", 1.)
        #self.plumedobj.cmd("setKbT", 1.)
        self.plumedobj.cmd("setNatoms",fragment.numatoms)
        self.plumedobj.cmd("setLogFile","test.log")
        self.plumedobj.cmd("init")
        
        #Units: length set to Angstrom and time to ps, energy in hartree
        self.plumedobj.cmd("UNITS LENGTH=A TIME=ps ENERGY=Ha")
        
        
        if bias_type == "1D_MTD":
            #height=1.2
            #sigma=0.35
            #biasfactor=6.0
            #1D metadynamics
            self.plumedobj.cmd("readInputLine","d: {} ATOMS={}".format(self.colvar_type, self.colvar_indices_string))
            self.plumedobj.cmd("")
            #p.cmd("readInputLine","RESTRAINT ARG=d AT=0 KAPPA=1")
            self.plumedobj.cmd("METAD ...")
            self.plumedobj.cmd("LABEL=MTD")
            self.plumedobj.cmd("ARG=dihed PACE={} HEIGHT={} SIGMA={} FILE={} BIASFACTOR={} TEMP={}".format(pace_num, height, sigma, hills_file, biasfactor, temperature))
            #p.cmd("WALKERS_N=SET_WALKERNUM")
            #p.cmd("WALKERS_ID=SET_WALKERID")
            #p.cmd("WALKERS_DIR=../")
            #p.cmd("WALKERS_RSTRIDE=10")
            self.plumedobj.cmd("... METAD")
            self.plumedobj.cmd("PRINT STRIDE={} ARG=d,MTD.bias FILE={}".format(stride_num, colvar_file))
        else:
            print("bias_type not implemented")
            exit()
    def run(self, coords=None, charges=None, forces=None, masses=None):
        #box=array.array('d',[10,0,0,0,10,0,0,0,10])
        #virial=array.array('d',[0,0,0,0,0,0,0,0,0])
        #masses=array.array('d',[1,1])
        #NOTE: Only set masses, charges etc. once ?
        masses=np.array(fragment.list_of_masses,dtype=np.float64)
        charges=array.array('d',[0,0])
        forces=array.array('d',[0,0,0,0,0,0])
        positions=array.array('d',[0,0,0,1,2,3])

        self.plumedobj.cmd("setStep",0)
        #self.plumedobj.cmd("setBox",box )
        self.plumedobj.cmd("setMasses", masses )
        self.plumedobj.cmd("setCharges", charges )
        self.plumedobj.cmd("setPositions", positions )
        self.plumedobj.cmd("setForces", forces )
        #p.cmd("setVirial", virial )
        
        self.plumedobj.cmd("calc")
        
        #Initialize bias array
        bias = np.zeros((1),dtype=np.float64)
        self.plumedobj.cmd("getBias", bias )
        bias=array.array('d',[0])
        
        print("forces are now:", forces)
        print("bias:", bias)
        
        exit()
        #NOTE: Return bias or modified forces?
        
        return bias