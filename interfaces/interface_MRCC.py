from functions_general import BC
import subprocess as sp
#MRCC interface. not ready

#MRCC Theory object. Fragment object is optional. Used??
class MRCCTheory:
    def __init__(self, mrccdir=None, fragment=None, charge=None, mult=None, printlevel=2,
                mrccinput=None, nprocs=1):

        #Printlevel
        self.printlevel=printlevel
        self.filename="mrcc"
        self.mrccdir=mrccdir
        self.charge=charge
        self.mult=mult
        self.mrccinput=mrccinput
        self.nprocs=nprocs


    #TODO: Parallelization is enabled most easily by OMP_NUM_THREADS AND MKL_NUM_THREADS. NOt sure if we can control this here



    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, nprocs=None, restart=False, label=None):

        if nprocs == None:
            nprocs = self.nprocs

        print(BC.OKBLUE, BC.BOLD, "------------RUNNING MRCC INTERFACE-------------", BC.END)

        print("Running MRCC object. Will use threads if OMP_NUM_THREADS and MKL_NUM_THREAD environment variables")
        print("Job label:", label)
        print("Creating inputfile: MINP")
        print("MRCC input:")
        print(self.mrccinput)

        # Coords provided to run or else taken from initialization.
        # if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords = self.coords

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems = self.elems
            else:
                qm_elems = elems

        #Grab energy and gradient
        #TODO: No qm/MM yet. need to check if possible in MRCC
        if Grad==True:
            print("Grad not ready")
            exit()
            write_mrcc_input(self.mrccinput,self.charge,self.mult,qm_elems,current_coords)
            run_mrcc(self.mrccdir,self.filename+'.out')
            self.energy=grab_energy_mrcc(self.filename+'.out')
            self.gradient = grab_gradient_mrcc()
        else:
            write_mrcc_input(self.mrccinput,self.charge,self.mult,qm_elems,current_coords)
            run_mrcc(self.mrccdir,self.filename+'.out')
            self.energy=grab_energy_mrcc(self.filename+'.out')

        #TODO: write in error handling here
        print(BC.OKBLUE, BC.BOLD, "------------ENDING MRCC INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point MRCC energy:", self.energy)
            return self.energy, self.gradient
        else:
            print("Single-point MRCC energy:", self.energy)
            return self.energy




def run_mrcc(mrccdir,filename):
    with open(filename, 'w') as ofile:
        process = sp.run([mrccdir + '/dmrcc'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

#TODO: Gradient option
def write_mrcc_input(mrccinput,charge,mult,elems,coords):
    with open("MINP", 'w') as inpfile:
        inpfile.write(mrccinput + '\n')
        inpfile.write('unit=angs\n')
        inpfile.write('charge={}\n'.format(charge))
        inpfile.write('mult={}\n'.format(mult))
        #inpfile.write('dens=2\n')
        inpfile.write('geom=xyz\n')
        inpfile.write('{}\n'.format(len(elems)))
        inpfile.write('\n')
        for el,c in zip(elems,coords):
            inpfile.write('{}   {} {} {}\n'.format(el,c[0],c[1],c[2]))
        inpfile.write('\n')

def grab_energy_mrcc(outfile):
    #Option 1. Grabbing all lines containing energy in outputfile. Take last entry.
    # CURRENT Option 2: grab energy from iface file. Higher level WF entry should be last
    with open("iface") as g:
        for line in f:
            if 'ENERGY' in line:
                energy=float(line.split()[5])
    
    #linetograb="energy"
    #with open(outfile) as f:
    #    for line in f:
    #        if linetograb.upper() in line.upper():
    #            final=line
    #print(final)
    #try:
    #    energy=float(final.split()[-1])
    #except:
    #    print("Problem reading energy from MRCC outputfile. Check:", outfile)
    #    exit()
    return energy


def grab_gradient_mrcc(self):
    pass
   # atomcount=0
   # with open('GRD') as grdfile:
   #     for i,line in enumerate(grdfile):
   #         if i==0:
   #             numatoms=int(line.split()[0])
   #             gradient=np.zeros((numatoms,3))
   #         if i>numatoms:
   #             gradient[atomcount,0] = float(line.split()[1])
   #             gradient[atomcount,1] = float(line.split()[2])
   #              gradient[atomcount,2] = float(line.split()[3])
   #             atomcount+=1
   # return gradient    