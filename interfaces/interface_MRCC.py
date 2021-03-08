#MRCC interface. not ready

#MRCC Theory object. Fragment object is optional. Used??
class MRCCTheory:
    def __init__(self, mrccdir=None, fragment=None, charge=None, mult=None, printlevel=2, cfourbasis=None, cfourmethod=None,
                mrccmemory=3100, nprocs=1):

        #Printlevel
        self.printlevel=printlevel

        self.charge=charge
        self.mult=mult
        self.cfourbasis=cfourbasis
        self.cfourmethod=cfourmethod
        self.mrccmemory=mrccmemory
        self.nprocs=nprocs

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, nprocs=None, restart=False):

        if nprocs == None:
            nprocs = self.nprocs

        print(BC.OKBLUE, BC.BOLD, "------------RUNNING MRCC INTERFACE-------------", BC.END)


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


        def write_cfour_input(method,basis,reference,charge,mult,frozencore,memory):
            with open("ZMAT", 'w') as inpfile:
                inpfile.write('ASH-created inputfile\n')
                for el,c in zip(elems,qm_elems):
                    inpfile.write('{} {} {} {}\n'.format(el,c[0],c[1],c[2]))
                inpfile.write('\n')
                inpfile.write('*CFOUR(CALC={},BASIS={},COORD=CARTESIAN,REF={},CHARGE={},MULT={},FROZEN_CORE={},GEO_MAXCYC=1,MEM_UNIT=MB,MEMORY={})\n'.format(
                    method,basis,reference,charge,mult,frozencore,memory))

        def run_cfour(cfourdir):
            fdg="dsgfs"


            #Grab energy and gradient
            #TODO: No qm/MM yet. need to check if possible in MRCC
            if Grad==True:

                write_cfour_input(self.method,self.basis,self.reference,self.charge,self.mult,self.frozen_core,self.memory)
                run_cfour(self.cfourdir)
                self.energy = 0.0
                #self.gradient = X
            else:
                write_cfour_input(self.method,self.basis,self.reference,self.charge,self.mult,self.frozen_core,self.memory)
                run_cfour(self.cfourdir)
                self.energy = 0.0

            #TODO: write in error handling here
            print(BC.OKBLUE, BC.BOLD, "------------ENDING MRCC INTERFACE-------------", BC.END)
            if Grad == True:
                print("Single-point MRCC energy:", self.energy)
                return self.energy, self.gradient
            else:
                print("Single-point MRCC energy:", self.energy)
                return self.energy

