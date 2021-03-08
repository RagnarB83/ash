import numpy as np
import shutil
import os
import subprocess as sp
import glob

import module_coords
from functions_general import BC


#Psi4 Theory object. Fragment object is optional. Only used for single-points.
#PSI4 runmode:
#   : library means that ASH will load Psi4 libraries and run psi4 directly
#   : inputfile means that ASH will create Psi4 inputfile and run a separate psi4 executable
#psi4dir only necessary for inputfile-based userinterface. Todo: Possibly unnexessary
#printsetting is by default set to 'File. Change to something else for stdout print
# PE: Polarizable embedding (CPPE). Pass pe_modulesettings dict as well
class Psi4Theory:
    def __init__(self, fragment=None, charge=None, mult=None, printsetting='False', psi4settings=None, psi4method=None,
                 runmode='library', psi4dir=None, pe=False, potfile='', outputname='psi4output.dat', label='psi4input',
                 psi4memory=3000, nprocs=1, printlevel=2,fchkwrite=False):

        #Printlevel
        self.printlevel=printlevel


        self.nprocs=nprocs
        self.psi4memory=psi4memory
        self.label=label
        self.outputname=outputname
        self.printsetting=printsetting
        self.runmode=runmode
        #CPPE Polarizable Embedding options
        self.pe=pe
        #Write fchk wavefunction file. Can be read by Multiwfn
        self.fchkwrite=fchkwrite
        #Potfile from user or passed on via QM/MM Theory object ?
        self.potfile=potfile
        #Determining runmode
        if self.runmode != 'library':
            print("Defining Psi4 object with runmode=psithon")
            if psi4dir is not None:
                print("Path to Psi4 provided:", psi4dir)
                self.psi4path=psi4dir
            else:
                self.psi4path=shutil.which('psi4')
                if self.psi4path==None:
                    print("Found no psi4 in path. Add Psi4 to Shell environment or provide psi4dir variable")
                    exit()
                else:
                    print("Found psi4 in path:", self.psi4path)

        #Checking if method is defined
        if psi4method == None:
            print("psi4method not set. Exiting")
            exit()
        if psi4settings == None:
            print("psi4settings dict not set. Exiting")
            exit()
        #All valid Psi4 methods that can be arguments in energy() function
        self.psi4method=psi4method
        #Settings dict
        self.psi4settings=psi4settings

        if fragment is not None:
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
        #print("frag elems", self.fragment.elems)
        if charge is not None:
            self.charge=int(charge)
        if mult is not None:
            self.mult=int(mult)


        #DFT-specific. Remove? Marked for deletion
        #self.psi4functional=psi4functional



    #Cleanup after run.
    def cleanup(self):
        print("Cleaning up old Psi4 files")
        try:
            os.remove('timer.dat')
            os.remove('psi4output.dat')
        except:
            pass
    #Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, PC=False, nprocs=None, pe=False, potfile='', restart=False ):

        if nprocs==None:
            nprocs=self.nprocs

        print(BC.OKBLUE,BC.BOLD, "------------RUNNING PSI4 INTERFACE-------------", BC.END)

        #If pe and potfile given as run argument
        if pe is not False:
            self.pe=pe
        if potfile != '':
            self.potfile=potfile

        #Coords provided to run or else taken from initialization.
        #if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords=self.coords

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list or self.elems
        if qm_elems is None:
            if elems is None:
                qm_elems=self.elems
            else:
                qm_elems = elems

        #PSI4 runmode:
        #   : library means that ASH will load Psi4 libraries and run psi4 directly
        #   : inputfile means that ASH will create Psi4 inputfile and run a separate psi4 executable

        if self.runmode=='library':
            print("Psi4 Runmode: Library")
            try:
                import psi4
            except:
                print(BC.FAIL,"Problem importing psi4. Make sure psi4 has been installed as part of same Python as ASH", BC.END)
                print(BC.WARNING,"If problematic, switch to inputfile based Psi4 interface instead.", BC.END)
                exit(9)
            #Changing namespace may prevent crashes due to multiple jobs running at same time
            if self.label=='label':
                psi4.core.IO.set_default_namespace("psi4job_ygg")
            else:
                psi4.core.IO.set_default_namespace(self.label)

            #Printing to stdout or not:
            if self.printsetting:
                print("Printsetting = True. Printing output to stdout...")
            else:
                print("Printsetting = False. Printing output to file: {}) ".format(self.outputname))
                psi4.core.set_output_file(self.outputname, False)

            #Psi4 scratch dir
            print("Setting Psi4 scratchdir to ", os.getcwd())
            psi4_io = psi4.core.IOManager.shared_object()
            psi4_io.set_default_path(os.getcwd())

            #Creating Psi4 molecule object using lists and manual information
            psi4molfrag = psi4.core.Molecule.from_arrays(
                elez=module_coords.elemstonuccharges(qm_elems),
                fix_com=True,
                fix_orientation=True,
                fix_symmetry='c1',
                molecular_charge=self.charge,
                molecular_multiplicity=self.mult,
                geom=current_coords)
            psi4.activate(psi4molfrag)

            #Adding MM charges as pointcharges if PC=True
            #Might be easier to use PE and potfile ??
            if PC==True:
                #Chargefield = psi4.QMMM()
                Chargefield = psi4.core.ExternalPotential()
                #Mmcoords seems to be in Angstrom
                for mmcharge,mmcoord in zip(MMcharges,current_MM_coords):
                    Chargefield.addCharge(mmcharge, mmcoord[0], mmcoord[1], mmcoord[2])
                psi4.core.set_global_option("EXTERN", True)
                psi4.core.EXTERN = Chargefield

            #Setting inputvariables
            print("Psi4 memory (MB): ", self.psi4memory)

            psi4.set_memory(str(self.psi4memory)+' MB')

            #Changing charge and multiplicity
            #psi4molfrag.set_molecular_charge(self.charge)
            #psi4molfrag.set_multiplicity(self.mult)

            #Setting RKS or UKS reference
            #For now, RKS always if mult 1 Todo: Make more flexible
            if self.mult == 1:
                self.psi4settings['reference'] = 'RHF'
            else:
                self.psi4settings['reference'] = 'UHF'

            #Controlling orbital read-in guess.
            if restart==True:
                self.psi4settings['guess'] = 'read'
                #Renameing orbital file
                PID = str(os.getpid())
                print("Restart Option On!")
                print("Renaming lastrestart.180 to {}".format(os.path.splitext( self.outputname)[0] + '.default.' + PID + '.180.npy'))
                os.rename('lastrestart.180', os.path.splitext( self.outputname)[0] + '.default.' + PID + '.180.npy')
            else:
                self.psi4settings['guess'] = 'sad'

            #Reading dict object with basic settings and passing to Psi4
            psi4.set_options(self.psi4settings)
            print("Psi4 settings:", self.psi4settings)

            #Reading module options dict and passing to Psi4
            #TODO: Make one for SCF, CC, PCM etc.
            #psi4.set_module_options(modulename, moduledict)

            #Reading PE module options if PE=True
            if self.pe==True:
                print(BC.OKGREEN,"Polarizable Embedding Option On! Using CPPE module inside Psi4", BC.END)
                print(BC.WARNING, "Potfile: ", self.potfile, BC.END)
                try:
                    if os.path.exists(self.potfile):
                        pass
                    else:
                        print(BC.FAIL, "Potfile: ", self.potfile, "does not exist!", BC.END)
                        exit()
                except:
                    exit()
                psi4.set_module_options('pe', {'potfile' : self.potfile})
                self.psi4settings['pe'] = 'true'

            #Controlling OpenMP parallelization. Controlled here, not via OMP_NUM_THREADS etc.
            psi4.set_num_threads(nprocs)

            #Namespace issue overlap integrals requires this when running with multiprocessing:
            # http://forum.psicode.org/t/wfn-form-h-errors/1304/2
            #psi4.core.clean()

            #Running energy or energy+gradient. Currently hardcoded to SCF-DFT jobs

            #TODO: Support pointcharges and PE embedding in Grad job?
            if Grad==True:
                print("Running gradient with Psi4 method:", self.psi4method)
                #grad=psi4.gradient('scf', dft_functional=self.psi4functional)
                grad=psi4.gradient(self.psi4method)
                self.gradient=np.array(grad)
                self.energy = psi4.variable("CURRENT ENERGY")
            else:
                #This might be unnecessary as I think all DFT functionals work as keyword to energy function. Hence psi4method works for all
                #self.energy = psi4.energy('scf', dft_functional=self.psi4functional)
                print("Running energy with Psi4 method:", self.psi4method)
                self.energy = psi4.energy(self.psi4method)
            #Keep restart file 180 as lastrestart.180
            PID = str(os.getpid())
            try:
                print("Renaming {} to lastrestart.180".format(os.path.splitext(self.outputname)[0]+'.default.'+PID+'.180.npy'))
                os.rename(os.path.splitext(self.outputname)[0]+'.default.'+PID+'.180.npy', 'lastrestart.180')
            except:
                pass

            #TODO: write in error handling here

            print(BC.OKBLUE, BC.BOLD, "------------ENDING PSI4-INTERFACE-------------", BC.END)

            if Grad == True:
                print("Single-point PSI4 energy:", self.energy)
                return self.energy, self.gradient
            else:
                print("Single-point PSI4 energy:", self.energy)
                return self.energy

        #Psithon INPUT-FILE BASED INTERFACE. Creates Psi4 inputfiles and runs Psithon as subprocessses
        elif self.runmode=='psithon':
            print("Psi4 Runmode: Psithon")
            print("Current directory:", os.getcwd())
            #Psi4 scratch dir
            #print("Setting Psi4 scratchdir to ", os.getcwd())
            #Possible option: Set scratch env-variable as subprocess??? TODO:
            #export PSI_SCRATCH=/path/to/existing/writable/local-not-network/directory/for/scratch/files
            #Better :
            #psi4_io.set_default_path('/scratch/user')
            #Setting inputvariables

            print("Psi4 Memory:", self.psi4memory)

            #Printing Psi4settings
            print("Psi4 method:", self.psi4method)
            print("Psi4 settings:", self.psi4settings)

            #Printing PE options and checking for ptfile
            if self.pe==True:
                print(BC.OKGREEN,"Polarizable Embedding Option On! Using CPPE module inside Psi4", BC.END)
                print(BC.WARNING, "Potfile: ", self.potfile, BC.END)
                try:
                    if os.path.exists(self.potfile):
                        pass
                    else:
                        print(BC.FAIL, "Potfile: ", self.potfile, "does not exist!", BC.END)
                        exit()
                except:
                    exit()

            #Write inputfile
            with open(self.label+'.inp', 'w') as inputfile:
                inputfile.write('psi4_io.set_default_path(\'{}\')\n'.format(os.getcwd()))
                inputfile.write('memory {} MB\n'.format(self.psi4memory))
                inputfile.write('molecule molfrag {\n')
                inputfile.write(str(self.charge)+' '+str(self.mult)+'\n')
                for el,c in zip(qm_elems, current_coords):
                    inputfile.write(el+' '+str(c[0])+' '+str(c[1])+' '+str(c[2])+'\n')
                inputfile.write('symmetry c1\n')
                inputfile.write('no_reorient\n')
                inputfile.write('no_com\n')
                inputfile.write('}\n')
                inputfile.write('\n')

                # Adding MM charges as pointcharges if PC=True
                # Might be easier to use PE and potfile ??
                if PC == True:
                    inputfile.write('Chrgfield = QMMM()\n')
                    # Mmcoords in Angstrom
                    for mmcharge, mmcoord in zip(MMcharges, current_MM_coords):
                        inputfile.write('Chrgfield.extern.addCharge({}, {}, {}, {})\n'.format(mmcharge, mmcoord[0], mmcoord[1], mmcoord[2]))
                    inputfile.write('psi4.set_global_option_python(\'EXTERN\', Chrgfield.extern)\n')
                inputfile.write('\n')
                #Adding Psi4 settings
                inputfile.write('set {\n')
                for key,val in self.psi4settings.items():
                    inputfile.write(key+' '+val+'\n')
                #Setting RKS or UKS reference. For now, RKS always if mult 1 Todo: Make more flexible
                if self.mult == 1:
                    self.psi4settings['reference'] = 'RHF'
                else:
                    inputfile.write('reference UHF \n')
                #Orbital guess
                if restart == True:
                    inputfile.write('guess read \n')
                else:
                    inputfile.write('guess sad \n')
                #PE
                if self.pe == True:
                    inputfile.write('pe true \n')
                #end
                inputfile.write('}\n')

                if self.pe==True:
                    inputfile.write('set pe { \n')
                    inputfile.write(' potfile {} \n'.format(self.potfile))
                    inputfile.write('}\n')

                #Writing job directive
                inputfile.write('\n')

                if restart==True:
                    #function add .npy extension to lastrestart.180
                    inputfile.write('wfn = core.Wavefunction.from_file(\'{}\')\n'.format('lastrestart.180'))
                    inputfile.write('newfile = wfn.get_scratch_filename(180)\n')
                    inputfile.write('wfn.to_file(newfile)\n')
                    inputfile.write('\n')

                #RUNNING
                if Grad==True:
                    #inputfile.write('scf_energy, wfn = gradient(\'scf\', dft_functional=\'{}\', return_wfn=True)\n'.format(self.psi4functional))
                    inputfile.write("energy, wfn = gradient(\'{}\', return_wfn=True)\n".format(self.psi4method))
                    inputfile.write("print(\"FINAL TOTAL ENERGY :\", wfn.energy())")
                else:
                    #inputfile.write('scf_energy, wfn = energy(\'scf\', dft_functional=\'{}\', return_wfn=True)\n'.format(self.psi4functional))
                    inputfile.write('energy, wfn = energy(\'{}\', return_wfn=True)\n'.format(self.psi4method))
                    inputfile.write("print(\"FINAL TOTAL ENERGY :\", energy)")
                    inputfile.write('\n')
                #Fchk write or not
                if self.fchkwrite == True:
                    inputfile.write('#Fchk write\n')
                    inputfile.write('fchk_writer = psi4.FCHKWriter(wfn)\n')
                    inputfile.write('fchk_writer.write(\'{}.fchk\')\n'.format(self.label))
        


            print("Running inputfile:", self.label+'.inp')
            #Running inputfile
            with open(self.label + '.out', 'w') as ofile:
                #Psi4 -m option for saving 180 file
                print("nprocs:", nprocs)
                process = sp.run(['psi4', '-m', '-i', self.label + '.inp', '-o', self.label + '.out', '-n', '{}'.format(str(nprocs)) ], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

            #Keep restart file 180 as lastrestart.180
            try:
                restartfile=glob.glob(self.label+'*180.npy')[0]
                print("restartfile:", restartfile)
                print("Psi4 Done. Renaming {} to lastrestart.180.npy".format(restartfile))
                os.rename(restartfile, 'lastrestart.180.npy')
            except:
                pass

            #Delete big WF files. Todo: move to cleanup function?
            wffiles=glob.glob('*.34')
            for wffile in wffiles:
                os.remove(wffile)

            #Grab energy and possibly gradient
            self.energy, self.gradient = grabPsi4EandG(self.label + '.out', len(qm_elems), Grad)

            #TODO: write in error handling here

            print(BC.OKBLUE, BC.BOLD, "------------ENDING PSI4-INTERFACE-------------", BC.END)

            if Grad == True:
                print("Single-point PSI4 energy:", self.energy)
                return self.energy, self.gradient
            else:
                print("Single-point PSI4 energy:", self.energy)
                return self.energy
        else:
            print("Unknown Psi4 runmode")
            exit()




#Psi4 interface.
# Inputfile version.
# Todo: replace with python-interface version

#create inputfile

#run Psi4 input


#check if Psi4 finished

#grab energy from output

#Slightly ugly
#Now grabbing specially printed line (FINAL TOTAL ENERGY). Can be either SCF energy or coupled cluster etc.
def grabPsi4EandG(outfile, numatoms, Grad):
    energy=None
    gradient = np.zeros((numatoms, 3))
    row=0
    gradgrab=False
    with open(outfile) as ofile:
        for line in ofile:
            if 'FINAL TOTAL ENERGY' in line:
                if 'print' not in line:
                    energy = float(line.split()[-1])
            if Grad == True:
                if gradgrab==True:
                    if len(line) < 2:
                        gradgrab=False
                        break
                    if '--' not in line:
                        if 'Atom' not in line:
                            val=line.split()
                            gradient[row] = [float(val[1]),float(val[2]),float(val[3])]
                            row+=1
                #SCF case
                if '  -Total Gradient:' in line:
                    gradgrab = True
                #CC case
                if '-Total gradient:' in line:
                    gradgrab = True
    if energy == None:
        print("Found no energy in Psi4 outputfile:", outfile)
        exit()
    return energy, gradient
