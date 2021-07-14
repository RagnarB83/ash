import os
import shutil
import time

from functions.functions_general import BC,blankline,print_time_rel
import modules.module_coords

#Polarizable Embedding theory object.
#Required at init: qm_theory and qmatoms, X, Y
#Currently only Polarizable Embedding (PE). Only available for Psi4, PySCF and Dalton.
#Peatoms: polarizable atoms. MMatoms: nonpolarizable atoms (e.g. TIP3P)
class PolEmbedTheory:
    def __init__(self, fragment=None, qm_theory=None, qmatoms=None, peatoms=None, mmatoms=None, pot_create=True,
                 potfilename='System', pot_option=None, pyframe=False, PElabel_pyframe='MM', daltondir=None, pdbfile=None):

        print(BC.WARNING,BC.BOLD,"------------Defining PolEmbedTheory object-------------", BC.END)
        self.pot_create=pot_create
        self.pyframe=pyframe
        self.pot_option=pot_option
        self.PElabel_pyframe = PElabel_pyframe
        self.potfilename = potfilename
        #Theory level definitions
        allowed_qmtheories=['Psi4Theory', 'PySCFTheory', 'DaltonTheory']
        self.qm_theory=qm_theory
        self.qm_theory_name = self.qm_theory.__class__.__name__
        if self.qm_theory_name in allowed_qmtheories:
            print(BC.OKGREEN, "QM-theory:", self.qm_theory_name, "is supported in Polarizable Embedding", BC.END)
        else:
            print(BC.FAIL, "QM-theory:", self.qm_theory_name, "is  NOT supported in Polarizable Embedding", BC.END)

        if self.pot_option=='LoProp':
            if daltondir is None:
                print("LoProp option chosen. This requires daltondir variable")
                exit()


        if pdbfile is not None:
            print("PDB file provided, will use residue information")

        # Region definitions
        if qmatoms is None:
            self.qmatoms = []
        else:
            self.qmatoms=qmatoms
        if peatoms is None:
            self.peatoms = []
        else:
            self.peatoms=peatoms
        if mmatoms is None:
            print("WARNING...mmatoms list is empty...")
            self.mmatoms = []
        else:
            self.mmatoms=mmatoms

        #If fragment object has been defined
        if fragment is not None:
            self.fragment=fragment
            self.coords=fragment.coords
            self.elems=fragment.elems
            self.connectivity=fragment.connectivity

            self.allatoms = list(range(0, len(self.elems)))
            self.qmcoords=[self.coords[i] for i in self.qmatoms]
            self.qmelems=[self.elems[i] for i in self.qmatoms]
            self.pecoords=[self.coords[i] for i in self.peatoms]
            self.peelems=[self.elems[i] for i in self.peatoms]
            self.mmcoords=[self.coords[i] for i in self.mmatoms]
            self.mmelems=[self.elems[i] for i in self.mmatoms]

            #print("List of all atoms:", self.allatoms)
            print("System size:", len(self.allatoms))
            print("QM region size:", len(self.qmatoms))
            print("PE region size", len(self.peatoms))
            print("MM region size", len(self.mmatoms))
            blankline()

            #Creating list of QM, PE, MM labels used by reading residues in PDB-file
            #Also making residlist necessary
            #TODO: This needs to be rewritten, only applies to water-solvent
            self.hybridatomlabels=[]
            self.residlabels=[]
            count=2
            rescount=0
            for i in self.allatoms:
                if i in self.qmatoms:
                    print("i : {} in qmatoms".format(i))
                    self.hybridatomlabels.append('QM')
                    self.residlabels.append(1)
                elif i in self.peatoms:
                    print("i : {} in peatoms".format(i))
                    self.hybridatomlabels.append(self.PElabel_pyframe)
                    self.residlabels.append(count)
                    rescount+=1
                elif i in self.mmatoms:
                    #print("i : {} in mmatoms".format(i))
                    self.hybridatomlabels.append('WAT')
                    self.residlabels.append(count)
                    rescount+=1
                if rescount==3:
                    count+=1
                    rescount=0

        print("self.hybridatomlabels:", self.hybridatomlabels)
        print("self.residlabels:", self.residlabels)
        #Create Potential file here. Usually true.
        if self.pot_create==True:
            print("Potfile Creation is on!")
            if self.pyframe==True:
                print("Using PyFrame")
                try:
                    import pyframe
                    print("PyFrame found")
                except:
                    print("Pyframe not found. Install pyframe via pip (https://pypi.org/project/PyFraME):")
                    print("pip install pyframe")
                    exit(9)
                #Create dummy pdb-file if PDB-file not provided
                if pdbfile is None:
                    modules.module_coords.write_pdbfile_dummy(self.elems, self.coords, self.potfilename, self.hybridatomlabels, self.residlabels)
                    file=self.potfilename+'.pdb'
                #Pyframe
                if self.pot_option=='SEP':
                    print("Pot option: SEP")
                    system = pyframe.MolecularSystem(input_file=file)
                    solventPol = system.get_fragments_by_name(names=[self.PElabel_pyframe])
                    solventNonPol = system.get_fragments_by_name(names=['WAT'])
                    system.add_region(name='solventpol', fragments=solventPol, use_standard_potentials=True,
                          standard_potential_model='SEP')
                    system.add_region(name='solventnonpol', fragments=solventNonPol, use_standard_potentials=True,
                          standard_potential_model='TIP3P')
                    project = pyframe.Project()
                    project.create_embedding_potential(system)
                    project.write_potential(system)
                    self.potfile=self.potfilename+'.pot'
                    print("Created potfile: ", self.potfile)
                elif self.pot_option=='TIP3P':
                    #Not sure if we use this much or at all. Needs to be checked.
                    print("Pot option: TIP3P")
                    system = pyframe.MolecularSystem(input_file=file)
                    solvent = system.get_fragments_by_name(names=['WAT'])
                    system.add_region(name='solvent', fragments=solvent, use_standard_potentials=True,
                          standard_potential_model='TIP3P')
                    project = pyframe.Project()
                    project.create_embedding_potential(system)
                    project.write_potential(system)
                    self.potfile=self.potfilename+'.pot'
                    print("Created potfile: ", self.potfile)
                #RB. TEST. Protein system using standard potentials
                elif self.pot_option=='Protein-SEP':
                    file=pdbfile
                    print("Pot option: Protein-SEP")
                    exit()
                    system = pyframe.MolecularSystem(input_file=file)
                    Polregion = system.get_fragments_by_name(names=[self.PElabel_pyframe])
                    NonPolregion = system.get_fragments_by_name(names=['WAT'])
                    system.add_region(name='solventpol', fragments=solventPol, use_standard_potentials=True,
                          standard_potential_model='SEP')
                    system.add_region(name='solventnonpol', fragments=solventNonPol, use_standard_potentials=True,
                          standard_potential_model='TIP3P')
                    project = pyframe.Project()
                    project.create_embedding_potential(system)
                    project.write_potential(system)
                    self.potfile=self.potfilename+'.pot'
                    print("Created potfile: ", self.potfile)

                elif self.pot_option=='LoProp':
                    print("Pot option: LoProp")
                    print("Note: dalton and loprop binaries need to be in shell PATH before running.")
                    #os.environ['PATH'] = daltondir + ':'+os.environ['PATH']
                    #print("Current PATH is:", os.environ['PATH'])
                    #TODO: Create pot file from scratch. Requires LoProp and Dalton I guess
                    system = pyframe.MolecularSystem(input_file=file)
                    core = system.get_fragments_by_name(names=['QM'])
                    system.set_core_region(fragments=core, program='Dalton', basis='pcset-1')
                    # solvent = system.get_fragments_by_distance(reference=core, distance=4.0)
                    solvent = system.get_fragments_by_name(names=[self.PElabel_pyframe])
                    system.add_region(name='solvent', fragments=solvent, use_mfcc=True, use_multipoles=True, 
                                      multipole_order=2, multipole_model='LoProp', multipole_method='DFT', multipole_xcfun='PBE0',
                                      multipole_basis='loprop-6-31+G*', use_polarizabilities=True, polarizability_model='LoProp',
                                      polarizability_method='DFT', polarizability_xcfun='PBE0', polarizability_basis='loprop-6-31+G*')
                    project = pyframe.Project()
                    print("Creating embedding potential")
                    project.create_embedding_potential(system)
                    project.write_core(system)
                    project.write_potential(system)
                    self.potfile=self.potfilename+'.pot'
                    print("Created potfile (via Dalton and LoProp): ", self.potfile)
                else:
                    print("Invalid option")
                    exit()
                #Copying pyframe-created potfile from dir:
                shutil.copyfile(self.potfilename+'/' + self.potfilename+'.pot', './'+self.potfilename+'.pot')

            #Todo: Manual potential file creation. Maybe only if pyframe is buggy
            else:
                print("Manual potential file creation (instead of Pyframe)")
                print("Not ready yet!")
                if self.pot_option == 'SEP':
                    numatomsolvent = 3
                    Ocharge = -0.67444000
                    Hcharge = 0.33722000
                    Opolz = 5.73935000
                    Hpolz = 2.30839000
                    numpeatoms=len(self.peatoms)
                    with open('System' + '.pot', 'w') as potfile:
                        potfile.write('! Generated by Pot-Gen-RB\n')
                        potfile.write('@COORDINATES\n')
                        potfile.write(str(numpeatoms) + '\n')
                        potfile.write('AA\n')
                        for i in range(0, numpeatoms):
                            atom = self.peatoms[i]
                            c = self.pecoords[i]
                            potfile.write(
                                atom + '   ' + str(c[0]) + '   ' + str(c[1]) + '   ' + str(c[2]) + '   ' + str(
                                    i + 1) + '\n')
                        potfile.write('@MULTIPOLES\n')
                        # Assuming simple pointcharge here. To be extended
                        potfile.write('ORDER 0\n')
                        potfile.write(str(numpeatoms) + '\n')
                        for i in range(0, numpeatoms):
                            atom = self.peatoms[i]
                            if atom == 'O':
                                SPcharge = Ocharge
                            elif atom == 'H':
                                SPcharge = Hcharge
                            potfile.write(str(i + 1) + '   ' + str(SPcharge) + '\n')
                        potfile.write('@POLARIZABILITIES\n')
                        potfile.write('ORDER 1 1\n')
                        potfile.write(str(numpeatoms) + '\n')
                        for i in range(0, numpeatoms):
                            atom = self.peatoms[i]
                            if atom == 'O':
                                SPpolz = Opolz
                            elif atom == 'H':
                                SPpolz = Hpolz
                            potfile.write(str(i + 1) + '    ' + str(SPpolz) + '   0.0000000' + '   0.0000000    ' + str(
                                SPpolz) + '   0.0000000    ' + str(SPpolz) + '\n')
                        potfile.write('EXCLISTS\n')
                        potfile.write(str(numpeatoms) + ' 3\n')
                        for j in range(1, numpeatoms, numatomsolvent):
                            potfile.write(str(j) + ' ' + str(j + 1) + ' ' + str(j + 2) + '\n')
                            potfile.write(str(j + 1) + ' ' + str(j) + ' ' + str(j + 2) + '\n')
                            potfile.write(str(j + 2) + ' ' + str(j) + ' ' + str(j + 1) + '\n')

                else:
                    print("Other pot options not yet available")
                    exit()
        else:
            print("Pot creation is off for this object. Assuming potfile has been provided")
            self.potfile=potfilename+'.pot'
        print_time_rel(module_init_time, modulename='PolEmbedTheory creation')
    def run(self, current_coords=None, elems=None, Grad=False, nprocs=1, potfile=None, restart=False):
    
        module_init_time=time.time()

        print(BC.WARNING, BC.BOLD, "------------RUNNING PolEmbedTheory MODULE-------------", BC.END)
        if restart==True:
            print("Restart Option On!")
        else:
            print("Restart Option Off!")
        print("QM Module:", self.qm_theory_name)

        #Check if potfile provide to run (rare use). If not, use object file
        if potfile is not None:
            self.potfile=potfile

        print("Using potfile:", self.potfile)

        #If no coords provided to run (from Optimizer or NumFreq or MD) then use coords associated with object.
        #if len(current_coords) != 0:
        if current_coords is not None:
            pass
        else:
            current_coords=self.coords

        #Updating QM coords and MM coords.
        #TODO: Should we use different name for updated QMcoords and MMcoords here??
        self.qmcoords=[current_coords[i] for i in self.qmatoms]

        if self.qm_theory_name == "Psi4Theory":
            #Calling Psi4 theory, providing current QM and MM coordinates.
            #Currently doing SP case only without Grad

            self.QMEnergy = self.qm_theory.run(current_coords=self.qmcoords, qm_elems=self.qmelems, Grad=False,
                                               nprocs=nprocs, pe=True, potfile=self.potfile, restart=restart)
        elif self.qm_theory_name == "DaltonTheory":
            print("self.potfile:", self.potfile)
            self.QMEnergy = self.qm_theory.run(current_coords=self.qmcoords, qm_elems=self.qmelems, Grad=False,
                                               nprocs=nprocs, pe=True, potfile=self.potfile, restart=restart)
        elif self.qm_theory_name == "PySCFTheory":
            self.QMEnergy = self.qm_theory.run(current_coords=self.qmcoords, qm_elems=self.qmelems, Grad=False,
                                               nprocs=nprocs, pe=True, potfile=self.potfile, restart=restart)

        elif self.qm_theory_name == "ORCATheory":
            print("not available for ORCATheory")
            exit()
        else:
            print("invalid QM theory")
            exit()

        #Todo: self.MM_Energy from PolEmbed calc?
        self.MMEnergy=0
        #Final QM/MM Energy
        self.PolEmbedEnergy = self.QMEnergy+self.MMEnergy
        self.energy=self.PolEmbedEnergy
        blankline()
        print("{:<20} {:>20.12f}".format("QM energy: ",self.QMEnergy))
        print("{:<20} {:>20.12f}".format("MM energy: ", self.MMEnergy))
        print("{:<20} {:>20.12f}".format("PolEmbed energy: ", self.PolEmbedEnergy))
        blankline()
        print_time_rel(module_init_time, modulename='PolEmbedTheory run', moduleindex=2)
        return self.PolEmbedEnergy
