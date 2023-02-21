#Module for calculating PhotoElectron/PhotoIonization Spectra
#####################################
import numpy as np
import os
import sys
import subprocess as sp
import struct
import copy
import time
import shutil
import math

#import ash
from ash.interfaces.interface_ORCA import scfenergygrab,tddftgrab,orbitalgrab,run_orca_plot,grabEOMIPs,check_stability_in_output, ORCATheory
from ash.functions.functions_general import ashexit, writestringtofile,BC,blankline,isint,islist,print_time_rel,print_line_with_mainheader,find_between
from ash.functions.functions_elstructure import modosplot,write_cube_diff,read_cube
import ash.constants

#Wrapper function around PhotoElectron()
def PhotoElectron(theory=None, fragment=None, numcores=1, memory=40000,label=None, 
                        Initialstate_charge=None, Initialstate_mult=None,
                        Ionizedstate_charge=None, Ionizedstate_mult=None, numionstates=None, 
                        initialorbitalfiles=None, densities='None', densgridvalue=100,
                        CAS=False, MRCI=False, MREOM=False, TDDFT=False,
                        tda=True,brokensym=False, HSmult=None, atomstoflip=None, check_stability=True,
                        CASCI=False, CAS_Initial=None, CAS_Final = None,
                        MRCI_Initial=None, MRCI_Final = None, MRCI_CASCI_Final=True,
                        EOM=False, btPNO=False, DLPNO=False, 
                        path_wfoverlap=None, tprintwfvalue=1e-6, noDyson=False):
    """
    Wrapper function around PhotoElectron Class
    """
    print_line_with_mainheader("PhotoElectron")
    timeA=time.time()
    photo=PhotoElectronClass(theory=theory, fragment=fragment, numcores=numcores, memory=memory,label=label, 
                        Initialstate_charge=Initialstate_charge, Initialstate_mult=Initialstate_mult,
                        Ionizedstate_charge=Ionizedstate_charge, Ionizedstate_mult=Ionizedstate_mult, numionstates=numionstates, 
                        initialorbitalfiles=initialorbitalfiles, densities=densities, densgridvalue=densgridvalue,
                        CAS=CAS, MRCI=MRCI, MREOM=MREOM, TDDFT=TDDFT,
                        tda=tda,brokensym=brokensym, HSmult=HSmult, atomstoflip=atomstoflip, check_stability=check_stability,
                        CASCI=CASCI, CAS_Initial=CAS_Initial, CAS_Final = CAS_Final,
                        MRCI_Initial=MRCI_Initial, MRCI_Final = MRCI_Final, MRCI_CASCI_Final=MRCI_CASCI_Final,
                        EOM=EOM, btPNO=btPNO, DLPNO=DLPNO, 
                        path_wfoverlap=path_wfoverlap, tprintwfvalue=tprintwfvalue, noDyson=noDyson)
    result = photo.run()
    print_time_rel(timeA, modulename='PhotoElectron', moduleindex=1)
    return result



class PhotoElectronClass:
    def __init__(self,theory=None, fragment=None, numcores=1, memory=40000,label=None, 
                        Initialstate_charge=None, Initialstate_mult=None,
                        Ionizedstate_charge=None, Ionizedstate_mult=None, numionstates=None, 
                        initialorbitalfiles=None, densities='None', densgridvalue=100,
                        CAS=False, MRCI=False, MREOM=False, TDDFT=False,
                        tda=True,brokensym=False, HSmult=None, atomstoflip=None, check_stability=True,
                        CASCI=False, CAS_Initial=None, CAS_Final = None,
                        MRCI_Initial=None, MRCI_Final = None, MRCI_CASCI_Final=True,
                        EOM=False, btPNO=False, DLPNO=False, 
                        path_wfoverlap=None, tprintwfvalue=1e-6, noDyson=False):
        """
        PhotoElectron module
        """
        module_init_time=time.time()
        blankline()
        print_line_with_mainheader("PhotoElectron: Calculating PES spectra via TDDFT/CAS/MRCI/EOM/MREOM and Dyson-norm approach")
        
        ###############
        #Early exits
        ################
        if CAS is True and MRCI is True:
            print("CAS and MRCI keywords can not both be True!")
            print("Note for MRCI: You must previously optimize orbitals (e.g. with CASSCF) and feed into MRCI")
            ashexit()
        if Initialstate_charge is None or Initialstate_mult is None or Ionizedstate_charge is None or Ionizedstate_mult is None:
            print("Provide charge and spin multiplicity of initial and ionized state: Initialstate_charge, Initialstate_mult, Ionizedstate_charge,Ionizedstate_mult ")
            ashexit()
        if isinstance(theory, ORCATheory) is False:
            print("Theory not supported for PhotoElectronSpectrum")
            print("Only ORCATheory supported for now")
            ashexit()
        #Setting TDDFT to True if CAS/MRCI/EOM is False
        if CAS is False and MRCI is False and EOM is False and TDDFT is False:
            print("No PES method chosen.")
            print("You must choose between:  TDDFT=True, CAS=True, MRCI=True, MREOM=True and EOM=True")
            ashexit()
        if len(Ionizedstate_mult) != len(numionstates):
            print("If 2 ionized state multiplicities are chosen (Ionizedstate_mult) then numionstates list must have 2 elements")
            ashexit()
        ################


        #Numionstates can be number or list of numbers (states for each multiplicity for CAS/MRCI)
        if isint(numionstates):
            self.numionstates_A=numionstates
            self.numionstates_B=numionstates
        elif islist(numionstates):
            if len(numionstates)==2:
                #A,B: first and second multiplicity in Ionizedstate_mult
                self.numionstates_A=numionstates[0]
                self.numionstates_B=numionstates[1]
                self.numionstates=self.numionstates_A+self.numionstates_B
            else:
                #A,B: first and second multiplicity in Ionizedstate_mult
                self.numionstates_A=numionstates[0]
                self.numionstates=self.numionstates_A 

        self.theory=theory
        self.fragment=fragment
        self.Initialstate_charge=Initialstate_charge
        self.Initialstate_mult=Initialstate_mult
        self.Ionizedstate_charge=Ionizedstate_charge
        self.Ionizedstate_mult=Ionizedstate_mult
        self.path_wfoverlap=path_wfoverlap
        self.tda=tda
        self.brokensym=brokensym
        self.HSmult=HSmult
        self.atomstoflip=atomstoflip
        self.initialorbitalfiles=initialorbitalfiles
        self.densities=densities
        self.densgridvalue=densgridvalue
        self.TDDFT=TDDFT
        self.CAS=CAS
        self.CAS_Initial=CAS_Initial
        self.CAS_Final = CAS_Final
        self.memory=memory 
        self.numcores=numcores 
        self.noDyson=noDyson
        self.CASCI=CASCI
        self.MRCI=MRCI
        self.MREOM=MREOM
        self.MRCI_Initial=MRCI_Initial
        self.MRCI_Final = MRCI_Final
        self.tprintwfvalue=tprintwfvalue
        self.MRCI_CASCI_Final=MRCI_CASCI_Final
        self.EOM=EOM 
        self.btPNO=btPNO 
        self.DLPNO=DLPNO 
        self.label=label
        self.check_stability=check_stability

        print("TDDFT:", self.TDDFT)
        print("CAS:", self.CAS)
        print("EOM:", self.EOM)
        print("MRCI:", self.MRCI)
        print("MREOM:", self.MREOM)

        #Initizalign final list (necessary)
        self.finaldysonnorms=[]


        #Getting charge/mult of states from function argument
        self.totnuccharge=self.fragment.nuccharge

        # Always just one StateI object with one charge and one spin multiplicity
        self.stateI = MolState(charge=self.Initialstate_charge, mult=self.Initialstate_mult,numionstates=1, label='Init_State')
        print(bcolors.OKBLUE, "StateI: Charge=", self.stateI.charge, "Multiplicity", self.stateI.mult, bcolors.ENDC)


        if type(self.Ionizedstate_mult) is int:
            #Only one mult for ionized-state. Using numionstates
            self.stateF1 = MolState(charge=self.Ionizedstate_charge, mult=self.Ionizedstate_mult,
                                        numionstates=self.numionstates, label=f'Final_State_mult{self.Ionizedstate_mult}')
            self.MultipleSpinStates = False
            self.Finalstates=[self.stateF1]
            print(bcolors.OKBLUE, "StateF_1: Charge=", self.Finalstates[0].charge, "Multiplicity", self.Finalstates[0].mult, bcolors.ENDC)
            print(bcolors.OKBLUE, "StateF_1: Numionstates=", self.Finalstates[0].numionstates, bcolors.ENDC)        
        #Case list provided for ionized state. Could mean multiple spin states: e.g.  Ionizedstate_mult=[5,7]
        elif type(self.Ionizedstate_mult) is list:
            if len(self.Ionizedstate_mult) == 1:
                self.MultipleSpinStates = False
                #Only one mult for ionized-state. Using numionstates
                self.stateF1 = MolState(charge=self.Ionizedstate_charge, mult=self.Ionizedstate_mult[0],
                                            numionstates=self.numionstates, label=f'Final_State_mult{self.Ionizedstate_mult[0]}')
                self.Finalstates = [self.stateF1]
                print(bcolors.OKBLUE, "StateF_1: Charge=", self.Finalstates[0].charge, "Multiplicity", self.Finalstates[0].mult,
                    bcolors.ENDC)
                print(bcolors.OKBLUE, "StateF_1: Numionstates=", self.numionstates_A, bcolors.ENDC)
            elif len(Ionizedstate_mult) == 2:
                self.MultipleSpinStates = True
                self.stateF1 = MolState(charge=self.Ionizedstate_charge, mult=self.Ionizedstate_mult[0],numionstates=self.numionstates_A,
                                            label=f'Final_State_mult{self.Ionizedstate_mult[0]}')
                self.stateF2 = MolState(charge=self.Ionizedstate_charge, mult=self.Ionizedstate_mult[1],numionstates=self.numionstates_B,
                                            label=f'Final_State_mult{self.Ionizedstate_mult[1]}')
                self.Finalstates = [self.stateF1,self.stateF2]
                print("Multiple spin states for Final State:")
                print(bcolors.OKBLUE, "StateF_1: Charge=", self.Finalstates[0].charge, "Multiplicity", self.Finalstates[0].mult,
                    bcolors.ENDC)
                print(bcolors.OKBLUE, "StateF_2: Charge=", self.Finalstates[1].charge, "Multiplicity", self.Finalstates[1].mult,
                    bcolors.ENDC)
                print(bcolors.OKBLUE, "StateF_1: Numionstates=", self.Finalstates[0].numionstates, bcolors.ENDC)
                print(bcolors.OKBLUE, "StateF_2: Numionstates=", self.Finalstates[1].numionstates, bcolors.ENDC)
            else:
                print("More than Two spin multiplicities are now allowed in Ionizedstate_mult argument")
                ashexit()
        else:
            print("Unknown type for Ionizedstate_mult value. Should be integer or list of integers")

        #PRINT SETTINGS
        #TODO: print everything here
        print("")
        print("Number of threads to be used by WFoverlap: ", numcores)
        os.environ["OMP_NUM_THREADS"] = str(numcores)
        
        if brokensym is True:
            print("Brokensym True. Will find BS-solution for StateI via spin-flipping. HSMult: ", HSmult)

        print("densities option is: ", self.densities, "(options are: SCF, All, None)")
        if self.densities == 'SCF':
            print("Will do densities (and difference densities) for Inital-state and Final-state SCF wavefunctions only.")
            shutil.rmtree('Calculated_densities', ignore_errors=True)
            os.mkdir('Calculated_densities')
        elif self.densities=='All':
            print("Will do densities (and difference densities) for all states: SCF and TDDFT states")
            shutil.rmtree('Calculated_densities', ignore_errors=True)
            os.mkdir('Calculated_densities')
        else:
            self.densities=None
            print("Will not calculate densities")

    #Modify ORCA object
    def setup_ORCA_object(self):
        #General ORCA setting
        self.theory.extraline=self.theory.extraline+"%method\n"+"frozencore FC_NONE\n"+"end\n"
        
        # For orbital analysis
        if 'NORMALPRINT' not in self.theory.orcasimpleinput.upper():
            self.theory.orcasimpleinput = self.theory.orcasimpleinput + ' Normalprint'
            
        if self.brokensym is True:
            self.theory.brokensym=True
            self.theory.HSmult=self.HSmult
            self.theory.atomstoflip=self.atomstoflip
            #Making sure UKS always present if brokensym feature active. Important for open-shell singlets
            if 'UKS' not in self.theory.orcasimpleinput.upper():
                self.theory.orcasimpleinput = self.theory.orcasimpleinput + ' UKS'

    def run_tddft_densities(self):
        print("Inside run_tddft_densities")
        #Here doing densities for each TDDFT-state. SCF-states already done.
        print("")
        print(bcolors.OKMAGENTA, "Densities option: All . Will do TDDFT-gradient calculation for each TDDFT-state (expensive)", bcolors.ENDC)
        os.chdir('Calculated_densities')

        #Adding Keepdens and Engrad to do TDDFT gradient
        self.theory.orcasimpleinput=self.theory.orcasimpleinput+' KeepDens Engrad'

        if '/C' not in self.theory.orcasimpleinput:
            self.theory.orcasimpleinput = self.theory.orcasimpleinput + ' AutoAux'

        for findex, fstate in enumerate(self.Finalstates):
            print(bcolors.OKGREEN, "Calculating Final State SCF + TDDFT DENSITY CALCULATION. Spin Multiplicity: ", fstate.mult, bcolors.ENDC)
            shutil.copyfile('../'+'Final_State_mult' + str(fstate.mult) + '.gbw','Final_State_mult' + str(fstate.mult) + '.gbw')
            os.rename('Final_State_mult' + str(fstate.mult) + '.gbw', self.theory.filename + '.gbw')


            #Looping over each TDDFT-state and doing TDDFT-calc
            for tddftstate in range(1,fstate.numionstates):
                print("-------------------------------------------------")
                print("Running TDDFT-gradient for State: ", tddftstate)
                #Adding Iroot to get state-specific gradient+density                                              ''
                if self.tda == False:
                    # Boolean for whether no_tda is on or not
                    print("Not sure if Full-TDDFT density is available. TO BE CHECKED.")
                    self.no_tda = True
                    tddftstring = "%tddft\n" + "tda false\n" + "nroots " + str(
                        self.numionstates - 1) + '\n' + "maxdim 25\n" + "Iroot {}\n".format(tddftstate) + "end\n" + "\n"
                else:
                    tddftstring = "%tddft\n" + "tda true\n" + "nroots " + str(
                        self.numionstates - 1) + '\n' + "maxdim 25\n" + "Iroot {}\n".format(tddftstate) + "end\n" + "\n"
                    # Boolean for whether no_tda is on or not
                    self.no_tda = False
                self.theory.extraline = "%method\n"+"frozencore FC_NONE\n"+"end\n" + tddftstring

                #Turning off stability analysis. Not available for gradient run.
                if 'stabperform true' in self.theory.orcablocks:
                    print("Turning off stability analysis.")
                    self.theory.orcablocks=self.theory.orcablocks.replace("stabperform true", "stabperform false")

                ash.Singlepoint(fragment=self.fragment, theory=self.theory, charge=fstate.charge, mult=fstate.mult)
                print("Calling orca_plot to create Cube-file for Final state TDDFT-state.")

                #Doing spin-density Cubefile for each cisr file
                run_orca_plot(orcadir=self.theory.orcadir, filename=self.theory.filename + '.gbw', option='cisspindensity',gridvalue=self.densgridvalue,
                            densityfilename=self.theory.filename+'.cisr' )
                os.rename(self.theory.filename + '.spindens.cube', 'Final_State_mult' + str(fstate.mult)+'TDDFTstate_'+str(tddftstate)+'.spindens.cube')
                #Doing eldensity Cubefile for each cisp file and then take difference with Initstate-SCF cubefile
                run_orca_plot(orcadir=self.theory.orcadir, filename=self.theory.filename + '.gbw', option='cisdensity',gridvalue=self.densgridvalue,
                            densityfilename=self.theory.filename+'.cisp' )
                os.rename(self.theory.filename + '.eldens.cube', 'Final_State_mult' + str(fstate.mult)+'TDDFTstate_'+str(tddftstate)+'.eldens.cube')

                final_dens = 'Final_State_mult' + str(fstate.mult)+'TDDFTstate_'+str(tddftstate)+'.eldens.cube'
                cubedict2 = read_cube(final_dens)
                write_cube_diff(self.stateI.cubedict,cubedict2,"Densdiff_SCFInit-TDDFTFinalmult" + str(fstate.mult)+'TDState'+str(tddftstate))
                print("Wrote Cube file containing density difference between Initial State and Final TDDFT State: ", tddftstate)
        self.theory.cleanup()
        os.chdir('..')

    def make_diffdensities_SCF(self):
        os.chdir('Calculated_densities')
        # Read Initial-state-SCF density Cube file into memory
        self.stateI.cubedict = read_cube(f"{self.stateI.label}.eldens.cube")
        for fstate in self.Finalstates:
            if self.densities == 'SCF' or self.densities == 'All':
                # Create difference density between Initial-SCF and Finalstate-SCFs
                cubedict2 = read_cube('Final_State_mult' + str(fstate.mult) + '.eldens.cube')
                write_cube_diff(self.stateI.cubedict,cubedict2,"Densdiff_SCFInit-SCFFinalmult"+str(fstate.mult))
                print("Wrote Cube file containing density difference between Initial State and Final State.")
        os.chdir('..')
        
    def run_MRCI(self):
        print("MRCI/MREOM option active!")
        print("Will do CASSCF orbital optimization for initial-state, followed by MRCI/MREOM")

        #CASSCF/MRCI wavefunction interpreted as restricted (important for get_MO_from_gbw)
        self.stateI.restricted = True
        for fstate in self.Finalstates:
            fstate.restricted = True

        if self.MRCI_CASCI_Final is True:
            print("Will do CAS-CI reference (using initial-state orbitals) for final-states")

            
            print("Using TprintWF value of ", self.tprintwfvalue)
            print("Modifying MRCI block for initial state, CAS({},{})".format(self.MRCI_Initial[0],self.MRCI_Initial[1]))
            print("{} electrons in {} orbitals".format(self.MRCI_Initial[0],self.MRCI_Initial[1]))
            print("WARNING: MRCI determinant-printing read will only work for ORCA-current or ORCA 5.0, not older ORCA versions like ORCA 4.2")
            if 'SORCI' in self.theory.orcasimpleinput:
                SORCI=True
                print("SORCI is True!")
            else:
                SORCI=False
                print("SORCI is False!")
            #USING CASSCF block to define reference
            #Add nel,norb and nroots lines back in. Also determinant printing option
            print("theory.orcablocks :", self.theory.orcablocks)
            
            #If CASSCF block present, trim and replace
            if '%casscf' in self.theory.orcablocks:
                            #Removing nel/norb/nroots lines if user added
                for line in self.theory.orcablocks.split('\n'):
                    if 'nel' in line:
                        self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
                    if 'norb' in line:
                        self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
                    if 'nroots' in line:
                        self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
                    if 'maxiter' in line:
                        self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
                self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')    
                self.theory.orcablocks = self.theory.orcablocks.replace('%casscf', '%casscf\n'  + "nel {}\n".format(self.MRCI_Initial[0]) +
                                                        "norb {}\n".format(self.MRCI_Initial[1]) + "nroots {}\n".format(1))
            else:
                self.theory.orcablocks= self.theory.orcablocks + '%casscf\n'  + "nel {}\n".format(self.MRCI_Initial[0]) + "norb {}\n".format(self.MRCI_Initial[1]) + "nroots {}\nend\n".format(1)
            print("theory.orcablocks :", self.theory.orcablocks)


            #Enforcing CAS-CI
            #if 'noiter' not in theory.orcasimpleinput.lower():
            #    theory.orcasimpleinput = theory.orcasimpleinput + ' noiter '
            if self.MREOM is True:
                if '%mdci' not in self.theory.orcablocks:
                    self.theory.orcablocks = self.theory.orcablocks + "%mdci\nSTol 1e-7\nmaxiter 2700\nDoSingularPT true\nend\n"
                else:
                    self.theory.orcablocks = self.theory.orcablocks.replace("%mdci\n","%mdci\nSTol 1e-7\nmaxiter 2700\nDoSingularPT true\nend\n")
                    
            #Defining simple MRCI block. States defined
            if '%mrci' not in self.theory.orcablocks:
                self.theory.orcablocks = self.theory.orcablocks + "%mrci\n" + "printwf det\nTPrintwf {}\n".format(self.tprintwfvalue) + "end"
            else:
                self.theory.orcablocks = self.theory.orcablocks.replace("%mrci\n","%mrci\n"+"printwf det\nTPrintwf {}\n".format(self.tprintwfvalue))
            
            #theory.orcablocks = "%mrci\n" + "printwf det\nTPrintwf 1e-16\n" + "newblock {} *\n refs cas({},{}) end\n".format(stateI.mult,MRCI_Initial[0],MRCI_Initial[1])+ "nroots {}\n end\n".format(1) + "end"
            self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')
            self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')

            #Adding MRCI+Q to simpleinputline unless MREOM
            #TODO: Remove as we may want another MRCI method ?
            if self.MREOM is True:
                if 'MREOM' not in self.theory.orcasimpleinput:
                    self.theory.orcasimpleinput = self.theory.orcasimpleinput + ' MR-EOM' 
            else:
                if 'MRCI+Q' not in self.theory.orcasimpleinput:
                    self.theory.orcasimpleinput = self.theory.orcasimpleinput + ' MRCI+Q' 

            #Calling SCF run
            #self.run_SCF_InitState()
            result = ash.Singlepoint(fragment=self.fragment, theory=self.theory, charge=self.stateI.charge, mult=self.stateI.mult)
            self.stateI.energy=result.energy

            #Keeping copy of input/outputfile and GBW file
            shutil.copyfile(self.theory.filename + '.out', './' + self.stateI.label + '.out')
            shutil.copyfile(self.theory.filename + '.inp', './' + self.stateI.label + '.inp')
            shutil.copyfile(self.theory.filename + '.gbw', './' + self.stateI.label + '.gbw')
            self.stateI.gbwfile=self.stateI.label+".gbw"
            self.stateI.outfile=self.stateI.label+".out"

            #Get orbital ranges (stateI is sufficient)
            internal_orbs,active_orbs,external_orbs = casscf_orbitalranges_grab(self.theory.filename+'.out')



            print("Modifying MRCI block for Final state, MRCI({},{})".format(self.MRCI_Final[0], self.MRCI_Final[1]))
            print("{} electrons in {} orbitals".format(self.MRCI_Final[0], self.MRCI_Final[1]))
            
            
            # Making sure multiplicties are sorted in ascending order and creating comma-sep string
            MRCI_mults = ','.join(str(x) for x in sorted([f.mult for f in self.Finalstates]))

            print("MRCI_mults:", MRCI_mults)
            for line in self.theory.orcablocks.split('\n'):
                if 'nel' in line:
                    self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
                if 'norb' in line:
                    self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
                if 'nroots' in line:
                    self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
                if 'mult' in line:
                    self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
            self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')

            #Add nel,norb and nroots lines back in.
            # And both spin multiplicities. Nroots for each
            #numionstates_string = ','.join(str(numionstates) for x in [f.mult for f in Finalstates])
            numionstates_string = ','.join(str(f.numionstates) for f in self.Finalstates)
            print("numionstates_string:", numionstates_string)
            self.theory.orcablocks = self.theory.orcablocks.replace('%casscf', '%casscf\n' + "nel {}\n".format(self.MRCI_Final[0]) +
                                                        "norb {}\n".format(
                                                            self.MRCI_Final[1]) + "nroots {}\n".format(numionstates_string) + "mult {}\n".format(MRCI_mults))

            #In Final-state MRCI we would typically use the previous CASSCF-orbitals. Hence CAS-CI and noiter
            if self.MRCI_CASCI_Final is True:
                if 'noiter' not in self.theory.orcasimpleinput.lower():
                    self.theory.orcasimpleinput = self.theory.orcasimpleinput + ' noiter '


            #Creating newblock blocks for each multiplicity
            #newblockstring=""
            #for mult in [f.mult for f in Finalstates]:
            #    newblockstring = newblockstring + "  newblock {} *\n".format(mult)+"  refs cas({},{}) end\n".format(MRCI_Final[0],MRCI_Final[1] )+ "nroots {}\n".format(mult)+"end\n"
            #theory.orcablocks = theory.orcablocks + "%mrci\n" + "printwf det\nTPrintwf 1e-16\nend"
            self.theory.orcablocks = self.theory.orcablocks.replace('\n\n', '\n')
            self.theory.orcablocks = self.theory.orcablocks.replace('\n\n', '\n')

            print(bcolors.OKGREEN, "Calculating Final State MRCI Spin Multiplicities: ", [f.mult for f in self.Finalstates], bcolors.ENDC)

            if self.initialorbitalfiles is not None:
                print("not tested for MRCI...")
                print("initialorbitalfiles keyword provided.")
                print("Will use file {} as guess GBW file for this Final state.".format(self.initialorbitalfiles[findex + 1]))
                shutil.copyfile(self.initialorbitalfiles[findex + 1], self.theory.filename + '.gbw')

            ash.Singlepoint(fragment=self.fragment, theory=self.theory, charge=self.Finalstates[0].charge, mult=self.Finalstates[0].mult)

            #Getting state-energies of all states for each spin multiplicity. MRCI vs. SORCI
            if SORCI is True:
                fstates_dict = mrci_state_energies_grab(self.theory.filename+'.out', SORCI=True)
            else:
                fstates_dict = mrci_state_energies_grab(self.theory.filename+'.out')
            # Saveing GBW and CIS file
            shutil.copyfile(self.theory.filename + '.gbw', './' + 'Final_State' + '.gbw')
            shutil.copyfile(self.theory.filename + '.out', './' + 'Final_State' + '.out')
            shutil.copyfile(self.theory.filename + '.inp', './' + 'Final_State' + '.inp')

            #Each fstate linked with same GBW file and outfile
            for fstate in self.Finalstates:
                fstate.gbwfile = "Final_State" + ".gbw"
                fstate.outfile = "Final_State" + ".out"

            #TODO: Saving files for density Cube file creation for MRCI

        self.FinalIPs = []
        self.Finalionstates = []
        FinalTDtransitionenergies =[]
        print(bcolors.OKBLUE,"Initial State energy:", self.stateI.energy, "au",bcolors.ENDC)
        print(bcolors.OKBLUE,"Final State energies:", fstates_dict, bcolors.ENDC)
        
        for fstate in self.Finalstates:
            fstate.ionstates = fstates_dict[fstate.mult]
            for ionstate in fstate.ionstates:
                fstate.IPs.append((ionstate-self.stateI.energy)*ash.constants.hartoeV)
            print("Mult: {} IPs: {}".format(fstate.mult,fstate.IPs))
            self.FinalIPs = self.FinalIPs + fstate.IPs
            self.Finalionstates = self.Finalionstates + fstate.ionstates

        #MRCI prepare determinants for Dyson
        print("Grabbing determinants from Initial State output")
        if SORCI is True:
            init_state = grab_dets_from_MRCI_output(self.stateI.outfile,SORCI=True)                    
        else:
            print("self.stateI.outfile:", self.stateI.outfile)
            init_state = grab_dets_from_MRCI_output(self.stateI.outfile)
        print("xx")
        det_init = format_ci_vectors(init_state[self.Initialstate_mult])

        writestringtofile(det_init, "dets_init")

        #Checking if wrong determinant in file
        delete_wrong_det("dets_init",self.stateI.mult)


        print("Grabbing determinants from Final State output")
        if SORCI is True:
            final_states = grab_dets_from_MRCI_output(self.Finalstates[0].outfile,SORCI=True)                    
        else:
            final_states = grab_dets_from_MRCI_output(self.Finalstates[0].outfile)
        
        for fstate in self.Finalstates:
            print("fstate: ", fstate)
            print("fstate.mult :", fstate.mult)
            det_final = format_ci_vectors(final_states[fstate.mult])
            #print("det_final : ", det_final)
            # Printing to file
            writestringtofile(det_final, "dets_final_mult" + str(fstate.mult))
            
            #Checking if wrong determinant in file and delete
            delete_wrong_det("dets_final_mult" + str(fstate.mult),fstate.mult)

    def run_CAS(self):
        print("Using TprintWF value of ", self.tprintwfvalue)
        print("Modifying CASSCF block for initial state, CAS({},{})".format(self.CAS_Initial[0],self.CAS_Initial[1]))
        print("{} electrons in {} orbitals".format(self.CAS_Initial[0],self.CAS_Initial[1]))

        #Removing nel/norb/nroots lines if user added
        for line in self.theory.orcablocks.split('\n'):
            if 'nel' in line:
                self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
            if 'norb' in line:
                self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
            if 'nroots' in line:
                self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
        self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')

        #Add nel,norb and nroots lines back in. Also determinant printing option
        self.theory.orcablocks = self.theory.orcablocks.replace('%casscf', '%casscf\n' + "printwf det\nci TPrintwf {} end\n".format(self.tprintwfvalue) + "nel {}\n".format(self.CAS_Initial[0]) +
                                                    "norb {}\n".format(
                                                        self.CAS_Initial[1]) + "nroots {}\n".format(1))
        self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')
        self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')

        #Calling SCF run
        self.run_SCF_InitState()
        #Note: Using SCF energy and not Final Single Point energy (does not work for TDDFT)
        print("After CASSCF run")
        self.stateI.energy=casscfenergygrab(theory.filename+'.out')
        print("stateI.energy: ", self.stateI.energy)


        # Initial state orbitals for MO-DOSplot
        self.stateI.hftyp='CASSCF'

        #CASSCF wavefunction interpreted as restricted
        self.stateI.restricted = True
        for fstate in self.Finalstates:
            fstate.restricted = True


        #Get orbital ranges (stateI is sufficient)
        internal_orbs,active_orbs,external_orbs = casscf_orbitalranges_grab(self.theory.filename+'.out')

        #FINAL
        ##CAS option: State-averaged calculation for both spin multiplicities.


        print("Modifying CASSCF block for final state, CAS({},{})".format(self.CAS_Final[0],self.CAS_Final[1]))
        print("{} electrons in {} orbitals".format(self.CAS_Final[0],self.CAS_Final[0]))
        #Making sure multiplicties are sorted in ascending order and creating comma-sep string
        CAS_mults=','.join(str(x) for x in sorted([f.mult for f in self.Finalstates]))
        print("CAS_mults:", CAS_mults)
        #Removing nel/norb/nroots lines
        for line in self.theory.orcablocks.split('\n'):
            if 'nel' in line:
                self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
            if 'norb' in line:
                self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
            if 'nroots' in line:
                self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
            if 'mult' in line:
                self.theory.orcablocks=self.theory.orcablocks.replace(line,'')
        self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')

        #Add nel,norb and nroots lines back in.
        # And both spin multiplicities. Nroots for each
        #numionstates_string = ','.join(str(numionstates) for x in [f.numionstates for f in Finalstates])
        numionstates_string = ','.join(str(f.numionstates) for f in self.Finalstates)
        self.theory.orcablocks = self.theory.orcablocks.replace('%casscf', '%casscf\n' + "nel {}\n".format(self.CAS_Final[0]) +
                                                    "norb {}\n".format(
                                                        self.CAS_Final[1]) + "nroots {}\n".format(numionstates_string) + "mult {}\n".format(CAS_mults))
        self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')
        self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')
        #CAS-CI option for Ionized FInalstate. CASSCF orb-opt done on Initial state but then CAS-CI using Init-state orbs on Final-states
        if self.CASCI is True:
            print("CASCI option on! Final ionized states performed at CAS-CI level using Initial-state orbitals.")
            self.theory.orcasimpleinput = self.theory.orcasimpleinput + ' noiter'


        print(bcolors.OKGREEN, "Calculating Final State CASSCF Spin Multiplicities: ", [f.mult for f in self.Finalstates], bcolors.ENDC)

        if self.initialorbitalfiles is not None:
            print("not tested for CASSCF...")
            print("initialorbitalfiles keyword provided.")
            print("Will use file {} as guess GBW file for this Final state.".format(self.initialorbitalfiles[findex + 1]))
            shutil.copyfile(self.initialorbitalfiles[findex + 1], self.theory.filename + '.gbw')

        ash.Singlepoint(fragment=self.fragment, theory=self.theory, charge=self.Finalstates[0].charge, mult=self.Finalstates[0].mult)

        #Getting state-energies of all states for each spin multiplicity (state-averaged calculation)
        fstates_dict = casscf_state_energies_grab(self.theory.filename+'.out')
        print("fstates_dict: ", fstates_dict)

        # Saveing GBW and CIS file
        shutil.copyfile(self.theory.filename + '.gbw', './' + 'Final_State' + '.gbw')
        shutil.copyfile(self.theory.filename + '.out', './' + 'Final_State' + '.out')
        shutil.copyfile(self.theory.filename + '.inp', './' + 'Final_State' + '.inp')

        #Each fstate linked with same GBW file and outfile
        for fstate in self.Finalstates:
            fstate.gbwfile = "Final_State" + ".gbw"
            fstate.outfile = "Final_State" + ".out"

        #TODO: Saving files for density Cube file creation for CASSCF


        self.FinalIPs = []
        self.Finalionstates = []
        FinalTDtransitionenergies =[]
        print(bcolors.OKBLUE,"Initial State energy:", self.stateI.energy, "au",bcolors.ENDC)
        print(bcolors.OKBLUE,"Final State energies:", fstates_dict, bcolors.ENDC)
        
        for fstate in self.Finalstates:
            fstate.ionstates = fstates_dict[fstate.mult]
            for ionstate in fstate.ionstates:
                fstate.IPs.append((ionstate-self.stateI.energy)*ash.constants.hartoeV)
            print("Mult: {} IPs: {}".format(fstate.mult,fstate.IPs))
            self.FinalIPs = self.FinalIPs + fstate.IPs
            self.Finalionstates = self.Finalionstates + fstate.ionstates

    # Calculate Ionized state via SCF+TDDFT approach
    def run_TDDFT(self):
        print("TDDFT option chosen:")
        print(bcolors.OKBLUE,"Total ion states:", self.numionstates, bcolors.ENDC)
        print(bcolors.OKBLUE,"TDDFT-calculated ion states:", self.numionstates-1, bcolors.ENDC)

        #Run Initial-State SCF
        self.run_SCF_InitState()

        ###################
        #FINAL STATE
        ###################
        #TDDFT-option SCF+TDDFT for each spin multiplicity
        #################################################

        #Looping over Finalstate-multiplicities (not individual states)
        for findex,fstate in enumerate(self.Finalstates):
            #Setting TDA/TDDFT states for each spin multiplicity
            if self.tda==False:
                # Boolean for whether no_tda is on or not
                self.no_tda = True
                tddftstring="%tddft\n"+"tda false\n"+"nroots " + str(fstate.numionstates-1) + '\n'+"maxdim 25\n"+"maxiter 15"+"end\n"+"\n"
            else:
                tddftstring="%tddft\n"+"tda true\n"+"nroots " + str(fstate.numionstates-1) + '\n'+"maxdim 25\n"+"end\n"+"\n"
                # Boolean for whether no_tda is on or not
                self.no_tda = False
            self.theory.extraline=self.theory.extraline+tddftstring

            print(bcolors.OKGREEN, "Calculating Final State SCF + TDDFT. Spin Multiplicity: ", fstate.mult, bcolors.ENDC)
            if self.initialorbitalfiles is not None:
                print("Initial orbitals keyword provided.")
                print("Will use file {} as guess GBW file for this Final state.".format(self.initialorbitalfiles[findex+1]))
                shutil.copyfile(self.initialorbitalfiles[findex+1], self.theory.filename + '.gbw')

            #Run SCF+TDDDFT
            ash.Singlepoint(fragment=self.fragment, theory=self.theory, charge=fstate.charge, mult=fstate.mult)
            stability = check_stability_in_output(self.theory.filename+'.out')
            if stability is False and check_stability is True:
                print("PES: Unstable final state. Exiting...")
                ashexit()
            
            #Grab SCF energy
            fstate.energy = scfenergygrab(self.theory.filename+'.out')

            #Grab TDDFT states from ORCA output
            fstate.TDtransitionenergies = tddftgrab(self.theory.filename+'.out')

            #Saving GBW and CIS files
            shutil.copyfile(self.theory.filename + '.gbw', './' + 'Final_State_mult' + str(fstate.mult) + '.gbw')
            shutil.copyfile(self.theory.filename + '.cis', './' + 'Final_State_mult' + str(fstate.mult) + '.cis')
            shutil.copyfile(self.theory.filename + '.out', './' + 'Final_State_mult' + str(fstate.mult) + '.out')
            shutil.copyfile(self.theory.filename + '.inp', './' + 'Final_State_mult' + str(fstate.mult) + '.inp')
            fstate.gbwfile="Final_State_mult"+str(fstate.mult)+".gbw"
            fstate.outfile="Final_State_mult"+str(fstate.mult)+".out"
            fstate.cisfile="Final_State_mult"+str(fstate.mult)+".cis"

            # Final state orbitals for MO-DOSplot
            fstate.occorbs_alpha, fstate.occorbs_beta, fstate.hftyp = orbitalgrab(self.theory.filename+'.out')

            #Calculate SCF eldensity and spindensity if requested
            if self.densities == 'SCF' or self.densities == 'All':
                #Electron density
                run_orca_plot(orcadir=self.theory.orcadir,filename=f"{self.theory.filename}.gbw", option='density', gridvalue=self.densgridvalue)
                #Move into Calculated_densities dir
                shutil.move(f"{self.theory.filename}.eldens.cube", 'Calculated_densities/' + f"{fstate.label}.eldens.cube")
                #Spin density (only if UHF). Otherwise orca_plot gets confused (takes difference between alpha-density and nothing)
                if fstate.hftyp == "UHF":
                    run_orca_plot(orcadir=self.theory.orcadir,filename=f"{self.theory.filename}.gbw", option='spindensity', gridvalue=self.densgridvalue)
                    #Move into Calculated_densities dir
                    shutil.move(f"{self.theory.filename}.spindens.cube", 'Calculated_densities/' + f"{fstate.label}.spindens.cube")

            print("Final state multiplicity properties:", fstate.__dict__)
            if fstate.hftyp == "UHF":
                fstate.restricted = False
            elif fstate.hftyp == "RHF":
                fstate.restricted = True

        blankline()
        blankline()
        print("All SCF and TDDFT calculations are now done (unless Densities=All, then we will calculate TDDFT densities at the end)!")


        #Printing initial results
        self.FinalIPs = []; self.Finalionstates = []; self.FinalTDtransitionenergies =[]
        print(bcolors.OKBLUE,"\nInitial State SCF energy:", self.stateI.energy, "au",bcolors.ENDC)
        print("")
        for fstate in self.Finalstates:
            print("---------------------------------------------------------------------------")
            print("SCF energy and TDDFT transition energies for FinalState mult: ", fstate.mult)
            # 1st vertical IP via deltaSCF
            GSIP=(fstate.energy-self.stateI.energy)*ash.constants.hartoeV
            fstate.GSIP=GSIP
            print(bcolors.OKBLUE,"Initial Final State SCF energy:", fstate.energy, "au", bcolors.ENDC)
            print(bcolors.OKBLUE,"1st vertical IP (delta-SCF):", fstate.GSIP,bcolors.ENDC)
            # TDDFT states
            print(bcolors.OKBLUE, "TDDFT transition energies (eV) for FinalState (mult: {}) : {}\n".format(fstate.mult, fstate.TDtransitionenergies), bcolors.ENDC, )

            # Adding GS-IP to IP-list and GS ion to ionstate
            fstate.IPs.append(fstate.GSIP)
            fstate.ionstates.append(fstate.energy)
            for e in fstate.TDtransitionenergies:
                fstate.ionstates.append(e / ash.constants.hartoeV + fstate.energy)
                fstate.IPs.append((e / ash.constants.hartoeV + fstate.energy - self.stateI.energy) * ash.constants.hartoeV)
            print("")
            print(bcolors.OKBLUE, "TDDFT-derived IPs (eV), delta-SCF IP plus TDDFT transition energies:\n", bcolors.ENDC, fstate.IPs)
            print(bcolors.OKBLUE, "Ion-state energies (au):\n", bcolors.ENDC, fstate.ionstates)
            print("")
            self.FinalIPs = self.FinalIPs + fstate.IPs
            self.Finalionstates = self.Finalionstates + fstate.ionstates
            self.FinalTDtransitionenergies = self.FinalTDtransitionenergies + fstate.TDtransitionenergies

    def run_EOM(self):
        print("EOM is True. Will do EOM-IP-CCSD calculations to calculate IPs directly.")

        #NOTE: EOM will not call Init-State SCF
        
        #Preserve old
        orig_orcablocks=copy.copy(theory.orcablocks)
        
        #Will calculate IPs directly
        print("Adding MDCI block for initial state")
        
        #Canonical EOM, btPNO or DLPNO
        if btPNO == True:
            if 'bt-PNO-IP-EOM-CCSD' not in self.theory.orcasimpleinput:
                self.theory.orcasimpleinput =  self.theory.orcasimpleinput + ' bt-PNO-IP-EOM-CCSD '
        elif DLPNO == True:
            if 'IP-EOM-DLPNO-CCSD' not in self.theory.orcasimpleinput:
                self.theory.orcasimpleinput =  self.theory.orcasimpleinput + ' IP-EOM-DLPNO-CCSD '
        else:
            if 'IP-EOM-CCSD' not in self.theory.orcasimpleinput:
                self.theory.orcasimpleinput =  self.theory.orcasimpleinput + ' IP-EOM-CCSD '
        
        self.FinalIPs=[]
        fstates_dict={}
        for fstate in self.Finalstates:
            print(bcolors.OKGREEN, "Calculating IPs directly via IP-EOM-CCSD. ", bcolors.ENDC)
            if fstate.mult > self.stateI.mult:
                print("Final state mult {}, setting DoBeta true".format(fstate.mult))
                Electron_ion_line='DoBeta true'
            else:
                print("Final state mult {}, setting DoAlpha true".format(fstate.mult))
                Electron_ion_line='DoAlpha true'
            #Add nel,norb and nroots lines back in. Also determinant printing option
            self.theory.orcablocks = orig_orcablocks + '\n%mdci\n' + 'nroots {}\n'.format(fstate.numionstates) + Electron_ion_line+'\n' + 'maxiter 200\n'+'end\n'
            self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')
            self.theory.orcablocks = self.theory.orcablocks.replace('\n\n','\n')        
        

            if self.initialorbitalfiles is not None:
                print("not tested for IP-EOM-CCSD...")
                print("initialorbitalfiles keyword provided.")
                print("Will use file {} as guess GBW file for this Final state.".format(self.initialorbitalfiles[findex + 1]))
                shutil.copyfile(self.initialorbitalfiles[findex + 1], self.theory.filename + '.gbw')
            #NOTE: Using initial state charge/mult here because EOM
            init_EOM = ash.Singlepoint(fragment=fragment, theory=self.theory, charge=self.Initialstate_charge, mult=self.Initialstate_mult)
            energy = init_EOM.energy
            self.stateI.energy= energy


            #Grab EOM-IPs and dominant singles amplitudes
            IPs, amplitudes = grabEOMIPs(theory.filename+'.out')
            print("IPs:", IPs)
            print("Dominant singles EOM amplitudes:", amplitudes)
            
            fstate.IPs=IPs
            fstate.dysonnorms=amplitudes
            
            #Collecting list of all IPs
            self.FinalIPs=self.FinalIPs+IPs
            
            
            #State_energies are Inititial-state energy + transition energy
            state_energies=[IP/ash.constants.hartoeV + self.stateI.energy for IP in IPs]
            
            #Equating the dominant singles amplitudes with dysonnorms.
            
            fstates_dict[fstate.mult] = state_energies
            print("fstates_dict:", fstates_dict)
            
            fstate.ionstates = fstates_dict[fstate.mult]


            # Saveing GBW and inp and outfiles
            shutil.copyfile(self.theory.filename + '.gbw', './' + 'Final_State_mult' + str(fstate.mult)+'.gbw')
            shutil.copyfile(self.theory.filename + '.out', './' + 'Final_State_mult' + str(fstate.mult)+'.out')
            shutil.copyfile(self.theory.filename + '.inp', './' + 'Final_State_mult' + str(fstate.mult)+ '.inp')

            #Each fstate linked with same GBW file and outfile
            fstate.gbwfile = 'Final_State_mult' + str(fstate.mult)+'.gbw'
            fstate.outfile = 'Final_State_mult' + str(fstate.mult)+'.out'
            print("")

            #Final states
            print(bcolors.OKBLUE,"Initial State CCSD energy:", self.stateI.energy, "au",bcolors.ENDC)
            print(bcolors.OKBLUE,"Final State CCSD+EOM-IP energies:", fstates_dict, bcolors.ENDC)
            #IPs already calculated
            print("FinalIPs:", self.FinalIPs)
            self.Finalionstates = []; self.finaldysonnorms=[]
            for fstate in self.Finalstates:
                print("Mult: {} IPs: {}".format(fstate.mult,fstate.IPs))  
                self.Finalionstates = self.Finalionstates + fstate.ionstates      
                self.finaldysonnorms = self.finaldysonnorms + fstate.dysonnorms 


        print(bcolors.WARNING,"WARNING: Dyson norms not calculated for IP-EOM-CCSD. Instead using dominant singles amplitudes as an approximation",bcolors.ENDC)
        print("Approximate Dyson norms: ", self.finaldysonnorms)


    def run_SCF_InitState(self):
        print(bcolors.OKGREEN, "\nCalculating Initial State SCF.",bcolors.ENDC)

        if self.initialorbitalfiles is not None:
            print("initialorbitalfiles keyword provided.")
            print("Will use file {} as guess GBW file for Initial state".format(self.initialorbitalfiles[0]))
            shutil.copyfile(self.initialorbitalfiles[0], self.theory.filename + '.gbw')

        self.InitSCF = ash.Singlepoint(fragment=self.fragment, theory=self.theory, charge=self.Initialstate_charge, mult=self.Initialstate_mult)
        finalsinglepointenergy = self.InitSCF.energy
        stability = check_stability_in_output(self.theory.filename+'.out')
        if stability is False and check_stability is True:
            print("PES: Unstable initial state. Exiting...")
            ashexit()

        #Grab energy of initial state
        self.stateI.energy=scfenergygrab(self.theory.filename+'.out')

        # Initial state orbitals for MO-DOSplot
        self.stateI.occorbs_alpha, self.stateI.occorbs_beta, self.stateI.hftyp = orbitalgrab(self.theory.filename+'.out')
        print("Initial state occupied MO energies (alpha):", self.stateI.occorbs_alpha)
        print("Initial state SCF-type:", self.stateI.hftyp)

        # Specify whether Initial state is restricted or not.
        if self.stateI.hftyp == "UHF":
            self.stateI.restricted = False
        elif self.stateI.hftyp == "RHF":
            self.stateI.restricted = True

        #Keeping copy of input/outputfile and GBW file
        shutil.copyfile(self.theory.filename + '.out', './' + self.stateI.label + '.out')
        shutil.copyfile(self.theory.filename + '.inp', './' + self.stateI.label + '.inp')
        shutil.copyfile(self.theory.filename + '.gbw', './' + self.stateI.label + '.gbw')
        self.stateI.gbwfile=self.stateI.label+".gbw"
        self.stateI.outfile=self.stateI.label+".out"
        #Calculate SCF eldensity and spindensity if requested
        if self.densities == 'SCF' or self.densities == 'All':
            #Electron density
            run_orca_plot(orcadir=self.theory.orcadir,filename=f"{self.theory.filename}.gbw", option='density', gridvalue=self.densgridvalue)
            #Move into Calculated_densities dir
            shutil.move(f"{self.theory.filename}.eldens.cube", 'Calculated_densities/' + f"{self.stateI.label}.eldens.cube")
            #Spin density (only if UHF). Otherwise orca_plot gets confused (takes difference between alpha-density and nothing)
            if self.stateI.hftyp == "UHF":
                run_orca_plot(orcadir=self.theory.orcadir,filename=f"{self.theory.filename}.gbw", option='spindensity', gridvalue=self.densgridvalue)
                #Move into Calculated_densities dir
                shutil.move(f"{self.theory.filename}.spindens.cube", 'Calculated_densities/' + f"{self.stateI.label}.spindens.cube")

    def mo_spectrum(self):
        #MO IP spectrum:
        self.stk_alpha,self.stk_beta=modosplot(self.stateI.occorbs_alpha,self.stateI.occorbs_beta,self.stateI.hftyp)
        moips=sorted(self.stk_alpha+self.stk_beta)
        print(bcolors.OKBLUE,"MO IPs (negative of MO energies of State I):", bcolors.ENDC)
        print(moips)
        print("")
        print("MO-IPs (alpha), eV : ", self.stk_alpha)
        print("MO-IPs (beta), eV : ", self.stk_beta)
        print("")
        print("") 

    def print_final_table(self):
        #Print table with info
        print("-------------------------------------------------------------------------")
        print("FINAL RESULTS for fragment (CalcLabel: {} FragLabel: {}  Formula: {})".format(self.label,self.fragment.label, self.fragment.prettyformula))
        print("-------------------------------------------------------------------------")
        print("Initial state:")
        print("{:>6} {:>7} {:^20} {:^5}".format("State no.", "Mult", "TotalE (Eh)", "State-type"))
        if self.EOM is True:
            print("{:>6d} {:>7d} {:20.11f} {:>8}".format(0, self.stateI.mult, self.stateI.energy, "CCSD"))            
        else:
            print("{:>6d} {:>7d} {:20.11f} {:>8}".format(0, self.stateI.mult, self.stateI.energy, "SCF"))
        print("")
        print("Final ionized states:")
        if self.CAS is True or self.MRCI is True:
            stype='CI'
            print("{:>6} {:>7} {:^20} {:8} {:10} {:>7}".format("State no.", "Mult", "TotalE (Eh)", "IE (eV)", "Dyson-norm", "State-type"))
            for i, (E, IE, dys) in enumerate(zip(self.Finalionstates,self.FinalIPs,self.finaldysonnorms)):
                #Getting spinmult
                if self.MultipleSpinStates is True:
                    #Change test. what mult we are in.. TODO: Check this for correctness
                    if i < self.Finalstates[0].numionstates:
                        spinmult=self.Finalstates[0].mult
                    else:
                        spinmult=self.Finalstates[1].mult
                else:
                    spinmult=self.stateF1.mult
                print("{:>6d} {:>7d} {:20.11f} {:>10.3f} {:>10.5f} {:>10}".format(i, spinmult, E, IE, dys,stype))
        elif self.EOM is True:
            stype='EOM'
            print("{:>6} {:>7} {:^20} {:8} {:10} {:>7}".format("State no.", "Mult", "TotalE (Eh)", "IE (eV)", "Dyson-norm", "State-type"))
            for i, (E, IE, dys) in enumerate(zip(self.Finalionstates,self.FinalIPs,self.finaldysonnorms)):
                #Getting spinmult
                if self.MultipleSpinStates is True:
                    #Change test. what mult we are in.. TODO: Check this for correctness
                    if i < self.Finalstates[0].numionstates:
                        spinmult=self.Finalstates[0].mult
                    else:
                        spinmult=self.Finalstates[1].mult
                else:
                    spinmult=self.stateF1.mult
                print("{:>6d} {:>7d} {:20.11f} {:>10.3f} {:>10.5f} {:>10}".format(i, spinmult, E, IE, dys,stype))        
        
        elif self.TDDFT is True:
            #Creating lists of all state labels and transition energies
            if self.tda is True: 
                tdtype = 'TDA'
            else: 
                tdtype = 'TDDFT'
            statelabels=[]; tdtransitions=[];spinmults=[]
            for fstate in self.Finalstates:
                statelabels +=  ['SCF']+[tdtype]*(fstate.numionstates-1)
                tdtransitions += [0.0]+fstate.TDtransitionenergies
                spinmults += [fstate.mult]*fstate.numionstates
            
            print("{:>6} {:>7} {:^20} {:8} {:10} {:>7} {:>15}".format("State no.", "Mult", "TotalE (Eh)", "IE (eV)", "Dyson-norm", "State-type", "TDDFT Exc.E. (eV)"))
            fstate=self.Finalstates[0]
            for i, (E, IE, dys,statelabel,TDtransenergy,spinmult) in enumerate(zip(self.Finalionstates,self.FinalIPs,self.finaldysonnorms,statelabels,tdtransitions,spinmults)):
                print("{:>6d} {:>7d} {:20.11f} {:>10.3f} {:>10.5f} {:>10} {:>17.3f}".format(i, spinmult, E, IE, dys,statelabel, TDtransenergy))

    def prepare_mos_file(self):
        print("Inside prepare_mos_file")
        print(bcolors.OKGREEN, "Grabbing AO matrix, MO coefficients and determinants from ORCA GBW file, CIS file (if TDDFT) or output (if CAS/MRCI)",
        bcolors.ENDC)
        # Specify frozencore or not.
        self.frozencore = 0
        print("")
        ####################
        # Initial State MOs
        ####################
        print("Doing initial state")
        # Get AO matrix from init state calculation
        saveAOmatrix(self.stateI.gbwfile, orcadir=self.theory.orcadir)


        # Grab MO coefficients and write to files mos_init and mos_final

        #Delete old mos_init file
        try:
            os.remove("mos_init")
        except:
            pass
        print("StateI GBW-file: ", self.stateI.gbwfile)
        print("StateI Restricted :", self.stateI.restricted)
        print("Frozencore: ", self.frozencore)
        mos_init = get_MO_from_gbw(self.stateI.gbwfile, self.stateI.restricted, self.frozencore,self.theory.orcadir)
        writestringtofile(mos_init, "mos_init")
        print("Created file mos_init")
        ####################
        # Final State MOs
        ####################
        print("Doing Final state")
        for fstate in self.Finalstates:
            print("StateF GBW-file: ", fstate.gbwfile)
            print("StateF Restricted :", fstate.restricted)
            print("Frozencore: ", self.frozencore)
            mos_final = get_MO_from_gbw(fstate.gbwfile, fstate.restricted, self.frozencore,self.theory.orcadir)
            writestringtofile(mos_final, "mos_final-mult"+str(fstate.mult))

    def TDDFT_dets_prep(self):
        # Create determinant file for ionized TDDFT states
        # Needs Outputfile, CIS-file, restricted-option, XXX, GS multiplicity, number of ion states and states to skip
        # States per Initial and Final options
        #TDDFT-only:
        # Number of multiplicity blocks I think. Should be 2 in general, 1 for GS and 1 for ionized
        # Not correct, should be actual multiplicites. Finalstate mult. If doing TDDFT-triplets then I guess we have more
        # Threshold for WF.
        self.wfthres = 2.0    
        #TDDFT: GETTING DETERMINANTS FROM CIS FILE
        # Final state. Create detfiles
        statestoskip = [0, 0]
        for fstate in self.Finalstates:
            # mults = [stateFmult]
            mults = [fstate.mult]
            det_final = get_dets_from_cis(fstate.outfile, fstate.cisfile, fstate.restricted, mults, fstate.charge, fstate.mult,
                                        self.totnuccharge, [1, fstate.numionstates], statestoskip, self.no_tda, self.frozencore, self.wfthres)
            # Printing to file
            for blockname, string in det_final.items():
                writestringtofile(string, "dets_final_mult"+str(fstate.mult))


        # Now doing initial state. Redefine necessary here.
        #det_init = get_dets_from_cis("Init_State1.out", "dummy", restricted_I, mults, stateIcharge, stateImult, totnuccharge,
        #                             statestoextract, statestoskip, no_tda, frozencore, wfthres)
        # RB simplification. Separate function for getting determinant-string for Initial State where only one.
        det_init = get_dets_from_single(self.stateI.outfile, self.stateI.restricted, self.stateI.charge, self.stateI.mult, self.totnuccharge, self.frozencore)
        # Printing to file
        for blockname, string in det_init.items():
            writestringtofile(string, "dets_init")

    def CAS_dets_prep(self):
        #CASSCF: GETTING GETERMINANTS FROM DETERMINANT-PRINTING OPTION in OUTPUTFILE
        #Combining with internal and external orbitals: internal_orbs,active_orbs,external_orbs
        #Initial
        print("Grabbing determinants from Initial State output")
        init_state = grab_dets_from_CASSCF_output(self.stateI.outfile)
        print("init_state:", init_state)
        #init_state_dict = [i.determinants for i in init_state]
        #init_state_dict2 = {Initialstate_mult : init_state_dict}
        #print("init_state_dict:", init_state_dict)
        #print("init_state_dict2:", init_state_dict2)
        det_init = format_ci_vectors(init_state[self.Initialstate_mult])
        #print("det_init:", det_init)
        # Printing to file
        writestringtofile(det_init, "dets_init")
        print("")
        #Final state. Just need to point to the one outputfile
        print("Grabbing determinants from Final State output")
        final_states = grab_dets_from_CASSCF_output(self.Finalstates[0].outfile)
        #print("final_states:", final_states)
        #final_states_dict = [i.determinants for i in final_states]
        #print("final_states_dict:", final_states_dict)
        #final_states_dict2 = {Initialstate_mult : final_states_dict}
        #print("final_states_dict2:", final_states_dict2)
        for fstate in self.Finalstates:
            #print("fstate: ", fstate)
            #print("fstate.mult :", fstate.mult)
            det_final = format_ci_vectors(final_states[fstate.mult])
            #print("det_final : ", det_final)
            # Printing to file
            writestringtofile(det_final, "dets_final_mult" + str(fstate.mult))

    def run_dyson_calc(self):
        print("Inside run_dyson_calc")
        # Run Wfoverlap to calculate Dyson norms. Will write to wfovl.out.
        # Check if binary exists
        if os.path.exists(self.path_wfoverlap) is False:
            print("Path {} does NOT exist !".format(self.path_wfoverlap))
            ashexit()
        print("Looping over Finalstate multiplicities")
        for fstate in self.Finalstates:
            print("\nRunning WFOverlap to calculate Dyson norms for Finalstate with mult: ", fstate.mult)
            # WFOverlap calculation needs files: AO_overl, mos_init, mos_final, dets_final, dets_init
            wfoverlapinput = """
            mix_aoovl=AO_overl
            a_mo=mos_final
            b_mo=mos_init
            a_det=dets_final
            b_det=dets_init
            a_mo_read=0
            b_mo_read=0
            ao_read=0
            moprint=1
            """
            wfoverlapinput = wfoverlapinput.replace("dets_final", "dets_final_mult"+str(fstate.mult))
            wfoverlapinput = wfoverlapinput.replace("mos_final", "mos_final-mult"+str(fstate.mult))

            #Calling wfoverlap
            run_wfoverlap(wfoverlapinput,self.path_wfoverlap,self.memory)

            #Grabbing Dyson norms from wfovl.out
            dysonnorms=grabDysonnorms()
            print(bcolors.OKBLUE,"\nDyson norms ({}):".format(len(dysonnorms)),bcolors.ENDC)
            print(dysonnorms)
            if len(dysonnorms) == 0:
                print("List of Dyson norms is empty. Something went wrong with WfOverlap calculation.")
                print("Setting Dyson norms to zero and continuing.")
                dysonnorms=len(fstate.IPs)*[0.0]
            self.finaldysonnorms=self.finaldysonnorms+dysonnorms


    def cleanup(self):
        files=['AO_overl', 'dets_final', 'dets_init', 'memlog', 'wfovl.inp', 'wfovl.out', 'mos_final', 'mos_init']
        print("Cleaning up files: ", files)
        for file in files:
            try:
                os.remove(file)
            except:
                pass

    def run(self):
        print_line_with_mainheader("PhotoElectron: run")
        module_init_time=time.time()

        print("Printing coordinates")
        self.fragment.print_coords()
        blankline()

        #Cleaning up old files to make sure no interference
        self.cleanup()

        #######################################################
        # CALL EOM, MRCI/MREOM, CAS or TDDFT to get states
        #######################################################
        if self.TDDFT is True:
            self.setup_ORCA_object()
            self.run_TDDFT()
            #Diff density
            if self.densities == 'SCF' or self.densities == 'All':
                self.make_diffdensities_SCF()
            # MO-spectrum 
            self.mo_spectrum()
            #For wfoverlap
            self.prepare_mos_file()
        elif self.EOM is True:
            print("Calling EOM")
            ashexit()
            self.setup_ORCA_object()
            self.run_EOM()

        elif self.CAS is True:
            print("CASSCF option active!")
            ashexit()
            self.setup_ORCA_object()
            if self.CASCI is True:
                print("CASCI option on! Initial state will be done with CASSCF while Final ionized states will do CAS-CI")
            #For wfoverlap
            self.prepare_mos_file()
        #Simplifies things. MREOM uses MRCI so let's use same logic.
        elif self.MRCI or self.MREOM is True:
            self.setup_ORCA_object()
            self.run_MRCI()
            #For wfoverlap
            self.prepare_mos_file()
            #Defining no MO-spectrum since WFT
            self.stk_alpha=[]; self.stk_beta=[]

        print("\nAll combined Final IPs:", self.FinalIPs)
        print("All combined Ion-state energies (au):", self.Finalionstates)



        #################################
        # PES intensities (Dyson norms)
        #################################
        if self.TDDFT is True:
            self.TDDFT_dets_prep()

        if self.noDyson is True:
            print("NoDyson is True. Setting all Dyson norms to 0.0")
            self.finaldysonnorms=[0.0]*self.numionstates
        else:
            #Call Dyson orb calc
            self.run_dyson_calc()

        print("")
        print(bcolors.OKBLUE, "Final combined Dyson norms ({}):".format(len(self.finaldysonnorms)), bcolors.ENDC)
        print(self.finaldysonnorms)
        print("self.FinalIPs:", self.FinalIPs)
        assert len(self.FinalIPs) == len(self.finaldysonnorms), "List of Dysonnorms not same size as list of IPs."
        print("")

        #########################
        #FINAL OUTPUT
        ########################
        #PRINT FINAL TABLE
        self.print_final_table()

        #Writing stuff to file. Useful for separate plotting of IPs and Dysonnorms
        print("")
        print("Printing IPs, Dyson-norms, MOs to file: PES-Results.txt")
        #Writing file in Configparser format for easy read-in below
        with open("PES-Results.txt", 'w') as resultfile:
            resultfile.write("[Results]\n")
            resultfile.write("IPs : {}\n".format(self.FinalIPs))
            resultfile.write("Dyson-norms : {}\n".format(self.finaldysonnorms))
            resultfile.write("MOs_alpha : {}\n".format(self.stk_alpha))
            resultfile.write("MOs_beta : {}\n".format(self.stk_beta))

        #################################
        # TDDFT DENSITIES (expensive)
        #################################
        #SCF density for Init and SCF-Finalstates have already been calculated
        #Here doing TDDFT-gradient calculation for each state if densities='All'
        if self.TDDFT is True:
            #Create Cube file of electron/spin density using orca_plot for INITIAL STATE
            if self.densities =='All':
                self.run_tddft_densities()

        print_time_rel(module_init_time, modulename='Photoelectron-run', moduleindex=2)
        return self.FinalIPs, self.finaldysonnorms


#####################################################################################
# Independent functions below
######################################################################################
def Read_old_PES_results():
    print("Reading file PES-Results.txt ...")
    # Parsing of files
    import json
    import configparser
    #from configparser import ConfigParser
    parser = configparser.ConfigParser()

    parser.read('PES-Results.txt')
    #Using JSON to load
    #From: https://stackoverflow.com/questions/335695/lists-in-configparser
    IPs = json.loads(parser.get("Results", "IPs"))
    dysonnorms = json.loads(parser.get("Results", "Dyson-norms"))
    mos_alpha = json.loads(parser.get("Results", "MOs_alpha"))
    mos_beta = json.loads(parser.get("Results", "MOs_beta"))

    return IPs, dysonnorms, mos_alpha, mos_beta

class bcolors:
    HEADER = '\033[95m' ; OKBLUE = '\033[94m'; OKGREEN = '\033[92m'; OKMAGENTA= '\033[95m'; WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'

eldict={'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Mo':42,'W':74,'Ru':44,'I':53}

#Readfile function
def readfile(filename):
  try:
    f=open(filename)
    out=f.readlines()
    f.close()
  except IOError:
    print('File {} does not exist!'.format(filename))
    ashexit()
  return out

#Get Atomic overlap matrix from GBW file
def saveAOmatrix(file, orcadir=None):
    NAO,Smat=get_smat_from_gbw(file, orcadir=orcadir)
    string='%i %i\n' % (NAO,NAO)
    for irow in range(NAO):
        for icol in range(NAO):
            string+='% .7e ' % (Smat[icol][irow])
        string+='\n'
    outfile='AO_overl'
    with open(outfile, 'w') as ofile:
        ofile.write(string)

#Get smat from GBW.
def get_smat_from_gbw(file1, file2='', orcadir=None):
    print("Inside get_smat_from_gbw")
    if os.path.isfile(file1) is False:
        print(f"file {file1} does exist")
        ashexit()
    print("file1:", file1)
    print("file2:", file2)
    if not file2:
      file2=file1

    # run orca_fragovl
    string=orcadir+'/orca_fragovl %s %s' % (file1,file2)
    try:
      proc=sp.Popen(string,shell=True,stdout=sp.PIPE,stderr=sp.PIPE)
    except OSError:
      print('Call has had some serious problems:',OSError)
      ashexit()
    comm=proc.communicate()
    #Python 3 decoding necessary
    comm=comm[0].decode('utf-8')
    #print(comm)
    #exit()
    out=comm.split('\n')

    #RBmod. Remove first lines (until MATRIX about to begin) as ORCA warnings can appear that change y-offset below
    indexskip=out.index("FRAGMENT-FRAGMENT OVERLAP MATRIX")
    out=out[indexskip:]


    # get size of matrix
    for line in reversed(out):
      #print line
      s=line.split()
      if len(s)>=1:
        NAO=int(line.split()[0])+1
        break

    # read matrix
    #Python3 conversion necessary here
    nblock=6
    ao_ovl=[ [ 0. for i in range(NAO) ] for j in range(NAO) ]
    for x in range(NAO):
      for y in range(NAO):
        block=int(x/nblock)
        xoffset=x%nblock+1
        #RB. Last number changed from 10 to 3 due to line-skipping above
        yoffset=block*(NAO+1)+y+3
        #Python3 issue with floats vs indices for block
        ao_ovl[x][y]=float( out[yoffset].split()[xoffset])

    return NAO,ao_ovl

#Get MO coefficients from GBW file.
def get_MO_from_gbw(filename,restr,frozencore,orcadir):
    # run orca_fragovl
    string=orcadir+'/orca_fragovl %s %s' % (filename,filename)
    try:
      proc=sp.Popen(string,shell=True,stdout=sp.PIPE,stderr=sp.PIPE)
    except OSError:
      print('Call have had some serious problems:',OSError)
      ashexit()
    comm=proc.communicate()
    #Python 3 decoding necessary
    comm=comm[0].decode('utf-8')
    #print(comm)
    data=comm.split('\n')
    #print(data)
    # get size of matrix
    for line in reversed(data):
      #print(line)
      s=line.split()
      if len(s)>=1:
        NAO=int(line.split()[0])+1
        break

    #job=QMin['IJOB']
    #restr=QMin['jobs'][job]['restr']

    # find MO block
    iline=-1
    while True:
      iline+=1
      if len(data)<=iline:
        print('MOs not found!')
        ashexit()
      line=data[iline]
      if 'FRAGMENT A MOs MATRIX' in line:
        break
    iline+=3
    # formatting
    nblock=6
    npre=11
    ndigits=16
    # get coefficients for alpha
    NMO_A=NAO
    MO_A=[ [ 0. for i in range(NAO) ] for j in range(NMO_A) ]
    for imo in range(NMO_A):
      #RB. Changed to floor division here
      jblock=imo//nblock
      jcol =imo%nblock
      for iao in range(NAO):
        shift=max(0,len(str(iao))-3)
        jline=iline + jblock*(NAO+1) + iao
        line=data[jline]
        val=float( line[npre+shift+jcol*ndigits : npre+shift+ndigits+jcol*ndigits] )
        MO_A[imo][iao]=val
    #Rb. changed to floor division here
    iline+=(NAO//nblock+1)*(NAO+1)
    #print("iline:", iline)
    # coefficients for beta
    if not restr:
      #RB. New definition of iline due to bug in original version of formatting change??
      iline=jline+2
      NMO_B=NAO
      MO_B=[ [ 0. for i in range(NAO) ] for j in range(NMO_B) ]
      for imo in range(NMO_B):
        #print("imo:", imo)
        #Changed to floor division here for Python3
        jblock=imo//nblock
        #print("jblock", jblock)
        jcol =imo%nblock
        #print("jcol:", jcol)
        #print("NAO:", NAO)
        #print("range(NAO)", range(NAO))
        #print("data[184]:", data[184])
        for iao in range(NAO):
          print("iao:", iao)
          shift=max(0,len(str(iao))-3)
          jline=iline + jblock*(NAO+1) + iao
          line=data[jline]
          val=float( line[npre+shift+jcol*ndigits : npre+shift+ndigits+jcol*ndigits] )
          MO_B[imo][iao]=val


    NMO=NMO_A      -  frozencore
    if restr:
        NMO=NMO_A      -  frozencore
    else:
        NMO=NMO_A+NMO_B-2*frozencore

    # make string
    string='''2mocoef
header
 1
MO-coefficients from Orca
 1
 %i   %i
 a
mocoef
(*)
''' % (NAO,NMO)
    x=0
    for imo,mo in enumerate(MO_A):
        if imo<frozencore:
            continue
        for c in mo:
            if x>=3:
                string+='\n'
                x=0
            string+='% 6.12e ' % c
            x+=1
        if x>0:
            string+='\n'
            x=0
    if not restr:
        x=0
        for imo,mo in enumerate(MO_B):
            if imo<frozencore:
                continue
            for c in mo:
                if x>=3:
                    string+='\n'
                    x=0
                string+='% 6.12e ' % c
                x+=1
            if x>0:
                string+='\n'
                x=0
    string+='orbocc\n(*)\n'
    x=0
    for i in range(NMO):
        if x>=3:
            string+='\n'
            x=0
        string+='% 6.12e ' % (0.0)
        x+=1

    return string


#RB. New function
#get determinant-string output for single-determinant case
def get_dets_from_single(logfile,restr,gscharge,gsmult,totnuccharge,frozencore):
    print("\nInside get_dets_from_single")
    # get infos from logfile
    data=readfile(logfile)
    infos={}
    for iline,line in enumerate(data):
      #print("line:", line)
      #if '# of contracted basis functions' in line:
      if 'Number of basis functions                   ...' in line:
        infos['nbsuse']=int(line.split()[-1])
      if 'Orbital ranges used for CIS calculation:' in line:
        s=data[iline+1].replace('.',' ').split()
        infos['NFC']=int(s[3])
        infos['NOA']=int(s[4])-int(s[3])+1
        infos['NVA']=int(s[7])-int(s[6])+1
        if restr:
          infos['NOB']=infos['NOA']
          infos['NVB']=infos['NVA']
        else:
          s=data[iline+2].replace('.',' ').split()
          infos['NOB']=int(s[4])-int(s[3])+1
          infos['NVB']=int(s[7])-int(s[6])+1
    if not 'NOA' in infos:
      charge=gscharge
      #charge=QMin['chargemap'][gsmult]
      nelec=float(totnuccharge-charge)
      infos['NOA']=int(nelec/2. + float(gsmult-1)/2. )
      infos['NOB']=int(nelec/2. - float(gsmult-1)/2. )
      infos['NVA']=infos['nbsuse']-infos['NOA']
      infos['NVB']=infos['nbsuse']-infos['NOB']
      infos['NFC']=0


    # get ground state configuration
    # make step vectors (0:empty, 1:alpha, 2:beta, 3:docc)
    if restr:
        occ_A=[ 3 for i in range(infos['NFC']+infos['NOA']) ]+[ 0 for i in range(infos['NVA']) ]
    if not restr:
        occ_A=[ 1 for i in range(infos['NFC']+infos['NOA']) ]+[ 0 for i in range(infos['NVA']) ]
        occ_B=[ 2 for i in range(infos['NFC']+infos['NOB']) ]+[ 0 for i in range(infos['NVB']) ]
    occ_A=tuple(occ_A)
    if not restr:
        occ_B=tuple(occ_B)

    # get eigenvectors
    eigenvectors={}
    eigenvectors[gsmult]=[]
    if restr:
        key=tuple(occ_A[frozencore:])
    else:
        key=tuple(occ_A[frozencore:]+occ_B[frozencore:])
    eigenvectors[gsmult].append( {key:1.0} )
    strings={}
    #print("Final (single-det case) eigenvectors:", eigenvectors)
    #print("format_ci_vectors(eigenvectors[gsmult] :", format_ci_vectors(eigenvectors[gsmult]))
    strings["dets."+str(gsmult)] = format_ci_vectors(eigenvectors[gsmult])
    return strings



#Get determinants from ORCA cisfile.
def get_dets_from_cis(logfile,cisfilename,restr,mults,gscharge,gsmult,totnuccharge,nstates_to_extract,nstates_to_skip,no_tda,frozencore,wfthres):
    print("Inside get_dets_from_cis")
    print("")
    print("logfile", logfile)
    print("cisfilename:", cisfilename)
    print("restr:", restr)
    print("mults:", mults)
    print("gscharge:", gscharge)
    print("gsmult:", gsmult)
    print("totnuccharge:", totnuccharge)
    print("nstates_to_extract:", nstates_to_extract)
    print("nstates_to_skip:", nstates_to_skip)
    print("no_tda:", no_tda)
    print("frozencore:", frozencore)
    print("wfthres", wfthres)
    print("nstates_to_extract:", nstates_to_extract)

    #print restr,mults,gsmult,nstates_to_extract
    #print "RB.....b"
    # get infos from logfile
    #logfile=os.path.join(os.path.dirname(filename),'ORCA.log')
    data=readfile(logfile)
    infos={}
    for iline,line in enumerate(data):
      #if '# of contracted basis functions' in line:
      if 'Number of basis functions                   ...' in line:
        infos['nbsuse']=int(line.split()[-1])
      if 'Orbital ranges used for CIS calculation:' in line:
        s=data[iline+1].replace('.',' ').split()
        infos['NFC']=int(s[3])
        infos['NOA']=int(s[4])-int(s[3])+1
        infos['NVA']=int(s[7])-int(s[6])+1
        if restr:
          infos['NOB']=infos['NOA']
          infos['NVB']=infos['NVA']
        else:
          s=data[iline+2].replace('.',' ').split()
          infos['NOB']=int(s[4])-int(s[3])+1
          infos['NVB']=int(s[7])-int(s[6])+1
    if not 'NOA' in  infos:
      charge=gscharge
      nelec=float(totnuccharge-charge)
      infos['NOA']=int(nelec/2. + float(gsmult-1)/2. )
      infos['NOB']=int(nelec/2. - float(gsmult-1)/2. )
      infos['NVA']=infos['nbsuse']-infos['NOA']
      infos['NVB']=infos['nbsuse']-infos['NOB']
      infos['NFC']=0
    else:
      # get all info from cis file
      CCfile=open(cisfilename,'rb')
      nvec  =struct.unpack('i', CCfile.read(4))[0]
      header=[ struct.unpack('i', CCfile.read(4))[0] for i in range(8) ]
      print(infos)
      print(header)
      if infos['NOA']!=header[1]-header[0]+1:
        print('Number of orbitals in %s not consistent' % filename)
        ashexit()
      if infos['NVA']!=header[3]-header[2]+1:
        print('Number of orbitals in %s not consistent' % filename)
        ashexit()
      if not restr:
        if infos['NOB']!=header[5]-header[4]+1:
          print('Number of orbitals in %s not consistent' % filename)
          ashexit()
        if infos['NVB']!=header[7]-header[6]+1:
          print('Number of orbitals in %s not consistent' % filename)
          ashexit()
      if no_tda:
        nstates_onfile=nvec/2
      else:
        nstates_onfile=nvec


    # get ground state configuration
    # make step vectors (0:empty, 1:alpha, 2:beta, 3:docc)
    if restr:
        occ_A=[ 3 for i in range(infos['NFC']+infos['NOA']) ]+[ 0 for i in range(infos['NVA']) ]
    if not restr:
        occ_A=[ 1 for i in range(infos['NFC']+infos['NOA']) ]+[ 0 for i in range(infos['NVA']) ]
        occ_B=[ 2 for i in range(infos['NFC']+infos['NOB']) ]+[ 0 for i in range(infos['NVB']) ]
    occ_A=tuple(occ_A)
    if not restr:
        occ_B=tuple(occ_B)

    # get infos
    nocc_A=infos['NOA']
    nvir_A=infos['NVA']
    nocc_B=infos['NOB']
    nvir_B=infos['NVB']

    # get eigenvectors
    eigenvectors={}
    for imult,mult in enumerate(mults):
        eigenvectors[mult]=[]
        if mult==gsmult:
            # add ground state
            if restr:
                key=tuple(occ_A[frozencore:])
            else:
                key=tuple(occ_A[frozencore:]+occ_B[frozencore:])
            eigenvectors[mult].append( {key:1.0} )
        #print("struct.unpack('d', CCfile.read(8))[0]:", struct.unpack('d', CCfile.read(8))[0])
        #for istate in range(nstates_to_extract[mult-1]):
        for istate in range(nstates_to_extract[-1] - 1):
            CCfile.read(40)
            dets={}
            #print("dets:", dets)
            #print("header:", header)
            #print("--------")
            #print("range(header[0],header[1]+1):", range(header[0],header[1]+1))
            for iocc in range(header[0],header[1]+1):
              #print("--------")
              #print("range(header[2],header[3]+1):", range(header[2],header[3]+1))
              for ivirt in range(header[2],header[3]+1):
                #print("istate, iocc, ivirt", istate, iocc, ivirt)
                #print("here1")
                dets[ (iocc,ivirt,1) ]=struct.unpack('d', CCfile.read(8))[0]
            #print("1 case done!!")
            #pprint.pprint(dets)
            if not restr:
              #print("here2")
              #print("range(header[4],header[5]+1):", range(header[4],header[5]+1))
              for iocc in range(header[4],header[5]+1):
                #print("range(header[6],header[7]+1):", range(header[6],header[7]+1))
                for ivirt in range(header[6],header[7]+1):
                  #print("istate, iocc, ivirt", istate, iocc, ivirt)
                  #print("struct.unpack('d', CCfile.read(8))[0]:", struct.unpack('d', CCfile.read(8))[0])
                  dets[ (iocc,ivirt,2) ]=struct.unpack('d', CCfile.read(8))[0]
            if no_tda:
              CCfile.read(40)
              for iocc in range(header[0],header[1]+1):
                for ivirt in range(header[2],header[3]+1):
                  dets[ (iocc,ivirt,1) ]+=struct.unpack('d', CCfile.read(8))[0]
                  dets[ (iocc,ivirt,1) ]/=2.
              if not restr:
                for iocc in range(header[4],header[5]+1):
                  for ivirt in range(header[6],header[7]+1):
                    #print("rb here...")
                    dets[ (iocc,ivirt,2) ]+=struct.unpack('d', CCfile.read(8))[0]
                    dets[ (iocc,ivirt,2) ]/=2.

            #pprint.pprint(dets)
            # truncate vectors
            norm=0.
            for k in sorted(dets,key=lambda x: dets[x]**2,reverse=True):
                factor=1.
                if norm>factor*wfthres:
                    del dets[k]
                    continue
                norm+=dets[k]**2
            #pprint.pprint(dets)
            # create strings and expand singlets
            dets2={}
            if restr:
                for iocc,ivirt,dummy in dets:
                    # singlet
                    if mult==1:
                        # alpha excitation
                        key=list(occ_A)
                        key[iocc]=2
                        key[ivirt]=1
                        dets2[tuple(key)]=dets[ (iocc,ivirt,dummy) ]*math.sqrt(0.5)
                        # beta excitation
                        key[iocc]=1
                        key[ivirt]=2
                        dets2[tuple(key)]=dets[ (iocc,ivirt,dummy) ]*math.sqrt(0.5)
                    # triplet
                    elif mult==3:
                        key=list(occ_A)
                        key[iocc]=1
                        key[ivirt]=1
                        dets2[tuple(key)]=dets[ (iocc,ivirt,dummy) ]
            else:
                for iocc,ivirt,dummy in dets:
                    if dummy==1:
                        key=list(occ_A+occ_B)
                        key[iocc]=0
                        key[ivirt]=1
                        dets2[tuple(key)]=dets[ (iocc,ivirt,dummy) ]
                    elif dummy==2:
                        key=list(occ_A+occ_B)
                        key[infos['NFC']+nocc_A+nvir_A + iocc]=0
                        key[infos['NFC']+nocc_A+nvir_A + ivirt]=2
                        dets2[tuple(key)]=dets[ (iocc,ivirt,dummy) ]
            #pprint.pprint(dets2)
            # remove frozen core
            dets3={}
            for key in dets2:
                problem=False
                if restr:
                    if any( [key[i]!=3 for i in range(frozencore) ] ):
                        problem=True
                else:
                    if any( [key[i]!=1 for i in range(frozencore) ] ):
                        problem=True
                    if any( [key[i]!=2 for i in range(nocc_A+nvir_A+frozencore, nocc_A+nvir_A + 2*frozencore) ] ):
                        problem=True
                if problem:
                    print('WARNING: Non-occupied orbital inside frozen core! Skipping ...')
                    continue
                    #sys.exit(86)
                if restr:
                    key2=key[frozencore:]
                else:
                    key2=key[frozencore:frozencore+nocc_A+nvir_A] + key[nocc_A+nvir_A+2*frozencore:]
                dets3[key2]=dets2[key]
            #pprint.pprint(dets3)
            # append
            eigenvectors[mult].append(dets3)
        # skip extra roots
        #RB Removing mult in range. Weird
        for istate in range(nstates_to_skip[-1]):
            CCfile.read(40)
            for iocc in range(header[0],header[1]+1):
              for ivirt in range(header[2],header[3]+1):
                CCfile.read(8)
            if not restr:
              for iocc in range(header[4],header[5]+1):
                for ivirt in range(header[6],header[7]+1):
                  CCfile.read(8)
            if no_da:
              CCfile.read(40)
              for iocc in range(header[0],header[1]+1):
                for ivirt in range(header[2],header[3]+1):
                  CCfile.read(8)
              if not restr:
                for iocc in range(header[4],header[5]+1):
                  for ivirt in range(header[6],header[7]+1):
                    CCfile.read(8)


    strings={}
    #print("Final (CIS) eigenvectors:", eigenvectors)
    for imult,mult in enumerate(mults):
        filename='dets.%i' % mult
        strings[filename]=format_ci_vectors(eigenvectors[mult])
    return strings

def format_ci_vectors(ci_vectors):

    # get nstates, norb and ndets
    alldets=set()
    for dets in ci_vectors:
        for key in dets:
            alldets.add(key)
    ndets=len(alldets)
    nstates=len(ci_vectors)
    norb=len(next(iter(alldets)))

    string='%i %i %i\n' % (nstates,norb,ndets)
    for det in sorted(alldets,reverse=True):
        for o in det:
            if o==0:
                string+='e'
            elif o==1:
                string+='a'
            elif o==2:
                string+='b'
            elif o==3:
                string+='d'
        for istate in range(len(ci_vectors)):
            if det in ci_vectors[istate]:
                string+=' %11.7f ' % ci_vectors[istate][det]
            else:
                string+=' %11.7f ' % 0.
        string+='\n'
    return string

#Run wfoverlap program
def run_wfoverlap(wfoverlapinput,path_wfoverlap,memory):
    wfoverlapfilefile = open('wfovl.inp', 'w')
    for l in wfoverlapinput:
        wfoverlapfilefile.write(l)
    wfoverlapfilefile.close()
    wfcommand="{} -m {} -f wfovl.inp".format(path_wfoverlap,memory)
    print("Running wfoverlap program:")
    print("may take a while...")
    print(wfcommand)
    print("Using memory: {} MB".format(memory))
    print("OMP num threads available to WFoverlap: ", os.environ["OMP_NUM_THREADS"])
    os.system('ldd {}'.format(path_wfoverlap))
    
    try:
        proc=sp.Popen(wfcommand,shell=True,stdout=sp.PIPE,stderr=sp.PIPE)
        wfoverlapout=proc.communicate()
        wfoverlapout=wfoverlapout[0].decode('utf-8')
        wfoverlapout=wfoverlapout.split('\n')
        with open("wfovl.out", 'w') as f:
            for b in wfoverlapout:
                f.write(b+'\n')
    except OSError:
        print("Problem calling wfoverlap program.")
    print("Wfoverlap done! See outputfile: wfovl.out")
    return

#Get Dysonnorms from output of wfoverlap
def grabDysonnorms():
    with open("wfovl.out") as wout:
        out=wout.readlines()
    #Getting Dyson norms from output
    dysonnorms=[]
    dysonorbs=[]
    dysonorbitalgrab=False
    for line in out:
        if dysonorbitalgrab==True:
            if 'MO' in line:
                dysonorbs.append(float(line.split()[-1]))
        if '<PsiA ' in line and dysonorbitalgrab==False:
            dysonnorms.append(float(line.split()[-1]))
        if 'Dyson orbitals in reference' in line:
            dysonorbitalgrab=True
    return dysonnorms


#Calculate HOMO number (0-indexing) from nuclear charge and total charge
def HOMOnumber(totnuccharge,charge,mult):
    numel=totnuccharge-charge
    HOMOnum_a="unset";HOMOnum_b="unset"
    offset=-1
    if mult == 1:
        #RHF case. HOMO is numel/2 -1
        HOMOnum_a=(numel/2)+offset
        HOMOnum_b=(numel/2)+offset
    elif mult > 1:
        #UHF case.
        numunpel=mult-1
        Doubocc=(numel-numunpel)/2
        HOMOnum_a=Doubocc+numunpel+offset
        HOMOnum_b=Doubocc+offset
    return int(HOMOnum_a),int(HOMOnum_b)


#CASSCF: Grabbing first root energy. Simplified because of problem
def casscfenergygrab(file):
    #grab=False
    string='Final CASSCF energy       :'
    with open(file) as f:
        for line in f:
            if string in line:
                Energy=float(line.split()[4])
                return Energy
                #Changing from 5 to -2
                #CIPSI: 5 Regular CASSCF: -2  ?
                #Pretty ugly. to be fixed. TODO
                #Energy=float(line.split()[-2])
                #if Energy == 0.0:
                #    Energy=float(line.split()[5])
            #if 'CAS-SCF STATES FOR BLOCK' in line:
            #    grab=True

#CASSCF: Grabbing all root energies
#Slightly tricky function because output differs for ICE-CASSCF and regular CASSCF.
#Should be good now.
def casscf_state_energies_grab(file):
    Finished=False
    grab=False
    mult_dict={}
    state_energies=[];Energy=0.0
    #string='STATE '
    #string2='ROOT '
    with open(file) as f:
        for line in f:
            #Stop grabbing lines once we have reached end of table
            if 'SA-CASSCF TRANSITION ENERGIES' in line:
                grab=False
            if 'Spin-Determinant CI Printing' in line:
                grab=False
            if 'Extended CI Printing (values > TPrintWF)' in line:
                grab=False
            #Grabbing STATE lines
            if grab is True and 'STATE ' in line:
                Energy=float(line.split()[5])
                state_energies.append(Energy)
                if len(state_energies) == roots:
                    mult_dict[mult] = state_energies
                    grab=False
            if grab is True and 'ROOT ' in line:
                Energy=float(line.split()[3])
                state_energies.append(Energy)
                if len(state_energies) == roots:
                    mult_dict[mult] = state_energies
                    grab=False
            if Finished is True and 'CAS-SCF STATES FOR BLOCK' in line:
                #New mult block. Resetting state-energies.
                state_energies=[];Energy=0.0
                mult=int(line.split()[6])
                #roots=int(line.split()[8])
                roots=int(line.split()[7:9][-1].replace('NROOTS=',''))
                grab=True
            #Only grabbing lines once CASSCF calc has converged
            if 'Final CASSCF energy' in line:
                Finished=True
    return mult_dict

#MRCI: Grabbing all root energies
#If SORCI then multiple MRCI output
def mrci_state_energies_grab(file,SORCI=False):
    
    #If SORCI True then multiple MRCI output sections. we want last one
    if SORCI is True:
        final_part=False
    else:
        final_part=True
    
    grab=False
    blockgrab=False
    grab_blockinfo=False
    block_dict={}
    mult_dict={}
    state_energies=[];Energy=0.0
    string='STATE '
    prev_grabbed_blockinfo=False
    with open(file) as f:
        for line in f:
            #print("line:", line)
            #print("prev_grabbed_blockinfo:", prev_grabbed_blockinfo)
            #print("grab_blockinfo:", grab_blockinfo)
            #Note. Grabbing block info from CASSCF output
            if '<<<<<<<<<<<<<<<<<<INITIAL CI STATE CHECK>>>>>>>>>>>>>>>>>>' in line:
                if prev_grabbed_blockinfo is False:
                    grab_blockinfo = True
                    prev_grabbed_blockinfo=True
                    continue
                else:
                    grab_blockinfo=False
            if grab_blockinfo is True:
                if 'BLOCK' in line:
                    blocknum = int(line.split()[1])
                    mult = int(line.split()[3])
                    roots = int(line.split("=")[-1])
                    block_dict[blocknum] = (mult,roots)
                    #print("block_dict:", block_dict)
                #Only reading 2 blocks (two multiplicities)
                #Unncessary?
                if len(block_dict) == 2:
                    grab_blockinfo = False
            #Grabbing actual MRCI state energies
            if grab is True and string in line:
                Energy=float(line.split()[3])
                state_energies.append(Energy)
                if len(state_energies) == current_roots:
                    mult_dict[currentmult] = state_energies
                    #print("mult_dict:", mult_dict)
                    state_energies=[]
            #Getting info about what block we are currently reading in the output
            if final_part is True:
                if '*              CI-BLOCK' in line:
                    blockgrab=True
                    currentblock=int(line.split()[-2])
                    currentmult=block_dict[currentblock][0]
                    current_roots = block_dict[currentblock][1]
            if 'TRANSITION ENERGIES' in line:
                grab = False
            if blockgrab is True:
                if 'Unselected CSF estimate:' in line:
                    grab=True
            if 'S O R C I (DDCI3-STEP)' in line:
                final_part=True
    return mult_dict


#CASSCF: Grab orbital ranges
def casscf_orbitalranges_grab(file):
    grab=False
    with open(file) as f:
        for line in f:
            if grab is True:
                if 'Internal' in line:
                    internal=int(line.split()[-2])
                if 'Active' in line:
                    active=int(line.split()[-2])
                if 'External' in line:
                    external=int(line.split()[-2])
            if 'Determined orbital ranges:' in line:
                grab=True
            if 'Number of rotation parameters' in line:
                grab=False

    return internal,active,external

def Gaussian(x, mu, strength, sigma):
    "Produces a Gaussian curve"
    bandshape = (strength)  * np.exp(-1*((x-mu))**2/(2*(sigma**2)))
    return bandshape


#Grab determinants from CASSCF-ORCA output with option PrintWF det
def grab_dets_from_CASSCF_output(file):

    class state_dets():
        def __init__(self, root,energy,mult):
            self.mult = mult
            self.root = root
            self.energy = energy
            self.determinants = {}
            self.configurations = {}
    list_of_states=[]
    detgrab=False
    grabrange=False
    with open(file) as f:
        for line in f:
            #Getting orbital ranges
            # Internal (doubly occ)and external orbitals (empty)
            if grabrange is True:

                if 'Internal' in line:
                    internal=int(line.split()[-2])
                    internal_tuple = tuple([3] * internal)
                if 'Active' in line:
                    active=int(line.split()[-2])
                if 'External' in line:
                    external=int(line.split()[-2])
                    external_tuple = tuple([0] * external)
            if 'Determined orbital ranges:' in line:
                grabrange=True
            if 'Number of rotation parameters' in line:
                grabrange=False

            if 'SA-CASSCF TRANSITION ENERGIES' in line:
                detgrab=False
            if 'DENSITY MATRIX' in line:
                detgrab=False
            if detgrab is True:

                if '[' in line and 'CFG' not in line:
                    det = line.split()[0]
                    #print("det:", det)
                    detlist=[i for i in det.replace('[','').replace(']','')]
                    detlist2=[]
                    #print("detlist:", detlist)
                    #Sticking with labelling: 3: doubly occ, 0: empty, 1 for up-alpha, 2 for down-beta
                    for j in detlist:
                        if j == '2':
                            detlist2.append(3)
                        elif j == '0':
                            detlist2.append(0)
                        elif j == 'u':
                            detlist2.append(1)
                        elif j == 'd':
                            detlist2.append(2)
                    #print("detlist2:", detlist2)
                    #combining
                    det_tuple=internal_tuple+tuple(detlist2)+external_tuple
                    #print("det_tuple : ", det_tuple)
                    #This is the CI coefficient of the determinant
                    coeff = float(line.split()[-1])
                    state.determinants[det_tuple] = coeff
                if '[' in line and 'CFG' in line:
                    cfg = line.split()[0]
                    #This is the weight (CI coefficient squared)
                    weight = float(line.split()[-1])
                    state.configurations[cfg] = weight

                    #CASE: CFG contains only 2 and 0s. That means a situation where Det info is not printed in CASSCF-module (but printed in MRCI)
                    #Removed in July-ish 2020 after Vijay update
                    #Added back in 29 Nov 2020 since still cases where det is not printed. Taking square of weight
                    #Vijay probably only changed the MRCI behaviour not the CASSCF behaviour
                    if '1' not in cfg:
                        #print("cfg : ", cfg)
                        print("WARNING: Found CFG with no SOMO.")
                        print("WARNING: Det info is probably missing (not printed). Taking CFG and weight and converting to determinant")
                        #print("line:", line)
                        bla = cfg.replace('[','').replace(']','').replace('CFG','')
                        #print("bla:", bla)
                        det = bla.replace(str(2),str(3))
                        #print("det:", det)
                        det2 = [int(i) for i in det]
                        det_tuple = internal_tuple + tuple(det2) + external_tuple
                        #print("det_tuple: ", det_tuple)
                        #Taking square-root
                        state.determinants[det_tuple] = math.sqrt(weight)

                if 'ROOT ' in line:
                    print("line:", line)
                    root=int(line.split()[1][0])
                    energy = float(line.split()[3])
                    state = state_dets(root, energy, mult)
                    list_of_states.append(state)
            if 'CAS-SCF STATES FOR BLOCK' in line:
                print("CAS LINE: ", line)
                mult =int(line.split()[6])
                print("Setting mult to: ", mult)
                detgrab = False
                print("Det grab set to False")
            if '  Extended CI Printing (values > TPrintWF)' in line:
                print("Det grab set to True")
                detgrab=True
            if '  Spin-Determinant CI Printing' in line:
                print("Det grab set to True")
                detgrab=True

    #print("list_of_states:", list_of_states)
    #print(list_of_states[0])
    #print(list_of_states[0].determinants)
    #print(list_of_states[0].configurations)


    #Going through
    for n,state in enumerate(list_of_states):
        print("------------------------")
        print("This is state {}  with mult {} and energy {} and root {}".format(n,state.mult, state.energy, state.root))
        print("length of state CFGs :", len(state.configurations))
        print("length of state determinants :", len(state.determinants))
        if len(state.determinants) == 0:
            print("WARNING!!! No determinant output found.")
            print("Must be because CFG and det is the same. Using CFG info ")
            print("WARNING!!!")
            print("state.configurations : ", state.configurations)
            for cfg in state.configurations.items():
                bla = cfg[0].replace('[','').replace(']','').replace('CFG','')
                det = bla.replace(str(2),str(3))
                det2 = [int(i) for i in det]
                #det_tuple = tuple(det2)
                det_tuple = internal_tuple + tuple(det2) + external_tuple
                coeff = cfg[1]
                state.determinants[det_tuple] = coeff
            #print("state.determinants: ", state.determinants)

    #print("list_of_states:", list_of_states)

    mults = list(set([state.mult for state in list_of_states]))
    #Return a dictionary with all mults and all states
    final = {}
    for mult in mults:
        final[mult] = [state.determinants for state in list_of_states if state.mult == mult ]
    #print("final :", final)
    return final



#Grab determinants from MRCI-ORCA output with option PrintWF det
def grab_dets_from_MRCI_output(file, SORCI=False):
    #If SORCI True then multiple MRCI output sections. we want last one
    if SORCI is True:
        final_part=False
    else:
        final_part=True

    class state_dets():
        def __init__(self, root,energy,mult):
            self.mult = mult
            self.root = root
            self.energy = energy
            self.determinants = {}
            self.configurations = {}
            self.ciblock = None
    list_of_states=[]
    detgrab=False
    grabrange=False
    dummycount=0
    with open(file) as f:
        for line in f:
            if 'Program Version 4.2.1 -  RELEASE' in line:
                ashexit(errormessage='MRCI-determinant read will not work for ORCA 4.2.1 and older!')
            if 'Total number of orbitals            ...' in line:
                totorbitals=int(line.split()[-1])
                #print("totorbitals:", totorbitals)
            #Getting orbital ranges
            # Internal (doubly occ)and external orbitals (empty)
            if grabrange is True:

                if 'Internal' in line:
                    internal=int(line.split()[-2])
                    internal_tuple = tuple([3] * internal)
                if 'Active' in line:
                    active=int(line.split()[-2])
                if 'External' in line:
                    external=int(line.split()[-2])
                    external_tuple = tuple([0] * external)
                    #First index of external list
                    external_first=int(line.split()[1])
                    
            if 'Determined orbital ranges:' in line:
                grabrange=True
            if 'Number of rotation parameters' in line:
                grabrange=False

            if 'SA-CASSCF TRANSITION ENERGIES' in line:
                detgrab=False
            if 'DENSITY MATRIX' in line or 'DENSITY GENERATION' in line:
                #print("here. density line. Setting detgrab to false")
                detgrab=False
            if 'TRANSITION ENERGIES' in line:
                detgrab=False
                #print("here. transitio energies line. Setting detgrab to false")
            #Determining CI BLOCK
            #if 'CI BLOCK' in line:


            #What block we are reading through
            if '          CI-BLOCK' in line:
                #Setting detgrab to False for each new CI-block. Prevents us from grabbing State-lines for Reference-space CI
                detgrab = False
                ciblock = int(line.split()[-2])
                #print("Inside CI Block : ", ciblock)
            if 'Building a CAS' in line:
                #Setting mult here. mult will be used when creating state
                mult = int(line.split()[-1])

            if detgrab is True:

                #Here reading CFG line. Grabbing configuration
                #Also
                if '[' in line and 'CFG' in line:
                    
                    hole_indices=[]
                    particle_indices=[]
                    #print("----------------------------------------------------------------------------------------")
                    #print("line:", line)
                    cfg = line.split()[-1]
                    #This is the weight of this configuration, not CI coefficient
                    weight = float(line.split()[0])
                    #print("weight:", weight)
                    state.configurations[cfg] = weight
                    #Reading CFG line and determining hole/particle excitations outside active space
                    if 'h---h---' in line and line.count('p')==0:
                        #CAS excitation
                        #print("Assignment: 0 HOLE  0 PARTICLE")
                        hole_indices=[]
                        particle_indices=[]
                        #print("hole_indices:", hole_indices)                         
                        #print("particle_indices:", particle_indices)    
                    elif 'h---h---' in line and line.count('p')==1:
                        #0-hole 1-particle
                        hole_indices=[]
                        #print("Assignment: 0 HOLE  1 PARTICLE")
                        particle_index = int(find_between(line,']p','\n'))
                        particle_indices.append(particle_index)
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)    
                    elif 'h---h---' in line and line.count('p')==2:
                        #0-hole 2-particle
                        hole_indices=[]
                        #print("Assignment: 0 HOLE  2 PARTICLE")
                        particle_indices = [int(i) for i in find_between(line,']p','\n').replace('p','').split()]
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)                
                    elif 'h---h ' in line and line.count('p')==0:
                        #1-hole 0-particle
                        particle_indices=[]
                        #print("Assignment: 1 HOLE  0 PARTICLE")                             
                        hole_index=int(find_between(line,'h---h','[').strip())
                        hole_indices.append(hole_index)
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)  
                    elif 'h---h ' in line and line.count('p')==1:
                        #1-hole 1-particle
                        #print("Assignment: 1 HOLE  1 PARTICLE")
                        hole_index=int(find_between(line,'h---h','[').strip())
                        hole_indices.append(hole_index)
                        particle_index = int(find_between(line,']p','\n'))
                        particle_indices.append(particle_index)                         
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)  
                    elif 'h---h ' in line and line.count('p')==2:
                        #1-hole 2-particle
                        #print("Assignment: 1 HOLE  2 PARTICLE")
                        hole_index=int(find_between(line,'h---h','[').strip())
                        hole_indices.append(hole_index)
                        particle_indices = [int(i) for i in find_between(line,']p','\n').replace('p','').split()]
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)
                    elif 'CFG h ' in line and line.count('p')==0:
                        # 2-hole 0-particle
                        #print("Assignment: 2 HOLE  0 PARTICLE")
                        hole_indices=[int(i) for i in find_between(line,'CFG h','[').replace('h','').split()]
                        particle_indices=[]
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)
                    elif 'CFG h ' in line and line.count('p')==1:
                        # 2-hole 1-particle
                        #print("Assignment: 2 HOLE  1 PARTICLE")
                        hole_indices=[int(i) for i in find_between(line,'CFG h','[').replace('h','').split()]
                        particle_index = int(find_between(line,']p','\n'))
                        particle_indices.append(particle_index)   
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)
                    elif 'CFG h ' in line and line.count('p')==2:
                        # 2-hole 2-particle
                        #print("Assignment: 2 HOLE  2 PARTICLE")
                        hole_indices=[int(i) for i in find_between(line,'CFG h','[').replace('h','').split()]
                        particle_indices = [int(i) for i in find_between(line,']p','\n').replace('p','').split()]
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)                                                    
                    else:
                        print("Bad line. exiting")
                        ashexit()    
                        
                if '[' in line and 'CFG' not in line:
                    dummycount+=1
                    #print("dummycount:", dummycount)
                    #print("Determinant line:", line)
                    det = line.split()[0]
                    #print("det:", det)
                    detlist=[i for i in det.replace('[','').replace(']','')]
                    detlist2=[]
                    #print("detlist:", detlist)
                    #Sticking with labelling: 3: doubly occ, 0: empty, 1 for up-alpha, 2 for down-beta
                    for j in detlist:
                        if j == '2':
                            detlist2.append(3)
                        elif j == '0':
                            detlist2.append(0)
                        elif j == 'u':
                            detlist2.append(1)
                        elif j == 'd':
                            detlist2.append(2)
                    #print("detlist2:", detlist2)
                    
                    
                    
                    #Modifying internal_tuple for possible holes
                    #print("internal_tuple :", internal_tuple)
                    #CASE: 1 HOLES  0 PARTICLES:
                    if len(hole_indices) == 1 and len(particle_indices) == 0:
                        holeindex=hole_indices[0]
                        #print("Modifying internal_tuple")                        
                        lst_internaltuple=list(internal_tuple)
                        #Getting spinlabel of internal electron where hole was created from detlist (first index in bracket)
                        spinlabelh1p0int=detlist2[0]
                        lst_internaltuple[holeindex] = spinlabelh1p0int
                        modinternal_tuple=tuple(lst_internaltuple)
                        #print("Mod internal_tuple :", modinternal_tuple)
                        #Removing hole orb from detlist
                        moddetlist2=detlist2[1:]
                        #print("Mod active tuple :", detlist2)
                        #Unmodified external
                        modexternal_tuple=external_tuple
                    #CASE: 2 HOLES  0 PARTICLES:
                    elif len(hole_indices) == 2 and len(particle_indices) == 0:
                        moddetlist2=detlist2
                        holeindex1=hole_indices[0]
                        holeindex2=hole_indices[1]
                        
                        #SubCase: Double internal hole. Means no spin-label in bracket for this
                        if holeindex1 == holeindex2:
                            #print("Same holeindex")
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = 0
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #No modification to detlist2 needed
                        #Subcase: Not doubly internal hole.
                        else:
                            #print("Not same holeindex")
                            #print("moddetlist2:", moddetlist2)
                            spinlabelh2p0int_1=detlist2[0]
                            spinlabelh2p0int_2=detlist2[1]
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = spinlabelh2p0int_1
                            lst_internaltuple[holeindex2] = spinlabelh2p0int_2
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #Modification to detlist
                            moddetlist2=detlist2[2:]
                        #Unmodified external
                        modexternal_tuple=external_tuple
                    #CASE: 0 HOLE  1 PARTICLE:
                    elif len(hole_indices) == 0 and len(particle_indices) == 1:
                        particleindex1=particle_indices[0]
                        #Particleposition in external list
                        particleposition=particleindex1-external_first
                        #print("particleposition in external list:", particleposition)
                        #print("Modifying external")
                        spinlabelh0p1ext=detlist2[-1]
                        lst_externaltuple=list(external_tuple)
                        lst_externaltuple[particleposition] = spinlabelh0p1ext
                        modexternal_tuple=tuple(lst_externaltuple)
                        #print("Mod external tuple :", modexternal_tuple)
                        #Removed particle spin from detlist
                        moddetlist2=detlist2[:-1]
                        
                        #Unmodified internal
                        modinternal_tuple=internal_tuple
                    #CASE: 1 HOLE  1 PARTICLE:
                    elif len(hole_indices) == 1 and len(particle_indices) == 1:
                        holeindex=hole_indices[0]
                        particleindex1=particle_indices[0]
                        #Particleposition in external list
                        particleposition=particleindex1-external_first
                        
                        #Modifying internal list
                        spinlabelh1p1int=detlist2[0]
                        lst_internaltuple=list(internal_tuple)
                        lst_internaltuple[holeindex] = spinlabelh1p1int
                        modinternal_tuple=tuple(lst_internaltuple)
                        #print("Mod internal_tuple :", modinternal_tuple)
                        #Modifying external list
                        spinlabelh1p1ext=detlist2[-1]
                        lst_externaltuple=list(external_tuple)
                        lst_externaltuple[particleposition] = spinlabelh1p1ext
                        modexternal_tuple=tuple(lst_externaltuple)
                        #print("Mod external tuple :", modexternal_tuple)
                        #Modifying detlist
                        moddetlist2=detlist2[1:-1]
                    #CASE: 0 HOLE  2 PARTICLES:                        
                    elif len(hole_indices) == 0 and len(particle_indices) == 2:
                        particleindex1=particle_indices[0]
                        particleindex2=particle_indices[1]
                        #Particleposition in external list
                        particleposition1=particleindex1-external_first
                        particleposition2=particleindex2-external_first                        
                        #Modifying external list
                        #SubCase: Double particle position. Means no spin-label in bracket for this
                        if particleindex1 == particleindex2:
                            #print("Particle indices the same")
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = 3
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            
                            #Modifying active detlist
                            moddetlist2=detlist2
                            
                        else:
                            #print("Particle indices NOT the same")
                            spinlabelh0p2_1ext=detlist2[-2]
                            spinlabelh0p2_2ext=detlist2[-1]
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = spinlabelh0p2_1ext
                            lst_externaltuple[particleposition2] = spinlabelh0p2_2ext
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            #Modifying active detlist
                            moddetlist2=detlist2[:-2]                        
                        
                        #Unmodified internal
                        modinternal_tuple=internal_tuple
                        
                        
                    #CASE: 1 HOLE  2 PARTICLES:                        
                    elif len(hole_indices) == 1 and len(particle_indices) == 2:
                        #Grabbing immediately
                        moddetlist2=detlist2
                        holeindex=hole_indices[0]
                        particleindex1=particle_indices[0]
                        particleindex2=particle_indices[1]
                        #Particleposition in external list
                        particleposition1=particleindex1-external_first
                        particleposition2=particleindex2-external_first
                        
                        #Modifying internal list
                        spinlabelh1p2_1int=detlist2[0]
                        lst_internaltuple=list(internal_tuple)
                        lst_internaltuple[holeindex] = spinlabelh1p2_1int
                        modinternal_tuple=tuple(lst_internaltuple)
                        #print("Mod internal_tuple :", modinternal_tuple)
                        #Modifying active detlist for hole
                        moddetlist2=moddetlist2[1:]
                        
                        #Modifying external list
                        #SubCase: Double particle position. Means no spin-label in bracket for this
                        if particleindex1 == particleindex2:
                            #print("Particle indices the same")
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = 3
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            

                            
                        else:
                            #print("Particle indices NOT the same")
                            spinlabelh1p2_1ext=detlist2[-2]
                            spinlabelh1p2_2ext=detlist2[-1]
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = spinlabelh1p2_1ext
                            lst_externaltuple[particleposition2] = spinlabelh1p2_2ext
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            #Modifying active detlist
                            moddetlist2=moddetlist2[:-2]
                        

                    #CASE: 2 HOLES  2 PARTICLES:                       
                    elif len(hole_indices) == 2 and len(particle_indices) == 2:
                        #Grab this immediately
                        moddetlist2=detlist2
                        holeindex1=hole_indices[0]
                        holeindex2=hole_indices[1]
                        particleindex1=particle_indices[0]
                        particleindex2=particle_indices[1]
                        #Particleposition in external list
                        particleposition1=particleindex1-external_first
                        particleposition2=particleindex2-external_first
                        
                        #Modifying internal list
                        #SubCase: Double internal hole. Means no spin-label in bracket for this
                        if holeindex1 == holeindex2:
                            #print("Same holeindex")
                            spinlabelh2p2int=0
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = spinlabelh2p2int
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #No modification to detlist2 needed
                        #Subcase: Not doubly internal hole.
                        else:
                            #print("Not same holeindex")
                            spinlabelh2p2_1int=detlist2[0]
                            spinlabelh2p2_2int=detlist2[1]
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = spinlabelh2p2_1int
                            lst_internaltuple[holeindex2] = spinlabelh2p2_2int
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #Modification to detlist
                            moddetlist2=moddetlist2[2:]
                        
                        #Modifying external list
                        #SubCase: Double particle position. Means no spin-label in bracket for this
                        if particleindex1 == particleindex2:
                            #print("Particle indices the same")
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = 3
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            #No modification to detlist2 needed
                        else:
                            #print("Particle indices NOT the same")
                            spinlabelh2p2_1ext=detlist2[-2]
                            spinlabelh2p2_2ext=detlist2[-1]
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = spinlabelh2p2_1ext
                            lst_externaltuple[particleposition2] = spinlabelh2p2_2ext
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            #Modifying active detlist
                            moddetlist2=moddetlist2[:-2]                        
                        
                        
                    #CASE: 2 HOLE  1 PARTICLE:   
                    elif len(hole_indices) == 2 and len(particle_indices) == 1:
                        moddetlist2=detlist2
                        holeindex1=hole_indices[0]
                        holeindex2=hole_indices[1]
                        particleindex1=particle_indices[0]
                        #Particleposition in external list
                        particleposition1=particleindex1-external_first
                        
                        #Modifying internal list
                        #SubCase: Double internal hole. Means no spin-label in bracket for this
                        if holeindex1 == holeindex2:
                            #print("Same holeindex")
                            spinlabelh2p1int=0
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = spinlabelh2p1int
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #No modification to detlist2 needed
                        #Subcase: Not doubly internal hole.
                        else:
                            #print("Not same holeindex")
                            spinlabelh2p1_1int=detlist2[0]
                            spinlabelh2p1_2int=detlist2[1]
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = spinlabelh2p1_1int
                            lst_internaltuple[holeindex2] = spinlabelh2p1_2int
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #Modification to detlist
                            moddetlist2=moddetlist2[2:]

                        #Modifying external
                        lst_externaltuple=list(external_tuple)
                        #print("external_tuple:", external_tuple)
                        spinlabelh2p1_1ext=detlist2[-1]
                        lst_externaltuple[particleposition1] = spinlabelh2p1_1ext
                        modexternal_tuple=tuple(lst_externaltuple)
                        #print("Mod external tuple :", modexternal_tuple)
                        
                        #Modifying detlist
                        moddetlist2=moddetlist2[:-1]                        
                        

                    #CASE: NO HOLES, NO PARTICLES                     
                    else:
                        modinternal_tuple=internal_tuple
                        modexternal_tuple=external_tuple
                        moddetlist2=detlist2
                        
                    #combining
                    det_tuple=modinternal_tuple+tuple(moddetlist2)+modexternal_tuple
                    #print("det_tuple ({}): {}".format(len(det_tuple),det_tuple))
                    
                    assert len(det_tuple) == totorbitals, "Orbital tuple ({}) not matching total number of orbitals ({})".format(len(det_tuple),totorbitals)
                    #if len(det_tuple) == 22:
                    #    print("problem")
                    #    ashexit()
                    #if len(det_tuple) != totorbitals:
                    #    print("XXXXXXXXX")
                    
                    #This is the CI coeffient
                    coeff = float(line.split()[-1])
                    #print("coeff : ", coeff)
                    state.determinants[det_tuple] = coeff
                    #print("state.determinants :", state.determinants)
                    #if dummycount == 7416:
                    #    ashexit()


                    #CASE: CFG contains only 2 and 0s. That means a situation where CFG and Det is same thing
                    # But det info is not printed so we need to add it
                    #DISABLING after Vijay update
                    #if '1' not in cfg:
                    #    print("cfg : ", cfg)
                    #    print("Found CFG without Det info. Adding to determinants")
                    #    print("line:", line)
                    #    bla = cfg.replace('[','').replace(']','').replace('CFG','')
                    #    print("bla:", bla)
                    #    det = bla.replace(str(2),str(3))
                    #    print("det:", det)
                    #    det2 = [int(i) for i in det]
                    #    det_tuple = internal_tuple + tuple(det2) + external_tuple
                    #    #print("det_tuple: ", det_tuple)
                    #    state.determinants[det_tuple] = coeff

                #Now creating state. Taking energy, root and mult (found earlier in beginning of CI block).
                if 'STATE' in line:
                    #print("STATE in line. Creating state")
                    #print("line:", line)
                    root=int(line.split()[1].replace(':',''))
                    #print("root:", root)
                    energy = float(line.split()[3])
                    state = state_dets(root,energy,mult)
                    list_of_states.append(state)
            #if 'CAS-SCF STATES FOR BLOCK' in line:
            #    mult =int(line.split()[6])
            #Now PT2-selection and CI-problem is solved. Final states coming next.
            #Checking if final_part of MRCI output or not (e.g. if SORCI)
            if final_part is True:
                if 'Unselected CSF estimate:' in line:
                    detgrab=True
            if 'S O R C I (DDCI3-STEP)' in line:
                final_part=True

    #print("list_of_states:", list_of_states)
    #print(list_of_states[0])
    #print(list_of_states[0].determinants)
    #print(list_of_states[0].configurations)


    #Going through
    #print("list_of_states[0].__dict__", list_of_states[0].__dict__)
    #for n,state in enumerate(list_of_states):
    #    print("------------------------")
    #    print("This is state {}  with mult {} and energy {} and root {}".format(n,state.mult, state.energy, state.root))
    #    print("length of state CFGs :", len(state.configurations))
    #    print("length of state determinants :", len(state.determinants))
    #    print("state.configurations : ", state.configurations)
    #    print("state.determinants : ", state.determinants)

    #print("list_of_states:", list_of_states)

    mults = list(set([state.mult for state in list_of_states]))
    #Return a dictionary with all mults and all states
    final = {}
    for mult in mults:
        final[mult] = [state.determinants for state in list_of_states if state.mult == mult ]
    #print("final :", final)
    return final

#Find wrong determinant in file
#Temporary function while MRCI printing is wrong. we just delete a determinant contribution

def delete_wrong_det(file,reference_mult):
    lines=[];wrongcount=0
    with open(file) as f:
        for count,line in enumerate(f):
            if count== 0:
                states=int(line.split()[0])
                orbitals=int(line.split()[1])
                determinants=int(line.split()[2])
            if count > 0:
                string=line.split()[0]
                a_count=string.count("a")
                b_count=string.count("b")
                unpaired_els=a_count - b_count
                mult=unpaired_els+1
                #print("string:", string)
                #print("a_count:", a_count)
                #print("b_count:", b_count)
                #print("unpaired_els:", unpaired_els)
                #print("mult:", mult)
                if mult != reference_mult:
                    wrongcount+=1
                    print("WRONG!!!! Skipping determinant")
                    print("line skipped:", line)
                else:
                    lines.append(line)
    determinants=determinants-wrongcount
    file2="temp"
    with open(file2,'w') as g:
        g.write("{} {} {}\n".format(str(states),str(orbitals),str(determinants)))
        for line in lines:
            g.write(line)
    shutil.copyfile("temp", './' + file)


def Read_old_results():
    print("Reading file PES-Results.txt ...")
    # Parsing of files
    import json
    import configparser
    #from configparser import ConfigParser
    parser = configparser.ConfigParser()

    parser.read('PES-Results.txt')
    #Using JSON to load
    #From: https://stackoverflow.com/questions/335695/lists-in-configparser
    IPs = json.loads(parser.get("Results", "IPs"))
    dysonnorms = json.loads(parser.get("Results", "Dyson-norms"))
    mos_alpha = json.loads(parser.get("Results", "MOs_alpha"))
    mos_beta = json.loads(parser.get("Results", "MOs_beta"))

    return IPs, dysonnorms, mos_alpha, mos_beta


def plot_PES_Spectrum(IPs=None, dysonnorms=None, mos_alpha=None, mos_beta=None, plotname='PES-plot',
                          start=None, finish=None, broadening=0.1, points=10000, hftyp_I=None, MOPlot=False, matplotlib=True):
    
    print("This is deprecated. To be removed...")
    
    
    if IPs is None or dysonnorms is None:
        print("plot_PES_Spectrum requires IPs and dysonnorms variables")
        ashexit()

    assert len(IPs) == len(dysonnorms), "List of Dysonnorms not same size as list of IPs." 

    if mos_alpha is None:
        MOPlot=False
        print("mos_alpha and mos_beta not provided. Skipping MO-DOS plot.")
    else:
        MOPlot=True

    blankline()
    print(bcolors.OKGREEN,"-------------------------------------------------------------------",bcolors.ENDC)
    print(bcolors.OKGREEN,"plot_PES_Spectrum: Plotting TDDFT-Dyson-norm spectrum and MO-spectrum",bcolors.ENDC)
    print(bcolors.OKGREEN,"-------------------------------------------------------------------",bcolors.ENDC)
    blankline()
    print("IPs ({}): {}".format(len(IPs),IPs))
    print("Dysonnorms ({}): {}".format(len(dysonnorms),dysonnorms))

    if start is None:
        start = IPs[0] - 8
        finish = IPs[-1] + 8

    #########################
    # Plot spectra.
    ########################
    print(bcolors.OKGREEN, "Plotting-range chosen:", start, "-", finish, "eV", "with ", points, "points and ",
              broadening, "eV broadening.", bcolors.ENDC)

    # X-range is electron binding energy
    x = np.linspace(start, finish, points)
    stkheight = 0.5
    strength = 1.0

    ######################
    # MO-dosplot
    ######################
    if MOPlot is True:
        if hftyp_I is None:
            print("hftyp_I not set (value: RHF or UHF). Assuming hftyp_I=RHF and ignoring beta MOs.")
            blankline()
        # Creates DOS out of electron binding energies (negative of occupied MO energies)
        # alpha
        occDOS_alpha = 0
        for count, peak in enumerate(mos_alpha):
            occdospeak = Gaussian(x, peak, strength, broadening)
            #virtdospeak = Gaussian(x, peak, strength, broadening)
            occDOS_alpha += occdospeak
        # beta
        if hftyp_I == "UHF":
            occDOS_beta = 0
            for count, peak in enumerate(mos_beta):
                occdospeak = Gaussian(x, peak, strength, broadening)
                #virtdospeak = Gaussian(x, peak, strength, broadening)
                occDOS_beta += occdospeak

        # Write dat/stk files for MO-DOS
        if MOPlot is True:
            datfile = open('MO-DOSPLOT' + '.dat', 'w')
            stkfile_a = open('MO-DOSPLOT' + '_a.stk', 'w')
            if hftyp_I == "UHF":
                stkfile_b = open('MO-DOSPLOT' + '_b.stk', 'w')

            for i in range(0, len(x)):
                datfile.write(str(x[i]) + " ")
                datfile.write(str(occDOS_alpha[i]) + " \n")
                if hftyp_I == "UHF":
                    datfile.write(str(occDOS_beta[i]) + "\n")
            datfile.close()
            # Creating stk file for alpha. Only including sticks for plotted region
            stk_alpha2 = []
            stk_alpha2height = []
            for i in mos_alpha:
                if i > x[-1]:
                    # print("i is", i)
                    continue
                else:
                    stkfile_a.write(str(i) + " " + str(stkheight) + "\n")
                    stk_alpha2.append(i)
                    stk_alpha2height.append(stkheight)
            stkfile_a.close()
            stk_beta2 = []
            stk_beta2height = []
            if hftyp_I == "UHF":
                for i in mos_beta:
                    if i > x[-1]:
                        continue
                    else:
                        stkfile_b.write(str(i) + " " + str(stkheight) + "\n")
                        stk_beta2.append(i)
                        stk_beta2height.append(stkheight)
                stkfile_b.close()

    ######################
    # TDDFT states DOS
    ######################
    tddftDOS = 0
    for peak, strength in zip(IPs, dysonnorms):
        tddospeak = Gaussian(x, peak, strength, broadening)
        tddftDOS += tddospeak

    #Save dat file
    with open(plotname+"-DOS.dat", 'w') as tdatfile:
        for i,j in zip(x,tddftDOS):
            tdatfile.write("{:13.10f} {:13.10f} \n".format(i,j))
    #Save stk file
    with open(plotname+"-DOS.stk", 'w') as tstkfile:
        for b,c in zip(IPs,dysonnorms):
            tstkfile.write("{:13.10f} {:13.10f} \n".format(b,c))


    ##################################
    # Plot with Matplotlib
    ####################################
    if matplotlib is True:
        print("Creating plot with Matplotlib")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        if MOPlot is True:
            # MO-DOSPLOT for initial state. Here assuming MO energies of initial state to be good approximations for IPs
            ax.plot(x, occDOS_alpha, 'C2', label='alphaMO')
            ax.stem(stk_alpha2, stk_alpha2height, label='alphaMO', basefmt=" ", markerfmt=' ', linefmt='C2-', use_line_collection=True)
            if hftyp_I == "UHF":
                ax.plot(x, occDOS_beta, 'C2', label='betaMO')
                ax.stem(stk_beta2, stk_beta2height, label='betaMO', basefmt=" ", markerfmt=' ', linefmt='C2-', use_line_collection=True)


        ##############
        # TDDFT-STATES
        ###############
        ax.plot(x, tddftDOS, 'C3', label='TDDFT')
        ax.stem(IPs, dysonnorms, label='TDDFT', markerfmt=' ', basefmt=' ', linefmt='C3-', use_line_collection=True)
        plt.xlabel('eV')
        plt.ylabel('Intensity')
        #################################
        plt.xlim(start, finish)
        plt.legend(shadow=True, fontsize='small')
        plt.savefig(plotname + '.png', format='png', dpi=200)
        # plt.show()
    else:
        print("Skipped Matplotlib part.")
    print(BC.OKGREEN,"ALL DONE!", BC.END)


#Potential adjusted KS-DFT according to Görling
#Gives adjusted MO spectrum for initial state
def potential_adjustor_DFT(theory=None, fragment=None, Initialstate_charge=None, Initialstate_mult=None,
                          Ionizedstate_charge=None, Ionizedstate_mult=None):
    
    print("="*30)
    print("Potential-adjustor DFT")
    print("="*30)
    #Calculate initial state with N electron (e.g. neutral)
    init_state = ash.Singlepoint(fragment=fragment, theory=theory, charge=Initialstate_charge, mult=Initialstate_mult)
    E_N = init_state.energy
    #Orbitals in eV
    occorbs_alpha, occorbs_beta, hftyp = orbitalgrab(theory.filename+'.out')
    
    print("occorbs_alpha (eV): ", occorbs_alpha)
    print("occorbs_beta (eV): ", occorbs_beta)
    
    #Calculate ionized state (N-1)
    result_Nmin1 = ash.Singlepoint(fragment=fragment, theory=theory, charge=Ionizedstate_charge, mult=Ionizedstate_mult)
    E_Nmin1 = result_Nmin1.energy
    #delta-SCF IP in eV
    print("")
    print("-"*60)
    print("")
    deltaE=(E_N-E_Nmin1)*ash.constants.hartoeV
    print("deltaE (IP) : {} eV".format(deltaE))
    #deltaPA for HOMO
    HOMO_index=HOMOnumber(fragment.nuccharge,Initialstate_charge,Initialstate_mult)
    print("HOMO_index:", HOMO_index)
    eps_HOMO=occorbs_alpha[HOMO_index[0]]
    print("eps_HOMO : {} eV".format(eps_HOMO))
    deltaPA=deltaE-eps_HOMO
    print("deltaPA : {} eV".format(deltaPA))
    
    #Adjust alpha and beta orbital sets
    PA_occorbs_alpha=  [orb+deltaPA for orb in occorbs_alpha]
    PA_occorbs_beta = [orb+deltaPA for orb in occorbs_beta]
    
    print("PA_occorbs_alpha (eV): ", PA_occorbs_alpha)
    print("PA_occorbs_beta (eV): ", PA_occorbs_beta)
    
    #IPs in eV as -1 times orbital energies

    PA_occorbs_alpha_IPs=[eig*-1 for eig in PA_occorbs_alpha]
    PA_occorbs_beta_IPs=[eig*-1 for eig in PA_occorbs_beta]
    print("PA_occorbs_alpha_IPs (eV):", PA_occorbs_alpha_IPs)
    print("PA_occorbs_beta_IPs (eV):", PA_occorbs_beta_IPs)
    
    #Return negative KS eigenvalues
    return PA_occorbs_alpha_IPs, PA_occorbs_beta_IPs


# Class for state (Initial, Final etc.) that may differ in charge or spin
#Will contain energies, MOs, transition energies, IPs etc.
class MolState:
    def __init__(self,charge,mult,numionstates,label):
        self.label=label
        self.charge=charge
        self.mult=mult
        self.tddftstates=[]
        self.dysonnorms=[]
        self.energy=0.0
        self.occorbs_alpha = []
        self.occorbs_beta= []
        self.hftyp = None
        self.TDtransitionenergies=[]
        self.restricted=None
        self.GSIP=None
        self.IPs=[]
        self.cubedict=None
        #Energy
        self.ionstates=[]
        #Number of calculated states for each mult
        self.numionstates=numionstates
        self.gbwfile=None
        self.outfile=None
        self.cisfile=None
        self.densitiesfile=None #New since ORCA5