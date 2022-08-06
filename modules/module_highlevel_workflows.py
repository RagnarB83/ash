#High-level WFT workflows
import numpy as np
import os
import shutil
import math
import copy

#import ash
import ash.constants
import ash.dictionaries_lists
import ash.interfaces.interface_ORCA
from ash.functions.functions_elstructure import num_core_electrons, check_cores_vs_electrons
from ash.functions.functions_general import ashexit, BC, print_line_with_mainheader, pygrep2, pygrep
from ash.modules.module_coords import elemlisttoformula, nucchargelist,elematomnumbers

# Allowed basis set families. Accessed by function basis_for_element and extrapolation
basisfamilies=['cc','aug-cc','cc-dkh','cc-dk','aug-cc-dkh','aug-cc-dk','def2','ma-def2','def2-zora', 'def2-dkh', 'def2-dk', 
            'ma-def2-zora','ma-def2-dkh', 'ma-def2-dk', 'cc-CV', 'aug-cc-CV', 'cc-CV-dkh', 'cc-CV-dk', 'aug-cc-CV-dkh', 'aug-cc-CV-dk',
            'cc-CV_3dTM-cc_L', 'aug-cc-CV_3dTM-cc_L', 'cc-f12', 'def2-x2c' ]

#PNO threshold dictionaries
LoosePNO_thresholds={'TCutPNO': 1e-6, 'TCutPairs': 1e-3, 'TCutDO': 2e-2, 'TCutMKN': 1e-3}
NormalPNO_thresholds={'TCutPNO': 3.33e-7, 'TCutPairs': 1e-4, 'TCutDO': 1e-2, 'TCutMKN': 1e-3}
NormalPNOreduced_thresholds={'TCutPNO': 1e-6, 'TCutPairs': 1e-4, 'TCutDO': 1e-2, 'TCutMKN': 1e-3}
TightPNO_thresholds={'TCutPNO': 1e-7, 'TCutPairs': 1e-5, 'TCutDO': 5e-3, 'TCutMKN': 1e-3}


#Flexible ORCA CCSD(T)/CBS protocol class.
# Regular CC, DLPNO-CC, DLPNO-CC with PNO extrapolation etc.
#pnoextrapolation=[6,7]  pnoextrapolation=[1e-6,1e-7,1.5,'TightPNO']   pnoextrapolation=[1e-6,3.33e-7,2.38,'NormalPNO']    

class ORCA_CC_CBS_Theory:
    def __init__(self, elements=None, scfsetting='TightSCF', extrainputkeyword='', extrablocks='', guessmode='Cmatrix', memory=5000, numcores=1, 
            cardinals=None, basisfamily=None, Triplesextrapolation=False, SCFextrapolation=True, alpha=None, beta=None, 
            stabilityanalysis=False, CVSR=False, CVbasis="W1-mtsmall", F12=False, Openshellreference=None, DFTreference=None, DFT_RI=False, auxbasis="autoaux-max",
            DLPNO=False, pnosetting='NormalPNO', pnoextrapolation=[1e-6,1e-7,1.5,'TightPNO'], FullLMP2Guess=False, 
            T1=False, T1correction=False, T1corrbasis_size='Large', T1corrpnosetting='NormalPNOreduced', 
            relativity=None, orcadir=None, FCI=False, atomicSOcorrection=False):

        print_line_with_mainheader("ORCA_CC_CBS_Theory")

        #Indicates that this is a QMtheory
        self.theorytype="QM"

        #CHECKS to exit early 
        if elements == None:
            print(BC.FAIL, "\nORCA_CC_CBS_Theory requires a list of elements to be given in order to set up basis sets", BC.END)
            print("Example: ORCA_CC_CBS_Theory(elements=['C','Fe','S','H','Mo'], basisfamily='def2',cardinals=[2,3], ...")
            print("Should be a list containing all elements that a fragment might contain")
            ashexit()
        else:
            #Removing redundant symbols (in case fragment.elems list was passed for example)
            elements = list(set(elements))

        #Check if F12 is chosen correctly
        if F12 == False and basisfamily == "cc-f12":
            print(BC.FAIL,"Basisfamily cc-f12 chosen but F12 is not active.")
            print("To use F12 instead of extrapolation set: F12=True, basisfamily='cc-f12', cardinals=[X] (i.e. single cardinal)",BC.END)
            ashexit()
        if F12 is True and 'f12' not in basisfamily:
            print(BC.FAIL,"F12 option chosen but an F12-basisfamily was not chosen. Choose basisfamily='cc-f12'",BC.END)
            ashexit()
        if F12 is True and len(cardinals) != 1:
            print(BC.FAIL,"For F12 calculations, set cardinals=[X] i.e. a list of one integer.", BC.END)
            ashexit()

        #Check if only 1 cardinal was chosen: meaning no extrapolation and just a single 
        if len(cardinals) == 1:
            print(BC.WARNING, "Only a single cardinal was chosen. This means that no extrapolation will be carried out", BC.END)
            self.singlebasis=True
        else:
            self.singlebasis=False

        #Main attributes
        self.cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        self.orcadir = orcadir
        self.elements=elements
        self.cardinals = cardinals
        self.alpha=alpha
        self.beta=beta
        self.SCFextrapolation=SCFextrapolation
        self.Triplesextrapolation=Triplesextrapolation
        self.basisfamily = basisfamily
        self.relativity = relativity
        self.stabilityanalysis=stabilityanalysis
        self.numcores=numcores
        self.CVSR=CVSR
        self.CVbasis=CVbasis
        self.F12=F12
        self.DFTreference=DFTreference
        self.DFT_RI=DFT_RI
        self.DLPNO=DLPNO
        self.memory=memory
        self.pnosetting=pnosetting
        self.pnoextrapolation=pnoextrapolation
        self.T1=T1
        self.T1correction=T1correction
        self.T1corrbasis_size=T1corrbasis_size
        self.T1corrpnosetting=T1corrpnosetting
        self.scfsetting=scfsetting

        self.extrainputkeyword=extrainputkeyword
        self.extrablocks=extrablocks
        self.FullLMP2Guess=FullLMP2Guess
        self.FCI=FCI
        self.atomicSOcorrection=atomicSOcorrection
        #ECP-flag may be set to True later
        self.ECPflag=False
        print("-----------------------------")
        print("ORCA_CC_CBS PROTOCOL")
        print("-----------------------------")
        print("Settings:")
        print("Cardinals chosen:", self.cardinals)
        print("Basis set family chosen:", self.basisfamily)
        print("SCF extrapolation:", self.SCFextrapolation)
        print("Separate Triples extrapolation:", self.Triplesextrapolation)
        if self.Triplesextrapolation == True:
            if len(self.cardinals) != 3:
                print("Separate triples extrapolation chosen but 3 cardinals numbers were not given.")
                print("Example: For CBS(3/4) CCSD with CBS(2/3) triples choose cardinals = [2,3,4]")
                ashexit()
        else:

            if len(self.cardinals) == 3:
                print("3 cardinal numbers were given but Triplesextrapolation == False.")
                print("This is not valid input. Exiting.")
                ashexit()
        print("Elements involved:", self.elements)
        print("Number of cores: ", self.numcores)
        print("Maxcore setting: ", self.memory, "MB")
        print("SCF setting: ", self.scfsetting)
        print("Relativity: ", self.relativity)
        print("Stability analysis:", self.stabilityanalysis)
        print("Core-Valence Scalar Relativistic correction (CVSR): ", self.CVSR)
        print("")
        print("DLPNO:", self.DLPNO)
        #DLPNO parameters
        dlpno_line=""
        if self.DLPNO == True:
            print("PNO setting: ", self.pnosetting)
            print("T1 : ", self.T1)
            print("T1correction : ", self.T1correction)
            if self.T1correction == True:
                print("T1corrbasis_size:", self.T1corrbasis_size)
                print("T1corrpnosetting:", self.T1corrpnosetting)
            print("FullLMP2Guess ", self.FullLMP2Guess)
            if self.pnosetting == "extrapolation":
                print("PNO extrapolation parameters:", self.pnoextrapolation)
            #Setting Full LMP2 to false in general for DLPNO
            dlpno_line="UseFullLMP2Guess {}".format(str(self.FullLMP2Guess).lower())
        print("")



        ######################################################
        # BLOCK-INPUT
        ######################################################

        #Enabling AutoAux for DKH whenna
        #NOTE: Only strictly necessary when doing 1-center approximation. 
        # Hopefully not too expensive. Should be robust.
        #See https://orcaforum.kofo.mpg.de/viewtopic.php?f=11&t=1392&start=10#p38051
        if self.relativity == 'DKH' and stabilityanalysis is True:
            extrablocks=extrablocks+"%rel \n DKH1CAUTOAUX true \nend"

        #Block input for SCF/MDCI block options.
        #Disabling FullLMP2 guess in general as not available for open-shell
        #Adding memory and extrablocks.
        self.blocks="""%maxcore {}
%scf
maxiter 1200
stabperform {}
guessmode {}
end

%mdci\n{}
maxiter 150\nend
{}\n""".format(memory,str(stabilityanalysis).lower(),guessmode, dlpno_line,extrablocks)

        #Stability analysis and Relativity Check
        if self.stabilityanalysis is True:
            #Adding 1-center approximation
            if self.relativity != None:
                print("Stability analysis and relativity requires 1-center approximation")
                print("Turning on")
                self.blocks = self.blocks + "\n%rel onecenter true end"



        #AUXBASIS choice
        if auxbasis == 'autoaux-max':
            finalauxbasis="auxc \"autoaux\"\nautoauxlmax true"
        elif auxbasis == 'autoaux':
            finalauxbasis="auxc \"autoaux\""
        else:
            finalauxbasis="auxc \"{}\"".format(auxbasis)

        #Getting basis sets and ECPs for each element for a given basis-family and cardinal
        
        #Distinguish between 3 cardinal calculations (Separate triples), 2 cardinal and 1 cardinal calculations
        if len(self.cardinals) == 3:
            #We now have 3 cardinals.
            self.Calc0_basis_dict={}
            for elem in elements:
                bas=basis_for_element(elem, basisfamily, cardinals[0])
                self.Calc0_basis_dict[elem] = bas
            print("Basis set definitions for each element:")
            print("Calc0_basis_dict:", self.Calc0_basis_dict)

            self.Calc1_basis_dict={}
            for elem in elements:
                bas=basis_for_element(elem, basisfamily, cardinals[1])
                self.Calc1_basis_dict[elem] = bas
            print("Basis set definitions for each element:")
            print("Calc1_basis_dict:", self.Calc1_basis_dict)

            self.Calc2_basis_dict={}
            for elem in elements:
                bas=basis_for_element(elem, basisfamily, cardinals[2])
                self.Calc2_basis_dict[elem] = bas
            print("Basis set definitions for each element:")
            print("Calc2_basis_dict", self.Calc2_basis_dict)

            #Adding basis set info for each element into blocks
            basis0_block="%basis\n"
            for el,bas_ecp in self.Calc0_basis_dict.items():
                basis0_block=basis0_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    basis0_block=basis0_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding basis set info for each element into blocks
            basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                basis1_block=basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    basis1_block=basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])
                    
            #Adding basis set info for each element into blocks
            basis2_block="%basis\n"
            for el,bas_ecp in self.Calc2_basis_dict.items():
                basis2_block=basis2_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    basis2_block=basis2_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding auxiliary basis to all defined blocks
            self.blocks0= self.blocks +basis0_block+finalauxbasis+"\nend" #Used in CCSD and CCSD(T) calcs
            self.blocks1= self.blocks +basis1_block+finalauxbasis+"\nend" #Used in CCSD and CCSD(T) calcs
            self.blocks2= self.blocks +basis2_block+finalauxbasis+"\nend" #Used in CCSD calcs only
        elif len(self.cardinals) == 2:

            self.Calc1_basis_dict={}
            for elem in elements:
                bas=basis_for_element(elem, basisfamily, cardinals[0])
                self.Calc1_basis_dict[elem] = bas
            print("Basis set definitions for each element:")
            print("Calc1_basis_dict:", self.Calc1_basis_dict)

            self.Calc2_basis_dict={}
            for elem in elements:
                bas=basis_for_element(elem, basisfamily, cardinals[1])
                self.Calc2_basis_dict[elem] = bas
            print("Basis set definitions for each element:")
            print("Calc2_basis_dict", self.Calc2_basis_dict)

            #Adding basis set info for each element into blocks
            basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                basis1_block=basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    basis1_block=basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding basis set info for each element into blocks
            basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                basis1_block=basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    basis1_block=basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding basis set info for each element into blocks
            basis2_block="%basis\n"
            for el,bas_ecp in self.Calc2_basis_dict.items():
                basis2_block=basis2_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    basis2_block=basis2_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])
            #Adding auxiliary basis to all defined blocks
            self.blocks1= self.blocks +basis1_block+finalauxbasis+"\nend" #Used in CCSD(T) calcs
            self.blocks2= self.blocks +basis2_block+finalauxbasis+"\nend"#Used in CCSD(T) calcs

        elif len(self.cardinals) == 1:
            #Single-basis calculation
            self.Calc1_basis_dict={}
            for elem in elements:
                bas=basis_for_element(elem, basisfamily, cardinals[0])
                self.Calc1_basis_dict[elem] = bas
            print("Basis set definitions for each element:")
            print("Calc1_basis_dict:", self.Calc1_basis_dict)
            #Adding basis set info for each element into blocks
            basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                basis1_block=basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    basis1_block=basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])
            
            self.blocks1= self.blocks +basis1_block+finalauxbasis+"\nend" 


        #Auxiliary basis to self.blcoks. Used by CVSR only:
        self.blocks=self.blocks+"%basis {} end".format(finalauxbasis)


        ###################
        #SIMPLE-INPUT LINE
        ###################

        #SCALAR RELATIVITY HAMILTONIAN
        if self.relativity == None:
            if self.basisfamily in ['cc-dkh', 'aug-cc-dkh', 'cc-dk', 'aug-cc-dk', 'def2-zora', 'def2-dkh', 'def2-dk', 'cc-CV_3dTM-cc_L', 'aug-cc-CV_3dTM-cc_L'
            'ma-def2-zora','ma-def2-dkh', 'cc-CV-dk', 'cc-CV-dkh', 'aug-cc-CV-dk', 'aug-cc-CV-dkh']:
                print("Relativity option is None but a relativistic basis set family chosen:", self.basisfamily)
                print("You probably want relativity keyword argument set to DKH or ZORA (relativity=\"NoRel\" option possible also but not recommended)")
                ashexit()
            self.extrainputkeyword = self.extrainputkeyword + '  '



        elif self.relativity == "NoRel":
            self.extrainputkeyword = self.extrainputkeyword + '  '
        elif self.relativity == 'DKH':
            self.extrainputkeyword = self.extrainputkeyword + ' DKH '
        elif self.relativity == 'ZORA':
            self.extrainputkeyword = self.extrainputkeyword + ' ZORA '
        elif self.relativity == 'X2C':
            self.extrainputkeyword = self.extrainputkeyword + ' X2C '
            print("Not ready")
            ashexit()

        #Possible DFT reference (functional name) NOTE: Hardcoding RIJCOSX SARC/J defgrid3 for now
        if self.DFTreference != None:
            if self.DFT_RI is True:
                self.extrainputkeyword = self.extrainputkeyword + ' {} RIJCOSX SARC/J defgrid3 '.format(self.DFTreference)
            else:
                self.extrainputkeyword = self.extrainputkeyword + ' {} NORI defgrid3 '.format(self.DFTreference)

        #Choosing CCSD(T) keyword depending on DLPNO and/or F12 approximations present
        if self.DLPNO == True:
            #Iterative DLPNO triples or not
            if self.T1 is True:
                self.ccsdtkeyword='DLPNO-CCSD(T1)'
            else:
                #DLPNO-F12 or not
                if self.F12 is True:
                    self.ccsdtkeyword='DLPNO-CCSD(T)-F12'
                else:
                    self.ccsdtkeyword='DLPNO-CCSD(T)'
            #Add PNO keyword in simpleinputline or not (if extrapolation)
            if self.pnosetting != "extrapolation":
                self.pnokeyword=self.pnosetting
            else:
                self.pnokeyword=""
        #Regular CCSD(T)
        else:
            #No PNO keyword
            self.pnokeyword=""
            self.pnosetting=None
            #F12 or not
            if self.F12 is True:
                self.ccsdtkeyword='CCSD(T)-F12/RI'
            else:
                self.ccsdtkeyword='CCSD(T)'
        
        if Openshellreference == 'QRO':
            self.extrainputkeyword = self.extrainputkeyword + ' UNO '
        elif Openshellreference == 'UHF':
            self.extrainputkeyword = self.extrainputkeyword + ' UHF '
        
        #Global F12-aux basis keyword
        if self.F12 is True:
            cardlabel=self.cardlabels[self.cardinals[0]]
            self.auxbasiskeyword="cc-pV{}Z-F12-OptRI".format(cardlabel)
        else:
            #Chosen elsewhere for non-F12
            self.auxbasiskeyword=""

        #NOTE: For F12 calculations ORCA determines the F12GAMMA parameter based on the F12-basis keyword is present in the simple-input
        #So we have to put a basis-set keyword there
        if self.F12 is True:
            self.mainbasiskeyword=self.Calc1_basis_dict[elements[0]][0]

        else:
            #For regular calculation we do not set a mainbasis-keyword
            self.mainbasiskeyword=""



        #CCSD or CCSD(T) simple input line
        #In case of separate triples we also define a CCSD line
        self.ccsdkeyword=self.ccsdtkeyword.replace("(T1)","").replace("(T)","")
        self.ccsd_line="! {} {} {} {} {} {} printbasis".format(self.ccsdkeyword, self.mainbasiskeyword, self.pnokeyword, self.scfsetting,self.extrainputkeyword,self.auxbasiskeyword)
        self.ccsdt_line="! {} {} {} {} {} {} printbasis".format(self.ccsdtkeyword, self.mainbasiskeyword, self.pnokeyword, self.scfsetting,self.extrainputkeyword,self.auxbasiskeyword)

        ##########################################################################################
        #Defining two theory objects for each basis set unless F12 or single-cardinal provided
        ##########################################################################################
        if self.singlebasis is True:
            #For single-basis CCSD(T) or single-basis F12 calculations
            self.ccsdt_1 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsdt_line, orcablocks=self.blocks1, numcores=self.numcores)
        else:
            #Extrapolations
            if self.Triplesextrapolation == True:
                #Separate CCSD extrapolations and (T) extrapolations
                #CCSD for last 2 cardinals
                self.ccsd_1 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsd_line, orcablocks=self.blocks1, numcores=self.numcores)
                self.ccsd_2 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsd_line, orcablocks=self.blocks2, numcores=self.numcores)
                #CCSD(T) for first and second cardinals
                self.ccsdt_0 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsdt_line, orcablocks=self.blocks0, numcores=self.numcores)
                self.ccsdt_1 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsdt_line, orcablocks=self.blocks1, numcores=self.numcores)
            else:
                #Regular direct CCSD(T) extrapolations on both cardinals
                self.ccsdt_1 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsdt_line, orcablocks=self.blocks1, numcores=self.numcores)
                self.ccsdt_2 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsdt_line, orcablocks=self.blocks2, numcores=self.numcores)


    def cleanup(self):
        print("Cleanup called")

    #T1 correction 
    def T1correction_Step(self, current_coords, elems,calc_label, numcores, charge=None, mult=None, basis='Large', pnosetting='NormalPNO'):
        print("\nNow doing T1 correction. Getting T0 and T1 triples from a single calculation")

        #Using basic theory line but changing from T0 to T1
        ccsdt_T1_line=self.ccsdt_line.replace('CCSD(T)','CCSD(T1)')
        #Using Large basis by default
        if basis == 'Large':
            if self.singlebasis == True:
                blocks = self.blocks1
            else:
                blocks = self.blocks2
        else:
            blocks = self.blocks1

        #PNO setting to use for T1 correction
        if pnosetting == 'NormalPNO':
            thresholdsetting=NormalPNO_thresholds
        elif pnosetting =='NormalPNOreduced':
            thresholdsetting=NormalPNOreduced_thresholds
        elif pnosetting =='LoosePNO':
            thresholdsetting=LoosePNO_thresholds
        elif pnosetting =='TightPNO':
            thresholdsetting=TightPNO_thresholds

        mdciblock=f"""\n%mdci
    TCutPNO {thresholdsetting['TCutPNO']}
    TCutPairs {thresholdsetting["TCutPairs"]}
    TCutDO {thresholdsetting["TCutDO"]}
    TCutMKN {thresholdsetting["TCutMKN"]}
    end
        """
        blocks = blocks + '\n' + mdciblock

        #Defining theory
        ccsdt_T1 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_T1_line, orcablocks=blocks, numcores=self.numcores)

        #Run T1
        unused = ccsdt_T1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
        print("ccsdt_T1.filename:", ccsdt_T1.filename)
        print("calc_label:", calc_label)
        shutil.copyfile(ccsdt_T1.filename+'.out', './' + calc_label + 'CCSDT_T1' + '.out')
        shutil.copyfile(ccsdt_T1.filename+'.gbw', './' + calc_label + 'CCSDT_T1' + '.gbw')

        #Grab both T0 and T1 from T1 calculation
        try:
            #Applies to open-shell DLPNO-CC
            triples_T0 = float(pygrep('The Total Conventional (T0) is', ccsdt_T1.filename+'.out')[-1].split()[-1])
        except TypeError:
            try:
                #Applies to closed-shell DLPNO-CC
                triples_T0 = float(pygrep('The Conventional (T0) is', ccsdt_T1.filename+'.out')[-1].split()[-1])
            except TypeError:
                #Applies when there are no triples (e.g. H2)
                triples_T0=0.0
        print("Triples T0 correlation energy:", triples_T0)
        
        triples_T1 = float(pygrep('Triples Correction (T)                     ...', ccsdt_T1.filename+'.out')[-1].split()[-1])
        print("Triples T1 correlation energy:", triples_T1)
    
        #T1 energy correction
        E_T1corr = triples_T1 - triples_T0
        print("T0->T1 correction:", E_T1corr)
        return E_T1corr

    #Core-Valence ScalarRelativistic Step
    #NOTE: Now no longer including relativity here. Best to include relativity from the beginning in all calculations and only do the CV as correction.
    #NOTE: Too messy to manage otherwise
    def CVSR_Step(self, current_coords, elems, reloption,calc_label, numcores, charge=None, mult=None):
        print("\nCVSR_Step")

        ccsdt_mtsmall_NoFC_line="! {} {} {}   nofrozencore {} {} {}".format(self.ccsdtkeyword,self.CVbasis,self.auxbasiskeyword,self.pnokeyword,self.scfsetting,self.extrainputkeyword)
        ccsdt_mtsmall_FC_line="! {} {}  {} {} {} {}".format(self.ccsdtkeyword,self.CVbasis,self.auxbasiskeyword,self.pnokeyword,self.scfsetting,self.extrainputkeyword)

        ccsdt_mtsmall_NoFC = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=self.blocks, numcores=self.numcores)
        ccsdt_mtsmall_FC = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=self.blocks, numcores=self.numcores)

        #Run
        energy_ccsdt_mtsmall_nofc = ccsdt_mtsmall_NoFC.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_NoFC' + '.out')
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.gbw', './' + calc_label + 'CCSDT_MTsmall_NoFC' + '.gbw')
        
        energy_ccsdt_mtsmall_fc = ccsdt_mtsmall_FC.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_FC' + '.out')
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.gbw', './' + calc_label + 'CCSDT_MTsmall_FC' + '.gbw')

        #Core-correlation is total energy difference between NoFC-DKH and FC-norel
        E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
        print("E_corecorr_and_SR:", E_corecorr_and_SR)
        return E_corecorr_and_SR


    # Do 2 calculations with different DLPNO TCutPNO thresholds and extrapolate to PNO limit. Other threshold follow cutoff_setting
    def PNOExtrapolationStep(self,elems=None, current_coords=None, theory=None, calc_label=None, numcores=None, charge=None, mult=None, triples=True):
        print("Inside PNOExtrapolationStep")
        
        cutoff_setting=self.pnoextrapolation[3]

        if cutoff_setting == 'NormalPNO' : 
            thresholdsetting= NormalPNO_thresholds
        elif cutoff_setting == 'TightPNO' :
            thresholdsetting= TightPNO_thresholds
        elif cutoff_setting == 'LoosePNO' :
            thresholdsetting= LoosePNO_thresholds
        else:
            thresholdsetting= NormalPNO_thresholds
        
        print("Cutoffsetting: ", cutoff_setting)
        print("Using these general thresholds:")
        print(f'TCutPairs: {thresholdsetting["TCutPairs"]:e}')
        print(f'TCutDO: {thresholdsetting["TCutDO"]:e}')
        print(f'TCutMKN: {thresholdsetting["TCutMKN"]:e}')
        print("")
        print(f"Running 2 TCutPNO calculations: {self.pnoextrapolation[0]:e} and {self.pnoextrapolation[1]:e}")

        mdciblockX=f"""\n%mdci
    TCutPNO {self.pnoextrapolation[0]}
    TCutPairs {thresholdsetting["TCutPairs"]}
    TCutDO {thresholdsetting["TCutDO"]}
    TCutMKN {thresholdsetting["TCutMKN"]}
    end
        """
        #TCutPNO option Y
        mdciblockY=f"""\n%mdci
    TCutPNO {self.pnoextrapolation[1]}
    TCutPairs {thresholdsetting["TCutPairs"]}
    TCutDO {thresholdsetting["TCutDO"]}
    TCutMKN {thresholdsetting["TCutMKN"]}
    end
        """

        #Keeping copy of original ORCA blocks of theory
        orcablocks_original=copy.copy(theory.orcablocks)

        #Add mdciblock to blocks of theory
        PNOXblocks = theory.orcablocks + mdciblockX
        PNOYblocks = theory.orcablocks + mdciblockY
        
        #Switching guessmodes for SCF as PNO-extrapolation uses same basis and CMatrix fucks with the orbitals requiring 
        PNOXblocks=PNOXblocks.replace("guessmode Cmatrix", "guessmode Fmatrix")
        PNOYblocks=PNOYblocks.replace("guessmode Cmatrix", "guessmode Fmatrix")

        theory.orcablocks = PNOXblocks
        
        theory.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
        PNOcalcX_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(theory.filename+'.out', DLPNO=self.DLPNO,F12=self.F12)
        shutil.copyfile(theory.filename+'.out', './' + calc_label + '_PNOX' + '.out')
        shutil.copyfile(theory.filename+'.gbw', './' + calc_label + '_PNOX' + '.gbw')

        theory.orcablocks = PNOYblocks
        #ash.Singlepoint(fragment=fragment, theory=theory)
        theory.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
        PNOcalcY_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(theory.filename+'.out', DLPNO=self.DLPNO,F12=self.F12)
        shutil.copyfile(theory.filename+'.out', './' + calc_label + '_PNOY' + '.out')
        shutil.copyfile(theory.filename+'.gbw', './' + calc_label + '_PNOY' + '.gbw')
        #print("PNOcalcY:", PNOcalcY_dict)
        
        #Setting theory.orcablocks back to original
        theory.orcablocks=orcablocks_original


        #Extrapolation to PNO limit

        E_SCF = PNOcalcY_dict['HF']
        #Extrapolation of CCSD part and (T) separately
        # NOTE: Is this correct??
        print("PNOcalcX_dict:", PNOcalcX_dict)
        print("PNOcalcY_dict:", PNOcalcY_dict)
        #Extrapolating the CCSD PNO-level energies to PNO limit.
        E_corrCCSD_final = PNO_extrapolation([PNOcalcX_dict['CCSD_corr'],PNOcalcY_dict['CCSD_corr']],self.pnoextrapolation[2])

        if triples == True:
            #Triples were calculated. Extrapolating to PNO limit
            E_corrCCT_final = PNO_extrapolation([PNOcalcX_dict['CCSD(T)_corr'],PNOcalcY_dict['CCSD(T)_corr']],self.pnoextrapolation[2])
            E_corrCC_final = E_corrCCSD_final + E_corrCCT_final
        else:
            #No triples
            E_corrCCT_final = 0.0
            E_corrCC_final = E_corrCCSD_final
        print("PNO extrapolated CCSD correlation energy:", E_corrCCSD_final, "Eh")
        print("PNO extrapolated triples correlation energy:", E_corrCCT_final, "Eh")
        print("PNO extrapolated full correlation energy:", E_corrCC_final, "Eh")
        
        return E_SCF, E_corrCCSD_final, E_corrCCT_final, E_corrCC_final



    #NOTE: TODO: PC info ??
    #TODO: coords and elems vs. fragment issue
    def run(self, current_coords=None, elems=None, Grad=False, numcores=None, charge=None, mult=None):

        print(BC.OKBLUE,BC.BOLD, "------------RUNNING ORCA_CC_CBS_Theory-------------", BC.END)

        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for ORCATheory.run method", BC.END)
            ashexit()


        if Grad == True:
            print(BC.FAIL,"No gradient available for ORCA_CC_CBS_Theory yet! Exiting", BC.END)
            ashexit()

        #Checking that there is a basis set defined for each element provided here
        #NOTE: ORCA will use default SVP basis set if basis set not defined for element
        for element in elems:
            if element not in self.Calc1_basis_dict:
                print("Error. No basis-set definition available for element: {}".format(element))
                print("Make sure to pass a list of all elements of molecule/benchmark-database when creating ORCA_CC_CBS_Theory object")
                print("Example: ORCA_CC_CBS_Theory(elements=[\"{}\" ] ".format(element))
                ashexit() 

        #Number of atoms and number of electrons
        numatoms=len(elems)
        numelectrons = int(nucchargelist(elems) - charge)

        #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
        #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
        if numelectrons == 1:
            print("Number of electrons is 1")
            print("Assuming hydrogen atom and skipping calculation")
            E_total = -0.500000
            print("Using hardcoded value: ", E_total)
            if self.FCI is True:
                E_dict = {'Total_E': E_total, 'E_SCF_CBS': E_total, 'E_CC_CBS': E_total, 'E_FCI_CBS': E_total, 'E_corrCCSD_CBS': 0.0, 
                        'E_corrCCT_CBS': 0.0, 'E_corr_CBS' : 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0, 'E_FCIcorrection': 0.0, 'T1energycorr': 0.0}
            else:
                E_dict = {'Total_E': E_total, 'E_SCF_CBS': E_total, 'E_CC_CBS': E_total, 'E_FCI_CBS': E_total, 'E_corrCCSD_CBS': 0.0, 'T1energycorr': 0.0,
                        'E_corrCCT_CBS': 0.0, 'E_corr_CBS' : 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
            self.energy_components=E_dict
            return E_total

        #Defining initial label here based on element and charge/mult of system
        formula=elemlisttoformula(elems)
        calc_label = "Frag_" + str(formula) + "_" + str(charge) + "_" + str(mult) + "_"
        print("Initial Calculation label: ", calc_label)


        #CONTROLLING NUMCORES
        #If numcores not provided to run, use self.numcores
        if numcores == None:
            numcores=self.numcores

        #Reduce numcores if required
        #NOTE: self.numcores is thus ignored if check_cores_vs_electrons reduces value based on system-size
        numcores = check_cores_vs_electrons(elems,numcores,charge)


        # EXTRAPOLATION TO PNO LIMIT BY 2 PNO calculations
        if self.pnosetting=="extrapolation":
            print("\nPNO Extrapolation option chosen.")
            print("Will run 2 jobs with PNO thresholds TCutPNO : {:e} and {:e} for each basis set cardinal calculation".format(self.pnoextrapolation[0],self.pnoextrapolation[1]))

            #SINGLE F12 EXPLICIT CORRELATION JOB or if only 1 cardinal was provided
            if self.singlebasis is True:
                print("="*70)
                print("Now doing Basis-1 job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[0]))
                print("="*70)
                #Note: naming as CBS despite single-basis
                E_SCF_CBS, E_corrCCSD_CBS, E_corrCCT_CBS,E_corr_CBS = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_1, calc_label=calc_label+'cardinal1', 
                    numcores=numcores, charge=charge, mult=mult)

            #REGULAR EXTRAPOLATION WITH 2 THEORIES
            else:
                if self.Triplesextrapolation==True:
                    print("Case: PNO extrapolation with separate CCSD and triples extrapolations")
                    #Doing large-basis CCSD job
                    print("="*70)
                    print("Now doing Basis-2 CCSD job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[2]))
                    print("="*70)
                    E_SCF_2, E_corrCCSD_2, E_corrCCT_2,E_corrCC_2 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsd_2, calc_label=calc_label+'cardinal2',
                        numcores=numcores, charge=charge, mult=mult, triples=False) #No triples

                    #Medium-basis CCSDT(T) job
                    print("Next running CCSD(T) calculations to get separate triples contribution:")

                    print("="*70)
                    print("Now doing Basis-1 CCSD(T) job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[1]))
                    print("="*70)

                    E_SCF_1, E_corrCCSD_1, E_corrCCT_1,E_corrCC_1 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_1, calc_label=calc_label+'cardinal1', 
                        numcores=numcores, charge=charge, mult=mult, triples=True)

                    print("="*70)
                    print("Now doing Basis-0 CCSD(T) job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[0]))
                    print("="*70)

                    E_SCF_0, E_corrCCSD_0, E_corrCCT_0,E_corrCC_0 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_0, calc_label=calc_label+'cardinal0', 
                        numcores=numcores, charge=charge, mult=mult, triples=True)

                    #Taking SCF and CCSD energies from largest CCSD job (cardinal no. 2) and largest CCSD(T) job (cardinal no. 1)
                    # CCSD energies from CCSD jobs (last 2 cardinals)
                    # (T) energies from the 2 CCSD(T) jobs (first 2 cardinals)
                    #Lists of energies
                    scf_energies = [E_SCF_1, E_SCF_2]
                    ccsdcorr_energies = [E_corrCCSD_1, E_corrCCSD_2]
                    triplescorr_energies = [E_corrCCT_0, E_corrCCT_1]

                    print("")
                    print(f"scf_energies (cardinals:{self.cardinals[1]},{self.cardinals[2]}): {scf_energies}")
                    print(f"ccsdcorr_energies (cardinals:{self.cardinals[1]},{self.cardinals[2]}) :{ccsdcorr_energies}")
                    print(f"triplescorr_energies (cardinals:{self.cardinals[0]},{self.cardinals[1]}) :{triplescorr_energies}")

                    #BASIS SET EXTRAPOLATION
                    #SCF extrapolation. WIll be overridden inside function if self.SCFextrapolation==True
                    print("SCF extrapolation:")
                    E_SCF_CBS = Extrapolation_twopoint_SCF(scf_energies, [self.cardinals[1],self.cardinals[2]], self.basisfamily, 
                        alpha=self.alpha, SCFextrapolation=self.SCFextrapolation) #2-point SCF extrapolation

                    #Separate CCSD, (T) and full-corr CBS energies
                    print("CCSD corr. extrapolation:")
                    E_corrCCSD_CBS = Extrapolation_twopoint_corr(ccsdcorr_energies, [self.cardinals[1],self.cardinals[2]], self.basisfamily, 
                        beta=self.beta) #2-point extrapolation using smaller cardinals
                    print("(T) corr. extrapolation:")
                    E_corrCCT_CBS = Extrapolation_twopoint_corr(triplescorr_energies, [self.cardinals[0],self.cardinals[1]], self.basisfamily, 
                        beta=self.beta) #2-point extrapolation
                    E_corr_CBS = E_corrCCSD_CBS + E_corrCCT_CBS

                else:
                    print("="*70)
                    print("Now doing Basis-1 job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[0]))
                    print("="*70)
                    E_SCF_1, E_corrCCSD_1, E_corrCCT_1,E_corrCC_1 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_1, calc_label=calc_label+'cardinal1', 
                        numcores=numcores, charge=charge, mult=mult)
                    print("="*70)
                    print("Basis-1 job done. Now doing Basis-2 job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[1]))
                    print("="*70)
                    E_SCF_2, E_corrCCSD_2, E_corrCCT_2,E_corrCC_2 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_2, calc_label=calc_label+'cardinal2',
                        numcores=numcores, charge=charge, mult=mult)
                
                    #Lists of energies
                    scf_energies = [E_SCF_1, E_SCF_2]
                    ccsdcorr_energies = [E_corrCCSD_1, E_corrCCSD_2]
                    triplescorr_energies = [E_corrCCT_1, E_corrCCT_2]
                    corr_energies = [E_corrCC_1, E_corrCC_2]
        
                    #BASIS SET EXTRAPOLATION
                    #SCF extrapolation. WIll be overridden inside function if self.SCFextrapolation==True
                    print("SCF extrapolation:")
                    E_SCF_CBS = Extrapolation_twopoint_SCF(scf_energies, self.cardinals, self.basisfamily, 
                        alpha=self.alpha, SCFextrapolation=self.SCFextrapolation) #2-point SCF extrapolation

                    #Separate CCSD, (T) and full-corr CBS energies
                    print("CCSD corr. extrapolation:")
                    E_corrCCSD_CBS = Extrapolation_twopoint_corr(ccsdcorr_energies, self.cardinals, self.basisfamily, 
                        beta=self.beta) #2-point extrapolation
                    print("(T) corr. extrapolation:")
                    E_corrCCT_CBS = Extrapolation_twopoint_corr(triplescorr_energies, self.cardinals, self.basisfamily, 
                        beta=self.beta) #2-point extrapolation
                    E_corr_CBS = E_corrCCSD_CBS + E_corrCCT_CBS
                    #E_corr_CBS = Extrapolation_twopoint_corr(corr_energies, self.cardinals, self.basisfamily, 
                    #    beta=self.beta) #2-point extrapolation     


        # OR no PNO extrapolation
        else:

            #SINGLE BASIS CORRELATION JOB
            if self.singlebasis is True:
                self.ccsdt_1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
                CCSDT_1_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNO, F12=self.F12)
                shutil.copyfile(self.ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
                shutil.copyfile(self.ccsdt_1.filename+'.gbw', './' + calc_label + 'CCSDT_1' + '.gbw')
                print("CCSDT_1_dict:", CCSDT_1_dict)
                E_SCF_CBS = CCSDT_1_dict['HF']
                E_corr_CBS = CCSDT_1_dict['full_corr']
                E_corrCCSD_CBS = CCSDT_1_dict['CCSD_corr']
                E_corrCCT_CBS = CCSDT_1_dict['CCSD(T)_corr']
            #REGULAR CBS EXTRAPOLATION WITHOUT PNO EXTRAPOLATION:
            else:
                #SEPARATE CCSD AND (T) EXTRAPOLATION
                if self.Triplesextrapolation==True:
                    print("\nNow doing separate CCSD and triples calculations")
                    print(f"Running largest CCSD calculation first with cardinal: {self.cardinals[2]}")

                    #Doing the CCSD calculations
                    #self.ccsd_1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
                    #CCSD_1_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNO)
                    #shutil.copyfile(self.ccsd_1.filename+'.out', './' + calc_label + 'CCSD_1' + '.out')
                    #shutil.copyfile(self.ccsd_1.filename+'.gbw', './' + calc_label + 'CCSD_1' + '.gbw')
                    #print("CCSD_1_dict:", CCSD_1_dict)

                    #Doing the most expensive CCSD job
                    self.ccsd_2.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
                    CCSD_2_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsd_2.filename+'.out', DLPNO=self.DLPNO)
                    shutil.copyfile(self.ccsd_2.filename+'.out', './' + calc_label + 'CCSD_2' + '.out')
                    shutil.copyfile(self.ccsd_2.filename+'.gbw', './' + calc_label + 'CCSD_2' + '.gbw')
                    print("CCSD_2_dict:", CCSD_2_dict)

                    print("Next running CCSD(T) calculations to get separate triples contribution:")
                    print(f"Running CCSD(T) calculations with cardinals: {self.cardinals[0]} and {self.cardinals[1]}")
                    #Doing the (T) calculations with the smaller basis
                    self.ccsdt_0.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
                    CCSDT_0_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_0.filename+'.out', DLPNO=self.DLPNO)
                    shutil.copyfile(self.ccsdt_0.filename+'.out', './' + calc_label + 'CCSDT_0' + '.out')
                    shutil.copyfile(self.ccsdt_0.filename+'.gbw', './' + calc_label + 'CCSDT_0' + '.gbw')
                    print("CCSDT_0_dict:", CCSDT_0_dict)

                    self.ccsdt_1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
                    CCSDT_1_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNO)
                    shutil.copyfile(self.ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
                    shutil.copyfile(self.ccsdt_1.filename+'.gbw', './' + calc_label + 'CCSDT_1' + '.gbw')
                    print("CCSDT_1_dict:", CCSDT_1_dict)

                    #Taking SCF and CCSD energies from largest CCSD job (cardinal no. 2) and largest CCSD(T) job (cardinal no. 1)
                    # CCSD energies from CCSD jobs (last 2 cardinals)
                    # (T) energies from the 2 CCSD(T) jobs (first 2 cardinals)
                    scf_energies = [CCSDT_1_dict['HF'], CCSD_2_dict['HF']]
                    ccsdcorr_energies = [CCSDT_1_dict['CCSD_corr'], CCSD_2_dict['CCSD_corr']]
                    triplescorr_energies = [CCSDT_0_dict['CCSD(T)_corr'], CCSDT_1_dict['CCSD(T)_corr']]
                    #Combining corr energies in a silly way
                    #corr_energies = list(np.array(ccsdcorr_energies)+np.array(triplescorr_energies))

                    print("")
                    print("scf_energies :", scf_energies)
                    print("ccsdcorr_energies :", ccsdcorr_energies)
                    print("triplescorr_energies :", triplescorr_energies)
                    #BASIS SET EXTRAPOLATION
                    #SCF extrapolation. WIll be overridden inside function if self.SCFextrapolation==True
                    print("SCF extrapolation:")
                    E_SCF_CBS = Extrapolation_twopoint_SCF(scf_energies, [self.cardinals[1],self.cardinals[2]], self.basisfamily, 
                        alpha=self.alpha, SCFextrapolation=self.SCFextrapolation) #2-point SCF extrapolation
                    #Separate CCSD, (T) and full-corr CBS energies
                    print("CCSD corr. extrapolation:")
                    E_corrCCSD_CBS = Extrapolation_twopoint_corr(ccsdcorr_energies, [self.cardinals[1],self.cardinals[2]], self.basisfamily, 
                        beta=self.beta) #2-point extrapolation using smaller cardinals
                    print("(T) corr. extrapolation:")
                    E_corrCCT_CBS = Extrapolation_twopoint_corr(triplescorr_energies, [self.cardinals[0],self.cardinals[1]], self.basisfamily, 
                        beta=self.beta) #2-point extrapolation
                    E_corr_CBS = E_corrCCSD_CBS + E_corrCCT_CBS

                #REGULAR DIRECT CCSD(T) EXTRAPOLATION
                else:
                    self.ccsdt_1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
                    CCSDT_1_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNO)
                    shutil.copyfile(self.ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
                    shutil.copyfile(self.ccsdt_1.filename+'.gbw', './' + calc_label + 'CCSDT_1' + '.gbw')
                    print("CCSDT_1_dict:", CCSDT_1_dict)

                    self.ccsdt_2.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
                    CCSDT_2_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_2.filename+'.out', DLPNO=self.DLPNO)
                    shutil.copyfile(self.ccsdt_2.filename+'.out', './' + calc_label + 'CCSDT_2' + '.out')
                    shutil.copyfile(self.ccsdt_2.filename+'.gbw', './' + calc_label + 'CCSDT_2' + '.gbw')
                    print("CCSDT_2_dict:", CCSDT_2_dict)

                    #List of all SCF energies  all CCSD-corr energies  and all (T) corr energies from the 2 jobs
                    scf_energies = [CCSDT_1_dict['HF'], CCSDT_2_dict['HF']]
                    ccsdcorr_energies = [CCSDT_1_dict['CCSD_corr'], CCSDT_2_dict['CCSD_corr']]
                    triplescorr_energies = [CCSDT_1_dict['CCSD(T)_corr'], CCSDT_2_dict['CCSD(T)_corr']]

                    print("")
                    print("scf_energies :", scf_energies)
                    print("ccsdcorr_energies :", ccsdcorr_energies)
                    print("triplescorr_energies :", triplescorr_energies)
                    #BASIS SET EXTRAPOLATION
                    #SCF extrapolation. WIll be overridden inside function if self.SCFextrapolation==True
                    print("SCF extrapolation:")
                    E_SCF_CBS = Extrapolation_twopoint_SCF(scf_energies, self.cardinals, self.basisfamily, 
                        alpha=self.alpha, SCFextrapolation=self.SCFextrapolation) #2-point SCF extrapolation

                    #Separate CCSD, (T) and full-corr CBS energies
                    print("CCSD corr. extrapolation:")
                    E_corrCCSD_CBS = Extrapolation_twopoint_corr(ccsdcorr_energies, self.cardinals, self.basisfamily, 
                        beta=self.beta) #2-point extrapolation
                    print("(T) corr. extrapolation:")
                    E_corrCCT_CBS = Extrapolation_twopoint_corr(triplescorr_energies, self.cardinals, self.basisfamily, 
                        beta=self.beta) #2-point extrapolation        
                    E_corr_CBS = E_corrCCSD_CBS + E_corrCCT_CBS

        ############################################################
        #T1 correction (only if T1correction = True and T1=False)
        ############################################################
        if self.T1 == False:
            if self.T1correction == True:
                T1energycorr = self.T1correction_Step(current_coords, elems, calc_label,numcores, charge=charge, mult=mult, 
                    basis=self.T1corrbasis_size, pnosetting=self.T1corrpnosetting)
                #Adding T1 energy correction to E_corr_CBS and E_corrCCT_CBS
                E_corr_CBS = E_corr_CBS + T1energycorr
                E_corrCCT_CBS = E_corrCCT_CBS + T1energycorr
            else:
                T1energycorr = 0.0
        else:
            T1energycorr = 0.0
        ############################################################
        #Core-correlation + scalar relativistic as joint correction
        ############################################################
        if self.CVSR is True:
            print("")
            print("Core-Valence Scalar Relativistic Correction is on!")
            #NOTE: We should only do CV if we are doing all-electron calculations. If we have heavy element then we have probably added an ECP (specialbasisfunction)
            
            # TODO: Option if W1-mtsmall basis set is not available? Do: cc-pwCVnZ-DK and cc-pwCVnZ ??
            #TODO: Do element check here to make sure there is an appropriate CV basis set available.
            #W1-mtsmall available for H-Ar
            #For Ca/Sc - Kr use : ??
            #Exit if heavier elements ?

            if self.ECPflag is True:
                print("ECPs present. Not doing ScalarRelativistic Correction. Switching to Core-Valence Correction only.")
                reloption=" "
                calc_label=calc_label+"CV_"
                print("Doing CVSR_Step with No Scalar Relativity and CV-basis: {}".format(self.CVbasis))
                E_corecorr_and_SR = self.CVSR_Step(current_coords, elems, reloption,calc_label,numcores, charge=charge, mult=mult)
            else:
                reloption="DKH"
                calc_label=calc_label+"CVSR_stepDKH"
                print("Doing CVSR_Step with Relativistic Option: {} and CV-basis: {}".format(reloption,self.CVbasis))
                E_corecorr_and_SR = self.CVSR_Step(current_coords, elems, reloption,calc_label,numcores, charge=charge, mult=mult)
        else:
            print("")
            print("Core-Valence Scalar Relativistic Correction is off!")
            E_corecorr_and_SR=0.0

        #Printing SCF and valence correlation energies (NOTE: contains T1 correction)
        print("E_SCF_CBS:", E_SCF_CBS)
        print("E_corr_CBS:", E_corr_CBS)
        print("E_corrCCSD_CBS:", E_corrCCSD_CBS)
        print("E_corrCCT_CBS:", E_corrCCT_CBS)


        ############################################################
        #Spin-orbit correction for atoms.
        ############################################################
        if numatoms == 1 and self.atomicSOcorrection is True:
            print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
            if charge == 0:
                print("Charge of atom is zero. Looking up in neutral dict")
                try:
                    E_SO = ash.dictionaries_lists.atom_spinorbitsplittings[elems[0]] / ash.constants.hartocm
                except KeyError:
                    print("Found no SO value for atom. Will set to 0.0 and continue")
                    E_SO = 0.0
            else:
                print("Charge of atom is not zero. Dictionary not available")
                print("Found no SO value for atom. Will set to 0.0 and continue")
                E_SO = 0.0
        else :
            E_SO = 0.0
        ############################################################
        #FINAL RESULT
        ############################################################
        print("")
        print("")

        if self.FCI is True:
            print("Extrapolating SCF-energy, CCSD-energy and CCSD(T) energy to Full-CI limit by Goodson formula")
            #Here using CBS-values for SCF, CCSD-corr and (T)-corr.
            #NOTE: We need to do separate extrapolation of corrCCSD_CBS and triples_CBS
            #CCSD(T)/CBS
            E_CC_CBS = E_SCF_CBS + E_corr_CBS + E_SO + E_corecorr_and_SR
            print("CCSD(T)/CBS energy :", E_CC_CBS, "Eh")
            #FCI/CBS
            E_FCI_CBS = FCI_extrapolation([E_SCF_CBS, E_corrCCSD_CBS, E_corrCCT_CBS])
            E_FCIcorrection = E_FCI_CBS-E_CC_CBS
            E_FINAL = E_FCI_CBS
            E_dict = {'Total_E' : E_FINAL, 'E_FCI_CBS': E_FCI_CBS, 'E_CC_CBS': E_CC_CBS, 'E_SCF_CBS' : E_SCF_CBS, 'E_corrCCSD_CBS': E_corrCCSD_CBS, 'T1energycorr' : T1energycorr,
                'E_corrCCT_CBS': E_corrCCT_CBS, 'E_corr_CBS' : E_corr_CBS, 'E_SO' : E_SO, 'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_FCIcorrection': E_FCIcorrection}
            print("FCI correction:", E_FCIcorrection, "Eh")
            print("FCI/CBS energy :", E_FCI_CBS, "Eh")
            print("")

        else:
            E_CC_CBS = E_SCF_CBS + E_corr_CBS + E_SO + E_corecorr_and_SR
            print("CCSD(T)/CBS energy :", E_CC_CBS, "Eh")
            E_FINAL = E_CC_CBS
            E_dict = {'Total_E' : E_FINAL, 'E_CC_CBS': E_CC_CBS, 'E_SCF_CBS' : E_SCF_CBS, 'E_corrCCSD_CBS': E_corrCCSD_CBS, 'E_corrCCT_CBS': E_corrCCT_CBS, 'T1energycorr' : T1energycorr,
                'E_corr_CBS' : E_corr_CBS, 'E_SO' : E_SO, 'E_corecorr_and_SR' : E_corecorr_and_SR}

        print("Final energy :", E_FINAL, "Eh")
        print("")
        print("Contributions:")
        print("--------------")
        print("E_SCF_CBS : ", E_SCF_CBS, "Eh")
        print("E_corr_CBS : ", E_corr_CBS, "Eh")
        print("E_corrCCSD_CBS : ", E_corrCCSD_CBS, "Eh")
        print("E_corrCCT_CBS : ", E_corrCCT_CBS, "Eh")
        print("T1 energy correction : ", T1energycorr, "Eh")
        print("Spin-orbit coupling : ", E_SO, "Eh")
        print("E_corecorr_and_SR : ", E_corecorr_and_SR, "Eh")
        

        #Setting energy_components as an accessible attribute
        self.energy_components=E_dict


        #Cleanup GBW file. Full cleanup ??
        # TODO: Keep output files for each step
        os.remove(self.ccsdt_1.filename+'.gbw')

        #return final energy and also dictionary with energy components
        #TODO: remove E_dict
        return E_FINAL


#NOTE: Temporary alias to maintain current compatibility. Will be changed when we introduce MRCC_CC_CBS_Theory 
CC_CBS_Theory = ORCA_CC_CBS_Theory

#MRCC version
class MRCC_CC_CBS_Theory:
    def __init__(self, elements=None, scfsetting='TightSCF', extrainputkeyword='', extrablocks='', memory=5000, numcores=1, 
            cardinals=None, basisfamily=None, SCFextrapolation=True, alpha=None, beta=None, 
            CVSR=False, CVbasis="W1-mtsmall", F12=False, Openshellreference=None, DFTreference=None, DFT_RI=False,
            LNO=False, lnosetting='XXX', 
            relativity=None, mrccdir=None, FCI=False, atomicSOcorrection=False):

        print_line_with_mainheader("MRCC_CC_CBS_Theory")

        #Indicates that this is a QMtheory
        self.theorytype="QM"

        #CHECKS to exit early 
        if elements == None:
            print(BC.FAIL, "\nMRCC_CC_CBS_Theory requires a list of elements to be given in order to set up basis sets", BC.END)
            print("Example: MRCC_CC_CBS_Theory(elements=['C','Fe','S','H','Mo'], basisfamily='def2',cardinals=[2,3], ...")
            print("Should be a list containing all elements that a fragment might contain")
            ashexit()
        else:
            #Removing redundant symbols (in case fragment.elems list was passed for example)
            elements = list(set(elements))

        #Check if F12 is chosen correctly
        if F12 == False and basisfamily == "cc-f12":
            print(BC.FAIL,"Basisfamily cc-f12 chosen but F12 is not active.")
            print("To use F12 instead of extrapolation set: F12=True, basisfamily='cc-f12', cardinals=[X] (i.e. single cardinal)",BC.END)
            ashexit()
        if F12 is True and 'f12' not in basisfamily:
            print(BC.FAIL,"F12 option chosen but an F12-basisfamily was not chosen. Choose basisfamily='cc-f12'",BC.END)
            ashexit()
        if F12 is True and len(cardinals) != 1:
            print(BC.FAIL,"For F12 calculations, set cardinals=[X] i.e. a list of one integer.", BC.END)
            ashexit()

        #Check if only 1 cardinal was chosen: meaning no extrapolation and just a single 
        if len(cardinals) == 1:
            print(BC.WARNING, "Only a single cardinal was chosen. This means that no extrapolation will be carried out", BC.END)
            self.singlebasis=True
        else:
            self.singlebasis=False

        #Main attributes
        self.cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        self.orcadir = orcadir
        self.elements=elements
        self.cardinals = cardinals
        self.alpha=alpha
        self.beta=beta
        self.SCFextrapolation=SCFextrapolation
        self.basisfamily = basisfamily
        self.relativity = relativity
        self.numcores=numcores
        self.CVSR=CVSR
        self.CVbasis=CVbasis
        self.F12=F12
        self.DFTreference=DFTreference
        self.DFT_RI=DFT_RI
        self.LNO=LNO
        self.memory=memory
        self.lnosetting=lnosetting
        self.scfsetting=scfsetting

        self.extrainputkeyword=extrainputkeyword
        self.extrablocks=extrablocks
        self.FCI=FCI
        self.atomicSOcorrection=atomicSOcorrection
        #ECP-flag may be set to True later
        self.ECPflag=False
        print("-----------------------------")
        print("MRCC_CC_CBS PROTOCOL")
        print("-----------------------------")
        print("Settings:")
        print("Cardinals chosen:", self.cardinals)
        print("Basis set family chosen:", self.basisfamily)
        print("SCFextrapolation:", self.SCFextrapolation)
        print("Elements involved:", self.elements)
        print("Number of cores: ", self.numcores)
        print("Maxcore setting: ", self.memory, "MB")
        print("SCF setting: ", self.scfsetting)
        print("Relativity: ", self.relativity)
        print("Core-Valence Scalar Relativistic correction (CVSR): ", self.CVSR)
        print("")
        print("LNO:", self.LNO)
        #DLPNO parameters
        dlpno_line=""
        if self.LNO == True:
            print("LNO setting: ", self.lnosetting)
        print("")

        #Getting basis sets and ECPs for each element for a given basis-family and cardinal
        
        self.Calc1_basis_dict={}
        for elem in elements:
            bas=basis_for_element(elem, basisfamily, cardinals[0])
            self.Calc1_basis_dict[elem] = bas
        print("Basis set definitions for each element:")
        print("Calc1_basis_dict:", self.Calc1_basis_dict)
        if self.singlebasis is False:
            self.Calc2_basis_dict={}
            for elem in elements:
                bas=basis_for_element(elem, basisfamily, cardinals[1])
                self.Calc2_basis_dict[elem] = bas
            print("Basis set definitions for each element:")
            print("Calc2_basis_dict", self.Calc2_basis_dict)

        #Adding basis set info for each element into blocks
        basis1_block="%basis\n"
        for el,bas_ecp in self.Calc1_basis_dict.items():
            basis1_block=basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
            if bas_ecp[1] != None:
                #Setting ECP flag to True
                self.ECPflag=True
                basis1_block=basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])
        #Adding auxiliary basis
        basis1_block=basis1_block+finalauxbasis
        basis1_block=basis1_block+"\nend"
        self.blocks1= self.blocks +basis1_block
        if self.singlebasis is False:
            basis2_block="%basis\n"
            for el,bas_ecp in self.Calc2_basis_dict.items():
                basis2_block=basis2_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    basis2_block=basis2_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])
            #Adding auxiliary basis
            basis2_block=basis2_block+finalauxbasis
            basis2_block=basis2_block+"\nend"
            self.blocks2= self.blocks +basis2_block

        #MOdifying self.blocks to add finalauxbasis. Used by CVSR only
        self.blocks=self.blocks+"%basis {} end".format(finalauxbasis)


        ###################
        #SIMPLE-INPUT LINE
        ###################


        #Possible DFT reference (functional name) NOTE: Hardcoding RIJCOSX SARC/J defgrid3 for now
        if self.DFTreference != None:
            if self.DFT_RI is True:
                self.extrainputkeyword = self.extrainputkeyword + ' {} RIJCOSX SARC/J defgrid3 '.format(self.DFTreference)
            else:
                self.extrainputkeyword = self.extrainputkeyword + ' {} NORI defgrid3 '.format(self.DFTreference)

        #Choosing CCSD(T) keyword depending on
        if self.LNO == True:
            self.ccsdtkeyword='LNO-CCSD(T)'
        elif self.FNO == True:
            self.ccsdtkeyword='FNO-CCSD(T)'
        elif self.DF == True:
            self.ccsdtkeyword='DF-CCSD(T)'
        else:
            self.ccsdtkeyword='CCSD(T)'
        
        if Openshellreference == 'QRO':
            qroline="qro=on"
            #qro = on
            #self.extrainputkeyword = self.extrainputkeyword + ' UNO '
        elif Openshellreference == 'UHF':
            self.extrainputkeyword = self.extrainputkeyword + ' UHF '
        
        #Global F12-aux basis keyword
        if self.F12 is True:
            cardlabel=self.cardlabels[self.cardinals[0]]
            self.auxbasiskeyword="cc-pV{}Z-F12-OptRI".format(cardlabel)
        else:
            #Chosen elsewhere for non-F12
            self.auxbasiskeyword=""

        #NOTE: For F12 calculations ORCA determines the F12GAMMA parameter based on the F12-basis keyword is present in the simple-input
        #So we have to put a basis-set keyword there
        if self.F12 is True:
            self.mainbasiskeyword=self.Calc1_basis_dict[elements[0]][0]

        else:
            #For regular calculation we do not set a mainbasis-keyword
            self.mainbasiskeyword=""



        #Final simple-input line
        self.ccsdt_line="! {} {} {} {} {} {} printbasis".format(self.ccsdtkeyword, self.mainbasiskeyword, self.pnokeyword, self.scfsetting,self.extrainputkeyword,self.auxbasiskeyword)

        ##########################################################################################
        #Defining two theory objects for each basis set unless F12 or single-cardinal provided
        ##########################################################################################
        if self.singlebasis is True:
            #For single-basis CCSD(T) or single-basis F12 calculations
            self.ccsdt_1 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsdt_line, orcablocks=self.blocks1, numcores=self.numcores)
        else:
            #Extrapolations
            self.ccsdt_1 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsdt_line, orcablocks=self.blocks1, numcores=self.numcores)
            self.ccsdt_2 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsdt_line, orcablocks=self.blocks2, numcores=self.numcores)


    def cleanup(self):
        print("Cleanup called")


    #Core-Valence step
    def CVSR_Step(self, current_coords, elems, reloption,calc_label, numcores, charge=None, mult=None):
        print("\nCVSR_Step")

        ccsdt_mtsmall_NoFC_line="! {} {} {}   nofrozencore {} {} {}".format(self.ccsdtkeyword,self.CVbasis,self.auxbasiskeyword,self.pnokeyword,self.scfsetting,self.extrainputkeyword)
        ccsdt_mtsmall_FC_line="! {} {}  {} {} {} {}".format(self.ccsdtkeyword,self.CVbasis,self.auxbasiskeyword,self.pnokeyword,self.scfsetting,self.extrainputkeyword)

        ccsdt_mtsmall_NoFC = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=self.blocks, numcores=self.numcores)
        ccsdt_mtsmall_FC = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=self.blocks, numcores=self.numcores)

        #Run
        energy_ccsdt_mtsmall_nofc = ccsdt_mtsmall_NoFC.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_NoFC' + '.out')
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.gbw', './' + calc_label + 'CCSDT_MTsmall_NoFC' + '.gbw')
        
        energy_ccsdt_mtsmall_fc = ccsdt_mtsmall_FC.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_FC' + '.out')
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.gbw', './' + calc_label + 'CCSDT_MTsmall_FC' + '.gbw')

        #Core-correlation is total energy difference between NoFC-DKH and FC-norel
        E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
        print("E_corecorr_and_SR:", E_corecorr_and_SR)
        return E_corecorr_and_SR

    #NOTE: TODO: PC info ??
    #TODO: coords and elems vs. fragment issue
    def run(self, current_coords=None, elems=None, Grad=False, numcores=None, charge=None, mult=None):

        print(BC.OKBLUE,BC.BOLD, "------------RUNNING MRCC_CC_CBS_Theory-------------", BC.END)

        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for MRCC_CC_CBS_Theory run", BC.END)
            ashexit()


        if Grad == True:
            print(BC.FAIL,"No gradient available for MRCC_CC_CBS_Theory yet! Exiting", BC.END)
            ashexit()

        #Checking that there is a basis set defined for each element provided here
        for element in elems:
            if element not in self.Calc1_basis_dict:
                print("Error. No basis-set definition available for element: {}".format(element))
                print("Make sure to pass a list of all elements of molecule/benchmark-database when creating CC_CBS_Theory object")
                print("Example: CC_CBS_Theory(elements=[\"{}\" ] ".format(element))
                ashexit() 

        #Number of atoms and number of electrons
        numatoms=len(elems)
        numelectrons = int(nucchargelist(elems) - charge)

        #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
        #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
        if numelectrons == 1:
            print("Number of electrons is 1")
            print("Assuming hydrogen atom and skipping calculation")
            E_total = -0.500000
            print("Using hardcoded value: ", E_total)
            if self.FCI is True:
                E_dict = {'Total_E': E_total, 'E_SCF_CBS': E_total, 'E_CC_CBS': E_total, 'E_FCI_CBS': E_total, 'E_corrCCSD_CBS': 0.0, 
                        'E_corrCCT_CBS': 0.0, 'E_corr_CBS' : 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0, 'E_FCIcorrection': 0.0}
            else:
                E_dict = {'Total_E': E_total, 'E_SCF_CBS': E_total, 'E_CC_CBS': E_total, 'E_FCI_CBS': E_total, 'E_corrCCSD_CBS': 0.0,
                        'E_corrCCT_CBS': 0.0, 'E_corr_CBS' : 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
            self.energy_components=E_dict
            return E_total

        #Defining initial label here based on element and charge/mult of system
        formula=elemlisttoformula(elems)
        calc_label = "Frag_" + str(formula) + "_" + str(charge) + "_" + str(mult) + "_"
        print("Initial Calculation label: ", calc_label)


        #CONTROLLING NUMCORES
        #If numcores not provided to run, use self.numcores
        if numcores == None:
            numcores=self.numcores

        #Reduce numcores if required
        #NOTE: self.numcores is thus ignored if check_cores_vs_electrons reduces value based on system-size
        numcores = check_cores_vs_electrons(elems,numcores,charge)

        #SINGLE BASIS CORRELATION JOB
        if self.singlebasis is True:

            self.ccsdt_1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
            CCSDT_1_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNO, F12=self.F12)
            shutil.copyfile(self.ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
            shutil.copyfile(self.ccsdt_1.filename+'.gbw', './' + calc_label + 'CCSDT_1' + '.gbw')
            print("CCSDT_1_dict:", CCSDT_1_dict)
            E_SCF_CBS = CCSDT_1_dict['HF']
            E_corr_CBS = CCSDT_1_dict['full_corr']
            E_corrCCSD_CBS = CCSDT_1_dict['CCSD_corr']
            E_corrCCT_CBS = CCSDT_1_dict['CCSD(T)_corr']
        #REGULAR EXTRAPOLATION WITH 2 THEORIES
        else:
            self.ccsdt_1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
            CCSDT_1_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNO)
            shutil.copyfile(self.ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
            shutil.copyfile(self.ccsdt_1.filename+'.gbw', './' + calc_label + 'CCSDT_1' + '.gbw')
            print("CCSDT_1_dict:", CCSDT_1_dict)

            self.ccsdt_2.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
            CCSDT_2_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_2.filename+'.out', DLPNO=self.DLPNO)
            shutil.copyfile(self.ccsdt_2.filename+'.out', './' + calc_label + 'CCSDT_2' + '.out')
            shutil.copyfile(self.ccsdt_2.filename+'.gbw', './' + calc_label + 'CCSDT_2' + '.gbw')
            print("CCSDT_2_dict:", CCSDT_2_dict)

            #List of all SCF energies (DZ,TZ,QZ), all CCSD-corr energies (DZ,TZ,QZ) and all (T) corr energies (DZ,TZ)
            scf_energies = [CCSDT_1_dict['HF'], CCSDT_2_dict['HF']]
            ccsdcorr_energies = [CCSDT_1_dict['CCSD_corr'], CCSDT_2_dict['CCSD_corr']]
            triplescorr_energies = [CCSDT_1_dict['CCSD(T)_corr'], CCSDT_2_dict['CCSD(T)_corr']]
            #Here combining corr energies in a silly way
            corr_energies = list(np.array(ccsdcorr_energies)+np.array(triplescorr_energies))

            print("")
            print("scf_energies :", scf_energies)
            print("ccsdcorr_energies :", ccsdcorr_energies)
            print("triplescorr_energies :", triplescorr_energies)
            print("corr_energies :", corr_energies)
        
        #BASIS SET EXTRAPOLATION
            E_SCF_CBS, E_corrCCSD_CBS = Extrapolation_twopoint(scf_energies, ccsdcorr_energies, self.cardinals, self.basisfamily, 
                alpha=self.alpha, beta=self.beta, SCFextrapolation=self.SCFextrapolation) #2-point extrapolation
            #Separate CCSD and (T) CBS energies
            E_SCF_CBS, E_corrCCT_CBS = Extrapolation_twopoint(scf_energies, triplescorr_energies, self.cardinals, self.basisfamily, 
                alpha=self.alpha, beta=self.beta, SCFextrapolation=self.SCFextrapolation) #2-point extrapolation

            #BASIS SET EXTRAPOLATION of SCF and full correlation energies
            E_SCF_CBS, E_corr_CBS = Extrapolation_twopoint(scf_energies, corr_energies, self.cardinals, self.basisfamily, 
                alpha=self.alpha, beta=self.beta, SCFextrapolation=self.SCFextrapolation) #2-point extrapolation

        ############################################################
        #Core-correlation + scalar relativistic as joint correction
        ############################################################
        if self.CVSR is True:
            print("")
            print("Core-Valence Scalar Relativistic Correction is on!")
            #NOTE: We should only do CV if we are doing all-electron calculations. If we have heavy element then we have probably added an ECP (specialbasisfunction)
            
            # TODO: Option if W1-mtsmall basis set is not available? Do: cc-pwCVnZ-DK and cc-pwCVnZ ??
            #TODO: Do element check here to make sure there is an appropriate CV basis set available.
            #W1-mtsmall available for H-Ar
            #For Ca/Sc - Kr use : ??
            #Exit if heavier elements ?

            if self.ECPflag is True:
                print("ECPs present. Not doing ScalarRelativistic Correction. Switching to Core-Valence Correction only.")
                reloption=" "
                calc_label=calc_label+"CV_"
                print("Doing CVSR_Step with No Scalar Relativity and CV-basis: {}".format(self.CVbasis))
                E_corecorr_and_SR = self.CVSR_Step(current_coords, elems, reloption,calc_label,numcores, charge=charge, mult=mult)
            else:
                reloption="DKH"
                calc_label=calc_label+"CVSR_stepDKH"
                print("Doing CVSR_Step with Relativistic Option: {} and CV-basis: {}".format(reloption,self.CVbasis))
                E_corecorr_and_SR = self.CVSR_Step(current_coords, elems, reloption,calc_label,numcores, charge=charge, mult=mult)
        else:
            print("")
            print("Core-Valence Scalar Relativistic Correction is off!")
            E_corecorr_and_SR=0.0

        #Printing SCF and valence correlation energies (NOTE: contains T1 correction)
        print("E_SCF_CBS:", E_SCF_CBS)
        print("E_corr_CBS:", E_corr_CBS)
        print("E_corrCCSD_CBS:", E_corrCCSD_CBS)
        print("E_corrCCT_CBS:", E_corrCCT_CBS)


        ############################################################
        #Spin-orbit correction for atoms.
        ############################################################
        if numatoms == 1 and self.atomicSOcorrection is True:
            print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
            if charge == 0:
                print("Charge of atom is zero. Looking up in neutral dict")
                try:
                    E_SO = ash.dictionaries_lists.atom_spinorbitsplittings[elems[0]] / ash.constants.hartocm
                except KeyError:
                    print("Found no SO value for atom. Will set to 0.0 and continue")
                    E_SO = 0.0
            else:
                print("Charge of atom is not zero. Dictionary not available")
                print("Found no SO value for atom. Will set to 0.0 and continue")
                E_SO = 0.0
        else :
            E_SO = 0.0
        ############################################################
        #FINAL RESULT
        ############################################################
        print("")
        print("")

        if self.FCI is True:
            print("Extrapolating SCF-energy, CCSD-energy and CCSD(T) energy to Full-CI limit by Goodson formula")
            #Here using CBS-values for SCF, CCSD-corr and (T)-corr.
            #NOTE: We need to do separate extrapolation of corrCCSD_CBS and triples_CBS
            #CCSD(T)/CBS
            E_CC_CBS = E_SCF_CBS + E_corr_CBS + E_SO + E_corecorr_and_SR
            print("CCSD(T)/CBS energy :", E_CC_CBS, "Eh")
            #FCI/CBS
            E_FCI_CBS = FCI_extrapolation([E_SCF_CBS, E_corrCCSD_CBS, E_corrCCT_CBS])
            E_FCIcorrection = E_FCI_CBS-E_CC_CBS
            E_FINAL = E_FCI_CBS
            E_dict = {'Total_E' : E_FINAL, 'E_FCI_CBS': E_FCI_CBS, 'E_CC_CBS': E_CC_CBS, 'E_SCF_CBS' : E_SCF_CBS, 'E_corrCCSD_CBS': E_corrCCSD_CBS,
                'E_corrCCT_CBS': E_corrCCT_CBS, 'E_corr_CBS' : E_corr_CBS, 'E_SO' : E_SO, 'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_FCIcorrection': E_FCIcorrection}
            print("FCI correction:", E_FCIcorrection, "Eh")
            print("FCI/CBS energy :", E_FCI_CBS, "Eh")
            print("")

        else:
            E_CC_CBS = E_SCF_CBS + E_corr_CBS + E_SO + E_corecorr_and_SR
            print("CCSD(T)/CBS energy :", E_CC_CBS, "Eh")
            E_FINAL = E_CC_CBS
            E_dict = {'Total_E' : E_FINAL, 'E_CC_CBS': E_CC_CBS, 'E_SCF_CBS' : E_SCF_CBS, 'E_corrCCSD_CBS': E_corrCCSD_CBS, 'E_corrCCT_CBS': E_corrCCT_CBS,
                'E_corr_CBS' : E_corr_CBS, 'E_SO' : E_SO, 'E_corecorr_and_SR' : E_corecorr_and_SR}

        print("Final energy :", E_FINAL, "Eh")
        print("")
        print("Contributions:")
        print("--------------")
        print("E_SCF_CBS : ", E_SCF_CBS, "Eh")
        print("E_corr_CBS : ", E_corr_CBS, "Eh")
        print("E_corrCCSD_CBS : ", E_corrCCSD_CBS, "Eh")
        print("E_corrCCT_CBS : ", E_corrCCT_CBS, "Eh")
        print("Spin-orbit coupling : ", E_SO, "Eh")
        print("E_corecorr_and_SR : ", E_corecorr_and_SR, "Eh")
        

        #Setting energy_components as an accessible attribute
        self.energy_components=E_dict


        #Cleanup GBW file. Full cleanup ??
        # TODO: Keep output files for each step
        os.remove(self.ccsdt_1.filename+'.gbw')

        #return final energy and also dictionary with energy components
        #TODO: remove E_dict
        return E_FINAL



    
############################
# EXTRAPOLATION FUNCTIONS
#############################

#Bistoni PNO extrapolation: https://pubs.acs.org/doi/10.1021/acs.jctc.0c00344
def PNO_extrapolation(E,F):
    """ PNO extrapolation by Bistoni and coworkers
    F = 1.5, good for both 5/6 and 6/7 extrapolations.
    where 5/6 and 6/7 refers to the X/Y TcutPNO threshold (10^-X and 10^-Y).
    F = 2.38 for TCutPNO=1e-6 and 3.33e-7 (Drosou et al)
    Args:
        E ([list]): list of energies
    """
    print("PNO extrapolation using F value:", F)
    #F=1.5
    E_C_PNO= E[0] + F*(E[1]-E[0])
    return E_C_PNO



#Dictionary of extrapolation parameters. Key: Basisfamilyandcardinals Value: list: [alpha, beta]
#Added default value of beta=3.0 (theoretical value), alpha=3.9
extrapolation_parameters_dict = { 'cc_23' : [4.42, 2.460], 'aug-cc_23' : [4.30, 2.510], 'cc_34' : [5.46, 3.050], 'aug-cc_34' : [5.790, 3.050],
'def2_23' : [10.390,2.4],  'def2_34' : [7.880,2.970], 'pc_23' : [7.02, 2.01], 'pc_34': [9.78, 4.09],  'ma-def2_23' : [10.390,2.4], 
'ma-def2_34' : [7.880,2.970], 'default' : [3.9,3.0], 'default_23' : [3.9,2.4]}



def Extrapolation_twopoint_SCF(scf_energies, cardinals, basis_family, alpha=None, SCFextrapolation=False):
    """
    Extrapolation function for general 2-point extrapolations
    :param scf_energies: list of SCF energies
    :param cardinals: list of basis-cardinal numbers
    :param basis_family: string (e.g. cc, def2, aug-cc)
    :param alpha: float
    :param SCFextrapolation: Boolean
    :return: extrapolated SCF energy
    """
    #NOTE: pc-n family uses different numbering. pc-1 is DZ(cardinal 2), pc-2 is TZ(cardinal 3), pc-4 is QZ(cardinal 4).
    if basis_family=='cc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='cc_23'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='cc-dk' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='cc_23'
    elif basis_family=='cc-CV_3dTM-cc_L' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='cc_23'
    elif basis_family=='cc-CV_3dTM-cc_L' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='cc_34'
    elif basis_family=='aug-cc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='aug-cc_23'
        #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='aug-cc-dk' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='aug-cc_23'
    elif basis_family=='cc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='cc_34'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='cc-dk' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='cc_34'
    elif basis_family=='aug-cc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='aug-cc_23'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='aug-cc-dk' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='aug-cc_23'
    elif basis_family=='def2' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='def2_23'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='def2-dkh' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='def2_23'
    elif basis_family=='def2' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='def2_34'
    elif basis_family=='def2-dk' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='def2_23'
    elif basis_family=='def2' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='def2_34'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='def2-dkh' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='def2_34'
    elif basis_family=='def2-dk' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='def2_34'
    elif basis_family=='ma-def2' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='ma-def2_23'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='ma-def2-dkh' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='ma-def2_23'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    elif basis_family=='ma-def2' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='ma-def2_34'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='ma-def2-dkh' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='ma-def2_34'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    elif basis_family=='pc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='pc_23'
    elif basis_family=='pc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='pc_34'
    else:
        print("WARNING: Unknown basis set family")
        extrap_dict_key='default'

        if all(x in cardinals for x in [2, 3]):
            #For 2/3 extrapolations beta=2.4 is clearly better
            extrap_dict_key="default_23"
            print("This is a 2/3 extrapolation,")
        else:
            print("This is a 3/4 or higher extrapolation")
            #For 3/4 extrapolations or higher we are close to 3.0 theoretical value
            #'default' : [3.9,3.0],
            extrap_dict_key='default'
        print("Using settings: alpha: {} ".format(extrapolation_parameters_dict[extrap_dict_key][0]))

    #Override settings if desired
    # If alpha/beta have not been set then we define based on basisfamily and cardinals
    if alpha == None:
        alpha=extrapolation_parameters_dict[extrap_dict_key][0]
    
    #Print energies
    print("Basis family is:", basis_family)
    print("Cardinals are:", cardinals)
    print("SCF energies are:", scf_energies[0], "and", scf_energies[1])

    #Whether to skip SCF extrapolation or not
    if SCFextrapolation == False:
        print("SCF extrapolation is INACTIVE")
        print(f"Using largest-basis (cardinal: {cardinals[1]}) calculated SCF energy instead")
        SCFfinal = scf_energies[1]
        print("SCF Final value is", SCFfinal)
    else:
        print("Used alpha extrapolation parameter:",alpha)
        eX=math.exp(-1*alpha*math.sqrt(cardinals[0]))
        eY=math.exp(-1*alpha*math.sqrt(cardinals[1]))
        SCFfinal=(scf_energies[0]*eY-scf_energies[1]*eX)/(eY-eX)
        print("SCF Extrapolated value is", SCFfinal)

    return SCFfinal

def Extrapolation_twopoint_corr(corr_energies, cardinals, basis_family, beta=None):
    """
    Extrapolation function for general 2-point extrapolations
    :param corr_energies: list of correlation energies
    :param cardinals: list of basis-cardinal numbers
    :param basis_family: string (e.g. cc, def2, aug-cc)
    :param beta: Float
    :return: extrapolated correlation energy
    """

    #NOTE: pc-n family uses different numbering. pc-1 is DZ(cardinal 2), pc-2 is TZ(cardinal 3), pc-4 is QZ(cardinal 4).
    if basis_family=='cc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='cc_23'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='cc-dk' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='cc_23'
    elif basis_family=='cc-CV_3dTM-cc_L' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='cc_23'
    elif basis_family=='cc-CV_3dTM-cc_L' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='cc_34'
    elif basis_family=='aug-cc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='aug-cc_23'
        #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='aug-cc-dk' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='aug-cc_23'
    elif basis_family=='cc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='cc_34'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='cc-dk' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='cc_34'
    elif basis_family=='aug-cc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='aug-cc_23'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='aug-cc-dk' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='aug-cc_23'
    elif basis_family=='def2' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='def2_23'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='def2-dkh' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='def2_23'
    elif basis_family=='def2' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='def2_34'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='def2-dkh' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='def2_34'
    elif basis_family=='ma-def2' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='ma-def2_23'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='ma-def2-dkh' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='ma-def2_23'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    elif basis_family=='ma-def2' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='ma-def2_34'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='ma-def2-dkh' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='ma-def2_34'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    elif basis_family=='pc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='pc_23'
    elif basis_family=='pc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='pc_34'
    else:
        print("WARNING: Unknown basis set family")
        extrap_dict_key='default'

        if all(x in cardinals for x in [2, 3]):
            #For 2/3 extrapolations beta=2.4 is clearly better
            extrap_dict_key="default_23"
            print("This is a 2/3 extrapolation, choosing beta=2.4.")
            print("Warning, choosing beta=2.4.")
            #'default_23' : [3.9,2.4]
        else:
            print("This is a 3/4 or higher extrapolation, choosing beta=3.0.")
            #For 3/4 extrapolations or higher we are close to 3.0 theoretical value
            #'default' : [3.9,3.0],
            extrap_dict_key='default'
        print("Using settings: beta: {}".format(extrapolation_parameters_dict[extrap_dict_key][1]))

    
    #Override settings if desired
    print("\nExtrapolation parameters:")
    # If beta has not been set then we define based on basisfamily and cardinals
    if beta == None:
        beta=extrapolation_parameters_dict[extrap_dict_key][1]
    
    print("Used beta :", beta)

    #Print energies
    print("Basis family is:", basis_family)
    print("Cardinals are:", cardinals)
    print("Correlation energies are:", corr_energies[0], "and", corr_energies[1])
    #Correlation extrapolation
    corr_final=(math.pow(cardinals[0],beta)*corr_energies[0] - math.pow(cardinals[1],beta) * corr_energies[1])/(math.pow(cardinals[0],beta)-math.pow(cardinals[1],beta))
    print("Correlation Extrapolated value is", corr_final)

    return corr_final



def FCI_extrapolation(E):
    """Full-CI extrapolation by Goodson. Extrapolates SCF-energy, SD correlation, T correlation to Full-CI at given basis set.
       Energies provided could be e.g. all at DZ level or alternatively at estimated CBS level

    Args:
        E (list): list of E_SCF, E_corr_CCSD, E_corr_T 
    """
    d1=E[0];d2=E[1];d3=E[2]
    E_FCI=d1/(1-((d2/d1)/(1-(d3/d2))))
    return E_FCI

def Extrapolation_W1_SCF_3point(E):
    """
    Extrapolation function for old-style 3-point SCF in W1 theory
    :param E: list of HF energies (floats)
    :return:
    Note: Reading list backwards
    """
    SCF_CBS = E[-1]-(E[-1]-E[-2])**2/(E[-1]-2*E[-2]+E[-3])
    return SCF_CBS


#https://www.cup.uni-muenchen.de/oc/zipse/teaching/computational-chemistry-2/topics/overview-of-weizmann-theories/weizmann-1-theory/
def Extrapolation_W1_SCF_2point(E):
    """
    Extrapolation function for new-style 2-point SCF in W1 theory
    :param E: list of HF energies (floats)
    :return:
    Note: Reading list backwards
    """
    print("This has not been tested. Proceed with caution")
    ashexit()
    SCF_CBS = E[-1]+(E[-1]-E[-2])/((4/3)**5 - 1)
    return SCF_CBS

def Extrapolation_W1F12_SCF_2point(E):
    """
    Extrapolation function for new-style 2-point SCF in W1-F12 theory
    :param E: list of HF energies (floats)
    :return:
    Note: Reading list backwards
    """
    SCF_CBS = E[-1]+(E[-1]-E[-2])/((3/2)**5 - 1)
    return SCF_CBS



def Extrapolation_W1_CCSD(E):
    """
    Extrapolation function (A+B/l^3) for 2-point CCSD in W1 theory
    :param E: list of CCSD energies (floats)
    :return:
    Note: Reading list backwards
    """
    CCSDcorr_CBS = E[-1]+(E[-1]-E[-2])/((4/3)**3.22 - 1)
    return CCSDcorr_CBS

def Extrapolation_W1F12_CCSD(E):
    """
    Extrapolation function (A+B/l^3) for 2-point CCSD in W1 theory
    :param E: list of CCSD energies (floats)
    :return:
    Note: Reading list backwards
    """
    CCSDcorr_CBS = E[-1]+(E[-1]-E[-2])/((3/2)**3.67 - 1)
    return CCSDcorr_CBS

def Extrapolation_W1F12_triples(E):
    """
    Extrapolation function (A+B/l^3) for 2-point triples in W1-F12 theory.
    Note: Uses regular CCSD(T) energies, not F12
    :param E: list of CCSD energies (floats)
    :return:
    Note: Reading list backwards
    """
    CCSDcorr_CBS = E[-1]+(E[-1]-E[-2])/((3/2)**3.22 - 1)
    return CCSDcorr_CBS



def Extrapolation_W2_CCSD(E):
    """
    Extrapolation function (A+B/l^3) for 2-point CCSD in W2 theory
    :param E: list of CCSD energies (floats)
    :return:
    Note: Reading list backwards
    """
    CCSDcorr_CBS = E[-1]+(E[-1]-E[-2])/((5/4)**3 - 1)
    return CCSDcorr_CBS

def Extrapolation_W1_triples(E):
    """
    Extrapolation function  for 2-point (T) in W1 theory
    :param E: list of triples correlation energies (floats)
    :return:
    Note: Reading list backwards
    """
    triples_CBS = E[-1]+(E[-1]-E[-2])/((3/2)**3.22 - 1)
    return triples_CBS

def Extrapolation_W2_triples(E):
    """
    Extrapolation function  for 2-point (T) in W2 theory
    :param E: list of triples correlation energies (floats)
    :return:
    Note: Reading list backwards
    """
    triples_CBS = E[-1]+(E[-1]-E[-2])/((4/3)**3 - 1)
    return triples_CBS



################################
# BASIS-SET CHOICE FUNCTIONS
################################

#NOTE: analogous function to basis_for_element but chooses F12 basis sets where available
def F12basis_for_element(element,basisfamily,cardinal):
    print("Not ready yet")
    ashexit()

#NOTE: Should we use the cc-pVn(+d)Z basis sets for Na-AR ???
#Note: return basisname and ECPname (None if no ECP)
#Function that gives basis-set name for specific element for basisfamily and cardinal
#Example : basis_for_element('Mo', "cc", 2) gives (cc-pVDZ-PP, SK-MCDHF-RSC)
#Example : basis_for_element('C', "cc", 4) gives (cc-pVQZ, None)
def basis_for_element(element,basisfamily,cardinal):
    """Function that chooses basis set for element based on basis-set family and cardinal number

    Args:
        element ([type]): [description]
        basisfamily ([type]): [description]
        cardinal ([type]): [description]

    Returns:
        [type]: [description]
    """
    atomnumber=elematomnumbers[element.lower()]

    if basisfamily not in basisfamilies:
        print(BC.FAIL,"Unknown basisfamily. Exiting",BC.END)
        ashexit()
    #CORRELATION CONSISTENT BASIS SETS: Non-relativistic all-electron until beyond Kr when we use cc-PP basis sets
    if basisfamily == "cc":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]

        #Special cases: cc-pV6Z only for specific light elements
        if cardinal == 6 and element not in ['H','He','Be','B','C','N','O','F','Ne','Al','Si','P','S','Cl','Ar']:
            print(BC.FAIL,"cc-pV6Z basis set only available for H-He,Be-Ne,Al-Ar. Take a look at literature.",BC.END)
            ashexit()

        if atomnumber <= 18 :
            return ("cc-pV{}Z".format(cardlabel), None)
        elif atomnumber == 19 :
            print("cc basis set for K is missing in ORCA. Take a look at literature.")
            ashexit()
        elif 20 <= atomnumber <= 36   : #Ca-Kr. Note: K missing
            return ("cc-pV{}Z".format(cardlabel), None)        
        elif 38 <= atomnumber <= 54   : #Sr-Xe. Note: Rb missing
            return ("cc-pV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")
        elif 72 <= atomnumber <= 86   : #Hf-Rn
            return ("cc-pV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")
        elif atomnumber == 56 or atomnumber == 88 or atomnumber == 92: #Ba, Ra, U
            return ("cc-pV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")
    elif basisfamily == "aug-cc":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]

        #Special cases: cc-pV6Z only for specific light elements
        if cardinal == 6 and element not in ['H','He','Be','B','C','N','O','F','Ne','Al','Si','P','S','Cl','Ar']:
            print(BC.FAIL,"cc-pV6Z basis set only available for H-He,Be-Ne,Al-Ar. Take a look at literature.",BC.END)
            ashexit()

        if atomnumber <= 18 :
            return ("aug-cc-pV{}Z".format(cardlabel), None)
        elif atomnumber == 19 :
            print("cc basis set for K is missing in ORCA. Take a look at literature.")
            ashexit()
        elif 20 <= atomnumber <= 36   : #Ca-Kr. Note: K missing
            return ("aug-cc-pV{}Z".format(cardlabel), None)        
        elif 38 <= atomnumber <= 54   : #Sr-Xe. Note: Rb missing
            return ("aug-cc-pV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")
        elif 72 <= atomnumber <= 86   : #Hf-Rn
            return ("aug-cc-pV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")
        elif atomnumber == 56 or atomnumber == 88 or atomnumber == 92: #Ba, Ra, U
            return ("aug-cc-pV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")

    #ALL-ELECTRON CORRELATION-CONSISTENT BASIS SETS FOR DKH calculations
    elif basisfamily == "cc-dkh" or basisfamily == "cc-dk":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]
        # No cc-pV6Z-DK basis available
        if cardinal == 6:
            print(BC.FAIL,"cc-pV6Z-DK basis set not available",BC.END)
            ashexit()

        #Going through atomnumbers
        if atomnumber <= 18 :
            return ("cc-pV{}Z-DK".format(cardlabel), None)
        elif atomnumber == 19 or atomnumber == 20:
            print(BC.FAIL,"cc-dkh basis sets for K and Ca is missing in ORCA. Take a look at literature.",BC.END)
            ashexit()
        elif 21 <= atomnumber <= 36   : #Sc-Kr.
            return ("cc-pV{}Z-DK".format(cardlabel), None)
        elif cardinal == 3:
            if 39 <= atomnumber <= 54: #Y-Xe for cc-pVTZ-DK
                return ("cc-pV{}Z-DK".format(cardlabel), None)
            elif 72 <= atomnumber <= 86: #Hf-Rn for cc-pVTZ-DK
                return ("cc-pV{}Z-DK".format(cardlabel), None)
        elif cardinal == 4:                   
            if 49 <= atomnumber <= 54 : #In-Xe for cc-pVQZ-DK
                return ("cc-pV{}Z-DK".format(cardlabel), None)
            elif 81 <= atomnumber <= 86: #Tl-Rn for cc-pVQZ-DK
                return ("cc-pV{}Z-DK".format(cardlabel), None)
        elif cardinal == 5 and atomnumber > 36:
            print(BC.FAIL,"cc-pV5Z-DK basis sets for elements beyond Kr is missing in ORCA. Take a look at literature.",BC.END)
            ashexit()

    elif basisfamily == "aug-cc-dkh" or basisfamily == "aug-cc-dk":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]
        # No aug-cc-pV6Z-DK basis available
        if cardinal == 6:
            print(BC.FAIL,"aug-cc-pV6Z-DK basis set not available",BC.END)
            ashexit()

        #Going through atomnumbers
        if atomnumber <= 18 :
            return ("aug-cc-pV{}Z-DK".format(cardlabel), None)
        elif atomnumber == 19 or atomnumber == 20:
            print(BC.FAIL,"cc-dkh basis sets for K and Ca is missing in ORCA. Take a look at literature.",BC.END)
            ashexit()
        elif 21 <= atomnumber <= 36   : #Sc-Kr.
            return ("aug-cc-pV{}Z-DK".format(cardlabel), None)
        elif cardinal == 3:
            if 39 <= atomnumber <= 54: #Y-Xe for cc-pVTZ-DK
                return ("aug-cc-pV{}Z-DK".format(cardlabel), None)
            elif 72 <= atomnumber <= 86: #Hf-Rn for cc-pVTZ-DK
                return ("aug-cc-pV{}Z-DK".format(cardlabel), None)
        elif cardinal == 4:                   
            if 49 <= atomnumber <= 54 : #In-Xe for cc-pVQZ-DK
                return ("aug-cc-pV{}Z-DK".format(cardlabel), None)
            elif 81 <= atomnumber <= 86: #Tl-Rn for cc-pVQZ-DK
                return ("aug-cc-pV{}Z-DK".format(cardlabel), None)
        elif cardinal == 5 and atomnumber > 36:
            print(BC.FAIL,"aug-cc-pV5Z-DK basis sets for elements beyond Kr is missing in ORCA. Take a look at literature.",BC.END)
            ashexit()

    #Core-valence cc-basis sets (cc-pCVnZ or cc-pwCVnZ)
    elif basisfamily == "cc-CV":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]

        #Special cases: cc-pV6Z only for specific light elements
        if cardinal == 6:
            print(BC.FAIL,"cc-pwCV6Z basis set not available.",BC.END)
            ashexit()

        if atomnumber <= 18 :
            return ("cc-pwCV{}Z".format(cardlabel), None)
        elif atomnumber == 19 :
            print("cc-CV basis set for K is missing in ORCA. Take a look at literature.")
            ashexit()
        elif atomnumber == 20:
            #NOTE: There is also a cc-pwCVDZ-PP version
            return ("cc-pwCV{}Z".format(cardlabel), None)     
        elif 21 <= atomnumber <= 30  and cardinal > 2: #Sc-Zn. Only for cardinal > 2
            return ("cc-pwCV{}Z".format(cardlabel), None)
        elif 31 <= atomnumber <= 36 or atomnumber == 20   : #Ga-Kr, Ca. Note: K missing
            return ("cc-pwCV{}Z".format(cardlabel), None)        

        elif 38 <= atomnumber <= 54 :
            return ("cc-pwCV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")
        elif 72 <= atomnumber <= 86 :
            return ("cc-pwCV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")
        elif atomnumber == 56 or atomnumber == 88 or atomnumber == 92: #Ba, Ra, U
            return ("cc-pwCV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")

    #Core-valence cc-basis sets (cc-pCVnZ or cc-pwCVnZ)
    elif basisfamily == "aug-cc-CV":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]

        #Special cases: cc-pV6Z only for specific light elements
        if cardinal == 6:
            print(BC.FAIL,"aug-cc-pwCV6Z basis set not available.",BC.END)
            ashexit()

        if atomnumber <= 18 :
            return ("aug-cc-pwCV{}Z".format(cardlabel), None)
        elif atomnumber == 19 :
            print("aug-cc-CV basis set for K is missing in ORCA. Take a look at literature.")
            ashexit()
        elif atomnumber == 20:
            #NOTE: There is also a cc-pwCVDZ-PP version
            return ("aug-cc-pwCV{}Z".format(cardlabel), None)     
        elif 21 <= atomnumber <= 30  and cardinal > 2: #Sc-Zn. Only for cardinal > 2
            return ("aug-cc-pwCV{}Z".format(cardlabel), None)
        elif 31 <= atomnumber <= 36 or atomnumber == 20   : #Ga-Kr, Ca. Note: K missing
            return ("aug-cc-pwCV{}Z".format(cardlabel), None)        

        elif 38 <= atomnumber <= 54 :
            return ("aug-cc-pwCV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")
        elif 72 <= atomnumber <= 86 :
            return ("aug-cc-pwCV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")
        elif atomnumber == 56 or atomnumber == 88 or atomnumber == 92: #Ba, Ra, U
            return ("aug-cc-pwCV{}Z-PP".format(cardlabel), "SK-MCDHF-RSC")



    #Core-valence cc-basis sets (cc-pCVnZ or cc-pwCVnZ) for DKH
    elif basisfamily == "cc-CV-dkh" or basisfamily == "cc-CV-dk":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]

        #Special cases: cc-pV6Z only for specific light elements
        if cardinal == 6:
            print(BC.FAIL,"cc-pwCV6Z-DK basis set not available.",BC.END)
            ashexit()

        if atomnumber <= 4 :
            return ("cc-pwCV{}Z-DK".format(cardlabel), None)
        elif 5 <= atomnumber <= 10 :
            print("Warning. No proper CV basis set available for element {}. Using valence basis set instead".format(element))
            #NOTE: Using DK valence basis sets instead here. Alternative: use nonrelativistic pwCV basis ?+
            #Could use cc-pCVnZ-DK for these ones. BUT NOT available in ORCA!
            return ("cc-pV{}Z-DK".format(cardlabel), None) 
        elif 11 <= atomnumber <= 12: #Na,Mg.
            return ("cc-pwCV{}Z-DK".format(cardlabel), None)
        elif 13 <= atomnumber <= 18 :
            print("Warning. No proper CV basis set available for element {}. Using valence basis set instead".format(element))
            #NOTE: Using DK valence basis sets instead here. Alternative: use nonrelativistic pwCV basis ?+
            #Could use cc-pCVnZ-DK for these ones. BUT NOT available in ORCA!
            return ("cc-pV{}Z-DK".format(cardlabel), None)
        elif atomnumber == 19:
            print("cc-CV-dkh basis set for K is missing in ORCA. Take a look at literature.")
            ashexit() 
        elif 20 <= atomnumber <= 30: #Ca-Zn.
            return ("cc-pwCV{}Z-DK".format(cardlabel), None)
        elif 39 <= atomnumber <= 54 and cardinal > 2:
            return ("cc-pwCV{}Z-DK".format(cardlabel), None)      
        elif 72 <= atomnumber <= 80 and cardinal == 3: #Hf-Hg
            return ("cc-pwCV{}Z-DK".format(cardlabel), None)            
        elif 81 <= atomnumber <= 86 and cardinal in [3,4]: #Tl-Rn for TZ and QZ
            return ("cc-pwCV{}Z-DK".format(cardlabel), None)

    #RB: Using cc-pwCVnZ-DK for 3d transition metals and cc-pVnZ-DK or aug-cc-pVNZ-DK for light atoms
    #NOTE: Very little available for 4d and 5d row
    elif basisfamily == "cc-CV_3dTM-cc_L" or basisfamily == "aug-cc-CV_3dTM-cc_L":
        if 'aug' in basisfamily:
            prefix="aug-"
        else:
            prefix=""
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]
        # No cc-pV6Z-DK basis available
        if cardinal == 6:
            print(BC.FAIL,"cc-pV6Z-DK/cc-pwCV6Z basis sets not available",BC.END)
            ashexit()

        #Going through atomnumbers
        if atomnumber <= 18 :
            return (prefix+"cc-pV{}Z-DK".format(cardlabel), None)
        elif atomnumber == 19 or atomnumber == 20:
            print(BC.FAIL,"cc-dkh basis sets for K and Ca is missing in ORCA. Take a look at literature.",BC.END)
            ashexit()
        elif 21 <= atomnumber <= 30   : #Sc-Zn
            return ("cc-pwCV{}Z-DK".format(cardlabel), None) #Skipping aug here anyway
        elif cardinal == 3:
            if 39 <= atomnumber <= 54: #Y-Xe for cc-pVTZ-DK
                return (prefix+"cc-pV{}Z-DK".format(cardlabel), None)
            elif 72 <= atomnumber <= 80: #Hf-Rn for cc-pVTZ-DK
                return (prefix+"cc-pV{}Z-DK".format(cardlabel), None)
        elif cardinal == 4:                   
            if 49 <= atomnumber <= 54 : #In-Xe for cc-pVQZ-DK
                return (prefix+"cc-pV{}Z-DK".format(cardlabel), None)
            elif 81 <= atomnumber <= 86: #Tl-Rn for cc-pVQZ-DK
                return (prefix+"cc-pV{}Z-DK".format(cardlabel), None)
        elif cardinal == 5 and atomnumber > 36:
            print(BC.FAIL,"cc-pV5Z-DK basis sets for elements beyond Kr is missing in ORCA. Take a look at literature.",BC.END)
            ashexit()


    #Core-valence cc-basis sets (cc-pCVnZ or cc-pwCVnZ) for DKH
    elif basisfamily == "aug-cc-CV-dkh" or basisfamily == "aug-cc-CV-dk":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]

        #Special cases: cc-pV6Z only for specific light elements
        if cardinal == 6:
            print(BC.FAIL,"aug-cc-pwCV6Z-DK basis set not available.",BC.END)
            ashexit()

        if atomnumber <= 4 :
            return ("aug-cc-pwCV{}Z-DK".format(cardlabel), None)
        elif 5 <= atomnumber <= 10 :
            print("Warning. No proper CV basis set available for element {}. Using valence basis set instead".format(element))
            #NOTE: Using DK valence basis sets instead here. Alternative: use nonrelativistic pwCV basis ?+
            #Could use cc-pCVnZ-DK for these ones. BUT NOT available in ORCA!
            return ("aug-cc-pV{}Z-DK".format(cardlabel), None) 
        elif 11 <= atomnumber <= 12: #Na,Mg.
            return ("aug-cc-pwCV{}Z-DK".format(cardlabel), None)
        elif 13 <= atomnumber <= 18 :
            print("Warning. No proper CV basis set available for element {}. Using valence basis set instead".format(element))
            #NOTE: Using DK valence basis sets instead here. Alternative: use nonrelativistic pwCV basis ?+
            #Could use cc-pCVnZ-DK for these ones. BUT NOT available in ORCA!
            return ("aug-cc-pV{}Z-DK".format(cardlabel), None)
        elif atomnumber == 19:
            print("aug-cc-CV-dkh basis set for K is missing in ORCA. Take a look at literature.")
            ashexit() 
        elif 20 <= atomnumber <= 30: #Ca-Zn.
            return ("aug-cc-pwCV{}Z-DK".format(cardlabel), None)
        elif 39 <= atomnumber <= 54 and cardinal == 3:
            return ("aug-cc-pwCV{}Z-DK".format(cardlabel), None)      
        elif 49 <= atomnumber <= 71 :
            return ("aug-cc-pwCV{}Z-DK3".format(cardlabel), None)
        elif 72 <= atomnumber <= 80 and cardinal == 3:
            return ("aug-cc-pwCV{}Z-DK".format(cardlabel), None)            
        elif 81 <= atomnumber <= 86 and cardinal in [3,4]: #Tl-Rn for TZ and QZ
            return ("aug-cc-pwCV{}Z-DK".format(cardlabel), None)

    #All-electron relativistic Karlsruhe basis sets: https://pubs.acs.org/doi/10.1021/acs.jctc.7b00593
    #For use with X2C Hamiltonian. Suitable for DKH also ??
    elif basisfamily == "def2-x2c":
        if cardinal > 4:
            print(BC.FAIL,"def2-x2c basis sets only available up to QZ level", BC.END)
            ashexit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("x2c-{}all".format(cardlabel), None)
        elif 1 <= atomnumber <= 86 :
            return ("x2c-{}all".format(cardlabel), None)
    elif basisfamily == "def2":
        if cardinal > 4:
            print(BC.FAIL,"def2 basis sets only available up to QZ level", BC.END)
            ashexit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("def2-{}".format(cardlabel), None)
        elif 36 < atomnumber <= 86 :
            return ("def2-{}".format(cardlabel), "def2-ECP")

    elif basisfamily == "ma-def2":
        if cardinal > 4:
            print(BC.FAIL,"ma-def2 basis sets only available up to QZ level", BC.END)
            ashexit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("ma-def2-{}".format(cardlabel), None)
        elif 36 < atomnumber <= 86 :
            return ("ma-def2-{}".format(cardlabel), "def2-ECP")

    #NOTE: Problem SARC QZ or DZ basis set not really available so extrapolations for heavy elements not really possible
    elif basisfamily == "def2-zora":
        if cardinal > 4:
            print(BC.FAIL,"def2-ZORA basis sets only available up to QZ level", BC.END)
            ashexit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("ZORA-def2-{}".format(cardlabel), None)
        elif 36 < atomnumber <= 86 and cardinal < 4:
            #NOTE: Problem SARC QZ basis set not really available
            return ("SARC-ZORA-{}".format(cardlabel), None)

    elif basisfamily == "def2-dkh" or basisfamily == "def2-dk":
        if cardinal > 4:
            print(BC.FAIL,"def2-DKH basis sets only available up to QZ level", BC.END)
            ashexit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("DKH-def2-{}".format(cardlabel), None)
        elif 36 < atomnumber <= 86 and cardinal < 4:
            
            return ("SARC-DKH-{}".format(cardlabel), None)
    #NOTE: Problem SARC QZ or DZ basis set not really available so extrapolations for heavy elements not really possible
    elif basisfamily == "ma-def2-zora":
        if cardinal > 4:
            print(BC.FAIL,"ma-def2-ZORA basis sets only available up to QZ level", BC.END)
            ashexit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("ma-ZORA-def2-{}".format(cardlabel), None)

    elif basisfamily == "ma-def2-dkh" or basisfamily == "ma-def2-dk":
        if cardinal > 4:
            print(BC.FAIL,"ma-def2-DKH basis sets only available up to QZ level", BC.END)
            ashexit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("ma-DKH-def2-{}".format(cardlabel), None)
    elif basisfamily == "cc-f12":
        cardlabels={2:'D',3:'T',4:'Q'}
        cardlabel=cardlabels[cardinal]
        #Special cases: cc-pV6Z only for specific light elements
        if cardinal > 4:
            print(BC.FAIL,"cc-pVnZ-F12 basis set only available up to QZ level.",BC.END)
            ashexit()

        if atomnumber <= 18 :
            return ("cc-pV{}Z-F12".format(cardlabel), None)
        elif 19 <= atomnumber <= 30 :
            print("cc-F12 basis set missing for K-Zn. Take a look at literature to see if this has changed.")
            ashexit()
        elif 31 <= atomnumber <= 36   : #Ga-Kr.
            return ("cc-pV{}Z-PP-F12".format(cardlabel), "SK-MCDHF-RSC")        
        elif 49 <= atomnumber <= 54   : #In-Xe.
            return ("cc-pV{}Z-PP-F12".format(cardlabel), "SK-MCDHF-RSC")
        elif 81 <= atomnumber <= 86   : #Tl-Rn
            return ("cc-pV{}Z-PP-F12".format(cardlabel), "SK-MCDHF-RSC")

    print(BC.FAIL,"There is probably no {} {}Z basis set available for element {} in ORCA. Exiting.".format(basisfamily, cardinal, element), BC.END)
    ashexit()


#OLd ideas:
#elements_basis_sets_ORCA = {1:["cc-pVDZ", "cc-pVTZ", "cc-pVTZ", "cc-pVQZ", "cc-pV5Z", "cc-pV6Z", 
#                        "aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVTZ", "aug-cc-pVQZ", "aug-cc-pV5Z", "aug-cc-pV6Z",
#                        "def2-SV(P)", "def2-SVP", "def2-TZVP", "def2-TZVPP", "def2-QZVP", "def2-QZVPP",
#                        "dhf-SV(P)", "dhf-SVP", "dhf-TZVP", "dhf-TZVPP", "dhf-QZVP", "dhf-QZVPP",
#                        "x2c-SV(P)all", "x2c-SVPall", "x2c-TZVPall", "x2c-TZVPPall", "x2c-QZVPall", "x2c-QZVPPall",
#                        "ma-def2-SV(P)", "ma-def2-SVP", "ma-def2-TZVP", "ma-def2-TZVPP", "ma-def2-QZVP", "ma-def2-QZVPP",
#                        "def2-SVPD", "def2-TZVPD", "def2-TZVPPD", "def2-QZVPD", "def2-QZVPPD",
#                       "ZORA-def2-SV(P)", "ZORA-def2-SVP", "ZORA-def2-TZVP", "ZORA-def2-TZVP(-f)", "ZORA-def2-TZVPP", "ZORA-def2-QZVPP",
#                       "DKH-def2-SV(P)", "DKH-def2-SVP", "DKH-def2-TZVP", "DKH-def2-TZVP(-f)", "DKH-def2-TZVPP", "DKH-def2-QZVPP", 
#                       "ma-ZORA-def2-SV(P)", "ma-ZORA-def2-SVP", "ma-ZORA-def2-TZVP", "ma-ZORA-def2-TZVPP", "ma-ZORA-def2-QZVPP",
#                      "ma-DKH-def2-SV(P)", "ma-DKH-def2-SVP", "ma-DKH-def2-TZVP", "ma-DKH-def2-TZVPP", "ma-DKH-def2-QZVPP",
#                       ]}
#Get hydrogen cc-pVDZ basis
#elements_basis_sets[1].ccpVDZ
