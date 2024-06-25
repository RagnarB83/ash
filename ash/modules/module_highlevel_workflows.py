#High-level WFT workflows
import numpy as np
import os
import shutil
import math
import copy
import time

from collections import defaultdict

#import ash
import ash.constants
import ash.dictionaries_lists
import ash.interfaces.interface_ORCA
from ash.interfaces.interface_ORCA import ICE_WF_CFG_CI_size, CCSD_natocc_grab, MP2_natocc_grab, CASSCF_natocc_grab,UHF_natocc_grab,create_orca_pcfile,ORCA_orbital_setup
from ash.functions.functions_elstructure import num_core_electrons, check_cores_vs_electrons
from ash.functions.functions_general import ashexit, BC, print_line_with_mainheader, pygrep2, pygrep,print_time_rel
from ash.modules.module_coords import elemlisttoformula, nucchargelist,elematomnumbers,check_charge_mult
from ash.modules.module_plotting import ASH_plot
from ash.modules.module_singlepoint import print_fragments_table

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


#TODO: CV correction. More general basis set for 3d TM and ligand systems. cc-pwCVTZ-DK (M) ?
class ORCA_CC_CBS_Theory:
    def __init__(self, elements=None, scfsetting='TightSCF', extrainputkeyword='', extrablocks='', guessmode='Cmatrix', memory=5000, numcores=1,
            cardinals=None, basisfamily=None, Triplesextrapolation=False, SCFextrapolation=True, alpha=None, beta=None,
            stabilityanalysis=False, CVSR=False, CVbasis="W1-mtsmall", F12=False, Openshellreference=None, DFTreference=None, DFT_RI=False, auxbasis="autoaux-max",
            DLPNO=False, pnosetting='extrapolation', pnoextrapolation=[1e-6,3.33e-7,2.38,'NormalPNO'], FullLMP2Guess=False, MP2_PNO_correction=False,
            OOCC=False,T1=False, T1correction=False, T1corrbasis_size='Small', T1corrpnosetting='NormalPNOreduced',
            relativity=None, orcadir=None, FCI=False, atomicSOcorrection=False,
            HOCCcorrection=False, HOCC_program='CFour', HOCC_method='CCSDT', HOCC_basis='DZ', HOCC_ref='RHF',
            DBOCcorrection=False, DBOC_program='CFour', DBOC_method='RHF', DBOC_basis='TZ', DBOC_ref='RHF'):

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
        #OOCC
        if OOCC is True and DLPNO==True:
            print(BC.FAIL,"OO-CC calculations can only be done with DLPNO=False.", BC.END)
            ashexit()
        if OOCC is True and F12==True:
            print(BC.FAIL,"OO-CC calculations can only be done with F12=False.", BC.END)
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
        self.OOCC=OOCC
        self.DFTreference=DFTreference
        self.DFT_RI=DFT_RI
        self.DLPNO=DLPNO
        self.memory=memory
        self.pnosetting=pnosetting
        self.pnoextrapolation=pnoextrapolation
        self.MP2_PNO_correction=MP2_PNO_correction
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
        #HOCC via CFour or MRCC
        self.HOCCcorrection=HOCCcorrection
        self.HOCC_program=HOCC_program
        self.HOCC_method=HOCC_method
        self.HOCC_basis=HOCC_basis
        self.HOCC_ref=HOCC_ref
        #DBOC via Cfour
        self.DBOCcorrection=DBOCcorrection
        self.DBOC_program=DBOC_program
        self.DBOC_method=DBOC_method
        self.DBOC_basis=DBOC_basis
        self.DBOC_ref=DBOC_ref

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
        print("F12:", self.F12)
        print("OOCC:", self.OOCC)
        print("Stability analysis:", self.stabilityanalysis)
        print("Core-Valence Scalar Relativistic correction (CVSR): ", self.CVSR)
        print("Atomic spin-orbit correction: ", self.atomicSOcorrection)
        print("Higher-order coupled cluster correction: ", self.HOCCcorrection)
        print("Diagonal Born-Oppenheimer correction: ", self.HOCCcorrection)
        if self.HOCCcorrection is True:
            print("HOCC program: ", self.HOCC_program)
            print("HOCC method: ", self.HOCC_method)
            print("HOCC basis: ", self.HOCC_basis)
            print("HOCC reference: ", self.HOCC_ref)
        if self.DBOCcorrection is True:
            print("DBOC program: ", self.DBOC_program)
            print("DBOC method: ", self.DBOC_method)
            print("DBOC basis: ", self.DBOC_basis)
            print("DBOC reference: ", self.DBOC_ref)
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
            if self.MP2_PNO_correction is True:
                print("MP2_PNO correction is active")
                print("This means the PNO-limit is estimated via a DLPNO-MP2->MP2 correction")
            if self.pnosetting == "extrapolation":
                print("PNO extrapolation parameters:", self.pnoextrapolation)
            #Setting Full LMP2 to false in general for DLPNO
            dlpno_line="UseFullLMP2Guess {}".format(str(self.FullLMP2Guess).lower())
        else:
            if self.T1correction == True or self.T1 == True:
                print("T1/T1correction is only available for DLPNO=True")
                print("Ignoring T1 input")
                self.T1=False
                self.T1correction=False
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
            self.basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                self.basis1_block=self.basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis1_block=self.basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding basis set info for each element into blocks
            self.basis2_block="%basis\n"
            for el,bas_ecp in self.Calc2_basis_dict.items():
                self.basis2_block=self.basis2_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis2_block=self.basis2_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding auxiliary basis to all defined blocks
            self.basis0_block=self.basis0_block+finalauxbasis+"\nend"
            self.basis1_block=self.basis1_block+finalauxbasis+"\nend"
            self.basis2_block=self.basis2_block+finalauxbasis+"\nend"

            self.blocks0= self.blocks +basis0_block #Used in CCSD and CCSD(T) calcs
            self.blocks1= self.blocks +self.basis1_block #Used in CCSD and CCSD(T) calcs
            self.blocks2= self.blocks +self.basis2_block  #Used in CCSD calcs only
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
            self.basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                self.basis1_block=self.basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis1_block=self.basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding basis set info for each element into blocks
            self.basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                self.basis1_block=self.basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis1_block=self.basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding basis set info for each element into blocks
            self.basis2_block="%basis\n"
            for el,bas_ecp in self.Calc2_basis_dict.items():
                self.basis2_block=self.basis2_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis2_block=self.basis2_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding auxiliary basis to all defined blocks
            self.basis1_block=self.basis1_block+finalauxbasis+"\nend"
            self.basis2_block=self.basis2_block+finalauxbasis+"\nend"

            self.blocks1= self.blocks +self.basis1_block #Used in CCSD(T) calcs
            self.blocks2= self.blocks +self.basis2_block #Used in CCSD(T) calcs

        elif len(self.cardinals) == 1:
            #Single-basis calculation
            self.Calc1_basis_dict={}
            for elem in elements:
                bas=basis_for_element(elem, basisfamily, cardinals[0])
                self.Calc1_basis_dict[elem] = bas
            print("Basis set definitions for each element:")
            print("Calc1_basis_dict:", self.Calc1_basis_dict)
            #Adding basis set info for each element into blocks
            self.basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                self.basis1_block=self.basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis1_block=self.basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])
            self.basis1_block=self.basis1_block+finalauxbasis+"\nend"
            self.blocks1= self.blocks +self.basis1_block


        #Auxiliary basis to self.blocks. Used by CVSR only:
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
            if self.pnosetting == "extrapolation":
                self.pnokeyword=""
            else:
                self.pnokeyword=self.pnosetting
        #Regular CCSD(T)
        else:
            #No PNO keyword
            self.pnokeyword=""
            self.pnosetting=None
            #F12 or not
            if self.F12 is True:
                self.ccsdtkeyword='CCSD(T)-F12/RI'
            elif self.OOCC is True:
                self.ccsdtkeyword='OOCCD(T)'
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


    #MP2 CPS correction step (A. Kubas and coworkers JCTC 2023)
    def MP2correction_Step(self, current_coords, elems,calc_label, numcores, charge=None, mult=None, basis=None, pnosetting='NormalPNO', PC=None, current_MM_coords=None, MMcharges=None):
        print("\nNow doing MP2 CPS correction. Getting CPS correction via DLPNO-MP2 and RI-MP2 difference")

        #Using basic theory line but changing DLPNO-CCSD(T) to DLPNO-MP2 and MP2
        dlpno_mp2_line=self.ccsdt_line.replace('DLPNO-CCSD(T)','DLPNO-MP2  ')
        can_mp2_line=self.ccsdt_line.replace('DLPNO-CCSD(T)','RI-MP2 ')
        #Using Large basis by default
        if basis == 1:
            blocks = self.basis1_block
        elif basis == 2:
            blocks = self.basis2_block
        else:
            print("error")
            ashexit()

        #Memory
        blocks = f"""%maxcore {self.memory}
#%scf
#TolRMSP 99999999
#ConvCheckMode 1
#DampFac 1.
#end
""" + blocks

        #PNO setting to use for MP2 correction
        if pnosetting == 'NormalPNO':
            thresholdsetting=NormalPNO_thresholds
        elif pnosetting =='NormalPNOreduced':
            thresholdsetting=NormalPNOreduced_thresholds
        elif pnosetting =='LoosePNO':
            thresholdsetting=LoosePNO_thresholds
        elif pnosetting =='TightPNO':
            thresholdsetting=TightPNO_thresholds

        mp2block=f"""\n%mp2
    TCutPNO {thresholdsetting['TCutPNO']}
    TCutDO {thresholdsetting["TCutDO"]}
    TCutMKN {thresholdsetting["TCutMKN"]}
end
        """

        #Defining and running canonical MP2 theory
        can_mp2 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=can_mp2_line, orcablocks=blocks, numcores=self.numcores)
        mp2_energy = can_mp2.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
        mp2_corr_energy = float(pygrep('RI-MP2 CORRELATION ENERGY:', can_mp2.filename+'.out')[-2].split()[-1])
        shutil.copyfile(can_mp2.filename+'.out', './' + calc_label + 'MP2' + '.out')
        shutil.copyfile(can_mp2.filename+'.gbw', './' + calc_label + 'MP2' + '.gbw')

        #Defining and running DLPNO-MP2 theory
        dlpno_mp2 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=dlpno_mp2_line, orcablocks=blocks + '\n' + mp2block, numcores=self.numcores)
        dlpno_mp2_energy = dlpno_mp2.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
        dlpnomp2_corr_energy = float(pygrep('DLPNO-MP2 CORRELATION ENERGY:', can_mp2.filename+'.out')[-2].split()[-1])
        shutil.copyfile(dlpno_mp2.filename+'.out', './' + calc_label + 'DLPNO-MP2' + '.out')
        shutil.copyfile(dlpno_mp2.filename+'.gbw', './' + calc_label + 'DLPNO-MP2' + '.gbw')

        print("RI-MP2 correlation energy:", mp2_corr_energy)
        print("DLPNO-MP2 correlation energy:", dlpnomp2_corr_energy)
        E_MP2corr = mp2_corr_energy - dlpnomp2_corr_energy
        print("deltaE_MP2_CPS correction:", E_MP2corr)

        return E_MP2corr


    #T1 correction
    def T1correction_Step(self, current_coords, elems,calc_label, numcores, charge=None, mult=None, basis='Large', pnosetting='NormalPNO', PC=None, current_MM_coords=None, MMcharges=None):
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
        unused = ccsdt_T1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
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
    def CVSR_Step(self, current_coords, elems, reloption,calc_label, numcores, charge=None, mult=None, PC=None, current_MM_coords=None, MMcharges=None):
        init_time=time.time()
        print("\nCVSR_Step")

        ccsdt_mtsmall_NoFC_line="! {} {} {}   nofrozencore {} {} {}".format(self.ccsdtkeyword,self.CVbasis,self.auxbasiskeyword,self.pnokeyword,self.scfsetting,self.extrainputkeyword)
        ccsdt_mtsmall_FC_line="! {} {}  {} {} {} {}".format(self.ccsdtkeyword,self.CVbasis,self.auxbasiskeyword,self.pnokeyword,self.scfsetting,self.extrainputkeyword)

        ccsdt_mtsmall_NoFC = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=self.blocks, numcores=self.numcores)
        ccsdt_mtsmall_FC = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=self.blocks, numcores=self.numcores)

        #Run
        energy_ccsdt_mtsmall_nofc = ccsdt_mtsmall_NoFC.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_NoFC' + '.out')
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.gbw', './' + calc_label + 'CCSDT_MTsmall_NoFC' + '.gbw')

        energy_ccsdt_mtsmall_fc = ccsdt_mtsmall_FC.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_FC' + '.out')
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.gbw', './' + calc_label + 'CCSDT_MTsmall_FC' + '.gbw')

        #Core-correlation is total energy difference between NoFC-DKH and FC-norel
        E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
        print("E_corecorr_and_SR:", E_corecorr_and_SR)
        print_time_rel(init_time, modulename='CVSR step', moduleindex=2)
        return E_corecorr_and_SR


    # Do 2 calculations with different DLPNO TCutPNO thresholds and extrapolate to PNO limit. Other threshold follow cutoff_setting
    def PNOExtrapolationStep(self,elems=None, current_coords=None, theory=None, calc_label=None, numcores=None, charge=None, mult=None, triples=True, PC=None, current_MM_coords=None, MMcharges=None):
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

        theory.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
        PNOcalcX_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(theory.filename+'.out', DLPNO=self.DLPNO,F12=self.F12)
        shutil.copyfile(theory.filename+'.out', './' + calc_label + '_PNOX' + '.out')
        shutil.copyfile(theory.filename+'.gbw', './' + calc_label + '_PNOX' + '.gbw')

        theory.orcablocks = PNOYblocks
        #ash.Singlepoint(fragment=fragment, theory=theory)
        theory.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
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


#TODO: CV correction. More general basis set for 3d TM and ligand systems. cc-pwCVTZ-DK (M) ?

    def run(self, current_coords=None, qm_elems=None,
            elems=None, Grad=False, numcores=None, charge=None, mult=None, PC=None, current_MM_coords=None, MMcharges=None, mm_elems=None):

        print(BC.OKBLUE,BC.BOLD, "------------RUNNING ORCA_CC_CBS_Theory-------------", BC.END)


        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for ORCATheory.run method", BC.END)
            ashexit()


        #Used by CFour and MRCC
        if mult > 1:
            openshell = True
        else:
            openshell = False

        if PC is True:
            print("Pointcharge embedding in ORCA_CC_CBS_Theory is active")
            elems=qm_elems
            #Creating PC file. Unnecessary
            #create_orca_pcfile("orca.pc", current_MM_coords, MMcharges)

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

            #Components
            E_dict = {'Total_E' : E_total, 'E_CC_CBS': E_total, 'E_SCF_CBS' : E_total, 'E_corrCCSD_CBS': 0.0, 'E_corrCCT_CBS': 0.0, 'T1energycorr' : 0.0,
            'E_corr_CBS' : 0.0, 'E_SO' : 0.0, 'E_corecorr_and_SR' : 0.0, 'E_HOCC': 0.0, 'E_DBOC': 0.0, 'E_FCIcorrection': 0.0}

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

        original_numcores = numcores
        #Reduce numcores if required
        #NOTE: self.numcores is thus ignored if check_cores_vs_electrons reduces value based on system-size
        numcores = check_cores_vs_electrons(elems,numcores,charge)


        #Settings things to 0.0
        #E_corr_MP2PNOcorr=0.0

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
                    numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)

            #EXTRAPOLATION WITH 2 BASIS SETS
            else:
                if self.Triplesextrapolation==True:
                    print("Case: PNO extrapolation with separate CCSD and triples extrapolations")
                    #Doing large-basis CCSD job
                    print("="*70)
                    print("Now doing Basis-2 CCSD job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[2]))
                    print("="*70)
                    E_SCF_2, E_corrCCSD_2, E_corrCCT_2,E_corrCC_2 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsd_2, calc_label=calc_label+'cardinal2',
                        numcores=numcores, charge=charge, mult=mult, triples=False, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges) #No triples

                    #Medium-basis CCSDT(T) job
                    print("Next running CCSD(T) calculations to get separate triples contribution:")

                    print("="*70)
                    print("Now doing Basis-1 CCSD(T) job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[1]))
                    print("="*70)

                    E_SCF_1, E_corrCCSD_1, E_corrCCT_1,E_corrCC_1 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_1, calc_label=calc_label+'cardinal1',
                        numcores=numcores, charge=charge, mult=mult, triples=True, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)

                    print("="*70)
                    print("Now doing Basis-0 CCSD(T) job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[0]))
                    print("="*70)

                    E_SCF_0, E_corrCCSD_0, E_corrCCT_0,E_corrCC_0 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_0, calc_label=calc_label+'cardinal0',
                        numcores=numcores, charge=charge, mult=mult, triples=True, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)

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
                    print("\nSCF extrapolation:")
                    E_SCF_CBS = Extrapolation_twopoint_SCF(scf_energies, [self.cardinals[1],self.cardinals[2]], self.basisfamily,
                        alpha=self.alpha, SCFextrapolation=self.SCFextrapolation) #2-point SCF extrapolation

                    #Separate CCSD, (T) and full-corr CBS energies
                    print("\nCCSD corr. extrapolation:")
                    E_corrCCSD_CBS = Extrapolation_twopoint_corr(ccsdcorr_energies, [self.cardinals[1],self.cardinals[2]], self.basisfamily,
                        beta=self.beta) #2-point extrapolation using smaller cardinals
                    print("\n(T) corr. extrapolation:")
                    E_corrCCT_CBS = Extrapolation_twopoint_corr(triplescorr_energies, [self.cardinals[0],self.cardinals[1]], self.basisfamily,
                        beta=self.beta) #2-point extrapolation
                    E_corr_CBS = E_corrCCSD_CBS + E_corrCCT_CBS

                else:
                    print("="*70)
                    print("Now doing Basis-1 job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[0]))
                    print("="*70)
                    E_SCF_1, E_corrCCSD_1, E_corrCCT_1,E_corrCC_1 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_1, calc_label=calc_label+'cardinal1',
                        numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
                    print("="*70)
                    print("Basis-1 job done. Now doing Basis-2 job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[1]))
                    print("="*70)
                    E_SCF_2, E_corrCCSD_2, E_corrCCT_2,E_corrCC_2 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_2, calc_label=calc_label+'cardinal2',
                        numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)

                    #Lists of energies
                    scf_energies = [E_SCF_1, E_SCF_2]
                    ccsdcorr_energies = [E_corrCCSD_1, E_corrCCSD_2]
                    triplescorr_energies = [E_corrCCT_1, E_corrCCT_2]
                    corr_energies = [E_corrCC_1, E_corrCC_2]

                    #BASIS SET EXTRAPOLATION
                    #SCF extrapolation. WIll be overridden inside function if self.SCFextrapolation==True
                    print("\nSCF extrapolation:")
                    E_SCF_CBS = Extrapolation_twopoint_SCF(scf_energies, self.cardinals, self.basisfamily,
                        alpha=self.alpha, SCFextrapolation=self.SCFextrapolation) #2-point SCF extrapolation

                    #Separate CCSDcorr extrapolation
                    print("\nCCSD corr. extrapolation:")
                    E_corrCCSD_CBS = Extrapolation_twopoint_corr(ccsdcorr_energies, self.cardinals, self.basisfamily,
                        beta=self.beta) #2-point extrapolation
                    #(T) extrapolation
                    print("\n(T) corr. extrapolation:")
                    E_corrCCT_CBS = Extrapolation_twopoint_corr(triplescorr_energies, self.cardinals, self.basisfamily,
                        beta=self.beta) #2-point extrapolation
                    E_corr_CBS = E_corrCCSD_CBS + E_corrCCT_CBS
                    #E_corr_CBS = Extrapolation_twopoint_corr(corr_energies, self.cardinals, self.basisfamily,
                    #    beta=self.beta) #2-point extrapolation

        # OR no PNO extrapolation
        else:
            #SINGLE BASIS CORRELATION JOB
            if self.singlebasis is True:
                self.ccsdt_1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
                CCSDT_1_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNO, F12=self.F12)
                shutil.copyfile(self.ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
                shutil.copyfile(self.ccsdt_1.filename+'.gbw', './' + calc_label + 'CCSDT_1' + '.gbw')
                print("CCSDT_1_dict:", CCSDT_1_dict)

                if self.MP2_PNO_correction is True:
                    CCSDT_1_MP2_CPS_corr = self.MP2correction_Step(current_coords, elems,"CCSDT_1step", numcores,
                                                            charge=charge, mult=mult, basis=1, pnosetting=self.pnosetting, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
                else:
                    CCSDT_1_MP2_CPS_corr=0.0

                E_SCF_CBS = CCSDT_1_dict['HF']
                E_corr_CBS = CCSDT_1_dict['full_corr']  + CCSDT_1_MP2_CPS_corr
                E_corrCCSD_CBS = CCSDT_1_dict['CCSD_corr'] + CCSDT_1_MP2_CPS_corr
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
                    self.ccsd_2.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
                    CCSD_2_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsd_2.filename+'.out', DLPNO=self.DLPNO)
                    shutil.copyfile(self.ccsd_2.filename+'.out', './' + calc_label + 'CCSD_2' + '.out')
                    shutil.copyfile(self.ccsd_2.filename+'.gbw', './' + calc_label + 'CCSD_2' + '.gbw')
                    print("CCSD_2_dict:", CCSD_2_dict)

                    if self.MP2_PNO_correction is True:
                        CCSDT_2_MP2_CPS_corr = self.MP2correction_Step(current_coords, elems,"CCSDT_2step", numcores,
                                                                charge=charge, mult=mult, basis=2, pnosetting=self.pnosetting, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
                    else:
                        CCSDT_2_MP2_CPS_corr=0.0

                    print("Next running CCSD(T) calculations to get separate triples contribution:")
                    print(f"Running CCSD(T) calculations with cardinals: {self.cardinals[0]} and {self.cardinals[1]}")
                    #Doing the (T) calculations with the smaller basis
                    self.ccsdt_0.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
                    CCSDT_0_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_0.filename+'.out', DLPNO=self.DLPNO)
                    shutil.copyfile(self.ccsdt_0.filename+'.out', './' + calc_label + 'CCSDT_0' + '.out')
                    shutil.copyfile(self.ccsdt_0.filename+'.gbw', './' + calc_label + 'CCSDT_0' + '.gbw')
                    print("CCSDT_0_dict:", CCSDT_0_dict)

                    self.ccsdt_1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)

                    CCSDT_1_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNO)
                    shutil.copyfile(self.ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
                    shutil.copyfile(self.ccsdt_1.filename+'.gbw', './' + calc_label + 'CCSDT_1' + '.gbw')
                    print("CCSDT_1_dict:", CCSDT_1_dict)

                    if self.MP2_PNO_correction is True:
                        CCSDT_1_MP2_CPS_corr = self.MP2correction_Step(current_coords, elems,"CCSDT_1step", numcores,
                                                                charge=charge, mult=mult, basis=1, pnosetting=self.pnosetting, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
                    else:
                        CCSDT_1_MP2_CPS_corr=0.0

                    #Taking SCF and CCSD energies from largest CCSD job (cardinal no. 2) and largest CCSD(T) job (cardinal no. 1)
                    # CCSD energies from CCSD jobs (last 2 cardinals)
                    # (T) energies from the 2 CCSD(T) jobs (first 2 cardinals)
                    scf_energies = [CCSDT_1_dict['HF'], CCSD_2_dict['HF']]
                    ccsdcorr_energies = [CCSDT_1_dict['CCSD_corr']+CCSDT_1_MP2_CPS_corr, CCSD_2_dict['CCSD_corr']+CCSDT_2_MP2_CPS_corr]
                    triplescorr_energies = [CCSDT_0_dict['CCSD(T)_corr'], CCSDT_1_dict['CCSD(T)_corr']]
                    #Combining corr energies in a silly way
                    #corr_energies = list(np.array(ccsdcorr_energies)+np.array(triplescorr_energies))

                    print("")
                    print("scf_energies :", scf_energies)
                    print("ccsdcorr_energies :", ccsdcorr_energies)
                    print("triplescorr_energies :", triplescorr_energies)
                    #BASIS SET EXTRAPOLATION
                    #SCF extrapolation. WIll be overridden inside function if self.SCFextrapolation==True
                    print("\nSCF extrapolation:")
                    E_SCF_CBS = Extrapolation_twopoint_SCF(scf_energies, [self.cardinals[1],self.cardinals[2]], self.basisfamily,
                        alpha=self.alpha, SCFextrapolation=self.SCFextrapolation) #2-point SCF extrapolation
                    #Separate CCSD, (T) and full-corr CBS energies
                    print("\nCCSD corr. extrapolation:")
                    E_corrCCSD_CBS = Extrapolation_twopoint_corr(ccsdcorr_energies, [self.cardinals[1],self.cardinals[2]], self.basisfamily,
                        beta=self.beta) #2-point extrapolation using smaller cardinals
                    print("\n(T) corr. extrapolation:")
                    E_corrCCT_CBS = Extrapolation_twopoint_corr(triplescorr_energies, [self.cardinals[0],self.cardinals[1]], self.basisfamily,
                        beta=self.beta) #2-point extrapolation
                    E_corr_CBS = E_corrCCSD_CBS + E_corrCCT_CBS

                #REGULAR DIRECT CCSD(T) EXTRAPOLATION
                else:
                    self.ccsdt_1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
                    CCSDT_1_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNO)
                    shutil.copyfile(self.ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
                    shutil.copyfile(self.ccsdt_1.filename+'.gbw', './' + calc_label + 'CCSDT_1' + '.gbw')
                    print("CCSDT_1_dict:", CCSDT_1_dict)

                    if self.MP2_PNO_correction is True:
                        CCSDT_1_MP2corr = self.MP2correction_Step(current_coords, elems,"CCSDT_1step", numcores,
                                                                charge=charge, mult=mult, basis=1, pnosetting=self.pnosetting, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
                    else:
                        CCSDT_1_MP2corr=0.0

                    self.ccsdt_2.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
                    CCSDT_2_dict = ash.interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_2.filename+'.out', DLPNO=self.DLPNO)
                    shutil.copyfile(self.ccsdt_2.filename+'.out', './' + calc_label + 'CCSDT_2' + '.out')
                    shutil.copyfile(self.ccsdt_2.filename+'.gbw', './' + calc_label + 'CCSDT_2' + '.gbw')
                    print("CCSDT_2_dict:", CCSDT_2_dict)

                    if self.MP2_PNO_correction is True:
                        CCSDT_2_MP2corr = self.MP2correction_Step(current_coords, elems,"CCSDT_2step", numcores,
                                                                charge=charge, mult=mult, basis=2, pnosetting=self.pnosetting, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
                    else:
                        CCSDT_2_MP2corr=0.0

                    #List of all SCF energies  all CCSD-corr energies  and all (T) corr energies from the 2 jobs
                    scf_energies = [CCSDT_1_dict['HF'], CCSDT_2_dict['HF']]
                    ccsdcorr_energies = [CCSDT_1_dict['CCSD_corr']+CCSDT_1_MP2corr, CCSDT_2_dict['CCSD_corr']+CCSDT_2_MP2corr]
                    triplescorr_energies = [CCSDT_1_dict['CCSD(T)_corr'], CCSDT_2_dict['CCSD(T)_corr']]

                    print("")
                    print("scf_energies :", scf_energies)
                    print("ccsdcorr_energies :", ccsdcorr_energies)
                    print("triplescorr_energies :", triplescorr_energies)
                    #BASIS SET EXTRAPOLATION
                    #SCF extrapolation. WIll be overridden inside function if self.SCFextrapolation==True
                    print("\nSCF extrapolation:")
                    E_SCF_CBS = Extrapolation_twopoint_SCF(scf_energies, self.cardinals, self.basisfamily,
                        alpha=self.alpha, SCFextrapolation=self.SCFextrapolation) #2-point SCF extrapolation

                    #Separate CCSD, (T) and full-corr CBS energies
                    print("\nCCSD corr. extrapolation:")
                    E_corrCCSD_CBS = Extrapolation_twopoint_corr(ccsdcorr_energies, self.cardinals, self.basisfamily,
                        beta=self.beta) #2-point extrapolation
                    print("\n(T) corr. extrapolation:")
                    E_corrCCT_CBS = Extrapolation_twopoint_corr(triplescorr_energies, self.cardinals, self.basisfamily,
                        beta=self.beta) #2-point extrapolation
                    E_corr_CBS = E_corrCCSD_CBS + E_corrCCT_CBS

        ############################################################
        #T1 correction (only if T1correction = True and T1=False)
        ############################################################
        if self.T1 == False:
            if self.T1correction == True:
                T1energycorr = self.T1correction_Step(current_coords, elems, calc_label,numcores, charge=charge, mult=mult,
                    basis=self.T1corrbasis_size, pnosetting=self.T1corrpnosetting, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
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
                E_corecorr_and_SR = self.CVSR_Step(current_coords, elems, reloption,calc_label,numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
            else:
                reloption="DKH"
                calc_label=calc_label+"CVSR_stepDKH"
                print("Doing CVSR_Step with Relativistic Option: {} and CV-basis: {}".format(reloption,self.CVbasis))
                E_corecorr_and_SR = self.CVSR_Step(current_coords, elems, reloption,calc_label,numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
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
        #Higher-order coupled cluster correction (HOCC)
        #via CFour or MRCC
        ############################################################
        if self.HOCCcorrection is True:
            print("\nDoing HOCC correction")
            if self.HOCC_program == 'CFour':
                from ash.interfaces.interface_CFour import run_CFour_HLC_correction
                print("Doing HOCC correction via the CFour program")
                E_HOCC = run_CFour_HLC_correction(coords=current_coords, elems=elems, charge=charge, mult=mult,
                                                  theory=None, method=self.HOCC_method, basis=self.HOCC_basis,
                                                  ref=self.HOCC_ref, numcores=original_numcores, openshell=openshell)

            elif self.HOCC_program == 'MRCC':
                print("Doing HOCC correction via the MRCC program")
                from ash.interfaces.interface_MRCC import run_MRCC_HLC_correction
                E_HOCC = run_MRCC_HLC_correction(coords=current_coords, elems=elems, charge=charge, mult=mult,
                                                 theory=None, method=self.HOCC_method, basis=self.HOCC_basis,
                                                 ref=self.HOCC_ref,numcores=original_numcores, openshell=openshell)
        else:
            E_HOCC = 0.0
        ############################################################
        #Diagonal Born-Oppenheimer correction (DBOC)
        #via CFour
        ############################################################
        if self.DBOCcorrection is True:
            print("Doing DBOC correction")
            if self.DBOC_program == 'CFour':
                from ash.interfaces.interface_CFour import run_CFour_DBOC_correction
                print("Doing DBOC correction via CFour")
                E_DBOC = run_CFour_DBOC_correction(coords=current_coords, elems=elems, charge=charge, mult=mult,
                                                   method=self.DBOC_method, basis=self.DBOC_basis, numcores=original_numcores,
                                                   openshell=openshell)
        else:
            E_DBOC = 0.0
        ############################################################
        #Multireference correction
        #via PYSCF
        #NOT READY
        ############################################################

        ############################################################
        #Full-CI correction (Goodson)
        ############################################################
        if self.FCI is True:
            print("Extrapolating SCF-energy, CCSD-energy and CCSD(T) energy to Full-CI limit by Goodson formula")
            if self.HOCCcorrection is True:
                print("Error: FCI=True and HOCCcorrection=True is incompatible")
                ashexit()
            #Here using CBS-values for SCF, CCSD-corr and (T)-corr.
            #NOTE: We need to do separate extrapolation of corrCCSD_CBS and triples_CBS
            E_CC_CBS = E_SCF_CBS + E_corr_CBS
            E_FCI_CBS = FCI_extrapolation([E_SCF_CBS, E_corrCCSD_CBS, E_corrCCT_CBS])
            print("FCI/CBS energy (Goodson formula):", E_FCI_CBS, "Eh")
            E_FCIcorrection = E_FCI_CBS-E_CC_CBS
            #'E_FCIcorrection': E_FCIcorrection
            print("FCI correction:", E_FCIcorrection, "Eh")
            print("")
        else:
            E_FCIcorrection=0.0
        ############################################################
        #FINAL RESULT
        ############################################################
        print("")
        print("")
        print("="*50)
        print("FINAL RESULTS")
        print("="*50)
        E_CC_CBS = E_SCF_CBS + E_corr_CBS
        print("CCSD(T)/CBS energy (without corrections):", E_CC_CBS, "Eh")
        E_FINAL = E_CC_CBS + E_corecorr_and_SR + E_SO + E_DBOC + E_HOCC + E_FCIcorrection
        print("Final CBS energy (with all corrections):", E_FINAL, "Eh")
        #Components
        E_dict = {'Total_E' : E_FINAL, 'E_CC_CBS': E_CC_CBS, 'E_SCF_CBS' : E_SCF_CBS, 'E_corrCCSD_CBS': E_corrCCSD_CBS, 'E_corrCCT_CBS': E_corrCCT_CBS, 'T1energycorr' : T1energycorr,
            'E_corr_CBS' : E_corr_CBS, 'E_SO' : E_SO, 'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_HOCC': E_HOCC, 'E_DBOC': E_DBOC,
            'E_FCIcorrection': E_FCIcorrection}

        print("")
        print("Contributions:")
        print("--------------")
        print("E_SCF_CBS : ", E_SCF_CBS, "Eh")
        print("E_corr_CBS : ", E_corr_CBS, "Eh")
        print("E_corrCCSD_CBS : ", E_corrCCSD_CBS, "Eh")
        print("E_corrCCT_CBS : ", E_corrCCT_CBS, "Eh")
        print("T1 energy correction : ", T1energycorr, "Eh")
        print("Spin-orbit coupling correction : ", E_SO, "Eh")
        print("E_corecorr_and_SR correction : ", E_corecorr_and_SR, "Eh")
        print("E_HOCC correction : ", E_HOCC, "Eh")
        print("E_DBOC correction : ", E_DBOC, "Eh")
        print("E_FCI correction : ", E_FCIcorrection, "Eh")

        print("FINAL ENERGY :", E_FINAL, "Eh")
        #Setting energy_components as an accessible attribute
        self.energy_components=E_dict

        #Cleanup GBW file. Full cleanup ??
        # TODO: Keep output files for each step
        os.remove(self.ccsdt_1.filename+'.gbw')

        #return only final energy now. Energy components accessible as energy_components attribute
        return E_FINAL

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
        ashexit()
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
        self.basis1_block="%basis\n"
        for el,bas_ecp in self.Calc1_basis_dict.items():
            self.basis1_block=self.basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
            if bas_ecp[1] != None:
                #Setting ECP flag to True
                self.ECPflag=True
                self.basis1_block=self.basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])
        #Adding auxiliary basis
        #self.basis1_block=self.basis1_block+finalauxbasis
        self.basis1_block=self.basis1_block+"\nend"
        self.blocks1= self.blocks +self.basis1_block
        if self.singlebasis is False:
            self.basis2_block="%basis\n"
            for el,bas_ecp in self.Calc2_basis_dict.items():
                self.basis2_block=self.basis2_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    self.basis2_block=self.basis2_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])
            #Adding auxiliary basis
            #self.basis2_block=self.basis2_block+finalauxbasis
            self.basis2_block=self.basis2_block+"\nend"
            self.blocks2= self.blocks +self.basis2_block

        #MOdifying self.blocks to add finalauxbasis. Used by CVSR only
        #self.blocks=self.blocks+"%basis {} end".format(finalauxbasis)


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
    def CVSR_Step(self, current_coords, elems, reloption,calc_label, numcores, charge=None, mult=None, PC=None, current_MM_coords=None, MMcharges=None):
        print("\nCVSR_Step")

        ccsdt_mtsmall_NoFC_line="! {} {} {}   nofrozencore {} {} {}".format(self.ccsdtkeyword,self.CVbasis,self.auxbasiskeyword,self.pnokeyword,self.scfsetting,self.extrainputkeyword)
        ccsdt_mtsmall_FC_line="! {} {}  {} {} {} {}".format(self.ccsdtkeyword,self.CVbasis,self.auxbasiskeyword,self.pnokeyword,self.scfsetting,self.extrainputkeyword)

        ccsdt_mtsmall_NoFC = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=self.blocks, numcores=self.numcores)
        ccsdt_mtsmall_FC = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=self.blocks, numcores=self.numcores)

        #Run
        energy_ccsdt_mtsmall_nofc = ccsdt_mtsmall_NoFC.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_NoFC' + '.out')
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.gbw', './' + calc_label + 'CCSDT_MTsmall_NoFC' + '.gbw')

        energy_ccsdt_mtsmall_fc = ccsdt_mtsmall_FC.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult, PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_FC' + '.out')
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.gbw', './' + calc_label + 'CCSDT_MTsmall_FC' + '.gbw')

        #Core-correlation is total energy difference between NoFC-DKH and FC-norel
        E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
        print("E_corecorr_and_SR:", E_corecorr_and_SR)
        return E_corecorr_and_SR

    #NOTE: TODO: PC info ??
    #TODO: coords and elems vs. fragment issue
    def run(self, current_coords=None, elems=None, Grad=False, numcores=None, charge=None, mult=None, PC=None, current_MM_coords=None, MMcharges=None, mm_elems=None):

        print(BC.OKBLUE,BC.BOLD, "------------RUNNING MRCC_CC_CBS_Theory-------------", BC.END)
        ashexit()
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
            #E_SCF_CBS, E_corrCCSD_CBS = Extrapolation_twopoint(scf_energies, ccsdcorr_energies, self.cardinals, self.basisfamily,
            #    alpha=self.alpha, beta=self.beta, SCFextrapolation=self.SCFextrapolation) #2-point extrapolation
            #Separate CCSD and (T) CBS energies
            #E_SCF_CBS, E_corrCCT_CBS = Extrapolation_twopoint(scf_energies, triplescorr_energies, self.cardinals, self.basisfamily,
            #    alpha=self.alpha, beta=self.beta, SCFextrapolation=self.SCFextrapolation) #2-point extrapolation

            #BASIS SET EXTRAPOLATION of SCF and full correlation energies
            #E_SCF_CBS, E_corr_CBS = Extrapolation_twopoint(scf_energies, corr_energies, self.cardinals, self.basisfamily,
            #    alpha=self.alpha, beta=self.beta, SCFextrapolation=self.SCFextrapolation) #2-point extrapolation

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
        #print("alpha setting: {} ".format(extrapolation_parameters_dict[extrap_dict_key][0]))

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


#2-point formula for CBS extrapolation of correlation energy by Riemann Zeta function
# Lesiuk, Jeziorski, JCTC 2019, 15, 5398-5403
def Extrapolation_Riemann_zeta_corr(corr_energies, cardinals):

    if len(corr_energies) != len(cardinals):
        print("corr_energies and cardinals lists should be same length")
        ashexit()
    if len(corr_energies) != 2:
        print("corr_energies list should be size 2")
        ashexit()

    #Riemann zeta: ksi(s), ksi(4) = pi^4/90
    rz4 = math.pi**4/90

    #a = L^4(E_L - E_L-1)
    a = cardinals[-1]**4*(corr_energies[1]-corr_energies[0])

    #sum_ang: sum(l^-4 for l=1 to L)
    sum_ang=sum([l**(-4) for l in range(1,cardinals[1]+1)])

    #E_L
    E_inf = corr_energies[-1] + a*(rz4-sum_ang)
    print("Cardinals are:", cardinals)
    print("Correlation energies are:", corr_energies[0], "and", corr_energies[1])
    print("Correlation Extrapolated value is", E_inf)
    return E_inf


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

#Make ORCATHeory object for ICE-CI
#TODO: Allow basis-set element dictionary
#TODO: Allow external basis set file
def make_ICE_theory(basis,tgen, tvar, numcores, nel=None, norb=None, nmin_nmax=False, ice_nmin=None,ice_nmax=None,
    autoice=False, basis_per_element=None, maxcorememory=10000, maxiter=20, etol=1e-6, moreadfile=None,label=""):
    print_line_with_mainheader("make_ICE_theory")
    print("Simple function to create ICE-CI ORCATheory object")
    print()
    print("Basis:", basis)
    print("Tgen:", tgen)
    print("Tvar:", tvar)
    print("numcores:", numcores)
    print("nel:", nel)
    print("norb:", norb)
    print("nmin_nmax:", nmin_nmax)
    print("ice_nmin:", ice_nmin)
    print("ice_nmax:", ice_nmax)
    print("autoice:", autoice)
    print("basis_per_element:", basis_per_element)
    print("maxcorememory:", maxcorememory)
    print("maxiter:", maxiter)
    print("etol:", etol)
    print("moreadfile:", moreadfile)
    print()
    icekeyword=""
    mp2nat_option="false"
    moreadoption="";moreadblockoption=""
    #Setting basis keyword to nothing if basis set dict provided
    if basis_per_element is not None:
        basis=""
    #If AutoICE is True then set keyword and request MP2 nat orbs
    if autoice is True:
        icekeyword="Auto-ICE"
        mp2nat_option="true"
    # if nmin_nmax is True then CAS selected by e.g. nmax=1.999 and nmax=0.001
    if nmin_nmax is True:
        print("nmin_nmax is active. Will select CAS based on ice_nmin and ice_nmin keywords")
        if ice_nmin is None or ice_nmax is None:
            print("Error: ice_nmin and ice_nmax need to be set.")
            ashexit()
        CAS_space_line=f"""nmin {ice_nmin}
nmax {ice_nmax}"""
    #Otherwise we selcect by nel and norb
    else:
        print("nmin_nmax is NOT active. Will select CAS based on nel and norb keywords")
        if nel is None or norb is None:
            print("Error: nel and norb need to be set.")
            ashexit()
        CAS_space_line=f"""nel {nel}
norb {norb}"""
    #Optional MO read
    if moreadfile is not None:
        moreadoption="MOREAD"
        moreadblockoption=f"%moinp \"{moreadfile}\""
    input=f"! {icekeyword} {basis} {moreadoption} tightscf"
    #Setting ICE-CI so that frozen-core is applied (1s oxygen frozen). Note: Auto-ICE also required.
    blocks=f"""
{moreadblockoption}
%maxcore {maxcorememory}
%ice
{CAS_space_line}
tgen {tgen}
tvar {tvar}
useMP2nat {mp2nat_option}
maxiter {maxiter}
etol {etol}
natorbs true
end

"""
    icetheory = ash.ORCATheory(orcasimpleinput=input, orcablocks=blocks, numcores=numcores, basis_per_element=basis_per_element, label=f'{label}ICE_tgen{tgen}_tvar_{tvar}', save_output_with_label=True)
    return icetheory

#New Workflow to do ICE-CI calculation using CASSCF module using any input orbitals
#no orbital selection
def newAuto_ICE_CAS(fragment=None, basis="cc-pVDZ", basisblock="", moreadfile=None, nmin=1.98, nmax=0.02, extrainput="",nat_occupations=None,
                 gtol=2.50e-04, numcores=1, charge=None, mult=None, CASCI=True, tgen=1e-4, memory=10000, no_moreadfile_in_CAS=False,
                 ciblockline=""):

    if fragment is None:
        print("Error: No fragment provided to Auto_ICE_CAS.")
        ashexit()
    if nat_occupations is None:
        print("Error: No natural occupations provided to Auto_ICE_CAS.")
        ashexit()

    if CASCI is True:
        print("Warning: CAS-CI is active. This means that no orbital-optimization is carried out.")
        print("In order to get ICE-CI natural orbitals we don't use the noiter option but instead change the CASSCF convergence threshold so that only 1 iteration is carried out")
        print("Otherwise the input orbitals are still present in the GBW file")
        #noiterkeyword="noiter"
        gtol=999999 #This is necessary
        print("Setting gtol to:", gtol)
        label=f"ICE-CASCI"
    else:
        #Regular CASSCF
        label=f"ICE-CASSCF"

    #Determine CAS space based on thresholds
    print(f"Natorb. ccupations:", nat_occupations)
    print("\nTable of natural occupation numbers")
    print("")
    print("{:<9} {:6} ".format("Orbital", f"Nat-occ"))
    print("----------------------------------------")
    init_flag=False
    final_flag=False
    for index,(nocc) in enumerate(nat_occupations):
        if init_flag is False and nocc<nmin:
            print("-"*40)
            init_flag=True
        if final_flag is False and nocc<nmax:
            print("-"*40)
            final_flag=True
        print(f"{index:<9} {nocc:9.4f}")
    nel,norb=ash.functions.functions_elstructure.select_space_from_occupations(nat_occupations, selection_thresholds=[nmin,nmax])
    print(f"Selecting CAS({nel},{norb}) based on thresholds: upper_sel_threshold={nmin} and lower_sel_threshold={nmax}")

    #Rare option: Use selection above but do not read in file
    autostart=True
    if no_moreadfile_in_CAS:
        print("Warning: no_moreadfile_in_CAS option active. This means that we use the active space definition above")
        print("But we will not read in orbitals in the CAS-CI ICE step. This is not recommended")
        print("Turning off moreadfile option and setting autostart to False")
        moreadfile=None
        autostart=False

    #ICE-theory: Fixed active space
    casblocks=f"""
    %maxcore {memory}
    {basisblock}
    %casscf
    gtol {gtol}
    nel {nel}
    norb {norb}
    cistep ice
    ci
    tgen {tgen}
    {ciblockline}
    end
    end
    """
    ice_cas_CI = ash.ORCATheory(orcasimpleinput=f"! CASSCF {basis} tightscf",
                                orcablocks=casblocks, moreadfile=moreadfile, autostart=autostart, label=f"{label}", numcores=numcores)

    result_ICE = ash.Singlepoint(fragment=fragment, theory=ice_cas_CI, charge=charge,mult=mult)

    ICEnatoccupations=CASSCF_natocc_grab(f"{ice_cas_CI.filename}.out")

    print("Table of natural occupation numbers")
    print("")
    print("{:<9} {:6} {:6}".format("Orbital", f"Initial", "ICE-nat-occ"))
    print("----------------------------------------")
    init_flag=False
    final_flag=False
    for index,(initocc,iceocc) in enumerate(zip(nat_occupations,ICEnatoccupations)):
        if init_flag is False and initocc<nmin:
            print("-"*40)
            init_flag=True
        if final_flag is False and initocc<nmax:
            print("-"*40)
            final_flag=True
        print("{:<9} {:9.4f} {:9.4f}".format(index,initocc,iceocc))

    return result_ICE



#Workflow to do active-space selection with MP2 or CCSD and then ICE-CI
def Auto_ICE_CAS(fragment=None, basis="cc-pVDZ", nmin=1.98, nmax=0.02, extrainput="",
                 initial_orbitals="RI-MP2", MP2_density="unrelaxed", CC_density="unrelaxed", moreadfile=None, gtol=2.50e-04,
                 numcores=1, charge=None, mult=None, CASCI=True, tgen=1e-4, memory=10000, no_moreadfile_in_CAS=False,
                 ciblockline=""):

    if fragment is None:
        print("Error: No fragment provided to Auto_ICE_CAS.")
        ashexit()

    if 'MP2' in initial_orbitals:
        print("MP2-type orbitals requested")
        print("MP2_density option:", MP2_density)
    if 'CCSD' in initial_orbitals:
        print("CCSD-type orbitals requested")
        print("CC_density option:", CC_density)


    if CASCI is True:
        print("Warning: CAS-CI is active. This means that no orbital-optimization is carried out.")
        print("In order to get ICE-CI natural orbitals we don't use the noiter option but instead change the CASSCF convergence threshold so that only 1 iteration is carried out")
        print("Otherwise the input orbitals are still present in the GBW file")
        #noiterkeyword="noiter"
        gtol=999999 #This is necessary
        print("Setting gtol to:", gtol)
        label=f"ICE-CASCI"
    else:
        #Regular CASSCF
        #noiterkeyword=""
        label=f"ICE-CASSCF"
    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, "QM", fragment, "bla", theory=None)

    #Make MP2 natural orbitals
    mp2blocks=f"""
    %maxcore {memory}
    %mp2
    natorbs true
    density {MP2_density}
    end
    """
    #Make CCSD natural orbitals
    ccsdblocks=f"""
    %maxcore {memory}
    %mdci
    natorbs true
    density {CC_density}
    end
    """
    print("initial_orbitals:", initial_orbitals)

    #Starting from scratch unless moreadfile is provided
    if moreadfile == None:
        autostart_option=False

    #NOTE: HF, UHF-UNO and QRO not yet ready
    #Need to somehow account for space-selection based on occupation numbers
    if initial_orbitals =="HF" or initial_orbitals =="RHF" :
        print("Performing HF orbital calculation")
        exit()
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} HF {basis}  tightscf", numcores=numcores,
                                 label='RHF', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.gbw"
        #Dummy occupations
        natoccgrab=UHF_natocc_grab
    elif initial_orbitals =="UHF-UNO" or initial_orbitals =="UHF" :
        print("Performing UHF natural orbital calculation")
        exit()
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} UHF {basis} normalprint UNO tightscf", numcores=numcores,
                                 label='UHF', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.unso"
        natoccgrab=UHF_natocc_grab
    elif initial_orbitals =="UHF-QRO":
        print("Performing UHF-QRO natural orbital calculation")
        exit()
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} UHF {basis}  UNO tightscf", numcores=numcores,
                                 label='UHF-QRO', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.qro"
        natoccgrab=UHF_natocc_grab
    elif initial_orbitals =="MP2" :
        natorbs = ash.ORCATheory(orcasimpleinput=f"! {extrainput} MP2 {basis} autoaux tightscf", orcablocks=mp2blocks, numcores=numcores,
                                 label='MP2', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.mp2nat"
        natoccgrab=MP2_natocc_grab
    elif initial_orbitals =="RI-MP2" :
        natorbs = ash.ORCATheory(orcasimpleinput=f"! RI-MP2 {basis} autoaux tightscf", orcablocks=mp2blocks, numcores=numcores,
                                 label='MP2', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.mp2nat"
        natoccgrab=MP2_natocc_grab
    elif initial_orbitals =="CCSD":
        natorbs = ash.ORCATheory(orcasimpleinput=f"! CCSD {basis} autoaux tightscf", orcablocks=ccsdblocks, numcores=numcores,
                                 label='CCSD', save_output_with_label=True, autostart=autostart_option, moreadfile=moreadfile)
        mofile=f"{natorbs.filename}.mdci.nat"
        natoccgrab=CCSD_natocc_grab
    else:
        print("Error: initial_orbitals must be HF, MP2, RI-MP2 or CCSD")
        ashexit()

    #Run MP2/CCSD natorb calculation
    ash.Singlepoint(theory=natorbs, fragment=fragment, charge=charge, mult=mult)

    #Determine CAS space based on thresholds
    nat_occupations=natoccgrab(natorbs.filename+'.out')
    print(f"{initial_orbitals} Natorb. ccupations:", nat_occupations)
    print("\nTable of natural occupation numbers")
    print("")
    print("{:<9} {:6} ".format("Orbital", f"{initial_orbitals}-nat-occ"))
    print("----------------------------------------")
    init_flag=False
    final_flag=False
    for index,(nocc) in enumerate(nat_occupations):
        if init_flag is False and nocc<nmin:
            print("-"*40)
            init_flag=True
        if final_flag is False and nocc<nmax:
            print("-"*40)
            final_flag=True
        print(f"{index:<9} {nocc:9.4f}")
    nel,norb=ash.functions.functions_elstructure.select_space_from_occupations(nat_occupations, selection_thresholds=[nmin,nmax])
    print(f"Selecting CAS({nel},{norb}) based on thresholds: upper_sel_threshold={nmin} and lower_sel_threshold={nmax}")

    #Rare option: Use selection above but do not read in file
    autostart=True
    if no_moreadfile_in_CAS:
        print("Warning: no_moreadfile_in_CAS option active. This means that we use the active space definition above")
        print("But we will not read in orbitals in the CAS-CI ICE step. This is not recommended")
        print("Turning off moreadfile option and setting autostart to False")
        mofile=None
        autostart=False


    #ICE-theory: Fixed active space
    casblocks=f"""
    %maxcore {memory}
    %casscf
    gtol {gtol}
    nel {nel}
    norb {norb}
    cistep ice
    ci
    tgen {tgen}
    {ciblockline}
    end
    end
    """
    ice_cas_CI = ash.ORCATheory(orcasimpleinput=f"! CASSCF  {basis} tightscf",
                                orcablocks=casblocks, moreadfile=mofile, autostart=autostart, label=f"{label}", numcores=numcores)

    result_ICE = ash.Singlepoint(fragment=fragment, theory=ice_cas_CI, charge=charge,mult=mult)

    ICEnatoccupations=CASSCF_natocc_grab(f"{ice_cas_CI.filename}.out")

    print("Table of natural occupation numbers")
    print("")
    print("{:<9} {:6} {:6}".format("Orbital", f"{initial_orbitals}", "ICE-nat-occ"))
    print("----------------------------------------")
    init_flag=False
    final_flag=False
    for index,(initocc,iceocc) in enumerate(zip(nat_occupations,ICEnatoccupations)):
        if init_flag is False and initocc<nmin:
            print("-"*40)
            init_flag=True
        if final_flag is False and initocc<nmax:
            print("-"*40)
            final_flag=True
        print("{:<9} {:9.4f} {:9.4f}".format(index,initocc,iceocc))

    return result_ICE








#Function to do ICE-CI FCI with multiple thresholds and simpler WF method comparison and plotting
#TODO: add second y-axis to ICE-CI plot (plot CFGs). Or maybe add info as point-label?
#TODO: MP2 and ICE-CI natural orbitals for CCSD(T) wf options
#TODO: Test MRCC and CFour
#TODO: Fix for 1-electron systems
#TODO: Finish MBE
def Reaction_FCI_Analysis(reaction=None, basis=None, basisfile=None, basis_per_element=None,
                Do_ICE_CI=True,
                Do_TGen_fixed_series=True, fixed_tvar=1e-11, Do_Tau3_series=True, Do_Tau7_series=True, Do_EP_series=True,
                tgen_thresholds=None, ice_nmin=1.999, ice_nmax=0,
                separate_MP2_nat_initial_orbitals=True,
                DoHF=True,DoMP2=True, DoCC=True, DoCC_CCSD=True, DoCC_CCSDT=True, DoCC_MRCC=False, DoCC_Bruckner=False, DoCC_CFour=False, DoCAS=False,
                active_space_for_each=None,
                DoCC_DFTorbs=True, KS_functionals=['BP86','BHLYP'], Do_OOCC=True, Do_OOMP2=True,
                maxcorememory=10000, numcores=1, ice_ci_maxiter=30, ice_etol=1e-6,
                upper_sel_threshold=1.999, lower_sel_threshold=0,
                plot=True, y_axis_label='None', yshift=0.3, ylimits=None, padding=0.4):


    if DoCAS is True and active_space_for_each == None:
        print("Error: active_space_for_each must be set if DoCAS is True.")
        ashexit()


    #Dealing with different basis sets on atoms.
    #TODO: basisfile. Common basis-file for ORCA and other codes?
    #basis: string keyword for name of basis
    #basisfile: ORCA basis-file that will be read in, in each ORCA alculation. basisfile="combined_SVP_on_C_ccDZ_on_H.basis"
    #basisdict: dictionary with elements as keys as basisname as value. Example: basisdict={'C':'SVP','H':'cc-pVDZ'}
    if basis_per_element != None:
        basis=""

    #Looping over TGen thresholds in ICE-CI
    #results_ice = {}
    #results_ice_genCFGs={}
    #results_ice_selCFGs={}
    #results_ice_SDCFGs={}
    #Dicts with tgen,tvar combos
    results_ice_tgen_tvar={} # (1e-4,1e-11) = -1.322
    results_ice_tgen_tvar_Evar=defaultdict(lambda: []) #Variational ICE-CI energies
    results_ice_tgen_tvar_EPT2=defaultdict(lambda: []) #EPT2 rest energies
    results_ice_tgen_tvar_genCFGs=defaultdict(lambda: [])
    results_ice_tgen_tvar_selCFGs=defaultdict(lambda: [])
    results_ice_tgen_tvar_SDCFGs=defaultdict(lambda: [])

    tgen_thresholds=[round(i,20) for i in tgen_thresholds]

    ################
    if separate_MP2_nat_initial_orbitals is True:
        print("Running MP2 natural orbital calculation")
        #TODO
        #Do MP2-natural orbital calculation here
        mp2blocks=f"""
        %maxcore 11000
        %mp2
        natorbs true
        density unrelaxed
        end
        """
        natmp2 = ash.ORCATheory(orcasimpleinput=f"! MP2 {basis} autoaux tightscf", orcablocks=mp2blocks, basis_per_element=basis_per_element,numcores=numcores, label='MP2', save_output_with_label=True)
        for frag in reaction.fragments:
            print("frag.label:", frag.label)
            frag_label = "Frag_" + str(frag.formula) + "_" + str(frag.charge) + "_" + str(frag.mult) + "_"
            ash.Singlepoint(fragment=frag, theory=natmp2)
            shutil.copyfile(natmp2.filename+'.mp2nat', f'./{frag_label}MP2natorbs.mp2nat')
            #Determine CAS space based on thresholds
            step1occupations=ash.interfaces.interface_ORCA.MP2_natocc_grab(natmp2.filename+'.out')
            print("MP2natoccupations:", step1occupations)
            nel,norb=ash.functions.functions_elstructure.select_space_from_occupations(step1occupations, selection_thresholds=[upper_sel_threshold,lower_sel_threshold])
            print(f"Selecting CAS({nel},{norb}) based on thresholds: upper_sel_threshold={upper_sel_threshold} and lower_sel_threshold={lower_sel_threshold}")
            reaction.properties["CAS"].append([nel,norb])
            print("reaction.properties CAS:", reaction.properties["CAS"])

            #Adding to orbital dictionary of Reaction
            #NOTE: Keep info in reaction object or fragment object?
            reaction.orbital_dictionary["MP2nat"].append(f'./{frag_label}MP2natorbs.mp2nat')
            #Cleanup
            natmp2.cleanup()

        #Determining frag that has largest number of active electrons
        largest_fragindex=[i[0] for i in reaction.properties["CAS"]].index(max([i[0] for i in reaction.properties["CAS"]]))


    ################
    #Iterating over TGen thresholds with fixed default TVar =
    #Not using Singlepoint_reaction as we need more flexibility due to orbital-info in theory object etc.
    if Do_ICE_CI is True:
        if Do_Tau3_series is True:
            tau3_e_series=[]
            print("Do_Tau3_series True")
            for tgen in tgen_thresholds:
                print("tgen:", tgen)
                tvar=round(tgen*1E-3,20)
                print("tvar determined to be:",tvar)
                #Creating ICE-CI theory
                if (tgen,tvar) not in results_ice_tgen_tvar:
                    for i,frag in enumerate(reaction.fragments):
                        ice = make_ICE_theory(basis, tgen, tvar,numcores, nel=reaction.properties["CAS"][i][0], norb=reaction.properties["CAS"][i][1], basis_per_element=basis_per_element,
                            maxcorememory=maxcorememory, maxiter=ice_ci_maxiter, etol=ice_etol, moreadfile=reaction.orbital_dictionary["MP2nat"][i], label=f"Frag_{str(frag.formula)}")
                        result_ICE = ash.Singlepoint(fragment=frag, theory=ice)
                        ice.cleanup()
                        energy_ICE = result_ICE.energy
                        reaction.energies.append(energy_ICE)
                        #WF info
                        #print("ice.properties E_var:", ice.properties["E_var"])
                        results_ice_tgen_tvar_Evar[(tgen,tvar)].append(ice.properties["E_var"])
                        #print("results_ice_tgen_tvar_Evar", results_ice_tgen_tvar_Evar)
                        results_ice_tgen_tvar_EPT2[(tgen,tvar)].append(ice.properties["E_PT2_rest"])
                        results_ice_tgen_tvar_genCFGs[(tgen,tvar)].append(ice.properties["num_genCFGs"])
                        results_ice_tgen_tvar_selCFGs[(tgen,tvar)].append(ice.properties["num_selected_CFGs"])
                        results_ice_tgen_tvar_SDCFGs[(tgen,tvar)].append(ice.properties["num_after_SD_CFGs"])
                        print("results_ice_tgen_tvar_Evar", results_ice_tgen_tvar_Evar)
                        print("results_ice_tgen_tvar_genCFGs:", results_ice_tgen_tvar_genCFGs)
                        #exit()
                    reaction.calculate_reaction_energy()
                    rel_energy_ICE = reaction.reaction_energy
                    results_ice_tgen_tvar[(tgen,tvar)] = rel_energy_ICE
                    tau3_e_series.append(rel_energy_ICE)
                    reaction.reset_energies()
                else:
                    print(f"Tgen/Tvar ({tgen}/{tvar}) combo already calculated")
                    tau3_e_series.append(results_ice_tgen_tvar[(tgen,tvar)])

        #print("results_ice_tgen_tvar:", results_ice_tgen_tvar)
        #print("results_ice_tgen_tvar_EPT2:", results_ice_tgen_tvar_EPT2)
        print("results_ice_tgen_tvar_genCFGs:", results_ice_tgen_tvar_genCFGs)
        if Do_Tau7_series is True:
            tau7_e_series=[]
            print("Do_Tau7_series True")
            for tgen in tgen_thresholds:
                print("tgen:", tgen)
                tvar=round(tgen*1E-7,20)
                print("tvar determined to be:",tvar)
                #Creating ICE-CI theory
                if (tgen,tvar) not in results_ice_tgen_tvar:
                    for i,frag in enumerate(reaction.fragments):
                        ice = make_ICE_theory(basis, tgen, tvar,numcores, nel=reaction.properties["CAS"][i][0], norb=reaction.properties["CAS"][i][1], basis_per_element=basis_per_element,
                            maxcorememory=maxcorememory, maxiter=ice_ci_maxiter, etol=ice_etol, moreadfile=reaction.orbital_dictionary["MP2nat"][i], label=f"Frag_{str(frag.formula)}")
                        result_ICE = ash.Singlepoint(fragment=frag, theory=ice)
                        ice.cleanup()
                        energy_ICE = result_ICE.energy
                        print("energy_ICE:", energy_ICE)
                        reaction.energies.append(energy_ICE)
                        print("reaction.energies:", reaction.energies)
                        #WF info
                        results_ice_tgen_tvar_Evar[(tgen,tvar)].append(ice.properties["E_var"])
                        results_ice_tgen_tvar_EPT2[(tgen,tvar)].append(ice.properties["E_PT2_rest"])
                        results_ice_tgen_tvar_genCFGs[(tgen,tvar)].append(ice.properties["num_genCFGs"])
                        results_ice_tgen_tvar_selCFGs[(tgen,tvar)].append(ice.properties["num_selected_CFGs"])
                        results_ice_tgen_tvar_SDCFGs[(tgen,tvar)].append(ice.properties["num_after_SD_CFGs"])
                    reaction.calculate_reaction_energy()
                    rel_energy_ICE = reaction.reaction_energy
                    results_ice_tgen_tvar[(tgen,tvar)] = rel_energy_ICE
                    tau7_e_series.append(rel_energy_ICE)
                    reaction.reset_energies()
                else:
                    print(f"Tgen/Tvar ({tgen}/{tvar}) combo already calculated")
                    tau7_e_series.append(results_ice_tgen_tvar[(tgen,tvar)])

        if Do_TGen_fixed_series is True:
            taufixed_e_series=[]
            print("Do_TGen_fixed_series True")
            tvar=round(fixed_tvar,20)
            print("Fixed Tvar :", tvar)
            for tgen in tgen_thresholds:
                print("tgen:", tgen)
                #Creating ICE-CI theory
                if (tgen,tvar) not in results_ice_tgen_tvar:
                    for i,frag in enumerate(reaction.fragments):
                        ice = make_ICE_theory(basis, tgen, tvar,numcores, nel=reaction.properties["CAS"][i][0], norb=reaction.properties["CAS"][i][1], basis_per_element=basis_per_element,
                            maxcorememory=maxcorememory, maxiter=ice_ci_maxiter, etol=ice_etol, moreadfile=reaction.orbital_dictionary["MP2nat"][i], label=f"Frag_{str(frag.formula)}")
                        result_ICE = ash.Singlepoint(fragment=frag, theory=ice)
                        ice.cleanup()
                        energy_ICE = result_ICE.energy
                        reaction.energies.append(energy_ICE)
                        #WF info
                        results_ice_tgen_tvar_Evar[(tgen,tvar)].append(ice.properties["E_var"])
                        results_ice_tgen_tvar_EPT2[(tgen,tvar)].append(ice.properties["E_PT2_rest"])
                        results_ice_tgen_tvar_genCFGs[(tgen,tvar)].append(ice.properties["num_genCFGs"])
                        results_ice_tgen_tvar_selCFGs[(tgen,tvar)].append(ice.properties["num_selected_CFGs"])
                        results_ice_tgen_tvar_SDCFGs[(tgen,tvar)].append(ice.properties["num_after_SD_CFGs"])
                    reaction.calculate_reaction_energy()
                    rel_energy_ICE = reaction.reaction_energy
                    results_ice_tgen_tvar[(tgen,tvar)] = rel_energy_ICE
                    taufixed_e_series.append(rel_energy_ICE)
                    reaction.reset_energies()
                else:
                    print(f"Tgen/Tvar ({tgen}/{tvar}) combo already calculated")
                    taufixed_e_series.append(results_ice_tgen_tvar[(tgen,tvar)])

        #Determining best ICE_CI value to center y-axis zoom on
        #NOTE: For now using last value
        rel_energy_ICE_last=rel_energy_ICE

        if Do_EP_series is True and Do_Tau3_series is True:
            #List of lists of [[1,2],[1,2]]
            E_extrap_tau3_2p=[[] for i in reaction.fragments]
            E_extrap_tau3_3p=[[] for i in reaction.fragments]

            print("ICE-CI extrapolations series")
            print("results_ice_tgen_tvar_Evar:", results_ice_tgen_tvar_Evar)
            print("results_ice_tgen_tvar_EPT2:", results_ice_tgen_tvar_EPT2)
            print("results_ice_tgen_tvar:", results_ice_tgen_tvar)

            print("reaction.fragments:", reaction.fragments)
            for i,frag in enumerate(reaction.fragments):
                E_pt2_tau3=[]
                E_var_tau3=[]
                E_tot_tau3=[]
                for tgen in tgen_thresholds:
                    if Do_Tau3_series:
                        tvar = round(tgen*1e-3,20)
                        E_v = results_ice_tgen_tvar_Evar[(tgen,tvar)][i]
                        E_pt2 = results_ice_tgen_tvar_EPT2[(tgen,tvar)][i]
                        E_var_tau3.append(E_v)
                        E_pt2_tau3.append(E_pt2)
                        E_tot_tau3.append(E_v+E_pt2)
                        if len(E_tot_tau3) > 1:
                            #2-point extrap. with last 2 points
                            twop_slope, twop_E_extrap = np.polyfit([abs(j) for j in E_pt2_tau3[-2:]],E_tot_tau3[-2:],1)
                            #3-point extrap. with last 3 points
                            threep_slope, threep_E_extrap = np.polyfit([abs(j) for j in E_pt2_tau3[-3:]],E_tot_tau3[-3:],1)
                            E_extrap_tau3_2p[i].append(twop_E_extrap)
                            if len(E_tot_tau3) > 2:
                                E_extrap_tau3_3p[i].append(threep_E_extrap)




            #Extrapolated Reaction energies
            if Do_Tau3_series:
                tau3_EP_series_2p=[]
                tau3_EP_series_3p=[]
                for i,tgen in enumerate(E_extrap_tau3_2p[0]):
                    list_of_energies_2p=[E_extrap_tau3_2p[f][i] for f,l in enumerate(E_extrap_tau3_2p)]
                    reaction_energy_2p = ash.ReactionEnergy(list_of_energies=list_of_energies_2p, stoichiometry=reaction.stoichiometry, unit=reaction.unit, silent=False)[0]
                    tau3_EP_series_2p.append(reaction_energy_2p)
                for i,tgen in enumerate(E_extrap_tau3_3p[0]):
                    list_of_energies_3p=[E_extrap_tau3_3p[f][i] for f,l in enumerate(E_extrap_tau3_3p)]
                    reaction_energy_3p = ash.ReactionEnergy(list_of_energies=list_of_energies_3p, stoichiometry=reaction.stoichiometry, unit=reaction.unit, silent=False)[0]
                    tau3_EP_series_3p.append(reaction_energy_3p)
                #print("tau3_EP_series_2p:", tau3_EP_series_2p)
                #print("tau3_EP_series_3p:", tau3_EP_series_3p)

            #Confidence interval
            #TODO
        print("results_ice_tgen_tvar:", results_ice_tgen_tvar)


    #Running regular single-reference WF methods
    results_cc={}
    if DoHF is True:
        hfblocks=f"""
        %maxcore 11000
        """
        hf = ash.ORCATheory(orcasimpleinput=f"! HF {basis} tightscf", orcablocks=hfblocks, basis_per_element=basis_per_element,numcores=numcores, label='HF', save_output_with_label=True)

        result_HF = ash.Singlepoint_reaction(reaction=reaction, theory=hf)
        results_cc['HF'] = result_HF.reaction_energy
        hf.cleanup()
    if DoMP2 is True:
        mp2blocks=f"""
        %maxcore 11000
        """
        mp2 = ash.ORCATheory(orcasimpleinput=f"! MP2 {basis} tightscf", orcablocks=mp2blocks, basis_per_element=basis_per_element,numcores=numcores, label='MP2', save_output_with_label=True)
        scsmp2 = ash.ORCATheory(orcasimpleinput=f"! SCS-MP2 {basis} tightscf", orcablocks=mp2blocks, basis_per_element=basis_per_element,numcores=numcores, label='SCSMP2', save_output_with_label=True)
        result_MP2 = ash.Singlepoint_reaction(reaction=reaction, theory=mp2)
        mp2.cleanup()
        result_SCSMP2 = ash.Singlepoint_reaction(reaction=reaction, theory=scsmp2)
        scsmp2.cleanup()
        results_cc['MP2'] = result_MP2.reaction_energy
        results_cc['SCS-MP2'] = result_SCSMP2.reaction_energy
        if Do_OOMP2 is True:
            oomp2 = ash.ORCATheory(orcasimpleinput=f"! OO-RI-MP2 autoaux {basis} tightscf", orcablocks=mp2blocks, basis_per_element=basis_per_element,numcores=numcores, label='OOMP2', save_output_with_label=True)
            scsoomp2 = ash.ORCATheory(orcasimpleinput=f"! OO-RI-SCS-MP2 {basis} autoaux tightscf", orcablocks=mp2blocks, basis_per_element=basis_per_element,numcores=numcores, label='OOSCSMP2', save_output_with_label=True)
            result_OOMP2 = ash.Singlepoint_reaction(reaction=reaction, theory=oomp2)
            oomp2.cleanup()
            result_SCSOOMP2 = ash.Singlepoint_reaction(reaction=reaction, theory=scsoomp2)
            scsoomp2.cleanup()
            results_cc['OO-MP2'] = result_OOMP2.reaction_energy
            results_cc['OO-SCS-MP2'] = result_SCSOOMP2.reaction_energy

    if DoCC is True:
        #Reducing numcores here for small systems. CC-code complains if numcores exceeds pairs.
        print("")
        nuccharges = [frag.nuccharge for frag in reaction.fragments]
        smallest_frag=nuccharges.index(min(nuccharges))
        newnumcores = check_cores_vs_electrons(reaction.fragments[smallest_frag].elems,numcores,reaction.fragments[smallest_frag].charge)
        if newnumcores != numcores:
            print("Reducing number of cores based on system-size")
            numcores=newnumcores
        ccblocks=f"""
        %maxcore 11000
        %mdci
        maxiter	300
        end
        """
        brucknerblocks=f"""
        %maxcore 11000
        %mdci
        maxiter 300
        Brueckner true
        end
        """
        if DoCC_CCSD is True:
            ccsd = ash.ORCATheory(orcasimpleinput=f"! CCSD {basis} tightscf", orcablocks=ccblocks, basis_per_element=basis_per_element,numcores=numcores, label='CCSD', save_output_with_label=True)
            bccd = ash.ORCATheory(orcasimpleinput=f"! CCSD {basis} tightscf", orcablocks=brucknerblocks, basis_per_element=basis_per_element,numcores=numcores, label='BCCD', save_output_with_label=True)
            #pccsd_1a = ash.ORCATheory(orcasimpleinput=f"! pCCSD/1a {basis} tightscf", orcablocks=ccblocks, basis_per_element=basis_per_element,numcores=numcores, label='pCCSD1a', save_output_with_label=True)
            #pccsd_2a = ash.ORCATheory(orcasimpleinput=f"! pCCSD/2a {basis} tightscf", orcablocks=ccblocks, basis_per_element=basis_per_element,numcores=numcores, label='pCCSD2a', save_output_with_label=True)

            result_CCSD = ash.Singlepoint_reaction(reaction=reaction, theory=ccsd)
            ccsd.cleanup()
            if DoCC_Bruckner is True:
                result_BCCD = ash.Singlepoint_reaction(reaction=reaction, theory=bccd)
                bccd.cleanup()
                results_cc['BCCD'] = result_BCCD.reaction_energy
            #result_pCCSD1a = ash.Singlepoint_reaction(reaction=reaction, theory=pccsd_1a)
            #pccsd_1a.cleanup()
            #result_pCCSD2a = ash.Singlepoint_reaction(reaction=reaction, theory=pccsd_2a)
            #pccsd_2a.cleanup()
            results_cc['CCSD'] = result_CCSD.reaction_energy


            if Do_OOCC is True:
                ooccd = ash.ORCATheory(orcasimpleinput=f"! OOCCD {basis} tightscf", orcablocks=ccblocks, basis_per_element=basis_per_element,numcores=numcores, label='OOCCD', save_output_with_label=True)
                result_OOCCD = ash.Singlepoint_reaction(reaction=reaction, theory=ooccd)
                ooccd.cleanup()
                results_cc['OOCCD'] = result_OOCCD.reaction_energy
            #results_cc['pCCSD/1a'] = result_pCCSD1a.reaction_energy
            #results_cc['pCCSD/2a'] = result_pCCSD2a.reaction_energy
        if DoCC_CCSDT is True:
            ccsdt = ash.ORCATheory(orcasimpleinput=f"! CCSD(T) {basis} tightscf", orcablocks=ccblocks, basis_per_element=basis_per_element,numcores=numcores, label='CCSDT', save_output_with_label=True)
            ccsdt_qro = ash.ORCATheory(orcasimpleinput=f"! UHF CCSD(T) {basis} UNO tightscf", orcablocks=ccblocks, basis_per_element=basis_per_element,numcores=numcores, label='CCSDT_QRO', save_output_with_label=True)
            #CCSD(T) extrapolated to FCI
            if basis_per_element ==None:
                ccsdt_fci_extrap = ORCA_CC_CBS_Theory(elements=reaction.fragments[0].elems, cardinals = [2], basisfamily="cc", numcores=1, FCI=True)
                result_CCSDT_FCI_extrap = ash.Singlepoint_reaction(reaction=reaction, theory=ccsdt_fci_extrap)
                ccsdt_fci_extrap.cleanup()
                results_cc['CCSD(T)-FCI-extrap'] = result_CCSDT_FCI_extrap.reaction_energy
            else:
                print("Warning: Basisdict per element provided. Can currently not do CCSD(T) FCI ORCA_CC_CBS_Theory job. SKipping")
            #CCSD(T) with MP2 and ICE orbitals
            #TODO: Need to figure out
            #ccsdt_mp2nat = ash.ORCATheory(orcasimpleinput=f"! CCSD(T) {basis} tightscf", orcablocks=ccblocks, numcores=numcores, label='CCSDT_mp2nat', save_output_with_label=True, moreadfile="ICEcalc_MP2natorbs.mp2nat")
            #ccsdt_icecinat = ash.ORCATheory(orcasimpleinput=f"! CCSD(T) {basis} tightscf", orcablocks=ccblocks, numcores=numcores, label='CCSDT_icecinat', save_output_with_label=True, moreadfile="ICEcalc_ICEnatorbs.gbw")
            #Run
            result_CCSDT = ash.Singlepoint_reaction(reaction=reaction, theory=ccsdt)
            ccsdt.cleanup()
            result_CCSDT_QRO = ash.Singlepoint_reaction(reaction=reaction, theory=ccsdt_qro)
            ccsdt_qro.cleanup()
            if DoCC_Bruckner is True:
                bccdt = ash.ORCATheory(orcasimpleinput=f"! CCSD(T) {basis} tightscf", orcablocks=brucknerblocks, basis_per_element=basis_per_element,numcores=numcores, label='BCCDT', save_output_with_label=True)
                result_BCCDT = ash.Singlepoint_reaction(reaction=reaction, theory=bccdt)
                bccdt.cleanup()
                results_cc['BCCD(T)'] = result_BCCDT.reaction_energy
            #relE_CCSDT_mp2nat = ash.Singlepoint_reaction(reaction=reaction, theory=ccsdt_mp2nat, unit=reaction.unit)
            #relE_CCSDT_icecinat = ash.Singlepoint_reaction(reaction=reaction, theory=ccsdt_icecinat, unit=reaction.unit)
            results_cc['CCSD(T)'] = result_CCSDT.reaction_energy
            results_cc['CCSD(T)-QRO'] = result_CCSDT_QRO.reaction_energy
            if Do_OOCC is True:
                ooccdt = ash.ORCATheory(orcasimpleinput=f"! OOCCD(T) {basis} tightscf", orcablocks=ccblocks, basis_per_element=basis_per_element,numcores=numcores, label='OOCCDT', save_output_with_label=True)
                result_OOCCDT = ash.Singlepoint_reaction(reaction=reaction, theory=ooccdt)
                ooccdt.cleanup()
                results_cc['OOCCD(T)'] = result_OOCCDT.reaction_energy

            #CCSD(T) with multiple DFT orbitals
            if DoCC_DFTorbs is True:
                print("Running CCSD(T) with multiple functionals")
                for functional in KS_functionals:
                    print("Doing orbitals from:", functional)
                    #Hybrid KS reference requires auxC basis (singles term) so adding autoaux
                    ccsdt_dft = ash.ORCATheory(orcasimpleinput=f"! CCSD(T) {functional} {basis} autoaux tightscf", orcablocks=ccblocks, numcores=numcores, label=f'CCSDT_{functional}', save_output_with_label=True)
                    result_CCSDT_DFT = ash.Singlepoint_reaction(reaction=reaction, theory=ccsdt_dft)
                    ccsdt_dft.cleanup()
                    results_cc[f'CCSD(T)-{functional}'] = result_CCSDT_DFT.reaction_energy


        if DoCC_MRCC is True:
            print("CC calculations using MRCC")
            print("not ready")
            #TODO: Define frozen core to be consistent with above
            ccsdt_mrccinput=f"""
            basis={basis}
            calc=CCSD(T)
            mem=9000MB
            scftype=UHF
            ccmaxit=150
            core=frozen
            """
            ccsd_t = ash.MRCCTheory(mrccinput=ccsdt_mrccinput,numcores=numcores, label='MRCC-CCSD(T)')
            ccsdt_mrccinput=f"""
            basis={basis}
            calc=CCSDT
            mem=9000MB
            scftype=UHF
            ccmaxit=150
            core=frozen
            """
            ccsdt = ash.MRCCTheory(mrccinput=ccsdt_mrccinput,numcores=numcores, label='MRCC-CCSDT')

            result_CCSD_T = ash.Singlepoint_reaction(reaction=reaction, theory=ccsd_t)
            ccsd_t.cleanup()
            result_CCSDT = ash.Singlepoint_reaction(reaction=reaction, theory=ccsdt)
            ccsdt.cleanup()
            results_cc['MRCC-CCSD(T)'] = result_CCSD_T.reaction_energy
            results_cc['MRCC-CCSDT'] = result_CCSDT.reaction_energy
        if DoCC_CFour is True:
            print("CC calculations using CFour")
            print("not ready")
            #TODO: Define frozen core to be consistent with above
            cfouroptions = {
            'CALC':'CCSD(T)',
            'BASIS':'PVDZ',
            'REF':'RHF',
            'FROZEN_CORE':'ON',
            'MEM_UNIT':'MB',
            'MEMORY':3100,
            'PROP':'FIRST_ORDER',
            'CC_PROG':'ECC',
            'SCF_CONV':10,
            'LINEQ_CONV':10,
            'CC_MAXCYC':300,
            'SYMMETRY':'OFF',
            'HFSTABILITY':'OFF'
            }
            ccsd_t = ash.CFourTheory(cfouroptions=cfouroptions,numcores=numcores, label='CFour-CCSD(T)')
            cfouroptions = {
            'CALC':'CCSDT',
            'BASIS':'PVDZ',
            'REF':'RHF',
            'FROZEN_CORE':'ON',
            'MEM_UNIT':'MB',
            'MEMORY':3100,
            'PROP':'FIRST_ORDER',
            'CC_PROG':'ECC',
            'SCF_CONV':10,
            'LINEQ_CONV':10,
            'CC_MAXCYC':300,
            'SYMMETRY':'OFF',
            'HFSTABILITY':'OFF'
            }
            ccsdt = ash.CFourTheory(cfouroptions=cfouroptions,numcores=numcores, label='CFour-CCSDT')

            result_CCSD_T = ash.Singlepoint_reaction(reaction=reaction, theory=ccsd_t)
            ccsd_t.cleanup()
            result_CCSDT = ash.Singlepoint_reaction(reaction=reaction, theory=ccsdt)
            ccsdt.cleanup()
            results_cc['C4-CCSD(T)'] = result_CCSD_T.reaction_energy
            results_cc['C4-CCSDT'] = result_CCSDT.reaction_energy


    #Running multireference methods
    if DoCAS is True:
        results_cas={}
        casscf_energies=[]
        caspt2_energies=[]
        nevpt2_energies=[]
        mrciq_energies=[]

        for i,frag in enumerate(reaction.fragments):
            casblocks=f"""%maxcore 11000
    %casscf
    nel {active_space_for_each[i][0]}
    norb {active_space_for_each[i][1]}
    maxiter	300
    end
    """
            caslabel=f'CASSCF_{active_space_for_each[i][0]}_{active_space_for_each[i][1]}'
            casscf = ash.ORCATheory(orcasimpleinput=f"! CASSCF {basis} tightscf", orcablocks=casblocks, basis_per_element=basis_per_element,
                                    numcores=numcores, label=caslabel, save_output_with_label=True,
                                    moreadfile=reaction.orbital_dictionary["MP2nat"][i])
            caspt2 = ash.ORCATheory(orcasimpleinput=f"! CASPT2 {basis} tightscf", orcablocks=casblocks, basis_per_element=basis_per_element,
                                    numcores=numcores, label=caslabel, save_output_with_label=True)
            nevpt2 = ash.ORCATheory(orcasimpleinput=f"! NEVPT2 {basis} tightscf", orcablocks=casblocks, basis_per_element=basis_per_element,
                                    numcores=numcores, label=caslabel, save_output_with_label=True)
            mrci_q = ash.ORCATheory(orcasimpleinput=f"! MRCI+Q {basis} tightscf", orcablocks=casblocks, basis_per_element=basis_per_element,
                                    numcores=numcores, label=caslabel, save_output_with_label=True)

            result_CASSCF = ash.Singlepoint(fragment=frag, theory=casscf)
            casscf_energies.append(result_CASSCF.energy)
            result_CASPT2 = ash.Singlepoint(fragment=frag, theory=caspt2)
            caspt2_energies.append(result_CASPT2.energy)
            result_NEVPT2 = ash.Singlepoint(fragment=frag, theory=nevpt2)
            nevpt2_energies.append(result_NEVPT2.energy)
            result_MRCIQ = ash.Singlepoint(fragment=frag, theory=mrci_q)
            mrciq_energies.append(result_MRCIQ.energy)



        #Reaction energies
        reaction_energy_casscf = ash.ReactionEnergy(list_of_energies=casscf_energies, stoichiometry=reaction.stoichiometry,
                                                    unit=reaction.unit, silent=False)[0]
        reaction_energy_caspt2 = ash.ReactionEnergy(list_of_energies=caspt2_energies, stoichiometry=reaction.stoichiometry,
                                                    unit=reaction.unit, silent=False)[0]
        reaction_energy_nevpt2 = ash.ReactionEnergy(list_of_energies=nevpt2_energies, stoichiometry=reaction.stoichiometry,
                                                    unit=reaction.unit, silent=False)[0]
        reaction_energy_mrciq = ash.ReactionEnergy(list_of_energies=mrciq_energies, stoichiometry=reaction.stoichiometry,
                                                    unit=reaction.unit, silent=False)[0]

        results_cas['CASSCF'] = reaction_energy_casscf
        results_cas['CASPT2'] = reaction_energy_caspt2
        results_cas['NEVPT2'] = reaction_energy_nevpt2
        results_cas['MRCI+Q'] = reaction_energy_mrciq

    ##########################################
    #Printing final results
    ##########################################
    SR_WF_indices=[];SR_WF_energies=[];SR_WF_labels=[]
    MR_WF_indices=[];MR_WF_energies=[];MR_WF_labels=[]


    print()
    print()
    if Do_ICE_CI is True:
        print("ICE-CI CIPSI wavefunction")
        print("Note: # CFGs refer to largest fragment calculated")
        print(f" TGen      TVar         Energy ({reaction.unit})        # gen. CFGs      # sel. CFGs    # max S+D CFGs")
        print("---------------------------------------------------------------------------------------------------------")
        for t, e in results_ice_tgen_tvar.items():
            gen_cfg=results_ice_tgen_tvar_genCFGs[(t[0],t[1])][largest_fragindex]
            sel_cg=results_ice_tgen_tvar_selCFGs[(t[0],t[1])][largest_fragindex]
            sd_cfg=results_ice_tgen_tvar_SDCFGs[(t[0],t[1])][largest_fragindex]
            tg=t[0]; tv=t[1]
            print("{:<9.1e} {:<9.1e} {:15.7f} {:15} {:15} {:15}".format(tg,tv,e,gen_cfg,sel_cg, sd_cfg))
        # Extrapolated values
        if Do_EP_series:
            print("------------")
            for tg,e_2p in zip(tgen_thresholds[1:],tau3_EP_series_2p):
                print("{:<9.1e} {:<9} {:15.7f}".format(tg,"Tau3-EP(2p)",e_2p))
            for tg,e_3p in zip(tgen_thresholds[2:],tau3_EP_series_3p):
                print("{:<9.1e} {:<9} {:15.7f}".format(tg,"Tau3-EP(3p)",e_3p))
        print()
        print()

        #y-limits based on last ICE calculation rel energy
        if ylimits == None:
            ylimits = [rel_energy_ICE_last-yshift,rel_energy_ICE_last+yshift]
        print(f"Using y-limits: {ylimits} {reaction.unit} in plot")

    print("Other methods:")
    print(f" WF           Energy ({reaction.unit})")
    print("-------------------------------------------------")
    #SR
    for i,(w, e) in enumerate(results_cc.items()):
        print("{:<22} {:<13.8f}".format(w,e))
        if plot is True:
            SR_WF_indices.append(i)
            SR_WF_energies.append(e)
            SR_WF_labels.append(w)
    #MR
    if DoCAS is True:
        for i,(w, e) in enumerate(results_cas.items()):
            print("{:<22} {:<13.8f}".format(w,e))
            if plot is True:
                MR_WF_indices.append(i)
                MR_WF_energies.append(e)
                MR_WF_labels.append(w)
    print();print()

    #Plotting if plot is True and if matplotlib worked
    #Create ASH_plot object named edplot
    if plot is True:

        if basis_per_element is not None:
            basislabel=""
            for i,j in basis_per_element.items():
                basislabel+=f" {i}({j})"
            basislabel=','.join(basis_per_element.values())
            print("basislabel:", basislabel)
        else:
            basislabel=basis

        if DoCAS is True:
            eplot = ASH_plot("Plotname", num_subplots=3, x_axislabels=["TGen", "Method","Method"], y_axislabels=[f'{y_axis_label} ({reaction.unit})',f'',f''], subplot_titles=[f"ICE-CI/{basislabel}",f"Singleref./{basislabel}",f"Multiref./{basislabel}"],
                ylimit=ylimits, horizontal=True, padding=padding)
        else:
            eplot = ASH_plot("Plotname", num_subplots=2, x_axislabels=["TGen", "Method"], y_axislabels=[f'{y_axis_label} ({reaction.unit})',f'{y_axis_label} ({reaction.unit})',], subplot_titles=[f"ICE-CI/{basislabel}",f"Singleref./{basislabel}"],
                ylimit=ylimits, horizontal=True, padding=padding)
        print("eplot:", eplot)
        if eplot.working is not False:
            #Inverting x-axis on subplot 0
            eplot.invert_x_axis(0) #
            if Do_ICE_CI is True:
                #Add dataseries to subplot 0 and using log-scale for ICE-CI data
                if Do_Tau3_series:
                    eplot.addseries(0, x_list=tgen_thresholds, y_list=tau3_e_series, label="Tau3", color='blue', line=True, scatter=True, x_scale_log=True)
                if Do_Tau7_series:
                    eplot.addseries(0, x_list=tgen_thresholds, y_list=tau7_e_series, label="Tau7", color='orange', line=True, scatter=True, x_scale_log=True)
                if Do_TGen_fixed_series:
                    eplot.addseries(0, x_list=tgen_thresholds, y_list=taufixed_e_series, label=f"Tvar:{fixed_tvar}", color='green', line=True, scatter=True, x_scale_log=True)
                if Do_EP_series:
                    if Do_Tau3_series:
                        eplot.addseries(0, x_list=tgen_thresholds[1:], y_list=tau3_EP_series_2p, label=f"EP-Tau3-2point", color='black', line=True, scatter=True, x_scale_log=True)
                        eplot.addseries(0, x_list=tgen_thresholds[2:], y_list=tau3_EP_series_3p, label=f"EP-Tau3-3point", color='purple', line=True, scatter=True, x_scale_log=True)
            #Plotting method labels on x-axis with rotation to make things fit
            eplot.addseries(1, x_list=SR_WF_indices, y_list=SR_WF_energies, x_labels=SR_WF_labels, label=reaction.label, color='red', line=True, scatter=True, xticklabelrotation=80)
            #Plotting method labels on x-axis with rotation to make things fit
            if DoCAS is True:
                eplot.addseries(2, x_list=MR_WF_indices, y_list=MR_WF_energies, x_labels=MR_WF_labels, label=reaction.label, color='purple', line=True, scatter=True, xticklabelrotation=80)
            #Save figure
            eplot.savefig(f'{reaction.label}_FCI')
        else:
            print("Could not plot data due to ASH_plot problem.")

#Simple FCI correction. Not working yet
def Reaction_FCI_correction(reaction=None, basis=None, basis_per_element=None, numcores=1, maxcorememory=4000,
        upper_sel_threshold=1.999, lower_sel_threshold=0.01 ):

    ice_ci_maxiter=30
    ice_etol=1e-6

    print("upper_sel_threshold:", upper_sel_threshold)
    print("lower_sel_threshold:", lower_sel_threshold)

    separate_MP2_nat_initial_orbitals=True
    if separate_MP2_nat_initial_orbitals is True:
        print("Running MP2 natural orbital calculation")
        #TODO
        #Do MP2-natural orbital calculation here
        mp2blocks=f"""
        %maxcore {maxcorememory}
        %mp2
        natorbs true
        density unrelaxed
        end
        """
        natmp2 = ash.ORCATheory(orcasimpleinput=f"! MP2 {basis} autoaux tightscf", orcablocks=mp2blocks, basis_per_element=basis_per_element,numcores=numcores, label='MP2', save_output_with_label=True)
        for frag in reaction.fragments:
            print("frag.label:", frag.label)
            frag_label = "Frag_" + str(frag.formula) + "_" + str(frag.charge) + "_" + str(frag.mult) + "_"
            ash.Singlepoint(fragment=frag, theory=natmp2)
            shutil.copyfile(natmp2.filename+'.mp2nat', f'./{frag_label}MP2natorbs.mp2nat')
            #Determine CAS space based on thresholds
            step1occupations=ash.interfaces.interface_ORCA.MP2_natocc_grab(natmp2.filename+'.out')
            print("MP2natoccupations:", step1occupations)
            nel,norb=ash.functions.functions_elstructure.select_space_from_occupations(step1occupations, selection_thresholds=[upper_sel_threshold,lower_sel_threshold])
            print(f"Selecting CAS({nel},{norb}) based on thresholds: upper_sel_threshold={upper_sel_threshold} and lower_sel_threshold={lower_sel_threshold}")
            reaction.properties["CAS"].append([nel,norb])
            print("reaction.properties CAS:", reaction.properties["CAS"])

            #Adding to orbital dictionary of Reaction
            #NOTE: Keep info in reaction object or fragment object?
            reaction.orbital_dictionary["MP2nat"].append(f'./{frag_label}MP2natorbs.mp2nat')
            #Cleanup
            natmp2.cleanup()

            #exit()

    #Determining frag that has largest number of active electrons
    largest_fragindex=[i[0] for i in reaction.properties["CAS"]].index(max([i[0] for i in reaction.properties["CAS"]]))

    #Frag_H2O1_1_2_MP2natorbs.mp2nat
    print("reaction.orbital_dictionary:", reaction.orbital_dictionary)


    tgen=1e-4
    tvar=1e-11

    ice_energies=[]
    ccsdt_energies_orca=[]
    ccsdt_energies_mrcc=[]
    for i,frag in enumerate(reaction.fragments):
        print("frag")

        #ICE-CI
        ice = make_ICE_theory(basis, tgen, tvar,numcores, nel=reaction.properties["CAS"][i][0],
                        norb=reaction.properties["CAS"][i][1], basis_per_element=basis_per_element,
                        maxcorememory=maxcorememory, maxiter=ice_ci_maxiter, etol=ice_etol,
                        moreadfile=reaction.orbital_dictionary["MP2nat"][i], label=f"Frag_{str(frag.formula)}")
        result_ICE = ash.Singlepoint(fragment=frag, theory=ice)
        energy_ICE = result_ICE.energy
        ice_energies.append(energy_ICE)
        ice.cleanup()
        #CC
        ccline=f"! CCSD(T) {basis} autoaux tightscf"
        virtorb_nat_threshold=lower_sel_threshold
        ccblocks=f"""
        %maxcore 11000
        %mdci
        maxiter	300
        Tnat {virtorb_nat_threshold}
        end
        """
        #ORCA-CCSD(T)
        ccsdt_orca = ash.ORCATheory(orcasimpleinput=ccline, orcablocks=ccblocks, basis_per_element=basis_per_element,numcores=numcores, label='CCSDT', save_output_with_label=True)
        result_ccsdt_orca = ash.Singlepoint(theory=ccsdt_orca, fragment=frag)
        e_ccsdt_orca = result_ccsdt_orca.energy
        ccsdt_energies_orca.append(e_ccsdt_orca)
        #MCC-CCSD(T)
        #https://www.mrcc.hu/MRCC/manual/html/single/manual_sp.xhtml
        #lnoepsv thresh
        ccsdt_mrccinput=f"""
        basis={basis}
        calc=FNO-CCSD(T)
        mem=9000MB
        scftype=UHF
        ccmaxit=150
        core=frozen
        ovirt=MP2
        lnoepsv={virtorb_nat_threshold}
        """
        ccsdt_mrcc = ash.MRCCTheory(mrccinput=ccsdt_mrccinput,numcores=numcores, label='CCSDT', save_output_with_label=True)
        result_ccsdt_mrcc = ash.Singlepoint(theory=ccsdt_mrcc, fragment=frag)
        e_ccsdt_mrcc = result_ccsdt_mrcc.energy
        ccsdt_energies_mrcc.append(e_ccsdt_mrcc)

    print("ice_energies:", ice_energies)
    print("ccsdt_energies_orca:", ccsdt_energies_orca)
    print("ccsdt_energies_mrcc:", ccsdt_energies_mrcc)
    reaction_energy_ice = ash.ReactionEnergy(list_of_energies=ice_energies, stoichiometry=reaction.stoichiometry, unit=reaction.unit, silent=False)[0]
    reaction_energy_cc_orca = ash.ReactionEnergy(list_of_energies=ccsdt_energies_orca, stoichiometry=reaction.stoichiometry, unit=reaction.unit, silent=False)[0]
    reaction_energy_cc_mrcc = ash.ReactionEnergy(list_of_energies=ccsdt_energies_mrcc, stoichiometry=reaction.stoichiometry, unit=reaction.unit, silent=False)[0]

    #delta_A=reaction_energy_ice-reaction_energy_cc
    #print(f"delta_A: {delta_A} {reaction.unit}")


#MRCI+Q/CBS theory, inspired by Reimann, Kaupp JCTC 2023, 19, 97-108
#Not ready
#TODO: MRCI-correction: freeze all occupied orbitals except 3s-3p. check both MRCI and CASPT2 step
#TODO: moreadfile option to read in orbitals. Maybe different orbitals for different cardinals ?
#TODO: Separate function to prepare orbitals
class ORCA_MRCI_CBS_Theory:
    def __init__(self, elements=None, scfsetting='TightSCF', extrainputkeyword='', extrablocks='', memory=5000, numcores=1,
            cardinals=None, basisfamily=None,  SCFextrapolation=True, alpha=None, beta=None, orcadir=None, relativity=None,
            atomicSOcorrection=False, auxbasis="autoaux-max", MRPT2_method="CASPT2", MRHL_method="MRCI+Q", MRHL_basis="small",
            active_space=None, F12=False, IPEA_shift=0.25) :

        print_line_with_mainheader("ORCA_MRCI_CBS_Theory")

        #Indicates that this is a QMtheory
        self.theorytype="QM"

        #CHECKS to exit early
        if elements == None:
            print(BC.FAIL, "\nORCA_MRCI_CBS_Theory requires a list of elements to be given in order to set up basis sets", BC.END)
            print("Example: ORCA_MRCI_CBS_Theory(elements=['C','Fe','S','H','Mo'], basisfamily='def2',cardinals=[2,3], ...")
            print("Should be a list containing all elements that a fragment might contain")
            ashexit()
        else:
            #Removing redundant symbols (in case fragment.elems list was passed for example)
            elements = list(set(elements))



        #Check if only 1 cardinal was chosen: meaning no extrapolation and just a single
        if cardinals == None:
            print("Error: cardinals must be given as a list of integers")
            ashexit()
        elif len(cardinals) == 1:
            print(BC.WARNING, "Only a single cardinal was chosen. This means that no extrapolation will be carried out", BC.END)
            self.singlebasis=True
        else:
            self.singlebasis=False


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
        if F12 is True and 'NEVPT2' not in MRPT2_method:
            print(BC.FAIL,"F12 calculations require MRPT2 to be NEVPT2.", BC.END)
            ashexit()


        #Main attributes
        self.cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        self.orcadir = orcadir
        self.elements=elements
        self.cardinals = cardinals
        self.alpha=alpha
        self.beta=beta
        self.SCFextrapolation=SCFextrapolation
        self.F12=F12
        self.basisfamily = basisfamily
        self.relativity = relativity
        self.numcores=numcores
        self.memory=memory
        self.scfsetting=scfsetting
        self.MRPT2_method=MRPT2_method
        self.MRHL_method = MRHL_method
        self.MRHL_basis=MRHL_basis
        self.active_space=active_space
        self.IPEA_shift=IPEA_shift

        self.extrainputkeyword=extrainputkeyword
        self.extrablocks=extrablocks
        self.atomicSOcorrection=atomicSOcorrection

        #ECP-flag may be set to True later
        self.ECPflag=False

        print("-----------------------------")
        print("ORCA_MRCI_CBS PROTOCOL")
        print("-----------------------------")
        print("Settings:")
        print("Cardinals chosen:", self.cardinals)
        print("Basis set family chosen:", self.basisfamily)
        print("F12:", self.F12)
        print("SCF extrapolation:", self.SCFextrapolation)
        print("Elements involved:", self.elements)
        print("Number of cores: ", self.numcores)
        print("Maxcore setting: ", self.memory, "MB")
        print("SCF setting: ", self.scfsetting)
        print("Relativity: ", self.relativity)
        print("Atomic spin-orbit correction: ", self.atomicSOcorrection)
        print("")

        ######################################################
        # BLOCK-INPUT
        ######################################################

        if self.MRPT2_method == "CASPT2":
            caspt2_settings=f"CASPT2_IPEAshift {self.IPEA_shift}"
        else:
            caspt2_settings=""
        if self.F12 is True:
            f12_option="""PTMethod FIC_NEVPT2
"""
        else:
            f12_option=""

        #Block input for CASSCF block options.
        #Disabling FullLMP2 guess in general as not available for open-shell
        #Adding memory and extrablocks.
        self.blocks=f"""%maxcore {memory}
%casscf
trafostep ri
PTSettings
{caspt2_settings}
f12 {str(self.F12).lower()}
end
{f12_option}
nel {active_space[0]}
norb {active_space[1]}
maxiter	300
end
{extrablocks}\n"""


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
            self.basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                self.basis1_block=self.basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis1_block=self.basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding basis set info for each element into blocks
            self.basis2_block="%basis\n"
            for el,bas_ecp in self.Calc2_basis_dict.items():
                self.basis2_block=self.basis2_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis2_block=self.basis2_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding auxiliary basis to all defined blocks
            self.basis0_block=self.basis0_block+finalauxbasis+"\nend"
            self.basis1_block=self.basis1_block+finalauxbasis+"\nend"
            self.basis2_block=self.basis2_block+finalauxbasis+"\nend"

            self.blocks0= self.blocks +basis0_block #Used in CCSD and CCSD(T) calcs
            self.blocks1= self.blocks +self.basis1_block #Used in CCSD and CCSD(T) calcs
            self.blocks2= self.blocks +self.basis2_block  #Used in CCSD calcs only
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
            self.basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                self.basis1_block=self.basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis1_block=self.basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding basis set info for each element into blocks
            self.basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                self.basis1_block=self.basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis1_block=self.basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding basis set info for each element into blocks
            self.basis2_block="%basis\n"
            for el,bas_ecp in self.Calc2_basis_dict.items():
                self.basis2_block=self.basis2_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis2_block=self.basis2_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])

            #Adding auxiliary basis to all defined blocks
            self.basis1_block=self.basis1_block+finalauxbasis+"\nend"
            self.basis2_block=self.basis2_block+finalauxbasis+"\nend"

            self.blocks1= self.blocks +self.basis1_block #Used in CCSD(T) calcs
            self.blocks2= self.blocks +self.basis2_block #Used in CCSD(T) calcs

        elif len(self.cardinals) == 1:
            #Single-basis calculation
            self.Calc1_basis_dict={}
            for elem in elements:
                bas=basis_for_element(elem, basisfamily, cardinals[0])
                self.Calc1_basis_dict[elem] = bas
            print("Basis set definitions for each element:")
            print("Calc1_basis_dict:", self.Calc1_basis_dict)
            #Adding basis set info for each element into blocks
            self.basis1_block="%basis\n"
            for el,bas_ecp in self.Calc1_basis_dict.items():
                self.basis1_block=self.basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
                if bas_ecp[1] != None:
                    #Setting ECP flag to True
                    self.ECPflag=True
                    self.basis1_block=self.basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])
            self.basis1_block=self.basis1_block+finalauxbasis+"\nend"
            self.blocks1= self.blocks +self.basis1_block


        #Auxiliary basis to self.blocks. Used by CVSR only:
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


        self.mrpt2_input_line=f"! {self.MRPT2_method} {self.scfsetting} {self.extrainputkeyword} {self.mainbasiskeyword} {self.auxbasiskeyword}"

        self.mrhl_input_line=f"! {self.MRHL_method} {self.scfsetting} {self.extrainputkeyword} {self.mainbasiskeyword}"

        ##########################################################################################
        #Defining two theory objects for each basis set unless F12 or single-cardinal provided
        #And 1 theory object for MRHL correction
        ##########################################################################################
        if self.singlebasis is True:
            #For single-basis CCSD(T) or single-basis F12 calculations
            self.mrpt2_1 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.mrpt2_input_line, orcablocks=self.blocks1, numcores=self.numcores)
        else:
            #Extrapolations

            #Regular direct CCSD(T) extrapolations on both cardinals
            self.mrpt2_1 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.mrpt2_input_line, orcablocks=self.blocks1, numcores=self.numcores)
            self.mrpt2_2 = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.mrpt2_input_line, orcablocks=self.blocks2, numcores=self.numcores)
        #HL correction
        if self.MRHL_method != "None":
            if self.MRHL_basis=="small":
                self.mrhl = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.mrhl_input_line,
                                                                    orcablocks=self.blocks1, numcores=self.numcores)
            elif self.MRHL_basis=="large":
                self.mrhl = ash.interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.mrhl_input_line,
                                                                    orcablocks=self.blocks2, numcores=self.numcores)
            elif self.MRHL_basis=="extrapolation":
                print("not ready")
            else:
                print("unknow MRHL_basis option")
                ashexit()


    def cleanup(self):
        print("Cleanup called")

    def run(self, current_coords=None, qm_elems=None,
            elems=None, Grad=False, numcores=None, charge=None, mult=None, PC=None, current_MM_coords=None, MMcharges=None, mm_elems=None):

        print(BC.OKBLUE,BC.BOLD, "------------RUNNING ORCA_MRCI_CBS_Theory-------------", BC.END)


        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for ORCATheory.run method", BC.END)
            ashexit()

        if PC is True:
            print("Pointcharge embedding in ORCA_MRCI_CBS_Theory is active")
            elems=qm_elems

        if Grad == True:
            print(BC.FAIL,"No gradient available for ORCA_MRCI_CBS_Theory yet! Exiting", BC.END)
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

        #Defining initial label here based on element and charge/mult of system
        formula=elemlisttoformula(elems)
        calc_label = "Frag_" + str(formula) + "_" + str(charge) + "_" + str(mult) + "_"
        print("Initial Calculation label: ", calc_label)


        #CONTROLLING NUMCORES
        #If numcores not provided to run, use self.numcores
        if numcores == None:
            numcores=self.numcores

        original_numcores = numcores
        #Reduce numcores if required
        #NOTE: self.numcores is thus ignored if check_cores_vs_electrons reduces value based on system-size
        numcores = check_cores_vs_electrons(elems,numcores,charge)

        MRPT2_1_dict={}
        MRPT2_2_dict={}

        ############
        #Basis-1
        ############
        self.mrpt2_1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult,
                         PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)

        casscf_energy = float(pygrep('Final CASSCF energy', self.mrpt2_1.filename+'.out')[4])
        caspt2_corr_energy = float(pygrep('Total Energy Correction :', self.mrpt2_1.filename+'.out')[-1])
        MRPT2_1_dict['CASSCF'] = casscf_energy
        MRPT2_1_dict['PT2_corr_energy'] = caspt2_corr_energy

        ############
        #Basis-2
        ############
        if self.singlebasis is False:
            self.mrpt2_2.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult,
                            PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)

            casscf_energy = float(pygrep('Final CASSCF energy', self.mrpt2_2.filename+'.out')[4])
            caspt2_corr_energy = float(pygrep('Total Energy Correction :', self.mrpt2_2.filename+'.out')[-1])
            MRPT2_2_dict['CASSCF'] = casscf_energy
            MRPT2_2_dict['PT2_corr_energy'] = caspt2_corr_energy


        ####################
        #MRCI HL correction
        ####################
        print("Now doing HL MRCI correction")
        #TODO: option to extrapolate to CBS
        if self.MRHL_method != "None":
            self.mrhl.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult,
                                PC=PC, current_MM_coords=current_MM_coords, MMcharges=MMcharges)

            #MRCI total energy (2 lines are found, taking last one)
            casscf_ref_energy = float(pygrep('Final CASSCF energy', self.mrhl.filename+'.out')[4])
            mrci_total_energy = float(pygrep2('STATE   0:  Energy', self.mrhl.filename+'.out')[-1].split()[3])
            E_corr_MRCI = mrci_total_energy-casscf_ref_energy

            if self.MRHL_basis == "small":
                delta_corr_MRCI = E_corr_MRCI - MRPT2_1_dict['PT2_corr_energy']
            elif self.MRHL_basis == "large":
                delta_corr_MRCI = E_corr_MRCI - MRPT2_2_dict['PT2_corr_energy']
            else:
                print("unknown")
                exit()
        else:
            delta_corr_MRCI=0.0

        print("Delta HL-MRCI correction:", delta_corr_MRCI, "Eh")

        ####################
        #EXTRAPOLATION
        ####################

        #List of all SCF energies  all CCSD-corr energies  and all (T) corr energies from the 2 jobs
        if self.singlebasis is True:
            E_SCF_CBS=MRPT2_1_dict['CASSCF']
            E_corr_MRPT2_CBS=MRPT2_1_dict['PT2_corr_energy']
        else:
            casscf_energies = [MRPT2_1_dict['CASSCF'], MRPT2_2_dict['CASSCF']]
            corr_energies= [MRPT2_1_dict['PT2_corr_energy'], MRPT2_2_dict['PT2_corr_energy']]
            print("")
            print("scf_energies :", casscf_energies)
            print("corr_energies:", corr_energies)

            #BASIS SET EXTRAPOLATION
            #SCF extrapolation. WIll be overridden inside function if self.SCFextrapolation==True
            print("\nSCF extrapolation:")
            E_SCF_CBS = Extrapolation_twopoint_SCF(casscf_energies, self.cardinals, self.basisfamily,
                alpha=self.alpha, SCFextrapolation=self.SCFextrapolation) #2-point SCF extrapolation

            E_corr_MRPT2_CBS = Extrapolation_twopoint_corr(corr_energies, self.cardinals, self.basisfamily,
                beta=self.beta) #2-point extrapolation





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
        print("="*50)
        print("FINAL RESULTS")
        print("="*50)
        E_FINAL = E_SCF_CBS + E_corr_MRPT2_CBS + delta_corr_MRCI + E_SO
        print("Final CBS energy (with all corrections):", E_FINAL, "Eh")
        #Components
        E_dict = {}

        print("")
        print("Contributions:")
        print("--------------")
        print("E_CASSCF_CBS : ", E_SCF_CBS, "Eh")
        print("E_corr_MRPT2_CBS : ", E_corr_MRPT2_CBS, "Eh")
        print("E_corrMRCI : ", delta_corr_MRCI, "Eh")
        print("Spin-orbit coupling correction : ", E_SO, "Eh")

        print("FINAL ENERGY :", E_FINAL, "Eh")
        #Setting energy_components as an accessible attribute
        self.energy_components=E_dict

        #Cleanup GBW file. Full cleanup ??


        #return only final energy now. Energy components accessible as energy_components attribute
        return E_FINAL


def iterative_MR_nat(fragment=None, theory=None,basis=None, numcores=1, maxiter=7, extrainput="", moreadfile=None):

    #Getting initial orbitals, e.g. ROHF
    extrablock="""
%scf
directresetfreq 1
diismaxeq 20
end"""
    orbfile, natocc = ORCA_orbital_setup(orbitals_option="ROHF", extrainput=extrainput, extrablock=extrablock,
                                         fragment=fragment, basis=basis, moreadfile=moreadfile,
                                         charge=fragment.charge, mult=fragment.mult)

    #Doing initial MR calculation
    CAS_nel=fragment.mult-1
    CAS_norb=CAS_nel

    #DDCI1, DDCI2, DDCI2.5, DDCI3, DDCI3.5
    MR_excitation_options_all = [{'1h':1,'1p':1,'1h1p':1,'2h':0, '2p':0, '2h1p':0, '1h2p':0,'2h2p':0, 'ss':0, 'isss':0, 'issa':0, 'ssss':0, 'sssa':0},
                              {'1h':1,'1p':1,'1h1p':1,'2h':1, '2p':1, '2h1p':0, '1h2p':0,'2h2p':0, 'ss':0, 'isss':0, 'issa':0, 'ssss':0, 'sssa':0},
                              {'1h':1,'1p':1,'1h1p':1,'2h':1, '2p':1, '2h1p':1, '1h2p':0,'2h2p':0, 'ss':0, 'isss':0, 'issa':0, 'ssss':0, 'sssa':0},
                              {'1h':1,'1p':1,'1h1p':1,'2h':1, '2p':1, '2h1p':1, '1h2p':1,'2h2p':0, 'ss':0, 'isss':0, 'issa':0, 'ssss':0, 'sssa':0},
                              {'1h':1,'1p':1,'1h1p':1,'2h':1, '2p':1, '2h1p':1, '1h2p':1,'2h2p':1, 'ss':0, 'isss':0, 'issa':0, 'ssss':0, 'sssa':0},
                              {'1h':1,'1p':1,'1h1p':1,'2h':1, '2p':1, '2h1p':1, '1h2p':1,'2h2p':1, 'ss':1, 'isss':1, 'issa':0, 'ssss':0, 'sssa':0},
                              {'1h':1,'1p':1,'1h1p':1,'2h':1, '2p':1, '2h1p':1, '1h2p':1,'2h2p':1, 'ss':1, 'isss':1, 'issa':1, 'ssss':1, 'sssa':1}]
    MRCI_natorbiterations_all=[5,5,5,5,5,5,5]
    MRCI_tsel_all=[1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]

    #LOOPING OVER MRCI
    for macroiter in range(0,maxiter):
        print("===============================")
        print("MACROITERATION:", macroiter)
        print("===============================")
        #Setting up MR calculation
        MR_excitation_options = MR_excitation_options_all[macroiter]
        print("Using  MR_excitation_options:", MR_excitation_options)
        MRCI_natorbiterations=MRCI_natorbiterations_all[macroiter]
        print("Using MRCI_natorbiterations:", MRCI_natorbiterations)
        MRCI_tsel=MRCI_tsel_all[macroiter]
        print("Using MRCI_tsel:", MRCI_tsel)
        blocks=f"""
%casscf
nel {CAS_nel}
norb {CAS_norb}
end

%mrci
newblock {fragment.mult} *
excitations none
refs cas({CAS_nel},{CAS_norb}) end
Flags[is]   {MR_excitation_options['1h']}
Flags[sa]   {MR_excitation_options['1p']}
Flags[ia]   {MR_excitation_options['1h1p']}
Flags[ijss] {MR_excitation_options['2h']}
Flags[ssab] {MR_excitation_options['2p']}
Flags[ijsa] {MR_excitation_options['2h1p']}
Flags[isab] {MR_excitation_options['1h2p']}
Flags[ijab] {MR_excitation_options['2h2p']}
Flags[ss] {MR_excitation_options['ss']}
Flags[isss] {MR_excitation_options['isss']}
Flags[issa] {MR_excitation_options['issa']}
Flags[ssss] {MR_excitation_options['ssss']}
Flags[sssa] {MR_excitation_options['sssa']}
nroots 1
end
AllSingles true
PrintWF det
natorbiters {MRCI_natorbiterations}
natorbs 2
tsel {MRCI_tsel}
end
"""
        mrtheory = ash.ORCATheory(orcasimpleinput=f"! {basis} CASSCF noiter", orcablocks=blocks, moreadfile=orbfile, numcores=numcores)

        print("Now")
        ash.Singlepoint(theory=mrtheory, fragment=fragment)
        os.rename("orca.out", f"orca_MR_iter{macroiter}.out")
        shutil.copy("orca.b0_s0.nat", f"orca_MR_iter{macroiter}.b0_s0.nat.gbw")
        orbfile="orca.b0_s0.nat"
        print("--------------------------")
        print("Now next")
