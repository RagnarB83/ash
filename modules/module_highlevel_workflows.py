#High-level WFT workflows
import numpy as np
import os
import ash
import shutil
import constants
import math
import copy
import dictionaries_lists
import interfaces.interface_ORCA
from functions.functions_elstructure import num_core_electrons, check_cores_vs_electrons
from functions.functions_general import ashexit, BC, print_line_with_mainheader
from modules.module_coords import elemlisttoformula, nucchargelist,elematomnumbers
from modules.module_coords import check_charge_mult

# Allowed basis set families. Accessed by function basis_for_element and extrapolation
basisfamilies=['cc','aug-cc','cc-dkh','cc-dk','aug-cc-dkh','aug-cc-dk','def2','ma-def2','def2-zora', 'def2-dkh',
            'ma-def2-zora','ma-def2-dkh', 'cc-CV', 'aug-cc-CV', 'cc-CV-dkh', 'cc-CV-dk', 'aug-cc-CV-dkh', 'aug-cc-CV-dk',
            'cc-CV_3dTM-cc_L', 'aug-cc-CV_3dTM-cc_L', 'cc-f12', 'def2-x2c' ]


#Flexible CCSD(T)/CBS protocol class. Simple. No core-correlation, scalar relativistic or spin-orbit coupling for now.
# Regular CC, DLPNO-CC, DLPNO-CC with PNO extrapolation etc.
#alpha and beta can be manually set. If not set then they are picked based on basisfamily
#NOTE: List of elements are required here
class CC_CBS_Theory:
    def __init__(self, elements=None, cardinals = None, basisfamily=None, relativity=None, orcadir=None, 
           stabilityanalysis=False, numcores=1, CVSR=False, CVbasis="W1-mtsmall", F12=False, Openshellreference=None, DFTreference=None, DFT_RI=False, auxbasis="autoaux-max",
                        DLPNO=False, memory=5000, pnosetting='extrapolation', pnoextrapolation=[6,7], FullLMP2Guess=False, T1=True, scfsetting='TightSCF',
                        alpha=None, beta=None, extrainputkeyword='', extrablocks='', FCI=False, guessmode='Cmatrix', atomicSOcorrection=False):
        """
        WORK IN PROGRESS
        CCSD(T)/CBS frozencore workflow

        :param elements: list of element symbols
        :param orcadir: ORCA directory
        :param stabilityanalysis: stability analysis on or off.
        :param numcores: number of cores
        :param memory: Memory in MB
        :param scfsetting: ORCA keyword (e.g. NormalSCF, TightSCF, VeryTightSCF)
        :param F12: True/False
        :param DLPNO: True/False  
        :param pnosetting: ORCA keyword: NormalPNO, LoosePNO, TightPNO or extrapolation
        :param pnoextrapolation: list. e.g. [5,6]
        ;param T1: Boolean (whether to do expensive iterative triples or not)
        :return: energy and dictionary with energy-components.
        """

        print_line_with_mainheader("CC_CBS_Theory")

        #Indicate that this is a QMtheory
        self.theorytype="QM"

        #CHECKS to exit early 
        if elements == None:
            print(BC.FAIL, "\nCC_CBS_Theory requires a list of elements to be given in order to set up basis sets", BC.END)
            print("Example: CC_CBS_Theory(elements=['C','Fe','S','H','Mo'], basisfamily='def2',cardinals=[2,3], ...")
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
        self.scfsetting=scfsetting
        self.alpha=alpha
        self.beta=beta
        self.extrainputkeyword=extrainputkeyword
        self.extrablocks=extrablocks
        self.FullLMP2Guess=FullLMP2Guess
        self.FCI=FCI
        self.atomicSOcorrection=atomicSOcorrection
        #ECP-flag may be set to True later
        self.ECPflag=False
        print("-----------------------------")
        print("CC_CBS PROTOCOL")
        print("-----------------------------")
        print("Settings:")
        print("Cardinals chosen:", self.cardinals)
        print("Basis set family chosen:", self.basisfamily)
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
            print("FullLMP2Guess ", self.FullLMP2Guess)
            if self.pnosetting == "extrapolation":
                print("pnoextrapolation:", self.pnoextrapolation)
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

        #SCALAR RELATIVITY HAMILTONIAN
        if self.relativity == None:
            if self.basisfamily in ['cc-dkh', 'aug-cc-dkh', 'cc-dk', 'aug-cc-dk', 'def2-zora', 'def2-dkh', 'cc-CV_3dTM-cc_L', 'aug-cc-CV_3dTM-cc_L'
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



        #Final simple-input line
        self.ccsdt_line="! {} {} {} {} {} {} printbasis".format(self.ccsdtkeyword, self.mainbasiskeyword, self.pnokeyword, self.scfsetting,self.extrainputkeyword,self.auxbasiskeyword)

        ##########################################################################################
        #Defining two theory objects for each basis set unless F12 or single-cardinal provided
        ##########################################################################################
        if self.singlebasis is True:
            #For single-basis CCSD(T) or single-basis F12 calculations
            self.ccsdt_1 = interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsdt_line, orcablocks=self.blocks1, numcores=self.numcores)
        else:
            #Extrapolations
            self.ccsdt_1 = interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsdt_line, orcablocks=self.blocks1, numcores=self.numcores)
            self.ccsdt_2 = interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=self.ccsdt_line, orcablocks=self.blocks2, numcores=self.numcores)


    def cleanup(self):
        print("Cleanup called")


    #Core-Valence ScalarRelativistic Step
    def CVSR_Step(self, current_coords, elems, reloption,calc_label, numcores, charge=None, mult=None):

        #Note: if reloption=='DKH' then we do DKH in NoFC and not in FC
        # if reloption=='' then no relativity
        #TODO: Option to use DKH for both, consistent with other calculations if using DKH ??

        ccsdt_mtsmall_NoFC_line="! {} {} {}   nofrozencore {} {} {} {}".format(self.ccsdtkeyword,reloption,self.CVbasis,self.auxbasiskeyword,self.pnokeyword,self.scfsetting,self.extrainputkeyword)
        ccsdt_mtsmall_FC_line="! {} {}  {} {} {} {}".format(self.ccsdtkeyword,self.CVbasis,self.auxbasiskeyword,self.pnokeyword,self.scfsetting,self.extrainputkeyword)

        ccsdt_mtsmall_NoFC = interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=self.blocks, numcores=self.numcores)
        ccsdt_mtsmall_FC = interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=self.blocks, numcores=self.numcores)

        #Run
        energy_ccsdt_mtsmall_nofc = ccsdt_mtsmall_NoFC.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.gbw', './' + calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.gbw')
        
        energy_ccsdt_mtsmall_fc = ccsdt_mtsmall_FC.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')
        shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.gbw', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.gbw')

        #Core-correlation is total energy difference between NoFC-DKH and FC-norel
        E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
        print("E_corecorr_and_SR:", E_corecorr_and_SR)
        return E_corecorr_and_SR


    # Do 2 calculations with different DLPNO thresholds and extrapolate
    def PNOExtrapolationStep(self,elems=None, current_coords=None, theory=None, calc_label=None, numcores=None, charge=None, mult=None):
        #elems=None, current_coords=None, theory=None, pnoextrapolation=None, DLPNO=None, F12=None, calc_label=None
        print("Inside PNOExtrapolationStep")
        #Adding TCutPNO option X
        #TightPNO options for other thresholds
        mdciblockX="""\n%mdci
    TCutPNO 1e-{}
    TCutPairs 1e-5
    TCutDO 5e-3
    TCutMKN 1e-3
    end
        
        """.format(self.pnoextrapolation[0])
        #TCutPNO option Y
        #TightPNO options for other thresholds
        mdciblockY="""\n%mdci
    TCutPNO 1e-{}
    TCutPairs 1e-5
    TCutDO 5e-3
    TCutMKN 1e-3
    end
        
        """.format(self.pnoextrapolation[1])

        #Keeping copy of original ORCA blocks of theory
        orcablocks_original=copy.copy(theory.orcablocks)
        #Add mdciblock to blocks of theory
        PNOXblocks = theory.orcablocks + mdciblockX
        PNOYblocks = theory.orcablocks + mdciblockY
        
        theory.orcablocks = PNOXblocks
        
        theory.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
        PNOcalcX_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(theory.filename+'.out', DLPNO=self.DLPNO,F12=self.F12)
        shutil.copyfile(theory.filename+'.out', './' + calc_label + '_PNOX' + '.out')
        shutil.copyfile(theory.filename+'.gbw', './' + calc_label + '_PNOX' + '.gbw')
        print("PNOcalcX:", PNOcalcX_dict)


        theory.orcablocks = PNOYblocks
        #ash.Singlepoint(fragment=fragment, theory=theory)
        theory.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
        PNOcalcY_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(theory.filename+'.out', DLPNO=self.DLPNO,F12=self.F12)
        shutil.copyfile(theory.filename+'.out', './' + calc_label + '_PNOY' + '.out')
        shutil.copyfile(theory.filename+'.gbw', './' + calc_label + '_PNOY' + '.gbw')
        print("PNOcalcY:", PNOcalcY_dict)
        
        #Setting theory.orcablocks back to original
        theory.orcablocks=orcablocks_original


        #Extrapolation to PNO limit

        E_SCF = PNOcalcY_dict['HF']
        #Extrapolation CCSD part and (T) separately
        # NOTE: Is this correct??
        E_corrCCSD_final = PNO_extrapolation([PNOcalcX_dict['CCSD_corr'],PNOcalcY_dict['CCSD_corr']])
        E_corrCCT_final = PNO_extrapolation([PNOcalcX_dict['CCSD(T)_corr'],PNOcalcY_dict['CCSD(T)_corr']])
        #Extrapolation of full correlation energy
        E_corrCC_final = PNO_extrapolation([PNOcalcX_dict['full_corr'],PNOcalcY_dict['full_corr']])

        print("PNO extrapolated CCSD correlation energy:", E_corrCCSD_final, "Eh")
        print("PNO extrapolated triples correlation energy:", E_corrCCT_final, "Eh")
        print("PNO extrapolated full correlation energy:", E_corrCC_final, "Eh")
        
        return E_SCF, E_corrCCSD_final, E_corrCCT_final, E_corrCC_final



    #NOTE: TODO: PC info ??
    #TODO: coords and elems vs. fragment issue
    def run(self, current_coords=None, elems=None, Grad=False, numcores=None, charge=None, mult=None):

        print(BC.OKBLUE,BC.BOLD, "------------RUNNING CC_CBS_Theory-------------", BC.END)

        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for ORCATheory.run method", BC.END)
            ashexit()


        if Grad == True:
            print(BC.FAIL,"No gradient available for CC_CBS_Theory yet! Exiting", BC.END)
            ashexit()

        #Checking that there is a basis set defined for each element provided here
        #NOTE: ORCA will use default SVP basis set if basis set not defined for element
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


        # EXTRAPOLATION TO PNO LIMIT BY 2 PNO calculations
        if self.pnosetting=="extrapolation":
            print("\nPNO Extrapolation option chosen.")
            print("Will run 2 jobs with PNO thresholds TCutPNO : 1e-{} and 1e-{} for each basis set cardinal".format(self.pnoextrapolation[0],self.pnoextrapolation[1]))
            print("="*70)
            print("Now doing Basis-1 job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[0]))
            print("="*70)
            #SINGLE F12 EXPLICIT CORRELATION JOB or if only 1 cardinal was provided
            if self.singlebasis is True:
                #Note: naming as CBS despite single-basis
                E_SCF_CBS, E_corrCCSD_CBS, E_corrCCT_CBS,E_corr_CBS = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_1, calc_label=calc_label+'cardinal1', numcores=numcores, charge=charge, mult=mult)

            #REGULAR EXTRAPOLATION WITH 2 THEORIES
            else:
                E_SCF_1, E_corrCCSD_1, E_corrCCT_1,E_corrCC_1 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_1, calc_label=calc_label+'cardinal1', numcores=numcores, charge=charge, mult=mult)
                print("="*70)
                print("Basis-1 job done. Now doing Basis-2 job: Family: {} Cardinal: {} ".format(self.basisfamily, self.cardinals[1]))
                print("="*70)
                E_SCF_2, E_corrCCSD_2, E_corrCCT_2,E_corrCC_2 = self.PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_2, calc_label=calc_label+'cardinal2', numcores=numcores, charge=charge, mult=mult)
                scf_energies = [E_SCF_1, E_SCF_2]
                ccsdcorr_energies = [E_corrCCSD_1, E_corrCCSD_2]
                triplescorr_energies = [E_corrCCT_1, E_corrCCT_2]
                corr_energies = [E_corrCC_1, E_corrCC_2]
    
            #BASIS SET EXTRAPOLATION
                E_SCF_CBS, E_corrCCSD_CBS = Extrapolation_twopoint(scf_energies, ccsdcorr_energies, self.cardinals, self.basisfamily, alpha=self.alpha, beta=self.beta) #2-point extrapolation
                #Separate CCSD and (T) CBS energies
                E_SCF_CBS, E_corrCCT_CBS = Extrapolation_twopoint(scf_energies, triplescorr_energies, self.cardinals, self.basisfamily, alpha=self.alpha, beta=self.beta) #2-point extrapolation

                #BASIS SET EXTRAPOLATION of SCF and full correlation energies
                E_SCF_CBS, E_corr_CBS = Extrapolation_twopoint(scf_energies, corr_energies, self.cardinals, self.basisfamily, alpha=self.alpha, beta=self.beta) #2-point extrapolation

        # OR no PNO extrapolation
        else:

            #SINGLE BASIS CORRELATION JOB
            if self.singlebasis is True:

                self.ccsdt_1.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
                CCSDT_1_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNO, F12=self.F12)
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
                CCSDT_1_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNO)
                shutil.copyfile(self.ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
                shutil.copyfile(self.ccsdt_1.filename+'.gbw', './' + calc_label + 'CCSDT_1' + '.gbw')
                print("CCSDT_1_dict:", CCSDT_1_dict)

                self.ccsdt_2.run(elems=elems, current_coords=current_coords, numcores=numcores, charge=charge, mult=mult)
                CCSDT_2_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_2.filename+'.out', DLPNO=self.DLPNO)
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
                E_SCF_CBS, E_corrCCSD_CBS = Extrapolation_twopoint(scf_energies, ccsdcorr_energies, self.cardinals, self.basisfamily, alpha=self.alpha, beta=self.beta) #2-point extrapolation
                #Separate CCSD and (T) CBS energies
                E_SCF_CBS, E_corrCCT_CBS = Extrapolation_twopoint(scf_energies, triplescorr_energies, self.cardinals, self.basisfamily, alpha=self.alpha, beta=self.beta) #2-point extrapolation

                #BASIS SET EXTRAPOLATION of SCF and full correlation energies
                E_SCF_CBS, E_corr_CBS = Extrapolation_twopoint(scf_energies, corr_energies, self.cardinals, self.basisfamily, alpha=self.alpha, beta=self.beta) #2-point extrapolation

        print("E_SCF_CBS:", E_SCF_CBS)
        print("E_corr_CBS:", E_corr_CBS)
        print("E_corrCCSD_CBS:", E_corrCCSD_CBS)
        print("E_corrCCT_CBS:", E_corrCCT_CBS)

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

        ############################################################
        #Spin-orbit correction for atoms.
        ############################################################
        if numatoms == 1 and self.atomicSOcorrection is True:
            print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
            if charge == 0:
                print("Charge of atom is zero. Looking up in neutral dict")
                try:
                    E_SO = dictionaries_lists.atom_spinorbitsplittings[elems[0]] / constants.hartocm
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
def PNO_extrapolation(E):
    """ PNO extrapolation by Bistoni and coworkers
    F is 1.5, good for both 5/6 and 6/7 extrapolations.
    where 5/6 and 6/7 refers to the X/Y TcutPNO threshold (10^-X and 10^-Y).
    Args:
        E ([list]): list of 
    """
    F=1.5
    E_C_PNO= E[0] + F*(E[1]-E[0])
    return E_C_PNO

def Extrapolation_twopoint(scf_energies, corr_energies, cardinals, basis_family, alpha=None, beta=None):
    """
    Extrapolation function for general 2-point extrapolations
    :param scf_energies: list of SCF energies
    :param corr_energies: list of correlation energies
    :param cardinals: list of basis-cardinal numbers
    :param basis_family: string (e.g. cc, def2, aug-cc)
    :return: extrapolated SCF energy and correlation energy
    """
    #Dictionary of extrapolation parameters. Key: Basisfamilyandcardinals Value: list: [alpha, beta]
    #Added default value of beta=3.0 (theoretical value), alpha=3.9
    extrapolation_parameters_dict = { 'cc_23' : [4.42, 2.460], 'aug-cc_23' : [4.30, 2.510], 'cc_34' : [5.46, 3.050], 'aug-cc_34' : [5.790, 3.050],
    'def2_23' : [10.390,2.4], 'def2_34' : [7.880,2.970], 'pc_23' : [7.02, 2.01], 'pc_34': [9.78, 4.09],  'ma-def2_23' : [10.390,2.4], 
    'ma-def2_34' : [7.880,2.970], 'default' : [3.9,3.0]}

    #NOTE: pc-n family uses different numbering. pc-1 is DZ(cardinal 2), pc-2 is TZ(cardinal 3), pc-4 is QZ(cardinal 4).
    if basis_family=='cc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='cc_23'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='cc-dk' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='cc_23'
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
    elif basis_family=='def2-dk' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='def2_23'
    elif basis_family=='def2' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='def2_34'
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='def2-dk' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='def2_34'
    elif basis_family=='ma-def2' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='ma-def2_23'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='ma-def2-dk' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='ma-def2_23'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    elif basis_family=='ma-def2' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='ma-def2_34'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    #Note: assuming extrapolation parameters are transferable here
    elif basis_family=='ma-def2-dk' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='ma-def2_34'
        print("Warning. ma-def2 family. Using extrapolation parameters from def2 family. UNTESTED!")
    elif basis_family=='pc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='pc_23'
    elif basis_family=='pc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='pc_34'
    else:
        print("WARNING: Unknown basis set family")
        extrap_dict_key='default'
        print("Using default settings: alpha: {} , beta: {}".format(extrapolation_parameters_dict[extrap_dict_key][0], extrapolation_parameters_dict[extrap_dict_key][1]))
        extrap_dict_key='default'
    
    #Override settings if desired
    print("\nExtrapolation parameters:")
    
    # If alpha/beta have not been set then we define based on basisfamily and cardinals
    if alpha == None and beta == None:
        alpha=extrapolation_parameters_dict[extrap_dict_key][0]
        beta=extrapolation_parameters_dict[extrap_dict_key][1]
    
    print("alpha :",alpha)
    print("beta :", beta)

    #Print energies
    print("Basis family is:", basis_family)
    print("Cardinals are:", cardinals)
    print("SCF energies are:", scf_energies[0], "and", scf_energies[1])
    print("Correlation energies are:", corr_energies[0], "and", corr_energies[1])


    eX=math.exp(-1*alpha*math.sqrt(cardinals[0]))
    eY=math.exp(-1*alpha*math.sqrt(cardinals[1]))
    SCFextrap=(scf_energies[0]*eY-scf_energies[1]*eX)/(eY-eX)
    corrextrap=(math.pow(cardinals[0],beta)*corr_energies[0] - math.pow(cardinals[1],beta) * corr_energies[1])/(math.pow(cardinals[0],beta)-math.pow(cardinals[1],beta))

    print("SCF Extrapolated value is", SCFextrap)
    print("Correlation Extrapolated value is", corrextrap)
    #print("Total Extrapolated value is", SCFextrap+corrextrap)

    return SCFextrap, corrextrap



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

    elif basisfamily == "def2-dkh":
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

    elif basisfamily == "ma-def2-dkh":
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
