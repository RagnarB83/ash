#High-level WFT workflows
import numpy as np
import os
import ash
import shutil
import constants
import math
import dictionaries_lists
import interfaces.interface_ORCA
from functions.functions_elstructure import num_core_electrons, check_cores_vs_electrons
from modules.module_highlevel_workflows import DLPNO_F12, FCI_F12, CVSR_Step
from functions.functions_general import BC, print_line_with_mainheader
from modules.module_coords import elemlisttoformula, nucchargelist,elematomnumbers


# Allowed basis set families. Accessed by function basis_for_element and extrapolation
basisfamilies=['cc','aug-cc','cc-dkh','cc-dk','aug-cc-dkh','aug-cc-dk','def2','ma-def2','def2-zora', 'def2-dkh',
            'ma-def2-zora','ma-def2-dkh', 'cc-CV', 'aug-cc-CV', 'cc-CV-dkh', 'cc-CV-dk', 'aug-cc-CV-dkh', 'aug-cc-CV-dk' ]

#NOTE: Should we use the cc-pVn(+d)Z basis sets for Na-AR ???
#Note: return basisname and ECPname (None if no ECP)
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
        exit()
    #CORRELATION CONSISTENT BASIS SETS: Non-relativistic all-electron until beyond Kr when we use cc-PP basis sets
    if basisfamily == "cc":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]

        #Special cases: cc-pV6Z only for specific light elements
        if cardinal == 6 and element not in ['H','He','Be','B','C','N','O','F','Ne','Al','Si','P','S','Cl','Ar']:
            print(BC.FAIL,"cc-pV6Z basis set only available for H-He,Be-Ne,Al-Ar. Take a look at literature.",BC.END)
            exit()

        if atomnumber <= 18 :
            return ("cc-pV{}Z".format(cardlabel), None)
        elif atomnumber == 19 :
            print("cc basis set for K is missing in ORCA. Take a look at literature.")
            exit()
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
            exit()

        if atomnumber <= 18 :
            return ("aug-cc-pV{}Z".format(cardlabel), None)
        elif atomnumber == 19 :
            print("cc basis set for K is missing in ORCA. Take a look at literature.")
            exit()
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
            exit()

        #Going through atomnumbers
        if atomnumber <= 18 :
            return ("cc-pV{}Z-DK".format(cardlabel), None)
        elif atomnumber == 19 or atomnumber == 20:
            print(BC.FAIL,"cc-dkh basis sets for K and Ca is missing in ORCA. Take a look at literature.",BC.END)
            exit()
        elif 21 <= atomnumber <= 36   : #Sc-Kr.
            return ("cc-pV{}Z".format(cardlabel), None)
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
            exit()

    elif basisfamily == "aug-cc-dkh" or basisfamily == "aug-cc-dk":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]
        # No aug-cc-pV6Z-DK basis available
        if cardinal == 6:
            print(BC.FAIL,"aug-cc-pV6Z-DK basis set not available",BC.END)
            exit()

        #Going through atomnumbers
        if atomnumber <= 18 :
            return ("aug-cc-pV{}Z-DK".format(cardlabel), None)
        elif atomnumber == 19 or atomnumber == 20:
            print(BC.FAIL,"cc-dkh basis sets for K and Ca is missing in ORCA. Take a look at literature.",BC.END)
            exit()
        elif 21 <= atomnumber <= 36   : #Sc-Kr.
            return ("aug-cc-pV{}Z".format(cardlabel), None)
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
            exit()

    #Core-valence cc-basis sets (cc-pCVnZ or cc-pwCVnZ)
    elif basisfamily == "cc-CV":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]

        #Special cases: cc-pV6Z only for specific light elements
        if cardinal == 6:
            print(BC.FAIL,"cc-pwCV6Z basis set not available.",BC.END)
            exit()

        if atomnumber <= 18 :
            return ("cc-pwCV{}Z".format(cardlabel), None)
        elif atomnumber == 19 :
            print("cc-CV basis set for K is missing in ORCA. Take a look at literature.")
            exit()
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
            exit()

        if atomnumber <= 18 :
            return ("aug-cc-pwCV{}Z".format(cardlabel), None)
        elif atomnumber == 19 :
            print("aug-cc-CV basis set for K is missing in ORCA. Take a look at literature.")
            exit()
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
            exit()

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
            exit() 
        elif 20 <= atomnumber <= 30: #Ca-Zn.
            return ("cc-pwCV{}Z-DK".format(cardlabel), None)
        elif 39 <= atomnumber <= 54 and cardinal > 2:
            return ("cc-pwCV{}Z-DK".format(cardlabel), None)      
        elif 49 <= atomnumber <= 71 :
            return ("cc-pwCV{}Z-DK3".format(cardlabel), None)
        elif 72 <= atomnumber <= 80 and cardinal == 3:
            return ("cc-pwCV{}Z-DK".format(cardlabel), None)            
        elif 81 <= atomnumber <= 86 and cardinal in [3,4]: #Tl-Rn for TZ and QZ
            return ("cc-pwCV{}Z-DK".format(cardlabel), None)



    #Core-valence cc-basis sets (cc-pCVnZ or cc-pwCVnZ) for DKH
    elif basisfamily == "aug-cc-CV-dkh" or basisfamily == "aug-cc-CV-dk":
        cardlabels={2:'D',3:'T',4:'Q',5:"5",6:"6"}
        cardlabel=cardlabels[cardinal]

        #Special cases: cc-pV6Z only for specific light elements
        if cardinal == 6:
            print(BC.FAIL,"aug-cc-pwCV6Z-DK basis set not available.",BC.END)
            exit()

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
            exit() 
        elif 20 <= atomnumber <= 30: #Ca-Zn.
            return ("aug-cc-pwCV{}Z-DK".format(cardlabel), None)
        elif 39 <= atomnumber <= 54 and cardinal > 2:
            return ("aug-cc-pwCV{}Z-DK".format(cardlabel), None)      
        elif 49 <= atomnumber <= 71 :
            return ("aug-cc-pwCV{}Z-DK3".format(cardlabel), None)
        elif 72 <= atomnumber <= 80 and cardinal == 3:
            return ("aug-cc-pwCV{}Z-DK".format(cardlabel), None)            
        elif 81 <= atomnumber <= 86 and cardinal in [3,4]: #Tl-Rn for TZ and QZ
            return ("aug-cc-pwCV{}Z-DK".format(cardlabel), None)


    elif basisfamily == "def2":
        if cardinal > 4:
            print(BC.FAIL,"def2 basis sets only available up to QZ level", BC.END)
            exit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("def2-{}".format(cardlabel), None)
        elif 36 < atomnumber < 86 :
            return ("def2-{}".format(cardlabel), "def2-ECP")

    elif basisfamily == "ma-def2":
        if cardinal > 4:
            print(BC.FAIL,"ma-def2 basis sets only available up to QZ level", BC.END)
            exit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("ma-def2-{}".format(cardlabel), None)
        elif 36 < atomnumber < 86 :
            return ("ma-def2-{}".format(cardlabel), "def2-ECP")

    #NOTE: Problem SARC QZ or DZ basis set not really available so extrapolations for heavy elements not really possible
    elif basisfamily == "def2-zora":
        if cardinal > 4:
            print(BC.FAIL,"def2-ZORA basis sets only available up to QZ level", BC.END)
            exit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("ZORA-def2-{}".format(cardlabel), None)
        elif 36 < atomnumber < 86 and cardinal == 3:
            #NOTE: Problem SARC QZ basis set not really available
            return ("SARC-ZORA-{}".format(cardlabel), None)

    elif basisfamily == "def2-dkh":
        if cardinal > 4:
            print(BC.FAIL,"def2-DKH basis sets only available up to QZ level", BC.END)
            exit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("DKH-def2-{}".format(cardlabel), None)
        elif 36 < atomnumber < 86 and cardinal == 3:
            
            return ("SARC-DKH-{}".format(cardlabel), None)
    #NOTE: Problem SARC QZ or DZ basis set not really available so extrapolations for heavy elements not really possible
    elif basisfamily == "ma-def2-zora":
        if cardinal > 4:
            print(BC.FAIL,"ma-def2-ZORA basis sets only available up to QZ level", BC.END)
            exit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("ma-ZORA-def2-{}".format(cardlabel), None)

    elif basisfamily == "ma-def2-dkh":
        if cardinal > 4:
            print(BC.FAIL,"ma-def2-DKH basis sets only available up to QZ level", BC.END)
            exit()
        cardlabels={2:'SVP',3:'TZVPP',4:'QZVPP'}
        cardlabel=cardlabels[cardinal]
        if atomnumber <= 36 :
            return ("ma-DKH-def2-{}".format(cardlabel), None)

    print(BC.FAIL,"There is probably no {} {}Z basis set available for element {} in ORCA. Exiting.".format(basisfamily, cardinal, element), BC.END)
    exit()





#Flexible CCSD(T)/CBS protocol class. Simple. No core-correlation, scalar relativistic or spin-orbit coupling for now.
# Regular CC, DLPNO-CC, DLPNO-CC with PNO extrapolation etc.
#alpha and beta can be manually set. If not set then they are picked based on basisfamily
#NOTE: List of elements are required here
class CC_CBS_Theory:
    def __init__(self, elements=None, cardinals = [2,3], basisfamily="def2", relativity=None, charge=None, orcadir=None, mult=None, 
           stabilityanalysis=False, numcores=1, CVSR=False, CVbasis="W1-mtsmall", F12=False, DFTreference=None, DFT_RI=False,
                        DLPNO=False, memory=5000, pnosetting='NormalPNO', pnoextrapolation=[5,6], T1=True, scfsetting='TightSCF',
                        alpha=None, beta=None, extrainputkeyword='', extrablocks=''):
        """
        WORK IN PROGRESS
        CCSD(T)/CBS frozencore workflow

        :param elements: list of element symbols
        :param charge: Charge of fragment (to be replaced)?
        :param orcadir: ORCA directory
        :param mult: Multiplicity of fragment (to be replaced)?
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

        if elements == None:
            print(BC.FAIL, "\nCC_CBS_Theory requires a list of elements to be given in order to set up basis set", BC.END)
            print("Example: CC_CBS_Theory(elements=['C','Fe','S','H','Mo'], basisfamily='def2',cardinals=[2,3], ...")
            print("Should be a list containing all elements that a fragment might contain")
            exit()
        else:
            #Removing redundant symbols (in case fragment.elems list was passed for example)
            elements = set(elements)

        if charge == None or mult == None:
            print(BC.FAIL,"Charge and mult keywords are required", BC.END)
            exit()


        #ORCA
        self.orcadir = orcadir

        self.elements=elements
        self.cardinals = cardinals
        self.basisfamily = basisfamily
        self.relativity = relativity
        self.charge = charge
        self.mult = mult
        self.stabilityanalysis=stabilityanalysis
        self.numcores=numcores
        self.CVSR=CVSR
        self.CVbasis=CVbasis
        self.F12=FCI_F12
        self.DFTreference=DFTreference
        self.DFT_RI=DFT_RI
        self.DLPNOflag=DLPNO
        self.memory=memory
        self.pnosetting=pnosetting
        self.pnoextrapolation=pnoextrapolation
        self.T1=T1
        self.scfsetting=scfsetting
        self.alpha=alpha
        self.beta=beta
        self.extrainputkeyword=extrainputkeyword
        self.extrablocks=extrablocks


        print("-----------------------------")
        print("CC_CBS PROTOCOL")
        print("-----------------------------")
        print("Settings:")
        print("Cardinals chosen:", self.cardinals)
        print("Basis set family chosen:", self.basisfamily)
        print("Elements involved:", self.elements)
        print("Number of cores: ", self.numcores)
        print("Maxcore setting: ", self.memory, "MB")
        print("")
        print("DLPNOflag:", self.DLPNOflag)
        dlpno_line=""
        if self.DLPNOflag == True:
            print("PNO setting: ", self.pnosetting)
            if self.pnosetting == "extrapolation":
                print("pnoextrapolation:", self.pnoextrapolation)
            #Setting Full LMP2 to false in general for DLPNO
            dlpno_line="UseFullLMP2Guess false"
            print("T1 : ", self.T1)
        print("SCF setting: ", self.scfsetting)
        print("Relativity: ", self.relativity)
        print("Stability analysis:", self.stabilityanalysis)
        print("Core-Valence Scalar Relativistic correction (CVSR): ", self.CVSR)
        print("")



        #Block input for SCF/MDCI block options.
        #Disabling FullLMP2 guess in general as not available for open-shell
        #Adding memory and extrablocks.
        blocks="""%maxcore {}
%scf
maxiter 1200
end
%mdci
{}
maxiter 150
end
{}""".format(memory,dlpno_line,extrablocks)
        if self.stabilityanalysis is True:
            blocks = blocks + "%scf stabperform true end"
            #Adding 1-center approximation
            if self.relativity != None:
                print("Stability analysis and relativity requires 1-center approximation")
                print("Turning on")
                blocks = blocks + "\n%rel onecenter true end"

        #Choosing whether DLPNO or not
        if self.DLPNOflag == True:
            #Iterative DLPNO triples or not
            if self.T1 is True:
                ccsdtkeyword='DLPNO-CCSD(T1)'
            else:
                #DLPNO-F12 or not
                if self.F12 is True:
                    print("Note: not supported yet ")
                    exit()
                    ccsdtkeyword='DLPNO-CCSD(T)-F12'
                else:
                    ccsdtkeyword='DLPNO-CCSD(T)'
            #Add PNO keyword in simpleinputline or not (if extrapolation)
            if self.pnosetting != "extrapolation":
                pnokeyword=self.pnosetting
            else:
                pnokeyword=""
        #Regular CCSD(T)
        else:
            #No PNO keywords
            pnokeyword=""
            self.pnosetting=None
            #F12 or not
            if self.F12 is True:
                ccsdtkeyword='CCSD(T)-F12'
                print("Note: not supported yet ")
                exit()
            else:
                ccsdtkeyword='CCSD(T)'


        #SCALAR RELATIVITY HAMILTONIAN AND SELECT CORRELATED AUX BASIS
        if self.relativity == None:
            if self.basisfamily in ['cc-dkh', 'aug-cc-dkh', 'cc-dk', 'aug-cc-dk', 'def2-zora', 'def2-dkh', 
            'ma-def2-zora','ma-def2-dkh', 'cc-CV-dk', 'cc-CV-dkh', 'aug-cc-CV-dk', 'aug-cc-CV-dkh']:
                print("Relativity option isNone but a relativistic basis set family chosen:", self.basisfamily)
                print("You probably want relativity keyword argument set to DKH or ZORA (relativity=\"NoRel\" option possible also but not recommended)")
                exit()

            self.extrainputkeyword = self.extrainputkeyword + '  '
            #Auxiliary basis set. 1 big one for now
            #TODO: look more into
            if 'def2' in self.basisfamily:
                auxbasis='AutoAux'
            else:
                if 'aug' in self.basisfamily:
                    #auxbasis='aug-cc-pV5Z/C'
                    auxbasis='AutoAux'               
                else:
                    #auxbasis='cc-pV5Z/C'
                    auxbasis='AutoAux'
        elif self.relativity == "NoRel":
            if 'def2' in self.basisfamily:
                auxbasis='AutoAux'
            else:
                if 'aug' in self.basisfamily:
                    auxbasis='AutoAux'                
                else:
                    auxbasis='AutoAux'
        elif self.relativity == 'DKH':
            self.extrainputkeyword = self.extrainputkeyword + ' DKH '
            if 'def2' in self.basisfamily:
                auxbasis='AutoAux'
            else:
                if 'aug' in self.basisfamily:
                    auxbasis='AutoAux'                
                else:
                    auxbasis='AutoAux'
        elif self.relativity == 'ZORA':
            self.extrainputkeyword = self.extrainputkeyword + ' ZORA '
            auxbasis='AutoAux'
            if 'def2' in self.basisfamily:
                auxbasis='AutoAux'
            else:
                if 'aug' in self.basisfamily:
                    auxbasis='AutoAux'                
                else:
                    auxbasis='AutoAux'
        elif self.relativity == 'X2C':
            self.extrainputkeyword = self.extrainputkeyword + ' X2C '
            auxbasis='AutoAux'
            print("Not ready")
            exit()

        #Possible DFT reference (functional name)
        #NOTE: Hardcoding RIJCOSX SARC/J defgrid3 for now
        if self.DFTreference != None:
            if self.DFT_RI is True:
                self.extrainputkeyword = self.extrainputkeyword + ' {} RIJCOSX SARC/J defgrid3 '.format(self.DFTreference)
            else:
                self.extrainputkeyword = self.extrainputkeyword + ' {} NORI defgrid3 '.format(self.DFTreference)

        ############################################################s
        #Frozen-core CCSD(T) calculations defined here
        ############################################################

        #Getting basis sets and ECPs for each element for a given basis-family and cardinal
        Calc1_basis_dict={}
        for elem in elements:
            bas=basis_for_element(elem, basisfamily, cardinals[0])
            Calc1_basis_dict[elem] = bas
        Calc2_basis_dict={}
        for elem in elements:
            bas=basis_for_element(elem, basisfamily, cardinals[1])
            Calc2_basis_dict[elem] = bas

        print("Calc1_basis_dict:", Calc1_basis_dict)
        print("Calc2_basis_dict", Calc2_basis_dict)

        #ccsdt_1_line, ccsdt_2_line = choose_inputlines_from_basisfamily(self.cardinals,self.basisfamily,ccsdtkeyword,auxbasis,pnokeyword,self.scfsetting,self.extrainputkeyword)
        ccsdt_1_line="! {}  {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {}  {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)

        #Adding basis set info for each element into blocks
        basis1_block="%basis\n"
        for el,bas_ecp in Calc1_basis_dict.items():
            basis1_block=basis1_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
            if bas_ecp[1] != None:
                basis1_block=basis1_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])
        basis1_block=basis1_block+"end"

        basis2_block="%basis\n"
        for el,bas_ecp in Calc2_basis_dict.items():
            basis2_block=basis2_block+"newgto {} \"{}\" end\n".format(el,bas_ecp[0])
            if bas_ecp[1] != None:
                basis2_block=basis2_block+"newecp {} \"{}\" end\n".format(el,bas_ecp[1])
        basis2_block=basis2_block+"end"
        print("basis1_block:", basis1_block)
        print("basis2_block:", basis2_block)

        #Final blocks input
        blocks1= blocks +basis1_block
        blocks2= blocks +basis2_block
        
        #Defining two theory objects for each basis set
        self.ccsdt_1 = interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_1_line, orcablocks=blocks1, numcores=self.numcores, charge=self.charge, mult=self.mult)
        self.ccsdt_2 = interfaces.interface_ORCA.ORCATheory(orcadir=self.orcadir, orcasimpleinput=ccsdt_2_line, orcablocks=blocks2, numcores=self.numcores, charge=self.charge, mult=self.mult)


    #NOTE: TODO: PC info ??
    #TODO: coords and elems vs. fragment issue
    def run(self, current_coords=None, elems=None, Grad=False, numcores=None):

        numatoms=len(elems)
        #Creating label here based on element and charge input
        formula=elemlisttoformula(elems)
        calc_label = "Frag" + str(formula) + "_" + str(self.charge) + "_"
        print("Calculation label: ", calc_label)
        
        #Calculate number of electrons
        numelectrons = int(nucchargelist(elems) - self.charge)
        #Reduce numcores if required
        numcores = check_cores_vs_electrons(elems,self.numcores,self.charge)

        #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
        #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
        if numelectrons == 1:
            print("Number of electrons is 1")
            print("Assuming hydrogen atom and skipping calculation")
            E_total = -0.500000
            print("Using hardcoded value: ", E_total)
            E_dict = {'Total_E': E_total, 'E_SCF_CBS': E_total, 'E_CCSDcorr_CBS': 0.0,
                    'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
            return E_total, E_dict
        
        # EXTRAPOLATION TO PNO LIMIT BY 2 PNO calculations
        if self.pnosetting=="extrapolation":
            print("PNO Extrapolation option chosen.")
            print("Will run 2 jobs with PNO thresholds TCutPNO : 1e-{} and 1e-{}".format(pnoextrapolation[0],pnoextrapolation[1]))
            E_SCF_1, E_corrCCSD_1, E_corrCCT_1,E_corrCC_1 = PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_1, pnoextrapolation=self.pnoextrapolation, DLPNO=self.DLPNO, F12=False, calc_label=calc_label)
            E_SCF_2, E_corrCCSD_2, E_corrCCT_2,E_corrCC_2 = PNOExtrapolationStep(elems=elems, current_coords=current_coords, theory=self.ccsdt_2, pnoextrapolation=self.pnoextrapolation, DLPNO=self.DLPNO, F12=False, calc_label=calc_label)
            scf_energies = [E_SCF_1, E_SCF_2]
            ccsdcorr_energies = [E_corrCCSD_1, E_corrCCSD_2]
            triplescorr_energies = [E_corrCCT_1, E_corrCCT_2]
            corr_energies = [E_corrCC_1, E_corrCC_2]
        # OR REGULAR
        else:
            #Running both theories
            #ash.Singlepoint(fragment=fragment, theory=self.ccsdt_1)
            self.ccsdt_1.run(elems=elems, current_coords=current_coords)
            CCSDT_1_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_1.filename+'.out', DLPNO=self.DLPNOflag)
            shutil.copyfile(self.ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
            print("CCSDT_1_dict:", CCSDT_1_dict)

            #ash.Singlepoint(fragment=fragment, theory=self.ccsdt_2)
            self.ccsdt_2.run(elems=elems, current_coords=current_coords)
            CCSDT_2_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(self.ccsdt_2.filename+'.out', DLPNO=self.DLPNOflag)
            shutil.copyfile(self.ccsdt_2.filename+'.out', './' + calc_label + 'CCSDT_2' + '.out')
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

        E_SCF_CBS, E_corr_CBS = Extrapolation_twopoint(scf_energies, corr_energies, self.cardinals, self.basisfamily, alpha=self.alpha, beta=self.beta) #2-point extrapolation

        print("E_SCF_CBS:", E_SCF_CBS)
        print("E_corr_CBS:", E_corr_CBS)

        ############################################################
        #Core-correlation + scalar relativistic as joint correction
        ############################################################
        if self.CVSR is True:
            print("")
            print("Core-Valence Scalar Relativistic Correction is on!")
            exit()
            #TODO: We should only do CV if we are doing all-electron calculations. If we have heavy element then we have probably added an ECP (specialbasisfunction)
            # Switch to doing only CV correction in that case ?
            # TODO: Option if W1-mtsmall basis set is not available?
            
            if self.ECPflag is True:
                print("ECPs present. Not doing ScalarRelativistic Correction. Switching to Core-Valence Correction only.")
                reloption=" "
                pnooption="NormalPNO"
                print("Doing CVSR_Step with No Scalar Relativity and CV-basis: {} and PNO-option: {}".format(CVbasis,pnooption))
                E_corecorr_and_SR = CV_Step(cvbasis,reloption,ccsdtkeyword,auxbasis,pnooption,scfsetting,extrainputkeyword,orcadir,blocks,numcores,charge,mult,fragment,calc_label)
            else:
                reloption="DKH"
                pnooption="NormalPNO"
                print("Doing CVSR_Step with Relativistic Option: {} and CV-basis: {} and PNO-option: {}".format(reloption,CVbasis,pnooption))
                E_corecorr_and_SR = CVSR_Step(cvbasis,reloption,ccsdtkeyword,auxbasis,pnooption,scfsetting,extrainputkeyword,orcadir,blocks,numcores,charge,mult,fragment,calc_label)
                
            
        else:
            print("")
            print("Core-Valence Scalar Relativistic Correction is off!")
            E_corecorr_and_SR=0.0

        ############################################################
        #Spin-orbit correction for atoms.
        ############################################################
        if numatoms == 1:
            print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
            if self.charge == 0:
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
        #Combining E_SCF_CBS, E_corr_CBS + SO + CV+SR
        print("")
        print("")
        E_FINAL = E_SCF_CBS + E_corr_CBS + E_SO+E_corecorr_and_SR
        print("Final CCSD(T)/CBS energy :", E_FINAL, "Eh")
        print("")
        print("Contributions:")
        print("--------------")
        print("E_SCF_CBS : ", E_SCF_CBS)
        print("E_corr_CBS : ", E_corr_CBS)
        print("Spin-orbit coupling : ", E_SO, "Eh")
        print("E_corecorr_and_SR : ", E_corecorr_and_SR, "Eh")
        E_dict = {'Total_E' : E_FINAL, 'E_SCF_CBS' : E_SCF_CBS, 'E_corr_CBS' : E_corr_CBS, 'E_SO' : E_SO, 'E_corecorr_and_SR' : E_corecorr_and_SR}


        #Cleanup GBW file. Full cleanup ??
        # TODO: Keep output files for each step
        os.remove(self.ccsdt_1.filename+'.gbw')

        #return final energy and also dictionary with energy components
        return E_FINAL, E_dict

    
    
################################
# BASIS SET and ECP CHOICES
################################


def choose_inputlines_from_basisfamily(cardinals,basisfamily,ccsdtkeyword,auxbasis,pnokeyword,scfsetting,extrainputkeyword):

    if cardinals == [2,3] and basisfamily=="def2":
        #Auxiliary basis set.
        auxbasis='def2-QZVPP/C'
        ccsdt_1_line="! {} def2-SVP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [2,3] and basisfamily=="def2-dk":
        #Auxiliary basis set.
        auxbasis='def2-QZVPP/C'
        ccsdt_1_line="! {} DKH-def2-SVP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} DKH-def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [3,4] and basisfamily=="def2-dk":
        #Auxiliary basis set.
        auxbasis='def2-QZVPP/C'
        ccsdt_1_line="! {} DKH-def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} DKH-def2-QZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [2,3] and basisfamily=="ma-def2-dk":
        #Auxiliary basis set.
        auxbasis='def2-QZVPP/C'
        ccsdt_1_line="! {} ma-DKH-def2-SVP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} ma-DKH-def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [3,4] and basisfamily=="ma-def2-dk":
        #Auxiliary basis set.
        auxbasis='def2-QZVPP/C'
        ccsdt_1_line="! {} ma-DKH-def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} ma-DKH-def2-QZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [2,3] and basisfamily=="def2-zora":
        #Auxiliary basis set.
        auxbasis='def2-QZVPP/C'
        ccsdt_1_line="! {} ZORA-def2-SVP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} ZORA-def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [3,4] and basisfamily=="def2-zora":
        #Auxiliary basis set.
        auxbasis='def2-QZVPP/C'
        ccsdt_1_line="! {} ZORA-def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} ZORA-def2-QZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [3,4] and basisfamily=="def2":
        #Auxiliary basis set.
        auxbasis='def2-QZVPP/C'
        ccsdt_1_line="! {} def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} def2-QZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)  
    elif cardinals == [2,3] and basisfamily=="ma-def2":
        #Auxiliary basis set.
        auxbasis='aug-cc-pVQZ/C'
        ccsdt_1_line="! {} ma-def2-SVP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} ma-def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [3,4] and basisfamily=="ma-def2":
        #Auxiliary basis set.
        auxbasis='aug-cc-pVQZ/C'
        ccsdt_1_line="! {} ma-def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} ma-def2-QZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [2,3] and basisfamily=="cc":
        #Auxiliary basis set.
        auxbasis='cc-pVQZ/C'
        ccsdt_1_line="! {} cc-pVDZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} cc-pVTZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [3,4] and basisfamily=="cc":
        #Auxiliary basis set.
        auxbasis='cc-pV5Z/C'
        ccsdt_1_line="! {} cc-pVTZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} cc-pVQZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [4,5] and basisfamily=="cc":
        #Auxiliary basis set.
        auxbasis='Autoaux'
        ccsdt_1_line="! {} cc-pVQZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} cc-pV5Z {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [2,3] and basisfamily=="aug-cc":
        #Auxiliary basis set.
        auxbasis='aug-cc-pVQZ/C'
        ccsdt_1_line="! {} aug-cc-pVDZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pVTZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [3,4] and basisfamily=="aug-cc":
        #Auxiliary basis set.
        auxbasis='aug-cc-pV5Z/C'
        ccsdt_1_line="! {} aug-cc-pVTZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pVQZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [4,5] and basisfamily=="aug-cc":
        #Auxiliary basis set.
        auxbasis='Autoaux'
        ccsdt_1_line="! {} aug-cc-pVQZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pV5Z {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        #TODO Note: 4/5 cc/aug-cc basis sets are available but we need extrapolation parameters

    #DKH correlation consistent basis sets
    elif cardinals == [2,3] and basisfamily=="cc-dk":
        #Auxiliary basis set.
        auxbasis='cc-pVQZ/C'
        ccsdt_1_line="! {} cc-pVDZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} cc-pVTZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [3,4] and basisfamily=="cc-dk":
        #Auxiliary basis set.
        auxbasis='autoaux'
        ccsdt_1_line="! {} cc-pVTZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} cc-pVQZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [4,5] and basisfamily=="cc-dk":
        #Auxiliary basis set.
        auxbasis='autoaux'
        ccsdt_1_line="! {} cc-pVQZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} cc-pV5Z-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [2,3] and basisfamily=="aug-cc-dk":
        #Auxiliary basis set.
        auxbasis='aug-cc-pVQZ/C'
        ccsdt_1_line="! {} aug-cc-pVDZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pVTZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [3,4] and basisfamily=="aug-cc-dk":
        #Auxiliary basis set.
        auxbasis='Autoaux'
        ccsdt_1_line="! {} aug-cc-pVTZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pVQZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [4,5] and basisfamily=="aug-cc-dk":
        #Auxiliary basis set.
        auxbasis='Autoaux'
        ccsdt_1_line="! {} aug-cc-pVQZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pV5Z-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        #TODO Note: 4/5 cc/aug-cc basis sets are available but we need extrapolation parameters
    
    #DKH CORE-VALENCE CORRELATION CONSISTENT BASIS SETS
    elif cardinals == [2,3] and basisfamily=="cc-pw-dk":
        #Auxiliary basis set.
        auxbasis='cc-pVQZ/C'
        ccsdt_1_line="! {} cc-pwCVDZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} cc-pwCVTZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [3,4] and basisfamily=="cc-pw-dk":
        #Auxiliary basis set.
        auxbasis='cc-pVQZ/C'
        ccsdt_1_line="! {} cc-pwCVTZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} cc-pwCVQZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [4,5] and basisfamily=="cc-pw-dk":
        #Auxiliary basis set.
        auxbasis='Autoaux'
        ccsdt_1_line="! {} cc-pwCVQZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} cc-pwCV5Z-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [2,3] and basisfamily=="aug-cc-pw-dk":
        #Auxiliary basis set.
        auxbasis='cc-pVQZ/C'
        ccsdt_1_line="! {} aug-cc-pwCVDZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pwCVTZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [3,4] and basisfamily=="aug-cc-pw-dk":
        #Auxiliary basis set.
        auxbasis='cc-pVQZ/C'
        ccsdt_1_line="! {} aug-cc-pwCVTZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pwCVQZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [4,5] and basisfamily=="aug-cc-pw-dk":
        #Auxiliary basis set.
        auxbasis='Autoaux'
        ccsdt_1_line="! {} aug-cc-pwCVQZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pwCV5Z-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    else:
        print("Unknown basisfamily or cardinals chosen...")
        exit()
    return ccsdt_1_line,ccsdt_2_line



#If heavy element present and using cc/aug-cc basisfamily then add special PP-basis and ECP in block
def special_element_basis(elems,cardinal,basisfamily,blocks):
    basis_dict = {('cc',2) : "cc-pVDZ-PP", ('aug-cc',2) : "aug-cc-pVDZ-PP", ('cc',3) : "cc-pVTZ-PP", ('aug-cc',3) : "aug-cc-pVTZ-PP", ('cc',4) : "cc-pVQZ-PP", ('aug-cc',4) : "aug-cc-pVQZ-PP"}
    auxbasis_dict = {('cc',2) : "cc-pVDZ-PP/C", ('aug-cc',2) : "aug-cc-pVDZ-PP/C", ('cc',3) : "cc-pVTZ-PP/C", ('aug-cc',3) : "aug-cc-pVTZ-PP/C", ('cc',4) : "cc-pVQZ-PP/C", ('aug-cc',4) : "aug-cc-pVQZ-PP/C"}
    for element in elems:
        if element in ['Rb', 'Sr','Y','Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh','Pd','Ag','Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba',
                        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn']:
            if 'cc' in basisfamily:
                specialbasis = basis_dict[(basisfamily,cardinal)]
                specialauxbasis = auxbasis_dict[(basisfamily,cardinal)]
                blocks = blocks + "\n%basis\n newgto {} \"{}\" end\n newecp {} \"SK-MCDHF-RSC\" end\nnewauxCGTO {} \"{}\" end \nend\n".format(element,specialbasis,element, element, specialauxbasis)
    return blocks

#Check if ECP-option was added by special_element_basis
def isECP(blocks):
    if 'newecp' in blocks:
        print("ECP information was added for heavy element")
        return True
    else:
        print("No ECP information was added")
        return False




#############################
# SPECIAL CALCULATION JOBS
#############################

# For theory object with DLPNO, do 2 calculations with different DLPNO thresholds and extrapolate
def PNOExtrapolationStep(elems=None, current_coords=None, theory=None, pnoextrapolation=None, DLPNO=None, F12=None, calc_label=None):

    print("Inside PNOExtrapolationStep")
    PNO_X=pnoextrapolation[0]
    PNO_Y=pnoextrapolation[1]
    
    #Adding TCutPNO option X
    #TightPNO options for other thresholds
    mdciblockX="""
    %mdci
    TCutPNO 1e-{}
    TCutPairs 1e-5
    TCutDO 5e-3
    TCutMKN 1e-3
    end
    
    """.format(PNO_X)
    #TCutPNO option Y
    #TightPNO options for other thresholds
    mdciblockY="""
    %mdci
    TCutPNO 1e-{}
    TCutPairs 1e-5
    TCutDO 5e-3
    TCutMKN 1e-3
    end
    
    """.format(PNO_Y)
    #Add mdciblock to blocks of theory
    PNOXblocks = theory.orcablocks + mdciblockX
    PNOYblocks = theory.orcablocks + mdciblockY
    
    theory.orcablocks = PNOXblocks
    
    #ash.Singlepoint(fragment=fragment, theory=theory)
    theory.run(elems=elems, current_coords=current_coords)
    resultdict_X = interfaces.interface_ORCA.grab_HF_and_corr_energies(theory.filename+'.out', DLPNO=DLPNO,F12=F12)
    shutil.copyfile(theory.filename+'.out', './' + calc_label + '_PNOX' + '.out')
    print("resultdict_X:", resultdict_X)


    
    theory.orcablocks = PNOYblocks
    #ash.Singlepoint(fragment=fragment, theory=theory)
    theory.run(elems=elems, current_coords=current_coords)
    resultdict_Y = interfaces.interface_ORCA.grab_HF_and_corr_energies(theory.filename+'.out', DLPNO=DLPNO,F12=F12)
    shutil.copyfile(theory.filename+'.out', './' + calc_label + '_PNOY' + '.out')
    print("resultdict_Y:", resultdict_Y)
    
    #Extrapolation to PNO limit

    E_SCF = resultdict_Y['HF']
    #Extrapolation CCSD part and (T) separately
    # TODO: Is this correct??
    E_corrCCSD_final = PNO_extrapolation([resultdict_X['CCSD_corr'],resultdict_Y['CCSD_corr']])
    E_corrCCT_final = PNO_extrapolation([resultdict_X['CCSD(T)_corr'],resultdict_Y['CCSD(T)_corr']])
    #Extrapolation of full correlation energy
    E_corrCC_final = PNO_extrapolation([resultdict_X['full_corr'],resultdict_Y['full_corr']])

    print("PNO extrapolated CCSD correlation energy:", E_corrCCSD_final, "Eh")
    print("PNO extrapolated triples correlation energy:", E_corrCCT_final, "Eh")
    print("PNO extrapolated full correlation energy:", E_corrCC_final, "Eh")
    
    return E_SCF, E_corrCCSD_final, E_corrCCT_final, E_corrCC_final

#Core-Valence ScalarRelativistic Step
def CVSR_Step(cvbasis,reloption,ccsdtkeyword,auxbasis,pnooption,scfsetting,extrainputkeyword,orcadir,blocks,numcores,charge,mult,fragment,calc_label):
    
    ccsdt_mtsmall_NoFC_line="! {} {} {}   nofrozencore {} {} {} {}".format(ccsdtkeyword,reloption,cvbasis,auxbasis,pnooption,scfsetting,extrainputkeyword)
    ccsdt_mtsmall_FC_line="! {} {}  {} {} {} {}".format(ccsdtkeyword,cvbasis,auxbasis,pnooption,scfsetting,extrainputkeyword)

    ccsdt_mtsmall_NoFC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)
    return E_corecorr_and_SR



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
    print("Extrapolation parameters:")
    
    # If alpha/beta have not been set then we define based on basisfamily and cardinals
    if alpha == None and beta == None:
        alpha=extrapolation_parameters_dict[extrap_dict_key][0]
        beta=extrapolation_parameters_dict[extrap_dict_key][1]
    
    print("alpha :",alpha)
    print("beta :", beta)

    #Print energies
    print("Basis family is:", basis_family)
    print("SCF energies are:", scf_energies[0], "and", scf_energies[1])
    print("Correlation energies are:", corr_energies[0], "and", corr_energies[1])


    eX=math.exp(-1*alpha*math.sqrt(cardinals[0]))
    eY=math.exp(-1*alpha*math.sqrt(cardinals[1]))
    SCFextrap=(scf_energies[0]*eY-scf_energies[1]*eX)/(eY-eX)
    corrextrap=(math.pow(cardinals[0],beta)*corr_energies[0] - math.pow(cardinals[1],beta) * corr_energies[1])/(math.pow(cardinals[0],beta)-math.pow(cardinals[1],beta))

    print("SCF Extrapolated value is", SCFextrap)
    print("Correlation Extrapolated value is", corrextrap)
    print("Total Extrapolated value is", SCFextrap+corrextrap)

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
    exit()
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










#Class Elements_basis_sets
#class Elements_basis_sets:
#    def __init__(self, elements):

#elements_basis_sets = {1:basis_sets}

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


#Function that gives basis-set name for specific element for basisfamily and cardinal
#Example : basis_for_element('Mo', "cc", 2) gives cc-pVDZ-PP
#Example : basis_for_element('C', "cc", 4) gives cc-pVQZ





# Takes basisfamily-name input, cardinals and elements and gives
# get_orcablockinput
# class BasisFamily:
#     def __init__(self, basisfamily, cardinals, elements):
        
#         self.basisfamily=basisfamily
#         self.elements=elements
#         self.cardinals=cardinals

#         self.cardinal1_basis_dict=None
#         self.cardinal2_basis_dict=None
#         self.cardinal1_ecp_dict=None
#         self.cardinal2_ecp_dict=None
#         if len(cardinals) == 3:
#             print("3 cardinals are not supported yet")
#             exit()

#         # Regular cc-pVnZ basis and cc-pVnZ-PP basis sets by Dunning.
#         # All-electron for elements H-Ar, Ca-Kr. ECP-based for heavier than XXX
#         #NOTE: Need to look at whether all-electron basis or ECP-basis s recommended for Ca or not ??
#         #NOTE: Ag, AU ????
#         if self.basisfamily == "cc":
#             if cardinals == [2,3]:

#                 #General basis names for H-Ar, Ca-Kr
#                 cardinal1_basis_dict['G'] = "cc-pVDZ"
#                 cardinal2_basis_dict['G'] = "cc-pVTZ"
#                 #Special rules for special elements

#                 #TODO: Transition metals etc
#                 if 'Mo' in elements:
#                     self.cardinal1_basis_dict['Mo'] = "cc-pVDZ-PP"
#                     self.cardinal1_ecp_dict['Mo'] = "SK-MCDHF-RSC"
#                     self.cardinal2_basis_dict['Mo'] = "cc-pVTZ-PP"
#                     self.cardinal2_ecp_dict['Mo'] = "SK-MCDHF-RSC"

#             elif cardinals == [3,4]:

#             elif cardinals == [4,5]:

#             elif cardinals == [5,6]:
#         elif self.basisfamily == "cc-dkh":

#         elif self.basisfamily == "cc-zora":

#         elif self.basisfamily == "aug-cc-dkh":

#         elif self.basisfamily == "aug-cc-zora":

#         elif self.basisfamily == "aug-cc":


#         #Core-valence cc-basis sets (cc-pCVnZ or cc-pwCVnZ)
#         elif self.basisfamily == "cc-CV":


#         #Core-valence cc-basis sets (aug-cc-pCVnZ or aug-cc-pwCVnZ)
#         elif self.basisfamily == "aug-cc-CV":


#         elif self.basisfamily == "def2":

#         elif self.basisfamily == "ma-def2":

#    def get_orcablockinput(self):
#
#        string="%basis\n"
#        for el in self.elements:
#            string.append("newgto {} \"{}\" end".format(el,bas))
 #       bas="""
 #       %basis
  #      newgto 
   #     end
   #     """
#
#        elems=['Fe','S']
#
#        ccpvdz_class  = BasisFamily("cc", [2,3], elems)
#
#        ccpvdz_class