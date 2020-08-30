import numpy as np
import functions_coords
import os
import glob
import ash
import subprocess as sp
import shutil
import constants
import math
from functions_ORCA import grab_HF_and_corr_energies
from interface_geometric import *
from interface_crest import *

#Various workflows and associated sub-functions



#Spin-orbit splittings:
#Currently only including neutral atoms. Data in cm-1 from : https://webhome.weizmann.ac.il/home/comartin/w1/so.txt
atom_spinorbitsplittings = {'H': 0.000, 'B': -10.17, 'C' : -29.58, 'N' : 0.00, 'O' : -77.97, 'F' : -134.70,
                      'Al' : -74.71, 'Si' : -149.68, 'P' : 0.00, 'S' : -195.77, 'Cl' : -294.12}

#Core electrons for elements in ORCA
atom_core_electrons = {'H': 0, 'He' : 0, 'Li' : 0, 'Be' : 0, 'B': 2, 'C' : 2, 'N' : 2, 'O' : 2, 'F' : 2, 'Ne' : 2,
                      'Na' : 2, 'Mg' : 2, 'Al' : 10, 'Si' : 10, 'P' : 10, 'S' : 10, 'Cl' : 10, 'Ar' : 10,
                       'K' : 10, 'Ca' : 10, 'Sc' : 10, 'Ti' : 10, 'V' : 10, 'Cr' : 10, 'Mn' : 10, 'Fe' : 10, 'Co' : 10,
                       'Ni' : 10, 'Cu' : 10, 'Zn' : 10, 'Ga' : 18, 'Ge' : 18, 'As' : 18, 'Se' : 18, 'Br' : 18, 'Kr' : 18,
                       'Rb' : 18, 'Sr' : 18, 'Y' : 28, 'Zr' : 28, 'Nb' : 28, 'Mo' : 28, 'Tc' : 28, 'Ru' : 28, 'Rh' : 28,
                       'Pd' : 28, 'Ag' : 28, 'Cd' : 28, 'In' : 36, 'Sn' : 36, 'Sb' : 36, 'Te' : 36, 'I' : 36, 'Xe' : 36,
                       'Cs' : 36, 'Ba' : 36, 'Lu' : 46, 'Hf' : 46, 'Ta' : 46, 'w' : 46, 'Re' : 46, 'Os' : 46, 'Ir' : 46,
                       'Pt' : 46, 'Au' : 46, 'Hg' : 46, 'Tl' : 68, 'Pb' : 68, 'Bi' : 68, 'Po' : 68, 'At' : 68, 'Rn' : 68}

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

def Extrapolation_twopoint(scf_energies, corr_energies, cardinals, basis_family):
    """
    Extrapolation function for general 2-point extrapolations
    :param scf_energies: list of SCF energies
    :param corr_energies: list of correlation energies
    :param cardinals: list of basis-cardinal numbers
    :param basis_family: string (e.g. cc, def2, aug-cc)
    :return: extrapolated SCF energy and correlation energy
    """
    #Dictionary of extrapolation parameters. Key: Basisfamilyandcardinals Value: list: [alpha, beta]
    extrapolation_parameters_dict = { 'cc_23' : [4.42, 2.460], 'aug-cc_23' : [4.30, 2.510], 'cc_34' : [5.46, 3.050], 'aug-cc_34' : [5.790, 3.050],
    'def2_23' : [10.390,2.4], 'def2_34' : [7.880,2.970], 'pc_23' : [7.02, 2.01], 'pc_34': [9.78, 4.09]}

    #NOTE: pc-n family uses different numbering. pc-1 is DZ(cardinal 2), pc-2 is TZ(cardinal 3), pc-4 is QZ(cardinal 4).
    if basis_family=='cc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='cc_23'
    elif basis_family=='aug-cc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='aug-cc_23'
    elif basis_family=='cc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='cc_34'
    elif basis_family=='aug-cc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='aug-cc_23'
    elif basis_family=='def2' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='def2_23'
    elif basis_family=='def2' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='def2_34'
    elif basis_family=='pc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='pc_23'
    elif basis_family=='pc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='pc_34'

    #Print energies
    print("Basis family is:", basis_family)
    print("SCF energies are:", scf_energies[0], "and", scf_energies[1])
    print("Correlation energies are:", corr_energies[0], "and", corr_energies[1])

    print("Extrapolation parameters:")
    alpha=extrapolation_parameters_dict[extrap_dict_key][0]
    beta=extrapolation_parameters_dict[extrap_dict_key][1]
    print("alpha :",alpha)
    print("beta :", beta)
    eX=math.exp(-1*alpha*math.sqrt(cardinals[0]))
    eY=math.exp(-1*alpha*math.sqrt(cardinals[1]))
    SCFextrap=(scf_energies[0]*eY-scf_energies[1]*eX)/(eY-eX)
    corrextrap=(math.pow(cardinals[0],beta)*corr_energies[0] - math.pow(cardinals[1],beta) * corr_energies[1])/(math.pow(cardinals[0],beta)-math.pow(cardinals[1],beta))

    print("SCF Extrapolated value is", SCFextrap)
    print("Correlation Extrapolated value is", corrextrap)
    print("Total Extrapolated value is", SCFextrap+corrextrap)

    return SCFextrap, corrextrap

def num_core_electrons(fragment):
    sum=0
    formula_list = functions_coords.molformulatolist(fragment.formula)
    for i in formula_list:
        els = atom_core_electrons[i]
        sum+=els
    return sum

#Note: Inner-shell correlation information: https://webhome.weizmann.ac.il/home/comartin/preprints/w1/node6.html
# Idea: Instead of CCSD(T), try out CEPA or pCCSD as alternative method. Hopefully as accurate as CCSD(T).
# Or DLPNO-CCSD(T) with LoosePNO ?

def W1theory_SP(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, scfsetting='TightSCF', numcores=1, 
                memory=5000, HFreference='QRO',extrainputkeyword='', extrablocks='', **kwargs):
    """
    Single-point W1 theory workflow.
    Differences: Basis sets may not be the same if 2nd-row element. TO BE CHECKED
    Scalar-relativistic step done via DKH. Same as modern W1 implementation.
    HF reference is important. UHF is not recommended. QRO gives very similar results to ROHF. To be set as default?
    Based on :
    https://webhome.weizmann.ac.il/home/comartin/w1/example.txt
    https://www.cup.uni-muenchen.de/oc/zipse/teaching/computational-chemistry-2/topics/overview-of-weizmann-theories/weizmann-1-theory/

    :param fragment: ASH fragment object
    :param charge: integer
    :param orcadir: string (path to ORCA)
    :param mult: integer (spin multiplicity)
    :param stabilityanalysis: boolean (currently not active)
    :param numcores: integer
    :param memory: integer (in MB)
    :param HFreference: string (UHF, QRO, ROHF)
    :return:
    """
    # If run_benchmark or other passed workflow_args then use them instead
    if 'workflow_args' in kwargs:
        print("Workflow args passed")
        workflow_args=kwargs['workflow_args']      
        if 'stabilityanalysis' in workflow_args:
            stabilityanalysis=workflow_args['stabilityanalysis']
        if 'scfsetting' in workflow_args:
            scfsetting=workflow_args['scfsetting']
        if 'memory' in workflow_args:
            memory=workflow_args['memory']
        if 'HFreference' in workflow_args:
            HFreference=workflow_args['HFreference']
        if 'extrainputkeyword' in workflow_args:
            extrainputkeyword=workflow_args['extrainputkeyword']
        if 'extrablocks' in workflow_args:
            extrablocks=workflow_args['extrablocks']    
    
    print("-----------------------------")
    print("W1theory_SP PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("HFreference : ", HFreference)
    print("")
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)
    numelectrons = int(fragment.nuccharge - charge)

    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        W1_total = -0.500000
        print("Using hardcoded value: ", W1_total)
        E_dict = {'Total_E': W1_total, 'E_SCF_CBS': W1_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return W1_total, E_dict

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #Disabling FullLMP2 guess in general as not available for open-shell
    #Adding memory and extrablocks.
    blocks="""
%maxcore {}
%scf
maxiter 1200
end
%mdci
maxiter 150
end
{}

""".format(memory,extrablocks)
    if stabilityanalysis is True:
        blocks = blocks + "%scf stabperform true end"
        
    #HF reference to use
    #If UHF then UHF will be enforced, also for closed-shell. unncessarily expensive
    if HFreference == 'UHF':
        print("HF reference = UHF chosen. Will enforce UHF (also for closed-shell)")
        hfkeyword='UHF'
    #ROHF option in ORCA requires dual-job. to be finished.
    elif HFreference == 'ROHF':
        print("ROHF reference is not yet available")
        exit()
    #QRO option is similar to ROHF. Recommended. Same as used by ORCA in DLPNO-CC.
    elif HFreference == 'QRO':
        print("HF reference = QRO chosen. Will use QROs for open-shell species)")
        hfkeyword='UNO'
    else:
        print("No HF reference chosen. Will use RHF for closed-shell and UHF for open-shell")
        hfkeyword=''

    ############################################################
    #Frozen-core calcs
    ############################################################
    #Special basis for H.
    # TODO: Add special basis for 2nd row block: Al-Ar
    # Or does ORCA W1-DZ choose this?
    ccsdt_dz_line="! CCSD(T) W1-DZ {} {} {}".format(scfsetting,hfkeyword,extrainputkeyword)

    ccsdt_tz_line="! CCSD(T) W1-TZ {} {} {}".format(scfsetting,hfkeyword,extrainputkeyword)

    ccsd_qz_line="! CCSD W1-QZ {} {} {}".format(scfsetting,hfkeyword,extrainputkeyword)

    ccsdt_dz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_dz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsd_qz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsd_qz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_dz)
    CCSDT_DZ_dict = grab_HF_and_corr_energies('orca-input.out')
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_DZ' + '.out')
    print("CCSDT_DZ_dict:", CCSDT_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = grab_HF_and_corr_energies('orca-input.out')
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsd_qz)
    CCSD_QZ_dict = grab_HF_and_corr_energies('orca-input.out')
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSD_QZ' + '.out')
    print("CCSD_QZ_dict:", CCSD_QZ_dict)

    #List of all SCF energies (DZ,TZ,QZ), all CCSD-corr energies (DZ,TZ,QZ) and all (T) corr energies (DZ,TZ)
    scf_energies = [CCSDT_DZ_dict['HF'], CCSDT_TZ_dict['HF'], CCSD_QZ_dict['HF']]
    ccsdcorr_energies = [CCSDT_DZ_dict['CCSD_corr'], CCSDT_TZ_dict['CCSD_corr'], CCSD_QZ_dict['CCSD_corr']]
    triplescorr_energies = [CCSDT_DZ_dict['CCSD(T)_corr'], CCSDT_TZ_dict['CCSD(T)_corr']]

    print("")
    print("scf_energies :", scf_energies)
    print("ccsdcorr_energies :", ccsdcorr_energies)
    print("triplescorr_energies :", triplescorr_energies)

    #Extrapolations
    #Choice for SCF: old 3-point formula or new 2-point formula. Need to check which is recommended nowadays
    E_SCF_CBS = Extrapolation_W1_SCF_3point(scf_energies) #3-point extrapolation
    #E_SCF_CBS = Extrapolation_W1_SCF_2point(scf_energies) #2-point extrapolation

    E_CCSDcorr_CBS = Extrapolation_W1_CCSD(ccsdcorr_energies) #2-point extrapolation
    E_triplescorr_CBS = Extrapolation_W1_triples(triplescorr_energies) #2-point extrapolation

    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_CCSDcorr_CBS:", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS:", E_triplescorr_CBS)

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    ccsdt_mtsmall_NoFC_line="! CCSD(T) DKH W1-mtsmall  {} nofrozencore {} {}".format(scfsetting,hfkeyword,extrainputkeyword)
    ccsdt_mtsmall_FC_line="! CCSD(T) W1-mtsmall {} {} {}".format(scfsetting,hfkeyword,extrainputkeyword)

    ccsdt_mtsmall_NoFC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        E_SO = atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
    else :
        E_SO = 0.0

    print("Spin-orbit correction (E_SO):", E_SO)

    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    W1_total = E_SCF_CBS + E_CCSDcorr_CBS + E_triplescorr_CBS +E_corecorr_and_SR  + E_SO
    print("Final W1 energy :", W1_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_CCSDcorr_CBS : ", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS : ", E_triplescorr_CBS)
    print("E_corecorr_and_SR : ", E_corecorr_and_SR)
    print("E_SO : ", E_SO)

    E_dict = {'Total_E' : W1_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS,
             'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_SO' : E_SO}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return W1_total, E_dict


def W1F12theory_SP(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1, scfsetting='TightSCF', 
                   memory=5000, HFreference='QRO',extrainputkeyword='', extrablocks='', **kwargs):
    """
    Single-point W1-F12 theory workflow.
    Differences: TBD
    Based on :
    https://webhome.weizmann.ac.il/home/comartin/OAreprints/240.pdf

    Differences: Core-valence and Rel done togeth using MTSmall as in W1 at the moment. TO be changed?
    No DBOC term
    

    :param fragment: ASH fragment object
    :param charge: integer
    :param orcadir: string (path to ORCA)
    :param mult: integer (spin multiplicity)
    :param stabilityanalysis: boolean (currently not active)
    :param numcores: integer
    :param memory: integer (in MB)
    :param HFreference: string (UHF, QRO, ROHF)
    :return:
    """
    
    # If run_benchmark or other passed workflow_args then use them instead
    if 'workflow_args' in kwargs:
        print("Workflow args passed")
        workflow_args=kwargs['workflow_args']      
        if 'stabilityanalysis' in workflow_args:
            stabilityanalysis=workflow_args['stabilityanalysis']
        if 'T1' in workflow_args:
            T1=workflow_args['T1']
        if 'scfsetting' in workflow_args:
            scfsetting=workflow_args['scfsetting']
        if 'memory' in workflow_args:
            memory=workflow_args['memory']
        if 'HFreference' in workflow_args:
            HFreference=workflow_args['HFreference']
        if 'extrainputkeyword' in workflow_args:
            extrainputkeyword=workflow_args['extrainputkeyword']
        if 'extrablocks' in workflow_args:
            extrablocks=workflow_args['extrablocks']
            
    print("-----------------------------")
    print("W1-F12 theory_SP PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("HFreference : ", HFreference)
    print("")
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)
    numelectrons = int(fragment.nuccharge - charge)

    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        W1_total = -0.500000
        print("Using hardcoded value: ", W1_total)
        E_dict = {'Total_E': W1_total, 'E_SCF_CBS': W1_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return W1_total, E_dict

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #Disabling FullLMP2 guess in general as not available for open-shell
    #Adding memory and extrablocks.
    blocks="""
%maxcore {}
%scf
maxiter 1200
end
%mdci
maxiter 150
end
{}

""".format(memory,extrablocks)
    if stabilityanalysis is True:
        blocks = blocks + "%scf stabperform true end"
        
    #HF reference to use
    #If UHF then UHF will be enforced, also for closed-shell. unncessarily expensive
    if HFreference == 'UHF':
        print("HF reference = UHF chosen. Will enforce UHF (also for closed-shell)")
        hfkeyword='UHF'
    #ROHF option in ORCA requires dual-job. to be finished.
    elif HFreference == 'ROHF':
        print("ROHF reference is not yet available")
        exit()
    #QRO option is similar to ROHF. Recommended. Same as used by ORCA in DLPNO-CC.
    elif HFreference == 'QRO':
        print("HF reference = QRO chosen. Will use QROs for open-shell species)")
        hfkeyword='UNO'
    else:
        print("No HF reference chosen. Will use RHF for closed-shell and UHF for open-shell")
        hfkeyword=''

    #Auxiliary basis set. One big one
    auxbasis='cc-pV5Z/C'

    ############################################################
    #Frozen-core calcs
    ############################################################
    #Special basis for H.

    #F12-calculations for SCF and CCSD
    ccsdf12_dz_line="! CCSD(T)-F12/RI cc-pVDZ-F12 cc-pVDZ-F12-CABS {} {} {} {}".format(scfsetting,auxbasis,hfkeyword,extrainputkeyword)
    ccsdf12_tz_line="! CCSD-F12/RI cc-pVTZ-F12 cc-pVTZ-F12-CABS {} {} {} {}".format(scfsetting,auxbasis,hfkeyword,extrainputkeyword)

    #Regular triples
    ccsdt_dz_line="! CCSD(T) W1-DZ tightscf {} {} ".format(hfkeyword,extrainputkeyword)
    ccsdt_tz_line="! CCSD(T) W1-TZ tightscf {} {} ".format(hfkeyword,extrainputkeyword)

    #F12
    ccsdf12_dz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_dz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdf12_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    #Regular
    ccsdt_dz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_dz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)    
    
    
    ash.Singlepoint(fragment=fragment, theory=ccsdf12_dz)
    CCSDF12_DZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDF12_DZ' + '.out')
    print("CCSDF12_DZ_dict:", CCSDF12_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdf12_tz)
    CCSDF12_TZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDF12_TZ' + '.out')
    print("CCSDF12_TZ_dict:", CCSDF12_TZ_dict)

    #Regular CCSD(T)
    ash.Singlepoint(fragment=fragment, theory=ccsdt_dz)
    CCSDT_DZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=False)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_DZ' + '.out')
    print("CCSDT_DZ_dict:", CCSDT_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=False)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)


    #List of all SCF energies (F12DZ,F12TZ), all CCSD-corr energies (F12DZ,F12TZ) and all (T) corr energies (DZ,TZ)
    scf_energies = [CCSDF12_DZ_dict['HF'], CCSDF12_TZ_dict['HF']]
    ccsdcorr_energies = [CCSDF12_DZ_dict['CCSD_corr'], CCSDF12_TZ_dict['CCSD_corr']]
    triplescorr_energies = [CCSDT_DZ_dict['CCSD(T)_corr'], CCSDT_TZ_dict['CCSD(T)_corr']]

    print("")
    print("scf_energies :", scf_energies)
    print("ccsdcorr_energies :", ccsdcorr_energies)
    print("triplescorr_energies :", triplescorr_energies)

    #Extrapolations
    #2-point SCF extrapolation of modified HF energies
    E_SCF_CBS = Extrapolation_W1F12_SCF_2point(scf_energies) #2-point extrapolation

    E_CCSDcorr_CBS = Extrapolation_W1F12_CCSD(ccsdcorr_energies) #2-point extrapolation
    E_triplescorr_CBS = Extrapolation_W1F12_triples(triplescorr_energies) #2-point extrapolation

    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_CCSDcorr_CBS:", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS:", E_triplescorr_CBS)

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    ccsdt_mtsmall_NoFC_line="! CCSD(T) DKH W1-mtsmall  tightscf nofrozencore {} {}".format(hfkeyword,extrainputkeyword)
    ccsdt_mtsmall_FC_line="! CCSD(T) W1-mtsmall tightscf {} {}".format(hfkeyword,extrainputkeyword)

    ccsdt_mtsmall_NoFC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        E_SO = atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
    else :
        E_SO = 0.0

    print("Spin-orbit correction (E_SO):", E_SO)

    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    W1F12_total = E_SCF_CBS + E_CCSDcorr_CBS + E_triplescorr_CBS +E_corecorr_and_SR  + E_SO
    print("Final W1-F12 energy :", W1F12_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_CCSDcorr_CBS : ", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS : ", E_triplescorr_CBS)
    print("E_corecorr_and_SR : ", E_corecorr_and_SR)
    print("E_SO : ", E_SO)

    E_dict = {'Total_E' : W1F12_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS,
             'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_SO' : E_SO}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return W1F12_total, E_dict


def DLPNO_W1F12theory_SP(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, 
                         numcores=1, memory=5000, pnosetting='NormalPNO', scfsetting='TightSCF',extrainputkeyword='', extrablocks='', **kwargs):
    """
    Single-point DLPNO W1-F12 theory workflow.
    Differences: TBD
    Based on :
    https://webhome.weizmann.ac.il/home/comartin/OAreprints/240.pdf
    with DLPNO

    Differences: Core-valence and Rel done togeth using MTSmall as in W1 at the moment. TO be changed?
    No DBOC term
    

    :param fragment: ASH fragment object
    :param charge: integer
    :param orcadir: string (path to ORCA)
    :param mult: integer (spin multiplicity)
    :param stabilityanalysis: boolean (currently not active)
    :param numcores: integer
    :param memory: integer (in MB)
    :param HFreference: string (UHF, QRO, ROHF)
    :return:
    """

    # If run_benchmark or other passed workflow_args then use them instead
    if 'workflow_args' in kwargs:
        print("Workflow args passed")
        workflow_args=kwargs['workflow_args']      
        if 'stabilityanalysis' in workflow_args:
            stabilityanalysis=workflow_args['stabilityanalysis']
        if 'pnosetting' in workflow_args:
            pnosetting=workflow_args['pnosetting']
        if 'scfsetting' in workflow_args:
            scfsetting=workflow_args['scfsetting']
        if 'memory' in workflow_args:
            memory=workflow_args['memory']
        if 'extrainputkeyword' in workflow_args:
            extrainputkeyword=workflow_args['extrainputkeyword']
        if 'extrablocks' in workflow_args:
            extrablocks=workflow_args['extrablocks']
            
    print("-----------------------------")
    print("DLPNO-W1-F12 theory_SP PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("PNO setting: ", pnosetting)
    print("")
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)
    numelectrons = int(fragment.nuccharge - charge)

    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        W1_total = -0.500000
        print("Using hardcoded value: ", W1_total)
        E_dict = {'Total_E': W1_total, 'E_SCF_CBS': W1_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return W1_total, E_dict

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #Disabling FullLMP2 guess in general as not available for open-shell
    #Adding memory and extrablocks.
    blocks="""
%maxcore {}
%scf
maxiter 1200
end
%mdci
UseFullLMP2Guess false
maxiter 150
end
{}

""".format(memory,extrablocks)
    if stabilityanalysis is True:
        blocks = blocks + "%scf stabperform true end"
        
    #Auxiliary basis set. One big one
    auxbasis='cc-pV5Z/C'

    ############################################################
    #Frozen-core calcs
    ############################################################
    #Special basis for H.

    #F12-calculations for SCF and CCSD contributions
    ccsdf12_dz_line="! DLPNO-CCSD-F12 cc-pVDZ-F12 cc-pVDZ-F12-CABS  {} {} {} {}".format(auxbasis,pnosetting,scfsetting,extrainputkeyword)
    ccsdf12_tz_line="! DLPNO-CCSD-F12 cc-pVTZ-F12 cc-pVTZ-F12-CABS  {} {} {} {}".format(auxbasis,pnosetting,scfsetting,extrainputkeyword)
    #TODO: Testing QZ CCSD step
    #ccsdf12_qz_line="! DLPNO-CCSD-F12 cc-pVQZ-F12 cc-pVQZ-F12-CABS  {} {} {}".format(auxbasis,pnosetting,scfsetting)
    
    #Regular triples
    ccsdt_dz_line="! DLPNO-CCSD(T) W1-DZ  {} {} {} {}".format(auxbasis,pnosetting,scfsetting,extrainputkeyword)
    ccsdt_tz_line="! DLPNO-CCSD(T) W1-TZ  {} {} {} {}".format(auxbasis,pnosetting,scfsetting,extrainputkeyword)

    #F12
    ccsdf12_dz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_dz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdf12_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    #ccsdf12_qz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_qz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    
    #Regular
    ccsdt_dz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_dz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)    
    
    
    ash.Singlepoint(fragment=fragment, theory=ccsdf12_dz)
    CCSDF12_DZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=True, DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDF12_DZ' + '.out')
    print("CCSDF12_DZ_dict:", CCSDF12_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdf12_tz)
    CCSDF12_TZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=True, DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDF12_TZ' + '.out')
    print("CCSDF12_TZ_dict:", CCSDF12_TZ_dict)

    #ash.Singlepoint(fragment=fragment, theory=ccsdf12_qz)
    #CCSDF12_QZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=True, DLPNO=True)
    #shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDF12_QZ' + '.out')
    #print("CCSDF12_QZ_dict:", CCSDF12_QZ_dict)

    #Regular CCSD(T)
    ash.Singlepoint(fragment=fragment, theory=ccsdt_dz)
    CCSDT_DZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=False, DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_DZ' + '.out')
    print("CCSDT_DZ_dict:", CCSDT_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=False, DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)


    #List of all SCF energies (F12DZ,F12TZ), all CCSD-corr energies (F12DZ,F12TZ) and all (T) corr energies (DZ,TZ)
    scf_energies = [CCSDF12_DZ_dict['HF'], CCSDF12_TZ_dict['HF']]
    ccsdcorr_energies = [CCSDF12_DZ_dict['CCSD_corr'], CCSDF12_TZ_dict['CCSD_corr']]
    triplescorr_energies = [CCSDT_DZ_dict['CCSD(T)_corr'], CCSDT_TZ_dict['CCSD(T)_corr']]
    #scf_energies = [CCSDF12_QZ_dict['HF']]
    #ccsdcorr_energies = [CCSDF12_QZ_dict['CCSD_corr']]
    #triplescorr_energies = [CCSDT_DZ_dict['CCSD(T)_corr'], CCSDT_TZ_dict['CCSD(T)_corr']]

    print("")
    print("scf_energies :", scf_energies)
    print("ccsdcorr_energies :", ccsdcorr_energies)
    print("triplescorr_energies :", triplescorr_energies)

    #Extrapolations
    #2-point SCF extrapolation of modified HF energies
    E_SCF_CBS = Extrapolation_W1F12_SCF_2point(scf_energies) #2-point extrapolation
    E_CCSDcorr_CBS = Extrapolation_W1F12_CCSD(ccsdcorr_energies) #2-point extrapolation
    #E_SCF_CBS = scf_energies[0]
    #E_CCSDcorr_CBS = ccsdcorr_energies[0]
    E_triplescorr_CBS = Extrapolation_W1F12_triples(triplescorr_energies) #2-point extrapolation

    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_CCSDcorr_CBS:", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS:", E_triplescorr_CBS)

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    ccsdt_mtsmall_NoFC_line="! DLPNO-CCSD(T) DKH W1-mtsmall   nofrozencore {} {} {} {}".format(auxbasis,pnosetting,scfsetting,extrainputkeyword)
    ccsdt_mtsmall_FC_line="! DLPNO-CCSD(T) W1-mtsmall  {} {} {} {}".format(auxbasis,pnosetting,scfsetting,extrainputkeyword)

    ccsdt_mtsmall_NoFC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        E_SO = atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
    else :
        E_SO = 0.0

    print("Spin-orbit correction (E_SO):", E_SO)

    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    DLPNOW1F12_total = E_SCF_CBS + E_CCSDcorr_CBS + E_triplescorr_CBS +E_corecorr_and_SR  + E_SO
    print("Final DLPNO-W1-F12 energy :", DLPNOW1F12_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_CCSDcorr_CBS : ", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS : ", E_triplescorr_CBS)
    print("E_corecorr_and_SR : ", E_corecorr_and_SR)
    print("E_SO : ", E_SO)

    E_dict = {'Total_E' : DLPNOW1F12_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS,
             'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_SO' : E_SO}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return DLPNOW1F12_total, E_dict


#DLPNO-test
def DLPNO_W1theory_SP(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1,
                      memory=5000, pnosetting='NormalPNO', T1=False, scfsetting='TightSCF',extrainputkeyword='', extrablocks='', **kwargs):
    """
    WORK IN PROGRESS
    DLPNO-version of single-point W1 theory workflow.
    Differences: DLPNO-CC enforces QRO reference (similar to ROHF). No other reference possible.

    :param fragment: ASH fragment
    :param charge: Charge of fragment (to be replaced)?
    :param orcadir: ORCA directory
    :param mult: Multiplicity of fragment (to be replaced)?
    :param stabilityanalysis: stability analysis on or off . Not currently active
    :param numcores: number of cores
    :param memory: Memory in MB
    :param scfsetting: ORCA keyword (e.g. NormalSCF, TightSCF, VeryTightSCF)
    :param pnosetting: ORCA keyword: NormalPNO, LoosePNO, TightPNO
    ;param T1: Boolean (whether to do expensive iterative triples or not)
    :return: energy and dictionary with energy-components
    """
    
    # If run_benchmark or other passed workflow_args then use them instead
    if 'workflow_args' in kwargs:
        print("Workflow args passed")
        workflow_args=kwargs['workflow_args']      
        if 'stabilityanalysis' in workflow_args:
            stabilityanalysis=workflow_args['stabilityanalysis']
        if 'pnosetting' in workflow_args:
            pnosetting=workflow_args['pnosetting']
        if 'T1' in workflow_args:
            T1=workflow_args['T1']
        if 'scfsetting' in workflow_args:
            scfsetting=workflow_args['scfsetting']
        if 'memory' in workflow_args:
            memory=workflow_args['memory']
        if 'extrainputkeyword' in workflow_args:
            extrainputkeyword=workflow_args['extrainputkeyword']
        if 'extrablocks' in workflow_args:
            extrablocks=workflow_args['extrablocks']
                
    print("-----------------------------")
    print("DLPNO_W1theory_SP PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("PNO setting: ", pnosetting)
    print("T1 : ", T1)
    print("SCF setting: ", scfsetting)
    print("")
    print("fragment:", fragment)
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)

    numelectrons = int(fragment.nuccharge - charge)

    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        W1_total = -0.500000
        print("Using hardcoded value: ", W1_total)
        E_dict = {'Total_E': W1_total, 'E_SCF_CBS': W1_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return W1_total, E_dict

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #Disabling FullLMP2 guess in general as not available for open-shell
    #Adding memory and extrablocks.
    blocks="""
%maxcore {}
%scf
maxiter 1200
end
%mdci
UseFullLMP2Guess false
maxiter 150
end
{}

""".format(memory,extrablocks)
    if stabilityanalysis is True:
        blocks = blocks + "%scf stabperform true end"

    #Auxiliary basis set. One big one
    auxbasis='cc-pV5Z/C'

    #Whether to use iterative triples or not. Default: regular DLPNO-CCSD(T)
    if T1 is True:
        ccsdtkeyword='DLPNO-CCSD(T1)'
    else:
        ccsdtkeyword='DLPNO-CCSD(T)'


    ############################################################s
    #Frozen-core calcs
    ############################################################
    #ccsdt_dz_line="! DLPNO-CCSD(T) {}cc-pVDZ {} tightscf ".format(prefix,auxbasis)
    #ccsdt_tz_line="! DLPNO-CCSD(T) {}cc-pVTZ {} tightscf ".format(prefix,auxbasis)
    #ccsd_qz_line="! DLPNO-CCSD {}cc-pVQZ {} tightscf ".format(prefix,auxbasis)
    ccsdt_dz_line="! {} W1-DZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
    ccsdt_tz_line="! {} W1-TZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
    ccsd_qz_line="! DLPNO-CCSD     W1-QZ {} {} {} {}".format(auxbasis, pnosetting, scfsetting,extrainputkeyword)


    ccsdt_dz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_dz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsd_qz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsd_qz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_dz)
    CCSDT_DZ_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_DZ' + '.out')
    print("CCSDT_DZ_dict:", CCSDT_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsd_qz)
    CCSD_QZ_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSD_QZ' + '.out')
    print("CCSD_QZ_dict:", CCSD_QZ_dict)

    #List of all SCF energies (DZ,TZ,QZ), all CCSD-corr energies (DZ,TZ,QZ) and all (T) corr energies (DZ,TZ)
    scf_energies = [CCSDT_DZ_dict['HF'], CCSDT_TZ_dict['HF'], CCSD_QZ_dict['HF']]
    ccsdcorr_energies = [CCSDT_DZ_dict['CCSD_corr'], CCSDT_TZ_dict['CCSD_corr'], CCSD_QZ_dict['CCSD_corr']]
    triplescorr_energies = [CCSDT_DZ_dict['CCSD(T)_corr'], CCSDT_TZ_dict['CCSD(T)_corr']]

    print("")
    print("scf_energies :", scf_energies)
    print("ccsdcorr_energies :", ccsdcorr_energies)
    print("triplescorr_energies :", triplescorr_energies)

    #Extrapolations
    #Choice for SCF: old 3-point formula or new 2-point formula. Need to check which is recommended nowadays
    E_SCF_CBS = Extrapolation_W1_SCF_3point(scf_energies) #3-point extrapolation
    #E_SCF_CBS = Extrapolation_W1_SCF_2point(scf_energies) #2-point extrapolation

    E_CCSDcorr_CBS = Extrapolation_W1_CCSD(ccsdcorr_energies) #2-point extrapolation
    E_triplescorr_CBS = Extrapolation_W1_triples(triplescorr_energies) #2-point extrapolation

    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_CCSDcorr_CBS:", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS:", E_triplescorr_CBS)

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    ccsdt_mtsmall_NoFC_line="! {} DKH W1-mtsmall  {} {} nofrozencore {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
    ccsdt_mtsmall_FC_line="! {} W1-mtsmall {}  {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)

    ccsdt_mtsmall_NoFC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile('orca-input.out', './'+ calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        E_SO = atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
    else :
        E_SO = 0.0

    print("Spin-orbit correction (E_SO):", E_SO)

    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    W1_total = E_SCF_CBS + E_CCSDcorr_CBS + E_triplescorr_CBS +E_corecorr_and_SR  + E_SO
    print("Final W1 energy :", W1_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_CCSDcorr_CBS : ", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS : ", E_triplescorr_CBS)
    print("E_corecorr_and_SR : ", E_corecorr_and_SR)
    print("E_SO : ", E_SO)

    E_dict = {'Total_E' : W1_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS,
             'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_SO' : E_SO}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return W1_total, E_dict




#DLPNO-F12
#Test: DLPNO-CCSD(T)-F12 protocol including CV+SR
def DLPNO_F12_SP(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1,
                      memory=5000, pnosetting='NormalPNO', T1=False, scfsetting='TightSCF', F12level='DZ',extrainputkeyword='', extrablocks='', **kwargs):
    """
    WORK IN PROGRESS
    DLPNO-CCSD(T)-F12 version of single-point W1-ish workflow.

    :param fragment: ASH fragment
    :param charge: Charge of fragment (to be replaced)?
    :param orcadir: ORCA directory
    :param mult: Multiplicity of fragment (to be replaced)?
    :param stabilityanalysis: stability analysis on or off . Not currently active
    :param numcores: number of cores
    :param memory: Memory in MB
    :param scfsetting: ORCA keyword (e.g. NormalSCF, TightSCF, VeryTightSCF)
    :param pnosetting: ORCA keyword: NormalPNO, LoosePNO, TightPNO
    ;param T1: Boolean (whether to do expensive iterative triples or not)
    :return: energy and dictionary with energy-components
    """
    
    # If run_benchmark or other passed workflow_args then use them instead
    if 'workflow_args' in kwargs:
        print("Workflow args passed")
        workflow_args=kwargs['workflow_args']      
        if 'stabilityanalysis' in workflow_args:
            stabilityanalysis=workflow_args['stabilityanalysis']
        if 'pnosetting' in workflow_args:
            pnosetting=workflow_args['pnosetting']
        if 'T1' in workflow_args:
            T1=workflow_args['T1']
        if 'scfsetting' in workflow_args:
            scfsetting=workflow_args['scfsetting']
        if 'memory' in workflow_args:
            memory=workflow_args['memory']
        if 'F12level' in workflow_args:
            F12level=workflow_args['F12level']
        if 'extrainputkeyword' in workflow_args:
            extrainputkeyword=workflow_args['extrainputkeyword']
        if 'extrablocks' in workflow_args:
            extrablocks=workflow_args['extrablocks']
            
    print("-----------------------------")
    print("DLPNO_F12_SP PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("PNO setting: ", pnosetting)
    print("T1 : ", T1)
    print("SCF setting: ", scfsetting)
    print("F12 basis level : ", F12level)
    print("")
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)

    numelectrons = int(fragment.nuccharge - charge)

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

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #Disabling FullLMP2 guess in general as not available for open-shell
    #Adding memory and extrablocks.
    blocks="""
%maxcore {}
%scf
maxiter 1200
end
%mdci
UseFullLMP2Guess false
maxiter 150
end
{}

""".format(memory,extrablocks)
    if stabilityanalysis is True:
        blocks = blocks + "%scf stabperform true end"

    #Auxiliary basis set. One big one
    auxbasis='cc-pV5Z/C'

    #Whether to use iterative triples or not. Default: regular DLPNO-CCSD(T)
    if T1 is True:
        
        print("test...")
        exit()
        ccsdtkeyword='DLPNO-CCSD(T1)'
    else:
        ccsdtkeyword='DLPNO-CCSD(T)-F12'


    ############################################################s
    #Frozen-core F12 calcs
    ############################################################

    ccsdt_f12_line="! {} cc-pV{}-F12 cc-pV{}-F12-CABS {} {} {} {}".format(ccsdtkeyword, F12level, F12level,auxbasis, pnosetting, scfsetting,extrainputkeyword)


    ccsdt_f12 = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_f12_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_f12)
    CCSDT_F12_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True,F12=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_F12' + '.out')
    print("CCSDT_F12_dict:", CCSDT_F12_dict)

    #List of all SCF energies (DZ,TZ,QZ), all CCSD-corr energies (DZ,TZ,QZ) and all (T) corr energies (DZ,TZ)
    scf_energies = [CCSDT_F12_dict['HF']]
    ccsdcorr_energies = [CCSDT_F12_dict['CCSD_corr']]
    triplescorr_energies = [CCSDT_F12_dict['CCSD(T)_corr']]

    print("")
    print("scf_energies :", scf_energies)
    print("ccsdcorr_energies :", ccsdcorr_energies)
    print("triplescorr_energies :", triplescorr_energies)

    #Final F12 energis
    E_SCF_CBS=scf_energies[0]
    E_CCSDcorr_CBS=ccsdcorr_energies[0]
    E_triplescorr_CBS=triplescorr_energies[0]

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    # Done regularly, not F12
    ############################################################
    print("Doing CV+SR at normal non-F12 level for now")
    ccsdt_mtsmall_NoFC_line="! DLPNO-CCSD(T) DKH W1-mtsmall  {} {} nofrozencore {} {}".format(auxbasis, pnosetting, scfsetting,extrainputkeyword)
    ccsdt_mtsmall_FC_line="! DLPNO-CCSD(T) W1-mtsmall {}  {} {} {}".format(auxbasis, pnosetting, scfsetting,extrainputkeyword)

    ccsdt_mtsmall_NoFC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile('orca-input.out', './'+ calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        E_SO = atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
    else :
        E_SO = 0.0

    print("Spin-orbit correction (E_SO):", E_SO)

    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    E_total = E_SCF_CBS + E_CCSDcorr_CBS + E_triplescorr_CBS +E_corecorr_and_SR  + E_SO
    print("Final DLPNO-CCSD(T)-F12 energy :", E_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_CCSDcorr_CBS : ", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS : ", E_triplescorr_CBS)
    print("E_corecorr_and_SR : ", E_corecorr_and_SR)
    print("E_SO : ", E_SO)

    E_dict = {'Total_E' : E_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS,
             'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_SO' : E_SO}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return E_total, E_dict


def DLPNO_W2theory_SP(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1,
                      memory=5000, pnosetting='NormalPNO', T1=False, scfsetting='TightSCF',extrainputkeyword='', extrablocks='', **kwargs):
    """
    WORK IN PROGRESS
    DLPNO-version of single-point W2 theory workflow.
    Differences: DLPNO-CC enforces QRO reference (similar to ROHF). No other reference possible.

    :param fragment: ASH fragment
    :param charge: Charge of fragment (to be replaced)?
    :param orcadir: ORCA directory
    :param mult: Multiplicity of fragment (to be replaced)?
    :param stabilityanalysis: stability analysis on or off . Not currently active
    :param numcores: number of cores
    :param memory: Memory in MB
    :param scfsetting: ORCA keyword (e.g. NormalSCF, TightSCF, VeryTightSCF)
    :param pnosetting: ORCA keyword: NormalPNO, LoosePNO, TightPNO
    ;param T1: Boolean (whether to do expensive iterative triples or not)
    :return: energy and dictionary with energy-components
    """
    
    # If run_benchmark or other passed workflow_args then use them instead
    if 'workflow_args' in kwargs:
        print("Workflow args passed")
        workflow_args=kwargs['workflow_args']      
        if 'stabilityanalysis' in workflow_args:
            stabilityanalysis=workflow_args['stabilityanalysis']
        if 'pnosetting' in workflow_args:
            pnosetting=workflow_args['pnosetting']
        if 'T1' in workflow_args:
            T1=workflow_args['T1']
        if 'scfsetting' in workflow_args:
            scfsetting=workflow_args['scfsetting']
        if 'memory' in workflow_args:
            memory=workflow_args['memory']
        if 'extrainputkeyword' in workflow_args:
            extrainputkeyword=workflow_args['extrainputkeyword']
        if 'extrablocks' in workflow_args:
            extrablocks=workflow_args['extrablocks']    
    
    print("-----------------------------")
    print("DLPNO_W2theory_SP PROTOCOL")
    print("-----------------------------")
    print("Not active yet")
    exit()
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)

    numelectrons = int(fragment.nuccharge - charge)

    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        W2_total = -0.500000
        print("Using hardcoded value: ", W2_total)
        E_dict = {'Total_E': W2_total, 'E_SCF_CBS': W2_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return W1_total, E_dict

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #Disabling FullLMP2 guess in general as not available for open-shell
    #Adding memory and extrablocks.
    blocks="""
%maxcore {}
%scf
maxiter 1200
end
%mdci
UseFullLMP2Guess false
maxiter 150
end
{}

""".format(memory,extrablocks)
    if stabilityanalysis is True:
        blocks = blocks + "%scf stabperform true end"
        

    #Auxiliary basis set. One big one
    #Todo: check whether it should be bigger
    auxbasis='cc-pV5Z/C'

    #Whether to use iterative triples or not. Default: regular DLPNO-CCSD(T)
    if T1 is True:
        ccsdtkeyword='DLPNO-CCSD(T1)'
    else:
        ccsdtkeyword='DLPNO-CCSD(T)'


    ############################################################s
    #Frozen-core calcs
    ############################################################
    #ccsdt_dz_line="! DLPNO-CCSD(T) {}cc-pVDZ {} tightscf ".format(prefix,auxbasis)
    #ccsdt_tz_line="! DLPNO-CCSD(T) {}cc-pVTZ {} tightscf ".format(prefix,auxbasis)
    #ccsd_qz_line="! DLPNO-CCSD {}cc-pVQZ {} tightscf ".format(prefix,auxbasis)
    ccsdt_tz_line="! {} W1-TZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
    ccsdt_qz_line="! {} W1-QZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
    ccsd_5z_line="! DLPNO-CCSD  haV5Z(+d) {} {} {} {}".format(auxbasis, pnosetting, scfsetting,extrainputkeyword)

    print("Need to check better if correct basis set.")

    #Defining W2 5Z basis
    #quintblocks = blocks + """%basis newgto H "cc-pV5Z" end
    #"""

    ccsdt_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_qz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_qz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsd_5z = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsd_5z_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_qz)
    CCSDT_QZ_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_QZ' + '.out')
    print("CCSDT_QZ_dict:", CCSDT_QZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsd_5z)
    CCSD_5Z_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSD_5Z' + '.out')
    print("CCSD_5Z_dict:", CCSD_5Z_dict)

    #List of all SCF energies (TZ,QZ,5Z), all CCSD-corr energies (TZ,QZ,5Z) and all (T) corr energies (TZ,qZ)
    scf_energies = [CCSDT_TZ_dict['HF'], CCSDT_QZ_dict['HF'], CCSD_5Z_dict['HF']]
    ccsdcorr_energies = [CCSDT_TZ_dict['CCSD_corr'], CCSDT_QZ_dict['CCSD_corr'], CCSD_5Z_dict['CCSD_corr']]
    triplescorr_energies = [CCSDT_TZ_dict['CCSD(T)_corr'], CCSDT_QZ_dict['CCSD(T)_corr']]

    print("")
    print("scf_energies :", scf_energies)
    print("ccsdcorr_energies :", ccsdcorr_energies)
    print("triplescorr_energies :", triplescorr_energies)

    #Extrapolations
    #Choice for SCF: old 3-point formula or new 2-point formula. Need to check which is recommended nowadays
    E_SCF_CBS = Extrapolation_W1_SCF_3point(scf_energies) #3-point extrapolation
    #E_SCF_CBS = Extrapolation_W1_SCF_2point(scf_energies) #2-point extrapolation

    E_CCSDcorr_CBS = Extrapolation_W2_CCSD(ccsdcorr_energies) #2-point extrapolation
    E_triplescorr_CBS = Extrapolation_W2_triples(triplescorr_energies) #2-point extrapolation

    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_CCSDcorr_CBS:", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS:", E_triplescorr_CBS)

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    ccsdt_mtsmall_NoFC_line="! {} DKH W1-mtsmall  {} {} nofrozencore {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
    ccsdt_mtsmall_FC_line="! {} W1-mtsmall {}  {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)

    ccsdt_mtsmall_NoFC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile('orca-input.out', './'+ calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        E_SO = atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
    else :
        E_SO = 0.0

    print("Spin-orbit correction (E_SO):", E_SO)

    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    W2_total = E_SCF_CBS + E_CCSDcorr_CBS + E_triplescorr_CBS +E_corecorr_and_SR  + E_SO
    print("Final W2 energy :", W2_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_CCSDcorr_CBS : ", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS : ", E_triplescorr_CBS)
    print("E_corecorr_and_SR : ", E_corecorr_and_SR)
    print("E_SO : ", E_SO)

    E_dict = {'Total_E' : W2_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS,
             'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_SO' : E_SO}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return W2_total, E_dict

#Thermochemistry protocol. Take list of fragments, stoichiometry, etc
#Requires orcadir, and theory level, typically an ORCATheory object
#Make more general. Not sure. ORCA makes most sense for geo-opt and HL theory
def thermochemprotocol(Opt_theory=None, SPprotocol=None, fraglist=None, stoichiometry=None, orcadir=None, numcores=None,
                       pnosetting='NormalPNO', F12level='DZ'):

    
    #DFT Opt+Freq  and Single-point High-level workflow
    FinalEnergies = []; list_of_dicts = []; ZPVE_Energies=[]
    for species in fraglist:
        #Only Opt+Freq for molecules, not atoms
        if species.numatoms != 1:
            #DFT-opt
            #ORCAcalc = ash.ORCATheory(orcadir=orcadir, charge=species.charge, mult=species.mult,
            #    orcasimpleinput=Opt_protocol_inputline, orcablocks=Opt_protocol_blocks, nprocs=numcores)
            #TODO: Check if this works in general. At least for ORCA.
            
            #Adding charge and mult to theory object, taken from each fragment object
            Opt_theory.charge = species.charge
            Opt_theory.mult = species.mult
            geomeTRICOptimizer(theory=Opt_theory,fragment=species)
            #DFT-FREQ
            thermochem = ash.NumFreq(fragment=species, theory=Opt_theory, npoint=2, runmode='serial')
            ZPVE = thermochem['ZPVE']
        else:
            #Setting ZPVE to 0.0.
            ZPVE=0.0
        #Single-point W1
        if SPprotocol == 'W1':
            FinalE, componentsdict = W1theory_SP(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, HFreference='QRO')
        elif SPprotocol == 'DLPNO-W1':
            FinalE, componentsdict = DLPNO_W1theory_SP(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, memory=5000, pnosetting=pnosetting, T1=False)
        elif SPprotocol == 'DLPNO-F12':
            FinalE, componentsdict = DLPNO_F12_SP(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, memory=5000, pnosetting=pnosetting, T1=False, F12level=F12level)
        elif SPprotocol == 'W1-F12':
            FinalE, componentsdict = W1F12theory_SP(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, memory=5000, HFreference='QRO')
        elif SPprotocol == 'DLPNO-W1-F12':
            FinalE, componentsdict = DLPNO_W1F12theory_SP(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, memory=5000, pnosetting=pnosetting)
        elif SPprotocol == 'DLPNO_CC_CBS':
            #TODO: Allow changing basisfamily and cardinals here?? Or should we stick with mostly simple non-changeable protocols here?
            FinalE, componentsdict = DLPNO_CC_CBS_SP(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, memory=5000, pnosetting=pnosetting)
        else:
            print("Unknown Singlepoint protocol")
            exit()
        FinalEnergies.append(FinalE+ZPVE); list_of_dicts.append(componentsdict)
        ZPVE_Energies.append(ZPVE)


    #Reaction Energy via list of total energies:
    scf_parts=[dict['E_SCF_CBS'] for dict in list_of_dicts]
    ccsd_parts=[dict['E_CCSDcorr_CBS'] for dict in list_of_dicts]
    triples_parts=[dict['E_triplescorr_CBS'] for dict in list_of_dicts]
    CV_SR_parts=[dict['E_corecorr_and_SR'] for dict in list_of_dicts]
    SO_parts=[dict['E_SO'] for dict in list_of_dicts]

    #Reaction Energy of total energiese and also different contributions
    print("")
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=scf_parts, unit='kcalpermol', label='ΔSCF')
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=ccsd_parts, unit='kcalpermol', label='ΔCCSD')
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=triples_parts, unit='kcalpermol', label='Δ(T)')
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=CV_SR_parts, unit='kcalpermol', label='ΔCV+SR')
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=SO_parts, unit='kcalpermol', label='ΔSO')
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=ZPVE_Energies, unit='kcalpermol', label='ΔZPVE')
    print("----------------------------------------------")
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalEnergies, unit='kcalpermol', label='Total ΔE')

    ash.print_time_rel(settings_ash.init_time,modulename='Entire thermochemprotocol')
    
    
    
    
    

#Functions to read and write energy-surface dictionary in simple format.
#Format: space-separated columns
# 1D: coordinate energy   e.g.    -180.0 -201.434343
# 2D: coordinate1 coordinate2 energy   e.g. e.g.   2.201 -180.0 -201.434343
#Output: dictionary: (tuple) : float   
# 1D: (coordinate1) : energy
# 2D: (coordinate1,coordinate2) : energy
#TODO: Make more general
def read_surfacedict_from_file(file, dimension=None):
    print("Attempting to read old results file...")
    dict = {}
    #If no file then return empty dict
    if os.path.isfile(file) is False:
        print("No file found.")
        return dict
    with open(file) as f:
        for line in f:
            if len(line) > 1:
                if dimension==1:
                    print("line:", line)
                    print(line.split())
                    key=float(line.split()[0])
                    val=float(line.split()[1])
                    dict[(key)]=val
                elif dimension==2:
                    print(line)
                    key1=float(line.split()[0])
                    key2=float(line.split()[1])
                    val=float(line.split()[2])                    
                    dict[(key1,key2)]=val
    return dict

def write_surfacedict_to_file(dict,file="surface_results.txt",dimension=None):
    with open(file, 'w') as f:
        for d in dict.items():
            if dimension==1:
                print("d:", d)
                x=d[0]
                e=d[1]
                f.write(str(x)+" "+str(e)+'\n')
            elif dimension==2:
                x=d[0][0]
                y=d[0][1]
                e=d[1]
                f.write(str(x)+" "+str(y)+" "+str(e)+'\n')

#Calculate 1D or 2D surface, either relaxed or unrelaxed.
# TODO: Parallelize surfacepoint calculations
def calc_surface(fragment=None, theory=None, workflow=None, type='Unrelaxed', resultfile='surface_results.txt', 
                 runmode='serial', coordsystem='dlc', **kwargs):    
    print("="*50)
    print("CALC_SURFACE FUNCTION")
    print("="*50)
    if 'numcores' in kwargs:
        numcores = kwargs['numcores']
    #Getting reaction coordinates and checking if 1D or 2D
    if 'RC1_range' in kwargs:
        RC1_range=kwargs['RC1_range']
        RC1_type=kwargs['RC1_type']
        RC1_indices=kwargs['RC1_indices']
        #Checking if list of lists. If so then we apply multiple constraints for this reaction coordinate (e.g. symmetric bonds)
        #Here making list of list in case only a single list was provided
        if any(isinstance(el, list) for el in RC1_indices) is False:
            RC1_indices=[RC1_indices]
        print("RC1_type:", RC1_type)
        print("RC1_indices:", RC1_indices)
        print("RC1_range:", RC1_range)
        
    if 'RC2_range' in kwargs:
        dimension=2
        RC2_range=kwargs['RC2_range']
        RC2_type=kwargs['RC2_type']
        RC2_indices=kwargs['RC2_indices']
        if any(isinstance(el, list) for el in RC2_indices) is False:
            RC2_indices=[RC2_indices]
        print("RC2_type:", RC2_type)
        print("RC2_indices:", RC2_indices)
        print("RC2_range:", RC2_range)
    else:
        dimension=1
    
    #Calc number of surfacepoints
    if dimension==2:
        range2=math.ceil((RC2_range[0]-RC2_range[1])/RC2_range[2])
        #print("range2", range2)
        range1=math.ceil((RC1_range[0]-RC1_range[1])/RC1_range[2])
        totalnumpoints=range2*range1
        #print("range1", range1)
        print("Number of surfacepoints to calculate ", totalnumpoints)
    elif dimension==1:
        totalnumpoints=math.ceil((RC1_range[0]-RC1_range[1])/RC1_range[2])
        print("Number of surfacepoints to calculate ", totalnumpoints)
    
    
    #Read dict from file. If file exists, read entries, if not, return empty dict
    surfacedictionary = read_surfacedict_from_file(resultfile, dimension=dimension)
    print("Initial surfacedictionary :", surfacedictionary)
    
    
    #Setting constraints once values are known
    def set_constraints(dimension=None,RCvalue1=None, RCvalue2=None):
        allconstraints = {}
        # Defining all constraints as dict to be passed to geometric
        if dimension == 2:
            RC2=[]
            RC1=[]
            #Creating empty lists for each RC type (Note: could be the same)
            allconstraints[RC1_type] = []
            allconstraints[RC2_type] = []
            
            for RC2_indexlist in RC2_indices:
                RC2.append(RC2_indexlist+[RCvalue2])
            allconstraints[RC2_type] = allconstraints[RC2_type] + RC2
            for RC1_indexlist in RC1_indices:
                RC1.append(RC1_indexlist+[RCvalue1])
            allconstraints[RC1_type] = allconstraints[RC1_type] + RC1
        elif dimension == 1:
            RC1=[]
            #Creating empty lists for each RC type (Note: could be the same)
            allconstraints[RC1_type] = []
            for RC1_indexlist in RC1_indices:
                RC1.append(RC1_indexlist+[RCvalue1])
            allconstraints[RC1_type] = allconstraints[RC1_type] + RC1
        return allconstraints
    
    pointcount=0
    
    #Create directory to keep track of surface XYZ files
    os.mkdir('surface_xyzfiles') 
    
    
    
    #PARALLEL CALCULATION
    if runmode=='parallel':
        print("Parallel runmode.")
        surfacepointfragments={}
        if type=='Unrelaxed':
            if dimension == 2:
                zerotheory = ash.ZeroTheory()
                for RCvalue1 in list(frange(RC1_range[0],RC1_range[1],RC1_range[2])):
                    for RCvalue2 in list(frange(RC2_range[0],RC2_range[1],RC2_range[2])):
                        pointcount+=1
                        print("=======================================")
                        print("Surfacepoint: {} / {}".format(pointcount,totalnumpoints))
                        print("RCvalue1: {} RCvalue2: {}".format(RCvalue1,RCvalue2))
                        print("=======================================")
                        if (RCvalue1,RCvalue2) not in surfacedictionary:
                            #Now setting constraints
                            allconstraints = set_constraints(dimension=2, RCvalue1=RCvalue1, RCvalue2=RCvalue2)
                            print("allconstraints:", allconstraints)
                            #Running zero-theory with optimizer just to set geometry
                            geomeTRICOptimizer(fragment=fragment, theory=zerotheory, coordsystem=coordsystem, constraints=allconstraints, constrainvalue=True)
                            #Shallow copy of fragment
                            newfrag = copy.copy(fragment)
                            newfrag.label = str(RCvalue1)+"_"+str(RCvalue1)
                            surfacepointfragments[(RCvalue1,RCvalue2)] = newfrag
                            #Single-point ORCA calculation on adjusted geometry
                            #energy = ash.Singlepoint(fragment=fragment, theory=theory)
                print("surfacepointfragments:", surfacepointfragments)
                #TODO: sort this list??
                surfacepointfragments_lists = list(surfacepointfragments.values())
                print("surfacepointfragments_lists: ", surfacepointfragments_lists)
                results = ash.Singlepoint_parallel(fragments=surfacepointfragments_lists, theories=[theory], numcores=numcores)
                print("Parallel calculation done!")
                print("results:", results)
                
                #Gathering results in dictionary
                for energy,coord in zip(results,surfacepointfragments_lists):
                    print("Coord : {}  Energy: {}".format(coord,energy))
                    surfacedictionary[coord] = energy
                    print("surfacedictionary:", surfacedictionary)
                    print("len surfacedictionary:", len(surfacedictionary))
                    print("totalnumpoints:", totalnumpoints)
                    if len(surfacedictionary) != totalnumpoints:
                        print("Dictionary not complete!")
                        print("len surfacedictionary:", len(surfacedictionary))
                        print("totalnumpoints:", totalnumpoints)
                
        exit()
    #SERIAL CALCULATION
    elif runmode=='serial':
        print("Serial runmode")
        if type=='Unrelaxed':
            zerotheory = ash.ZeroTheory()
            if dimension == 2:
                for RCvalue1 in list(frange(RC1_range[0],RC1_range[1],RC1_range[2])):
                    for RCvalue2 in list(frange(RC2_range[0],RC2_range[1],RC2_range[2])):
                        pointcount+=1
                        print("=======================================")
                        print("Surfacepoint: {} / {}".format(pointcount,totalnumpoints))
                        print("RCvalue1: {} RCvalue2: {}".format(RCvalue1,RCvalue2))
                        print("=======================================")
                        
                        if (RCvalue1,RCvalue2) not in surfacedictionary:
                            #Now setting constraints
                            allconstraints = set_constraints(dimension=2, RCvalue1=RCvalue1, RCvalue2=RCvalue2)
                            print("allconstraints:", allconstraints)
                            #Running zero-theory with optimizer just to set geometry
                            geomeTRICOptimizer(fragment=fragment, theory=zerotheory, coordsystem=coordsystem, constraints=allconstraints, constrainvalue=True)
                            
                            #Write geometry to disk
                            fragment.write_xyzfile(xyzfilename="RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")
                            fragment.print_system(filename="RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".ygg")
                            shutil.move("RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz", "surface_xyzfiles/RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")
                            #Single-point ORCA calculation on adjusted geometry
                            if theory is not None:
                                energy = ash.Singlepoint(fragment=fragment, theory=theory)
                            surfacedictionary[(RCvalue1,RCvalue2)] = energy
                            #Writing dictionary to file
                            write_surfacedict_to_file(surfacedictionary,"surface_results.txt", dimension=2)

                        else:
                            print("RC1, RC2 values in dict already. Skipping.")
                    print("surfacedictionary:", surfacedictionary)
            elif dimension == 1:
                for RCvalue1 in list(frange(RC1_range[0],RC1_range[1],RC1_range[2])):
                    pointcount+=1
                    print("=======================================")
                    print("Surfacepoint: {} / {}".format(pointcount,totalnumpoints))
                    print("RCvalue1: {}".format(RCvalue1))
                    print("=======================================")
                    
                    if (RCvalue1) not in surfacedictionary:
                        #Now setting constraints
                        allconstraints = set_constraints(dimension=1, RCvalue1=RCvalue1)
                        print("allconstraints:", allconstraints)
                        #Running zero-theory with optimizer just to set geometry
                        geomeTRICOptimizer(fragment=fragment, theory=zerotheory, coordsystem=coordsystem, constraints=allconstraints, constrainvalue=True)
                        
                        #Write geometry to disk: RC1_2.02.xyz
                        fragment.write_xyzfile(xyzfilename="RC1_"+str(RCvalue1)+".xyz")
                        fragment.print_system(filename="RC1_"+str(RCvalue1)+".ygg")
                        shutil.move("RC1_"+str(RCvalue1)+".xyz", "surface_xyzfiles/"+"RC1_"+str(RCvalue1)+".xyz")
                        #Single-point ORCA calculation on adjusted geometry
                        energy = ash.Singlepoint(fragment=fragment, theory=theory)
                        surfacedictionary[(RCvalue1)] = energy
                        #Writing dictionary to file
                        write_surfacedict_to_file(surfacedictionary,"surface_results.txt", dimension=1)
                        print("surfacedictionary:", surfacedictionary)
                    else:
                        print("RC1 value in dict already. Skipping.")
        elif type=='Relaxed':
            zerotheory = ash.ZeroTheory()
            if dimension == 2:
                for RCvalue1 in list(frange(RC1_range[0],RC1_range[1],RC1_range[2])):
                    for RCvalue2 in list(frange(RC2_range[0],RC2_range[1],RC2_range[2])):
                        pointcount+=1
                        print("=======================================")
                        print("Surfacepoint: {} / {}".format(pointcount,totalnumpoints))
                        print("RCvalue1: {} RCvalue2: {}".format(RCvalue1,RCvalue2))
                        print("=======================================")
                        if (RCvalue1,RCvalue2) not in surfacedictionary:
                            #Now setting constraints
                            allconstraints = set_constraints(dimension=2, RCvalue1=RCvalue1, RCvalue2=RCvalue2)
                            print("allconstraints:", allconstraints)
                            #Running 
                            energy = geomeTRICOptimizer(fragment=fragment, theory=theory, coordsystem=coordsystem, constraints=allconstraints, constrainvalue=True)
                            surfacedictionary[(RCvalue1,RCvalue2)] = energy
                            #Writing dictionary to file
                            write_surfacedict_to_file(surfacedictionary,"surface_results.txt", dimension=2)
                            
                            #Write geometry to disk
                            fragment.write_xyzfile(xyzfilename="RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")
                            fragment.print_system(filename="RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".ygg")
                            shutil.move("RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz", "surface_xyzfiles/RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")

                        else:
                            print("RC1, RC2 values in dict already. Skipping.")
                    print("surfacedictionary:", surfacedictionary)
            elif dimension == 1:
                for RCvalue1 in list(frange(RC1_range[0],RC1_range[1],RC1_range[2])):
                    pointcount+=1
                    print("=======================================")
                    print("Surfacepoint: {} / {}".format(pointcount,totalnumpoints))
                    print("RCvalue1: {}".format(RCvalue1))
                    print("=======================================")
                    
                    if (RCvalue1) not in surfacedictionary:
                        #Now setting constraints
                        allconstraints = set_constraints(dimension=1, RCvalue1=RCvalue1)
                        print("allconstraints:", allconstraints)
                        #Running zero-theory with optimizer just to set geometry
                        energy = geomeTRICOptimizer(fragment=fragment, theory=theory, coordsystem=coordsystem, constraints=allconstraints, constrainvalue=True)
                        surfacedictionary[(RCvalue1)] = energy
                        #Writing dictionary to file
                        write_surfacedict_to_file(surfacedictionary,"surface_results.txt", dimension=1)
                        print("surfacedictionary:", surfacedictionary)
                        
                        #Write geometry to disk
                        fragment.write_xyzfile(xyzfilename="RC1_"+str(RCvalue1)+".xyz")
                        fragment.print_system(filename="RC1_"+str(RCvalue1)+".ygg")
                        shutil.move("RC1_"+str(RCvalue1)+".xyz", "surface_xyzfiles/"+"RC1_"+str(RCvalue1)+".xyz")
                    else:
                        print("RC1 value in dict already. Skipping.")
    return surfacedictionary

# Calculate surface from XYZ-file collection. Single-point only for now
def calc_surface_fromXYZ(xyzdir=None, theory=None, dimension=None, resultfile=None ):
    
    print("="*50)
    print("CALC_SURFACE_FROMXYZ FUNCTION")
    print("="*50)
    print("XYZdir:", xyzdir)
    print("Theory:", theory)
    print("Dimension:", dimension)
    print("Resultfile:", resultfile)
    print("")
    #Read dict from file. If file exists, read entries, if not, return empty dict
    surfacedictionary = read_surfacedict_from_file(resultfile, dimension=dimension)
    print("Initial surfacedictionary :", surfacedictionary)

    #Looping over XYZ files
    for file in glob.glob(xyzdir+'/*.xyz'):
        relfile=os.path.basename(file)
        #Getting RC values from XYZ filename e.g. RC1_2.0-RC2_180.0.xyz
        if dimension == 2:
            RCvalue1=float(relfile.split('-')[0][4:])
            RCvalue2=float(relfile.split('-')[1][4:].replace('.xyz',''))
            print("XYZ-file: {}     RC1: {} RC2: {}".format(relfile,RCvalue1,RCvalue2))
            if (RCvalue1,RCvalue2) not in surfacedictionary:
                mol=ash.Fragment(xyzfile=file)
                energy = ash.Singlepoint(theory=theory, fragment=mol)
                print("Energy of file {} : {} Eh".format(relfile, energy))
                theory.cleanup()
                surfacedictionary[(RCvalue1,RCvalue2)] = energy
                #Writing dictionary to file
                write_surfacedict_to_file(surfacedictionary,"surface_results.txt", dimension=2)
                print("surfacedictionary:", surfacedictionary)
                print("")
            else:
                print("RC1 and RC2 values in dict already. Skipping.")
        elif dimension == 1:
            #RC1_2.02.xyz
            RCvalue1=float(relfile.replace('.xyz','').replace('RC1_',''))
            print("XYZ-file: {}     RC1: {} ".format(relfile,RCvalue1))
            if (RCvalue1) not in surfacedictionary:
                mol=ash.Fragment(xyzfile=file)
                energy = ash.Singlepoint(theory=theory, fragment=mol)
                print("Energy of file {} : {} Eh".format(relfile, energy))
                theory.cleanup()
                surfacedictionary[(RCvalue1)] = energy
                #Writing dictionary to file
                write_surfacedict_to_file(surfacedictionary,"surface_results.txt", dimension=1)
                print("surfacedictionary:", surfacedictionary)
                print("")            
            else:
                print("RC1 value in dict already. Skipping.")

    return surfacedictionary







#DLPNO-test CBS protocol. Simple. No core-correlation, scalar relativistic or spin-orbit coupling for now
def DLPNO_CC_CBS_SP(cardinals = "2/3", basisfamily="def2", fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1,
                      memory=5000, pnosetting='NormalPNO', T1=False, scfsetting='TightSCF', extrainputkeyword='', extrablocks='', **kwargs):
    """
    WORK IN PROGRESS
    DLPNO-CCSD(T)/CBS frozencore workflow

    :param fragment: ASH fragment
    :param charge: Charge of fragment (to be replaced)?
    :param orcadir: ORCA directory
    :param mult: Multiplicity of fragment (to be replaced)?
    :param stabilityanalysis: stability analysis on or off.
    :param numcores: number of cores
    :param memory: Memory in MB
    :param scfsetting: ORCA keyword (e.g. NormalSCF, TightSCF, VeryTightSCF)
    :param pnosetting: ORCA keyword: NormalPNO, LoosePNO, TightPNO
    ;param T1: Boolean (whether to do expensive iterative triples or not)
    :return: energy and dictionary with energy-components
    """
    # If run_benchmark or other passed workflow_args then use them instead
    if 'workflow_args' in kwargs:
        print("Workflow args passed")
        workflow_args=kwargs['workflow_args']
        if 'cardinals' in workflow_args:
            cardinals=workflow_args['cardinals']
        if 'basisfamily' in workflow_args:
            basisfamily=workflow_args['basisfamily']        
        if 'stabilityanalysis' in workflow_args:
            stabilityanalysis=workflow_args['stabilityanalysis']
        if 'pnosetting' in workflow_args:
            pnosetting=workflow_args['pnosetting']
        if 'T1' in workflow_args:
            T1=workflow_args['T1']
        if 'scfsetting' in workflow_args:
            scfsetting=workflow_args['scfsetting']
        if 'memory' in workflow_args:
            memory=workflow_args['memory']
        if 'extrainputkeyword' in workflow_args:
            extrainputkeyword=workflow_args['extrainputkeyword']
        if 'extrablocks' in workflow_args:
            extrablocks=workflow_args['extrablocks']

    print("-----------------------------")
    print("DLPNO_CC_CBS_SP PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("Cardinals chosen:", cardinals)
    print("Basis set family chosen:", basisfamily)
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("PNO setting: ", pnosetting)
    print("T1 : ", T1)
    print("SCF setting: ", scfsetting)
    print("Stability analysis:", stabilityanalysis)
    print("")
    print("fragment:", fragment)
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)

    numelectrons = int(fragment.nuccharge - charge)

    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        E_total = -0.500000
        print("Using hardcoded value: ", E_total)
        E_dict = {'Total_E': W1_total, 'E_SCF_CBS': W1_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return W1_total, E_dict

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #Disabling FullLMP2 guess in general as not available for open-shell
    #Adding memory and extrablocks.
    blocks="""
%maxcore {}
%scf
maxiter 1200
end
%mdci
UseFullLMP2Guess false
maxiter 150
end
{}

""".format(memory,extrablocks)
    if stabilityanalysis is True:
        blocks = blocks + "%scf stabperform true end"

    #Auxiliary basis set. One big one
    auxbasis='cc-pV5Z/C'

    #Whether to use iterative triples or not. Default: regular DLPNO-CCSD(T)
    if T1 is True:
        ccsdtkeyword='DLPNO-CCSD(T1)'
    else:
        ccsdtkeyword='DLPNO-CCSD(T)'


    ############################################################s
    #Frozen-core DLPNO-CCSD(T) calculations defined here
    ############################################################
    if cardinals == "2/3" and basisfamily=="def2":
        ccsdt_1_line="! {} def2-SVP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
    elif cardinals == "3/4" and basisfamily=="def2":
        ccsdt_1_line="! {} def2-TZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} def2-QZVPP {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
    elif cardinals == "2/3" and basisfamily=="cc":
        ccsdt_1_line="! {} cc-pVDZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} cc-pVTZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
    elif cardinals == "3/4" and basisfamily=="cc":
        ccsdt_1_line="! {} cc-pVTZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} cc-pVQZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
    elif cardinals == "2/3" and basisfamily=="aug-cc":
        ccsdt_1_line="! {} aug-cc-pVDZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pVTZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
    elif cardinals == "3/4" and basisfamily=="aug-cc":
        ccsdt_1_line="! {} aug-cc-pVTZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pVQZ {} {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting,extrainputkeyword)
        
    #Defining two theory objects for each basis set
    ccsdt_1 = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_1_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_2 = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_2_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    #Running both theories
    ash.Singlepoint(fragment=fragment, theory=ccsdt_1)
    CCSDT_1_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_1' + '.out')
    print("CCSDT_1_dict:", CCSDT_1_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_2)
    CCSDT_2_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_2' + '.out')
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
    
    #Extrapolations

    E_SCF_CBS, E_corr_CBS = Extrapolation_twopoint(scf_energies, corr_energies, [2,3], 'def2') #3-point extrapolation

    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_corr_CBS:", E_corr_CBS)

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    #DISABLED FOR NOW

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    #DISABLED FOR NOW
    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    E_total = E_SCF_CBS + E_corr_CBS 
    print("Final DLPNO-CCSD(T)/CBS energy :", E_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_corr_CBS : ", E_corr_CBS)

    E_dict = {'Total_E' : E_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS, 'E_corr_CBS' : E_corr_CBS}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return E_total, E_dict

#Provide crest/xtb info, MLtheory object (e.g. ORCA), HLtheory object (e.g. ORCA)
def confsampler_protocol(fragment=None, crestdir=None, xtbmethod='GFN2-xTB', MLtheory=None, 
                         HLtheory=None, orcadir=None, numcores=1, charge=None, mult=None):
    print("="*50)
    print("CONFSAMPLER FUNCTION")
    print("="*50)
    
    #1. Calling crest
    #call_crest(fragment=molecule, xtbmethod='GFN2-xTB', crestdir=crestdir, charge=charge, mult=mult, solvent='H2O', energywindow=6 )
    call_crest(fragment=fragment, xtbmethod=xtbmethod, crestdir=crestdir, charge=charge, mult=mult, numcores=numcores)

    #2. Grab low-lying conformers from crest_conformers.xyz as list of ASH fragments.
    list_conformer_frags, xtb_energies = get_crest_conformers()

    print("list_conformer_frags:", list_conformer_frags)
    print("")
    print("Crest Conformer Searches done. Found {} conformers".format(len(xtb_energies)))
    print("xTB energies: ", xtb_energies)

    #3. Run ML (e.g. DFT) geometry optimizations for each crest-conformer

    ML_energies=[]
    print("")
    for index,conformer in enumerate(list_conformer_frags):
        print("")
        print("Performing ML Geometry Optimization for Conformer ", index)
        geomeTRICOptimizer(fragment=conformer, theory=MLtheory, coordsystem='tric')
        ML_energies.append(conformer.energy)
        #Saving ASH fragment and XYZ file for each ML-optimized conformer
        os.rename('Fragment-optimized.ygg', 'Conformer{}_ML.ygg'.format(index))
        os.rename('Fragment-optimized.xyz', 'Conformer{}_ML.xyz'.format(index))

    print("")
    print("ML Geometry Optimization done")
    print("ML_energies: ", ML_energies)

    #4.Run high-level thery. Provide HLtheory object (typically ORCATheory)
    HL_energies=[]
    for index,conformer in enumerate(list_conformer_frags):
        print("")
        print("Performing High-level calculation for ML-optimized Conformer ", index)
        HLenergy = ash.Singlepoint(theory=HLTheory, fragment=conformer)
        HL_energies.append(HLenergy)


    print("")
    print("=================")
    print("FINAL RESULTS")
    print("=================")

    #Printing total energies
    print("")
    print(" Conformer   xTB-energy    ML-energy    HL-energy (Eh)")
    print("----------------------------------------------------------------")

    min_xtbenergy=min(xtb_energies)
    min_MLenergy=min(ML_energies)
    min_HLenergy=min(HL_energies)

    for index,(xtb_en,ML_en,HL_en) in enumerate(zip(xtb_energies,ML_energies, HL_energies)):
        print("{:10} {:13.10f} {:13.10f} {:13.10f}".format(index,xtb_en, ML_en, HL_en))

    print("")
    #Printing relative energies
    min_xtbenergy=min(xtb_energies)
    min_MLenergy=min(ML_energies)
    min_HLenergy=min(HL_energies)
    harkcal = 627.50946900
    print(" Conformer   xTB-energy    ML-energy    HL-energy (kcal/mol)")
    print("----------------------------------------------------------------")
    for index,(xtb_en,ML_en,HL_en) in enumerate(zip(xtb_energies,ML_energies, HL_energies)):
        rel_xtb=(xtb_en-min_xtbenergy)*harkcal
        rel_ML=(ML_en-min_MLenergy)*harkcal
        rel_HL=(HL_en-min_HLenergy)*harkcal
        print("{:10} {:13.10f} {:13.10f} {:13.10f}".format(index,rel_xtb, rel_ML, rel_HL))

    print("")
    print("Workflow done!")