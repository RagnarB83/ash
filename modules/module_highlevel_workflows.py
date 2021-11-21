#High-level WFT workflows
import numpy as np
import os
import ash
import shutil
import constants
import math
import dictionaries_lists
import interfaces.interface_ORCA
from functions.functions_elstructure import check_cores_vs_electrons, num_core_electrons


#Note: Inner-shell correlation information: https://webhome.weizmann.ac.il/home/comartin/preprints/w1/node6.html
# Idea: Instead of CCSD(T), try out CEPA or pCCSD as alternative method. Hopefully as accurate as CCSD(T).
# Or DLPNO-CCSD(T) with LoosePNO ?

def W1theory(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, scfsetting='TightSCF', numcores=1, 
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
    if 'workflow_args' in kwargs and kwargs['workflow_args'] is not None:
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
    print("W1theory PROTOCOL")
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
    #Reduce numcores if required
    numcores = check_cores_vs_electrons(fragment,numcores,charge)
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

    ccsdt_dz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_dz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_tz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsd_qz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsd_qz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_dz)
    CCSDT_DZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_dz.filename+'.out')
    shutil.copyfile(ccsdt_dz.filename+'.out', './' + calc_label + 'CCSDT_DZ' + '.out')
    print("CCSDT_DZ_dict:", CCSDT_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_tz.filename+'.out')
    shutil.copyfile(ccsdt_tz.filename+'.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsd_qz)
    CCSD_QZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsd_qz.filename+'.out')
    shutil.copyfile(ccsd_qz.filename+'.out', './' + calc_label + 'CCSD_QZ' + '.out')
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

    ccsdt_mtsmall_NoFC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'out', './' + calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        try:
            E_SO = dictionaries_lists.atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
        except KeyError:
            print("Found no SO value for atom. Will set to 0.0 and continue")
            E_SO = 0.0
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
    os.remove(ccsdt_mtsmall_FC.filename+'.gbw')

    #return final energy and also dictionary with energy components
    return W1_total, E_dict


def W1F12theory(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1, scfsetting='TightSCF', 
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
    if 'workflow_args' in kwargs and kwargs['workflow_args'] is not None:
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
    print("W1-F12 theory PROTOCOL")
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
    #Reduce numcores if required
    numcores = check_cores_vs_electrons(fragment,numcores,charge)
    
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
    ccsdf12_dz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_dz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdf12_tz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_tz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    #Regular
    ccsdt_dz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_dz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_tz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)    
    
    
    ash.Singlepoint(fragment=fragment, theory=ccsdf12_dz)
    CCSDF12_DZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdf12_dz.filename+'out', F12=True)
    shutil.copyfile(ccsdf12_dz.filename+'out', './' + calc_label + 'CCSDF12_DZ' + '.out')
    print("CCSDF12_DZ_dict:", CCSDF12_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdf12_tz)
    CCSDF12_TZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdf12_tz.filename+'out', F12=True)
    shutil.copyfile(ccsdf12_tz.filename+'out', './' + calc_label + 'CCSDF12_TZ' + '.out')
    print("CCSDF12_TZ_dict:", CCSDF12_TZ_dict)

    #Regular CCSD(T)
    ash.Singlepoint(fragment=fragment, theory=ccsdt_dz)
    CCSDT_DZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_dz.filename+'out', F12=False)
    shutil.copyfile(ccsdt_dz.filename+'out', './' + calc_label + 'CCSDT_DZ' + '.out')
    print("CCSDT_DZ_dict:", CCSDT_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_tz.filename+'out', F12=False)
    shutil.copyfile(ccsdt_tz.filename+'out', './' + calc_label + 'CCSDT_TZ' + '.out')
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

    ccsdt_mtsmall_NoFC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile(ccsdt_mtsmall_FC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        try:
            E_SO = dictionaries_lists.atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
        except KeyError:
            print("Found no SO value for atom. Will set to 0.0 and continue")
            E_SO = 0.0
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
    os.remove(ccsdt_mtsmall_NoFC+'.gbw')

    #return final energy and also dictionary with energy components
    return W1F12_total, E_dict


def DLPNO_W1F12theory(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, 
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
    if 'workflow_args' in kwargs and kwargs['workflow_args'] is not None:
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
    print("DLPNO-W1-F12 theory PROTOCOL")
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
    #Reduce numcores if required
    numcores = check_cores_vs_electrons(fragment,numcores,charge)

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
    ccsdf12_dz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_dz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdf12_tz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_tz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    #ccsdf12_qz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_qz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    
    #Regular
    ccsdt_dz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_dz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_tz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)    
    
    
    ash.Singlepoint(fragment=fragment, theory=ccsdf12_dz)
    CCSDF12_DZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdf12_dz.filename+'.out', F12=True, DLPNO=True)
    shutil.copyfile(ccsdf12_dz.filename+'.out', './' + calc_label + 'CCSDF12_DZ' + '.out')
    print("CCSDF12_DZ_dict:", CCSDF12_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdf12_tz)
    CCSDF12_TZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdf12_tz.filename+'.out', F12=True, DLPNO=True)
    shutil.copyfile(ccsdf12_tz.filename+'.out', './' + calc_label + 'CCSDF12_TZ' + '.out')
    print("CCSDF12_TZ_dict:", CCSDF12_TZ_dict)

    #ash.Singlepoint(fragment=fragment, theory=ccsdf12_qz)
    #CCSDF12_QZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=True, DLPNO=True)
    #shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDF12_QZ' + '.out')
    #print("CCSDF12_QZ_dict:", CCSDF12_QZ_dict)

    #Regular CCSD(T)
    ash.Singlepoint(fragment=fragment, theory=ccsdt_dz)
    CCSDT_DZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_dz.filename+'.out', F12=False, DLPNO=True)
    shutil.copyfile(ccsdt_dz.filename+'.out', './' + calc_label + 'CCSDT_DZ' + '.out')
    print("CCSDT_DZ_dict:", CCSDT_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_tz.filename+'.out', F12=False, DLPNO=True)
    shutil.copyfile(ccsdt_tz.filename+'.out', './' + calc_label + 'CCSDT_TZ' + '.out')
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

    ccsdt_mtsmall_NoFC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile(ccsdt_mtsmall_FC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        try:
            E_SO = dictionaries_lists.atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
        except KeyError:
            print("Found no SO value for atom. Will set to 0.0 and continue")
            E_SO = 0.0
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
    os.remove(ccsdt_mtsmall_FC.filename+'.gbw')

    #return final energy and also dictionary with energy components
    return DLPNOW1F12_total, E_dict


#DLPNO-test
def DLPNO_W1theory(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1,
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
    if 'workflow_args' in kwargs and kwargs['workflow_args'] is not None:
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
    print("DLPNO_W1theory PROTOCOL")
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
    #Reduce cores if needed
    numcores = check_cores_vs_electrons(fragment,numcores,charge)
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


    ccsdt_dz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_dz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_tz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsd_qz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsd_qz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_dz)
    CCSDT_DZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_dz.filename+'.out', DLPNO=True)
    shutil.copyfile(ccsdt_dz.filename+'.out', './' + calc_label + 'CCSDT_DZ' + '.out')
    print("CCSDT_DZ_dict:", CCSDT_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_tz.filename+'.out', DLPNO=True)
    shutil.copyfile(ccsdt_tz.filename+'.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsd_qz)
    CCSD_QZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsd_qz.filename+'.out', DLPNO=True)
    shutil.copyfile(ccsd_qz.filename+'.out', './' + calc_label + 'CCSD_QZ' + '.out')
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

    ccsdt_mtsmall_NoFC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './'+ calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile(ccsdt_mtsmall_FC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        try:
            E_SO = dictionaries_lists.atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
        except KeyError:
            print("Found no SO value for atom. Will set to 0.0 and continue")
            E_SO = 0.0
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
    os.remove(ccsdt_mtsmall_FC.filename+'.gbw')

    #return final energy and also dictionary with energy components
    return W1_total, E_dict




#DLPNO-F12
#Test: DLPNO-CCSD(T)-F12 protocol including CV+SR
def DLPNO_F12(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1,
                      memory=5000, pnosetting='NormalPNO', T1=False, scfsetting='TightSCF', F12level='DZ', extrainputkeyword='', extrablocks='', **kwargs):
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
    if 'workflow_args' in kwargs and kwargs['workflow_args'] is not None:
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
    print("DLPNO_F12 PROTOCOL")
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
    #Reduce cores if needed
    numcores = check_cores_vs_electrons(fragment,numcores,charge)

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


    ccsdt_f12 = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_f12_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_f12)
    CCSDT_F12_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_f12.filename+'.out', DLPNO=True,F12=True)
    shutil.copyfile(ccsdt_f12.filename+'.out', './' + calc_label + 'CCSDT_F12' + '.out')
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

    ccsdt_mtsmall_NoFC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './'+ calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile(ccsdt_mtsmall_FC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        try:
            E_SO = dictionaries_lists.atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
        except KeyError:
            print("Found no SO value for atom. Will set to 0.0 and continue")
            E_SO = 0.0
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
    os.remove(ccsdt_mtsmall_FC.filename+'.gbw')

    #return final energy and also dictionary with energy components
    return E_total, E_dict


def DLPNO_W2theory(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1,
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
    if 'workflow_args' in kwargs and kwargs['workflow_args'] is not None:
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
    print("DLPNO_W2theory PROTOCOL")
    print("-----------------------------")
    print("Not active yet")
    exit()
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)

    numelectrons = int(fragment.nuccharge - charge)
    #Check if numcores should be reduced
    numcores = check_cores_vs_electrons(fragment,numcores,charge)
    
    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        W2_total = -0.500000
        print("Using hardcoded value: ", W2_total)
        E_dict = {'Total_E': W2_total, 'E_SCF_CBS': W2_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return W2_total, E_dict



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

    ccsdt_tz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_qz = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_qz_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsd_5z = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsd_5z_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_tz.filename+'.out', DLPNO=True)
    shutil.copyfile(ccsdt_tz.filename+'.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_qz)
    CCSDT_QZ_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_qz.filename+'.out', DLPNO=True)
    shutil.copyfile(ccsdt_qz.filename+'.out', './' + calc_label + 'CCSDT_QZ' + '.out')
    print("CCSDT_QZ_dict:", CCSDT_QZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsd_5z)
    CCSD_5Z_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsd_5z.filename+'.out', DLPNO=True)
    shutil.copyfile(ccsd_5z.filename+'.out', './' + calc_label + 'CCSD_5Z' + '.out')
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

    ccsdt_mtsmall_NoFC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile(ccsdt_mtsmall_NoFC.filename+'.out', './'+ calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile(ccsdt_mtsmall_FC.filename+'.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        try:
            E_SO = dictionaries_lists.atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
        except KeyError:
            print("Found no SO value for atom. Will set to 0.0 and continue")
            E_SO = 0.0
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
    os.remove(ccsdt_mtsmall_FC+'.gbw')

    #return final energy and also dictionary with energy components
    return W2_total, E_dict

#Flexible CCSD(T)/CBS protocol. Simple. No core-correlation, scalar relativistic or spin-orbit coupling for now.
# Regular CC, DLPNO-CC, DLPNO-CC with PNO extrapolation etc.
#alpha and beta can be manually set. If not set then they are picked based on basisfamily
def CC_CBS(cardinals = [2,3], basisfamily="def2", relativity=None, fragment=None, charge=None, orcadir=None, mult=None, 
           stabilityanalysis=False, numcores=1, CVSR=False, CVbasis="W1-mtsmall", F12=False, DFTreference=None, DFT_RI=False,
                        DLPNO=False, memory=5000, pnosetting='NormalPNO', pnoextrapolation=[5,6], T1=False, scfsetting='TightSCF',
                        alpha=None, beta=None,
                        extrainputkeyword='', extrablocks='', **kwargs):
    """
    WORK IN PROGRESS
    CCSD(T)/CBS frozencore workflow

    :param fragment: ASH fragment
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
    # If run_benchmark or other passed workflow_args then use them instead
    if 'workflow_args' in kwargs and kwargs['workflow_args'] is not None:
        print("Workflow args passed")
        workflow_args=kwargs['workflow_args']
        if 'cardinals' in workflow_args:
            cardinals=workflow_args['cardinals']
        if 'basisfamily' in workflow_args:
            basisfamily=workflow_args['basisfamily']
        if 'CVbasis' in workflow_args:
            CVbasis=workflow_args['CVbasis']
        if 'stabilityanalysis' in workflow_args:
            stabilityanalysis=workflow_args['stabilityanalysis']
        if 'pnosetting' in workflow_args:
            pnosetting=workflow_args['pnosetting']
        if 'CVSR' in workflow_args:
            CVSR=workflow_args['CVSR']
        if 'pnoextrapolation' in workflow_args:
            pnoextrapolation=workflow_args['pnoextrapolation']
        if 'T1' in workflow_args:
            T1=workflow_args['T1']
        if 'DLPNO' in workflow_args:
            DLPNO=workflow_args['DLPNO']
        if 'scfsetting' in workflow_args:
            scfsetting=workflow_args['scfsetting']
        if 'memory' in workflow_args:
            memory=workflow_args['memory']
        if 'extrainputkeyword' in workflow_args:
            extrainputkeyword=workflow_args['extrainputkeyword']
        if 'extrablocks' in workflow_args:
            extrablocks=workflow_args['extrablocks']

    print("-----------------------------")
    print("CC_CBS PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("Cardinals chosen:", cardinals)
    print("Basis set family chosen:", basisfamily)
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("DLPNO:", DLPNO)
    if DLPNO == True:
        print("PNO setting: ", pnosetting)
        if pnosetting == "extrapolation":
            print("pnoextrapolation:", pnoextrapolation)
        print("T1 : ", T1)
    print("SCF setting: ", scfsetting)
    print("Relativity: ", relativity)
    print("Stability analysis:", stabilityanalysis)
    print("Core-Valence Scalar Relativistic correction (CVSR): ", CVSR)
    print("")
    print("fragment:", fragment)
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)

    numelectrons = int(fragment.nuccharge - charge)
    #Reduce numcores if required
    numcores = check_cores_vs_electrons(fragment,numcores,charge)


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



    #Choosing whether DLPNO or not
    if DLPNO == True:
        #Iterative DLPNO triples or not
        if T1 is True:
            ccsdtkeyword='DLPNO-CCSD(T1)'
        else:
            #DLPNO-F12 or not
            if F12 is True:
                print("Note: not supported yet ")
                exit()
                ccsdtkeyword='DLPNO-CCSD(T)-F12'
            else:
                ccsdtkeyword='DLPNO-CCSD(T)'
        #Add PNO keyword in simpleinputline or not (if extrapolation)
        if pnosetting != "extrapolation":
            pnokeyword=pnosetting
        else:
            pnokeyword=""
    #Regular CCSD(T)
    else:
        #No PNO keywords
        pnokeyword=""
        pnosetting=None
        #F12 or not
        if F12 is True:
            ccsdtkeyword='CCSD(T)-F12'
            print("Note: not supported yet ")
            exit()
        else:
            ccsdtkeyword='CCSD(T)'


    #SCALAR RELATIVITY HAMILTONIAN AND SELECT CORRELATED AUX BASIS
    #TODO: Handle RIJK/RIJCOSX and AUXBASIS FOR HF/DFT reference also
    if relativity == None:
        extrainputkeyword = extrainputkeyword + '  '
         #Auxiliary basis set. 1 big one for now
         #TODO: look more into
        if 'def2' in basisfamily:
            auxbasis='def2-QZVPP/C'
        else:
            if 'aug' in basisfamily:
                auxbasis='aug-cc-pV5Z/C'                
            else:
                auxbasis='cc-pV5Z/C'
    elif relativity == 'DKH':
        extrainputkeyword = extrainputkeyword + ' DKH '
        if 'def2' in basisfamily:
            auxbasis='def2-QZVPP/C'
        else:
            if 'aug' in basisfamily:
                auxbasis='aug-cc-pV5Z/C'                
            else:
                auxbasis='cc-pV5Z/C'
    elif relativity == 'ZORA':
        extrainputkeyword = extrainputkeyword + ' ZORA '
        auxbasis='cc-pV5Z/C'
        if 'def2' in basisfamily:
            auxbasis='def2-QZVPP/C'
        else:
            if 'aug' in basisfamily:
                auxbasis='aug-cc-pV5Z/C'                
            else:
                auxbasis='cc-pV5Z/C'
    elif relativity == 'X2C':
        extrainputkeyword = extrainputkeyword + ' X2C '
        auxbasis='cc-pV5Z/C'
        print("Not ready")
        exit()

    #Possible DFT reference (functional name)
    #NOTE: Hardcoding RIJCOSX SARC/J defgrid3 for now
    if DFTreference != None:
        if DFT_RI is True:
            extrainputkeyword = extrainputkeyword + ' {} RIJCOSX SARC/J defgrid3 '.format(DFTreference)
        else:
            extrainputkeyword = extrainputkeyword + ' {} NORI defgrid3 '.format(DFTreference)

    ############################################################s
    #Frozen-core CCSD(T) calculations defined here
    ############################################################
    #Choosing 
    ccsdt_1_line,ccsdt_2_line=choose_inputlines_from_basisfamily(cardinals,basisfamily,ccsdtkeyword,auxbasis,pnokeyword,scfsetting,extrainputkeyword)



    #Adding special-ECP basis like cc-pVnZ-PP for heavy elements if present
    blocks1 = special_element_basis(fragment,cardinals[0],basisfamily,blocks)
    blocks2 = special_element_basis(fragment,cardinals[1],basisfamily,blocks)
    
    #Check if we are using an ECP
    ECPflag=isECP(blocks1)
    
    
    #Defining two theory objects for each basis set
    ccsdt_1 = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_1_line, orcablocks=blocks1, numcores=numcores, charge=charge, mult=mult)
    ccsdt_2 = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_2_line, orcablocks=blocks2, numcores=numcores, charge=charge, mult=mult)
    
    
    # EXTRAPOLATION TO PNO LIMIT BY 2 PNO calculations
    if pnosetting=="extrapolation":
        print("PNO Extrapolation option chosen.")
        print("Will run 2 jobs with PNO thresholds TCutPNO : 1e-{} and 1e-{}".format(pnoextrapolation[0],pnoextrapolation[1]))
        E_SCF_1, E_corrCCSD_1, E_corrCCT_1,E_corrCC_1 = PNOExtrapolationStep(fragment=fragment, theory=ccsdt_1, pnoextrapolation=pnoextrapolation, DLPNO=DLPNO, F12=False, calc_label=calc_label)
        E_SCF_2, E_corrCCSD_2, E_corrCCT_2,E_corrCC_2 = PNOExtrapolationStep(fragment=fragment, theory=ccsdt_2, pnoextrapolation=pnoextrapolation, DLPNO=DLPNO, F12=False, calc_label=calc_label)
        scf_energies = [E_SCF_1, E_SCF_2]
        ccsdcorr_energies = [E_corrCCSD_1, E_corrCCSD_2]
        triplescorr_energies = [E_corrCCT_1, E_corrCCT_2]
        corr_energies = [E_corrCC_1, E_corrCC_2]
    # OR REGULAR
    else:
        #Running both theories
        ash.Singlepoint(fragment=fragment, theory=ccsdt_1)
        CCSDT_1_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_1.filename+'.out', DLPNO=DLPNO)
        shutil.copyfile(ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
        print("CCSDT_1_dict:", CCSDT_1_dict)

        ash.Singlepoint(fragment=fragment, theory=ccsdt_2)
        CCSDT_2_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_2.filename+'.out', DLPNO=DLPNO)
        shutil.copyfile(ccsdt_2.filename+'.out', './' + calc_label + 'CCSDT_2' + '.out')
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

    E_SCF_CBS, E_corr_CBS = Extrapolation_twopoint(scf_energies, corr_energies, cardinals, basisfamily, alpha=alpha, beta=beta) #2-point extrapolation

    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_corr_CBS:", E_corr_CBS)

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    if CVSR is True:
        print("")
        print("Core-Valence Scalar Relativistic Correction is on!")
        #TODO: We should only do CV if we are doing all-electron calculations. If we have heavy element then we have probably added an ECP (specialbasisfunction)
        # Switch to doing only CV correction in that case ?
        # TODO: Option if W1-mtsmall basis set is not available?
        
        if ECPflag is True:
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
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        try:
            E_SO = dictionaries_lists.atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
        except KeyError:
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
    os.remove(ccsdt_1.filename+'.gbw')

    #return final energy and also dictionary with energy components
    return E_FINAL, E_dict

    
    
    

#FCI/CBS protocol. No core-correlation, scalar relativistic or spin-orbit coupling for now.
# Extrapolates CC series to Full-CI and to CBS 
def FCI_CBS(cardinals = [2,3], basisfamily="def2", fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1,
                      memory=5000, DLPNO=True, pnosetting='NormalPNO', F12=False, T1=False, scfsetting='TightSCF', extrainputkeyword='', extrablocks='', **kwargs):
    """
    WORK IN PROGRESS
    FCI/CBS frozencore workflow

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
    if 'workflow_args' in kwargs and kwargs['workflow_args'] is not None:
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
    print("FCI_CBS PROTOCOL")
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
    #Reduce numcores if required
    numcores = check_cores_vs_electrons(fragment,numcores,charge)




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


    #Whether to use iterative triples or not. Default: regular DLPNO-CCSD(T)
    if DLPNO == True:
        pnokeyword=pnosetting
        if T1 is True:
            ccsdtkeyword='DLPNO-CCSD(T1)'
        else:
            if F12 is True:
                ccsdtkeyword='DLPNO-CCSD(T)-F12'
            else:
                ccsdtkeyword='DLPNO-CCSD(T)'   
    else:
        pnokeyword=""
        if F12 is True:
            ccsdtkeyword='CCSD(T)-F12'
        else:
            ccsdtkeyword='CCSD(T)'



    ############################################################s
    #Frozen-core CCSD(T) calculations defined here
    ############################################################
    ccsdt_1_line,ccsdt_2_line=choose_inputlines_from_basisfamily(cardinals,basisfamily,ccsdtkeyword,auxbasis,pnokeyword,scfsetting,extrainputkeyword)


    #Adding special-ECP basis like cc-pVnZ-PP for heavy elements if present
    blocks1 = special_element_basis(fragment,cardinals[0],basisfamily,blocks)
    blocks2 = special_element_basis(fragment,cardinals[1],basisfamily,blocks)
    
    #Check if we are using an ECP
    ECPflag=isECP(blocks1)
    
    #Defining two theory objects for each basis set
    ccsdt_1 = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_1_line, orcablocks=blocks1, numcores=numcores, charge=charge, mult=mult)
    ccsdt_2 = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_2_line, orcablocks=blocks2, numcores=numcores, charge=charge, mult=mult)

    #Running both theories
    ash.Singlepoint(fragment=fragment, theory=ccsdt_1)
    CCSDT_1_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_1.filename+'.out', DLPNO=DLPNO)
    shutil.copyfile(ccsdt_1.filename+'.out', './' + calc_label + 'CCSDT_1' + '.out')
    print("CCSDT_1_dict:", CCSDT_1_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_2)
    CCSDT_2_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_2.filename+'.out', DLPNO=DLPNO)
    shutil.copyfile(ccsdt_2.filename+'.out', './' + calc_label + 'CCSDT_2' + '.out')
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
    # TODO: Extrapolation formula appropriate for separate CCSD and triples extraplation ?? Use W1 formulas instead?
    E_SCF_CBS, E_corrCC_CBS = Extrapolation_twopoint(scf_energies, corr_energies, cardinals, basisfamily) #2-point extrapolation
    E_SCF_CBS, E_corrCCSD_CBS = Extrapolation_twopoint(scf_energies, ccsdcorr_energies, cardinals, basisfamily) #2-point extrapolation
    E_SCF_CBS, E_corrCCT_CBS = Extrapolation_twopoint(scf_energies, triplescorr_energies, cardinals, basisfamily) #2-point extrapolation
    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_corrCC_CBS:", E_corrCC_CBS)
    print("E_corrCCSD_CBS:", E_corrCCSD_CBS)
    print("E_corrCCT_CBS:", E_corrCCT_CBS)
    E_total_CC = E_SCF_CBS + E_corrCC_CBS
    
    
    
    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    if CVSR is True:
        print("")
        print("Core-Valence Scalar Relativistic Correction is on!")
        #TODO: We should only do CV if we are doing all-electron calculations. If we have heavy element then we have probably added an ECP (specialbasisfunction)
        # Switch to doing only CV correction in that case ?
        # TODO: Option if W1-mtsmall basis set is not available?
        
        if ECPflag is True:
            print("ECPs present. Not doing ScalarRelativistic Correction. Switching to Core-Valence Correction only.")
            cvbasis="W1-mtsmall"
            reloption=" "
            pnooption="NormalPNO"
            print("Doing CVSR_Step with No Scalar Relativity and CV-basis: {} and PNO-option: {}".format(cvbasis,pnooption))
            E_corecorr_and_SR = CV_Step(cvbasis,reloption,ccsdtkeyword,auxbasis,pnooption,scfsetting,extrainputkeyword,orcadir,blocks,numcores,charge,mult,fragment,calc_label)
        else:
            cvbasis="W1-mtsmall"
            reloption="DKH"
            pnooption="NormalPNO"
            print("Doing CVSR_Step with Relativistic Option: {} and CV-basis: {} and PNO-option: {}".format(reloption,cvbasis,pnooption))
            E_corecorr_and_SR = CVSR_Step(cvbasis,reloption,ccsdtkeyword,auxbasis,pnooption,scfsetting,extrainputkeyword,orcadir,blocks,numcores,charge,mult,fragment,calc_label)
            
        
    else:
        print("")
        print("Core-Valence Scalar Relativistic Correction is off!")
        E_corecorr_and_SR=0.0

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        try:
            E_SO = dictionaries_lists.atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
        except KeyError:
            print("Found no SO value for atom. Will set to 0.0 and continue")
            E_SO = 0.0
    else :
        E_SO = 0.0

    
    ############################################################
    #FCI extrapolation via Goodson
    ############################################################
    print("FCI extrapolation by Goodson in use")
    #Here using CBS-values for SCF, CCSD-corr and (T)-corr.
    E_FCI_CBS = FCI_extrapolation([E_SCF_CBS, E_corrCCSD_CBS, E_corrCCT_CBS])
    
    
    ############################################################
    #FINAL RESULT PRINTING
    ############################################################
    #Combining E_FCI_CBS + SO + CV+SR
    print("")
    print("")
    E_FINAL = E_FCI_CBS + E_SO + E_corecorr_and_SR

    print("")
    print("")

    print("Final FCI/CBS energy :", E_FINAL, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_corrCC_CBS : ", E_corrCC_CBS)
    print("CCSD(T)/CBS energy :", E_total_CC, "Eh")
    print("FCI correction : ", E_FCI_CBS-E_total_CC, "Eh")
    print("FCI correlation energy : ", E_FCI_CBS-E_SCF_CBS, "Eh")
    print("Spin-orbit coupling : ", E_SO, "Eh")
    print("E_corecorr_and_SR : ", E_corecorr_and_SR, "Eh")
    E_dict = {'Total_E' : E_FINAL, 'E_FCI_CBS' : E_FCI_CBS, 'E_SCF_CBS' : E_SCF_CBS, 'E_corrCC_CBS' : E_corrCC_CBS, 'E_total_CC': E_total_CC, 'E_SO' : E_SO, 'E_corecorr_and_SR' : E_corecorr_and_SR}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove(ccsdt_1.filename+'.gbw')

    #return final energy and also dictionary with energy components
    return E_FCI_CBS, E_dict








  

#FCI-F12 protocol. No core-correlation, scalar relativistic or spin-orbit coupling for now.
# Extrapolates CC series to Full-CI and uses F12 to deal with basis set limit
#Limitation: availability of F12 basis sets for H-Ar only.
def FCI_F12(F12level='DZ', fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1, CVSR=False,
                      memory=5000, DLPNO=True, pnosetting='NormalPNO', pnoextrapolation=[5,6], scfsetting='TightSCF', extrainputkeyword='', extrablocks='', **kwargs):
    """
    WORK IN PROGRESS
    FCI/CBS frozencore workflow

    :param fragment: ASH fragment
    :param charge: Charge of fragment (to be replaced)?
    :param orcadir: ORCA directory
    :param mult: Multiplicity of fragment (to be replaced)?
    :param stabilityanalysis: stability analysis on or off.
    :param numcores: number of cores
    :param memory: Memory in MB
    :param scfsetting: ORCA keyword (e.g. NormalSCF, TightSCF, VeryTightSCF)
    :param pnosetting: ORCA keyword: NormalPNO, LoosePNO, TightPNO or extrapolation
    :param pnoextrapolation: [5,6] or [6,7] for TCutPNO extrapolation
    :return: energy and dictionary with energy-components
    """
    # If run_benchmark or other passed workflow_args then use them instead
    if 'workflow_args' in kwargs and kwargs['workflow_args'] is not None:
        print("Workflow args passed")
        workflow_args=kwargs['workflow_args']      
        if 'stabilityanalysis' in workflow_args:
            stabilityanalysis=workflow_args['stabilityanalysis']
        if 'pnosetting' in workflow_args:
            pnosetting=workflow_args['pnosetting']
        if 'F12level' in workflow_args:
            F12level=workflow_args['F12level']
        if 'scfsetting' in workflow_args:
            scfsetting=workflow_args['scfsetting']
        if 'memory' in workflow_args:
            memory=workflow_args['memory']
        if 'extrainputkeyword' in workflow_args:
            extrainputkeyword=workflow_args['extrainputkeyword']
        if 'extrablocks' in workflow_args:
            extrablocks=workflow_args['extrablocks']

    print("-----------------------------")
    print("FCI_F12 PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("F12level :", F12level)
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("PNO setting: ", pnosetting)
    print("SCF setting: ", scfsetting)
    print("Stability analysis:", stabilityanalysis)
    print("")
    print("fragment:", fragment)
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)

    numelectrons = int(fragment.nuccharge - charge)
    #Reduce numcores if required
    numcores = check_cores_vs_electrons(fragment,numcores,charge)

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


    #Whether to use iterative triples or not. Default: regular DLPNO-CCSD(T)
    if DLPNO is True:
        ccsdtkeyword='DLPNO-CCSD(T)-F12'
    else:
        ccsdtkeyword='CCSD(T)-F12/RI'
        pnosetting=""


    #Auxiliary basis set. One big one
    auxbasis='cc-pV5Z/C'


    ############################################################s
    #Frozen-core F12 calcs
    ############################################################

    ccsdt_f12_line="! {} cc-pV{}-F12 cc-pV{}-F12-CABS {} {} {} {}".format(ccsdtkeyword, F12level, F12level,auxbasis, pnosetting, scfsetting,extrainputkeyword)
    ccsdt_f12 = interfaces.interface_ORCA.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_f12_line, orcablocks=blocks, numcores=numcores, charge=charge, mult=mult)

    #PNO extrapolation or not
    if pnosetting=="extrapolation":
        E_SCF_CBS, E_corrCCSD_CBS, E_corrCCT_CBS,E_corrCC_CBS = PNOExtrapolationStep(fragment=fragment, theory=ccsdt_f12, pnoextrapolation=pnoextrapolation, DLPNO=True, F12=True, calc_label=calc_label)
        
    #Regular single-PNO-setting job
    else:
        ash.Singlepoint(fragment=fragment, theory=ccsdt_f12)
        CCSDT_F12_dict = interfaces.interface_ORCA.grab_HF_and_corr_energies(ccsdt_f12.filename+'.out', DLPNO=DLPNO,F12=True)
    
        shutil.copyfile(ccsdt_f12.filename+'.out', './' + calc_label + 'CCSDT_F12' + '.out')
        print("CCSDT_F12_dict:", CCSDT_F12_dict)

        #List of  SCF energy, CCSDcorr energy, (T)corr energy
        E_SCF_CBS = CCSDT_F12_dict['HF']
        E_corrCCSD_CBS = CCSDT_F12_dict['CCSD_corr']
        E_corrCCT_CBS = CCSDT_F12_dict['CCSD(T)_corr']
        E_corrCC_CBS = E_corrCCSD_CBS + E_corrCCT_CBS


    E_total_CC = E_SCF_CBS + E_corrCC_CBS
    
    
    
    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    if CVSR is True:
        print("")
        print("Core-Valence Scalar Relativistic Correction is on!")
        #TODO: We should only do CV if we are doing all-electron calculations. If we have heavy element then we have probably added an ECP (specialbasisfunction)
        # Switch to doing only CV correction in that case ?
        # TODO: Option if W1-mtsmall basis set is not available?
        
        if ECPflag is True:
            print("ECPs present. Not doing ScalarRelativistic Correction. Switching to Core-Valence Correction only.")
            cvbasis="W1-mtsmall"
            reloption=" "
            pnooption="NormalPNO"
            print("Doing CVSR_Step with No Scalar Relativity and CV-basis: {} and PNO-option: {}".format(cvbasis,pnooption))
            E_corecorr_and_SR = CV_Step(cvbasis,reloption,ccsdtkeyword,auxbasis,pnooption,scfsetting,extrainputkeyword,orcadir,blocks,numcores,charge,mult,fragment,calc_label)
        else:
            cvbasis="W1-mtsmall"
            reloption="DKH"
            pnooption="NormalPNO"
            print("Doing CVSR_Step with Relativistic Option: {} and CV-basis: {} and PNO-option: {}".format(reloption,cvbasis,pnooption))
            E_corecorr_and_SR = CVSR_Step(cvbasis,reloption,ccsdtkeyword,auxbasis,pnooption,scfsetting,extrainputkeyword,orcadir,blocks,numcores,charge,mult,fragment,calc_label)
            
        
    else:
        print("")
        print("Core-Valence Scalar Relativistic Correction is off!")
        E_corecorr_and_SR=0.0
    
    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        try:
            E_SO = dictionaries_lists.atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
        except KeyError:
            print("Found no SO value for atom. Will set to 0.0 and continue")
            E_SO = 0.0
    else :
        E_SO = 0.0
    
    ############################################################
    #FCI extrapolation via Goodson
    ############################################################
    print("FCI extrapolation by Goodson in use")
    #Here using CBS-values for SCF, CCSD-corr and (T)-corr.
    E_FCI_CBS = FCI_extrapolation([E_SCF_CBS, E_corrCCSD_CBS, E_corrCCT_CBS])
    
    ############################################################
    #FINAL RESULT PRINTING
    ############################################################
    #Combining E_FCI_CBS + SO + CV+SR
    E_FINAL = E_FCI_CBS + E_SO + E_corecorr_and_SR
    print("")
    print("")

    print("Final FCI/CBS energy :", E_FINAL, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_corrCC_CBS : ", E_corrCC_CBS)
    print("CCSD(T)/CBS energy :", E_total_CC, "Eh")
    print("FCI correction : ", E_FCI_CBS-E_total_CC, "Eh")
    print("FCI correlation energy : ", E_FCI_CBS-E_SCF_CBS, "Eh")
    print("Spin-orbit coupling : ", E_SO, "Eh")
    print("E_corecorr_and_SR : ", E_corecorr_and_SR, "Eh")
    E_dict = {'Total_E' : E_FINAL, 'E_FCI_CBS' : E_FCI_CBS, 'E_SCF_CBS' : E_SCF_CBS, 'E_corrCC_CBS' : E_corrCC_CBS, 'E_total_CC': E_total_CC, 'E_SO' : E_SO, 'E_corecorr_and_SR' : E_corecorr_and_SR}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove(ccsdt_f12.filename+'.gbw')

    #return final energy and also dictionary with energy components
    return E_FCI_CBS, E_dict





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
        auxbasis='aug-cc-pV5Z/C'
        ccsdt_1_line="! {} aug-cc-pVTZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pVQZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
    elif cardinals == [4,5] and basisfamily=="aug-cc-dk":
        #Auxiliary basis set.
        auxbasis='Autoaux'
        ccsdt_1_line="! {} aug-cc-pVQZ-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        ccsdt_2_line="! {} aug-cc-pV5Z-DK {} {} {} {}".format(ccsdtkeyword, auxbasis, pnokeyword, scfsetting,extrainputkeyword)
        #TODO Note: 4/5 cc/aug-cc basis sets are available but we need extrapolation parameters
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
    else:
        print("Unknown basisfamily or cardinals chosen...")
        exit()
    return ccsdt_1_line,ccsdt_2_line



#If heavy element present and using cc/aug-cc basisfamily then add special PP-basis and ECP in block
def special_element_basis(fragment,cardinal,basisfamily,blocks):
    basis_dict = {('cc',2) : "cc-pVDZ-PP", ('aug-cc',2) : "aug-cc-pVDZ-PP", ('cc',3) : "cc-pVTZ-PP", ('aug-cc',3) : "aug-cc-pVTZ-PP", ('cc',4) : "cc-pVQZ-PP", ('aug-cc',4) : "aug-cc-pVQZ-PP"}
    auxbasis_dict = {('cc',2) : "cc-pVDZ-PP/C", ('aug-cc',2) : "aug-cc-pVDZ-PP/C", ('cc',3) : "cc-pVTZ-PP/C", ('aug-cc',3) : "aug-cc-pVTZ-PP/C", ('cc',4) : "cc-pVQZ-PP/C", ('aug-cc',4) : "aug-cc-pVQZ-PP/C"}
    for element in fragment.elems:
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


# For theory object with DLPNO, do 2 calculations with different DLPNO thresholds and extrapolate
def PNOExtrapolationStep(fragment=None, theory=None, pnoextrapolation=None, DLPNO=None, F12=None, calc_label=None):

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
    
    ash.Singlepoint(fragment=fragment, theory=theory)
    resultdict_X = interfaces.interface_ORCA.grab_HF_and_corr_energies(theory.filename+'.out', DLPNO=DLPNO,F12=F12)
    shutil.copyfile(theory.filename+'.out', './' + calc_label + '_PNOX' + '.out')
    print("resultdict_X:", resultdict_X)


    
    theory.orcablocks = PNOYblocks
    ash.Singlepoint(fragment=fragment, theory=theory)
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
