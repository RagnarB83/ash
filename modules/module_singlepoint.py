"""
    Singlepoint module:

    Function Singlepoint

    class ZeroTheory
    """
import numpy as np
import time
import ash
from ash.functions.functions_general import ashexit, BC,print_time_rel,print_line_with_mainheader
from ash.modules.module_coords import check_charge_mult
from ash.modules.module_results import ASH_Results
#from ash.modules.module_highlevel_workflows import ORCA_CC_CBS_Theory

#Single-point energy function
def Singlepoint_gradient(fragment=None, theory=None, charge=None, mult=None):
    result = Singlepoint(fragment=fragment, theory=theory, Grad=True, charge=charge, mult=mult)
    return result

#Single-point energy function
def Singlepoint(fragment=None, theory=None, Grad=False, charge=None, mult=None):
    """Singlepoint function: runs a single-point energy calculation using ASH theory and ASH fragment.

    Args:
        fragment (ASH fragment, optional): An ASH fragment. Defaults to None.
        theory (ASH theory, optional): Any valid ASH theory. Defaults to None.
        Grad (bool, optional): Do gradient or not Defaults to False.
        charge (int, optional): Specify charge of system. Overrides fragment charge information.
        mult (int, optional): Specify mult of system. Overrides fragment charge information.
        
    Returns:
        float: Energy
        or
        float,np.array : Energy and gradient array
    """
    print_line_with_mainheader("Singlepoint function")
    module_init_time=time.time()
    if fragment is None or theory is None:
        print(BC.FAIL,"Singlepoint requires a fragment and a theory object",BC.END)
        ashexit()
    coords=fragment.coords
    elems=fragment.elems

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Singlepoint", theory=theory)


    # Run a single-point energy job with gradient
    if Grad ==True:
        print()
        print(BC.WARNING,"Doing single-point Energy+Gradient job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
        # An Energy+Gradient calculation where we change the number of cores to 12
        energy,gradient= theory.run(current_coords=coords, elems=elems, Grad=True, charge=charge, mult=mult)
        print("Energy: ", energy)
        print_time_rel(module_init_time, modulename='Singlepoint', moduleindex=1)
        result = ASH_Results(label="Singlepoint", energy=energy, gradient=gradient, charge=charge, mult=mult)
        return result
    # Run a single-point energy job without gradient (default)
    else:
        print()
        print("Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label))
        #Run
        energy = theory.run(current_coords=coords, elems=elems, charge=charge, mult=mult)

        #Previously some theories like CC_CBS_Theory returned energy and componentsdict as a tuple
        #TODO: This can probably be deleted soon.
        if type(energy) is tuple:
            componentsdict=energy[1]
            energy=energy[0]

        print("Energy: ", energy)
        #Now adding total energy to fragment
        fragment.set_energy(energy)
        print_time_rel(module_init_time, modulename='Singlepoint', moduleindex=1)
        result = ASH_Results(label="Singlepoint", energy=energy, charge=charge, mult=mult)
        return result



#Single-point energy function that runs calculations on 1 fragment using multiple theories. Returns a list of energies.
#TODO: allow Grad option?
def Singlepoint_theories(theories=None, fragment=None, charge=None, mult=None):
    print_line_with_mainheader("Singlepoint_theories function")
    module_init_time=time.time()
    print("Will run single-point calculation on the fragment with multiple theories")

    energies=[]

    #Looping through fragmengs
    for theory in theories:
        #Check charge/mult
        charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Singlepoint_theories", theory=theory)

        #Running single-point. 
        result = Singlepoint(theory=theory, fragment=fragment, charge=charge, mult=mult)
        
        print("Theory Label: {} Energy: {} Eh".format(theory.label, result.energy))
        theory.cleanup()
        energies.append(result.energy)

    #Printing final table
    print_theories_table(theories,energies,fragment)
    result = ASH_Results(label="Singlepoint_theories", energies=energies, charge=charge, mult=mult)
    print_time_rel(module_init_time, modulename='Singlepoint_theories', moduleindex=1)
    return result

#Pretty table of fragments and theories
def print_theories_table(theories,energies,fragment):
    print()
    print("="*70)
    print("Singlepoint_theories: Table of energies of each theory:")
    print("="*70)

    print("\n{:15} {:15} {:>7} {:>7} {:>20}".format("Theory class", "Theory Label", "Charge","Mult", "Energy(Eh)"))
    print("-"*70)
    for t, e in zip(theories,energies):
        print("{:15} {:15} {:>7} {:>7} {:>20.10f}".format(t.__class__.__name__, str(t.label), fragment.charge, fragment.mult, e))
    print()

#Pretty table of fragments and energies
def print_fragments_table(fragments,energies,tabletitle="Singlepoint_fragments: ", unit='Eh'):
    print()
    print("="*70)
    print("{}Table of energies of each fragment:".format(tabletitle))
    print("="*70)
    print("{:10} {:<20} {:>7} {:>7} {:>20}".format("Formula", "Label", "Charge","Mult", f"Energy({unit})"))
    print("-"*70)
    for frag, e in zip(fragments,energies):
        if frag.label==None:
            label="None"
        else:
            label=frag.label
        print("{:10} {:<20} {:>7} {:>7} {:>20.10f}".format(frag.formula, label, frag.charge, frag.mult, e))
    print()

#Single-point energy function that runs calculations on multiple fragments. Returns a list of energies.
#Assuming fragments have charge,mult info defined.
#If stoichiometry provided then print reaction energy
def Singlepoint_fragments(theory=None, fragments=None, stoichiometry=None, relative_energies=False, unit='kcal/mol'):
    print_line_with_mainheader("Singlepoint_fragments function")
    module_init_time=time.time()
    print("Will run single-point calculation on each fragment")
    print("Theory:", theory.__class__.__name__)

    energies=[];filenames=[]

    #Looping through fragments
    for frag in fragments:

        if frag.charge == None or frag.mult == None:
            print(BC.FAIL,"Error: Singlepoint_fragments requires charge/mult information to be associated with each fragment.", BC.END)
            ashexit()
        #Setting charge/mult  from fragment
        charge=frag.charge; mult=frag.mult
        
        #Running single-point
        result = Singlepoint(theory=theory, fragment=frag, charge=charge, mult=mult)
        
        print("Fragment {} . Label: {} Energy: {} Eh".format(frag.formula, frag.label, result.energy))
        theory.cleanup()
        energies.append(result.energy)
        #Adding energy as the fragment attribute
        frag.set_energy(result.energy)
        print("")

    #Create Results object
    result = ASH_Results(label="Singlepoint_fragments", energies=energies, charge=charge, mult=mult)   

    #Print table
    print_fragments_table(fragments,energies)

    #Print table
    if relative_energies is True:
        print()
        print("relative_energies option is True!")
        conversionfactor = { 'kcal/mol' : 627.50946900, 'kcalpermol' : 627.50946900, 'kJ/mol' : 2625.499638, 'kJpermol' : 2625.499638, 
                        'eV' : 27.211386245988, 'cm-1' : 219474.6313702, 'Eh' : 1.0, 'mEh' : 1000, 'meV' : 27211.386245988 }
        convfactor=conversionfactor[unit]
        relenergies=[(i - min(energies)) * convfactor for i in energies]
        print_fragments_table(fragments,relenergies, unit=unit)
        result.relative_energies = relenergies
        result.labels = [f.label for f in fragments]
 
    #Printing reaction energy if stoichiometry was provided
    if stoichiometry != None:
        print("Stoichiometry provided:", stoichiometry)
        r = ReactionEnergy(list_of_energies=energies, stoichiometry=stoichiometry, list_of_fragments=fragments, unit='kcal/mol', label='ΔE')
        result.reaction_energy = r[0]
    print_time_rel(module_init_time, modulename='Singlepoint_fragments', moduleindex=1)
    return result


#Single-point energy function that runs calculations on multiple fragments. Returns a list of energies.
#Assuming fragments have charge,mult info defined.
def Singlepoint_fragments_and_theories(theories=None, fragments=None, stoichiometry=None):

    print_line_with_mainheader("Singlepoint_fragments_and_theories")
    module_init_time=time.time()
    #List of lists
    all_energies=[]

    #Looping over theories and getting energies for list of fragments
    for theory in theories:
        energies=Singlepoint_fragments(theory=theory, fragments=fragments, stoichiometry=stoichiometry )
        all_energies.append(energies)

    print("\n")
    print("SINGLEPOINT_FRAGMENTS_AND_THEORIES ALL DONE")
    print("\n")
    print("="*60)
    print("Singlepoint_fragments_and_theories: FINAL RESULTS")
    print("="*60)
    #Table
    for t,elist in zip(theories,all_energies):
        print("\nTheory:", t.__class__.__name__)
        print("Label:", t.label)
        print_fragments_table(fragments,energies,tabletitle="")
        #Reaction energy if stoichiometry provided
        if stoichiometry != None:
            print("Stoichiometry provided:", stoichiometry)
            ReactionEnergy(list_of_energies=elist, stoichiometry=stoichiometry, list_of_fragments=fragments, unit='kcal/mol', label='{}'.format(t.label))

                
            print("_"*60)
    print("\nFinal list of list of total energies:", all_energies)

    result = ASH_Results(label="Singlepoint_fragments_and_theories", energies=all_energies)
    if stoichiometry != None:
        print("Final reaction energies:")
        for elist,t in zip(all_energies,theories):
            r = ReactionEnergy(list_of_energies=elist, stoichiometry=stoichiometry, list_of_fragments=fragments, unit='kcal/mol', label='{}'.format(t.label))
            result.reaction_energies.append(r[0])
    print()
    #return all_energies
    print_time_rel(module_init_time, modulename='Singlepoint_fragments_and_theories', moduleindex=1)
    return result


#Single-point energy function that runs calculations on an ASH reaction object
#Assuming fragments have charge,mult info defined.
def Singlepoint_reaction(theory=None, reaction=None, unit=None, orbitals_stored=None):
    print_line_with_mainheader("Singlepoint_reaction function")
    module_init_time=time.time()

    print("Will run single-point calculation on each fragment defined in reaction")
    print("Theory:", theory.__class__.__name__)
    print("Resetting energies in reaction object")
    reaction.energies=[]
    reaction.reset_energies()

    #Looping through fragments defined in Reaction object
    list_of_componentsdicts=[]
    componentsdict={}
    for i,frag in enumerate(reaction.fragments):
        #Orbital file for ORCATheory
        try:
            theory.moreadfile=reaction.orbital_dictionary[orbitals_stored][i]
        except:
            pass
        #Running single-point
        result = Singlepoint(theory=theory, fragment=frag, charge=frag.charge, mult=frag.mult)
        energy = result.energy
        print("Fragment {} . Label: {} Energy: {} Eh".format(frag.formula, frag.label, energy))
        theory.cleanup()
        reaction.energies.append(energy)
        #Check if ORCATheory object contains ICE-CI info
        if isinstance(theory,ash.ORCATheory):
            print("theory.properties:", theory.properties)
            #Add selected properties to Reaction object
            try:
                reaction.properties["E_var"].append(theory.properties["E_var"])
                reaction.properties["E_PT2_rest"].append(theory.properties["E_PT2_rest"])
                reaction.properties["num_genCFGs"].append(theory.properties["num_genCFGs"])
                reaction.properties["num_selected_CFGs"].append(theory.properties["num_selected_CFGs"])
                reaction.properties["num_after_SD_CFGs"].append(theory.properties["num_after_SD_CFGs"])
            except:
                pass
        # Keeping CC components if using ORCA_CC_CBS_Theory
        if isinstance(theory,ash.ORCA_CC_CBS_Theory):
            componentsdict=theory.energy_components
            list_of_componentsdicts.append(componentsdict)
        #Adding energy as the fragment attribute
        frag.set_energy(energy)
        print("")

    #Print table
    print_fragments_table(reaction.fragments,reaction.energies, tabletitle="Singlepoint_reaction: ")

    #Setting unit of reaction if given (will override reaction.unit definition)
    #NOTE: Needed?
    if unit is not None:
        reaction.unit=unit
    
    reaction.calculate_reaction_energy()
    
    result = ASH_Results(label="Singlepoint_reaction", energies=reaction.energies,
        reaction_energy=reaction.reaction_energy)

    if isinstance(theory,ash.ORCA_CC_CBS_Theory):
        print("-"*70)
        print("CCSD(T)/CBS components")
        print("-"*70)
        finaldict={}
        #Contributions to CCSD(T) energies
        if 'E_SCF_CBS' in componentsdict:
            scf_parts=[d['E_SCF_CBS'] for d in list_of_componentsdicts]
            deltaSCF=ReactionEnergy(stoichiometry=reaction.stoichiometry, list_of_fragments=reaction.fragments, list_of_energies=scf_parts, unit=unit, label='ΔSCF')[0]
            finaldict['deltaSCF']=deltaSCF
        if 'E_corrCCSD_CBS' in componentsdict:
            ccsd_parts=[d['E_corrCCSD_CBS'] for d in list_of_componentsdicts]
            delta_CCSDcorr=ReactionEnergy(stoichiometry=reaction.stoichiometry, list_of_fragments=reaction.fragments, list_of_energies=ccsd_parts, unit=unit, label='ΔCCSD')[0]
            finaldict['delta_CCSDcorr']=delta_CCSDcorr
        if 'E_corrCCT_CBS' in componentsdict:
            triples_parts=[d['E_corrCCT_CBS'] for d in list_of_componentsdicts]
            delta_Tcorr=ReactionEnergy(stoichiometry=reaction.stoichiometry, list_of_fragments=reaction.fragments, list_of_energies=triples_parts, unit=unit, label='Δ(T)')[0]
            finaldict['delta_Tcorr']=delta_Tcorr
        if 'E_corr_CBS' in componentsdict:
            valencecorr_parts=[d['E_corr_CBS'] for d in list_of_componentsdicts]
            delta_CC_corr=ReactionEnergy(stoichiometry=reaction.stoichiometry, list_of_fragments=reaction.fragments, list_of_energies=valencecorr_parts, unit=unit, label='ΔCCSD+Δ(T) corr')[0]
            finaldict['delta_CC_corr']=delta_CC_corr
        if 'E_SO' in componentsdict:
            SO_parts=[d['E_SO'] for d in list_of_componentsdicts]
            delta_SO_corr=ReactionEnergy(stoichiometry=reaction.stoichiometry, list_of_fragments=reaction.fragments, list_of_energies=SO_parts, unit=unit, label='ΔSO')[0]
            finaldict['delta_SO_corr']=delta_SO_corr
        if 'E_corecorr_and_SR' in componentsdict:
            CV_SR_parts=[d['E_corecorr_and_SR'] for d in list_of_componentsdicts]
            delta_CVSR_corr=ReactionEnergy(stoichiometry=reaction.stoichiometry, list_of_fragments=reaction.fragments, list_of_energies=CV_SR_parts, unit=unit, label='ΔCV+SR')[0]
            finaldict['delta_CVSR_corr']=delta_CVSR_corr
        if 'T1energycorr' in componentsdict:
            T1corr_parts=[d['T1energycorr'] for d in list_of_componentsdicts]
            delta_T1_corr=ReactionEnergy(stoichiometry=reaction.stoichiometry, list_of_fragments=reaction.fragments, list_of_energies=T1corr_parts, unit=unit, label='ΔΔT1corr')[0]
            finaldict['delta_T1_corr']=delta_T1_corr
        if 'E_FCIcorrection' in componentsdict:
            fcicorr_parts=[d['E_FCIcorrection'] for d in list_of_componentsdicts]
            delta_FCI_corr=ReactionEnergy(stoichiometry=reaction.stoichiometry, list_of_fragments=reaction.fragments, list_of_energies=fcicorr_parts, unit=unit, label='ΔFCIcorr')[0]
            finaldict['delta_FCI_corr']=delta_FCI_corr
        result.energy_contributions = finaldict
    print_time_rel(module_init_time, modulename='Singlepoint_reaction', moduleindex=1)
    return result
    #return reaction.reaction_energy


#Single-point energy function that communicates via fragment
#NOTE: NOT SURE IF WE WANT TO GO THIS ROUTE
def newSinglepoint(fragment=None, theory=None, Grad=False):
    """Singlepoint function: runs a single-point energy calculation using ASH theory and ASH fragment.

    Args:
        fragment (ASH fragment, optional): An ASH fragment. Defaults to None.
        theory (ASH theory, optional): Any valid ASH theory. Defaults to None.
        Grad (bool, optional): Do gradient or not Defaults to False.

    Returns:
        float: Energy
        or
        float,np.array : Energy and gradient array
    """
    ashexit()
    module_init_time=time.time()
    print("")
    if fragment is None or theory is None:
        print(BC.FAIL,"Singlepoint requires a fragment and a theory object",BC.END)
        ashexit()
    
    #Case QM/MM: we don't pass whole fragment?
    if isinstance(theory, ash.QMMMTheory):
        print("this is QM/MM. not ready")
        ashexit()
    #Regular single-point
    else:
        # Run a single-point energy job with gradient
        if Grad ==True:
            print(BC.WARNING,"Doing single-point Energy+Gradient job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
            # An Energy+Gradient calculation
            energy,gradient= theory.run(fragment=fragment, Grad=True)
            print("Energy: ", energy)
            print_time_rel(module_init_time, modulename='Singlepoint', moduleindex=1)
            return energy,gradient
        # Run a single-point energy job without gradient (default)
        else:
            print(BC.WARNING,"Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
            #energy = theory.run(current_coords=coords, elems=elems)
            energy = theory.run(fragment=fragment)
            print("Energy: ", energy)
            #Now adding total energy to fragment
            fragment.energy=energy
            print_time_rel(module_init_time, modulename='Singlepoint', moduleindex=1)
            return energy




# Theory object that always gives zero energy and zero gradient. Useful for setting constraints
class ZeroTheory:
    def __init__(self, fragment=None, printlevel=None, numcores=1, label=None):
        """Class Zerotheory: Simple dummy theory that gives zero energy and a zero-valued gradient array
            Note: includes unnecessary attributes for consistency.

        Args:
            fragment (ASH fragment, optional): A valid ASH fragment. Defaults to None.
            printlevel (int, optional): Printlevel:0,1,2 or 3. Defaults to None.
            numcores (int, optional): Number of cores. Defaults to 1.
            label (str, optional): String label. Defaults to None.
        """
        self.numcores=numcores
        self.printlevel=printlevel
        self.label=label
        self.fragment=fragment
        self.filename="zerotheory"
        #Indicate that this is a QMtheory
        self.theorytype="QM"
    def run(self, current_coords=None, elems=None, Grad=False, PC=False, numcores=None, charge=None, mult=None, label=None ):
        self.energy = 0.0
        #Gradient as np array 
        self.gradient = np.zeros((len(elems), 3))
        if Grad==False:
            return self.energy
        else:
            return self.energy,self.gradient



def ReactionEnergy(list_of_energies=None, stoichiometry=None, list_of_fragments=None, unit='kcal/mol', label=None, reference=None, silent=False):
    """Calculate reaction energy from list of energies (or energies from list of fragments) and stoichiometry

    Args:
        list_of_energies ([type], optional): A list of total energies in hartrees. Defaults to None.
        stoichiometry (list, optional): A list of integers, e.g. [-1,-1,1,1]. Defaults to None.
        list_of_fragments (list, optional): A list of ASH fragments . Defaults to None.
        unit (str, optional): Unit for relative energy. Defaults to 'kcal/mol'.
        label (string, optional): Optional label for energy. Defaults to None.
        reference (float, optional): Optional shift-parameter of energy Defaults to None.

    Returns:
        tuple : energy and error in chosen unit
    """
    conversionfactor = { 'kcal/mol' : 627.50946900, 'kcalpermol' : 627.50946900, 'kJ/mol' : 2625.499638, 'kJpermol' : 2625.499638, 
                        'eV' : 27.211386245988, 'cm-1' : 219474.6313702, 'Eh' : 1.0, 'mEh' : 1000, 'meV' : 27211.386245988 }
    if label is None:
        label=''
    reactant_energy=0.0 #hartree
    product_energy=0.0 #hartree
    if stoichiometry is None:
        print("stoichiometry list is required")
        ashexit()


    #List of energies option
    if list_of_energies is not None:

        if len(list_of_energies) != len(stoichiometry):
            print("Number of energies not equal to number of stoichiometry values")
            print("Exiting.")
            ashexit()

        for i,stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy=reactant_energy+list_of_energies[i]*abs(stoich)
            if stoich > 0:
                product_energy=product_energy+list_of_energies[i]*abs(stoich)
        reaction_energy=(product_energy-reactant_energy)*conversionfactor[unit]
        if reference is None:
            error=None
            if silent is False:
                print(BC.BOLD, "Reaction_energy({}): {} {} {}".format(label,BC.OKGREEN,reaction_energy, unit), BC.END)
        else:
            error=reaction_energy-reference
            if silent is False:
                print(BC.BOLD, "Reaction_energy({}): {} {} {} (Error: {})".format(label,BC.OKGREEN,reaction_energy, unit, error), BC.END)
    else:
        print("\nNo list of total energies provided. Using internal energy of each fragment instead.")
        print("")
        for i,stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy=reactant_energy+list_of_fragments[i].energy*abs(stoich)
            if stoich > 0:
                product_energy=product_energy+list_of_fragments[i].energy*abs(stoich)
        reaction_energy=(product_energy-reactant_energy)*conversionfactor[unit]
        if reference is None:
            error=None
            if silent is False:
                print(BC.BOLD, "Reaction_energy({}): {} {} {}".format(label,BC.OKGREEN,reaction_energy, unit), BC.END)
        else:
            error=reaction_energy-reference
            if silent is False:
                print(BC.BOLD, "Reaction_energy({}): {} {} {} (Error: {})".format(label,BC.OKGREEN,reaction_energy, unit, error), BC.END)
    return reaction_energy, error

