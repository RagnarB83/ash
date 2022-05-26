"""
    Singlepoint module:

    Function Singlepoint

    class ZeroTheory
    """
import numpy as np
import time
#import ash
from ash.functions.functions_general import ashexit, BC,print_time_rel,print_line_with_mainheader
from ash.modules.module_coords import check_charge_mult

#Single-point energy function
def Singlepoint_gradient(fragment=None, theory=None, charge=None, mult=None):
    energy, gradient = Singlepoint(fragment=fragment, theory=theory, Grad=True, charge=charge, mult=mult)
    return energy, gradient

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
        print(BC.WARNING,"Doing single-point Energy+Gradient job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
        # An Energy+Gradient calculation where we change the number of cores to 12
        energy,gradient= theory.run(current_coords=coords, elems=elems, Grad=True, charge=charge, mult=mult)
        print("Energy: ", energy)
        print_time_rel(module_init_time, modulename='Singlepoint', moduleindex=1)
        return energy,gradient
    # Run a single-point energy job without gradient (default)
    else:
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
        return energy



#Single-point energy function that runs calculations on 1 fragment using multiple theories. Returns a list of energies.
#TODO: allow Grad option?
def Singlepoint_theories(theories=None, fragment=None, charge=None, mult=None):
    print_line_with_mainheader("Singlepoint_theories function")

    print("Will run single-point calculation on the fragment with multiple theories and return a list of energies")

    energies=[]

    #Looping through fragmengs
    for theory in theories:
        #Check charge/mult
        charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Singlepoint_theories", theory=theory)

        #Running single-point. 
        energy = ash.Singlepoint(theory=theory, fragment=fragment, charge=charge, mult=mult)
        
        print("Theory Label: {} Energy: {} Eh".format(theory.label, energy))
        theory.cleanup()
        energies.append(energy)

    #Printing final table
    print_theories_table(theories,energies,fragment)
    return energies

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
def print_fragments_table(fragments,energies,tabletitle="Singlepoint_fragments: "):
    print()
    print("="*70)
    print("{}Table of energies of each fragment:".format(tabletitle))
    print("="*70)
    print("{:10} {:<20} {:>7} {:>7} {:>20}".format("Formula", "Label", "Charge","Mult", "Energy(Eh)"))
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
def Singlepoint_fragments(theory=None, fragments=None, stoichiometry=None):
    print_line_with_mainheader("Singlepoint_fragments function")

    print("Will run single-point calculation on each fragment and return a list of energies")
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
        energy = ash.Singlepoint(theory=theory, fragment=frag, charge=charge, mult=mult)
        
        print("Fragment {} . Label: {} Energy: {} Eh".format(frag.formula, frag.label, energy))
        theory.cleanup()
        energies.append(energy)
        #Adding energy as the fragment attribute
        frag.set_energy(energy)
        print("")

    #Print table
    print_fragments_table(fragments,energies)
    
    #Printing reaction energy if stoichiometry was provided
    if stoichiometry != None:
        print("Stoichiometry provided:", stoichiometry)
        ReactionEnergy(list_of_energies=energies, stoichiometry=stoichiometry, list_of_fragments=fragments, unit='kcal/mol', label='ΔE')
    return energies


#Single-point energy function that runs calculations on multiple fragments. Returns a list of energies.
#Assuming fragments have charge,mult info defined.
def Singlepoint_fragments_and_theories(theories=None, fragments=None, stoichiometry=None):

    print_line_with_mainheader("Singlepoint_fragments_and_theories")

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

    if stoichiometry != None:
        print("Final reaction energies:")
        for elist,t in zip(all_energies,theories):
            ReactionEnergy(list_of_energies=elist, stoichiometry=stoichiometry, list_of_fragments=fragments, unit='kcal/mol', label='{}'.format(t.label))
    print()
    return all_energies


#Single-point energy function that runs calculations on an ASH reaction object
#Assuming fragments have charge,mult info defined.
def Singlepoint_reaction(theory=None, reaction=None, unit='kcal/mol'):
    print_line_with_mainheader("Singlepoint_reaction function")

    print("Will run single-point calculation on each fragment defined in reaction and return the reaction energy")
    print("Theory:", theory.__class__.__name__)

    #Looping through fragments defined in Reaction object
    for frag in reaction.fragments:
        
        #Running single-point
        energy = ash.Singlepoint(theory=theory, fragment=frag, charge=frag.charge, mult=frag.mult)
        print("Fragment {} . Label: {} Energy: {} Eh".format(frag.formula, frag.label, energy))
        theory.cleanup()
        reaction.energies.append(energy)
        #Adding energy as the fragment attribute
        frag.set_energy(energy)
        print("")

    #Print table
    print_fragments_table(reaction.fragments,reaction.energies, tabletitle="Singlepoint_reaction: ")

    reaction.calculate_reaction_energy(unit=unit)
    
    return reaction.reaction_energy


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
                        'eV' : 27.211386245988, 'cm-1' : 219474.6313702 }
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
            print("Check ")

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

