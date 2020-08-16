import ash
import shutil
import numpy as np
import functions_coords
from functions_general import *
import os
import math


#Reaction class. Used for benchmarking
class Reaction:
    def __init__(self, index, filenames, stoichiometry, refenergy, unit, correction=0.0):
        self.index = index
        self.filenames = filenames
        self.stoichiometry = stoichiometry
        #Reference reaction energies
        self.refenergy = refenergy
        self.unit =unit
        #Calculated reaction energy
        self.calcenergy = None
        #List of total energies of each species
        self.totalenergies = []
        #List of molecular formulas
        self.formulas= None
        # Correction to energy: e.g. ZPE. If not provided then 0.0
        self.correction = correction
        #Corrected energy and corrected error
        self.calcenergy_corrected = None
        self.error = None

#Read benchmark-set reference file ("Reference_data") inside indicated directory 
def read_referencedata_file(benchmarksetpath):
    #Open file and get database info as dict
    database_dict={}
    corrections=False
    count=0
    
    #with open(benchmarksetpath+"Reference_data.txt") as ref_file:
    if os.path.isfile(benchmarksetpath+'corrections.txt') is True:
        print("Found corrections.txt file. Grabbing corrections")
        corrections=True
        corrections_dict={}
        with open(benchmarksetpath+'corrections.txt') as correction_file:
            for line in correction_file:
                if '#' not in line:
                    rindex=int(line.split()[0])
                    corr = float(line.split()[1])
                    corrections_dict[rindex] = corr
    
    with open(benchmarksetpath+"Reference_data.txt") as ref_file:
        for line in ref_file:
            if '#TESTSET_INFO' in line:
                if 'Unit' in line:
                    unit=line.split()[-1]
                if 'Numentries' in line:
                    numentries=int(line.split()[-1])
            if '#' not in line:
                count+=1
                filenames=[]
                stoichiometry=[]
                #Getting index (first line)
                # Next entries are either strings (filename) or stoichiometry-indices (integers). 
                # Check if integer or float, if neither then assume filename string
                for i,word in enumerate(line.split()):
                    if i == 0:
                        index=int(word)
                    elif isint(word):
                        stoichiometry.append(int(word))
                    elif isfloat(word):
                        refenergy=float(line.split()[-1])
                    else:
                        filenames.append(word)
                #New reaction
                if corrections is True:
                    newreaction = Reaction(index, filenames, stoichiometry, refenergy, unit, correction=corrections_dict[index])                    
                else:
                    newreaction = Reaction(index, filenames, stoichiometry, refenergy, unit)
                
                
                #print("New reaction: ", newreaction.__dict__)
                #Add to dict.
                database_dict[index] = newreaction
        if count != index or count != numentries:
            print("Reaction lines does not match indices or number of entries in header. Mistake in file?!")
            exit()
    return database_dict

#Get pretty reaction string from molecule-filenames and stoichiometry
def get_reaction_string(filenames, stoichiometry):
    string =""
    #Index when sign changes from reactant to product
    index=stoichiometry.index(1)
    for i,file in enumerate(filenames):
        if i == index:
            #=>→ 
            string+=" ⟶   "
        string+=file
    return string

#run_benchmark
#Reuseorbs option: Reuse orbitals within same reaction. This only makes sense if reaction contains very similar geometries (e.g. IE/EA reaction)
def run_benchmark(set=None, theory=None, workflow=None, orcadir=None, numcores=None, reuseorbs=False, corrections=None):
    """[summary]

    Args:
        set ([type], optional): [description]. Defaults to None.
        theory ([type], optional): [description]. Defaults to None.
        workflow ([type], optional): [description]. Defaults to None.
        orcadir ([type], optional): [description]. Defaults to None.
        numcores ([type], optional): [description]. Defaults to None.
        reuseorbs (bool, optional): [description]. Defaults to False.
        corrections ([type], optional): [description]. Defaults to None.
    """
    print("")
    print("")
    print(BC.WARNING,"="*30,BC.END)
    print(BC.WARNING,"BENCHMARKING FUNCTION",BC.END)
    print(BC.WARNING,"="*30,BC.END)
    print("Dataset: ", set)
    print("Reuse orbitals option: ", reuseorbs)
    ashpath = os.path.dirname(ash.__file__)
    benchmarksetpath=ashpath+"/databases/Benchmarking-sets/"+set+"/data/"
    #Read reference data and define reactions
    print("")
    database_dict = read_referencedata_file(benchmarksetpath)
    print("")
    #One way of providing corrections: give list of floats
    if corrections is not None:
        print("Corrections provided as input: ", corrections)
        assert len(corrections) == len(database_dict), "Length of list corrections not matching length of test set"
        for i,corr in enumerate(corrections):
            database_dict[i+1].correction = corr
    
    
    #Always same unit
    unit=database_dict[1].unit
    try:
        os.mkdir("benchmarks_calcs")
    except FileExistsError:
        pass
    os.chdir("benchmarks_calcs")
    
    errors=[]
    
    #Dictionary of energies of calculated fragments so that we don't have to calculate same fragment multiple times
    all_calc_energies ={}
    
    for reactionindex in database_dict:
        reaction=database_dict[reactionindex]
        #TODO: Get longest reaction string here to make sure final is good
        #reactionstring=get_reaction_string(reaction.filenames, reaction.stoichiometry)
        
        print("")
        print("-"*70)
        print(BC.WARNING,"Reaction {} : {} {} ".format(reactionindex, BC.OKBLUE, reaction.filenames),BC.END)
        print(BC.WARNING,"Stoichiometry:", BC.OKBLUE,reaction.stoichiometry,BC.END)
        print(BC.WARNING,"Reference energy:", BC.OKBLUE,reaction.refenergy, unit, BC.END)
        print("-"*70)

        #Reading XYZ file and grabbing charge and multiplicity
        energies=[]
        for file in reaction.filenames:
            
            #If previously calculated fragment, grab energy from all_calc_energies and skip
            if file in all_calc_energies:
                print("File {} already calculated. Skipping calculation".format(file))
                energy = all_calc_energies[file]
                reaction.totalenergies.append(energy)
                energies.append(energy)
                continue
            
            frag = ash.Fragment(xyzfile=benchmarksetpath+file+'.xyz', readchargemult=True, conncalc=False)
            # Setting charge and mult for theory
            if theory is not None:
                print("theory:", theory)
                theory.charge=frag.charge
                theory.mult=frag.mult
                energy = ash.Singlepoint(fragment=frag, theory=theory)
                all_calc_energies[file] = energy
                reaction.totalenergies.append(energy)
                shutil.copyfile('orca-input.out', './' + file  + '.out')
                
                #If reuseorbs False (default) then delete ORCA files in each step
                #If True, keep file, including orca-input.gbw which enables Autostart
                if reuseorbs is False:
                    theory.cleanup()
                print("")
            elif workflow is not None:
                if orcadir is None:
                    print("Please provide orcadir variable to run_benchmark_set")
                    exit()
                energy, energydict = workflow(fragment=frag, charge=frag.charge, mult=frag.mult, orcadir=orcadir, numcores=numcores)
                all_calc_energies[file] = energy
                reaction.totalenergies.append(energy)
            #List of all energies
            energies.append(energy)
        print("")
        reaction_energy, error = ash.ReactionEnergy(stoichiometry=reaction.stoichiometry, list_of_energies=energies, unit='eV', label=reactionindex, 
                                                    reference=reaction.refenergy)
        reaction.calcenergy = reaction_energy
        reaction.calcenergy_corrected = reaction_energy + reaction.correction
        #Adding error with correction
        reaction.error = error + reaction.correction
        
        #Cleanup after reaction is done. Theory only.
        if theory is not None:
            theory.cleanup()
        
    print("")
    print(BC.WARNING,"="*70, BC.END)
    print(BC.WARNING,"FINAL RESULTS FOR TESTSET: ", BC.OKBLUE, set, BC.END)
    print(BC.WARNING,"="*70, BC.END)
    print("Unit:", unit)
    corrections = [database_dict[r].correction for r in database_dict]
    print("List of corrections applied:", corrections)
    print("")
    #Calculating errors (have been corrected)
    errors = [database_dict[r].error for r in database_dict]
    abserrors = [abs(i) for i in errors]
    MAE=sum(abserrors)/len(abserrors)
    ME=sum(errors)/len(errors)
    MaxError=max(errors, key=abs)
    RMSE=math.sqrt(sum([i**2 for i in errors])/len(errors))
    
    #Print nice table
    print(BC.WARNING, "{:7s} {:55s}  {:13s} {:13s} {:13s}   {:17s}".format("Index", "Reaction", "Ref.", "Calc.", "Calc.+corr.", "Error"), BC.END)
    print("-"*120)
    for rindex in database_dict:
        r=database_dict[rindex]
        if r.error == MaxError:
            colorcode=BC.FAIL
        else:
            colorcode=BC.END
        reactionstring=get_reaction_string(r.filenames, r.stoichiometry)
        #print(" {:<10} {:<40s}  {:<13.4f} {:<13.4f}{} {:<13.4f}{}".format(rindex, ' '.join(r.filenames), r.refenergy, r.calcenergy, colorcode, r.error,BC.END))
        print(" {:<7} {:<55s}  {:<13.4f} {:<13.4f} {:<13.4f}{} {:>8.4f}{}".format(rindex, reactionstring, r.refenergy, r.calcenergy, r.calcenergy_corrected, colorcode, r.error,BC.END))
    print("-"*120)
    print(" {:<10s} {:13.4f} {:<10s} ".format("MAE", MAE, unit))
    print(" {:<10s} {:13.4f} {:<10s} ".format("ME", ME, unit))
    print(" {:<10s} {:13.4f} {:<10s} ".format("RMSE", RMSE, unit))
    print(" {:<10s} {:13.4f} {:<10s} ".format("MaxError", MaxError, unit))
