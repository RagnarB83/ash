import os
from pathlib import Path
from ash.functions.functions_general import ashexit,writestringtofile
import shutil
from ash.modules.module_coords import check_charge_mult
from ash import Singlepoint


# interface to forcebalance

class ForceBalance:
    def __init__(self,theory=None, fragment=None, charge=None, mult=None, 
                 inputfilename="input.in", jobtype="newton",
                 forcefield_name="systemFF.xml", target_name="system_XX",
                 target_type="Interaction_OpenMM",
                 convergence_objective=0.01,
                 penalty_additive=0.01,trust0=1.0, backup=False):

        if fragment.charge is None or fragment.mult is None:
            print("Charge and multiplicity must be defined inside Fragments for ForceBalance")
            ashexit()

        self.theory=theory
        self.fragment=fragment
        self.inputfilename=inputfilename
        self.forcefield_name=forcefield_name

        # target type: Interaction_OpenMM, AbInitio_OpenMM

        inputfilestring=f"""$options
 jobtype {jobtype}
 forcefield {forcefield_name}
 penalty_additive {penalty_additive}
 convergence_objective {convergence_objective}
 trust0 {trust0}
 backup {backup}
 $end
 
 $target
 name {target_name}
 type {target_type}
 $end
 """
        writestringtofile(inputfilestring,self.inputfilename)
        # Delete old structure
        for d in ["forcefield","targets","temp","result"]:
            if os.path.exists(d):
                shutil.rmtree(d)
        os.mkdir("forcefield")
        os.mkdir("targets")
        os.mkdir("temp")
        os.mkdir("result")


    def run(self, debug=False, continue_=False):

        # Create XML file for molecule
        print("Now creating XML file")
        # Create list of atomnames, used in PDB topology and XML file
        atomnames_full=[j+str(i) for i,j in enumerate(self.fragment.elems)]
        xmlfile = write_xmlfile(filename=self.forcefield_name, resnames=["DUM"], atomnames_per_res=[atomnames_full], atomtypes_per_res=[self.fragment.elems],
                                        elements_per_res=[self.fragment.elems], masses_per_res=[self.fragment.masses],
                                        charges_per_res=[[0.0]*self.fragment.numatoms],sigmas_per_res=[[0.0]*self.fragment.numatoms],
                                        coulomb14scale=0.833333,
                                        epsilons_per_res=[[0.0]*self.fragment.numatoms], skip_nb=False)

        # Move system.xml file to forcefield directory
        shutil.move(self.forcefield_name, f"forcefield/{self.forcefield_name}")

        # Run Theory to get energy and gradient
        result = Singlepoint(theory=self.theory, fragment=self.fragment, Grad=True)
        energy = result.energy
        gradient = result.gradient
        # Quantum data file
        with open("qdata.txt","w") as f:
            f.write("JOB 0\n")
            f.write(f"ENERGY {energy}\n")
            f.write("FORCES ")
            for g in gradient:
                for gg in g:
                    f.write(f"{gg} ")
            f.write("\n\n")

        from forcebalance.parser import parse_inputs
        from forcebalance.forcefield import FF
        from forcebalance.objective import Objective
        from forcebalance.optimizer import Optimizer
        try:
            # The general options and target options that come from parsing the input file
            options, tgt_opts = parse_inputs(self.inputfilename)
            # Set the continue_ option.
            if continue_: options['continue'] = True
            # The FF, objective function and optimizer
            forcefield  = FF(options)
            objective   = Objective(options, tgt_opts, forcefield)
            optimizer   = Optimizer(options, objective, forcefield)
            optimizer.Run()
        except:
            import traceback
            traceback.print_exc()
            if debug:
                import pdb
                pdb.post_mortem()


################################
# General XML-writing function.
################################
def write_xmlfile(resnames=None, atomnames_per_res=None, atomtypes_per_res=None, elements_per_res=None, connectivity_dict=None,
                            masses_per_res=None, charges_per_res=None, sigmas_per_res=None,
                            epsilons_per_res=None, filename="system.xml", 
                            coulomb14scale=0.833333, lj14scale=0.5, charmm=False
                            skip_nb=False, skip_harmb=False, skip_harma=False, skip_torsion=False):
    
    print("Inside write_xmlfile")
    assert len(resnames) == len(atomnames_per_res) == len(atomtypes_per_res)

    ################################
    # BONDS and CONNECTIVITY
    if connectivity_dict is None:
        print("Error: connectivity_dict need to be defined")
        ashexit()

    # Creating simple unique list of lists of bonds from connectivity_dict
    bonds=[]
    for k, v in connectivity_dict.items():
        for i in v:
            if i > k:
                bonds.append([k, i])
    print("bonds:",bonds)
    ##################################



    ########################
    bondlines=[]
    anglelines=[]
    torsionlines=[]

    #TODO: need to look-up bond parameters from some dict that has been guessed or something

    for bond in bonds:
        classA="XX"
        classB="YY"
        r_eq=0.18210
        k=188949.44000
        bondforce = f"<Bond class1=\"{classA}\" class2=\"{classB}\" length=\"{r_eq}\" k=\"{k}\"/>\n"
        print(bondforce)
        bondlines.append(bondforce)
    print(bondlines)
    anglelines=None
    torsionlines=None

    #TODO: Parameterize (this is for Forcebalance)
    #NOTE: need to add parameterize to user-chosen Nonbonded atomtype lines, bondforce lines, angleforce lines and periodicforce lines

    # Create list of all AtomTypelines (unique)
    atomtypelines = []
    for resname, atomtypelist, elemlist, masslist in zip(resnames, atomtypes_per_res, elements_per_res, masses_per_res):
        for atype, elem, mass in zip(atomtypelist, elemlist, masslist):
            atomtypeline = "<Type name=\"{}\" class=\"{}\" element=\"{}\" mass=\"{}\"/>\n".format(atype, atype, elem,
                                                                                                  str(mass))
            if atomtypeline not in atomtypelines:
                atomtypelines.append(atomtypeline)
    

    # Create list of all nonbonded lines (unique)
    nonbondedlines = []
    LJforcelines = []
    for resname, atomtypelist, chargelist, sigmalist, epsilonlist in zip(resnames, atomtypes_per_res, charges_per_res,
                                                                         sigmas_per_res, epsilons_per_res):
        for atype, charge, sigma, epsilon in zip(atomtypelist, chargelist, sigmalist, epsilonlist):
            if charmm:
                #LJ parameters zero here
                nonbondedline = "<Atom type=\"{}\" charge=\"{}\" sigma=\"{}\" epsilon=\"{}\"/>\n".format(atype, charge,0.0, 0.0)
                #Here we set LJ parameters
                ljline = "<Atom type=\"{}\" sigma=\"{}\" epsilon=\"{}\"/>\n".format(atype, sigma, epsilon)
                if nonbondedline not in nonbondedlines:
                    nonbondedlines.append(nonbondedline)
                if ljline not in LJforcelines:
                    LJforcelines.append(ljline)
            else:
                nonbondedline = "<Atom type=\"{}\" charge=\"{}\" sigma=\"{}\" epsilon=\"{}\"/>\n".format(atype, charge,
                                                                                                        sigma, epsilon)
                if nonbondedline not in nonbondedlines:
                    nonbondedlines.append(nonbondedline)

    # WRITE XML-file
    with open(filename, 'w') as xmlfile:
        xmlfile.write("<ForceField>\n")
        xmlfile.write("<AtomTypes>\n")
        for atomtypeline in atomtypelines:
            xmlfile.write(atomtypeline)
        xmlfile.write("</AtomTypes>\n")
        xmlfile.write("<Residues>\n")
        #Looping over residues
        for resname, atomnamelist, atomtypelist in zip(resnames, atomnames_per_res, atomtypes_per_res):
            xmlfile.write("<Residue name=\"{}\">\n".format(resname))
            #Looping over atoms
            for i, (atomname, atomtype) in enumerate(zip(atomnamelist, atomtypelist)):
                xmlfile.write("<Atom name=\"{}\" type=\"{}\"/>\n".format(atomname, atomtype))
            #Adding possible bonds
            if bonds is not None:
                print("Writing bonds")
                #<Bond from="0" to="1"/>
                for b in bonds:
                    xmlfile.write(f"<Bond from=\"{b[0]}\" to=\"{b[1]}\"/>\n")


            xmlfile.write("</Residue>\n")
        xmlfile.write("</Residues>\n")
        # HARMONIC BOND FORCE
        if skip_harmb is False:
            xmlfile.write("<HarmonicBondForce>\n")
            for bondline in bondlines:
                xmlfile.write(bondline)
            xmlfile.write("</HarmonicBondForce>\n")
        # HARMONIC ANGLE FORCE
        if skip_harma is False:
            xmlfile.write("<HarmonicAngleForce>\n")
            for angleline in anglelines:
                xmlfile.write(angleline)
            xmlfile.write("</HarmonicAngleForce>\n")
        # PERIODIC TORSION FORCE
        if skip_torsion is False:
            xmlfile.write("<PeriodicTorsionForce>\n")
            for torsionline in torsionlines:
                xmlfile.write(torsionline)
            xmlfile.write("</PeriodicTorsionForce>\n")
        if skip_nb is False:
            if charmm:
                #Writing both Nonbnded force block and also LennardJonesForce block
                xmlfile.write("<NonbondedForce coulomb14scale=\"{}\" lj14scale=\"{}\">\n".format(coulomb14scale, lj14scale))
                for nonbondedline in nonbondedlines:
                    xmlfile.write(nonbondedline)
                xmlfile.write("</NonbondedForce>\n")
                xmlfile.write("<LennardJonesForce lj14scale=\"{}\">\n".format(lj14scale))
                for ljline in LJforcelines:
                    xmlfile.write(ljline)
                xmlfile.write("</LennardJonesForce>\n")
            else:
                #Only NonbondedForce block
                xmlfile.write("<NonbondedForce coulomb14scale=\"{}\" lj14scale=\"{}\">\n".format(coulomb14scale, lj14scale))
                for nonbondedline in nonbondedlines:
                    xmlfile.write(nonbondedline)
                xmlfile.write("</NonbondedForce>\n")
        xmlfile.write("</ForceField>\n")
    print("Wrote XML-file:", filename)
    return filename
