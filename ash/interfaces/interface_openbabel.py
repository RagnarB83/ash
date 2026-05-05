import subprocess as sp
import os
import shutil
import time
import numpy as np

from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
import ash.settings_ash
from ash.modules.module_coords import reformat_element

# Interface to OpenBabel for running implemented theories (e.g. UFF)
# TODO: Move other OpenBabel functionality to this file

class OpenBabelTheory():
    def __init__(self, forcefield="UFF", chargemodel=None, label="OpenBabelTheory", 
                 printlevel=2, user_atomcharges=None):
        self.label = label
        self.printlevel = printlevel
        self.theorytype = 'QM'
        self.theorynamelabel = 'OpenBabel'
        self.forcefield=forcefield #UFF, GAFF, MMF94, Ghemical, etc. See https://openbabel.org/docs/dev/Forcefields/FF.html for options
        self.chargemodel=chargemodel #gasteiger, mmff94, qeq, qtpie. See https://openbabel.org/docs/dev/Forcefields/ChargeModels.html for options

        self.user_atomcharges=user_atomcharges
        from openbabel import openbabel as ob
        from openbabel import pybel

    def cleanup(self):
        print("No cleanup implemented")

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time = time.time()
        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)

        from openbabel import openbabel as ob
        from openbabel import pybel

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        # Create an OBMol object and populate it with the current geometry
        mol = ob.OBMol()
        for elem, coord in zip(qm_elems, current_coords):
            # Create the atom object
            atom = mol.NewAtom() 
            atomic_num = ob.GetAtomicNum(elem)
            atom.SetAtomicNum(atomic_num)
            atom.SetVector(coord[0], coord[1], coord[2])

        print("Determining connectivity and bond orders for FF...")
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()

        # Turn off auto charges
        #mol.SetAutomaticPartialCharge(False)
        #mol.SetPartialChargesPerceived()
        #mol.SetAutomaticFormalCharge(False)

        def print_charges(mol):
            # Print charges for each atom
            for i in range(1, mol.NumAtoms() + 1):
                atom = mol.GetAtom(i)
                # Get charge from your dict, default to 0.0 if not found
                charge = atom.GetPartialCharge()
                print(f"Atom {i} charge: {charge}")
        def set_charges(mol,usercharges):
            # Set charges for each atom
            for i in range(1, mol.NumAtoms() + 1):
                atom = mol.GetAtom(i)
                atom.SetPartialCharge(usercharges[i-1])
        print("Initial charges in mol (before applying any charge model):")
        print_charges(mol)

        # Charge model
        if self.chargemodel is not None:
            print("Charge model is active")
            print("Charge model is currently disabled")
            #ashexit()
            if self.user_atomcharges is not None:
                print("Setting charges to user-atomcharges")
                set_charges(mol,self.user_atomcharges)
            else:
                self.cm = ob.OBChargeModel.FindType(self.chargemodel)
                print("Computing charges using OpenBabel charge model:", self.chargemodel)
                success = self.cm.ComputeCharges(mol)
                if not success:
                    raise RuntimeError("Failed to compute charges")
            print("Charges (after applying charge model):")
            print_charges(mol)
            self.ff = ob.OBForceField.FindForceField(self.forcefield)
            self.ff.Setup(mol)
            self.ff.GetPartialCharges(mol)
            # NOTE: still not working
        else:
            print("No chargemodel is active")
            self.ff = ob.OBForceField.FindForceField(self.forcefield)
            success = self.ff.Setup(mol)

        print("Computing regular FF energy:")
        self.energy = self.ff.Energy() / ash.constants.hartokj 
        print("FF energy:", self.energy)
        #elec_energy = self.ff.E_Electrostatic()
        #print("Electrostatic energy:", elec_energy)
        if Grad:
            self.gradient = np.zeros((len(qm_elems), 3))
            for i in range(len(qm_elems)):
                atom = mol.GetAtom(i + 1)
                f = self.ff.GetGradient(atom)
                self.gradient[i, 0] = f.GetX()*-1
                self.gradient[i, 1] = f.GetY()*-1
                self.gradient[i, 2] = f.GetZ()*-1
            self.gradient = self.gradient * 0.00020155
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        # Returning energy and gradient
        if Grad is True:
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            return self.energy, self.gradient
        # Returning energy without gradient
        else:
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            return self.energy
        

###################################
# Other Openbabel functionality
###################################

#Function to convert Mol file to PDB-file via OpenBabel
def mol_to_pdb(file):
    #OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: mol_to_pdb requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("mol", file))
    mol.write(format='pdb', filename=os.path.splitext(file)[0]+'.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0]+'.pdb')
    return os.path.splitext(file)[0]+'.pdb'

#Function to convert SDF file to PDB-file via OpenBabel
def sdf_to_pdb(file):
    #OpenBabel
    try:
        from openbabel import openbabel
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: sdf_to_pdb requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("sdf", file))

    #Write do disk as PDB-file
    mol.write(format='pdb', filename=os.path.splitext(file)[0]+'temp.pdb', overwrite=True)
    #Read-in again (this will create a Residue)
    newmol = next(pybel.readfile("pdb", os.path.splitext(file)[0]+'temp.pdb'))
    os.remove(os.path.splitext(file)[0]+'temp.pdb')

    #Atomlabel = {0:'C1',1:'X',2:'C',3:'C',4:'C',5:'C',6:'C',7:'C',8:'C',9:'C',10:'C',11:'C',12:'C'}
    #Change atomnames (AtomIDs) to something sensible (OpenBabel does not do this by default)
    print("Creating new atomnames for PDBfile")
    #Note: currently just combining element and atomindex to get a unique atomname (otherwise Modeller will not work)
    #TODO: make something better (element-specific numbering?)
    for res in pybel.ob.OBResidueIter(newmol.OBMol):
        for i,atom in enumerate(openbabel.OBResidueAtomIter(res)):
            atomname = res.GetAtomID(atom)
            #print("atomname:", atomname)
            res.SetAtomID(atom,atomname.strip()+str(i+1))
            atomname = res.GetAtomID(atom)
            #print("atomname:", atomname)
            #res.SetAtomID(atom,Atomlabel[i])

    #Write final PDB-file
    newmol.write(format='pdb', filename=os.path.splitext(file)[0]+'.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0]+'.pdb')
    return os.path.splitext(file)[0]+'.pdb'

#Function to read in PDB-file and write new one with CONECT lines (geometry needs to be sensible)
#NOTE: Requires OpenBabel which seems unnecessary, probably better to use OpenMM functionality instead
def writepdb_with_connectivity(file):
    #OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: writepdb_with_connectivity requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("pdb", file))
    mol.write(format='pdb', filename=os.path.splitext(file)[0]+'_withcon.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0]+'_withcon.pdb')
    return os.path.splitext(file)[0]+'_withcon.pdb'

#Function to read in XYZ-file (small molecule) and create PDB-file with CONECT lines (geometry needs to be sensible)
def xyz_to_pdb_with_connectivity(file, resname="UNL"):
    print("xyz_to_pdb_with_connectivity function:")
    #OpenBabel
    try:
        from openbabel import openbabel
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: xyz_to_pdb_with_connectivity requires OpenBabel library but it could not be imported")
        print("You can install OpenBabel like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    #Read in XYZ-file
    mol = next(pybel.readfile("xyz", file))
    #Write do disk as PDB-file
    mol.write(format='pdb', filename=os.path.splitext(file)[0]+'temp.pdb', overwrite=True)
    #Read-in again (this will create a Residue)
    newmol = next(pybel.readfile("pdb", os.path.splitext(file)[0]+'temp.pdb'))

    os.remove(os.path.splitext(file)[0]+'temp.pdb')

    #Atomlabel = {0:'C1',1:'X',2:'C',3:'C',4:'C',5:'C',6:'C',7:'C',8:'C',9:'C',10:'C',11:'C',12:'C'}
    #Change atomnames (AtomIDs) to something sensible (OpenBabel does not do this by default)
    print("Creating new atomnames for PDBfile")
    #Note: currently just combining element and atomindex to get a unique atomname (otherwise Modeller will not work)
    #TODO: make something better (element-specific numbering?)
    for res in pybel.ob.OBResidueIter(newmol.OBMol):
        #Setting residue name
        res.SetName(resname)
        for i,atom in enumerate(openbabel.OBResidueAtomIter(res)):
            atomname = res.GetAtomID(atom)
            #print("atomname:", atomname)
            res.SetAtomID(atom,atomname.strip()+str(i+1))
            atomname = res.GetAtomID(atom)
            #print("atomname:", atomname)
            #res.SetAtomID(atom,Atomlabel[i])

    #Write final PDB-file
    newmol.write(format='pdb', filename=os.path.splitext(file)[0]+'.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0]+'.pdb')
    return os.path.splitext(file)[0]+'.pdb'

#Function to convert PDB-file to SMILES string
def pdb_to_smiles(fname: str) -> str:
    #OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: pdb_to_smiles requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("pdb", fname))
    smi = mol.write(format="smi")
    return smi.split()[0].strip()

#Function to convert SMILES string to elements and coordinates list
def smiles_to_coords(smiles_string):
    #OpenBabel
    try:
        from openbabel import pybel
        from openbabel import openbabel
    except ModuleNotFoundError:
        print("Error: smiles_to_coords requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    print("Reading SMILES by OpenBabel")
    mol = pybel.readstring("smi", smiles_string)
    print("Guessing 3D coordinates (uses MMFF94 forcefield)")
    mol.make3D()
    b_mol = mol.OBMol
    atomnums = []
    coords = []
    for atom in openbabel.OBMolAtomIter(b_mol):
        atomnums.append(atom.GetAtomicNum())
        coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    elems = [reformat_element(atn, isatomnum=True) for atn in atomnums]
    #frag = Fragment(elems=elems, coords=coords, charge=charge, mult=mult)
    return elems, coords