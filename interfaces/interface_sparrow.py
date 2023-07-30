import subprocess as sp
import os
import shutil
import time
import numpy as np
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.constants import ang2bohr
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader


class SparrowTheory:
    def __init__(self, sparrowdir=None, filename='sparrow', printlevel=2,
                method=None, numcores=1):

        self.theorynamelabel="Sparrow"
        self.theorytype="QM"
        self.analytic_hessian=False
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        if method is None:
            print(f"{self.theorynamelabel}Theory requires a method keyword")
            ashexit()
        
        try:
            import scine_utilities as su
            import scine_sparrow as scine_sparrow
            self.su=su
            self.scine_sparrow=scine_sparrow
        except:
            print("Problem importing sparrow. Did you install correctly? See: https://github.com/qcscine/sparrow")
            print("Try: conda install scine-sparrow-python")


        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.method=method
        self.numcores=numcores
    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup():
        print("Sparrow cleanup not yet implemented.")
    #TODO: Parallelization is enabled most easily by OMP_NUM_THREADS AND MKL_NUM_THREADS. NOt sure if we can control this here
    #NOTE: Should be possible by adding to subprocess call

    #Request Hessian calculation. Will call run.
    def Hessian(self, fragment=None, Hessian=None, numcores=None, label=None, charge=None, mult=None):
        #Check charge/mult
        charge,mult = check_charge_mult(charge, mult, self.theorytype, fragment, "xTBTheory.Opt", theory=self)
        
        self.run (current_coords=fragment.coords, qm_elems=fragment.elems,
            Grad=True, Hessian=True, numcores=numcores, label=label,
            charge=charge, mult=mult)
        print("Hessian:", self.hessian)
        return self.hessian

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, Hessian=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for SparrowTheory.run method", BC.END)
            ashexit()

        print("Job label:", label)

        #Coords provided to run
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()

        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        #Creating Sparrow objects
        #Geometry
        structure = self.su.AtomCollection(len(qm_elems))
        structure.elements = [self.su.ElementType(elematomnumbers[q.lower()]) for q in qm_elems]
        structure.positions = current_coords*ang2bohr
        #Manager
        manager = self.su.core.ModuleManager()
        #Creating Calculator
        self.calculator = manager.get('calculator', self.method)
        self.calculator.structure = structure

        #Run energy
        self.calculator.set_required_properties([self.su.Property.Energy])
        print("Calculating energy")
        results_e = self.calculator.calculate()
        self.energy=results_e.energy
        #Run gradient
        if Grad==True:
            print("Calculating gradient")
            self.calculator.set_required_properties([self.su.Property.Gradients])
            results_g = self.calculator.calculate()
            #print(dir(results_g))
            #print(results_g.__dict__)
            self.gradient=results_g.gradients

        if Hessian==True:
            self.calculator.set_required_properties([self.su.Property.Hessian])
            print("Calculating Hessian")
            results_h = self.calculator.calculate()
            self.hessian=results_h.hessian

        #TODO: write in error handling here
        print(BC.OKBLUE, BC.BOLD, "------------ENDING Sparrow INTERFACE-------------", BC.END)
        if Grad == True:
            print("Single-point Sparrow energy:", self.energy)
            print_time_rel(module_init_time, modulename='Sparrow run', moduleindex=2)
            if PC is True:
                return self.energy, self.gradient, self.pcgradient
            else:
                return self.energy, self.gradient
        else:
            print("Single-point Sparrow energy:", self.energy)
            print_time_rel(module_init_time, modulename='Sparrow run', moduleindex=2)
            return self.energy
