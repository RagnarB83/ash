import numpy as np
import os
import shutil
import time

import ash.constants
from ash.modules.module_QMMM import QMMMTheory
from ash.interfaces.interface_OpenMM import OpenMMTheory
from ash.modules.module_coords import print_coords_for_atoms,print_internal_coordinate_table,write_XYZ_for_atoms,write_xyzfile,write_coords_all
from ash.functions.functions_general import ashexit, blankline,BC,print_time_rel,print_line_with_mainheader,print_line_with_subheader1
from ash.modules.module_coords import check_charge_mult
from ash.modules.module_freq import write_hessian,calc_hessian_xtb, approximate_full_Hessian_from_smaller
from ash.modules.module_results import ASH_Results

##################################################
# NEW Interface to geomeTRIC Optimization Library
##################################################
#Attempt to write a simpler more modular interface

#Wrapper function around GeomeTRICOptimizerClass
#NOTE: theory and fragment given to Optimizer function but not part of Class initialization. Only passed to run method
def geomeTRICOptimizer(theory=None, fragment=None, charge=None, mult=None, coordsystem='tric', frozenatoms=None, constraints=None, 
                       constrainvalue=False, maxiter=100, ActiveRegion=False, actatoms=None,
                       convergence_setting=None, conv_criteria=None, print_atoms_list=None, TSOpt=False, hessian=None, partial_hessian_atoms=None,
                       modelhessian=None, subfrctor=1, MM_PDB_traj_write=False, printlevel=2):
    """
    Wrapper function around GeomeTRICOptimizerClass
    """
    #print_line_with_mainheader("geomeTRICOptimizer")
    timeA=time.time()
    #NOTE: Class does not take fragment and theory
    optimizer=GeomeTRICOptimizerClass(charge=charge, mult=mult, coordsystem=coordsystem, frozenatoms=frozenatoms, 
                        maxiter=maxiter, ActiveRegion=ActiveRegion, actatoms=actatoms, TSOpt=TSOpt, 
                        hessian=hessian, partial_hessian_atoms=partial_hessian_atoms,modelhessian=modelhessian,
                        convergence_setting=convergence_setting, conv_criteria=conv_criteria, 
                        print_atoms_list=print_atoms_list, subfrctor=subfrctor, MM_PDB_traj_write=MM_PDB_traj_write,
                        printlevel=printlevel)
    
    #Providing theory and fragment to run method. Also constraints
    result = optimizer.run(theory=theory, fragment=fragment, charge=charge, mult=mult,
                           constraints=constraints, constrainvalue=constrainvalue)
    print_time_rel(timeA, modulename='geomeTRIC', moduleindex=1)

    return result

# Class for optimization. 
class GeomeTRICOptimizerClass:
        def __init__(self,theory=None, charge=None, mult=None, coordsystem='tric', 
                     frozenatoms=None, maxiter=50, ActiveRegion=False, actatoms=None, 
                       convergence_setting=None, conv_criteria=None, TSOpt=False, hessian=None,
                       print_atoms_list=None, partial_hessian_atoms=None, modelhessian=None, 
                       subfrctor=1, MM_PDB_traj_write=False, printlevel=2):
            print_line_with_mainheader("geomeTRICOptimizer initialization")
            print("Creating optimizer object")

            ###############################
            #Going through user options
            ###############################

            #Active region and coordsystem
            if actatoms==None:
                actatoms=[]
            if frozenatoms==None:
                frozenatoms=[]

            if ActiveRegion == True and coordsystem == "tric":
                #TODO: Look into this more
                print("Activeregion true and coordsystem = tric are not compatible")
                print("Switching to HDLC")
                coordsystem='hdlc'
            
            #Defining some attributes
            self.maxiter=maxiter
            self.printlevel=printlevel
            self.actatoms=actatoms
            self.frozenatoms=frozenatoms
            self.coordsystem=coordsystem
            self.print_atoms_list=print_atoms_list
            self.ActiveRegion=ActiveRegion
            self.TSOpt=TSOpt
            self.subfrctor=subfrctor
            #For MM or QM/MM whether to write PDB-trajectory or not
            self.MM_PDB_traj_write=MM_PDB_traj_write
            #Hessian stuff
            self.hessian=hessian
            self.modelhessian=modelhessian
            self.partial_hessian_atoms=partial_hessian_atoms

            #Constraints by default set to None
            self.constraints=None
            ######################

            #Setup convergence criteria (sets self.conv_criteria)
            self.convergence_criteria(convergence_setting,conv_criteria)

            ######################
            #SOME PRINTING of settings
            ######################
            print("Printlevel:", self.printlevel)
            print("Coordinate system: ", self.coordsystem)
            print("Max iterations: ", self.maxiter)
            print("Frozen atoms:", self.frozenatoms)
            print("Active Region:", self.ActiveRegion)
            if self.ActiveRegion is True:
                print("Number of active atoms:", len(self.actatoms))
            print("TS Optimization:", self.TSOpt)
            print("Hessian Option:", self.hessian)
                    
        #Requires info on theory and fragment
        def print_atoms_output_setting(self,theory,fragment):
            #What atoms to print in outputfile in each opt-step. Example choice: QM-region only
            #If not specified then active-region or all-atoms
            if self.print_atoms_list == None:
                #Print-atoms list not specified. What to do: 
                if self.ActiveRegion == True:
                    #If QM/MM object then QM-region:
                    if isinstance(theory,QMMMTheory):
                        print("Theory class: QMMMTheory")
                        print("Will by default print only QM-region in output (use print_atoms_list option to change)")
                        self.print_atoms_list=theory.qmatoms
                    else:
                        #Print actatoms since using Active Region (can be too much)
                        self.print_atoms_list=self.actatoms
                else:
                    #No act-region. Print all atoms
                    self.print_atoms_list=fragment.allatoms

        def convergence_criteria(self,convergence_setting,conv_criteria):
           ########################################
            #Dealing with convergence criteria
            ########################################
            if convergence_setting is None or convergence_setting == 'ORCA':
                #default
                if conv_criteria is None:
                    self.conv_criteria = {'convergence_energy' : 5e-6, 'convergence_grms' : 1e-4, 'convergence_gmax' : 3.0e-4, 'convergence_drms' : 2.0e-3, 
                            'convergence_dmax' : 4.0e-3 }
            elif convergence_setting == 'Chemshell':
                self.conv_criteria = {'convergence_energy' : 1e-6, 'convergence_grms' : 3e-4, 'convergence_gmax' : 4.5e-4, 'convergence_drms' : 1.2e-3, 
                                'convergence_dmax' : 1.8e-3 }
            elif convergence_setting == 'ORCA_TIGHT':
                self.conv_criteria = {'convergence_energy' : 1e-6, 'convergence_grms' : 3e-5, 'convergence_gmax' : 1.0e-4, 'convergence_drms' : 6.0e-4, 
                            'convergence_dmax' : 1.0e-3 }
            elif convergence_setting == 'GAU':
                self.conv_criteria = {'convergence_energy' : 1e-6, 'convergence_grms' : 3e-4, 'convergence_gmax' : 4.5e-4, 'convergence_drms' : 1.2e-3, 
                            'convergence_dmax' : 1.8e-3 }
            elif convergence_setting == 'GAU_TIGHT':
                self.conv_criteria = {'convergence_energy' : 1e-6, 'convergence_grms' : 1e-5, 'convergence_gmax' : 1.5e-5, 'convergence_drms' : 4.0e-5, 
                                'convergence_dmax' : 6e-5 }
            elif convergence_setting == 'GAU_VERYTIGHT':
                self.conv_criteria = {'convergence_energy' : 1e-6, 'convergence_grms' : 1e-6, 'convergence_gmax' : 2e-6, 'convergence_drms' : 4.0e-6, 
                                'convergence_dmax' : 6e-6 }        
            elif convergence_setting == 'SuperLoose':
                        self.conv_criteria = { 'convergence_energy' : 1e-1, 'convergence_grms' : 1e-1, 'convergence_gmax' : 1e-1, 'convergence_drms' : 1e-1, 
                            'convergence_dmax' : 1e-1 }
            else:
                print("Unknown convergence setting. Exiting...")
                ashexit()

        #Option to populate constraints dictionary
        def set_constraints(self,con):
            self.constraints=con
        #Parse the constraints into bond, angle, dihedral
        def define_constraints(self,constraints):
            ########################################
            #CONSTRAINTS
            ########################################
            # For QM/MM we need to convert full-system atoms into active region atoms 
            #constraints={'bond':[[8854,37089]]}
            if self.ActiveRegion == True:
                if constraints != None:
                    print("Constraints set. Active region true")
                    print("User-defined constraints (fullsystem-indices):", constraints)
                    constraints=constraints_indices_convert(constraints,self.actatoms)
                    print("Converting constraints indices to active-region indices")
                    print("Constraints (actregion-indices):", constraints)

            #Delete constraintsfile
            os.remove('constraints.txt')
            #Getting individual constraints from constraints dict
            if constraints is not None:
                try:
                    bondconstraints = constraints['bond']
                except:
                    bondconstraints = None
                try:
                    angleconstraints = constraints['angle']
                except:
                    angleconstraints = None
                try:
                    dihedralconstraints = constraints['dihedral']
                except:
                    dihedralconstraints = None
            else:
                bondconstraints=None
                angleconstraints=None
                dihedralconstraints=None
            
            return bondconstraints, angleconstraints, dihedralconstraints

        def write_constraintsfile(self,frozenatoms,bondconstraints,constrainvalue,angleconstraints,dihedralconstraints):
            ########################################
            # CONSTRAINTS
            ########################################
            #Write constraints to constraints.txt file
            #Frozen atom option. Only for small systems. Not QM/MM etc.
            self.constraintsfile=None
            if len(frozenatoms) > 0 :
                print("Writing frozen atom constraints")
                self.constraintsfile='constraints.txt'
                with open("constraints.txt", 'a') as confile:
                    confile.write('$freeze\n')
                    for frozat in frozenatoms:
                        #Changing from zero-indexing (ASH) to 1-indexing (geomeTRIC)
                        frozenatomindex=frozat+1
                        confile.write(f'xyz {frozenatomindex}\n')
            #Bond constraints
            if bondconstraints is not None :
                self.constraintsfile='constraints.txt'
                with open("constraints.txt", 'a') as confile:
                    if constrainvalue is True:
                        confile.write('$set\n')            
                    else:
                        confile.write('$freeze\n')
                    for bondpair in bondconstraints:
                        #Changing from zero-indexing (ASH) to 1-indexing (geomeTRIC)
                        #print("bondpair", bondpair)
                        if constrainvalue is True:
                            confile.write(f'distance {bondpair[0]+1} {bondpair[1]+1} {bondpair[2]}\n')                    
                        else:    
                            confile.write(f'distance {bondpair[0]+1} {bondpair[1]+1}\n')
            #Angle constraints
            if angleconstraints is not None :
                self.constraintsfile='constraints.txt'
                with open("constraints.txt", 'a') as confile:
                    if constrainvalue is True:
                        confile.write('$set\n')            
                    else:
                        confile.write('$freeze\n')
                    for angleentry in angleconstraints:
                        #Changing from zero-indexing (ASH) to 1-indexing (geomeTRIC)
                        #print("angleentry", angleentry)
                        if constrainvalue is True:
                            confile.write(f'angle {angleentry[0]+1} {angleentry[1]+1} {angleentry[2]+1} {angleentry[3]}\n')
                        else:
                            confile.write(f'angle {angleentry[0]+1} {angleentry[1]+1} {angleentry[2]+1}\n')
            if dihedralconstraints is not None:
                self.constraintsfile='constraints.txt'
                with open("constraints.txt", 'a') as confile:
                    if constrainvalue is True:
                        confile.write('$set\n')            
                    else:
                        confile.write('$freeze\n')
                    for dihedralentry in dihedralconstraints:
                        #Changing from zero-indexing (ASH) to 1-indexing (geomeTRIC)
                        #print("dihedralentry", dihedralentry)
                        if constrainvalue is True:
                            confile.write(f'dihedral {dihedralentry[0]+1} {dihedralentry[1]+1} {dihedralentry[2]+1} {dihedralentry[3]+1} {dihedralentry[4]}\n')
                        else:
                            confile.write(f'dihedral {dihedralentry[0]+1} {dihedralentry[1]+1} {dihedralentry[2]+1} {dihedralentry[3]+1}\n')

        def cleanup(self):
            #Clean-up before we begin
            tmpfiles=['geometric_OPTtraj.log','geometric_OPTtraj.xyz','geometric_OPTtraj_Full.xyz','geometric_OPTtraj_QMregion.xyz', 'optimization_energies.log',
                'constraints.txt','initialxyzfiletric.xyz','geometric_OPTtraj.tmp','dummyprefix.tmp','dummyprefix.log','Fragment-optimized.ygg','Fragment-optimized.xyz',
                'Fragment-optimized_Active.xyz','geometric_OPTtraj-PDB.pdb']
            for tmpfile in tmpfiles:
                try:
                    shutil.rmtree(tmpfile)
                except FileNotFoundError:
                    pass
                except NotADirectoryError:
                    os.remove(tmpfile)
                else:
                    pass

        def hessian_option(self,fragment,actatoms,theory,charge,mult,modelhessian):
            #Do xtB Hessian to get Hessian file if requestd
            if self.hessian == "xtb":
                print("xTB Hessian option requested")
                #Calling xtb to get Hessian, written to disk. Returns name of Hessianfile
                hessianfile = calc_hessian_xtb(fragment=fragment, actatoms=actatoms, numcores=theory.numcores, use_xtb_feature=True, charge=charge, mult=mult)
                self.hessian="file:"+hessianfile
            #NumFreq 1 and 2-point Hessians
            elif self.hessian == "1point":
                print("Requested Hessian from Numfreq 1-point approximation (running in serial)")
                result_freq = ash.NumFreq(theory=theory, fragment=fragment, printlevel=0, npoint=1, runmode='serial', numcores=theory.numcores)
                hessianfile="Hessian_from_theory"
                shutil.copyfile("Numfreq_dir/Hessian",hessianfile)
                self.hessian='file:'+str(hessianfile)
            elif self.hessian == "2point":
                print("Requested Hessian from Numfreq 2-point approximation (running in serial)")
                result_freq = ash.NumFreq(theory=theory, fragment=fragment, printlevel=0, npoint=2, runmode='serial', numcores=theory.numcores)
                hessianfile="Hessian_from_theory"
                shutil.copyfile("Numfreq_dir/Hessian",hessianfile)
                self.hessian='file:'+str(hessianfile)
            elif self.hessian == "partial":
                print("Partial Hessian option requested")
                
                if self.partial_hessian_atoms is None:
                    print("hessian='partial' option requires setting the partial_hessian_atoms option. Exiting.")
                    ashexit()

                print("Now doing partial Hessian calculation using atoms:", self.partial_hessian_atoms)
                #Note: hardcoding runmode='serial' for now
                result_freq = ash.NumFreq(theory=theory, fragment=fragment, printlevel=0, npoint=1, hessatoms=self.partial_hessian_atoms, runmode='serial', numcores=1)
                #Combine partial exact Hessian with model Hessian(Almloef, Lindh, Schlegel or unit)
                #Large Hessian is the actatoms Hessian if actatoms provided
                
                combined_hessian = approximate_full_Hessian_from_smaller(fragment,result_freq.hessian, self.partial_hessian_atoms, large_atomindices=actatoms, restHessian=modelhessian)

                #Write combined Hessian to disk
                hessianfile="Hessian_from_partial"
                write_hessian(combined_hessian,hessfile=hessianfile)
                self.hessian="file:"+hessianfile

        #If using Active region then we write only those coordinates to disk (initialxyzfiletric)
        def setup_active_region_geometry(self,fragment):
            if len(self.actatoms) == 0:
                print("Error: List of active atoms (actatoms) provided is empty. This is not allowed.")
                ashexit()
            #Sorting list, otherwise trouble
            self.actatoms.sort()
            print("Active Region option Active. Passing only active-region coordinates to geomeTRIC.")
            print("Active atoms list:", self.actatoms)
            print("Number of active atoms:", len(self.actatoms))

            #Check that the actatoms list does not contain atom indices higher than the number of atoms
            largest_atom_index=max(self.actatoms)
            if largest_atom_index >= fragment.numatoms:
                print(BC.FAIL,f"Found active-atom index ({largest_atom_index}) that is larger or equal (>=) than the number of atoms of system ({fragment.numatoms})!",BC.END)
                print(BC.FAIL,"This does not make sense. Please provide a correct actatoms list. Exiting.",BC.END)
                ashexit()
            #Get active region coordinates and elements
            actcoords, actelems = fragment.get_coords_for_atoms(self.actatoms)
            
            #Writing act-region coords (only) of ASH fragment to disk as XYZ file and reading into geomeTRIC
            write_xyzfile(actelems, actcoords, 'initialxyzfiletric')

        #Running geomeTRIC object
        def run(self, theory=None, fragment=None, charge=None, mult=None, constraints=None, constrainvalue=False):
            if self.printlevel > 1:
                print()
                print_line_with_subheader1("Running geomeTRIC object")
                print(BC.WARNING, f"\nDoing geometry optimization on fragment. Formula: {fragment.prettyformula} Label: {fragment.label} ", BC.END)            
            #Cleanup of temp-files before we begin
            self.cleanup() #NOTE: This deletes constraintsfile

            #################
            # CONSTRAINTS
            #################
            #If constraints not provided to run method, then we look at self.constraints
            if constraints == None:
                print("No constraints provided to run method")
                if self.constraints != None:
                    constraints=self.constraints
                else:
                    print("No previously defined constraints found either")
            
            print("\nConstraints: ", constraints)
            print("constrainvalue: ", constrainvalue)

            #Constraints
            bondconstraints, angleconstraints, dihedralconstraints = self.define_constraints(constraints)
            self.write_constraintsfile(self.frozenatoms,bondconstraints,constrainvalue,angleconstraints,
                                       dihedralconstraints)
            #################
            #EARLY EXITS:
            #Check charge/mult
            charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "geomeTRICOptimizer", theory=theory)
            fragment.charge=charge
            fragment.mult=mult
            
            #Check if atom and do Singlepoint instead if so
            if fragment.numatoms == 1:
                print("System has 1 atoms.")
                print("Doing single-point energy calculation instead")
                result = ash.Singlepoint(fragment=fragment, theory=theory, charge=charge, mult=mult)
                return result.energy
            
            #ActiveRegion option where geomeTRIC only sees the QM part that is being optimized
            if self.ActiveRegion == True:
                self.setup_active_region_geometry(fragment)
            #Whole system
            else:
                #Write coordinates from ASH fragment to disk as XYZ-file and reading into geomeTRIC
                fragment.write_xyzfile("initialxyzfiletric.xyz")

            #Determine geometry-printout in each iteration. Requires knowledge on theory and fragment
            self.print_atoms_output_setting(theory,fragment)

            #Hessian option
            self.hessian_option(fragment,self.actatoms,theory,charge,mult,self.modelhessian)



            ######################
            #CALLING LIBRARY
            ######################
            try:
                import geometric
            except:
                blankline()
                print(BC.FAIL,"Problem importing geomeTRIC module!", BC.END)
                print(BC.WARNING,"Either install geomeTRIC using pip:\n conda install geometric\n or \n pip install geometric\n or manually from Github (https://github.com/leeping/geomeTRIC)", BC.END)
                ashexit(code=9)

            #Read geometry from XYZ-file into geomeTRIC Molecule object
            mol_geometric_frag=geometric.molecule.Molecule("initialxyzfiletric.xyz")

            #Defining ASHengineclass engine object containing geometry and theory. ActiveRegion boolean passed.
            #Also now passing list of atoms to print in each step.
            ashengine = ASHengineclass(mol_geometric_frag,theory, ActiveRegion=self.ActiveRegion, actatoms=self.actatoms, 
                print_atoms_list=self.print_atoms_list, MM_PDB_traj_write=self.MM_PDB_traj_write,
                charge=charge, mult=mult, conv_criteria=self.conv_criteria, fragment=fragment, printlevel=self.printlevel)
            
            #Defining args object, containing engine object
            final_geometric_args=geomeTRICArgsObject(ashengine,self.constraintsfile,coordsys=self.coordsystem, 
                maxiter=self.maxiter, conv_criteria=self.conv_criteria, transition=self.TSOpt, hessian=self.hessian, subfrctor=self.subfrctor)

            print("Convergence criteria:", self.conv_criteria)
            print("Hessian option:", self.hessian)

            if self.TSOpt == True:
                print("Starting saddlepoint optimization")
            else:
                print("Starting optimization")

            ###################################
            # RUNNING
            ###################################
            geometric.optimize.run_optimizer(**vars(final_geometric_args))
            time.sleep(1)

             ###################################

            blankline()
            print(f"geomeTRIC Geometry optimization converged in {ashengine.iteration_count} steps!")
            blankline()

            #QM/MM: Doing final energy evaluation if Truncated PC option was on
            if isinstance(theory,QMMMTheory):
                if theory.TruncatedPC is True:
                    print("Truncated PC approximation was active. Doing final energy calculation with full PC environment")
                    theory.TruncatedPC=False
                    finalenergy, finalgrad = theory.run(current_coords=ashengine.full_current_coords, elems=fragment.elems, 
                        Grad=True,  charge=charge, mult=mult)
                        #label='FinalIter',
                else:
                    finalenergy=ashengine.energy
            else:
                #Updating energy and coordinates of ASH fragment before ending
                finalenergy=ashengine.energy

            print("Final optimized energy:",  finalenergy)

            #Replacing coordinates in fragment
            fragment.replace_coords(fragment.elems,ashengine.full_current_coords, conn=False)
            #Writing out fragment file and XYZ file
            fragment.print_system(filename='Fragment-optimized.ygg')
            fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
            fragment.set_energy(finalenergy)
            
            #Active region XYZ-file
            if self.ActiveRegion==True:
                write_XYZ_for_atoms(fragment.coords, fragment.elems, self.actatoms, "Fragment-optimized_Active")
            #QM-region XYZ-file
            if isinstance(theory,QMMMTheory):
                write_XYZ_for_atoms(fragment.coords, fragment.elems, theory.qmatoms, "Fragment-optimized_QMregion")

            #Printing internal coordinate table
            print_internal_coordinate_table(fragment,actatoms=self.print_atoms_list)
            blankline()

            #Now returning final Results object
            #TODO: Return dictionary of energy, gradient, coordinates etc, coordinates along trajectory ??
            result = ASH_Results(label="Optimizer", energy=finalenergy, initial_geometry=None, 
                    geometry=fragment.coords)
            return result


class geomeTRICArgsObject:
    def __init__(self,eng,constraintsfile, coordsys, maxiter, conv_criteria, transition,hessian,subfrctor):
        self.coordsys=coordsys
        self.maxiter=maxiter
        self.transition=transition
        self.hessian=hessian
        self.subfrctor=subfrctor
        #self.convergence_criteria=conv_criteria
        #self.converge=conv_criteria
        #Setting these to be part of kwargs that geometric reads
        self.convergence_energy = conv_criteria['convergence_energy']
        self.convergence_grms = conv_criteria['convergence_grms']
        self.convergence_gmax = conv_criteria['convergence_gmax']
        self.convergence_drms = conv_criteria['convergence_drms']
        self.convergence_dmax = conv_criteria['convergence_dmax']
        self.prefix='geometric_OPTtraj'
        self.input='dummyinputname'
        self.constraints=constraintsfile
        #Created log.ini file here. Missing from pip installation for some reason?
        #Storing log.ini in ash dir
        path = os.path.dirname(ash.__file__)
        self.logIni=path+'/log.ini'
        self.customengine=eng

#Defining ASH engine class used to communicate with geomeTRIC
class ASHengineclass:
    def __init__(self,geometric_molf, theory, ActiveRegion=False, actatoms=None,print_atoms_list=None, charge=None, mult=None, conv_criteria=None, fragment=None,
        MM_PDB_traj_write=False, printlevel=2):
        #MM_PDB_traj_write on/off. Can be pretty big files
        self.MM_PDB_traj_write=MM_PDB_traj_write
        #Defining M attribute of engine object as geomeTRIC Molecule object
        self.M=geometric_molf
        #Defining theory from argument
        self.theory=theory
        self.ActiveRegion=ActiveRegion
        #Defining current_coords for full system (not only act region)
        self.full_current_coords=[]
        #Manual iteration count
        self.iteration_count=0
        #Defining initial E
        self.energy = 0
        #Active atoms
        self.actatoms=actatoms
        #Print-list atoms (set above)
        self.print_atoms_list=print_atoms_list
        self.charge=charge
        self.mult=mult
        self.conv_criteria=conv_criteria
        self.fragment=fragment
        self.printlevel=printlevel

    def load_guess_files(self,dirname):
        print("geometric called load_guess_files option for ASHengineclass.")
        print("This option is currently unsupported in ASH. Continuing.")
    def save_guess_files(self,dirname):
        print("geometric called save_guess_files option option for ASHengineclass.")
        print("This option is currently unsupported in ASH. Continuing.")
    #Optimizer may call this to see if the engine class is doing DFT with grid to print warning
    def detect_dft(self):
        print("geometric called detect_dft option option for ASHengineclass.")
        #TODO: Return True or False
        return True
    #geometric checks if calc_bondorder method is implemented for the ASHengine. Disabled until we implement this
    #def calc_bondorder(self,coords,dirname):
    #    print("geometric called calc_bondorder option option for ASHengineclass.")
    #    print("This option is currently unsupported in ASH. Continuing.")
    #TODO: geometric will regularly do ClearCalcs in an optimization
    def clearCalcs(self):
        print("geometric called clearCalcs option for ASHengineclass.")
        print("This option is currently unsupported in ASH. Continuing.")
    #Writing out trajectory file for full system in case of ActiveRegion. Note: Actregion coordinates are done done by GeomeTRIC
    def write_trajectory_full(self):
        print("Writing trajectory for Full system to file: geometric_OPTtraj_Full.xyz")
        with open("geometric_OPTtraj_Full.xyz", "a") as trajfile:
            trajfile.write(str(self.fragment.numatoms)+"\n")
            trajfile.write(f"Iteration {self.iteration_count} Energy {self.energy} \n")
            for el,cor in zip(self.fragment.elems,self.full_current_coords):
                trajfile.write(el + "  " + str(cor[0]) + " " + str(cor[1]) + " " + str(cor[2]) +
                            "\n")
    #QM/MM: Writing out trajectory file for QM-region if QM/MM.
    def write_trajectory_qmregion(self):
        print("Writing trajectory for QM-region to file: geometric_OPTtraj_QMregion.xyz")
        with open("geometric_OPTtraj_QMregion.xyz", "a") as trajfile:
            trajfile.write(str(len(self.theory.qmatoms))+"\n")
            trajfile.write(f"Iteration {self.iteration_count} Energy {self.energy} \n")
            qm_coords, qm_elems = self.fragment.get_coords_for_atoms(self.theory.qmatoms)
            for el,cor in zip(qm_elems,qm_coords):
                trajfile.write(el + "  " + str(cor[0]) + " " + str(cor[1]) + " " + str(cor[2]) +
                            "\n")
    def write_energy_logfile(self):
        #QM/MM: Writing out logfile containing QM-energy, MM-energy, QM/MM-energy
        print("Writing logfile with energies: optimization_energies.log")
        with open("optimization_energies.log", "a") as trajfile:
            if self.iteration_count == 0:
                trajfile.write(f"Iteration QM-energy       (Eh) MM-Energy (Eh)  QM/MM-Energy (Eh)\n")
            trajfile.write(f"{self.iteration_count}         {self.theory.QMenergy} {self.theory.MMenergy} {self.theory.QM_MM_energy}\n")

    def write_pdbtrajectory(self):
        print("Writing PDB-trajectory to file: geometric_OPTtraj-PDB.pdb")
        pdbtrajectoryfile="geometric_OPTtraj-PDB.pdb"
        # Get OpenMM positions
        #STILL problem with PBC
        state = self.theory.mm_theory.simulation.context.getState(getEnergy=False, getPositions=True, getForces=False,enforcePeriodicBox=True)
        newpos = state.getPositions()
        self.theory.mm_theory.openmm.app.PDBFile.writeFile(self.theory.mm_theory.topology, newpos, file=open(pdbtrajectoryfile, 'a'))

    #Defining calculator.
    #Read_data and copydir not used (dummy variables)
    def calc(self,coords,tmp, read_data=None, copydir=None):
        #print("read_data:", read_data)
        #Note: tmp and read_data not used. Needed for geomeTRIC version compatibility
        print("Convergence criteria:", self.conv_criteria)

        print()
        #Updating coords in object
        #Need to combine with rest of full-system coords
        timeA=time.time()
        self.M.xyzs[0] = coords.reshape(-1, 3) * ash.constants.bohr2ang
        #print_time_rel(timeA, modulename='geometric ASHcalc.calc reshape', moduleindex=2)
        timeA=time.time()
        currcoords=self.M.xyzs[0]
        #Special act-region (for QM/MM) since GeomeTRIC does not handle huge system and constraints
        if self.ActiveRegion==True:
            #Defining full_coords as original coords temporarily
            #full_coords = np.array(fragment.coords)
            full_coords = self.fragment.coords
            
            #Replacing act-region coordinates in full_coords with coords from currcoords
            for act_i,curr_i in zip(self.actatoms,currcoords):
                full_coords[act_i] = curr_i
            #print_time_rel(timeA, modulename='geometric ASHcalc.calc replacing act-region', moduleindex=2)
            timeA=time.time()
            self.full_current_coords = full_coords
            
            #Write out fragment with updated coordinates for the purpose of doing restart
            self.fragment.replace_coords(self.fragment.elems, self.full_current_coords, conn=False)
            self.fragment.print_system(filename='Fragment-currentgeo.ygg')
            self.fragment.write_xyzfile(xyzfilename="Fragment-currentgeo.xyz")
            #print_time_rel(timeA, modulename='geometric ASHcalc.calc replacecoords and printsystem', moduleindex=2)
            timeA=time.time()

            #PRINTING TO OUTPUT SPECIFIC GEOMETRY IN EACH GEOMETRIC ITERATION (now: self.print_atoms_list)
            print(f"Current geometry (Å) in step {self.iteration_count} (print_atoms_list region)")
            
            print("-------------------------------------------------")
            
            #print_atoms_list
            #Previously act: print_coords_for_atoms(self.full_current_coords, fragment.elems, self.actatoms)
            print_coords_for_atoms(self.full_current_coords, self.fragment.elems, self.print_atoms_list)
            #print_time_rel(timeA, modulename='geometric ASHcalc.calc printcoords atoms', moduleindex=2)
            timeA=time.time()
            print("Note: Only print_atoms_list region printed above")
            #Request Engrad calc for full system

            E, Grad = self.theory.run(current_coords=self.full_current_coords, elems=self.fragment.elems, charge=self.charge, mult=self.mult, Grad=True)
            #label='Iter'+str(self.iteration_count)
            #print_time_rel(timeA, modulename='geometric ASHcalc.calc theory.run', moduleindex=2)
            timeA=time.time()
            
            if self.printlevel >2:
                print("printlevel >2. Writing full grad to disk")
                write_coords_all(Grad, self.fragment.elems, indices=self.fragment.allatoms, file="Grad", description="Grad (au/Bohr):")
            #Trim Full gradient down to only act-atoms gradient
            Grad_act = np.array([Grad[i] for i in self.actatoms])
            if self.printlevel >2:
                print("printlevel >2. Writing active grad to disk")
                act_elems=[self.fragment.elems[i] for i in self.actatoms]
                write_coords_all(Grad_act, act_elems, indices=[i for i in range(0, len(self.actatoms))], file="Grad_act", description="Grad_act (au/Bohr):")
            #print_time_rel(timeA, modulename='geometric ASHcalc.calc trim full gradient', moduleindex=2)
            timeA=time.time()
            self.energy = E

            print("Writing trajectory for Active Region to file: geometric_OPTtraj.xyz")

            #Now writing trajectory for full system
            self.write_trajectory_full()
            
            #Case QM/MM:
            if isinstance(self.theory,QMMMTheory):
                #Writing trajectory for QM-region only
                self.write_trajectory_qmregion()
                #Writing logfile with QM,MM and QM/MM energies
                self.write_energy_logfile()

                #Case MMtheory is OpenMM: Write out PDB-trajectory via OpenMM
                if isinstance(self.theory.mm_theory,OpenMMTheory):
                    if self.MM_PDB_traj_write is True:
                        self.write_pdbtrajectory()

            #print_time_rel(timeA, modulename='geometric ASHcalc.calc writetraj full', moduleindex=2)
            timeA=time.time()
            self.iteration_count += 1
            return {'energy': E, 'gradient': Grad_act.flatten()}
        else:
            self.full_current_coords=currcoords
            #PRINTING ACTIVE GEOMETRY IN EACH GEOMETRIC ITERATION
            #print("Current geometry (Å) in step {}".format(self.iteration_count))
            print(f"Current geometry (Å) in step {self.iteration_count} (print_atoms_list region)")
            print("---------------------------------------------------")
            print_coords_for_atoms(currcoords, self.fragment.elems, self.print_atoms_list)
            print("")
            print("Note: printed only print_atoms_list (this is not necessary all atoms) ")
            E,Grad=self.theory.run(current_coords=currcoords, elems=self.M.elem, charge=self.charge, mult=self.mult,
                                Grad=True)
            #label='Iter'+str(self.iteration_count)
            self.iteration_count += 1
            self.energy = E
            return {'energy': E, 'gradient': Grad.flatten()}


#Function Convert constraints indices to actatom indices
def constraints_indices_convert(con,actatoms):
    try:
        bondcons=con['bond']
    except KeyError:
        bondcons=[]
    try:
        anglecons=con['angle']
    except KeyError:
        anglecons=[]
    try:
        dihedralcons=con['dihedral']
    except KeyError:
        dihedralcons=[]
    #Looping over constraints-class (bond,angle-dihedral)
    #list-item:
    for bc in bondcons:
        bc[0]=fullindex_to_actindex(bc[0],actatoms)
        bc[1]=fullindex_to_actindex(bc[1],actatoms)
    for ac in anglecons:
        ac[0]=fullindex_to_actindex(ac[0],actatoms)
        ac[1]=fullindex_to_actindex(ac[1],actatoms)
        ac[2]=fullindex_to_actindex(ac[2],actatoms)
    for dc in dihedralcons:
        dc[0]=fullindex_to_actindex(dc[0],actatoms)
        dc[1]=fullindex_to_actindex(dc[1],actatoms)
        dc[2]=fullindex_to_actindex(dc[2],actatoms)
        dc[3]=fullindex_to_actindex(dc[3],actatoms)
    return con

#Simple function to convert atom indices from full system to Active region. Single index case
def fullindex_to_actindex(fullindex,actatoms):
    actindex=actatoms.index(fullindex)
    return actindex
