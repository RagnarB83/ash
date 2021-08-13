import numpy as np
import constants
from modules.module_coords import print_coords_all,print_coords_for_atoms,print_internal_coordinate_table,write_XYZ_for_atoms,write_xyzfile
from functions.functions_general import blankline,BC,print_time_rel
import os
import shutil
import ash
import time

################################################
# Interface to geomeTRIC Optimization Library
################################################
# https://github.com/leeping/geomeTRIC/blob/master/examples/constraints.txt
# bond,angle and dihedral constraints work. If only atom indices provided and constrainvalue is False then constraint at current position
# If constrainvalue=True then last entry should be value of constraint




#Function to convert atom indices from full system to Active region. Used in case of QM/MM
#Single index case
def fullindex_to_actindex(fullindex,actatoms):
    actindex=actatoms.index(fullindex)
    return actindex



def geomeTRICOptimizer(theory=None,fragment=None, coordsystem='hdlc', frozenatoms=None, constraintsinputfile=None, constraints=None, 
                       constrainvalue=False, maxiter=50, ActiveRegion=False, actatoms=None, convergence_setting=None, conv_criteria=None,
                       print_atoms_list=None):
    """
    Wrapper function around geomeTRIC code. Take theory and fragment info from ASH
    Supports frozen atoms and bond/angle/dihedral constraints in native code. Use frozenatoms and bondconstraints etc. for this.
    New feature: Active Region for huge systems. Use ActiveRegion=True and provide actatoms list.
    Active-atom coords (e.g. only QM region) are only provided to geomeTRIC during optimization while rest is frozen.
    Needed as discussed here: https://github.com/leeping/geomeTRIC/commit/584869707aca1dbeabab6fe873fdf139d384ca66#diff-2af7dd72b77dac63cea64c052a549fe0
    """
    module_init_time=time.time()
    if fragment==None:
        print("geomeTRIC requires fragment object")
        exit()
    try:
        import geometric
    except:
        blankline()
        print(BC.FAIL,"geomeTRIC module not found!", BC.END)
        print(BC.WARNING,"Either install geomeTRIC using pip:\n pip install geometric\n or manually from Github (https://github.com/leeping/geomeTRIC)", BC.END)
        exit(9)

    if fragment.numatoms == 1:
        print("System has 1 atoms.")
        print("Doing single-point energy calculation instead")
        energy = ash.Singlepoint(fragment=fragment, theory=theory)
        return energy
        #E = self.theory.run(current_coords=fragment.coords, elems=fragment.elems, Grad=False)

    if ActiveRegion == True and coordsystem == "tric":
        #TODO: Look into this more
        print("Activeregion true and coordsystem = tric are not compatible")
        exit()

    if actatoms==None:
        actatoms=[]
    if frozenatoms==None:
        frozenatoms=[]

    #Clean-up
    try:
        os.remove('geometric_OPTtraj.log')
        os.remove('geometric_OPTtraj.xyz')
        os.remove('geometric_OPTtraj_Full.xyz')
        os.remove('constraints.txt')
        os.remove('initialxyzfiletric.xyz')
        shutil.rmtree('geometric_OPTtraj.tmp')
        shutil.rmtree('dummyprefix.tmp')
        os.remove('dummyprefix.log')
    except:
        pass
    

    #NOTE: We are now sorting actatoms and qmatoms list both here and in QM/MM object
    #: Alternatively we could sort the actatoms list and qmatoms list in QM/MM object before doing anything. Need to check carefully though....
    #if is_integerlist_ordered(actatoms) is False:
    #    print("Problem. Actatoms list is not sorted in ascending order. Please sort this list (and possibly qmatoms list also))")
    #    exit()
    
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
            #atomindex:
            for i,bc_i in enumerate(bc):
                #replacing
                bc[i]=fullindex_to_actindex(bc_i,actatoms)
        for ac in anglecons:
            #atomindex:
            for j,ac_j in enumerate(ac):
                #replacing
                ac[j]=fullindex_to_actindex(ac_j,actatoms)
        for dc in dihedralcons:
            #atomindex:
            for k,dc_k in enumerate(dc):
                #replacing
                dc[k]=fullindex_to_actindex(dc_k,actatoms)
        return con

    #CONSTRAINTS
    # For QM/MM we need to convert full-system atoms into active region atoms 
    #constraints={'bond':[[8854,37089]]}
    if ActiveRegion == True:
        if constraints != None:
            print("Constraints set. Active region true")
            print("User-defined constraints (fullsystem-indices):", constraints)
            constraints=constraints_indices_convert(constraints,actatoms)
            print("Converting constraints indices to active-region indices")
            print("Constraints (actregion-indices):", constraints)


    #Delete constraintsfile unless asked for
    if constraintsinputfile is None:
        try:
            os.remove('constraints.txt')
        except:
            pass
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
    
    blankline()
    print(BC.WARNING, "Doing geometry optimization on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,fragment.label), BC.END)
    print("Launching geomeTRIC optimization module")
    print("Coordinate system: ", coordsystem)
    print("Max iterations: ", maxiter)
    #print("Frozen atoms: ", frozenatoms)
    print("Constraints: ", constraints)

    #What atoms to print in outputfile in each opt-step. Example choice: QM-region only
    #If not specified then active-region or all-atoms
    if print_atoms_list == None:
        #Print-atoms list not specified. What to do: 
        if ActiveRegion == True:
            #If QM/MM object then QM-region:
            if theory.__class__.__name__ == "QMMMTheory":
                print("Theory class: QMMMTheory")
                print("Will by default print only QM-region in output (use print_atoms_list option to change)")
                print_atoms_list=theory.qmatoms
            else:
                #Print actatoms since using Active Region (can be too much)
                print_atoms_list=actatoms
        else:
            #No act-region. Print all atoms
            print_atoms_list=fragment.allatoms

    print("Atomlist to print in output:", print_atoms_list)


    #ActiveRegion option where geomeTRIC only sees the QM part that is being optimized
    if ActiveRegion == True:
        #Sorting list, otherwise trouble
        actatoms.sort()
        print("Active Region option Active. Passing only active-region coordinates to geomeTRIC.")
        print("Number of active atoms:", len(actatoms))
        actcoords, actelems = fragment.get_coords_for_atoms(actatoms)
        
        #Writing act-region coords (only) of ASH fragment to disk as XYZ file and reading into geomeTRIC
        write_xyzfile(actelems, actcoords, 'initialxyzfiletric')
        mol_geometric_frag=geometric.molecule.Molecule("initialxyzfiletric.xyz")
    else:
        #Write coordinates from ASH fragment to disk as XYZ-file and reading into geomeTRIC
        fragment.write_xyzfile("initialxyzfiletric.xyz")
        mol_geometric_frag=geometric.molecule.Molecule("initialxyzfiletric.xyz")

    #Defining ASH engine class used to communicate with geomeTRIC
    class ASHengineclass:
        def __init__(self,geometric_molf, theory, ActiveRegion=False, actatoms=None,print_atoms_list=None):
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
            
            
        #Defining calculator
        def clearCalcs(self):
            print("ClearCalcs option chosen by geomeTRIC. Not sure why")
        def calc(self,coords,tmp, read_data=None):
            #Note: tmp and read_data not used. Needed for geomeTRIC version compatibility
            print("Convergence criteria:", conv_criteria)
            blankline()
            #Updating coords in object
            #Need to combine with rest of full-syme coords
            self.M.xyzs[0] = coords.reshape(-1, 3) * constants.bohr2ang
            currcoords=self.M.xyzs[0]
            #Special act-region (for QM/MM) since GeomeTRIC does not handle huge system and constraints
            if self.ActiveRegion==True:
                #Defining full_coords as original coords temporarily
                full_coords = np.array(fragment.coords)
                #Replacing act-region coordinates with coords from currcoords
                for i, c in enumerate(full_coords):
                    if i in self.actatoms:
                        #Silly. Pop-ing first coord from currcoords until done
                        curr_c, currcoords = currcoords[0], currcoords[1:]
                        full_coords[i] = curr_c
                self.full_current_coords=full_coords
                #Write out fragment with updated coordinates for the purpose of doing restart
                fragment.replace_coords(fragment.elems, self.full_current_coords, conn=False)
                fragment.print_system(filename='Fragment-currentgeo.ygg')

                #PRINTING TO OUTPUT SPECIFIC GEOMETRY IN EACH GEOMETRIC ITERATION (now: self.print_atoms_list)
                print("Current geometry (Å) in step {} (print_atoms_list region)".format(self.iteration_count))
                
                print("-------------------------------------------------")
                
                #print_atoms_list
                #Previously act: print_coords_for_atoms(self.full_current_coords, fragment.elems, self.actatoms)
                print_coords_for_atoms(self.full_current_coords, fragment.elems, self.print_atoms_list)
                print("Note: printed only print_atoms_list (this not necessary all active atoms) ")
                #Request Engrad calc for full system
                E, Grad = self.theory.run(current_coords=self.full_current_coords, elems=fragment.elems, Grad=True)
                #Trim Full gradient down to only act-atoms gradient
                Grad_act = np.array([Grad[i] for i in self.actatoms])
                self.energy = E

                #Writing out trajectory file for full system. Act system done by GeomeTRIC
                with open("geometric_OPTtraj_Full.xyz", "a") as trajfile:
                    trajfile.write(str(fragment.numatoms)+"\n")
                    trajfile.write("Iteration {} Energy {} \n".format(self.iteration_count,self.energy))
                    for el,cor in zip(fragment.elems,self.full_current_coords):
                        trajfile.write(el + "  " + str(cor[0]) + " " + str(cor[1]) + " " + str(cor[2]) +
                                       "\n")

                self.iteration_count += 1
                return {'energy': E, 'gradient': Grad_act.flatten()}
            else:
                self.full_current_coords=currcoords
                #PRINTING ACTIVE GEOMETRY IN EACH GEOMETRIC ITERATION
                #print("Current geometry (Å) in step {}".format(self.iteration_count))
                print("Current geometry (Å) in step {} (print_atoms_list region)".format(self.iteration_count))
                print("---------------------------------------------------")
                #Disabled: print_coords_all(currcoords, fragment.elems)
                print_coords_for_atoms(currcoords, fragment.elems, self.print_atoms_list)
                print("Note: printed only print_atoms_list (this not necessary all atoms) ")
                
                E,Grad=self.theory.run(current_coords=currcoords, elems=self.M.elem, Grad=True)
                self.iteration_count += 1
                self.energy = E
                return {'energy': E, 'gradient': Grad.flatten()}



    class geomeTRICArgsObject:
        def __init__(self,eng,constraintsfile, coordsys, maxiter, conv_criteria):
            self.coordsys=coordsys
            self.maxiter=maxiter

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

    #Define constraints provided. Write constraints.txt file
    #Frozen atom option. Only for small systems. Not QM/MM etc.
    constraintsfile=None
    if len(frozenatoms) > 0 :
        print("Writing frozen atom constraints")
        constraintsfile='constraints.txt'
        with open("constraints.txt", 'a') as confile:
            confile.write('$freeze\n')
            for frozat in frozenatoms:
                #Changing from zero-indexing (ASH) to 1-indexing (geomeTRIC)
                frozenatomindex=frozat+1
                confile.write('xyz {}\n'.format(frozenatomindex))
    #Bond constraints
    if bondconstraints is not None :
        constraintsfile='constraints.txt'
        with open("constraints.txt", 'a') as confile:
            if constrainvalue is True:
                confile.write('$set\n')            
            else:
                confile.write('$freeze\n')
            for bondpair in bondconstraints:
                #Changing from zero-indexing (ASH) to 1-indexing (geomeTRIC)
                #print("bondpair", bondpair)
                if constrainvalue is True:
                    confile.write('distance {} {} {}\n'.format(bondpair[0]+1,bondpair[1]+1, bondpair[2] ))                    
                else:    
                    confile.write('distance {} {}\n'.format(bondpair[0]+1,bondpair[1]+1))
    #Angle constraints
    if angleconstraints is not None :
        constraintsfile='constraints.txt'
        with open("constraints.txt", 'a') as confile:
            if constrainvalue is True:
                confile.write('$set\n')            
            else:
                confile.write('$freeze\n')
            for angleentry in angleconstraints:
                #Changing from zero-indexing (ASH) to 1-indexing (geomeTRIC)
                #print("angleentry", angleentry)
                if constrainvalue is True:
                    confile.write('angle {} {} {} {}\n'.format(angleentry[0]+1,angleentry[1]+1,angleentry[2]+1,angleentry[3] ))
                else:
                    confile.write('angle {} {} {}\n'.format(angleentry[0]+1,angleentry[1]+1,angleentry[2]+1))
    if dihedralconstraints is not None:
        constraintsfile='constraints.txt'
        with open("constraints.txt", 'a') as confile:
            if constrainvalue is True:
                confile.write('$set\n')            
            else:
                confile.write('$freeze\n')
            for dihedralentry in dihedralconstraints:
                #Changing from zero-indexing (ASH) to 1-indexing (geomeTRIC)
                #print("dihedralentry", dihedralentry)
                if constrainvalue is True:
                    confile.write('dihedral {} {} {} {} {}\n'.format(dihedralentry[0]+1,dihedralentry[1]+1,dihedralentry[2]+1,dihedralentry[3]+1, dihedralentry[4] ))
                else:
                    confile.write('dihedral {} {} {} {}\n'.format(dihedralentry[0]+1,dihedralentry[1]+1,dihedralentry[2]+1,dihedralentry[3]+1))
    if constraintsinputfile is not None:
        constraintsfile=constraintsinputfile

    #Dealing with convergence criteria
    if convergence_setting is None or convergence_setting == 'ORCA':
        #default
        if conv_criteria is None:
            conv_criteria = {'convergence_energy' : 5e-6, 'convergence_grms' : 1e-4, 'convergence_gmax' : 3.0e-4, 'convergence_drms' : 2.0e-3, 
                     'convergence_dmax' : 4.0e-3 }
    elif convergence_setting == 'Chemshell':
        conv_criteria = {'convergence_energy' : 1e-6, 'convergence_grms' : 3e-4, 'convergence_gmax' : 4.5e-4, 'convergence_drms' : 1.2e-3, 
                        'convergence_dmax' : 1.8e-3 }
    elif convergence_setting == 'ORCA_TIGHT':
        conv_criteria = {'convergence_energy' : 1e-6, 'convergence_grms' : 3e-5, 'convergence_gmax' : 1.0e-4, 'convergence_drms' : 6.0e-4, 
                     'convergence_dmax' : 1.0e-3 }
    elif convergence_setting == 'GAU':
        conv_criteria = {'convergence_energy' : 1e-6, 'convergence_grms' : 3e-4, 'convergence_gmax' : 4.5e-4, 'convergence_drms' : 1.2e-3, 
                     'convergence_dmax' : 1.8e-3 }
    elif convergence_setting == 'GAU_TIGHT':
        conv_criteria = {'convergence_energy' : 1e-6, 'convergence_grms' : 1e-5, 'convergence_gmax' : 1.5e-5, 'convergence_drms' : 4.0e-5, 
                        'convergence_dmax' : 6e-5 }
    elif convergence_setting == 'GAU_VERYTIGHT':
        conv_criteria = {'convergence_energy' : 1e-6, 'convergence_grms' : 1e-6, 'convergence_gmax' : 2e-6, 'convergence_drms' : 4.0e-6, 
                        'convergence_dmax' : 6e-6 }        
    elif convergence_setting == 'SuperLoose':
                conv_criteria = { 'convergence_energy' : 1e-1, 'convergence_grms' : 1e-1, 'convergence_gmax' : 1e-1, 'convergence_drms' : 1e-1, 
                     'convergence_dmax' : 1e-1 }
    else:
        print("Unknown convergence setting. Exiting...")
        exit(1)

    print("convergence_setting:", convergence_setting)
    print("conv_criteria:", conv_criteria)




    #Defining ASHengineclass engine object containing geometry and theory. ActiveRegion boolean passed.
    #Also now passing list of atoms to print in each step.
    ashengine = ASHengineclass(mol_geometric_frag,theory, ActiveRegion=ActiveRegion, actatoms=actatoms, print_atoms_list=print_atoms_list)
    #Defining args object, containing engine object
    args=geomeTRICArgsObject(ashengine,constraintsfile,coordsys=coordsystem, maxiter=maxiter, conv_criteria=conv_criteria)

    #Starting geomeTRIC
    geometric.optimize.run_optimizer(**vars(args))
    time.sleep(1)
    blankline()
    print("geomeTRIC Geometry optimization converged in {} steps!".format(ashengine.iteration_count))
    blankline()


    #Updating energy and coordinates of ASH fragment before ending
    fragment.set_energy(ashengine.energy)
    print("Final optimized energy:",  fragment.energy)
    #
    #print("fragment.elems: ", fragment.elems)
    #print("ashengine.full_current_coords : ", ashengine.full_current_coords)
    #Replacing coordinates in fragment
    fragment.replace_coords(fragment.elems,ashengine.full_current_coords, conn=False)
    
    #Writing out fragment file and XYZ file
    fragment.print_system(filename='Fragment-optimized.ygg')
    fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')

    if ActiveRegion==True:
        write_XYZ_for_atoms(fragment.coords, fragment.elems, actatoms, "Fragment-optimized_Active")

    #Printing internal coordinate table
    #TODO: Make a lot better
    print_internal_coordinate_table(fragment,actatoms=print_atoms_list)

    blankline()
    #Now returning final energy
    #TODO: Return dictionary of energy, gradient, coordinates etc, coordinates along trajectory ??
    
    print_time_rel(module_init_time, modulename='geomeTRIC', moduleindex=1)
    return ashengine.energy
