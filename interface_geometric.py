import numpy as np
import constants
from functions_coords import *
from functions_general import *
import os
import shutil
import yggdrasill
import time
#########################
# Interface to geomeTRIC Optimization Library
########################

'''Wrapper function around geomTRIC code. Take theory and fragment info from Yggdrasill
Supports frozen atoms and bond constraints in native code. Use frozenatoms and bondconstraints for this.
New feature: Active Region for huge systems. Use ActiveRegion=True and provide actatoms list.
Active-atom coords (e.g. only QM region) are only provided to geomeTRIC during optimization while rest is frozen.
Needed as discussed here: https://github.com/leeping/geomeTRIC/commit/584869707aca1dbeabab6fe873fdf139d384ca66#diff-2af7dd72b77dac63cea64c052a549fe0
'''
# TODO: Get other constraints (angle-constraints etc.) working.
# Todo: Add optional print-coords in each step option. Maybe only print QM-coords (if QM/MM).
def geomeTRICOptimizer(theory=None,fragment=None, coordsystem='tric', frozenatoms=[], bondconstraints=[],
                       maxiter=50, ActiveRegion=False, actatoms=[]):
    try:
        os.remove('geometric_OPTtraj.log')
        os.remove('geometric_OPTtraj.xyz')
        os.remove('constraints.txt')
        os.remove('initialxyzfiletric.xyz')
        shutil.rmtree('geometric_OPTtraj.tmp')
        shutil.rmtree('dummyprefix.tmp')
        os.remove('dummyprefix.log')
    except:
        pass
    blankline()
    print("Launching geomeTRIC optimization module")
    print("Coordinate system: ", coordsystem)
    print("Max iterations: ", maxiter)
    print("Frozen atoms: ", frozenatoms)
    print("Bond constraints: ", bondconstraints)
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

    #ActiveRegion option where geomeTRIC only sees the QM part that is being optimized
    if ActiveRegion == True:
        print("Active Region option Active. Passing only active-region coordinates to geomeTRIC.")
        print("Number of active atoms:", len(actatoms))
        actcoords, actelems = fragment.get_coords_for_atoms(actatoms)
        #Writing act-region coords (only) of Yggdrasill fragment to disk as XYZ file and reading into geomeTRIC
        write_xyzfile(actelems, actcoords, 'initialxyzfiletric')
        mol_geometric_frag=geometric.molecule.Molecule("initialxyzfiletric.xyz")
    else:
        #Write coordinates from Yggdrasill fragment to disk as XYZ-file and reading into geomeTRIC
        fragment.write_xyzfile("initialxyzfiletric.xyz")
        mol_geometric_frag=geometric.molecule.Molecule("initialxyzfiletric.xyz")

    #Defining Yggdrasill engine class used to communicate with geomeTRIC
    class Yggdrasillengineclass:
        def __init__(self,geometric_molf, theory, ActiveRegion=False, actatoms=actatoms):
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
        #Defining calculator
        def clearCalcs(self):
            print("ClearCalcs option chosen by geomeTRIC. Not sure why")
        def calc(self,coords,tmp):
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

                #PRINTING ACTIVE GEOMETRY IN EACH GEOMETRIC ITERATION
                print("Current geometry (Å) in step {} (active region)".format(self.iteration_count))
                print("---------------------------------------------------")
                print_coords_for_atoms(full_coords, fragment.elems, self.actatoms)

                #Request Engrad calc for full system
                E, Grad = self.theory.run(current_coords=full_coords, elems=fragment.elems, Grad=True)
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
                #PRINTING ACTIVE GEOMETRY IN EACH GEOMETRIC ITERATION
                print("Current geometry (Å) in step {}".format(self.iteration_count))
                print("---------------------------------------------------")
                print_coords_all(currcoords, fragment.elems)
                E,Grad=self.theory.run(current_coords=currcoords, elems=self.M.elem, Grad=True)
                self.iteration_count += 1
                self.energy = E
                return {'energy': E, 'gradient': Grad.flatten()}


    class geomeTRICArgsObject:
        def __init__(self,eng,constraints, coordsys, maxiter):
            self.coordsys=coordsys
            self.maxiter=maxiter
            self.prefix='geometric_OPTtraj'
            self.input='dummyinputname'
            self.constraints=constraints
            #Created log.ini file here. Missing from pip installation for some reason?
            #Storing log.ini in yggdrasill dir
            path = os.path.dirname(yggdrasill.__file__)
            self.logIni=path+'/log.ini'
            self.customengine=eng

    #Define constraints provided. Write constraints.txt file
    #Frozen atom option. Only for small systems. Not QM/MM etc.
    if len(frozenatoms) > 0 :
        print("Writing frozen atom constraints")
        constraints='constraints.txt'
        with open("constraints.txt", 'w') as confile:
            confile.write('$freeze\n')
            for frozat in frozenatoms:
                #Changing from zero-indexing (Yggdrasill) to 1-indexing (geomeTRIC)
                frozenatomindex=frozat+1
                confile.write('xyz {}\n'.format(frozenatomindex))
    #Bond constraints
    elif len(bondconstraints) > 0 :
        constraints='constraints.txt'
        with open("constraints.txt", 'w') as confile:
            confile.write('$freeze\n')
            for bondpair in bondconstraints:
                #Changing from zero-indexing (Yggdrasill) to 1-indexing (geomeTRIC)
                print("bondpair", bondpair)
                confile.write('distance {} {}\n'.format(bondpair[0]+1,bondpair[1]+1))
    else:
        constraints=None

    #Defining Yggdrasill engine object containing geometry and theory. ActiveRegion boolean passed.
    yggdrasillengine = Yggdrasillengineclass(mol_geometric_frag,theory, ActiveRegion=ActiveRegion, actatoms=actatoms)
    #Defining args object, containing engine object
    args=geomeTRICArgsObject(yggdrasillengine,constraints,coordsys=coordsystem, maxiter=maxiter)

    #Starting geomeTRIC
    geometric.optimize.run_optimizer(**vars(args))
    time.sleep(1)
    blankline()
    #print("geomeTRIC Geometry optimization converged in {} steps!".format(geometric.iteration))
    print("geomeTRIC Geometry optimization converged in {} steps!".format(yggdrasillengine.iteration_count))
    blankline()

    #Updating energy and coordinates of Yggdrasill fragment before ending
    fragment.set_energy(yggdrasillengine.energy)
    print("Final optimized energy:",  fragment.energy)
    #
    fragment.replace_coords(fragment.elems,yggdrasillengine.full_current_coords, conn=False)
    #Writing out fragment file and XYZ file
    fragment.print_system(filename='Fragment-optimized.ygg')
    fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')
