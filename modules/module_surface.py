"""Surface Scans module:
    Primary functions:
    calc_surface : Calculate 1D/2D surface from geometric parameters
    calc_surface_fromXYZ: Calculate 1D/2D surface from XYZ files
    read_surfacedict_from_file: Read dictionary of surfacepoints from file

    """
import math
import os
import glob
import shutil
import copy
import time
#import ash
from ash.functions.functions_general import frange, BC, print_line_with_mainheader,print_line_with_subheader1,print_time_rel, ashexit
import ash.interfaces.interface_geometric
from ash.modules.module_freq import calc_rotational_constants
import ash.functions.functions_parallel
from ash.modules.module_coords import check_charge_mult

# TODO: Finish parallelize surfacepoint calculations
def calc_surface(fragment=None, theory=None, charge=None, mult=None, scantype='Unrelaxed', resultfile='surface_results.txt', keepoutputfiles=True, keepmofiles=False,
                 runmode='serial', coordsystem='dlc', maxiter=50, extraconstraints=None, convergence_setting=None, 
                 ActiveRegion=False, actatoms=None, **kwargs):
    """Calculate 1D/2D surface

    Args:
        fragment (ASH fragment, optional): ASH fragment object. Defaults to None.
        theory (ASH theory, optional): ASH theory object. Defaults to None.
        scantype (str, optional): Type of scan: 'Unrelaxed' or 'Relaxed'. Defaults to 'Unrelaxed'.
        resultfile (str, optional): Name of resultfile. Defaults to 'surface_results.txt'.
        runmode (str, optional): Runmode: 'serial' or 'parallel. Defaults to 'serial'.
        coordsystem (str, optional): Coordinate system for geomeTRICOptimizer. Defaults to 'dlc'.
        maxiter (int, optional): Max number of Opt iterations. Defaults to 50.
        extraconstraints (dict, optional): Dictionary of additional constraints for geomeTRICOptimizer. Defaults to None.
        convergence_setting (str, optional): Convergence setting for geomeTRICOptimizer. Defaults to None.
        ActiveRegion (bool,optional): To use activeregion or not in optimization
        actatoms (list,optional): List of active atoms

    Returns:
        [type]: [description]
    """
    module_init_time=time.time()
    print_line_with_mainheader("CALC_SURFACE FUNCTION")

    #Check charge/mult
    charge,mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "calc_surface", theory=theory)

    if 'numcores' in kwargs:
        numcores = kwargs['numcores']
    #Getting reaction coordinates and checking if 1D or 2D
    if 'RC1_range' in kwargs:
        RC1_range=kwargs['RC1_range']
        RC1_type=kwargs['RC1_type']
        RC1_indices=kwargs['RC1_indices']
        #Checking if list of lists. If so then we apply multiple constraints for this reaction coordinate (e.g. symmetric bonds)
        #Here making list of list in case only a single list was provided
        if any(isinstance(el, list) for el in RC1_indices) is False:
            RC1_indices=[RC1_indices]
        print("RC1_type:", RC1_type)
        print("RC1_indices:", RC1_indices)
        print("RC1_range:", RC1_range)
        
    if 'RC2_range' in kwargs:
        dimension=2
        RC2_range=kwargs['RC2_range']
        RC2_type=kwargs['RC2_type']
        RC2_indices=kwargs['RC2_indices']
        if any(isinstance(el, list) for el in RC2_indices) is False:
            RC2_indices=[RC2_indices]
        print("RC2_type:", RC2_type)
        print("RC2_indices:", RC2_indices)
        print("RC2_range:", RC2_range)
    else:
        dimension=1
    
    #Calc number of surfacepoints
    if dimension==2:
        range2=math.ceil(abs((RC2_range[0]-RC2_range[1])/RC2_range[2]))
        #print("range2", range2)
        range1=math.ceil(abs((RC1_range[0]-RC1_range[1])/RC1_range[2]))

        #Create lists of point-values
        RCvalue1_list=list(frange(RC1_range[0],RC1_range[1],RC1_range[2]))
        RCvalue1_list.append(float(RC1_range[1]))    #Adding last specified value to list also
        RCvalue2_list=list(frange(RC2_range[0],RC2_range[1],RC2_range[2]))
        RCvalue2_list.append(float(RC2_range[1]))    #Adding last specified value to list also
        print("RCvalue1_list: ", RCvalue1_list)
        print("RCvalue2_list: ", RCvalue2_list)
        totalnumpoints=len(RCvalue1_list)*len(RCvalue2_list)

    elif dimension==1:
        #totalnumpoints=math.ceil(abs((RC1_range[0]-RC1_range[1])/RC1_range[2]))
        #Create lists of point-values
        RCvalue1_list=list(frange(RC1_range[0],RC1_range[1],RC1_range[2]))
        RCvalue1_list.append(float(RC1_range[1]))    #Adding last specified value to list also
        print("RCvalue1_list: ", RCvalue1_list)
        totalnumpoints=len(RCvalue1_list)

    print("Number of surfacepoints to calculate ", totalnumpoints)

    #Read dict from file. If file exists, read entries, if not, return empty dict
    surfacedictionary = read_surfacedict_from_file(resultfile, dimension=dimension)
    print("Initial surfacedictionary :", surfacedictionary)
    
    #
    if theory.__class__.__name__ == "ZeroTheory":
        keepoutputfiles=False
        keepomofile=False
    print("keepoutputfiles: ", keepoutputfiles)
    print("keepmofiles: ", keepmofiles)





    pointcount=0
    
    #Create directory to keep track of surface XYZ files
    try:
        shutil.rmtree("surface_xyzfiles")
    except:
        pass
    try:
        shutil.rmtree("surface_outfiles")
    except:
        pass
    try:
        shutil.rmtree("surface_fragfiles")
    except:
        pass
    try:
        shutil.rmtree("surface_mofiles")
    except:
        pass
    os.mkdir('surface_xyzfiles')
    os.mkdir('surface_outfiles')
    os.mkdir('surface_fragfiles')
    os.mkdir('surface_mofiles')
    #try:
    #    os.mkdir('surface_xyzfiles') 
    #    os.mkdir('surface_outfiles')
    #    os.mkdir('surface_fragfiles')
    #    os.mkdir('surface_mofiles')
    #except FileExistsError:
    #    print("")
    #    print(BC.FAIL,"surface_xyzfiles, surface_fragfiles, surface_mofiles and surface_outfiles directories exist already in dir. Please remove them", BC.END)
     #   ashexit()

    #PARALLEL CALCULATION
    if runmode=='parallel':
        print("Parallel runmode.")
        surfacepointfragments={}
        if scantype=='Unrelaxed':
            if dimension == 2:
                zerotheory = ash.ZeroTheory()
                for RCvalue1 in RCvalue1_list:
                    for RCvalue2 in RCvalue2_list:
                        pointcount+=1
                        print("=======================================")
                        print("Surfacepoint: {} / {}".format(pointcount,totalnumpoints))
                        print("RCvalue1: {} RCvalue2: {}".format(RCvalue1,RCvalue2))
                        print("=======================================")
                        pointlabel='RC1_'+str(RCvalue1)+'-'+'RC2_'+str(RCvalue2)
                        if (RCvalue1,RCvalue2) not in surfacedictionary:
                            #Now setting constraints
                            allconstraints = set_constraints(dimension=2, RCvalue1=RCvalue1, RCvalue2=RCvalue2, extraconstraints=extraconstraints,
                                                             RC1_type=RC1_type, RC2_type=RC2_type, RC1_indices=RC1_indices, RC2_indices=RC2_indices)
                            print("allconstraints:", allconstraints)
                            #Running zero-theory with optimizer just to set geometry
                            ash.interfaces.interface_geometric.geomeTRICOptimizer(fragment=fragment, theory=zerotheory, maxiter=maxiter, coordsystem=coordsystem, 
                            constraints=allconstraints, constrainvalue=True, convergence_setting=convergence_setting,
                            ActiveRegion=ActiveRegion, actatoms=actatoms)
                            #Shallow copy of fragment
                            newfrag = copy.copy(fragment)
                            #newfrag.label = str(RCvalue1)+"_"+str(RCvalue2)
                            #Label can be tuple
                            newfrag.label = (RCvalue1,RCvalue2)
                            
                            newfrag.write_xyzfile(xyzfilename="RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")
                            shutil.move("RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz", "surface_xyzfiles/RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")
                            surfacepointfragments[(RCvalue1,RCvalue2)] = newfrag
                            #Single-point ORCA calculation on adjusted geometry
                            #energy = ash.Singlepoint(fragment=fragment, theory=theory, charge=charge, mult=mult)
                print("surfacepointfragments:", surfacepointfragments)
                #TODO: sort this list??
                surfacepointfragments_lists = list(surfacepointfragments.values())
                print("surfacepointfragments_lists: ", surfacepointfragments_lists)
                surfacedictionary = ash.functions.functions_parallel.Singlepoint_parallel(fragments=surfacepointfragments_lists, theories=[theory], numcores=numcores)
                print("Parallel calculation done!")
                print("surfacedictionary:", surfacedictionary)

                if len(surfacedictionary) != totalnumpoints:
                    print("Dictionary not complete!")
                    print("len surfacedictionary:", len(surfacedictionary))
                    print("totalnumpoints:", totalnumpoints)
            elif dimension == 1:
                print("not ready")
                ashexit()
        elif scantype=="Relaxed":
            print("not ready")
            if dimension == 2:
                print("not ready")
            if dimension == 1:
                print("not ready")
            ashexit()
    #SERIAL CALCULATION
    elif runmode=='serial':
        print("Serial runmode")
        if scantype=='Unrelaxed':
            zerotheory = ash.ZeroTheory()
            if dimension == 2:
                for RCvalue1 in RCvalue1_list:
                    for RCvalue2 in RCvalue2_list:
                        pointcount+=1
                        print("==================================================")
                        print("Surfacepoint: {} / {}".format(pointcount,totalnumpoints))
                        print("RCvalue1: {} RCvalue2: {}".format(RCvalue1,RCvalue2))
                        print("Unrelaxed scan. Will use Zerotheory and geometric to set geometry.")
                        print("==================================================")
                        pointlabel='RC1_'+str(RCvalue1)+'-'+'RC2_'+str(RCvalue2)
                        if (RCvalue1,RCvalue2) not in surfacedictionary:
                            #Now setting constraints
                            allconstraints = {}
                            allconstraints = set_constraints(dimension=2, RCvalue1=RCvalue1, RCvalue2=RCvalue2, extraconstraints=extraconstraints,
                                                             RC1_type=RC1_type, RC2_type=RC2_type, RC1_indices=RC1_indices, RC2_indices=RC2_indices)
                            print("x allconstraints:", allconstraints)
                            #Running zero-theory with optimizer just to set geometry
                            ash.interfaces.interface_geometric.geomeTRICOptimizer(fragment=fragment, theory=zerotheory, maxiter=maxiter, coordsystem=coordsystem, 
                            constraints=allconstraints, constrainvalue=True, convergence_setting=convergence_setting, charge=charge, mult=mult,
                            ActiveRegion=ActiveRegion, actatoms=actatoms)
                            
                            #Write geometry to disk
                            fragment.write_xyzfile(xyzfilename="RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")
                            fragment.print_system(filename="RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".ygg")
                            shutil.move("RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz", "surface_xyzfiles/RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")
                            shutil.move("RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".ygg", "surface_fragfiles/RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".ygg")
                            #Single-point ORCA calculation on adjusted geometry
                            if theory is not None:
                                energy = ash.Singlepoint(fragment=fragment, theory=theory, charge=charge, mult=mult)
                                print("RCvalue1: {} RCvalue2: {} Energy: {}".format(RCvalue1,RCvalue2, energy))
                                if keepoutputfiles == True:
                                    shutil.copyfile(theory.filename+'.out', 'surface_outfiles/'+str(theory.filename)+'_'+pointlabel+'.out')
                                if keepmofiles == True:
                                    shutil.copyfile(theory.filename+'.gbw', 'surface_mofiles/'+str(theory.filename)+'_'+pointlabel+'.gbw')
                            surfacedictionary[(RCvalue1,RCvalue2)] = energy

                            #Writing dictionary to file
                            write_surfacedict_to_file(surfacedictionary,resultfile, dimension=2)
                            calc_rotational_constants(fragment)
                        else:
                            print("RC1, RC2 values in dict already. Skipping.")
                    print("surfacedictionary:", surfacedictionary)
                    
            elif dimension == 1:
                for RCvalue1 in RCvalue1_list:
                    pointcount+=1
                    print("==================================================")
                    print("Surfacepoint: {} / {}".format(pointcount,totalnumpoints))
                    print("RCvalue1: {}".format(RCvalue1))
                    print("Unrelaxed scan. Will use Zerotheory and geometric to set geometry.")
                    print("==================================================")
                    pointlabel='RC1_'+str(RCvalue1)
                    if (RCvalue1) not in surfacedictionary:
                        #Now setting constraints
                        allconstraints = set_constraints(dimension=1, RCvalue1=RCvalue1, extraconstraints=extraconstraints,
                                                         RC1_type=RC1_type, RC1_indices=RC1_indices)
                        print("allconstraints:", allconstraints)
                        #Running zero-theory with optimizer just to set geometry
                        ash.interfaces.interface_geometric.geomeTRICOptimizer(fragment=fragment, theory=zerotheory, maxiter=maxiter, coordsystem=coordsystem, 
                        constraints=allconstraints, constrainvalue=True, convergence_setting=convergence_setting, charge=charge, mult=mult,
                        ActiveRegion=ActiveRegion, actatoms=actatoms)
                        
                        #Write geometry to disk: RC1_2.02.xyz
                        fragment.write_xyzfile(xyzfilename="RC1_"+str(RCvalue1)+".xyz")
                        fragment.print_system(filename="RC1_"+str(RCvalue1)+".ygg")
                        shutil.move("RC1_"+str(RCvalue1)+".xyz", "surface_xyzfiles/"+"RC1_"+str(RCvalue1)+".xyz")
                        shutil.move("RC1_"+str(RCvalue1)+".ygg", "surface_fragfiles/"+"RC1_"+str(RCvalue1)+".ygg")
                        #Single-point ORCA calculation on adjusted geometry
                        energy = ash.Singlepoint(fragment=fragment, theory=theory, charge=charge, mult=mult)
                        print("RCvalue1: {} Energy: {}".format(RCvalue1,energy))
                        if keepoutputfiles == True:
                            shutil.copyfile(theory.filename+'.out', 'surface_outfiles/'+str(theory.filename)+'_'+pointlabel+'.out')
                        if keepmofiles == True:
                            shutil.copyfile(theory.filename+'.gbw', 'surface_mofiles/'+str(theory.filename)+'_'+pointlabel+'.gbw')
                        surfacedictionary[(RCvalue1)] = energy
                        #Writing dictionary to file
                        write_surfacedict_to_file(surfacedictionary,resultfile, dimension=1)
                        print("surfacedictionary:", surfacedictionary)
                        calc_rotational_constants(fragment)
                    else:
                        print("RC1 value in dict already. Skipping.")
        elif scantype=='Relaxed':
            zerotheory = ash.ZeroTheory()
            if dimension == 2:
                for RCvalue1 in RCvalue1_list:
                    for RCvalue2 in RCvalue2_list:
                        pointcount+=1
                        print("==================================================")
                        print("Surfacepoint: {} / {}".format(pointcount,totalnumpoints))
                        print("RCvalue1: {} RCvalue2: {}".format(RCvalue1,RCvalue2))
                        print("Relaxed scan. Will relax geometry using theory level with the included contraints.")
                        print("==================================================")
                        pointlabel='RC1_'+str(RCvalue1)+'-'+'RC2_'+str(RCvalue2)
                        if (RCvalue1,RCvalue2) not in surfacedictionary:
                            #Now setting constraints
                            allconstraints = set_constraints(dimension=2, RCvalue1=RCvalue1, RCvalue2=RCvalue2, extraconstraints=extraconstraints,
                                                             RC1_type=RC1_type, RC2_type=RC2_type, RC1_indices=RC1_indices, RC2_indices=RC2_indices)
                            print("allconstraints:", allconstraints)
                            #Running 
                            energy = ash.interfaces.interface_geometric.geomeTRICOptimizer(fragment=fragment, theory=theory, maxiter=maxiter, coordsystem=coordsystem, 
                            constraints=allconstraints, constrainvalue=True, convergence_setting=convergence_setting, charge=charge, mult=mult,
                            ActiveRegion=ActiveRegion, actatoms=actatoms)
                            print("RCvalue1: {} RCvalue2: {} Energy: {}".format(RCvalue1,RCvalue2, energy))
                            if keepoutputfiles == True:
                                shutil.copyfile(theory.filename+'.out', 'surface_outfiles/'+str(theory.filename)+'_'+pointlabel+'.out')
                            if keepmofiles == True:
                                shutil.copyfile(theory.filename+'.gbw', 'surface_mofiles/'+str(theory.filename)+'_'+pointlabel+'.gbw')
                            surfacedictionary[(RCvalue1,RCvalue2)] = energy
                            #Writing dictionary to file
                            write_surfacedict_to_file(surfacedictionary,resultfile, dimension=2)
                            calc_rotational_constants(fragment)
                            #Write geometry to disk
                            fragment.write_xyzfile(xyzfilename="RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")
                            fragment.print_system(filename="RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".ygg")
                            shutil.move("RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz", "surface_xyzfiles/RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")
                            shutil.move("RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".ygg", "surface_fragfiles/RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".ygg")
                        else:
                            print("RC1, RC2 values in dict already. Skipping.")
                    print("surfacedictionary:", surfacedictionary)
            elif dimension == 1:
                for RCvalue1 in RCvalue1_list:
                    pointcount+=1
                    print("==================================================")
                    print("Surfacepoint: {} / {}".format(pointcount,totalnumpoints))
                    print("RCvalue1: {}".format(RCvalue1))
                    print("Relaxed scan. Will relax geometry using theory level with the included contraints.")
                    print("==================================================")
                    pointlabel='RC1_'+str(RCvalue1)
                    if (RCvalue1) not in surfacedictionary:
                        #Now setting constraints
                        allconstraints = set_constraints(dimension=1, RCvalue1=RCvalue1, extraconstraints=extraconstraints,
                                                         RC1_type=RC1_type, RC1_indices=RC1_indices)
                        print("allconstraints:", allconstraints)
                        #Running zero-theory with optimizer just to set geometry
                        energy = ash.interfaces.interface_geometric.geomeTRICOptimizer(fragment=fragment, theory=theory, maxiter=maxiter, coordsystem=coordsystem, 
                        constraints=allconstraints, constrainvalue=True, convergence_setting=convergence_setting, charge=charge, mult=mult,
                        ActiveRegion=ActiveRegion, actatoms=actatoms)
                        print("RCvalue1: {} Energy: {}".format(RCvalue1, energy))
                        if keepoutputfiles == True:
                            shutil.copyfile(theory.filename+'.out', 'surface_outfiles/'+str(theory.filename)+'_'+pointlabel+'.out')
                        if keepmofiles == True:
                            shutil.copyfile(theory.filename+'.gbw', 'surface_mofiles/'+str(theory.filename)+'_'+pointlabel+'.gbw')
                        surfacedictionary[(RCvalue1)] = energy
                        #Writing dictionary to file
                        write_surfacedict_to_file(surfacedictionary,resultfile, dimension=1)
                        print("surfacedictionary:", surfacedictionary)
                        calc_rotational_constants(fragment)
                        #Write geometry to disk
                        fragment.write_xyzfile(xyzfilename="RC1_"+str(RCvalue1)+".xyz")
                        fragment.print_system(filename="RC1_"+str(RCvalue1)+".ygg")
                        shutil.move("RC1_"+str(RCvalue1)+".xyz", "surface_xyzfiles/"+"RC1_"+str(RCvalue1)+".xyz")
                        shutil.move("RC1_"+str(RCvalue1)+".ygg", "surface_fragfiles/"+"RC1_"+str(RCvalue1)+".ygg")
                    else:
                        print("RC1 value in dict already. Skipping.")
    print_time_rel(module_init_time, modulename='calc_surface', moduleindex=0)                    
    return surfacedictionary

# Calculate surface from XYZ-file collection.
#Both unrelaxed (single-point) and relaxed (opt) is now possible
# Parallelization and MOREAD complete
# TODO: Parallelization and Relaxed mode
def calc_surface_fromXYZ(xyzdir=None, theory=None, charge=None, mult=None, dimension=None, resultfile='surface_results.txt', scantype='Unrelaxed',runmode='serial',
                         coordsystem='dlc', maxiter=50, extraconstraints=None, convergence_setting=None, numcores=None,
                         RC1_type=None, RC2_type=None, RC1_indices=None, RC2_indices=None, keepoutputfiles=True, keepmofiles=False,
                         read_mofiles=False, mofilesdir=None):
    """Calculate 1D/2D surface from XYZ files

    Args:
        xyzdir (str, optional): Path to directory with XYZ files. Defaults to None.
        theory (ASH theory, optional): ASH theory object. Defaults to None.
        dimension (int, optional): Dimension of surface. Defaults to None.
        resultfile (str, optional): Name of resultfile. Defaults to None.
        scantype (str, optional): Type of scan: 'Unrelaxed' or 'Relaxed' Defaults to 'Unrelaxed'.
        runmode (str, optional): Runmode: 'serial' or 'parallel'. Defaults to 'serial'.
        coordsystem (str, optional): Coordinate system for geomeTRICOptimizer. Defaults to 'dlc'.
        maxiter (int, optional): Max number of iterations for geomeTRICOptimizer. Defaults to 50.
        extraconstraints (dict, optional): Dictionary of constraints for geomeTRICOptimizer. Defaults to None.
        convergence_setting (str, optional): Convergence setting for geomeTRICOptimizer. Defaults to None.
        numcores (float, optional): Number of cores. Defaults to None.
        RC1_type (str, optional):  Reaction-coordinate type (bond,angle,dihedral). Defaults to None.
        RC2_type (str, optional): Reaction-coordinate type (bond,angle,dihedral). Defaults to None.
        RC1_indices (list, optional):  List of atom-indices involved for RC1. Defaults to None.
        RC2_indices (list, optional): List of atom-indices involved for RC2. Defaults to None.

    Returns:
        [type]: [description]
    """
    module_init_time=time.time()
    print_line_with_mainheader("CALC_SURFACE_FROMXYZ FUNCTION")

    #Checking if charge and mult has been provided
    if charge == None or mult == None:
        print(BC.FAIL, "Error. charge and mult has not been defined for calc_surface_fromXYZ", BC.END)
        ashexit()

    print("XYZdir:", xyzdir)
    print("Theory:", theory)
    print("Dimension:", dimension)
    print("Resultfile:", resultfile)
    print("Scan type:", scantype)
    print("keepoutputfiles:", keepoutputfiles)
    print("keepmofiles:", keepmofiles)
    print("read_mofiles:", read_mofiles)
    print("mofilesdir:", mofilesdir)
    print("runmode:", runmode)
    if read_mofiles == True:
        if mofilesdir == None:
            print("mofilesdir not set. Exiting")
            ashexit()
    print("");print("")
    #Read dict from file. If file exists, read entries, if not, return empty dict
    surfacedictionary = read_surfacedict_from_file(resultfile, dimension=dimension)
    print("Initial surfacedictionary :", surfacedictionary)
    print("")

    #Points
    totalnumpoints=len(glob.glob(xyzdir+'/*.xyz'))
    if totalnumpoints == 0:
        print("Found no XYZ-files in directory. Exiting")
        ashexit()
    print("totalnumpoints:", totalnumpoints)
    if len(surfacedictionary) == totalnumpoints:
        print("Surface dictionary size {} matching total number of XYZ files {}. We should have all data".format(len(surfacedictionary),totalnumpoints))
        print("Exiting.")
        return surfacedictionary


    #Case Relaxed Scan: Create directory to keep track of optimized surface XYZ files
    if scantype=="Relaxed":
        try:
            os.mkdir('surface_xyzfiles') 
        except FileExistsError:
            print("")
            print(BC.FAIL,"surface_xyzfiles directory exist already in dir. Please remove it", BC.END)
            ashexit()

    #Create directory to keep track of surface outfiles for runmode=serial
    #Note: for runmode_parallel we have separate dirs for each surfacepoint where we have inputfile, outputfile and MOfile
    if runmode=='serial':
        try:
            shutil.rmtree("'surface_outfiles'")
        except:
            pass
        try:
            os.mkdir('surface_outfiles')
        except FileExistsError:
            print("")
            #print(BC.FAIL,"surface_outfiles directory exist already in dir. Removing...", BC.END)
            
        try:
            shutil.rmtree("'surface_mofiles'")
        except:
            pass
        try:
            os.mkdir('surface_mofiles')
        except FileExistsError:
            print("")

    #New Surfacepoint class to organize the data, at least for parallel mode
    #Using list to collect the Surfacepoint objects
    list_of_surfacepoints=[]
    class Surfacepoint:
        def __init__(self,RC1,RC2=None):
            self.RC1=RC1
            self.RC2=RC2
            self.energy=0.0
            self.xyzfile=None
            self.fragment=None


    ###########################
    #PARALLEL CALCULATION
    ##########################
    if runmode=='parallel':
        print("Parallel runmode.")

        if numcores == None:
            print("numcores argument required for parallel runmode")
            ashexit()

        surfacepointfragments={}
        #Looping over XYZ files to get coordinates
        print("")
        print("Reading XYZ files, expecting format:  RC1_value1-RC2_value2.xyz     Example:  RC1_2.0-RC2_180.0.xyz")
        print("")
        for count,file in enumerate(glob.glob(xyzdir+'/*.xyz')):
            relfile=os.path.basename(file)
            #Getting RC values from XYZ filename e.g. RC1_2.0-RC2_180.0.xyz
            if dimension == 2:
                #Cleaner splitting.
                #TODO: Should we use other symbol than "-" inbetween RC1 and RC2 values?
                start="RC1_"; end="-RC2_"
                RCvalue1=float(relfile.split(start)[1].split(end)[0])
                RCvalue2=float(relfile.split(end)[1].split(".xyz")[0])
                if (RCvalue1,RCvalue2) not in surfacedictionary:
                    #Creating new surfacepoint object
                    newsurfacepoint=Surfacepoint(RCvalue1,RCvalue2)
                    newsurfacepoint.xyzfile=xyzdir+'/'+relfile
                    #NOTE: Currently putting fragment into surfacepoint. Could also just point to xyzfile. Currently more memory-demanding
                    #NOTE: Using tuple as a label for fragment
                    newfrag=ash.Fragment(xyzfile=xyzdir+'/'+relfile, label=(RCvalue1,RCvalue2), charge=charge, mult=mult)
                    #"RC1"+str(RCvalue1)+"_RC2"+str(RCvalue2)
                    newsurfacepoint.fragment=newfrag
                    list_of_surfacepoints.append(newsurfacepoint)
                    #surfacepointfragments[(RCvalue1,RCvalue2)] = newfrag
                    
            elif dimension == 1:
                print("relfile:", relfile)
                if 'RC2' in relfile:
                    print(BC.FAIL,"RC2 information in filename string. Chosen dimension wrong or filename wrong. Exiting", BC.END)
                    ashexit()
                #RC1_2.02.xyz
                RCvalue1=float(relfile.replace('.xyz','').replace('RC1_',''))
                print("XYZ-file: {}     RC1: {} ".format(relfile,RCvalue1))
                if (RCvalue1) not in surfacedictionary:
                    #Creating new surfacepoint object
                    newsurfacepoint=Surfacepoint(RCvalue1)
                    newsurfacepoint.xyzfile=xyzdir+'/'+relfile
                    #NOTE: Currently putting fragment into surfacepoint. Could also just point to xyzfile. Currently more memory-demanding
                    #NOTE: Using tuple as a label for fragment
                    newfrag=ash.Fragment(xyzfile=xyzdir+'/'+relfile, label=(RCvalue1,), charge=charge, mult=mult)
                    newsurfacepoint.fragment=newfrag
                    list_of_surfacepoints.append(newsurfacepoint)

        #This is an ordered list of fragments only. Same order as list_of_surfacepoints, though does not matter since we use dicts
        #Used by ash.Singlepoint_parallel
        surfacepointfragments_lists=[point.fragment for point in list_of_surfacepoints]
        
        if scantype=='Unrelaxed':

            if read_mofiles == True:
                #print("Will read MO-file: {}".format(mofilesdir+'/'+str(theory.filename)+'_'+pointlabel+'.gbw'))
                #if theory.__class__.__name__ == "ORCATheory":
                #    theory.moreadfile=mofilesdir+'/'+str(theory.filename)+'_'+pointlabel+'.gbw'
                results = ash.Singlepoint_parallel(fragments=surfacepointfragments_lists, theories=[theory], numcores=numcores, mofilesdir=mofilesdir)
            else:
                results = ash.Singlepoint_parallel(fragments=surfacepointfragments_lists, theories=[theory], numcores=numcores)
            print("Parallel calculation done!")
            
            #Gathering results in FINAL dictionary.
            for dictitem in results:
                print("Surfacepoint: {} Energy: {}".format(dictitem, results[dictitem]))
                surfacedictionary[dictitem] = results[dictitem]
            print("")

            if len(surfacedictionary) != totalnumpoints:
                print("Dictionary not complete!")
                print("len surfacedictionary:", len(surfacedictionary))
                print("totalnumpoints:", totalnumpoints)

            #Replacing tuple-key with number to make things cleaner for 1d surface dicts.
            if dimension == 1:
                newsurfacedictionary={}
                for k,v in surfacedictionary.items():
                    newsurfacedictionary[k[0]]=v
                surfacedictionary=newsurfacedictionary

            print("surfacedictionary:", surfacedictionary)
            #Write final surface to file
            write_surfacedict_to_file(surfacedictionary,resultfile, dimension=dimension)
        elif scantype=='Relaxed':
            print("calc_surface_fromXYZ Relaxed option not possible in parallel mode yet. Exiting")
            ashexit()
        
        
        
        
    else:
        ###########################
        #SERIAL CALCULATION
        ##########################
        #Looping over XYZ files
        for count,file in enumerate(glob.glob(xyzdir+'/*.xyz')):
            relfile=os.path.basename(file)
            #Getting RC values from XYZ filename e.g. RC1_2.0-RC2_180.0.xyz
            if dimension == 2:
                #Cleaner splitting.
                #TODO: Should we use other symbol than "-" inbetween RC1 and RC2 values?
                start="RC1_"; end="-RC2_"
                RCvalue1=float(relfile.split(start)[1].split(end)[0])
                RCvalue2=float(relfile.split(end)[1].split(".xyz")[0])
                pointlabel='RC1_'+str(RCvalue1)+'-'+'RC2_'+str(RCvalue2)
                print("==================================================================")
                print("Surfacepoint: {} / {}".format(count+1,totalnumpoints))
                print("XYZ-file: {}     RC1: {}   RC2: {}".format(relfile,RCvalue1,RCvalue2))
                print("==================================================================")
                
                #Adding MO-file for point to theory level object if requested
                if read_mofiles == True:
                    print("Will read MO-file: {}".format(mofilesdir+'/'+str(theory.filename)+'_'+pointlabel+'.gbw'))
                    if theory.__class__.__name__ == "ORCATheory":
                        theory.moreadfile=mofilesdir+'/'+str(theory.filename)+'_'+pointlabel+'.gbw'

                
                if (RCvalue1,RCvalue2) not in surfacedictionary:
                    mol=ash.Fragment(xyzfile=file)
                    if scantype=="Unrelaxed":

                        energy = ash.Singlepoint(theory=theory, fragment=mol, charge=charge, mult=mult)
                    elif scantype=="Relaxed":
                        #Now setting constraints
                        allconstraints = set_constraints(dimension=2, RCvalue1=RCvalue1, RCvalue2=RCvalue2, extraconstraints=extraconstraints,
                                                        RC1_type=RC1_type, RC2_type=RC2_type, RC1_indices=RC1_indices, RC2_indices=RC2_indices)
                        print("allconstraints:", allconstraints)
                        energy = ash.interfaces.interface_geometric.geomeTRICOptimizer(fragment=mol, theory=theory, 
                                                    maxiter=maxiter, coordsystem=coordsystem, constraints=allconstraints, constrainvalue=True, 
                                                    convergence_setting=convergence_setting, charge=charge, mult=mult)
                        #Write geometry to disk in dir : surface_xyzfiles
                        fragment.write_xyzfile(xyzfilename="RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")
                        fragment.print_system(filename="RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".ygg")
                        shutil.move("RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz", "surface_xyzfiles/RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".xyz")
                        shutil.move("RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".ygg", "surface_fragfiles/RC1_"+str(RCvalue1)+"-RC2_"+str(RCvalue2)+".ygg")
                    
                    print("Energy of file {} : {} Eh".format(relfile, energy))
                    if keepoutputfiles == True:
                        shutil.copyfile(theory.filename+'.out', 'surface_outfiles/'+str(theory.filename)+'_'+pointlabel+'.out')
                    if keepmofiles == True:
                        shutil.copyfile(theory.filename+'.gbw', 'surface_mofiles/'+str(theory.filename)+'_'+pointlabel+'.gbw')
                    #theory.cleanup()
                    surfacedictionary[(RCvalue1,RCvalue2)] = energy
                    #Writing dictionary to file
                    write_surfacedict_to_file(surfacedictionary,resultfile, dimension=2)
                    print("surfacedictionary:", surfacedictionary)
                    calc_rotational_constants(mol)
                    print("")
                else:
                    print("RC1 and RC2 values in dict already. Skipping.")
            elif dimension == 1:
                #RC1_2.02.xyz
                RCvalue1=float(relfile.replace('.xyz','').replace('RC1_',''))
                pointlabel='RC1_'+str(RCvalue1)
                #print("XYZ-file: {}     RC1: {} ".format(relfile,RCvalue1))
                print("==================================================================")
                print("Surfacepoint: {} / {}".format(count+1,totalnumpoints))
                print("XYZ-file: {}     RC1: {} ".format(relfile,RCvalue1))
                print("==================================================================")

                #Adding MO-file for point to theory level object if requested
                if read_mofiles == True:
                    print("Will read MO-file: {}".format(mofilesdir+'/'+str(theory.filename)+'_'+pointlabel+'.gbw'))
                    if theory.__class__.__name__ == "ORCATheory":
                        theory.moreadfile=mofilesdir+'/'+str(theory.filename)+'_'+pointlabel+'.gbw'

                if (RCvalue1) not in surfacedictionary:
                    mol=ash.Fragment(xyzfile=file)
                    if scantype=="Unrelaxed":
                        energy = ash.Singlepoint(theory=theory, fragment=mol, charge=charge, mult=mult)
                    elif scantype=="Relaxed":
                        #Now setting constraints
                        allconstraints = set_constraints(dimension=1, RCvalue1=RCvalue1, extraconstraints=extraconstraints,
                                                        RC1_type=RC1_type, RC1_indices=RC1_indices)
                        print("allconstraints:", allconstraints)
                        energy = ash.interfaces.interface_geometric.geomeTRICOptimizer(fragment=mol, theory=theory, 
                                                    maxiter=maxiter, coordsystem=coordsystem, constraints=allconstraints, constrainvalue=True, 
                                                    convergence_setting=convergence_setting, charge=charge, mult=mult)
                        #Write geometry to disk in dir : surface_xyzfiles
                        fragment.write_xyzfile(xyzfilename="RC1_"+str(RCvalue1)+".xyz")
                        fragment.print_system(filename="RC1_"+str(RCvalue1)+".ygg")
                        shutil.move("RC1_"+str(RCvalue1)+".xyz", "surface_xyzfiles/"+"RC1_"+str(RCvalue1)+".xyz")
                        shutil.move("RC1_"+str(RCvalue1)+".ygg", "surface_fragfiles/"+"RC1_"+str(RCvalue1)+".ygg")
                    print("Energy of file {} : {} Eh".format(relfile, energy))
                    if keepoutputfiles == True:
                        shutil.copyfile(theory.filename+'.out', 'surface_outfiles/'+str(theory.filename)+'_'+pointlabel+'.out')
                    if keepmofiles == True:
                        shutil.copyfile(theory.filename+'.gbw', 'surface_mofiles/'+str(theory.filename)+'_'+pointlabel+'.gbw')
                    #theory.cleanup()
                    surfacedictionary[(RCvalue1)] = energy
                    #Writing dictionary to file
                    write_surfacedict_to_file(surfacedictionary,resultfile, dimension=1)
                    print("surfacedictionary:", surfacedictionary)
                    calc_rotational_constants(mol)
                    print("")            
                else:
                    print("RC1 value in dict already. Skipping.")
    print_time_rel(module_init_time, modulename='calc_surface_fromXYZ', moduleindex=0)
    return surfacedictionary

def calc_numerical_gradient():
    print("TODO")
    ashexit()




#######################################################################
# Constraints function. Used by calc_surface and calc_surface_fromXYZ
######################################################################
#Setting constraints once values are known
#Add extraconstraints if provided
#TODO: Only works if RC constraints do not overwrite the extraconstraints. Need to fix
def set_constraints(dimension=None,RCvalue1=None, RCvalue2=None, extraconstraints=None,
                    RC1_type=None, RC2_type=None, RC1_indices=None, RC2_indices=None ):
    """Set constraints for calc_surface and calc_surface_fromXYZ

    Args:
        dimension (int, optional): Dimension of scan. Defaults to None.
        RCvalue1 (float, optional): Current value of RC1. Defaults to None.
        RCvalue2 (float, optional): Current value of RC1. Defaults to None.
        extraconstraints (dict, optional): Dictionary of additional constraints for geomeTRICOptimizer. Defaults to None.
        RC1_type (str, optional):  Reaction-coordinate type (bond,angle,dihedral). Defaults to None.
        RC2_type (str, optional): Reaction-coordinate type (bond,angle,dihedral). Defaults to None.
        RC1_indices (list, optional):  List of atom-indices involved for RC1. Defaults to None.
        RC2_indices (list, optional): List of atom-indices involved for RC2. Defaults to None.

    Returns:
        [type]: [description]
    """
    allcon = {}
    if extraconstraints is not None:
        allcon = copy.copy(extraconstraints)
    else:
        allcon = {}
    # Defining all constraints as dict to be passed to geometric
    if dimension == 2:
        RC2=[]
        RC1=[]
        #Creating empty lists for each RC type (Note: could be the same)
        if RC1_type not in allcon:
            allcon[RC1_type] = []
        if RC2_type not in allcon:
            allcon[RC2_type] = []
        for RC2_indexlist in RC2_indices:
            RC2.append(RC2_indexlist+[RCvalue2])
        allcon[RC2_type] = allcon[RC2_type] + RC2
        for RC1_indexlist in RC1_indices:
            RC1.append(RC1_indexlist+[RCvalue1])
        allcon[RC1_type] = allcon[RC1_type] + RC1
    elif dimension == 1:
        RC1=[]
        #Creating empty lists for each RC type (Note: could be the same)
        if RC1_type not in allcon:
            allcon[RC1_type] = []
        for RC1_indexlist in RC1_indices:
            RC1.append(RC1_indexlist+[RCvalue1])
        allcon[RC1_type] = allcon[RC1_type] + RC1
    return allcon


#Functions to read and write energy-surface dictionary in simple format.
#Format: space-separated columns
# 1D: coordinate energy   e.g.    -180.0 -201.434343
# 2D: coordinate1 coordinate2 energy   e.g. e.g.   2.201 -180.0 -201.434343
#Output: dictionary: (tuple) : float   
# 1D: (coordinate1) : energy
# 2D: (coordinate1,coordinate2) : energy
#TODO: Make more general
#TODO: Replace with Json read/write instead??
#Example: https://stackoverflow.com/questions/47568211/how-to-read-and-write-dictionaries-to-external-files-in-python
def read_surfacedict_from_file(file, dimension=None):
    """Read surface dictionary from file

    Args:
        file (str): Name of file to read.
        dimension (int): Dimension of surface. Defaults to None.

    Returns:
        dict: Dictionary of surface-points with energies
    """
    print("Attempting to read old results file:", file)
    dictionary = {}
    #If no file then return empty dict
    if os.path.isfile(file) is False:
        print("No file found.")
        return dictionary
    with open(file) as f:
        for line in f:
            if '#' not in line:
                if len(line) > 1:
                    if dimension==1:
                        key=float(line.split()[0])
                        val=float(line.split()[1])
                        dictionary[(key)]=val
                    elif dimension==2:
                        key1=float(line.split()[0])
                        key2=float(line.split()[1])
                        val=float(line.split()[2])                    
                        dictionary[(key1,key2)]=val
    
    if len(dictionary) > 0:
        print("Dictionary read ")
        return dictionary
    else:
        print("Could not read anything from file")
        return None

def write_surfacedict_to_file(surfacedict,file="surface_results.txt",dimension=None):
    """Write surface dictionary to file

    Args:
        surfacedict (dict): Dictionary of surface-points {(coord1,coord2):energy}
        file (str, optional): Filename. Defaults to "surface_results.txt".
        dimension (int, optional): Dimension of surface. Defaults to None.
    """
    if dimension == None:
        print("write_surfacedict_to_file: Dimension needs to be given")
        ashexit()
    with open(file, 'w') as f:
        for d in surfacedict.items():
            if dimension==1:
                x=d[0]
                #Converting from 1-element tuple to number if it happens to be a tuple
                if type(x) == tuple:
                    x=x[0]
                e=d[1]
                f.write(str(x)+" "+str(e)+'\n')
            elif dimension==2:
                x=d[0][0]
                y=d[0][1]
                e=d[1]
                f.write(str(x)+" "+str(y)+" "+str(e)+'\n')



