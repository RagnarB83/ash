import numpy as np
import os
import sys
import re
import time
import glob

from ash.functions.functions_general import ashexit, natural_sort, print_line_with_mainheader,print_time_rel

#Standalone function to call plumed binary and get fes.dat
def call_plumed_sum_hills(path_to_plumed,hillsfile,ndim=None, binsg=[99,99], ming=[0,0], maxg=[10,10]):
        print("Running plumed sum_hills on hills file")
        print("path_to_plumed:", path_to_plumed)
        print("HILLS file:", hillsfile)
        print("Plumed will create fes.dat")
        #Either run plumed binary, requiring to st
        os.environ['PATH'] = path_to_plumed+'/bin'+os.pathsep+os.environ['PATH']
        os.environ['LD_LIBRARY_PATH'] = path_to_plumed+'/lib'+os.pathsep+os.environ['LD_LIBRARY_PATH']
        #os.system(f'plumed sum_hills --hills {hillsfile} --min {} --max {}')
        if ndim == None:
            #Default grid
            print("Calling plumed sum_hills with default grid settings")
            os.system(f'plumed sum_hills --hills HILLS')
        elif ndim == 1:
            print(f"Calling plumed sum_hills with  grid settings: --bin {binsg[0]} --min {ming[0]} --max {maxg[0]}")
            os.system(f'plumed sum_hills --hills HILLS --bin {binsg[0]} --min {ming[0]} --max {maxg[0]}')
        elif ndim == 2:
            print(f"Calling plumed sum_hills with  grid settings: --bin {binsg[0]},{binsg[1]} --min {ming[0]},{ming[1]} --max {maxg[0]},{maxg[1]}")
            os.system(f'plumed sum_hills --hills HILLS --bin {binsg[0]},{binsg[1]} --min {ming[0]},{ming[1]} --max {maxg[0]},{maxg[1]}')

#Metadynamics visualization function for Plumed runs
def MTD_analyze(plumed_ash_object=None, path_to_plumed=None, Plot_To_Screen=False, CV1_type=None, CV2_type=None, temperature=None,
                CV1_indices=None, CV2_indices=None, plumed_length_unit=None, plumed_energy_unit=None, plumed_time_unit=None,
                CV1_grid_limits=None,CV2_grid_limits=None):
    #Energy-unit used by Plumed-ASH should be eV in general
    print_line_with_mainheader("Metadynamics Analysis Script")
    try:
        import matplotlib.pyplot as plt
    except:
        print("Problem importing matplotlib (make it is installed in your environment). Plotting is not possible but continuing.")


    ###############################
    #USER SETTINGS
    ###############################
    colormap='RdYlBu_r'
    #Colormap to use in 2CV plots.
    # Perceptually uniform sequential: viridis, plasma, inferno, magma, cividis
    #Others: # RdYlBu_r
    #See https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    ##############################
    WellTemp=True

    #Get everything from provided plumed_ash object
    if plumed_ash_object != None:
        print("plumed_ash object provided. Reading")
        path_to_plumed=plumed_ash_object.path_to_plumed
        CV1_type = plumed_ash_object.CV1_type
        CV2_type = plumed_ash_object.CV2_type
        CV1_indices = plumed_ash_object.CV1_indices
        CV2_indices = plumed_ash_object.CV2_indices
        temperature=plumed_ash_object.temperature
        plumed_length_unit=plumed_ash_object.plumed_length_unit
        plumed_energy_unit=plumed_ash_object.plumed_energy_unit
        plumed_time_unit=plumed_ash_object.plumed_time_unit
    #Or from plumed info file
    elif os.path.isfile("plumed_ash.info") == True:
        print("Found file plumed_ash.info file in directory. Reading")
        with open("plumed_ash.info") as pinfo:
            for line in pinfo:
                if 'path_to_plumed' in line:
                    print(line)
                    path_to_plumed=line.split()[-1]
                if 'CV1_type' in line:
                    CV1_type = line.split()[-1]
                if 'CV2_type' in line:
                    CV2_type = line.split()[-1]
                if 'CV1_indices' in line:
                    CV1_indices = line.split()[-1]
                if 'CV2_indices' in line:
                    CV2_indices = line.split()[-1]
                if 'temperature' in line:
                    temperature = line.split()[-1]
                if 'plumed_length_unit' in line:
                    plumed_length_unit = line.split()[-1]
                if 'plumed_energy_unit' in line:
                    plumed_energy_unit = line.split()[-1]
                if 'plumed_time_unit' in line:
                    plumed_time_unit = line.split()[-1]

    #Or from keywords (TO BE CHECKED)
    else:
        print("No Plumed_ASH object provided or plumed_ash.info file. Trying to get information via keywords")
        if CV1_type==None or temperature==None:
            print("give CV1_type and temperature")
            ashexit()
        #Run Plumed script to get fes.dat from HILLS
        #Setting PATH and LD_LIBRARY_PATH for PLUMED. LD-lib path to C library may also be required
        if path_to_plumed==None:
            print("Set path_to_plumed argument. Example: MTD_analyze(path_to_plumed=/home/bjornsson/plumed-install)")
            ashexit()


    ######################
    # UNITS
    ######################
    pi=3.14159265359
    #Dict of energy conversions: Energy-unit to kcal/mol
    energy_conversion_dict= {'eV':1/23.060541945329334, 'kj/mol':4.184 }
    # possibly conversion from energy-unit to kcal/mol.
    energy_scaling=energy_conversion_dict[plumed_energy_unit]

    #Possible nm to Ang conversion. TODO: Generalize
    if plumed_length_unit =="A":
        distance_scaling=1
    else:
        print("No plumed_length_unit defined. Assuming default plumed (nm)")
        distance_scaling=10

    #1D
    if CV2_type == None or CV2_type == "None":
        print("1D MTD")
        CVnum=1
        if CV1_type.upper() =='TORSION' or CV1_type.upper()=='ANGLE' or CV1_type.upper()=='DIHEDRAL':
            finalcvunit_1='°'
            CV1_scaling=180/pi
        elif CV1_type.upper() == 'RMSD' or CV1_type.upper()=='DISTANCE' or CV1_type.upper()=='BOND':
            finalcvunit_1='Å'
            CV1_scaling=distance_scaling
        else:
            print("1unknown. exiting");exit()
        print("Final CV1 units:", finalcvunit_1)
    #2D
    else:
        print("2D MTD")
        CVnum=2
        if CV1_type.upper() =='TORSION' or CV1_type.upper()=='ANGLE' or CV1_type.upper()=='DIHEDRAL':
            finalcvunit_1='°'
            CV1_scaling=180/pi
        elif CV1_type.upper() == 'RMSD' or CV1_type.upper()=='DISTANCE' or CV1_type.upper()=='BOND':
            finalcvunit_1='Å'
            CV1_scaling=distance_scaling
            print("")
        else:
            print("2unknown. exiting");exit()
        print("Final CV1 unit:", finalcvunit_1)
        if CV2_type.upper() =='TORSION' or CV2_type.upper()=='ANGLE' or CV2_type.upper()=='DIHEDRAL':
            finalcvunit_2='°'
            CV2_scaling=180/pi
        elif CV2_type.upper() == 'RMSD' or CV2_type.upper()=='DISTANCE' or CV2_type.upper()=='BOND':
            finalcvunit_2='Å'
            CV2_scaling=distance_scaling
        else:
            print("3unknown. exiting");ashexit()
        print("Final CV2 units:", finalcvunit_2)
        if finalcvunit_2 != finalcvunit_1:
            print("differ. possible problem")


    print("CV1_type:", CV1_type)
    print("CV2_type:", CV2_type)
    ######################

    ########################
    #READING MAIN DATA
    ########################
    print("Assuming current dir contains COLVAR/HILLS files")
    print("Reading HILLS files")
    #Checking if Multiple Walker MTD or not (by analyzing whether HILLS.X files are present or not)
    try:
        f = open("HILLS.0")
        f.close()
        print("Found numbered HILLS.X file. This is a multiple-walker run.")
        MultipleWalker=True
    except IOError:
        try:
            f = open("HILLS")
            f.close()
            print("This is a single-walker run")
            MultipleWalker=False
        except FileNotFoundError:
            print("Found no HILLS.X or HILLS file. Exiting...")
            ashexit()

    #The plumed sum_hills command that is run.
    print("")
    if MultipleWalker==True:
        #Removing old HILLS.ALL if present
        try:
            os.remove('HILLS.ALL')
        except:
            pass
        #Gathering HILLS files
        #HILLSFILELIST=sorted(glob.glob("HILLS*"))
        HILLSFILELIST=natural_sort(glob.glob("HILLS*"))
        #Which COLVAR file to look at
        #COLVARFILELIST=sorted(glob.glob("COLVAR*"))
        COLVARFILELIST=natural_sort(glob.glob("COLVAR*"))
        print("MW= True. Concatenating files to HILLS.ALL")
        #os.system('cat HILLS.* > HILLS.ALL')
        print("HILLSFILELIST:", HILLSFILELIST)

        with open('HILLS.ALL', 'w') as outfile:
            for hfile in HILLSFILELIST:
                with open(hfile) as infile:
                    for line in infile:
                        outfile.write(line)

        print("Running plumed to sum hills...")
        print("")
        #RUN PLUMED_ASH OBJECT function
        if CV1_grid_limits is None:
            call_plumed_sum_hills(path_to_plumed,"HILLS.ALL",CVnum)
        else:
            #Changing input unit from Angstrom to nm or degree to radian
            if CVnum == 1:
                call_plumed_sum_hills(path_to_plumed,'HILLS.ALL',ndim=CVnum, ming=[CV1_grid_limits[0]/ CV1_scaling], maxg=[CV1_grid_limits[1]/ CV1_scaling])
            elif CVnum == 2:
                #Changing input unit from Angstrom to nm or degree to radian
                call_plumed_sum_hills(path_to_plumed,'HILLS.ALL',ndim=CVnum, ming=[CV1_grid_limits[0]/ CV1_scaling,CV2_grid_limits[0]/ CV2_scaling],
                     maxg=[CV1_grid_limits[1]/ CV1_scaling,CV2_grid_limits[1]/ CV2_scaling])

    else:
        print("Calling call_plumed_sum_hills")
        #call_plumed_sum_hills(path_to_plumed,"HILLS")
        if CV1_grid_limits == None:
            call_plumed_sum_hills(path_to_plumed,"HILLS",CVnum)
        else:
            #Changing input unit from Angstrom to nm or degree to radian
            if CVnum == 1:
                call_plumed_sum_hills(path_to_plumed,'HILLS',ndim=CVnum, ming=[CV1_grid_limits[0]/ CV1_scaling], maxg=[CV1_grid_limits[1]/ CV1_scaling])
            elif CVnum == 2:
                #Changing input unit from Angstrom to nm or degree to radian
                call_plumed_sum_hills(path_to_plumed,'HILLS',ndim=CVnum, ming=[CV1_grid_limits[0]/ CV1_scaling,CV2_grid_limits[0]/ CV2_scaling], maxg=[CV1_grid_limits[1]/ CV1_scaling,CV2_grid_limits[1]/ CV2_scaling])
        HILLSFILELIST=['HILLS']
        #Single COLVAR file
        COLVARFILELIST=['COLVAR']

    print("")
    print("COLVAR files:", COLVARFILELIST)
    print("HILLS files:", HILLSFILELIST)
    ###########################################
    # 0 K PES curve for comparison on plot
    ##########################################
    PotCurve=True
    if PotCurve==True:
        #Getting 0 Kelvin potential energy curve from file
        #File should be :  X-value: Torsion in Deg  Y-value: Energy in hartree
        potcurve_degs=[]
        potcurve_energy_au=[]
        try:
            with open("potcurve") as potfile:
                for line in potfile:
                    if '#' not in line:
                        potcurve_degs.append(float(line.split()[0]))
                        potcurve_energy_au.append(float(line.split()[1]))
            potcurve_energy_kcal=np.array(potcurve_energy_au)*627.509
            potcurve_Relenergy_kcal=potcurve_energy_kcal-min(potcurve_energy_kcal)
        except FileNotFoundError:
            PotCurve=False
            print("File potcurve not found. Add file if desired.")
    ########################################



    #READING fes.dat
    #Reaction coordinates (radian if torsion)
    rc=[]
    rc2=[]
    #Free energy (kJ/mol)
    free_energy=[]

    #Derivative of Free Energy vs. reaction-coordinate. Probably not useful
    derivG=[]
    derivG2=[]

    #Reading file
    ##! FIELDS dihed1 dihed2 file.free der_dihed1 der_dihed2
    with open("fes.dat") as fesfile:
        for line in fesfile:
            if '#' not in line and len(line.split()) > 0:
                if CVnum==1:
                    rc.append(float(line.split()[0]))
                    free_energy.append(float(line.split()[1]))
                    derivG.append(float(line.split()[2]))
                else:
                    rc.append(float(line.split()[0]))
                    rc2.append(float(line.split()[1]))
                    free_energy.append(float(line.split()[2]))
                    derivG.append(float(line.split()[3]))
                    derivG2.append(float(line.split()[4]))


    #READ HILLS. Only necessary for Well-Tempered Metadynamics and plotting of Gaussian height
    if WellTemp==True:
        time_hills=[]
        gaussheight=[]
        time_hills_list=[]
        gaussheightkcal_list=[]
        for hillsf in HILLSFILELIST:
            with open(hillsf) as hillsfx:
                for line in hillsfx:
                    if 'FIELDS' in line:
                        biasfcolnum=int(line.split().index('biasf'))
                    if '#' not in line:
                        if biasfcolnum==6:
                            if len(line) > 25:
                                time_hills.append(float(line.split()[0]))
                                gaussheight.append(float(line.split()[3]))
                        if biasfcolnum==8:
                            if len(line) > 25:
                                time_hills.append(float(line.split()[0]))
                                gaussheight.append(float(line.split()[5]))
            gaussheight_kcal=np.array(gaussheight)/energy_scaling
            time_hills_list.append(time_hills)
            gaussheightkcal_list.append(gaussheight_kcal)
            time_hills=[];gaussheight_kcal=[];gaussheight=[]

    #READ COLVAR
    time=[]
    colvar_value=[]
    colvar2_value=[]
    biaspot_value=[]

    colvar_value_deg_list=[]
    colvar2_value_deg_list=[]
    biaspot_value_kcal_list=[]
    time_list=[]
    finalcolvar_value_list=[]
    finalcolvar2_value_list=[]

    print("CVnum:", CVnum)
    for colvarfile in COLVARFILELIST:
        with open(colvarfile) as colvarf:
            for line in colvarf:
                if 'FIELDS' in line:
                    fields=line.split()[2:]
                    number_of_fields=len(fields)
                #    biascolnum = [i for i, s in enumerate(line.split()) if 'bias' in s][0]
                if '#' not in line:
                    if CVnum == 1:
                        if number_of_fields >= 3:
                            if len(line) > 25:
                                time.append(float(line.split()[0]))
                                colvar_value.append(float(line.split()[1]))
                                biaspot_value.append(float(line.split()[2]))
                    elif CVnum == 2:
                        if number_of_fields >= 4:
                            if len(line) > 10:
                                time.append(float(line.split()[0]))
                                colvar_value.append(float(line.split()[1]))
                                colvar2_value.append(float(line.split()[2]))
                                biaspot_value.append(float(line.split()[3]))
                    else:
                        print("unknown format of COLVAR file. More than 2 CVs ??")
                        ashexit()

    #convert to deg if torsion/angle
    if CV1_type.upper()=='TORSION' or CV1_type.upper()=='ANGLE' or CV1_type.upper()=='DIHEDRAL':
        #General CV1 scaling factor for radian to deg
        #Scaling
        rc_deg=np.array(rc)*CV1_scaling
        final_rc=rc_deg
        colvar_value_deg=np.array(colvar_value)*CV1_scaling
        # New. For multiple COLVAR files we create lists of colvar_value_deg, colvar2_value_deg and biaspot_value_kcal
        colvar_value_deg_list.append(colvar_value_deg)
        finalcolvar_value_list=colvar_value_deg_list

    #Possible conversion from nm to Angstrom
    elif CV1_type.upper()=='RMSD' or CV1_type.upper()=='DISTANCE' or CV1_type.upper()=='BOND':
        #General CV1 scaling factor for nm to Angstrom (default)
        rc_ang=np.array(rc)*CV1_scaling
        final_rc=rc_ang
        #print("final_rc:", final_rc)
        colvar_value=np.array(colvar_value)*CV1_scaling
        #print("colvar_value:", colvar_value)
        finalcolvar_value_list.append(colvar_value)
        #print("finalcolvar_value_list:", finalcolvar_value_list)
    else:
        finalcolvar_value_list.append(colvar_value)

    #convert to deg if torsion/angle
    if CV2_type != None:
        if CV2_type.upper()=='TORSION' or CV2_type.upper()=='ANGLE' or CV1_type.upper()=='DIHEDRAL':
            #General CV1 scaling factor for radian to deg
            rc2_deg=np.array(rc2)*CV2_scaling
            final_rc2=rc2_deg
            colvar2_value_deg=np.array(colvar2_value)*CV2_scaling
            # New. For multiple COLVAR files we create lists of colvar_value_deg, colvar2_value_deg and biaspot_value_kcal
            colvar2_value_deg_list.append(colvar2_value_deg)
            finalcolvar2_value_list=colvar2_value_deg_list
        #Possible conversion from nm to Angstrom
        elif CV2_type.upper()=='RMSD' or CV2_type.upper()=='DISTANCE':
            #General CV1 scaling factor for nm to Angstrom (default)
            rc2_ang=np.array(rc2)*CV2_scaling
            final_rc2=rc2_ang
            colvar2_value=np.array(colvar2_value)*CV2_scaling
            finalcolvar2_value_list.append(colvar2_value)
        else:
            finalcolvar2_value_list.append(colvar2_value)

        #Possible energy conversion
        biaspot_value_kcal=np.array(biaspot_value)/energy_scaling


        biaspot_value_kcal_list.append(biaspot_value_kcal)
        time_list.append(time)
        time=[];biaspot_value_kcal=[];colvar2_value_deg=[];colvar_value_deg=[]
        biaspot_value=[];colvar2_value=[];colvar_value=[]


    print("final_rc:", final_rc)
    #Convert free energy from kJ/mol to kcal/mol
    free_energy_kcal=np.array(free_energy)/energy_scaling
    print("free_energy_kcal:", free_energy_kcal)
    Relfreeenergy_kcal=free_energy_kcal-min(free_energy_kcal)
    print("Relfreeenergy_kcal:", Relfreeenergy_kcal)



    ###################
    # Matplotlib part
    ###################
    print("Data preparation done")
    print("Now plotting via Matplotlib")

    if CVnum==1:
        print("Making plots for 1 CV:")
        #Space between subplots
        plt.subplots_adjust(hspace=0.6)
        plt.subplots_adjust(wspace=0.4)
        #Subplot 1: Free energy surface. From fes.dat via HILLS file (single-walker) or HILLS.X files (multiple-walker)
        plt.subplot(2, 2, 1)
        plt.gca().set_title('Free energy vs. CV', fontsize='small', style='italic', fontweight='bold')
        plt.xlabel('{} ({})'.format(CV1_type,finalcvunit_1), fontsize='small')
        plt.ylabel('Energy (kcal/mol)', fontsize='small')
        if CV1_type=='Torsion':
            plt.xlim([-180,180])
        #plt.plot(rc_deg, free_energy_kcal, marker='o', linestyle='-', markerwidth is , linewidth=1, label='G (kcal/mol)')
        plt.plot(final_rc, Relfreeenergy_kcal, marker='o', linestyle='-', linewidth=1, markersize=3, label='G({} K)'.format(temperature))
        if PotCurve==True:
            plt.plot(potcurve_degs, potcurve_Relenergy_kcal, marker='o', linestyle='-', markersize=3, linewidth=1, label='E(0 K)', color='orange')
        plt.legend(shadow=False, frameon=False, fontsize='xx-small', loc='upper left')

        #Subplot 2: CV vs. time. From COLVAR file/files.
        plt.subplot(2, 2, 2)
        plt.gca().set_title('CV vs. time', fontsize='small', style='italic', fontweight='bold')
        plt.xlabel('Time (ps)', fontsize='small')
        plt.ylabel('{} ({})'.format(CV1_type,finalcvunit_1), fontsize='small')
        #New: Using first timelist to get x-axis limit
        plt.xlim([0,max(time_list[0])+5])

        #New. For MW-MTD we have multiple trajectories. Time should be the same
        for num,(t,cv) in enumerate(zip(time_list,finalcolvar_value_list)):
            plt.plot(t, cv, marker='o', linestyle='-', linewidth=0.5, markersize=2, label='Walker'+str(num))
        #lg = plt.legend(shadow=True, fontsize='xx-small', bbox_to_anchor=(1.3, 1.0), loc='upper right')

        #Subplot 3: Bias potential from COLVAR
        plt.subplot(2, 2, 3)
        #plt.title.set_text('Bias potential')
        plt.gca().set_title('Bias potential', fontsize='small', style='italic', fontweight='bold')
        plt.xlabel('{} ({})'.format(CV1_type,finalcvunit_1), fontsize='small')
        plt.ylabel('Bias potential (kcal/mol)', fontsize='small')
        if CV1_type=='Torsion':
            plt.xlim([-180,180])
        #elif CV=='RMSD':
        #    plt.xlim([min(),180])
        for num,(cv,biaspot) in enumerate(zip(finalcolvar_value_list,biaspot_value_kcal_list)):
            print("len cv:", len(cv))
            print("len biaspot:", len(biaspot))
            plt.scatter(cv, biaspot, marker='o', linestyle='-', s=3, linewidth=1, label='Walker'+str(num))
        #lg2 = plt.legend(shadow=True, fontsize='xx-small', bbox_to_anchor=(0.0, 0.0), loc='lower left')

        if WellTemp==True:
            #Subplot 4: Gaussian height from HILLS
            plt.subplot(2, 2, 4)
            plt.gca().set_title('G-height vs. time', fontsize='small', style='italic', fontweight='bold')
            plt.xlabel('Time (ps)', fontsize='small')
            plt.ylabel('G-height (kcal/mol)', fontsize='small')
            plt.xlim([0,max(time_hills_list[0])+5])
            plt.ylim([0,min(gaussheightkcal_list[0])*50])
            for num,(th,gh) in enumerate(zip(time_hills_list,gaussheightkcal_list)):
                #plt.scatter(th, gh, marker='o', linestyle='-', s=3, linewidth=1, label='G height')
                plt.plot(th, gh, marker='o', linestyle='-', markersize=2, linewidth=0.5, label='W'+str(num))
            plt.legend(shadow=True, fontsize=3, loc='lower right', bbox_to_anchor=(1.2, 0.0))

    elif CVnum==2:
        print("Making plots for 2 CV:")

        def flatten(list):
            return [item for sublist in list for item in sublist]

        #2CV-MW plots will be too messy so combinining walker information

        finalcolvar_value_list_flat=flatten(finalcolvar_value_list)
        finalcolvar2_value_list_flat=flatten(finalcolvar2_value_list)
        biaspot_value_kcal_list_flat=flatten(biaspot_value_kcal_list)
        time_hills_flat=flatten(time_hills)
        time_flat=flatten(time_list)
        gaussheight_kcal_flat=flatten(gaussheight_kcal)

        #Space between subplots
        plt.subplots_adjust(hspace=0.6)
        plt.subplots_adjust(wspace=0.6)

        #Subplot 1: Free energy surface
        plt.subplot(2, 2, 1)
        plt.gca().set_title('Free energy vs. CV', fontsize='small', style='italic', fontweight='bold')
        plt.xlabel('{} ({})'.format(CV1_type,CV1_indices), fontsize='small')
        plt.ylabel('{} ({})'.format(CV2_type,CV2_indices), fontsize='small')
        if CV1_type=='Torsion':
            plt.xlim([-180,180])
            plt.ylim([-180,180])
        else:
            print("Subplot 1 free energy surface")
            print("Choosing sensible x and y values based on min and max")
            #print("final_rc:", final_rc)
            #print("final_rc2:", final_rc2)
            #min_x=min(final_rc)
            #max_x=max(final_rc)
            #min_y=min(final_rc2)
            #max_y=max(final_rc2)
            #plt.xlim([min_x,max_x])
            #plt.ylim([min_y,max_y])
            #plt.xlim(CV1_plot_limits)
            #plt.ylim(CV2_plot_limits)
        cm = plt.cm.get_cmap(colormap)
        colorscatter=plt.scatter(final_rc, final_rc2, c=Relfreeenergy_kcal, marker='o', linestyle='-', linewidth=1, cmap=cm)
        cbar = plt.colorbar(colorscatter)
        cbar.set_label('ΔG (kcal/mol)', fontweight='bold', fontsize='xx-small')

        #Subplot 2: CV vs. time
        plt.subplot(2, 2, 2)
        plt.gca().set_title('CV vs. time', fontsize='small', style='italic', fontweight='bold')
        plt.xlabel('{} ({})'.format(CV1_type,CV1_indices), fontsize='small')
        plt.ylabel('{} ({})'.format(CV2_type,CV2_indices), fontsize='small')
        #plt.xlim([0,max(time)+5])
        cm = plt.cm.get_cmap('RdYlBu_r')
        colorscatter=plt.scatter(finalcolvar_value_list_flat, finalcolvar2_value_list_flat, c=time_flat, marker='o', s=2, linestyle='-', linewidth=1, cmap=cm)
        cbar = plt.colorbar(colorscatter)
        #cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Time (ps)', fontweight='bold', fontsize='xx-small')

        #Subplot 3: Bias potential
        plt.subplot(2, 2, 3)
        plt.gca().set_title('Bias potential', fontsize='small', style='italic', fontweight='bold')
        plt.xlabel('{} ({})'.format(CV1_type,CV1_indices), fontsize='small')
        plt.ylabel('{} ({})'.format(CV2_type,CV2_indices), fontsize='small')
        cm = plt.cm.get_cmap(colormap)
        colorscatter2=plt.scatter(finalcolvar_value_list_flat, finalcolvar2_value_list_flat, c=biaspot_value_kcal_list_flat, marker='o', linestyle='-', linewidth=1, cmap=cm)
        cbar2 = plt.colorbar(colorscatter2)
        cbar2.set_label('Biaspot (kcal/mol)', fontweight='bold', fontsize='xx-small')
        #lg = plt.legend(fontsize='xx-small', bbox_to_anchor=(1.05, 1.0), loc='lower left')

        #Subplot 4: Gaussian height
        plt.subplot(2, 2, 4)
        plt.gca().set_title('G height vs. time', fontsize='small', style='italic', fontweight='bold')
        plt.xlabel('Time (ps)', fontsize='small')
        plt.ylabel('G height (kcal/mol)', fontsize='small')
        plt.xlim([0,max(time_hills_list[0])+5])
        #plt.xlim([0,max(time_hills_list[0])+5])
        plt.ylim([0, min(gaussheightkcal_list[0]) * 100])
        for num,(th,gh) in enumerate(zip(time_hills_list,gaussheightkcal_list)):
            plt.plot(th, gh, marker='o', linestyle='-', markersize=2, linewidth=0.5, label='W'+str(num))
        #plt.legend(shadow=True, fontsize='xx-small')
        plt.legend(fontsize=3, bbox_to_anchor=(1.2, 0.0), loc='lower right')

    #Saving figure
    maxtime=int(max(time_list[0]))
    plt.savefig("MTD_Plot-"+str(maxtime)+"ps"+".png",
                dpi=300,
                format='png')

    print("Plotted to file: ", "MTD_Plot-"+str(maxtime)+"ps"+".png"  )
    if Plot_To_Screen is True:
        plt.show()



#Possibly obsolete
def read_plumed_inputfile(plumedinputfile="plumed.in"):
    #Get temperature from plumed.in in dir or dir above
    dihed1atoms=[]
    dihed2atoms=[]
    angle1atoms=[]
    angle2atoms=[]
    distance1atoms=[]
    distance2atoms=[]
    print("")
    try:
        with open(plumedinputfile) as pluminpfile:
            print("Found plumed inputfile file. Reading variables")
            for line in pluminpfile:
                if '#' not in line:
                    if 'TORSION' in line:
                        CV='Torsion'
                        cvunit='°'
                        if len(dihed1atoms) > 0:
                            x=line.split()[-1]
                            y=line.split('=')[-1]
                            for z in y.split(','):
                                dihed2atoms.append(int(z))
                        else:
                            x=line.split()[-1]
                            y=line.split('=')[-1]
                            for z in y.split(','):
                                dihed1atoms.append(int(z))

                    elif 'RMSD' in line:
                        CV='RMSD'
                        #The unit we will plot
                        cvunit='Å'
                    if 'TEMP' in line:
                        for x in line.split():
                            if 'TEMP' in x:
                                temperature=float(x.split('=')[1])
        print("Found temperature:", temperature)
    except:
        print("Found no plumed.in in dir")
        print("Trying dir above...")
        try:
            with open("../plumed.in") as pluminpfile:
                print("Found plumed.in file. Reading variables")
                for line in pluminpfile:
                    if '#' not in line:
                        if 'TORSION' in line:
                            CV = 'Torsion'
                            cvunit = '°'
                            if len(dihed1atoms) > 0:
                                x=line.split()[-1]
                                y=line.split('=')[-1]
                                for z in y.split(','):
                                    dihed2atoms.append(int(z))
                            else:
                                x=line.split()[-1]
                                y=line.split('=')[-1]
                                for z in y.split(','):
                                    dihed1atoms.append(int(z))
                        elif 'DISTANCE' in line:
                            CV = 'Distance'
                            #What we end up with
                            cvunit = 'Å'
                        elif 'ANGLE' in line:
                            CV = 'Angle'
                            cvunit = '°'
                            if len(angle1atoms) > 0:
                                x=line.split()[-1]
                                y=line.split('=')[-1]
                                for z in y.split(','):
                                    angle2atoms.append(int(z))
                            else:
                                x=line.split()[-1]
                                y=line.split('=')[-1]
                                for z in y.split(','):
                                    angle1atoms.append(int(z))
                        elif 'DISTANCE' in line:
                            CV = 'Distance'
                            cvunit = 'Å'
                            if len(distance1atoms) > 0:
                                x = line.split()[-1]
                                y = line.split('=')[-1]
                                for z in y.split(','):
                                    distance2atoms.append(int(z))
                            else:
                                x = line.split()[-1]
                                y = line.split('=')[-1]
                                for z in y.split(','):
                                    distance1atoms.append(int(z))

                        elif 'ANGLE' in line:
                            CV = 'Angle'
                            cvunit = '°'
                            if len(angle1atoms) > 0:
                                x=line.split()[-1]
                                y=line.split('=')[-1]
                                for z in y.split(','):
                                    angle2atoms.append(int(z))
                            else:
                                x=line.split()[-1]
                                y=line.split('=')[-1]
                                for z in y.split(','):
                                    angle1atoms.append(int(z))
                        elif 'RMSD' in line:
                            CV = 'RMSD'
                            #The unit we will plot
                            cvunit = 'Å'
                        if 'TEMP' in line:
                            for x in line.split():
                                if 'TEMP' in x:
                                    temperature=float(x.split('=')[1])
            print("Found temperature:", temperature)
        except:
            print("Unknown exception occurred when reading plumed.in")
            print("Setting temp to unknown")
            temperature="Unknown"


#Old Interface to Plumed (deprecated)
#PLUMED_ASH class: Works with ASE but not confirmed to work with OpenMM at present.

class plumed_ASH():
    def __init__(self, path_to_plumed_kernel=None, bias_type="MTD", fragment=None, CV1_type=None, CV1_indices=None,
                CV2_type=None, CV2_indices=None,
                temperature=300.0, hills_file="HILLS", colvar_file="COLVAR", height=0.01243, sigma1=None, sigma2=None, biasfactor=6.0, timestep=None,
                stride_num=10, pace_num=500, dynamics_program=None,
                numwalkers=None, debug=False):
        print_line_with_mainheader("Plumed ASH interface")

        if dynamics_program == None:
            print("Please specify dynamics program: e.g. dynamics_program=\"ASE\" or dynamics_program=\"OpenMM\"" )
            ashexit()
        # Making sure both Plumed kernel and Python wrappers are available
        if path_to_plumed_kernel == None:
            print("plumed_MD requires path_to_plumed_kernel argument to be set")
            print("Should point to: /path/to/libplumedKernel.so or /path/to/libplumedKernel.dylib")
            ashexit()
        #Path to Plumed (used by MTD_analyze)
        if '.dylib' in path_to_plumed_kernel:
            self.path_to_plumed=path_to_plumed_kernel.replace("/lib/libplumedKernel.dylib","")
        else:
            self.path_to_plumed=path_to_plumed_kernel.replace("/lib/libplumedKernel.so","")

        try:
            import plumed
        except:
            print("Found no plumed library. Install via: pip install plumed")
            ashexit()
        if timestep==None:
            print("timestep= needs to be provided to plumed object")
            ashexit()
        self.plumed=plumed
        self.plumedobj=self.plumed.Plumed(kernel=path_to_plumed_kernel)

        #Basic settings in Plumed object
        self.plumedobj.cmd("setMDEngine","python")
        #Timestep needs to be set
        self.plumedobj.cmd("setTimestep", timestep)
        #Not sure about KbT
        #self.plumedobj.cmd("setKbT", 2.478957)
        self.plumedobj.cmd("setNatoms",fragment.numatoms)
        self.plumedobj.cmd("setLogFile","plumed.log")

        self.debug=debug

        #Initialize object
        self.plumedobj.cmd("init")

        #Choose Plumed units based on what the dynamics program is:
        #By using same units as dynamics program, we can avoid unit-conversion of forces
        if dynamics_program == "ASE":
            print("Dynamics program ASE is recognized. Setting Plumed units to Angstrom (distance), eV (energy) and ps (time).")
            print("sigma and height values should reflect this. ")
            self.plumed_length_unit="A" #Plumed-label for Angstrom
            self.plumed_energy_unit="eV"
            self.plumed_time_unit="ps"
        elif dynamics_program == "OpenMM":
            print("Dynamics program OpenMM is recognized. Setting Plumed units to nm (distance), kJ/mol (energy) and ps (time).")
            print("sigma and height values should reflect this. ")
            self.plumed_length_unit="nm"
            self.plumed_energy_unit="kj/mol"
            self.plumed_time_unit="ps"
        else:
            print("unknown dynamics_program. Exiting")
            ashexit()
        self.plumedobj.cmd("readInputLine","UNITS LENGTH={} ENERGY={} TIME={}".format(self.plumed_length_unit,self.plumed_energy_unit,self.plumed_time_unit))
        self.CV1_type=CV1_type
        self.CV2_type=CV2_type
        self.CV1_indices=CV1_indices
        self.CV2_indices=CV2_indices
        self.temperature=temperature
        print("")
        print("Settings")
        print("")
        print("Path to Plumed kernel:", path_to_plumed_kernel)
        print("Dynamics program:", dynamics_program)
        print("Bias type:", bias_type)
        print("CV1 type:", self.CV1_type)
        print("CV1 indices:", self.CV1_indices)
        print("CV2 type:", self.CV2_type)
        print("CV2 indices:", self.CV2_indices)
        print("")
        print("Temperature:", temperature)
        print("Gaussian height: {} {}".format(height, self.plumed_energy_unit))


        if self.CV1_type.upper() == "DISTANCE" or self.CV1_type.upper() == "RMSD":
            print("Gaussian sigma for CV1: {} {}".format(sigma1,self.plumed_length_unit))
        elif self.CV1_type.upper() == "TORSION" or self.CV1_type.upper() == "ANGLE":
            print("Gaussian sigma for CV1: {} {}".format(sigma1,"rad"))
        else:
            print("unknown CV1 type. Exiting")
            ashexit()
        if self.CV2_type != None:
            if self.CV2_type.upper() == "DISTANCE" or self.CV2_type.upper() == "RMSD":
                print("Gaussian sigma for CV2: {} {}".format(sigma2,self.plumed_length_unit))
            elif self.CV2_type.upper() == "TORSION" or self.CV2_type.upper() == "ANGLE":
                print("Gaussian sigma for CV2: {} {}".format(sigma2,"rad"))
            else:
                print("unknown CV2 type. Exiting")
                ashexit()
        print("Bias factor:", biasfactor)
        print("Timestep: {} {}".format(timestep, self.plumed_time_unit))
        print("")
        print("HILLS filename:", hills_file)
        print("COLVAR filename:", colvar_file)
        print("Stride number", stride_num)
        print("Pace number", pace_num)

        self.numwalkers=numwalkers
        #Store masses
        self.masses=np.array(fragment.list_of_masses,dtype=np.float64)
        print("Masses:", self.masses)

        if bias_type == "MTD":
            #1D metadynamics
            CV1_indices_string = ','.join(map(str, [i+1 for i in CV1_indices])) #Change from 0 to 1 based indexing and converting to text-string
            self.plumedobj.cmd("readInputLine","cv1: {} ATOMS={}".format(self.CV1_type.upper(), CV1_indices_string))
            CV_string="cv1"
            sigma_string=sigma1
            #2D metadynamics if CV2_type has been set
            if CV2_type != None:
                CV2_indices_string = ','.join(map(str, [i+1 for i in CV2_indices]))
                self.plumedobj.cmd("readInputLine","cv2: {} ATOMS={}".format(self.CV2_type.upper(), CV2_indices_string))
                CV_string="cv1,cv2"
                sigma_string=str(sigma1)+","+str(sigma2)

            self.plumedobj.cmd("readInputLine","METAD LABEL=MTD ARG={} PACE={} HEIGHT={} SIGMA={} FILE={} BIASFACTOR={} TEMP={}".format(CV_string,pace_num,
                height, sigma_string, hills_file, biasfactor, temperature))

            #Multiple walker option. Not confirmed to work
            if numwalkers != None:
                print("not ready")
                ashexit()
                self.plumedobj.cmd("readInputLine",str(numwalkers))
                self.plumedobj.cmd("readInputLine","WALKERS_ID=SET_WALKERID") #NOTE: How to set this??
                self.plumedobj.cmd("readInputLine","WALKERS_DIR=../")
                self.plumedobj.cmd("readInputLine","WALKERS_RSTRIDE={}".format(stride_num))
            self.plumedobj.cmd("readInputLine","PRINT STRIDE={} ARG={},MTD.bias FILE={}".format(stride_num, CV_string, colvar_file))
        else:
            print("bias_type not implemented")
            ashexit()
        #Write Plumed info file
        with open ("plumed_ash.info", 'w') as pfile:
            pfile.write("path_to_plumed {}\n".format(self.path_to_plumed))
            pfile.write("CV1_type {}\n".format(self.CV1_type))
            pfile.write("CV1_indices {}\n".format(self.CV1_indices))
            if self.CV2_type != None:
                pfile.write("CV2_type {}\n".format(self.CV2_type))
                pfile.write("CV2_indices {}\n".format(self.CV2_indices))
            pfile.write("temperature {}\n".format(self.temperature))
            pfile.write("plumed_length_unit {}\n".format(self.plumed_length_unit))
            pfile.write("plumed_energy_unit {}\n".format(self.plumed_energy_unit))
            pfile.write("plumed_time_unit {}\n".format(self.plumed_time_unit))


    def run(self, coords=None, forces=None, step=None):
        #module_init_time = time.time()
        #Setting step
        self.plumedobj.cmd("setStep",step)
        #Setting masses. Must be done after Step
        self.plumedobj.cmd("setMasses", np.array(self.masses))

        #Necessary?
        box=np.zeros(9)
        virial=np.zeros(9)
        self.plumedobj.cmd("setBox", box )
        self.plumedobj.cmd("setVirial", virial )

        #Setting current coordinates and forces
        self.plumedobj.cmd("setPositions", coords )
        self.plumedobj.cmd("setForces", forces )

        #Running
        print("Running Plumed bias calculation")
        if self.debug is True:
            print("forces before:", forces)
        self.plumedobj.cmd("calc")
        print("Plumed calc done")
        if self.debug is True:
            print("forces after:", forces)
        #bias = np.zeros((1),dtype=np.float64)
        #self.plumedobj.cmd("getBias", bias )
        # print("forces are now:", forces)
        #print("bias:", bias)
        #print("virial:", virial)
        #print("coords", coords)
        energy=999.9999
        #print_time_rel(module_init_time, modulename='Plumed run', moduleindex=2)
        return energy,forces

    def close(self):
        #Properly close Plumed object (flushes outputfiles etc)
        self.plumedobj.finalize()
