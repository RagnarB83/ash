import numpy as np
import os
import sys
import re
import glob

from functions.functions_general import natural_sort, print_line_with_mainheader
#Interface to Plumed


#PLUMED_ASH class

class plumed_ASH():
    def __init__(self, path_to_plumed_kernel=None, bias_type="1D_MTD", fragment=None, colvar_type=None, colvar_indices=None,
               temperature=300.0, hills_file="HILLS", colvar_file="COLVAR", height=None, sigma=None, biasfactor=None, timestep=None,
               stride_num=10, pace_num=500):
        
        if timestep==None:
            print("timestep= needs to be provided to plumed object")
            exit()

        
        if path_to_plumed_kernel == None:
            print("plumed_MD requires path_to_plumed_kernel argument to be set")
            print("Should point to: /path/to/libplumedKernel.so")
            exit()
        try:
            import plumed
        except:
            print("Found no plumed library. Install via: pip install plumed")
            exit()
        self.plumed=plumed
        
        #Store masses
        self.masses=np.array(fragment.list_of_masses,dtype=np.float64)
        
        if colvar_type=="distance" or colvar_type=="bondlength":
            self.colvar_type="DISTANCE"
        elif colvar_type=="torsion" or colvar_type=="dihedral":
            self.colvar_type="TORSION"
        elif colvar_type=="angle":
            self.colvar_type="ANGLE"
        elif colvar_type=="rmsd":
            self.colvar_type="RMSD"
        else:
            print("Specify colvar_type argumentt.")
            print("Options: distance, angle, torsion, rmsd")
            exit()
        #Change 0 to 1 basedindexing and converting to text-string
        self.colvar_indices_string=','.join(map(str, [i+1 for i in colvar_indices]))


        #os.environ["PLUMED_KERNEL"]=path_to_plumed_library
        #p=plumed.Plumed()
        self.plumedobj=self.plumed.Plumed(kernel=path_to_plumed_kernel)
        
        
        #Basic settings
        self.plumedobj.cmd("setMDEngine","python")
        #Timestep needs to be set
        self.plumedobj.cmd("setTimestep", timestep)
        print("timestep:", timestep)
        #Not sure about KbT
        self.plumedobj.cmd("setKbT", 2.478957)
        self.plumedobj.cmd("setNatoms",fragment.numatoms)
        self.plumedobj.cmd("setLogFile","plumed.log")
        
        #Initialize object
        self.plumedobj.cmd("init")



        #Units: length set to Angstrom and time to ps, energy in hartree
        self.plumedobj.cmd("readInputLine","UNITS LENGTH=A TIME=ps")
        
        
        if bias_type == "1D_MTD":
            #height=1.2
            #sigma=0.35
            #biasfactor=6.0
            #1D metadynamics
            self.plumedobj.cmd("readInputLine","d: {} ATOMS={}".format(self.colvar_type, self.colvar_indices_string))
            #p.cmd("readInputLine","RESTRAINT ARG=d AT=0 KAPPA=1")
            self.plumedobj.cmd("readInputLine","METAD LABEL=MTD ARG=d PACE={} HEIGHT={} SIGMA={} FILE={} BIASFACTOR={} TEMP={}".format(pace_num, 
                height, sigma, hills_file, biasfactor, temperature))
            #p.cmd("WALKERS_N=SET_WALKERNUM")
            #p.cmd("WALKERS_ID=SET_WALKERID")
            #p.cmd("WALKERS_DIR=../")
            #p.cmd("WALKERS_RSTRIDE=10")
            #self.plumedobj.cmd("readInputLine","... METAD")
            self.plumedobj.cmd("readInputLine","PRINT STRIDE={} ARG=d,MTD.bias FILE={}".format(stride_num, colvar_file))
        else:
            print("bias_type not implemented")
            exit()
    def run(self, coords=None, forces=None, step=None):
        #box=array.array('d',[10,0,0,0,10,0,0,0,10])
        #virial=array.array('d',[0,0,0,0,0,0,0,0,0])
        #masses=array.array('d',[1,1])
        #NOTE: Only set masses, charges etc. once ?
        #masses=np.array(fragment.list_of_masses,dtype=np.float64)
        #charges=array.array('d',[0,0])
        #forces=array.array('d',[0,0,0,0,0,0])
        #positions=array.array('d',[0,0,0,1,2,3])
        print("plumed run")
        print("coords:", coords)
        print("forces:", forces)
        print("step:", step)
        self.plumedobj.cmd("setStep",step)
        #Setting masses. Must be done after Step
        self.plumedobj.cmd("setMasses", np.array(self.masses))

        #self.plumedobj.cmd("setBox",box )
        
        #self.plumedobj.cmd("setCharges", charges )
        print("here")
        box=np.zeros(9)
        virial=np.zeros(9)
        self.plumedobj.cmd("setBox", box )
        self.plumedobj.cmd("setVirial", virial )

        self.plumedobj.cmd("setPositions", coords )
        self.plumedobj.cmd("setForces", forces )
        
        
        print("Running calc")
        self.plumedobj.cmd("calc")
        print("Calc done")
        bias = np.zeros((1),dtype=np.float64)
        self.plumedobj.cmd("getBias", bias )
        #Initialize bias array
        
        print("forces are now:", forces)
        print("bias:", bias)
        print("virial:", virial)
        print("coords", coords)
        
        energy=999.9999
        #NOTE: Return bias or modified forces?
        
        return


#Metadynamics visualization tool

def MTD_analyze(path_to_plumed=None, Plot_To_Screen=False):

    print_line_with_mainheader("Metadynamics Analysis Script")
    try:
        import matplotlib.pyplot as plt
    except:
        print("Problem importing matplotlib.")
        

    ###############################
    #USER SETTINGS
    ###############################
    #Run Plumed script to get fes.dat from HILLS
    #Setting PATH and LD_LIBRARY_PATH for PLUMED. LD-lib path to C library may also be required
    if path_to_plumed==None:
        print("Set path_to_plumed argument. Example: MTD_analyze(path_to_plumed=/home/bjornsson/plumed-install)")
        exit()
    
    os.environ['PATH'] = path_to_plumed+'/bin:$PATH'
    os.environ['LD_LIBRARY_PATH'] = path_to_plumed+'/lib:$LD_LIBRARY_PATH'

    #Colormap to use in 2CV plots.
    # Perceptually uniform sequential: viridis, plasma, inferno, magma, cividis
    #Others: # RdYlBu_r
    #See https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    colormap='RdYlBu_r'


    # If WellTempered MetaDynamics (generally recommended) is used. For regular MTD a pointless plot would be generated for Welltemp=True
    WellTemp=True

    #PLUMED uses kJ/mol be default in its files.
    # kJ/mol units in Plumed are assumed by the script, but gets converted to kcal/mol here.
    # Dihedrals and angles are assumed to be in radians from Plumed and are converted to degrees
    # Distances and RMSDs are assumed to be in nm from Plumed and are converted to Å

    ################################
    #END OF USER-REQUIRED SETTINGS
    ################################


    #try:
    #    path=sys.argv[1]
    #    os.chdir(path)
    #    print("Changing dir to path: ", path)
    #except:
    print("Assuming current dir contains COLVAR/HILLS files")


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
            exit()


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
        os.system('plumed sum_hills --hills HILLS.ALL')
    else:
        os.system('plumed sum_hills --hills HILLS')
        #HILLSFILE="HILLS"
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
    pi=3.14159265359
    #Get temperature from plumed.in in dir or dir above
    dihed1atoms=[]
    dihed2atoms=[]
    angle1atoms=[]
    angle2atoms=[]
    distance1atoms=[]
    distance2atoms=[]
    print("")

    try:
        with open("plumed.in") as pluminpfile:
            print("Found plumed.in file. Reading variables")
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

    print("CV:", CV)
    print("CV unit:", cvunit)

    if CV=='Torsion':
        print("Atoms in CV1:", dihed1atoms)
        CV1atoms = dihed1atoms
        if len(dihed2atoms)>0:
            print("Atoms in CV2:", dihed2atoms)
            CV2atoms=dihed2atoms
    elif CV=='Angle':
        print("Atoms in CV1:", angle1atoms)
        CV1atoms = angle1atoms
        if len(angle2atoms)>0:
            print("Atoms in CV2:", angle2atoms)
            CV2atoms = angle2atoms
    elif CV=='Distance':
        print("Atoms in CV1:", distance1atoms)
        CV1atoms = distance1atoms
        if len(distanceatoms)>0:
            print("Atoms in CV2:", distance2atoms)
            CV2atoms = distance2atoms


    #READ HILLS. Only necessary for Well-Tempered Metadynamics and plotting of Gaussian height
    if WellTemp==True:
        time_hills=[]
        gaussheight=[]
        time_hills_list=[]
        gaussheightkcal_list=[]
        for hillsfile in HILLSFILELIST:
            with open(hillsfile) as hillsf:
                for line in hillsf:
                    if 'FIELDS' in line:
                        biasfcolnum=int(line.split().index('biasf'))
                    if '#' not in line:
                        if biasfcolnum==6:
                            time_hills.append(float(line.split()[0]))
                            gaussheight.append(float(line.split()[3]))
                        if biasfcolnum==8:
                            time_hills.append(float(line.split()[0]))
                            gaussheight.append(float(line.split()[5]))
            gaussheight_kcal=np.array(gaussheight)/4.184
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

    for colvarfile in COLVARFILELIST:
        with open(colvarfile) as colvarf:
            for line in colvarf:
                if 'FIELDS' in line:
                    biascolnum = [i for i, s in enumerate(line.split()) if '.bias' in s][0]
                if '#' not in line:
                    try:
                        #1 CVs
                        if biascolnum==4:
                            CVnum=1
                            biaspot_value.append(float(line.split()[2]))
                            time.append(float(line.split()[0]))
                            colvar_value.append(float(line.split()[1]))
                        #2 CVs
                        elif biascolnum==5:
                            CVnum=2
                            biaspot_value.append(float(line.split()[3]))
                            time.append(float(line.split()[0]))
                            colvar_value.append(float(line.split()[1]))
                            colvar2_value.append(float(line.split()[2]))
                        else:
                            print("unknown format of COLVAR file. More than 2 CVs ??")
                            exit()
                    except:
                        pass
        #convert to deg if torsion
        if CV=='Torsion' or CV=='Angle':
            colvar_value_deg=np.array(colvar_value)*180/pi
            colvar2_value_deg=np.array(colvar2_value)*180/pi
            # New. For multiple COLVAR files we create lists of colvar_value_deg, colvar2_value_deg and biaspot_value_kcal
            colvar_value_deg_list.append(colvar_value_deg)
            colvar2_value_deg_list.append(colvar2_value_deg)

            finalcolvar_value_list=colvar_value_deg_list
            finalcolvar2_value_list=colvar2_value_deg_list

        elif CV=='RMSD' or CV=='Distance':
            #Converting from nm to A
            colvar_value=np.array(colvar_value)*10
            colvar2_value=np.array(colvar2_value)*10
            finalcolvar_value_list.append(colvar_value)
            finalcolvar2_value_list.append(colvar2_value)

            #finalcolvar_value_list=np.array(finalcolvar_value_list)
            #finalcolvar2_value_list=np.array(finalcolvar2_value_list)
        else:
            finalcolvar_value_list.append(colvar_value)
            finalcolvar2_value_list.append(colvar2_value)


        #Convert to kcal
        biaspot_value_kcal=np.array(biaspot_value)/4.184


        biaspot_value_kcal_list.append(biaspot_value_kcal)
        time_list.append(time)
        time=[];biaspot_value_kcal=[];colvar2_value_deg=[];colvar_value_deg=[]
        biaspot_value=[];colvar2_value=[];colvar_value=[]

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
    #rc is in rad. convert to deg
    if CV=='Torsion' or CV=='Angle':
        rc_deg=np.array(rc)*180/pi
        rc2_deg=np.array(rc2)*180/pi
        final_rc=rc_deg
        final_rc2=rc2_deg
    #rc is is in nm. convert to Å
    elif CV=='RMSD' or CV=='Distance':
        rc_ang=np.array(rc)*10
        rc2_ang=np.array(rc2)*10
        final_rc=rc_ang
        final_rc2=rc2_ang
    else:
        print("Unknown CV...oops...")
        exit()

    #Convert free energy from kJ/mol to kcal/mol
    free_energy_kcal=np.array(free_energy)/4.184
    Relfreeenergy_kcal=free_energy_kcal-min(free_energy_kcal)

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
        plt.xlabel('{} ({})'.format(CV,cvunit), fontsize='small')
        plt.ylabel('Energy (kcal/mol)', fontsize='small')
        if CV=='Torsion':
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
        plt.ylabel('{} ({})'.format(CV,cvunit), fontsize='small')
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
        plt.xlabel('{} ({})'.format(CV,cvunit), fontsize='small')
        plt.ylabel('Bias potential (kcal/mol)', fontsize='small')
        if CV=='Torsion':
            plt.xlim([-180,180])
        #elif CV=='RMSD':
        #    plt.xlim([min(),180])
        for num,(cv,biaspot) in enumerate(zip(finalcolvar_value_list,biaspot_value_kcal_list)):
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
        plt.xlabel('{} ({})'.format(CV,CV1atoms), fontsize='small')
        plt.ylabel('{} ({})'.format(CV,CV2atoms), fontsize='small')
        if CV=='Torsion':
            plt.xlim([-180,180])
            plt.ylim([-180,180])
        cm = plt.cm.get_cmap(colormap)
        colorscatter=plt.scatter(final_rc, final_rc2, c=Relfreeenergy_kcal, marker='o', linestyle='-', linewidth=1, cmap=cm)
        cbar = plt.colorbar(colorscatter)
        cbar.set_label('ΔG (kcal/mol)', fontweight='bold', fontsize='xx-small')

        #Subplot 2: CV vs. time
        plt.subplot(2, 2, 2)
        plt.gca().set_title('CV vs. time', fontsize='small', style='italic', fontweight='bold')
        plt.xlabel('{} ({})'.format(CV,CV1atoms), fontsize='small')
        plt.ylabel('{} ({})'.format(CV,CV2atoms), fontsize='small')
        #plt.xlim([0,max(time)+5])
        cm = plt.cm.get_cmap('RdYlBu_r')
        colorscatter=plt.scatter(finalcolvar_value_list_flat, finalcolvar2_value_list_flat, c=time_flat, marker='o', s=2, linestyle='-', linewidth=1, cmap=cm)
        cbar = plt.colorbar(colorscatter)
        #cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Time (ps)', fontweight='bold', fontsize='xx-small')

        #Subplot 3: Bias potential
        plt.subplot(2, 2, 3)
        plt.gca().set_title('Bias potential', fontsize='small', style='italic', fontweight='bold')
        plt.xlabel('{} ({})'.format(CV,CV1atoms), fontsize='small')
        plt.ylabel('{} ({})'.format(CV,CV2atoms), fontsize='small')
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