import os
import sys
import ash
import functions_general
import time
import configparser
import distutils
import distutils.util
parser = configparser.ConfigParser()
from pathlib import Path
userhome = str(Path.home())
ashpath = os.path.dirname(ash.__file__)

############################
# ASH DEFAULT SETTINGS
############################
#Defining some default ASH settings here in a dictionary
settings_dict={}

# (will be overriden by ash_user_settings file variables if present)
settings_dict["debugflag"] = False

#Julia usage
settings_dict["load_julia"] = True

#Whether to use ANSI color escape sequences in output or not.
settings_dict["use_ANSI_color"] = True

#Print logo or not 
settings_dict["print_logo"] = True

#Print inputfile or not in beginning of job
settings_dict["print_input"] = True

#Global Connectivity settings
settings_dict["scale"] = 1.0
settings_dict["tol"] = 0.1
settings_dict["conndepth"] = 10


#Read additional user configuration file if present. Should be present in $HOME.
#WILL overwrite default settings above
parser.read(userhome+"/"+"ash_user_settings.ini")

def try_read_setting(stringvalue,datatype):
    try:
        if datatype == "string":
                settings_dict[stringvalue] = str(parser.get("Settings",stringvalue))
        elif datatype == "float":
            settings_dict[stringvalue] = float(parser.get("Settings",stringvalue))
        elif datatype == "int":
            settings_dict[stringvalue] = int(parser.get("Settings",stringvalue))
        elif datatype == "bool":
            if parser.get("Settings",stringvalue) == 'True':
                settings_dict[stringvalue] = True
            elif parser.get("Settings",stringvalue) == 'False':
                settings_dict[stringvalue] = False
        else:
            settings_dict[stringvalue] = parser.get("Settings",stringvalue)
    except:
        pass
        #print("EXCEPTION!!!!. stringvalue:", stringvalue)
        
# Keywords to look up in ash_user_settings.ini
try_read_setting("orcadir","string")
try_read_setting("daltondir","string")
try_read_setting("xtbdir","string")
try_read_setting("psi4dir","string")
try_read_setting("cfourdir","string")
try_read_setting("crestdir","string")
try_read_setting("scale","float")
try_read_setting("tol","float")
try_read_setting("use_ANSI_color","bool")
try_read_setting("print_input","bool")
try_read_setting("print_logo","bool")
try_read_setting("debugflag","bool")
try_read_setting("load_julia","bool")


def init():
    """
    ASH initial output. Used to print header (logo, version etc.), set initial time, print inputscript etc.
    """
    #Initializes time
    global init_time
    init_time=time.time()
    
    #Comment out to skip printing of header
    if settings_dict["print_logo"] is True:
        ash.print_ash_header()

    print("ASH path:", ashpath)
    print("ASH Settings after reading defaults and ~/ash_user_settings.ini : ")
    print(settings_dict)
    print("Setting initial time")
    print("Note: ASH uses ANSI escape sequences for displaying color. Use less -R to display or set LESS=-R environment variable")
    print("To turn off escape sequences, set:   use_ANSI_color = False   in  ~/ash_user_settings.ini")
    print("")
    
    #Print input script
    if settings_dict["print_input"] is True:
        inputfilepath=inputfile= os.getcwd()+"/"+sys.argv[0]
        print("Input script:", inputfilepath )
        print(functions_general.BC.WARNING,"="*100)
        with open(inputfilepath) as x: f = x.readlines()
        for line in f:
            print(line,end="")
        print("="*100,functions_general.BC.END)
        print("")

    
