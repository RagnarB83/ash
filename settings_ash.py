import os
import sys
import ash
import time
import configparser
import distutils
import distutils.util
parser = configparser.ConfigParser()
from pathlib import Path
userhome = str(Path.home())

#Defining some default ASH settings here
# (will be overriden by ash_user_settings file variables if present)

#Whether to use ANSI color escape sequences in output or not.
use_ANSI_color = True

#Print logo or not 
print_logo = True

#Print inputfile or not in beginning of job
print_input=True

#Global Connectivity settings
scale = 1.0
tol = 0.1
conndepth = 10


#Path to codes can be defined here (incompatible with git pull though. If regularly updating code via git, use configuration file below instead)
#orcadir='/path/to/orca'
#xtbdir='/path/to/xtbdir'

#Read additional user configuration file if present. Should be present in $HOME.
#WILL overwrite settings above
#Introduced to bypass git conflicts of settings_ash.py. Also useful if user does not have access to source-code
#Format of file ash_user_settings.ini:
#[Settings]
#orcadir = /Applications/orca_4.2.1

ashpath = os.path.dirname(ash.__file__)
parser.read(userhome+"/"+"ash_user_settings.ini")
try:
    orcadir = parser.get("Settings","orcadir")
    scale = float(parser.get("Settings","scale"))
    tol = float(parser.get("Settings","tol"))
    #Handling Booleans
    use_ANSI_color = bool(distutils.util.strtobool(parser.get("Settings","use_ANSI_color")))
    print_input = bool(distutils.util.strtobool(parser.get("Settings","print_input")))
    print_logo = bool(distutils.util.strtobool(parser.get("Settings","print_logo")))
    
except:
    pass


def init():
    """
    ASH initial output. Used to print header (logo, version etc.), set initial time, print inputscript etc.
    """
    #Initializes time
    global init_time
    init_time=time.time()
    
    #Comment out to skip printing of header
    if print_logo is True:
        ash.print_ash_header()

    print("ASH path:", ashpath)
    print("Using global settings:\nConnectivity scale: {} and tol: {}".format(scale,tol))
    print("Setting initial time")
    print("Note: ASH uses ANSI escape sequences for displaying color. Use less -R to display or set LESS=-R environment variable")
    print("To turn off escape sequences, see settings_ash.py")
    print("")
    
    #Print input script
    if print_input is True:
        inputfilepath=inputfile= os.getcwd()+"/"+sys.argv[0]
        print("Input script:", inputfilepath )
        print(ash.functions_general.BC.WARNING,"="*100)
        with open(inputfilepath) as x: f = x.readlines()
        for line in f:
            print(line,end="")
        print("="*100,ash.functions_general.BC.END)
        print("")

    
