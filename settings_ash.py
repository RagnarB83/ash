import os
import sys
import ash
import time
import configparser
parser = configparser.ConfigParser()

#Defining some ASH settings here

#Whether to use ANSI color escape sequences in output or not.
use_ANSI_color = True

#Print inputfile or not in beginning of job
print_input=True

#Global Connectivity settings
scale = 1.0
tol = 0.1
conndepth = 10

#Path to codes can be defined here (incompatible with git pull though. If regularly updating code via git, use configuration file below instead)
#orcadir='/path/to/orca'
#xtbdir='/path/to/xtbdir'

#Read additional user configuration file if present. Should be present inside ASH source-code dir. TODO: Move to ~ instead?
#Introduced to bypass git conflicts of settings_ash.py

#Format of file ash_user_settings.ini:
#[Settings]
#orcadir = /Applications/orca_4.2.1

ashpath = os.path.dirname(ash.__file__)
parser.read(ashpath+"/"+"ash_user_settings.ini")
try:
    orcadir = parser.get("Settings","orcadir")
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

    
