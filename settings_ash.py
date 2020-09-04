import os
import ash
import time
import configparser
parser = configparser.ConfigParser()

#Defining some ASH settings here

#Whether to use ANSI color escape sequences in output or not.
use_ANSI_color = True

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

file = "ash_user_settings.ini"
ashpath = os.path.dirname(ash.__file__)
parser.read(ashpath+"/"+file)
try:
    orcadir = parser.get("Settings","orcadir")
except:
    pass


#Init function to print header and set time
def init():
    #Comment out to skip printing of header
    ash.print_ash_header()


    
    #Initializes time
    global init_time
    init_time=time.time()
    
    print(f"Using global settings: \n Connectivity scale: {scale} and tol: {tol}")
    print("Setting initial time")
    print("Note: ASH uses ANSI escape sequences for displaying color. Use less -R to display or set LESS=-R environment variable")
    print("To turn off escape sequences, see settings_ash.py")
    