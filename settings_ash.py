import os
import sys
import configparser
from pathlib import Path

import ash
parser = configparser.ConfigParser()
userhome = str(Path.home()) #Path to user's home dir
ashpath = os.path.dirname(ash.__file__) #Path to ASH

# Check if interactive session
interactive_session = bool(getattr(sys, 'ps1', sys.flags.interactive))

############################
# ASH DEFAULT SETTINGS
############################
# Defining some default ASH settings here in a dictionary
settings_dict = {}

# Settings will be overriden by ash_user_settings file variables if present
settings_dict["debugflag"] = False

# Julia usage
settings_dict["load_julia"] = False
settings_dict["julia_library"] = "pythoncall" #pythoncall is default. pyjulia another option

# Whether to use ANSI color escape sequences in output or not.
settings_dict["use_ANSI_color"] = False

# Print logo or not
settings_dict["print_logo"] = True

# Print inputfile or not in beginning of job
settings_dict["print_input"] = True

# Global Connectivity settings
settings_dict["scale"] = 1.0
settings_dict["tol"] = 0.1
settings_dict["conndepth"] = 10
settings_dict["connectivity_code"] = "julia"
settings_dict["nonbondedMM_code"] = "julia"
# Exit command
settings_dict["print_exit_footer"] = True
settings_dict["print_full_timings"] = True

############################
# ASH READ USER SETTINGS
############################

# Read additional user configuration file if present. Should be present in $HOME.
# WILL overwrite default settings above
parser.read(userhome + "/" + "ash_user_settings.ini")


def try_read_setting(stringvalue, datatype):
    try:
        if datatype == "string":
            settings_dict[stringvalue] = str(parser.get("Settings", stringvalue))
        elif datatype == "float":
            settings_dict[stringvalue] = float(parser.get("Settings", stringvalue))
        elif datatype == "int":
            settings_dict[stringvalue] = int(parser.get("Settings", stringvalue))
        elif datatype == "bool":
            if parser.get("Settings", stringvalue) == 'True':
                settings_dict[stringvalue] = True
            elif parser.get("Settings", stringvalue) == 'False':
                settings_dict[stringvalue] = False
        else:
            settings_dict[stringvalue] = parser.get("Settings", stringvalue)
    except:
        pass

#NOTE: Warning. If user added quotation marks around string then things go awry. Look into
# Keywords to look up in ash_user_settings.ini
try_read_setting("orcadir", "string")
try_read_setting("daltondir", "string")
try_read_setting("xtbdir", "string")
try_read_setting("psi4dir", "string")
try_read_setting("cfourdir", "string")
try_read_setting("crestdir", "string")
try_read_setting("connectivity_code", "string")
try_read_setting("nonbondedMM_code", "string")
try_read_setting("scale", "float")
try_read_setting("tol", "float")
try_read_setting("use_ANSI_color", "bool")
try_read_setting("print_input", "bool")
try_read_setting("print_exit_footer", "bool")
try_read_setting("print_full_timings", "bool")
try_read_setting("print_logo", "bool")
try_read_setting("debugflag", "bool")
try_read_setting("load_julia", "bool")
try_read_setting("julia_library", "string")
