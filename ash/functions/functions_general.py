
import os
import sys
import numpy as np
import time
from functools import wraps
import math
import shutil
import re
import atexit

import ash.settings_ash
from ash import ashpath

# ANSI colors: http://jafrog.com/2013/11/23/colors-in-terminal.html
if ash.settings_ash.settings_dict["use_ANSI_color"] is True:
    class BC:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        OKMAGENTA = '\033[95m'
        OKRED = '\033[31m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        END = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
else:
    class BC:
        HEADER = ''
        OKBLUE = ''
        OKGREEN = ''
        OKMAGENTA = ''
        OKRED = ''
        WARNING = ''
        FAIL = ''
        END = ''
        BOLD = ''
        UNDERLINE = ''



def check_program_location(directory,directory_name, bin_name):
    if directory != None:
        finaldirectory = directory
        print(BC.OKGREEN,f"Using directory path provided: {finaldirectory}", BC.END)
    else:
        print(BC.WARNING, f"No {directory_name} argument passed. Attempting to find {directory_name} variable in ASH settings file (~/ash_user_settings.ini)", BC.END)
        try:
            finaldirectory=ash.settings_ash.settings_dict[directory_name]
            print(BC.OKGREEN,f"Using {directory_name} path provided from ASH settings file (~/ash_user_settings.ini): ", finaldirectory, BC.END)
        except KeyError:
            print(BC.WARNING,f"Found no {directory_name} variable in ASH settings file either.",BC.END)
            print(BC.WARNING,f"Checking for {bin_name} in PATH environment variable.",BC.END)
            try:
                finaldirectory = os.path.dirname(shutil.which(f'{bin_name}'))
                print(BC.OKGREEN,f"Found {bin_name} binary in PATH. Using the following directory:", finaldirectory, BC.END)
            except TypeError:
                print(BC.FAIL,f"Found no {bin_name} binary in PATH environment variable either. Giving up.", BC.END)
                ashexit()
    return finaldirectory

#Create ASH environment shell-file in home-dir
#Simple shell script to active ASH environment for future calcs
def create_ash_env_file():
    ash_dir=ash.ashpath
    path_to_python3_dir=os.path.dirname(sys.executable)
    #Create set_environment_ash.sh file
    ash_multiline_string=f"""
#!/bin/bash

#################################
# Setting up ASH environment
#################################

ulimit -s unlimited

ASHPATH={ash_dir}
python3path={path_to_python3_dir}

#PYTHONPATH for finding ASH usually not recommended.
#Better to install into Python environment (pip install)
#export PYTHONPATH=$ASHPATH:$ASHPATH/lib:$PYTHONPATH

export PATH=$python3path:$PATH
export LD_LIBRARY_PATH=$ASHPATH/lib:$LD_LIBRARY_PATH

#################################
echo \"Sourced ASH environment file!\"
echo \"Importing ASH within Python should now work!\"
echo \"ASH is located in $ASHPATH\"
echo \"The Python interpreter that you should be using is located in $python3path \"
    """


    with open(f"{os.path.expanduser('~')}/set_environment_ash.sh", "w") as f:
        f.write(ash_multiline_string)
    print("Created file:   set_environment_ash.sh      in your home-directory.")
    print("Sourcing this file (in your shell) will activate ASH for future shell sessions:")
    print("source ~/set_environment_ash.sh")
    print()
    print("You can add this line to your ~/.bashrc (or ~/.bash_profile or ~/.zshrc) and job-submission script")
    exit()


# Julia load interface
# TODO: Avoid reloading
julia_loaded = False

def is_interactive() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

#General function to exit ASH
#NOTE: By default we exit with errorcode 1
def ashexit(errormessage=None, code=1):
    print(BC.FAIL,"ASH exiting with code:", code, BC.END)

    if errormessage != None:
        print(BC.FAIL,"Error message:", errormessage, BC.END)

    #If in Jupyter notebook, then we do a softer return
    if is_interactive():
        raise SystemExit("ASH exiting due to error")
    else:
        sys.exit(1)

def basename(filename):
    return os.path.splitext(filename)[0]

#Attempt to generally find a 3rd-party program based on path, exename etc.
#Either programdir variable is already set, else we try to find based on programdirname or exename
def find_program(programdir,programdirname,exename,theorynamelabel):
    if programdir == None:
        print(BC.WARNING, f"No {programdirname} argument passed to {theorynamelabel}Theory. Attempting to find {programdir} variable inside settings_ash", BC.END)
        try:
            print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
            finalpath=ash.settings_ash.settings_dict[programdirname]
        except KeyError:
            print(BC.WARNING,f"Found no {programdirname} variable in settings_ash module either.",BC.END)
            try:
                finalpath = os.path.dirname(os.path.dirname(shutil.which(exename)))
                print(BC.OKGREEN,f"Found {exename} executable in PATH. Setting {programdir} to:", finalpath, BC.END)
            except:
                print(BC.FAIL,f"Found no {exename} executable in PATH.", BC.END)
                ashexit()
    else:
        print("Program directory chosen to be:", programdir)
     #Check if dir exists
        if os.path.exists(programdir):
            print("It exists.")
        else:
            print(f"Chosen directory : {programdir} does not exist. Exiting...")
            ashexit()
        finalpath = programdir
    return finalpath

def load_pythoncall():
    print("Now trying pythoncall/juliacall package. This will fail if :\n\
            - Juliacall Python package has not been installed (via pip)\n")
            #- PythonCall julia packages has not been installed (via Julia Pkg)\n")
            #- Julia Hungarian package has not been installed")
    from juliacall import Main as JuliaMain
    JuliaMain.include(ashpath + "/functions/functions_julia.jl")
    return JuliaMain

#def load_pyjulia():
#    print("Now loading PyJulia. This will fail if :\n\
#        - Julia PyCall package has not been installed\n\
#
#    from julia import Main as JuliaMain
#    #NOTE: Reading old Pyjulia function file here instead.
#    JuliaMain.include(ashpath + "/functions/functions_julia_oldpyjulia.jl")
#    return JuliaMain

def load_julia_interface(julia_library=None):
    print("\nCalling Julia interface")

    #If not set (rare) then get the settings_ash value
    if julia_library == None:
        julia_library=ash.settings_ash.settings_dict["julia_library"]

    print("Note: PythonCall/Juliacall is recommended (default).")
    # Loading pythoncall or pyjulia
    print("Library is set to:", julia_library)
    #Global variables
    global julia_loaded
    global JuliaMain

    #Only load if not loaded before
    if julia_loaded is False:
        print("Now loading Julia interface.")
        currtime=time.time()
        #Checking for Julia binary in PATH
        print("This requires a Julia installation available in PATH")
        try:
            juliapath=os.path.dirname(shutil.which('julia'))
            print("Found Julia in dir:", juliapath)
        except TypeError:
            print("Possible Problem. No julia binary found in PATH environment variable.")
            print("Make sure the path to your desired Julia's bin directory is available in your shell-configuration or jobscript")
            print("Will continue as pythoncall may be able to install Julia for you.")
            #ashexit()

        #Importing the necessary interface library
        print("Loading a Python/Julia interface library")
        if julia_library == "pythoncall":
            print("Library: pythoncall/juliacall")
            try:
                JuliaMain = load_pythoncall()
                print("Julia interface successfully loaded")
            except:
                print("Problem loading pythoncall/juliacall.")
                ashexit()
        #elif julia_library == "pyjulia":
        #    try:
        #        JuliaMain = load_pyjulia()
        #        print("Julia interface successfully loaded")
        #    except:
        #        print("Problem loading pyjulia")
        #        ashexit()
        else:
            print("Unknown Julia library:", julia_library)
            ashexit()
        julia_loaded = True  #Means an attempt was made to load Julia.
        print_time_rel(currtime, modulename='loading julia interface', moduleindex=4)
    return JuliaMain.Juliafunctions


# Get ranges of integers from list. Returns string of ranges. Used to communitcate with crest and xtb
# example: input: [1,2,3,4,5,6,20,21,22,23,500,700,701,702,1000,1100,1101]
# output: '1-6,20-23,500,700-702,1000,1100-1101'
def int_ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    l_of_tuples = list(zip(edges, edges))

    newstring = ""
    for i in l_of_tuples:
        if i[0] != i[1]:
            newstring += str(i[0]) + '-' + str(i[1]) + ','
        else:
            newstring += str(i[0]) + ','
    # remove final ,
    newstring = newstring[0:-1]
    return newstring


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn:" + fn.__name__ + " took " + str(t2 - t1) + " seconds")
        return result

    return measure_time


# Grep-style function to find a line in file and return a list of words
# TODO: Make more advanced
def pygrep(string, file, errors=None):
    with open(file, errors=errors) as f:
        for line in f:
            if string in line:
                stringlist = line.split()
                return stringlist


# Multiple match version. Replace pygrep ?
def pygrep2(string, file, print_output=False, errors=None):
    l = []
    with open(file, errors=errors) as f:
        for line in f:
            if string in line:
                l.append(line)
    if print_output is True:
        print(*l)
    return l

#Simple function to do find and replace string in file
def find_replace_string_in_file(file,findstring,replstring):
    with open(file, 'r') as f:
        filedata = f.read()
    # Replace the target string
    filedata = filedata.replace(findstring, replstring)
    # Write the file out again
    with open(file, 'w') as f:
        f.write(filedata)


# Give difference of two lists, sorted. List1: Bigger list
def listdiff(list1, list2):
    diff = (list(set(list1) - set(list2)))
    diff.sort()
    return diff


# Range function for floats
# Using round to deal with floating-point problem : 0.6+0.3 =0.89999
def frange(start, stop=None, step=None, rounddigits=4):
    # if stop and step argument is None set start=0.0 and step = 1.0
    start = float(start)
    if stop is None:
        stop = start + 0.0
        start = 0.0
    if step is None:
        step = 1.0
    # print("start= ", start, "stop= ", stop, "step= ", step)
    count = 0
    while True:
        temp = round(float(start + count * step), rounddigits)
        # print("temp:", temp)
        if step > 0 and temp >= stop:
            break
        elif step < 0 and temp <= stop:
            break
        yield temp
        count += 1


#Function to find n highest max values and indices
#Quite ugly
def n_max_values(l,num):
    maxweight=max(l)
    maxweight_index=l.index(max(l))
    current_indices=[maxweight_index]
    #Looping over

    for step in range(1,num):
        current_highest_val=0.0
        for index,weight in enumerate(l):
            if index in current_indices: #Skipping previously found max
                pass
            else:
                if weight > current_highest_val:
                    current_highest_val=weight
                    current_highest_index=index
        current_indices.append(current_highest_index)
    return current_indices


# FUNCTIONS TO PRINT MODULE AND SUBMODULE HEADERS


# Print string if printlevel equals or larger than reference
def print_if_level(var, printlevel,refprintlevel):
    if printlevel >= refprintlevel:
        print(var)

# Debug print. Behaves like print but reads global debug var first
def printdebug(string, var=''):
    if ash.settings_ash.settings_dict["debugflag"] is True:
        print(BC.OKRED, string, var, BC.END)

# mainmodule header
def print_line_with_mainheader(line):
    length = len(line)
    offset = 12
    outer_line = f"{BC.OKGREEN}{'#' * (length + offset)}{BC.END}"
    midline = f"{BC.OKGREEN}#{' ' * (length + offset - 2)}#{BC.END}"
    inner_line = f"{BC.OKGREEN}#{' ' * (offset//2 - 1)}{BC.BOLD}{line}{' ' * (offset//2 - 1)}#{BC.END}"
    print("\n")
    print(outer_line.center(80))
    print(midline.center(80))
    print(inner_line.center(80))
    print(midline.center(80))
    print(outer_line.center(80))
    #print("\n")


# Submodule header
def print_line_with_subheader1(line):
    print("")
    print(f"{BC.OKBLUE}{'-' * 80}{BC.END}")
    print(f"{BC.OKBLUE}{BC.BOLD}{line.center(80)}{BC.END}")
    print(f"{BC.OKBLUE}{'-' * 80}{BC.END}")
    print("")

# Submodule header
def print_line_with_subheader1_end():
    print("")
    print(f"{BC.OKBLUE}{'-' * 80}{BC.END}")

# Smaller header
def print_line_with_subheader2(line):
    print("")
    length = len(line)
    print(f"{BC.OKBLUE}{'-' * length}{BC.END}")
    print(f"{BC.OKBLUE}{BC.BOLD}{line}{BC.END}")
    print(f"{BC.OKBLUE}{'-' * length}{BC.END}")


# Inserts line into file for matched string.
# option: Once=True means only added for first match
def insert_line_into_file(file, string, addedstring, Once=True):
    Added = False
    with open(file, 'r') as ffr:
        contents = ffr.readlines()
    with open(file, 'w') as ffw:
        for l in contents:
            ffw.write(l)
            if string in l:
                if Added is False:
                    ffw.write(addedstring + '\n')
                    if Once is True:
                        Added = True


def blankline():
    print("")


# Can variable be converted into integer
def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


# Is integer odd
def isodd(n):
    if (n % 2) == 0:
        return False
    else:
        return True


# Compare sign of two numbers. Return True if same sign, return False if opposite sign
def is_same_sign(a, b):
    if a * b < 0:
        return False
    elif a * b > 0:
        return True


# Is it possible to interpret string/number as float.
# Note: integer variable/string can be interpreted. See also is_string_float_withdecimal below
def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Is string a float with single decimal point
def is_string_float_withdecimal(s):
    # Checking if single . in string
    if s.count('.') != 1:
        return False
    # Check if number can be interpreted as float
    try:
        float(s)
        return True
    except ValueError:
        return False


#Search list of lists. Returns list-index if match

def search_list_of_lists_for_index(i, l):
    return next((c for c, f in enumerate(l) if i in f), None)

#convert list of lists to dict
def create_conn_dict(l):
    index = {}
    for c, sublist in enumerate(l):
        for value in sublist:
            if value not in index:
                index[value] = c
    return index


def search_list_of_lists_for_index_old(i,l):
    for c,f in enumerate(l):
        if i in f:
            return c
            break

# Check if list of integers is sorted or not.
def is_integerlist_ordered(list):
    list_s = sorted(list)
    if list == list_s:
        return True
    else:
        return False


def islist(l):
    if type(l) == list:
        return True
    else:
        return False


# Read lines of file by slurping.
#def readlinesfile(filename):
#    try:
#        f = open(filename)
#        out = f.readlines()
#        f.close()
#    except IOError:
#        print('File %s does not exist!' % (filename))
#        ashexit()
#    return out


# Find substring of string between left and right parts
def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]


# Read list of integers from file. Output list of integers. Ignores blanklines, return chars, non-int characters
# offset option: shifts integers by a value (e.g. 1 or -1)
def read_intlist_from_file(filename, offset=0):
    intlist = []
    try:
        with open(filename, "r") as f:
            for line in f:
                for l in line.split():
                    # Removing non-numeric part
                    l = ''.join(i for i in l if i.isdigit())
                    if isint(l):
                        intlist.append(int(l) + offset)
    except FileNotFoundError:
        print(f"File '{filename}' does not exists!")
        ashexit()
    intlist.sort()
    return intlist


# Read list of flaots from file. Output list of floats.
# Works for single-line with numbers and multi-lines
def read_floatlist_from_file(filename):
    floatlist = []
    try:
        with open(filename, "r") as f:
            for line in f:
                for l in line.split():
                    if isfloat(l):
                        floatlist.append(float(l))
    except FileNotFoundError:
        print(f"File '{filename}' does not exists!")
        ashexit()
    floatlist.sort()
    return floatlist


#Read simple datafile (e.g. .dat and .stk files from ORCA).
# Separator is Python default whitespace.
def read_datafile(filename, separator=None):
    x=[]
    y=[]
    with open(filename) as f:
        for line in f:
            if '#' not in line:
                if separator == None:
                    x.append(float(line.split()[0]))
                    y.append(float(line.split()[1]))
                else:
                    x.append(float(line.split(separator)[0]))
                    y.append(float(line.split(separator)[1]))
    if len(x) != len(y):
        print(f"Warning:Length of x ({len(x)}) and y {len(y)} are different!")

    return np.array(x),np.array(y)

#Write simple datafile
# Separator is Python default whitespace.
def write_datafile(x, y, filename="new.dat", separator="     "):
    if len(x) != len(y):
        print(f"Error:Length of x ({len(x)}) and y {len(y)} are different!")
        ashexit()
    with open(filename, 'w') as f:
        f.write("# Created by ASH\n")
        for i,j in zip(x,y):
            f.write(f"{i}{separator}{j}\n")
    print("Wrote new datafile:", filename)

# Fast numpy-array to file
# https://stackoverflow.com/questions/53820891/speed-of-writing-a-numpy-array-to-a-text-file
#Note: float_format needs to match dimension of array
def fast_nparray_write(a, float_format="%-12.7f %-12.7f %-12.7f %-8.4f", filename="bla5", writemode="w"):
    with open(filename,writemode) as f:
        #fmt = ' '.join([float_format]*a.shape[1])
        #fmt = '\n'.join([fmt]*a.shape[0])
        #print("fmt:", fmt)
        #print(type(fmt))
        fmt = '\n'.join([float_format]*a.shape[0])
        data = fmt % tuple(a.ravel())
        f.write(data)


# Write a string to file simply
def writestringtofile(string, file,writemode='w'):
    with open(file, writemode) as f:
        f.write(string)


# Write a Python list to file simply
def writelisttofile(pylist, file, separator=" "):
    with open(file, 'w') as f:
        for l in pylist:
            f.write(str(l) + separator)
    print("Wrote list to file:", file)


# Natural (human) sorting of list
def natural_sort(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# Reverse read function.
def reverse_lines(filename, BUFSIZE=20480):
    # f = open(filename, "r")
    filename.seek(0, 2)
    p = filename.tell()
    remainder = ""
    while True:
        sz = min(BUFSIZE, p)
        p -= sz
        filename.seek(p)
        buf = filename.read(sz) + remainder
        if '\n' not in buf:
            remainder = buf
        else:
            i = buf.index('\n')
            for L in buf[i + 1:].split("\n")[::-1]:
                yield L
            remainder = buf[:i]
        if p == 0:
            break
    yield remainder


def clean_number(number):
    return np.real_if_close(number)




# Function to get unique values
def uniq(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


# Extract column from matrix
def column(matrix, i):
    return [row[i] for row in matrix]


# Various function to print time of module/step. Will add time also to Timings object
#Printing if currprintlevel
def print_time_rel(timestamp, modulename='Unknown', moduleindex=4, currprintlevel=1, currthreshold=1):
    secs = time.time() - timestamp
    #print("secs:", secs)
    mins = secs / 60
    if currprintlevel >= currthreshold:
        print_line_with_subheader2(
            "Time to calculate step ({}): {:4.3f} seconds, {:3.1f} minutes.".format(modulename, secs, mins))
    # Adding time to Timings object
    timingsobject.add(modulename, secs, moduleindex=moduleindex)


def print_time_rel_and_tot(timestampA, timestampB, modulename='Unknown', moduleindex=4):
    secsA = time.time() - timestampA
    minsA = secsA / 60
    # hoursA=minsA/60
    secsB = time.time() - timestampB
    minsB = secsB / 60
    # hoursB=minsB/60
    print("-------------------------------------------------------------------")
    print("Time to calculate step ({}): {:3.1f} seconds, {:3.1f} minutes.".format(modulename, secsA, minsA))
    print("Total Walltime: {:3.1f} seconds, {:3.1f} minutes.".format(secsB, minsB))
    print("-------------------------------------------------------------------")
    # Adding time to Timings object
    timingsobject.add(modulename, secsA, moduleindex=moduleindex)


def print_time_tot_color(time_initial, modulename='Unknown', moduleindex=4):
    # hoursA=minsA/60
    secs = time.time() - time_initial
    mins = secs / 60
    # hoursB=minsB/60
    print(BC.WARNING, "-------------------------------------------------------------------", BC.END)
    print(BC.WARNING, "ASH Total Walltime: {:3.1f} seconds, {:3.1f} minutes.".format(secs, mins), BC.END)
    print(BC.WARNING, "-------------------------------------------------------------------", BC.END)
    # Adding time to Timings object
    timingsobject.add(modulename, secs, moduleindex=moduleindex)


# Keep track of module runtimes
class Timings:
    def __init__(self):
        self.simple_dict = {}
        self.module_count = {}
        self.module_indices = {}
        self.totalsumtime = 0

    def add(self, modulename, mtime, moduleindex=4):

        # Adding time to dictionary
        if modulename in self.simple_dict:
            self.simple_dict[modulename] += mtime
        else:
            self.simple_dict[modulename] = mtime

        # Adding moduleindex to dictionary
        if modulename not in self.module_indices:
            self.module_indices[modulename] = moduleindex

        # Adding times called
        if modulename in self.module_count:
            self.module_count[modulename] += 1
        else:
            self.module_count[modulename] = 1

        self.totalsumtime += mtime

    # Distinguish and sort between:
    # workflows (thermochem_protol, PES, calc_surface etc.): 0
    # jobtype (optimizer,Singlepoint,Anfreq,Numfreq): 1
    # theory-run (ORCAtheory run, QM/MM run, MM run etc.): 2
    # various: 3
    # others (calc connectivity etc.): 4

    def print(self, inittime):
        totalwalltime = time.time() - inittime
        print("To turn off timing output add to settings file: ~/ash_user_settings.ini")
        print("print_full_timings = False   ")
        print("")
        print("{:35}{:>20}{:>20}{:>17}".format("Modulename", "Time (sec)", "Percentage of total", "Times called"))
        print("-" * 100)

        # Lists of dictitems by module_labels
        # Workflows: thermochemprotocol, calc_surface, benchmarking etc.
        dictitems_index0 = [i for i in self.simple_dict if self.module_indices[i] == 0]
        # Jobtype: Singlepoint, Opt, freq
        dictitems_index1 = [i for i in self.simple_dict if self.module_indices[i] == 1]
        # Theory run: ORCATHeory, QM/MM Theory etc
        dictitems_index2 = [i for i in self.simple_dict if self.module_indices[i] == 2]
        # NOTE: Was not using index 3. Now using for object creation
        dictitems_index3=[i for i in self.simple_dict if self.module_indices[i] == 3]
        # Other small modules. 4 is default
        dictitems_index4 = [i for i in self.simple_dict if self.module_indices[i] == 4]

        if len(dictitems_index0) != 0:
            print("Workflow modules")
            print("-" * 30)
            for dictitem in dictitems_index0:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        if len(dictitems_index1) != 0:
            print("Jobtype modules")
            print("-" * 30)
            for dictitem in dictitems_index1:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        if len(dictitems_index2) != 0:
            print("Theory-run modules")
            print("-" * 30)
            for dictitem in dictitems_index2:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        if len(dictitems_index3) != 0:
            print("Object creation")
            print("-" * 30)
            for dictitem in dictitems_index3:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        if len(dictitems_index4) != 0:
            print("Other modules")
            print("-" * 30)
            for dictitem in dictitems_index4:
                mmtime = self.simple_dict[dictitem]
                time_per = 100 * (mmtime / totalwalltime)
                print("{:35}{:>20.2f}{:>10.1f}{:>20}".format(dictitem, mmtime, time_per, self.module_count[dictitem]))
            print("")
        print("")
        print("{:35}{:>20.2f}".format("Sum of all moduletimes (flawed)", self.totalsumtime))
        print("{:35}{:>20.2f}{:>10}".format("Total walltime", totalwalltime, 100.0))


#General pretty table of things
def print_pretty_table(list_of_objects=None, list_of_labels=None, title=None,  spacing=15, float_decimals=4,
                       divider_line_length=70):

    format_spec_string=f"{spacing}"
    format_spec_float=f"{spacing}.{float_decimals}f"
    #Header
    print("="*divider_line_length)
    print(f"{title}")
    print("="*divider_line_length)
    headerstring=f""
    for label in list_of_labels:
        headerstring+= f"{label:>{format_spec_string}}"
    print(headerstring)
    print("-"*divider_line_length)
    tablestring=f""
    for i in range(len(list_of_objects[0])):
        for j in range(len(list_of_objects)):
            val = list_of_objects[j][i]
            if type(val) == float:
                tablestring+=f"{val:>{format_spec_float}}"
            else:
                tablestring+=f"{val:>{format_spec_string}}"
        tablestring+=f"\n"
    print(tablestring)
    print("-"*divider_line_length)
    print()


# Creating object
timingsobject = Timings()
