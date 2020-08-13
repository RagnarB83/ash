import ash
import time

#Defining some ASH settings here

#Whether to use ANSI color escape sequences in output or not.
use_ANSI_color = True

#Global Connectivity settings
scale = 1.0
tol = 0.1
conndepth = 10

#Init function to print header and set time
def init():
    #Uncomment to skip header
    ash.print_ash_header()
    #global scale
    #global tol
    #global conndepth

    # Changing Scale, Tol, connectivity-recursivedepth
    # Scale=1.0, tol=0.1 and conndepth=10 should be good
    #scale = 1.0
    #tol = 0.1
    #conndepth = 10

    global init_time
    init_time=time.time()
    
    print(f"Setting global settings: \n Connectivity scale: {scale} and tol: {tol}")
    print("Setting initial time")
    print("Note: ASH uses ANSI escape sequences for displaying color. Use less -R to display or set LESS=-R environment variable")
    print("To turn off escape sequences, see settings_ash.py")
    