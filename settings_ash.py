import ash
import time
#Defining global variables here: called as settings_ash.scale etc.
def init():
    ash.print_ash_header()
    global scale
    global tol
    global conndepth
    global init_time
    # Changing Scale, Tol, connectivity-recursivedepth
    # Scale=1.0, tol=0.1 and conndepth=10 should be good
    scale = 1.0
    tol = 0.1
    conndepth = 10
    init_time=time.time()
    print(f"Setting global settings: \n Connectivity scale: {scale} and tol: {tol}")