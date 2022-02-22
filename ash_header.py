"""
Functions to print header, footer, logo, inputscript etc.
"""
import time
import os
import sys

import settings_ash
import ash
from functions.functions_general import ashexit, BC, print_time_tot_color, timingsobject, print_line_with_mainheader, \
    print_line_with_subheader1

programversion = "0.8dev"


# ASH footer
def print_footer():
    print()
    print_time_tot_color(init_time)


def print_timings():
    """
    Print timings of each module
    """
    print()
    print_line_with_subheader1("Total timings of all modules")

    timingsobject.print(init_time)


def print_header():
    """
    ASH initial output. Used to print header (logo, version etc.), set initial time, print inputscript etc.
    """

    # Initializes time
    global init_time
    init_time = time.time()
    #########################################
    # Print main header w/wo logo
    #########################################
    # Getting commit version number from file VERSION (updated by ashpull) inside module dir
    try:
        with open(os.path.dirname(ash.__file__) + "/VERSION") as f:
            git_commit_number = int(f.readline())
    except:
        git_commit_number = "Unknown"

    print(f"{BC.OKGREEN}{'-' * 80}{BC.END}")
    print(f"{BC.OKGREEN}{'-' * 80}{BC.END}")
    if settings_ash.settings_dict["print_logo"] is True:
        print_logo()
    print(f"{BC.WARNING}A MULTISCALE MODELLING PROGRAM{BC.END}".center(90))
    print(f"{BC.WARNING}{BC.BOLD}Version: {programversion}{BC.END}".center(95))
    print(f"{BC.WARNING}Git commit version: {git_commit_number}{BC.END}".center(90))
    print(f"{BC.OKGREEN}{'-' * 80}{BC.END}")
    print(f"{BC.OKGREEN}{'-' * 80}{BC.END}")

    print("ASH path:", settings_ash.ashpath)
    
    #Check Python version
    pythonversion=(sys.version_info[0],sys.version_info[1],sys.version_info[2])
    print("Python version: {}.{}.{}".format(pythonversion[0],pythonversion[1],pythonversion[2]))
    print("Python interpreter:", sys.executable)
    if pythonversion < (3,6,0):
      print("ASH requires Python version 3.6.0 or higher")
      ashexit()
    
    print("\nASH Settings after reading defaults and ~/ash_user_settings.ini : ")
    print("See https://ash.readthedocs.io/en/latest/basics.html#ash-settings on how to change settings.")
    # print(settings_ash.settings_dict)
    for key, val in settings_ash.settings_dict.items():
        print("\t", key, ": ", val)

    print("\nNote: ASH can use ANSI escape sequences for displaying color. Use e.g. less -R to display")
    print("To turn on/off escape sequences, set: 'use_ANSI_color = False' in")
    print("~/ash_user_settings.ini")
    print()

    # Print input script unless interactive session
    if settings_ash.settings_dict["print_input"] is True:
        if settings_ash.interactive_session is False:
            inputfilepath = os.getcwd() + "/" + sys.argv[0]
            print("Input script:", inputfilepath)
            print(f"{BC.WARNING}{'=' * 80}")
            with open(inputfilepath) as f:
                for line in f:
                    print("   >", line, end="")
            print(f"{BC.WARNING}{'=' * 80}",BC.END)
            print()


def print_logo():


    # http://asciiflow.com
    # https://textik.com/#91d6380098664f89
    # https://www.gridsagegames.com/rexpaint/

    ascii_banner = """
   ▄████████    ▄████████    ▄█    █▄    
  ███    ███   ███    ███   ███    ███   
  ███    ███   ███    █▀    ███    ███   
  ███    ███   ███         ▄███▄▄▄▄███▄▄ 
▀███████████ ▀███████████ ▀▀███▀▀▀▀███▀  
  ███    ███          ███   ███    ███   
  ███    ███    ▄█    ███   ███    ███   
  ███    █▀   ▄████████▀    ███    █▀    
                                         
    """

    ascii_banner_center = """
                            ▄████████    ▄████████    ▄█    █▄    
                           ███    ███   ███    ███   ███    ███   
                           ███    ███   ███    █▀    ███    ███   
                           ███    ███   ███         ▄███▄▄▄▄███▄▄ 
                         ▀███████████ ▀███████████ ▀▀███▀▀▀▀███▀  
                           ███    ███          ███   ███    ███   
                           ███    ███    ▄█    ███   ███    ███   
                           ███    █▀   ▄████████▀    ███    █▀    
                                         
    """
    ascii_tree = """                                                                               
                  [      ▓          ▒     ╒                  ▓                  
                  ╙█     ▓▓         ▓  -µ ╙▄        ¬       ▓▀             
               ▌∩   ▀█▓▌▄▄▓▓▓▄,  ▀▄▓▌    ▓  ▀█      ▌   █ ╓▓Γ ▄▓▀¬              
              ╓Γ╘     % ╙▀▀▀▀▀▓▓▄  ▓▓   █▓,╒  ▀▓   ▓▌ Æ▀▓▓▓ ▄▓▓¬ .     ▌        
              ▓  ^█    ▌  ▄    ▀▓▓ ▓▓m ▐▓ ▓▓  ▐▌▓▌▓▀   Å▓▓▓▓▀     ▐   ▐▓    ▄   
            ╘▓▌ \  ▀▓▄▐▓ ▐▓     ▀▓▓▓▓m ▓▌▄▓▓█ ▓┘ ▓▓ ▄█▀▀▓▓▄   ▐▓  ▐▄  j▓   ▓    
          █  ╙▓  ╙▀▄▄▀▓▓µ ▓   ▄▄▀▀▓▓▓ ▓▓▓Σ▄▓▓▓Γ  ▓▓▓▀ █▓▀^▓  ▄▓   ▓   ▐▓  █     
        ▌  ▀▀█,▓▓  ▄╙▀▓▓▓▄▓▌▄▓▌   ▓▓  ▓▓▓▀▓▓▓▓▓  ▓▓▓▓▓▀ ╞ ▓▄▓▀ ╓─ ▓▄ ,▓█▀▀ .▀   
        ▐▄    ▀▓▓  ▓   ▓▓▓▓▓Γ╙▓▄  ▓▓▓▓▓▌▓▓▀▓Γ ▓▓▄▓▓▓▀  ▄ ▓▓▀  ▀   ▐▓▄▓▀   ▄▓    
   ╓┴▀███▓▓▓▄   ▓▓▓    █▓▓▓ ║ ¬▓▓▓▓▓▓▓▓▀  ▓▌  ▐▓▓▓▓  ▄▓▓▓▀  ▄▓██▀▀  ▓▓▄▓█▀Γ     
           ╙▀▓▓▓▓▓▓     ▓▓▓  █▄  ▀▓▓▓▓▓  j▓▄  ▓▓▓▓▓▓▓▓▀   █▓▓▀   ▄▓▓▓▀▐         
          ,▄µ  ▀▓▓▓▓▓▄  ▓▓▓   ▓▓▄▓▓▀▓▓▓▓▓▓▓  ▄▓▓▓▓▓▓▓▓▓▓▓▓▀  ▄▓▓▀▀¬             
    ▄▓█▀▀▀▀▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▄  ²▓▓▓    ▀▓▓▓▓ ▄▓▓▓▓▓▓▀▀▀¬   ▄▓▓▀¬ ▄Φ` ▄`       ▓  
  Σ▀Σ  ▄██▀▀Σ     ¬▀▓▓▓▓▓▓▓▓  j▓▓▌     ▐▓▓▓▓▓▓▓▓▀       .▓▓▀  ▄▓  ▄▓Γ   ▄▄▄▀▀   
     ╒▓               ▀▓▓▓▓▓▄,▓▓▓▓▓▓▓▓▄ ▓▓▓▓▓▓▓▓▓▄,     ▓▓▓▄▄▓▓▓█▀▀  ▓▓▀¬       
     ╙⌐       ▄         ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▐▀▓▓▓▓▓▓▓▓▓▓▓▓▀▀▓▓▓▓▓▓▓▀          
          ▄█▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▀▀¬  ▐▀▓▓▓▓▓▓▓▓▓                      ▐▀▀▄         
       ▄▀▀▐▓▓▀▐  ¬*▐▀▀▀▀             ▓▓▓▓▓▓▓                                 
     ╒Γ   ▐▓                         ▓▓▓▓▓▓▓                                    
     Γ    ╫                          ▓▓▓▓▓▓▓                                    
                                     ▓▓▓▓▓▓▓                                    
                                    ▓▓▓▓▓▓▓▓▄                                   
                                   ▓▓▓▓▓▓▓▓▓▓▄                                  
                                ,▓▓▓▓▓▓▓▓▓▓▓▓▓▄                                 
                          ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▄▄▌▌▄▄▄                       
                ,▄æ∞▄▄▄█▓▀▀▐¬▐▀▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▄ ▐▀▓▓▓▓▓▀▀▀▀Σ╙           
         ╙Φ¥¥▀▀▀▀   ╒▓   ▄▓█▀▀▓▓▓▓▓▓▓▓▀  ▓▓▓▓▓▓ ▀▓▓▄▄▐▀▀▀▀▀▐▐▀██                
                    ▓▌▄▓▓▀  ▓▓▓▓▓▀▀▓▓     ▓▓ ▓▓▓   ▐▀▀█▓▓▌¥4▀▀▀▀▓▓▄             
               `ΓΦ▀▓▀▐   ▄▄▓▓ ▓▓  ▐▌▓     ▓▓▓ ▓▀▀▓▄     ,▓▓      ▀▀▀▀▀          
                 ▄▀,  ▄█▀▀ ▓¬ ▓τ  ▓ ▐▓   ▓  ▓  ¥ ▓        ▀▌                    
               ^    ▓▀    █  ▓▀ ▄█¬ ▐▓  ▌   Γ   ▐▓τ*       ▀█▄                  
                   ▓   /Γ ƒ▀ⁿ  █    ▓   Γ      ╓▀▐⌐          ¬▀²                
                   ▓      \    ▀    ╙µ  ⌡                                       
                                     └                                                                                                                      
    """
    # print(BC.OKBLUE,ascii_banner3,BC.END)
    # print(BC.OKBLUE,ascii_banner2,BC.END)
    print(f"{BC.OKGREEN}{ascii_banner_center}{BC.END}")
    print(f"{BC.OKGREEN}{ascii_tree}{BC.END}")

