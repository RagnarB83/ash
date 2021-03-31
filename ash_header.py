"""
Functions to print header, logo, inputscript etc.
"""
import time
import os
import sys

import settings_ash
import ash
from functions_general import BC,print_time_tot_color



programversion = 0.2

#ASH footer
def print_footer():
    print("")


    print_time_tot_color(init_time)


def print_header():
    """
    ASH initial output. Used to print header (logo, version etc.), set initial time, print inputscript etc.
    """


    #Initializes time
    global init_time
    init_time=time.time()
    
    #Comment out to skip printing of header
    if settings_ash.settings_dict["print_logo"] is True:
        print_logo()

    print("ASH path:", settings_ash.ashpath)
    print("ASH Settings after reading defaults and ~/ash_user_settings.ini : ")
    print(settings_ash.settings_dict)
    print("Setting initial time")
    print("Note: ASH uses ANSI escape sequences for displaying color. Use less -R to display or set LESS=-R environment variable")
    print("To turn off escape sequences, set:   use_ANSI_color = False   in  ~/ash_user_settings.ini")
    print("")
    
    #Print input script unless interactive session
    if settings_ash.settings_dict["print_input"] is True:
        if settings_ash.interactive_session == False:
            inputfilepath=inputfile= os.getcwd()+"/"+sys.argv[0]
            print("Input script:", inputfilepath )
            print(BC.WARNING,"="*100)
            with open(inputfilepath) as x: f = x.readlines()
            for line in f:
                print(line,end="")
            print("="*100,BC.END)
            print("")

    

def print_logo():
    
    #Getting commit version number from file VERSION (updated by ashpull) inside module dir
    try:
        with open(os.path.dirname(ash.__file__)+"/VERSION") as f:
            git_commit_number = int(f.readline())
    except:
        git_commit_number="Unknown"


    #http://asciiflow.com
    #https://textik.com/#91d6380098664f89
    #https://www.gridsagegames.com/rexpaint/

    ascii_banner="""
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
    print(BC.OKGREEN,"----------------------------------------------------------------------------------",BC.END)
    print(BC.OKGREEN,"----------------------------------------------------------------------------------",BC.END)
    #print(BC.OKBLUE,ascii_banner3,BC.END)
    #print(BC.OKBLUE,ascii_banner2,BC.END)
    print(BC.OKGREEN,ascii_banner_center,BC.END)
    print(BC.OKGREEN,ascii_tree,BC.END)
    print(BC.WARNING,BC.BOLD,"ASH version", programversion,BC.END)
    print(BC.WARNING, "Git commit version: ", git_commit_number, BC.END)
    print(BC.WARNING,"A COMPCHEM AND QM/MM ENVIRONMENT", BC.END)
    print(BC.OKGREEN,"----------------------------------------------------------------------------------",BC.END)
    print(BC.OKGREEN,"----------------------------------------------------------------------------------",BC.END)