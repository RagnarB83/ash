# Author: Vilhjalmur Asgeirsson, 2019.
# global variables within KNARR program

global settings_types, job_types, calculator_types

settings_types = {"MAIN": 0,
                  "CALCULATOR": 1,
                  "OPTIMIZER": 2,
                  "PATH": 3,
                  "DYNAMICS": 4,
                  "SADDLE": 5,
                  "NEB": 6,
                  "FREQ": 7}

job_types = {"POINT": 0,
             "FREQ": 1,
             "OPT": 2,
             "RMSD": 3,
             "PATH": 4,
             "DYNAMICS": 5,
             "SADDLE": 6,
             "NEB": 7,
             "INSTANTON": 8}

calculator_types = {"ORCA": 0,
                    "XTB": 1}

optimizer_types = {"GVPO": 0,
                   "LVPO": 1,
                   "AVPO": 2,
                   "FIRE": 4,
                   "LBFGS": 5,
                   "BOFILL": 7}

global au_to_ev, au_to_eva, amu2au, au2cm, ev2cm, c

au_to_a = 0.529177
au_to_ev = 27.211396132
au_to_eva = au_to_ev / au_to_a
au_to_eva2 = au_to_ev / (au_to_a ** 2)
amu2au = 1.66053886e-27 / 9.10938215e-31
au2cm = 219474.63067
ev2cm = 8065.54
c = 2.99792458e10

global hbar_ev, h_ev, hbar, kB, time_unit, mu, no_kb

hbar_ev = 6.582119514e-16
h_ev = 4.135667662e-15
hbar = 6.46538e-2
kB = 8.61738573e-5  # eV/K
time_unit = 1.018046e-14
mu = 1.0
no_kb = False

global energystring, forcestring

energystring = 'eV'
forcestring = 'eV/Ang'
lengthstring = 'A'

global printlevel, output_extension

printlevel = 0
extension = 0  # 0=xyz 1=con

global seed

seed = 1234

global debug_a, debug_b, debug_c

debug_a = 1.5
debug_b = 2.0
debug_c = 3.0
debug_d = 4.0
debug_e = 1.-1


# henkelman potential
global boost, boosted, boost_time, boost_temp

boost = -1000.0 #for flat boost
boosted = False
boost_time = 1.0
boost_temp = 0.0
gauss_A = 0.8
gauss_B = -0.05
gauss_alpha = 1.0


#lennard jones
ljrcut = 100.0
ljsigma = 1.0
ljepsilon = 1.0

