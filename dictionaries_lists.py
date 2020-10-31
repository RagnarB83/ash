#does not work to import this for some reason

#TODO: Look into: https://github.com/awvwgk/QCElemental


#Dictionary to handle fragments. e.g. mainfrag will contain list of unique elements, charge and multiplicity:
#{'mainfrag': [['C', 'Cl', 'Co', 'H', 'N', 'O'], '1', '1']}
fragdictionary = {}


#List of elements
elements=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

#Spin multiplicities of all element atoms
atom_spinmults = {'H':2, 'He': 1, 'Li':2, 'Be':1, 'B':2, 'C':3, 'N':4, 'O':3, 'F':2, 'Ne':1, 'Na':2, 'Mg':1, 'Al':2, 'Si':3, 'P':4, 'S':3, 'Cl':2, 'Ar':1, 'K':2, 'Ca':1, 'Sc':2, 'Ti':3, 'V':4, 'Cr':7, 'Mn':6, 'Fe':5, 'Co':4, 'Ni':3, 'Cu':2, 'Zn':1, 'Ga':2, 'Ge':3, 'As':4, 'Se':3, 'Br':2, 'Kr':1, 'Rb':2, 'Sr':1, 'Y':2, 'Zr':3, 'Nb':6, 'Mo':7, 'Tc':6, 'Ru':5, 'Rh':4, 'Pd':1, 'Ag':2, 'Cd':1, 'In':2, 'Sn':3, 'Sb':4, 'Te':3, 'I':2, 'Xe':1, 'Cs':2, 'Ba':1, 'La':2, 'Ce':1, 'Pr':4, 'Nd':5, 'Pm':6, 'Sm':7, 'Eu':8, 'Gd':9, 'Tb':6, 'Dy':5, 'Ho':4, 'Er':3, 'Tm':2, 'Yb':1, 'Lu':2, 'Hf':3, 'Ta':4, 'W':5, 'Re':6, 'Os':5, 'Ir':4, 'Pt':3, 'Au':2, 'Hg':1, 'Tl':2, 'Pb':3, 'Bi':4, 'Po':3, 'At':2, 'Rn':1, 'Fr':2, 'Ra':1, 'Ac':2, 'Th':3, 'Pa':4, 'U':5, 'Np':6, 'Pu':7, 'Am':8, 'Cm':9, 'Bk':6, 'Cf':5, 'Es':5, 'Fm':3, 'Md':2, 'No':1, 'Lr':2, 'Rf':3, 'Db':4, 'Sg':5, 'Bh':6, 'Hs':5, 'Mt':4, 'Ds':3, 'Rg':2, 'Cn':1, 'Nh':2, 'Fl':3, 'Mc':4, 'Lv':3, 'Ts':2, 'Og':1 }


#Atom masses
atommasses = [1.00794, 4.002602, 6.94, 9.0121831, 10.81, 12.01070, 14.00670, 15.99940, 18.99840316, 20.1797, 22.98976928, 24.305, 26.9815385, 28.085, 30.973762, 32.065, 35.45, 39.948, 39.0983, 40.078, 44.955908, 47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.63, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637, 95.96, 97, 101.07, 102.9055, 106.42, 107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 126.90447, 131.293, 132.905452, 137.327, 138.90547, 140.116, 140.90766, 144.242, 145, 150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259, 168.93422, 173.054, 174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592, 204.38, 207.2, 208.9804, 209, 210, 222, 223, 226, 227, 232.0377, 231.03588, 238.02891, 237, 244, 243, 247, 247, 251, 252, 257, 258, 259, 262 ]


#Core electrons for elements in ORCA
atom_core_electrons = {'H': 0, 'He' : 0, 'Li' : 0, 'Be' : 0, 'B': 2, 'C' : 2, 'N' : 2, 'O' : 2, 'F' : 2, 'Ne' : 2,
                      'Na' : 2, 'Mg' : 2, 'Al' : 10, 'Si' : 10, 'P' : 10, 'S' : 10, 'Cl' : 10, 'Ar' : 10,
                       'K' : 10, 'Ca' : 10, 'Sc' : 10, 'Ti' : 10, 'V' : 10, 'Cr' : 10, 'Mn' : 10, 'Fe' : 10, 'Co' : 10,
                       'Ni' : 10, 'Cu' : 10, 'Zn' : 10, 'Ga' : 18, 'Ge' : 18, 'As' : 18, 'Se' : 18, 'Br' : 18, 'Kr' : 18,
                       'Rb' : 18, 'Sr' : 18, 'Y' : 28, 'Zr' : 28, 'Nb' : 28, 'Mo' : 28, 'Tc' : 28, 'Ru' : 28, 'Rh' : 28,
                       'Pd' : 28, 'Ag' : 28, 'Cd' : 28, 'In' : 36, 'Sn' : 36, 'Sb' : 36, 'Te' : 36, 'I' : 36, 'Xe' : 36,
                       'Cs' : 36, 'Ba' : 36, 'Lu' : 46, 'Hf' : 46, 'Ta' : 46, 'w' : 46, 'Re' : 46, 'Os' : 46, 'Ir' : 46,
                       'Pt' : 46, 'Au' : 46, 'Hg' : 46, 'Tl' : 68, 'Pb' : 68, 'Bi' : 68, 'Po' : 68, 'At' : 68, 'Rn' : 68}

#Spin-orbit splittings:
#Currently only including neutral atoms. Data in cm-1 from : https://webhome.weizmann.ac.il/home/comartin/w1/so.txt
atom_spinorbitsplittings = {'H': 0.000, 'B': -10.17, 'C' : -29.58, 'N' : 0.00, 'O' : -77.97, 'F' : -134.70,
                      'Al' : -74.71, 'Si' : -149.68, 'P' : 0.00, 'S' : -195.77, 'Cl' : -294.12}

#List of transition metal elements (with d-electrons)
tmlist=['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']

#Dictionary for converting between regular numbers and Roman numerals.
#For oxidation state printing
oxnumbers={-1:'-I',0:'0',-2:'-II',-3:'-III',-4:'-IV',-5:'-V',-6:'-VI',-7:'-VII',-8:'-VIII',-9:'-IX',-10:'-X',1:'I',2:'II',3:'III',4:'IV',5:'V',6:'VI',7:'VII',8:'VIII',9:'IX',10:'X'}


#Element dictionary: Key: atomic number. Value: List of [Atomsymbol,covalent radius,valence electrons]
#Valence electrons: For p-block including whole electron shell (s,p). For TM: including s,p,d (and next s if applicable)
#Covalent radii from Alvarez
#Note: Carbon requires special treatment: sp3, sp2, sp. Sp3 in dict
#Note: Mn ls and hs options. hs in dict
#Note: Fe ls and hs options. hs in dict
#Note Co ls and hs options: hs in dict
#Elements added: H-Cd. Rest to be done
eldict_covrad_valel={1:['H',0.31,1],2:['He',0.28,2],3:['Li',1.28,1],4:['Be',0.96,2],5:['B',0.84,3],6:['C',0.76,4],7:['N',0.71,5],8:['O',0.66,6],9:['F',0.57,7],10:['Ne',0.58,8],11:['Na',1.66],12:['Mg',1.41],13:['Al',1.21],14:['Si',1.11],15:['P',1.07],16:['S',1.05],17:['Cl',1.02,7],18:['Ar',1.06],19:['K',2.03],20:['Ca',1.76],21:['Sc',1.70],22:['Ti',1.6],23:['V',1.53],24:['Cr',1.39],25:['Mn',1.61],26:['Fe',1.52,16],27:['Co',1.50],28:['Ni',1.24],29:['Cu',1.32],30:['Zn',1.22],31:['Ga',1.22],32:['Ge',1.20],33:['As',1.19],34:['Se',1.20],35:['Br',1.20],36:['Kr',1.16],37:['Rb',2.2],38:['Sr',1.95],39:['Y',1.9],40:['Zr',1.75],41:['Nb',1.64],42:['Mo',1.54,14],43:['Tc',1.47],44:['Ru',1.46],45:['Rh',1.42],46:['Pd',1.39,18],47:['Ag',1.45],48:['Cd',1.44]}



# Element class
class Element:
    def __init__(self, name, atomicSymbol, C6, pol):
        self.name = name
        self.atomicSymbol = atomicSymbol
        # Atom C6 parameter. Chu and Dalgarno
        self.C6 = C6
        #atom polarizability. Chu and Dalgarno
        self.pol = pol

#Element dictionary based on atomic number keys. Contains element name, element symbol, free atomic C6 parameter and atomic polarizability
# Dictionary is also used later for other book-keeping (atomic r2,r3 and r4 parameters from computation)
#H-parameters not in paper.
# C6 parameter taken from Table in Tkatchenko and Scheffler paper: http://th.fhi-berlin.mpg.de/th/publications/PRL-102-073005-2009.pdf
#Polarizability taken from 0.6668 Angstrom^3 value in Transport properties of Ions in Gases 1988. Converted to Bohr^3
#http://onlinelibrary.wiley.com/store/10.1002/3527602852.app3/asset/app3.pdf?v=1&t=ixyxx0ci&s=87e0d5806cf1005fdb0a82cbaaef6956d8f63ebe
#Also recent Axel Becke and Kannemann 2012 paper. http://aip.scitation.org/doi/pdf/10.1063/1.3676064

#Update 15feb2017. Added C6 and alpha parameters beyond Kr. Elements 37-80 (except lanthanides).
# Taken from:
#VASP version 5.4.1 source code. Tkatchenko-Scheffler VdW code. vdwforcefield.F
# http://th.fhi-berlin.mpg.de/site/uploads/Publications/Submitted_NJP_%20WLiu_20130222.pdf
#Becke parameters are different. Not used.
#Too lazy to do elements 81-86. who cares

elems_C6_polz = {1:Element('hydrogen', 'H', 6.5, 4.5), 2:Element('helium', 'He', 1.42, 1.38), 3:Element('lithium', 'Li', 1392, 164),
4:Element('beryllium', 'Be', 227, 38), 5:Element('boron', 'B', 99.5, 21), 6:Element('carbon', 'C', 46.6, 12), 7:Element('nitrogen', 'N', 24.2, 7.4),
8:Element('oxygen', 'O', 15.6, 5.4), 9:Element('fluorine', 'F', 9.52, 3.8), 10:Element('neon', 'Ne', 6.20, 2.67), 11:Element('sodium', 'Na', 1518, 163),
12:Element('magnesium', 'Mg', 626, 71), 13:Element('aluminum', 'Al', 528, 60), 14:Element('silicon', 'Si', 305, 37), 15:Element('phosphorus', 'P', 185, 25),
16:Element('sulfur', 'S', 134, 19.6), 17:Element('chlorine', 'Cl', 94.6, 15), 18:Element('argon', 'Ar', 64.2, 11.1), 19:Element('potassium', 'K', 3923, 294),
20:Element('calcium', 'Ca', 2163, 160), 21:Element('scandium', 'Sc', 1383, 120), 22:Element('titanium', 'Ti', 1044, 98), 23:Element('vanadium', 'V', 832, 84),
24:Element('chromium', 'Cr', 602, 78), 25:Element('manganese', 'Mn', 552, 63), 26:Element('iron', 'Fe', 482, 56), 27:Element('cobalt', 'Co', 408, 50),
28:Element('nickel', 'Ni', 373, 48), 29:Element('copper', 'Cu', 253, 42), 30:Element('zinc', 'Zn', 284, 40), 31:Element('gallium', 'Ga', 498, 60), 32:Element('germanium', 'Ge', 354, 41),
33:Element('arsenic', 'As', 246, 29), 34:Element('selenium', 'Se', 210, 25), 35:Element('bromine', 'Br', 162, 20), 36:Element('krypton', 'Kr', 130, 16.7),
37:Element('rubidium', 'Rb', 4691.0, 319.2), 38:Element('strontium', 'Sr', 3170.0, 199.0),
39:Element('yttrium', 'Y', 1968.58, 126.737), 40:Element('zirconium', 'Zr', 1677.91, 119.97), 41:Element('niobium', 'Nb', 1263.61, 101.603), 42:Element('molybdenum', 'Mo', 1028.73, 88.42),
43:Element('technetium', 'Tc', 1390.87, 80.08), 44:Element('ruthenium', 'Ru', 609.75, 65.90), 45:Element('rhodium', 'Rh', 469, 56.1),46:Element('palladium', 'Pd', 157.5, 23.7),
47:Element('silver', 'Ag', 339, 50.6), 48:Element('cadmium', 'Cd', 452.0, 39.7),
49:Element('indium', 'In', 707.05, 70.22), 50:Element('tin', 'Sn', 587.42, 55.95), 51:Element('antimony', 'Sb', 459.322, 43.67), 52:Element('tellurium', 'Te', 396., 37.65), 53:Element('iodine', 'I', 385.0, 35.), 54:Element('xenon', 'Xe', 285.90, 27.30),
55:Element('cesium', 'Cs', 6582.08, 427.12), 56:Element('barium', 'Ba', 5727.0, 275.0),
72:Element('hafnium', 'Hf', 1274.8, 99.52), 73:Element('tantalum', 'Ta', 1019.92, 82.53), 74:Element('tungsten', 'W', 847.93, 71.041), 75:Element('rhenium', 'Re', 710.2, 63.04), 76:Element('osmium', 'Os', 596.67, 55.055),
77:Element('iridium', 'Ir', 359.1, 42.51), 78:Element('platinum', 'Pt', 347.1, 39.68), 79:Element('gold', 'Au', 298, 36.5), 80:Element('mercury', 'Hg', 392.0, 33.9)}


