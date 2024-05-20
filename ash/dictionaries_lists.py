"""
Various lists and dictionaries of elements etc.
"""


# List of elements
elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
            'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
            'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
            'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
#Other form
eldict={'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Mo':42,'W':74,'Ru':44,'I':53}
# Conversion of common atom types or atomnames to elements.
# Used by: module_coords: conv_atomtypes_elems and read_gromacsfile. Also in read_pdbfile when atomnames option is active
# Added MW as atomtype for M-site. Problem for QM-code
atomtypes_dict = {'H': 'H', 'HA': 'H', 'HB': 'H', 'HW': 'H', 'HH': 'H', 'HN': 'H', 'HD': 'H', 'HZ': 'H', 'HG': 'H',
                  'HE': 'H', 'HT': 'H', 'H1': 'H', 'H2': 'H', 'H3': 'H', 'HT1': 'H', 'HT2': 'H', 'HT3': 'H', 'HC': 'H', 'HF':'H',
                  'HN1': 'H', 'HN2': 'H',
                  'HB1': 'H', 'HB2': 'H', 'HG1': 'H', 'HG2': 'H', 'HG11': 'H', 'HG12': 'H', 'HG13': 'H', 'HG21': 'H',
                  'HG22': 'H', 'HG23': 'H', 'HH11': 'H', 'HH12': 'H', 'HH21': 'H', 'HH22': 'H', 'HD11': 'H',
                  'HD12': 'H', 'HD13': 'H',
                  'HD21': 'H', 'HD22': 'H', 'HD23': 'H', 'HE21': 'H', 'HE22': 'H',
                  'HE1': 'H', 'HE2': 'H', 'HE3': 'H', 'HD1': 'H', 'HD2': 'H', '1HH1': 'H', '2HH1': 'H', '1HH2': 'H',
                  '2HH2': 'H',
                  '1HG1': 'H', '2HG1': 'H', '3HG1': 'H', '1HG2': 'H', '2HG2': 'H', '3HG2': 'H', '1HD1': 'H',
                  '2HD1': 'H', '3HD1': 'H', '1HD2': 'H', '2HD2': 'H', '3HD2': 'H', 'H4':'H',
                  'HD3': 'H', '1HE2': 'H', '2HE2': 'H', 'HZ1': 'H', 'HZ2': 'H', 'HZ3': 'H', 'HB3': 'H', 'HA1': 'H',
                  'HA2': 'H', 'HA3': 'H', 'HH2': 'H', 'HOB': 'H', 'HW1': 'H', 'HW2': 'H', 'HO1':'H', 'H5':'H', 'H61':'H',
                  'HO': 'H', 'H62':'H', 'HO6':'H', 'H12':'H', 'H52':'H', 'H22':'H', 'HO22':'H', 'H32':'H', 'HO32':'H',
                  'H631':'H','H632':'H','H633':'H', 'H43':'H', 'HO43':'H', 'H33':'H', 'HO33':'H', 'H23':'H', 'HO23':'H',
                  'H53':'H', 'HO53':'H', 'H42':'H', 'HO42':'H', 'H621':'H', 'H622':'H', 'HO62':'H', 'H13':'H',
                  'H01':'H','H02':'H','H03':'H','H04':'H','H05':'H', 'H06':'H', 'H07':'H','H08':'H', 'H09':'H',
                  'H0A':'H', 'H0B':'H',
                  'NA': 'Na', 'SOD': 'Na',
                  'C': 'C', 'CD': 'C', 'CH': 'C', 'CH2': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CZ': 'C', 'CE': 'C',
                  'CG1': 'C', 'CG2': 'C', 'CD1': 'C', 'CD2': 'C', 'CE1': 'C', 'CE2': 'C', 'CE3': 'C', 'C5':'C', 'C3':'C', 'C6':'C',
                  'CZ3': 'C', 'CZ2': 'C', 'CX': 'C', 'CBC': 'C', 'CAC': 'C', 'CDC': 'C', 'C1':'C', 'C2':'C', 'CT':'C', 'C4':'C',
                  'C12':'C', 'C52':'C', 'C22':'C', 'C32':'C', 'C42':'C', 'C62':'C', 'C13':'C', 'C53':'C', 'C23':'C', 'C33':'C',
                  'C43':'C', 'C63':'C', 'C00':'C', 'C01':'C','C02':'C','C03':'C', 'C04':'C',
                  'S': 'S', 'SD': 'S', 'SG': 'S', 'S1': 'S', 'S2': 'S', 'S3': 'S', 'S4': 'S', 'S5': 'S', 'S6': 'S',
                  'S7': 'S', 'S8': 'S', 'S9': 'S',
                  'O': 'O', 'OE': 'O', 'OG': 'O', 'OH': 'O', 'OW': 'O', 'OT': 'O', 'OD': 'O', 'OP': 'O', 'OE1': 'O', 'O2':'O',
                  'OE2': 'O', 'OD1': 'O', 'OD2': 'O', 'OG1': 'O', 'OT1': 'O', 'OH2': 'O', 'OXT': 'O', 'O1':'O', 'O5':'O', 'O04':'O',
                  'OT2': 'O', 'OHB': 'O', 'OB1': 'O', 'OB2': 'O', 'OA1': 'O', 'OA2': 'O', 'O3':'O', 'O4':'O', 'O6':'O',
                  'O52':'O', 'O22':'O', 'O32':'O', 'O42':'O', 'O62':'O', 'O53':'O', 'O23':'O', 'O33':'O', 'O43':'O',
                  'N': 'N', 'NZ': 'N', 'NH': 'N', 'NE': 'N', 'ND': 'N', 'NH1': 'N', 'NH2': 'N', 'NE1': 'N', 'NE2': 'N',
                  'ND1': 'N', 'ND2': 'N', 'N1':'N', 'N01':'N', 'N02':'N', 'N03':'N', 'N04':'N', 'N05':'N',
                  'CL': 'Cl', 'CLA': 'Cl',
                  'MG': 'Mg',
                  'FE1': 'Fe', 'FE2': 'Fe', 'FE3': 'Fe', 'FE4': 'Fe', 'FE5': 'Fe', 'FE6': 'Fe', 'FE7': 'Fe',
                  'FE8': 'Fe', 'FE9': 'Fe',
                  'MO1': 'Mo', 'V1': 'V',
                  'MW': 'M'}


# Element class
class Element:
    def __init__(self, name, symbol, atomnumber):
        # Fine-structure constant (2018 CODATA recommended value)
        self.name = None
        self.symbol = symbol
        self.atomnumber = atomnumber


element_dict_atname = {'h': Element('hydrogen', 'H', 1), 'he': Element('helium', 'He', 2),
                       'li': Element('lithium', 'Li', 3),
                       'be': Element('beryllium', 'Be', 4), 'b': Element('boron', 'B', 5),
                       'c': Element('carbon', 'C', 6), 'n': Element('nitrogen', 'N', 7),
                       'o': Element('oxygen', 'O', 8), 'f': Element('fluorine', 'F', 9),
                       'ne': Element('neon', 'Ne', 10), 'na': Element('sodium', 'Na', 11),
                       'mg': Element('magnesium', 'Mg', 12), 'al': Element('aluminum', 'Al', 13),
                       'si': Element('silicon', 'Si', 14), 'p': Element('phosphorus', 'P', 15),
                       's': Element('sulfur', 'S', 16), 'cl': Element('chlorine', 'Cl', 17),
                       'ar': Element('argon', 'Ar', 18), 'k': Element('potassium', 'K', 19),
                       'ca': Element('calcium', 'Ca', 20), 'sc': Element('scandium', 'Sc', 21),
                       'ti': Element('titanium', 'Ti', 22), 'v': Element('vanadium', 'V', 23),
                       'cr': Element('chromium', 'Cr', 24), 'mn': Element('manganese', 'Mn', 25),
                       'fe': Element('iron', 'Fe', 26), 'co': Element('cobalt', 'Co', 27),
                       'ni': Element('nickel', 'Ni', 28), 'cu': Element('copper', 'Cu', 29),
                       'zn': Element('zinc', 'Zn', 30), 'ga': Element('gallium', 'Ga', 31),
                       'ge': Element('germanium', 'Ge', 32),
                       'as': Element('arsenic', 'As', 33), 'se': Element('selenium', 'Se', 34),
                       'br': Element('bromine', 'Br', 35), 'kr': Element('krypton', 'Kr', 36),
                       'rb': Element('rubidium', 'Rb', 37), 'sr': Element('strontium', 'Sr', 38),
                       'y': Element('yttrium', 'Y', 39), 'zr': Element('zirconium', 'Zr', 40),
                       'nb': Element('niobium', 'Nb', 41), 'mo': Element('molybdenum', 'Mo', 42),
                       'tc': Element('technetium', 'Tc', 43), 'ru': Element('ruthenium', 'Ru', 44),
                       'rh': Element('rhodium', 'Rh', 45), 'pd': Element('palladium', 'Pd', 46),
                       'ag': Element('silver', 'Ag', 47), 'cd': Element('cadmium', 'Cd', 48),
                       'in': Element('indium', 'In', 49), 'sn': Element('tin', 'Sn', 50),
                       'sb': Element('antimony', 'Sb', 51), 'te': Element('tellurium', 'Te', 52),
                       'i': Element('iodine', 'I', 53), 'xe': Element('xenon', 'Xe', 54),
                       'cs': Element('cesium', 'Cs', 55), 'ba': Element('barium', 'Ba', 56),
                       'hf': Element('hafnium', 'Hf', 72), 'ta': Element('tantalum', 'Ta', 73),
                       'w': Element('tungsten', 'W', 74), 're': Element('rhenium', 'Re', 75),
                       'os': Element('osmium', 'Os', 76),
                       'ir': Element('iridium', 'Ir', 77), 'pt': Element('platinum', 'Pt', 78),
                       'au': Element('gold', 'Au', 79), 'hg': Element('mercury', 'Hg', 80), 'tl': Element('thallium', 'Tl', 81),
                       'pb': Element('lead', 'Pb', 82),'bi': Element('bismuth', 'Bi', 83),'po': Element('polonium', 'Po', 84),
                       'at': Element('astatine', 'At', 85), 'rn': Element('radon', 'Rn', 86)}

# NOTE: redundant? remove?
element_dict_atnum = {1: Element('hydrogen', 'H', 1), 2: Element('helium', 'He', 2), 3: Element('lithium', 'Li', 3),
                      4: Element('beryllium', 'Be', 4), 5: Element('boron', 'B', 5), 6: Element('carbon', 'C', 6),
                      7: Element('nitrogen', 'N', 7),
                      8: Element('oxygen', 'O', 8), 9: Element('fluorine', 'F', 9), 10: Element('neon', 'Ne', 10),
                      11: Element('sodium', 'Na', 11),
                      12: Element('magnesium', 'Mg', 12), 13: Element('aluminum', 'Al', 13),
                      14: Element('silicon', 'Si', 14), 15: Element('phosphorus', 'P', 15),
                      16: Element('sulfur', 'S', 16), 17: Element('chlorine', 'Cl', 17), 18: Element('argon', 'Ar', 18),
                      19: Element('potassium', 'K', 19),
                      20: Element('calcium', 'Ca', 20), 21: Element('scandium', 'Sc', 21),
                      22: Element('titanium', 'Ti', 22), 23: Element('vanadium', 'V', 23),
                      24: Element('chromium', 'Cr', 24), 25: Element('manganese', 'Mn', 25),
                      26: Element('iron', 'Fe', 26), 27: Element('cobalt', 'Co', 27),
                      28: Element('nickel', 'Ni', 28), 29: Element('copper', 'Cu', 29), 30: Element('zinc', 'Zn', 30),
                      31: Element('gallium', 'Ga', 31), 32: Element('germanium', 'Ge', 32),
                      33: Element('arsenic', 'As', 33), 34: Element('selenium', 'Se', 34),
                      35: Element('bromine', 'Br', 35), 36: Element('krypton', 'Kr', 36),
                      37: Element('rubidium', 'Rb', 37), 38: Element('strontium', 'Sr', 38),
                      39: Element('yttrium', 'Y', 39), 40: Element('zirconium', 'Zr', 40),
                      41: Element('niobium', 'Nb', 41), 42: Element('molybdenum', 'Mo', 42),
                      43: Element('technetium', 'Tc', 43), 44: Element('ruthenium', 'Ru', 44),
                      45: Element('rhodium', 'Rh', 45), 46: Element('palladium', 'Pd', 46),
                      47: Element('silver', 'Ag', 47), 48: Element('cadmium', 'Cd', 48),
                      49: Element('indium', 'In', 49), 50: Element('tin', 'Sn', 50), 51: Element('antimony', 'Sb', 51),
                      52: Element('tellurium', 'Te', 52), 53: Element('iodine', 'I', 53),
                      54: Element('xenon', 'Xe', 54),
                      55: Element('cesium', 'Cs', 55), 56: Element('barium', 'Ba', 56),
                      72: Element('hafnium', 'Hf', 72), 73: Element('tantalum', 'Ta', 73),
                      74: Element('tungsten', 'W', 74), 75: Element('rhenium', 'Re', 75),
                      76: Element('osmium', 'Os', 76),
                      77: Element('iridium', 'Ir', 77), 78: Element('platinum', 'Pt', 78),
                      79: Element('gold', 'Au', 79), 80: Element('mercury', 'Hg', 80)}

# Ground-state spin multiplicities of all elements
atom_spinmults = {'H': 2, 'He': 1, 'Li': 2, 'Be': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 3, 'F': 2, 'Ne': 1, 'Na': 2, 'Mg': 1,
                  'Al': 2, 'Si': 3, 'P': 4, 'S': 3, 'Cl': 2, 'Ar': 1, 'K': 2, 'Ca': 1, 'Sc': 2, 'Ti': 3, 'V': 4,
                  'Cr': 7, 'Mn': 6, 'Fe': 5, 'Co': 4, 'Ni': 3, 'Cu': 2, 'Zn': 1, 'Ga': 2, 'Ge': 3, 'As': 4, 'Se': 3,
                  'Br': 2, 'Kr': 1, 'Rb': 2, 'Sr': 1, 'Y': 2, 'Zr': 3, 'Nb': 6, 'Mo': 7, 'Tc': 6, 'Ru': 5, 'Rh': 4,
                  'Pd': 1, 'Ag': 2, 'Cd': 1, 'In': 2, 'Sn': 3, 'Sb': 4, 'Te': 3, 'I': 2, 'Xe': 1, 'Cs': 2, 'Ba': 1,
                  'La': 2, 'Ce': 1, 'Pr': 4, 'Nd': 5, 'Pm': 6, 'Sm': 7, 'Eu': 8, 'Gd': 9, 'Tb': 6, 'Dy': 5, 'Ho': 4,
                  'Er': 3, 'Tm': 2, 'Yb': 1, 'Lu': 2, 'Hf': 3, 'Ta': 4, 'W': 5, 'Re': 6, 'Os': 5, 'Ir': 4, 'Pt': 3,
                  'Au': 2, 'Hg': 1, 'Tl': 2, 'Pb': 3, 'Bi': 4, 'Po': 3, 'At': 2, 'Rn': 1, 'Fr': 2, 'Ra': 1, 'Ac': 2,
                  'Th': 3, 'Pa': 4, 'U': 5, 'Np': 6, 'Pu': 7, 'Am': 8, 'Cm': 9, 'Bk': 6, 'Cf': 5, 'Es': 5, 'Fm': 3,
                  'Md': 2, 'No': 1, 'Lr': 2, 'Rf': 3, 'Db': 4, 'Sg': 5, 'Bh': 6, 'Hs': 5, 'Mt': 4, 'Ds': 3, 'Rg': 2,
                  'Cn': 1, 'Nh': 2, 'Fl': 3, 'Mc': 4, 'Lv': 3, 'Ts': 2, 'Og': 1}

# Atom masses
atommasses = [1.00794, 4.002602, 6.94, 9.0121831, 10.81, 12.01070, 14.00670, 15.99940, 18.99840316, 20.1797,
              22.98976928, 24.305, 26.9815385, 28.085, 30.973762, 32.065, 35.45, 39.948, 39.0983, 40.078, 44.955908,
              47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 69.723, 72.63, 74.921595,
              78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637, 95.96, 97, 101.07, 102.9055, 106.42,
              107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 126.90447, 131.293, 132.905452, 137.327, 138.90547,
              140.116, 140.90766, 144.242, 145, 150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259,
              168.93422, 173.054, 174.9668, 178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569,
              200.592, 204.38, 207.2, 208.9804, 209, 210, 222, 223, 226, 227, 232.0377, 231.03588, 238.02891, 237, 244,
              243, 247, 247, 251, 252, 257, 258, 259, 262]

# Core electron definitions for all elements in ORCA. Used in frozen-core post-HF methods
atom_core_electrons = {'H': 0, 'He': 0, 'Li': 0, 'Be': 0, 'B': 2, 'C': 2, 'N': 2, 'O': 2, 'F': 2, 'Ne': 2,
                       'Na': 2, 'Mg': 2, 'Al': 10, 'Si': 10, 'P': 10, 'S': 10, 'Cl': 10, 'Ar': 10,
                       'K': 10, 'Ca': 10, 'Sc': 10, 'Ti': 10, 'V': 10, 'Cr': 10, 'Mn': 10, 'Fe': 10, 'Co': 10,
                       'Ni': 10, 'Cu': 10, 'Zn': 10, 'Ga': 18, 'Ge': 18, 'As': 18, 'Se': 18, 'Br': 18, 'Kr': 18,
                       'Rb': 18, 'Sr': 18, 'Y': 28, 'Zr': 28, 'Nb': 28, 'Mo': 28, 'Tc': 28, 'Ru': 28, 'Rh': 28,
                       'Pd': 28, 'Ag': 28, 'Cd': 28, 'In': 36, 'Sn': 36, 'Sb': 36, 'Te': 36, 'I': 36, 'Xe': 36,
                       'Cs': 36, 'Ba': 36, 'Lu': 46, 'Hf': 46, 'Ta': 46, 'W': 46, 'Re': 46, 'Os': 46, 'Ir': 46,
                       'Pt': 46, 'Au': 46, 'Hg': 46, 'Tl': 68, 'Pb': 68, 'Bi': 68, 'Po': 68, 'At': 68, 'Rn': 68}

# Spin-orbit splittings:
# Currently only including neutral atoms. Data in cm-1 from : https://webhome.weizmann.ac.il/home/comartin/w1/so.txt
#Br and I taken from https://cccbdb.nist.gov/elecspin.asp
atom_spinorbitsplittings = {'H': 0.000, 'B': -10.17, 'C': -29.58, 'N': 0.00, 'O': -77.97, 'F': -134.70,
                            'Al': -74.71, 'Si': -149.68, 'P': 0.00, 'S': -195.77, 'Cl': -294.12, 'Br':-1228.4, 'I':-2534.4}


# List of transition metal elements (with d-electrons)
tmlist = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
          'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']

# Dictionary for converting between regular numbers and Roman numerals.
# For oxidation state printing
oxnumbers = {-1: '-I', 0: '0', -2: '-II', -3: '-III', -4: '-IV', -5: '-V', -6: '-VI', -7: '-VII', -8: '-VIII',
             -9: '-IX', -10: '-X', 1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX',
             10: 'X'}

# Element dictionary: Key: atomic number. Value: List of [Atomsymbol,covalent radius,valence electrons]
# Valence electrons: For p-block including whole electron shell (s,p). For TM: including s,p,d (and next s if applicable)
# Covalent radii from Alvarez
# Note: Carbon requires special treatment: sp3, sp2, sp. Sp3 in dict
# Note: Mn ls and hs options. hs in dict
# Note: Fe ls and hs options. hs in dict
# Note Co ls and hs options: hs in dict
# Elements added: H-Cd. Rest to be done
eldict_covrad_valel = {1: ['H', 0.31, 1], 2: ['He', 0.28, 2], 3: ['Li', 1.28, 1], 4: ['Be', 0.96, 2], 5: ['B', 0.84, 3],
                       6: ['C', 0.76, 4], 7: ['N', 0.71, 5], 8: ['O', 0.66, 6], 9: ['F', 0.57, 7], 10: ['Ne', 0.58, 8],
                       11: ['Na', 1.66], 12: ['Mg', 1.41], 13: ['Al', 1.21], 14: ['Si', 1.11], 15: ['P', 1.07],
                       16: ['S', 1.05], 17: ['Cl', 1.02, 7], 18: ['Ar', 1.06], 19: ['K', 2.03], 20: ['Ca', 1.76],
                       21: ['Sc', 1.70], 22: ['Ti', 1.6], 23: ['V', 1.53], 24: ['Cr', 1.39], 25: ['Mn', 1.61],
                       26: ['Fe', 1.52, 16], 27: ['Co', 1.50], 28: ['Ni', 1.24], 29: ['Cu', 1.32], 30: ['Zn', 1.22],
                       31: ['Ga', 1.22], 32: ['Ge', 1.20], 33: ['As', 1.19], 34: ['Se', 1.20], 35: ['Br', 1.20],
                       36: ['Kr', 1.16], 37: ['Rb', 2.2], 38: ['Sr', 1.95], 39: ['Y', 1.9], 40: ['Zr', 1.75],
                       41: ['Nb', 1.64], 42: ['Mo', 1.54, 14], 43: ['Tc', 1.47], 44: ['Ru', 1.46], 45: ['Rh', 1.42],
                       46: ['Pd', 1.39, 18], 47: ['Ag', 1.45], 48: ['Cd', 1.44]}


# Element_pol class
class Element_pol:
    def __init__(self, name, atomicSymbol, C6, pol):
        # Fine-structure constant (2018 CODATA recommended value)
        FSC = 0.0072973525693
        Phi = FSC ** (4 / 3)
        self.name = name
        self.atomicSymbol = atomicSymbol
        # Atom C6 parameter. Chu and Dalgarno
        self.C6 = C6
        # atom polarizability. Chu and Dalgarno
        self.pol = pol
        # Atomic vdW radius based on the 2020 paper by Tkatchenko: https://arxiv.org/pdf/2007.02992.pdf
        self.Rvdw_au = (self.pol / Phi) ** (1 / 7)
        self.Rvdw_ang = self.Rvdw_au * 0.52917721067


# Element dictionary based on atomic number keys. Contains element name, element symbol, free atomic C6 parameter and
# atomic polarizability
# Dictionary is also used later for other book-keeping (atomic r2,r3 and r4 parameters from computation)
# H-parameters not in paper.
# C6 parameter taken from Table in Tkatchenko and Scheffler paper:
# http://th.fhi-berlin.mpg.de/th/publications/PRL-102-073005-2009.pdf
# Polarizability taken from 0.6668 Angstrom^3 value in Transport properties of Ions in Gases 1988. Converted to Bohr^3
# http://onlinelibrary.wiley.com/store/10.1002/3527602852.app3/asset/app3.pdf?v=1&t=ixyxx0ci&s=87e0d5806cf1005fdb0a82cbaaef6956d8f63ebe
# Also recent Axel Becke and Kannemann 2012 paper. http://aip.scitation.org/doi/pdf/10.1063/1.3676064

# Update 15feb2017. Added C6 and alpha parameters beyond Kr. Elements 37-80 (except lanthanides).
# Taken from:
# VASP version 5.4.1 source code. Tkatchenko-Scheffler VdW code. vdwforcefield.F
# http://th.fhi-berlin.mpg.de/site/uploads/Publications/Submitted_NJP_%20WLiu_20130222.pdf
# Becke parameters are different. Not used.
# Too lazy to do elements 81-86. who cares

elems_C6_polz = {1: Element_pol('hydrogen', 'H', 6.5, 4.5), 2: Element_pol('helium', 'He', 1.42, 1.38),
                 3: Element_pol('lithium', 'Li', 1392, 164),
                 4: Element_pol('beryllium', 'Be', 227, 38), 5: Element_pol('boron', 'B', 99.5, 21),
                 6: Element_pol('carbon', 'C', 46.6, 12), 7: Element_pol('nitrogen', 'N', 24.2, 7.4),
                 8: Element_pol('oxygen', 'O', 15.6, 5.4), 9: Element_pol('fluorine', 'F', 9.52, 3.8),
                 10: Element_pol('neon', 'Ne', 6.20, 2.67), 11: Element_pol('sodium', 'Na', 1518, 163),
                 12: Element_pol('magnesium', 'Mg', 626, 71), 13: Element_pol('aluminum', 'Al', 528, 60),
                 14: Element_pol('silicon', 'Si', 305, 37), 15: Element_pol('phosphorus', 'P', 185, 25),
                 16: Element_pol('sulfur', 'S', 134, 19.6), 17: Element_pol('chlorine', 'Cl', 94.6, 15),
                 18: Element_pol('argon', 'Ar', 64.2, 11.1), 19: Element_pol('potassium', 'K', 3923, 294),
                 20: Element_pol('calcium', 'Ca', 2163, 160), 21: Element_pol('scandium', 'Sc', 1383, 120),
                 22: Element_pol('titanium', 'Ti', 1044, 98), 23: Element_pol('vanadium', 'V', 832, 84),
                 24: Element_pol('chromium', 'Cr', 602, 78), 25: Element_pol('manganese', 'Mn', 552, 63),
                 26: Element_pol('iron', 'Fe', 482, 56), 27: Element_pol('cobalt', 'Co', 408, 50),
                 28: Element_pol('nickel', 'Ni', 373, 48), 29: Element_pol('copper', 'Cu', 253, 42),
                 30: Element_pol('zinc', 'Zn', 284, 40), 31: Element_pol('gallium', 'Ga', 498, 60),
                 32: Element_pol('germanium', 'Ge', 354, 41),
                 33: Element_pol('arsenic', 'As', 246, 29), 34: Element_pol('selenium', 'Se', 210, 25),
                 35: Element_pol('bromine', 'Br', 162, 20), 36: Element_pol('krypton', 'Kr', 130, 16.7),
                 37: Element_pol('rubidium', 'Rb', 4691.0, 319.2), 38: Element_pol('strontium', 'Sr', 3170.0, 199.0),
                 39: Element_pol('yttrium', 'Y', 1968.58, 126.737), 40: Element_pol('zirconium', 'Zr', 1677.91, 119.97),
                 41: Element_pol('niobium', 'Nb', 1263.61, 101.603),
                 42: Element_pol('molybdenum', 'Mo', 1028.73, 88.42),
                 43: Element_pol('technetium', 'Tc', 1390.87, 80.08), 44: Element_pol('ruthenium', 'Ru', 609.75, 65.90),
                 45: Element_pol('rhodium', 'Rh', 469, 56.1), 46: Element_pol('palladium', 'Pd', 157.5, 23.7),
                 47: Element_pol('silver', 'Ag', 339, 50.6), 48: Element_pol('cadmium', 'Cd', 452.0, 39.7),
                 49: Element_pol('indium', 'In', 707.05, 70.22), 50: Element_pol('tin', 'Sn', 587.42, 55.95),
                 51: Element_pol('antimony', 'Sb', 459.322, 43.67), 52: Element_pol('tellurium', 'Te', 396., 37.65),
                 53: Element_pol('iodine', 'I', 385.0, 35.), 54: Element_pol('xenon', 'Xe', 285.90, 27.30),
                 55: Element_pol('cesium', 'Cs', 6582.08, 427.12), 56: Element_pol('barium', 'Ba', 5727.0, 275.0),
                 72: Element_pol('hafnium', 'Hf', 1274.8, 99.52), 73: Element_pol('tantalum', 'Ta', 1019.92, 82.53),
                 74: Element_pol('tungsten', 'W', 847.93, 71.041), 75: Element_pol('rhenium', 'Re', 710.2, 63.04),
                 76: Element_pol('osmium', 'Os', 596.67, 55.055),
                 77: Element_pol('iridium', 'Ir', 359.1, 42.51), 78: Element_pol('platinum', 'Pt', 347.1, 39.68),
                 79: Element_pol('gold', 'Au', 298, 36.5), 80: Element_pol('mercury', 'Hg', 392.0, 33.9)}
