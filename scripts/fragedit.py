#!/usr/bin/env python3
"""
fragedit.py

"""
import sys

#Standalone script to grab subcoordinates

#Function to reformat element string to be correct('cu' or 'CU' become 'Cu')
#Can also convert atomic-number (isatomnum flag)
def reformat_element(elem,isatomnum=False):
    if isatomnum is True:
        el_correct=element_dict_atnum[elem].symbol    
    else:
        try:
            el_correct=element_dict_atname[elem.lower()].symbol
        except KeyError:
            print("Element-string: {} not found in element-dictionary!".format(elem))
            print("This is not a valid element as defined in ASH source-file: dictionaries_lists.py")
            print("Fix element-information in coordinate-file.")
            exit()
    return el_correct
class Element:
    def __init__(self, name,symbol,atomnumber):
        #Fine-structure constant (2018 CODATA recommended value)
        self.name = None
        self.symbol = symbol
        self.atomnumber=atomnumber  

element_dict_atname = {'h':Element('hydrogen', 'H', 1), 'he':Element('helium', 'He', 2), 'li':Element('lithium', 'Li', 3),
'be':Element('beryllium', 'Be', 4), 'b':Element('boron', 'B', 5), 'c':Element('carbon', 'C', 6), 'n':Element('nitrogen', 'N', 7),
'o':Element('oxygen', 'O', 8), 'f':Element('fluorine', 'F', 9), 'ne':Element('neon', 'Ne', 10), 'na':Element('sodium', 'Na', 11),
'mg':Element('magnesium', 'Mg', 12), 'al':Element('aluminum', 'Al', 13), 'si':Element('silicon', 'Si', 14), 'p':Element('phosphorus', 'P', 15),
's':Element('sulfur', 'S', 16), 'cl':Element('chlorine', 'Cl', 17), 'ar':Element('argon', 'Ar', 18), 'k':Element('potassium', 'K', 19),
'ca':Element('calcium', 'Ca', 20), 'sc':Element('scandium', 'Sc', 21), 'ti':Element('titanium', 'Ti', 22), 'v':Element('vanadium', 'V', 23),
'cr':Element('chromium', 'Cr', 24), 'mn':Element('manganese', 'Mn', 25), 'fe':Element('iron', 'Fe', 26), 'co':Element('cobalt', 'Co', 27),
'ni':Element('nickel', 'Ni', 28), 'cu':Element('copper', 'Cu', 29), 'zn':Element('zinc', 'Zn', 30), 'ga':Element('gallium', 'Ga', 31), 'ge':Element('germanium', 'Ge', 32),
'as':Element('arsenic', 'As', 33), 'se':Element('selenium', 'Se', 34), 'br':Element('bromine', 'Br', 35), 'kr':Element('krypton', 'Kr', 36),
'rb':Element('rubidium', 'Rb', 37), 'sr':Element('strontium', 'Sr', 38),
'y':Element('yttrium', 'Y', 39), 'zr':Element('zirconium', 'Zr', 40), 'nb':Element('niobium', 'Nb', 41), 'mo':Element('molybdenum', 'Mo', 42),
'tc':Element('technetium', 'Tc', 43), 'ru':Element('ruthenium', 'Ru', 44), 'rh':Element('rhodium', 'Rh', 45),'pd':Element('palladium', 'Pd', 46),
'ag':Element('silver', 'Ag', 47), 'cd':Element('cadmium', 'Cd', 48),
'in':Element('indium', 'In', 49), 'sn':Element('tin', 'Sn', 50), 'sb':Element('antimony', 'Sb', 51), 'te':Element('tellurium', 'Te', 52), 'i':Element('iodine', 'I', 53), 'xe':Element('xenon', 'Xe', 54),
'cs':Element('cesium', 'Cs', 55), 'ba':Element('barium', 'Ba', 56),
'hf':Element('hafnium', 'Hf', 72), 'ta':Element('tantalum', 'Ta', 73), 'w':Element('tungsten', 'W', 74), 're':Element('rhenium', 'Re', 75), 'os':Element('osmium', 'Os', 76),
'ir':Element('iridium', 'Ir', 77), 'pt':Element('platinum', 'Pt', 78), 'au':Element('gold', 'Au', 79), 'hg':Element('mercury', 'Hg', 80)}

element_dict_atnum = {1:Element('hydrogen', 'H', 1), 2:Element('helium', 'He', 2), 3:Element('lithium', 'Li', 3),
4:Element('beryllium', 'Be', 4), 5:Element('boron', 'B', 5), 6:Element('carbon', 'C', 6), 7:Element('nitrogen', 'N', 7),
8:Element('oxygen', 'O', 8), 9:Element('fluorine', 'F', 9), 10:Element('neon', 'Ne', 10), 11:Element('sodium', 'Na', 11),
12:Element('magnesium', 'Mg', 12), 13:Element('aluminum', 'Al', 13), 14:Element('silicon', 'Si', 14), 15:Element('phosphorus', 'P', 15),
16:Element('sulfur', 'S', 16), 17:Element('chlorine', 'Cl', 17), 18:Element('argon', 'Ar', 18), 19:Element('potassium', 'K', 19),
20:Element('calcium', 'Ca', 20), 21:Element('scandium', 'Sc', 21), 22:Element('titanium', 'Ti', 22), 23:Element('vanadium', 'V', 23),
24:Element('chromium', 'Cr', 24), 25:Element('manganese', 'Mn', 25), 26:Element('iron', 'Fe', 26), 27:Element('cobalt', 'Co', 27),
28:Element('nickel', 'Ni', 28), 29:Element('copper', 'Cu', 29), 30:Element('zinc', 'Zn', 30), 31:Element('gallium', 'Ga', 31), 32:Element('germanium', 'Ge', 32),
33:Element('arsenic', 'As', 33), 34:Element('selenium', 'Se', 34), 35:Element('bromine', 'Br', 35), 36:Element('krypton', 'Kr', 36),
37:Element('rubidium', 'Rb', 37), 38:Element('strontium', 'Sr', 38),
39:Element('yttrium', 'Y', 39), 40:Element('zirconium', 'Zr', 40), 41:Element('niobium', 'Nb', 41), 42:Element('molybdenum', 'Mo', 42),
43:Element('technetium', 'Tc', 43), 44:Element('ruthenium', 'Ru', 44), 45:Element('rhodium', 'Rh', 45),46:Element('palladium', 'Pd', 46),
47:Element('silver', 'Ag', 47), 48:Element('cadmium', 'Cd', 48),
49:Element('indium', 'In', 49), 50:Element('tin', 'Sn', 50), 51:Element('antimony', 'Sb', 51), 52:Element('tellurium', 'Te', 52), 53:Element('iodine', 'I', 53), 54:Element('xenon', 'Xe', 54),
55:Element('cesium', 'Cs', 55), 56:Element('barium', 'Ba', 56),
72:Element('hafnium', 'Hf', 72), 73:Element('tantalum', 'Ta', 73), 74:Element('tungsten', 'W', 74), 75:Element('rhenium', 'Re', 75), 76:Element('osmium', 'Os', 76),
77:Element('iridium', 'Ir', 77), 78:Element('platinum', 'Pt', 78), 79:Element('gold', 'Au', 79), 80:Element('mercury', 'Hg', 80)}

#Read XYZ file
def read_xyzfile(filename):
    #Will accept atom-numbers as well as symbols
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
            'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
            'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
            'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    print("Reading coordinates from XYZfile {} ".format(filename))
    coords=[]
    elems=[]
    with open(filename) as f:
        for count,line in enumerate(f):
            if count == 0:
                numatoms=int(line.split()[0])
            if count > 1:
                if len(line.strip()) > 0:
                    if isint(line.split()[0]) is True:
                        #Grabbing element as atomnumber and reformatting
                        #el=dictionaries_lists.element_dict_atnum[int(line.split()[0])].symbol
                        el=reformat_element(int(line.split()[0]),isatomnum=True)
                        elems.append(el)
                    else:
                        #Grabbing element as symbol and reformatting just in case
                        el=reformat_element(line.split()[0])
                        elems.append(el)
                    coords.append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
    assert len(coords) == numatoms, "Number of coordinates does not match header line"
    assert len(coords) == len(elems), "Number of coordinates does not match elements."
    return elems,coords

def read_yggfile(fragfile):
    coordline=False
    elems=[]
    coords=[]
    print("Reading coordinates from ASH fragmentfile", fragfile)
    with open(fragfile) as file:
        for line in file:
            if '=====' in line:
                coordline=False
            if coordline==True:
                el=line.split()[1]
                c_x=float(line.split()[2]);c_y=float(line.split()[3]);c_z=float(line.split()[4])
                elems.append(el)
                coords.append([c_x, c_y,c_z])
            if '-----------------------' in line:
                coordline=True
    return elems,coords

#Is variable an integer
def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False
#Read list of integers from file. Output list of integers. Ignores blanklines, return chars, non-int characters
#offset option: shifts integers by a value (e.g. 1 or -1)
def read_intlist_from_file(file,offset=0):
    atomslist=[]
    lines=readlinesfile(file)
    for line in lines:
        for l in line.split():
            #Removing non-numeric part
            l = ''.join(i for i in l if i.isdigit())
            if isint(l):
                atomslist.append(int(l)+offset)
    atomslist.sort()
    return atomslist
#Read lines of file by slurping.
def readlinesfile(filename):
  try:
    f=open(filename)
    out=f.readlines()
    f.close()
  except IOError:
    print('File %s does not exist!' % (filename))
    exit(12)
  return out

#Write XYZfile provided list of elements and list of list of coords and filename
def write_xyzfile(elems,coords,name,printlevel=2):
    with open(name+'.xyz', 'w') as ofile:
        ofile.write(str(len(elems))+'\n')
        ofile.write("title"+'\n')
        for el,c in zip(elems,coords):
            line="{:4} {:16.12f} {:16.12f} {:16.12f}".format(el,c[0], c[1], c[2])
            ofile.write(line+'\n')
    if printlevel >= 2:
        print("Wrote XYZ file:", name+'.xyz')
        
#Standalone fragment-editing script for Ash

#Reads in Ash fragment file and atomslist file and output XYZ coordinate file that can be visualized in e.g. Chemcraft and edited

#Fragfile is always first argument
try:
    fragfile=sys.argv[1]
    print("Reading file:", fragfile)
except:
    print("Error:")
    print("Please provide an Ash fragment file and atomindices file as argument:")
    print("Examples:")
    print("fragedit.py file.ygg qmatoms")
    print("fragedit.py file.xyz qmatoms")
    print("fragedit.py file.xyz qmatoms index1  (if using 1-based atom indices)")
    exit(1)


#Read coordinates from XYZ-files
if '.xyz' in fragfile:
    elems,coords=read_xyzfile(fragfile)
elif '.ygg' in fragfile:
    elems,coords=read_yggfile(fragfile)
else:
    print("Unknown file format. Only .xyz and .ygg files currently supported")
    print("")
    exit()
# PDB-file could be a bit tricky to support. Should we use element-column or atomname column?
#elif '.pdb' in fragfile:
#    elems,coords=read_pdbfile(fragfile)


#Try to process a list-of-atom-indices file if provided
try:
    selatoms_file = sys.argv[2]
    atomslist=read_intlist_from_file(selatoms_file)
    atomslist.sort()
    print("Read atom indices from file:", selatoms_file)
    print("atomslist:", atomslist)
except IndexError:
    print("Please provide file containing atom indices as 2nd argument")
    exit()
#Indexing argument
try:
    extraarg=sys.argv[3]
except:
    extraarg=None

#Grab the relevant elems and coords
if extraarg=="index1":
    print("!!Warning!!: using 1-based indexing as chosen by user")
    subcoords=[coords[i-1] for i in atomslist]
    subelems=[elems[i-1] for i in atomslist]
else:
    print("!!Warning!!: assuming 0-based indexing by default")
    print("Do: fragedit.py fragfile atomslistfile index1  if you want to use 1-based indexing")
    subcoords=[coords[i] for i in atomslist]
    subelems=[elems[i] for i in atomslist]
print("Grabbing XYZ coordinates from coordinatefile: {} for atom indices: {}".format(fragfile,atomslist))
write_xyzfile(subelems,subcoords,"fragment")
