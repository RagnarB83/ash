import settings_ash
import os
import numpy as np
import time
from functools import wraps

#ANSI colors: http://jafrog.com/2013/11/23/colors-in-terminal.html
if settings_ash.use_ANSI_color is True:
    class BC:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        OKMAGENTA= '\033[95m'
        OKRED= '\033[31m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        END = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
else:
    class BC:
        HEADER = ''; OKBLUE = ''; OKGREEN = ''; OKMAGENTA= ''; OKRED= ''; WARNING = ''; FAIL = ''; END = ''; BOLD = ''; UNDERLINE = ''
        
#Get ranges of integers from list. Returns string of ranges. Used to communitcate with crest and xtbs
def int_ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    l_of_tuples=list(zip(edges, edges))

    newstring=""
    for i in l_of_tuples:
        if i[0] != i[1]:
            newstring+=str(i[0])+'-'+str(i[1])+','
        else:
            newstring+=str(i[0])+','
    #remove final ,
    newstring=newstring[0:-1]
    return newstring


#Debug print. Behaves like print but reads global debug var first
def printdebug(string,var=''):
    global debugflag
    if debugflag is True:
        print(BC.OKRED,string,var,BC.END)

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn:" + fn.__name__ + " took " + str(t2-t1) + " seconds")
        return result
    return measure_time

#Grep-style function to find a line in file and return a list of words
#TODO: Make more advanced
def pygrep(string,file):
    with open(file) as f:
        for line in f:
            if string in line:
                stringlist=line.split()
                return stringlist



# Give difference of two lists, sorted. List1: Bigger list
def listdiff(list1, list2):
    diff = (list(set(list1) - set(list2)))
    diff.sort()
    return diff

# Range function for floats
def frange(start, stop=None, step=None):
    # if stop and step argument is None set start=0.0 and step = 1.0
    start = float(start)
    if stop == None:
        stop = start + 0.0
        start = 0.0
    if step == None:
        step = 1.0

    #print("start= ", start, "stop= ", stop, "step= ", step)

    count = 0
    while True:
        temp = float(start + count * step)
        if step > 0 and temp >= stop:
            break
        elif step < 0 and temp <= stop:
            break
        yield temp
        count += 1


def print_line_with_mainheader(line):
    print(BC.OKGREEN,"--------------------------------------------------",BC.END)
    print(BC.OKGREEN,BC.BOLD,line,BC.END)
    print(BC.OKGREEN,"--------------------------------------------------",BC.END)

def print_line_with_subheader1(line):
    print(BC.OKBLUE,"--------------------------------------------------",BC.END)
    print(BC.OKBLUE,BC.BOLD,line,BC.END)
    print(BC.OKBLUE,"--------------------------------------------------",BC.END)


#Inserts line into file for matched string.
#option: Once=True means only added for first match
def insert_line_into_file(file,string,addedstring, Once=True):
    Added=False
    with open(file, 'r') as ffr:
        contents = ffr.readlines()
    with open(file, 'w') as ffw:
        for l in contents:
            ffw.write(l)
            if string in l:
                if Added is False:
                    ffw.write(addedstring+'\n')
                    if Once is True:
                        Added=True

def blankline():
    print("")

#Is variable an integer
def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False

#Is integer odd
def isodd(n):
    if (n % 2) == 0:
        return False
    else:
        return True


#Is it possible to interpret string/number as float.
#Note: integer variable/string can be interpreted. See also is_string_float_withdecimal below
def isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#Is string a float with single decimal point
def is_string_float_withdecimal(s):
    #Checking if single . in string
    if s.count('.') != 1:
        return False
    #Check if number can be interpreted as float
    try:
        float(s)
        return True
    except ValueError:
        return False

#Check if list of integers is sorted or not.
def is_integerlist_ordered(list):
    list_s=sorted(list)
    if list == list_s:
        return True
    else:
        return False
    
def islist(l):
    if type(l) == list:
        return True
    else:
        return False

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

#Find substring of string between left and right parts
def find_between(s, start, end):
  return (s.split(start))[1].split(end)[0]


#Read list of integers from file. Output list of integers. Ignores blanklines, return chars, non-int characters
#offset option: shifts integers by a value (e.g. 1 or -1)
def read_intlist_from_file(file,offset=0):
    list=[]
    lines=readlinesfile(file)
    for line in lines:
        for l in line.split():
            #Removing non-numeric part
            l = ''.join(i for i in l if i.isdigit())
            if isint(l):
                list.append(int(l)+offset)
    list.sort()
    return list

#Read list of flaots from file. Output list of floats.
#Works for single-line with numbers and multi-lines
def read_floatlist_from_file(file):
    list=[]
    lines=readlinesfile(file)
    for line in lines:
        for l in line.split():
            if isfloat(l):
                list.append(float(l))
    list.sort()
    return list



#Write a string to file simply
def writestringtofile(string,file):
    with open(file, 'w') as f:
        f.write(string)

#Natural (human) sorting of list
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

# Reverse read function.
def reverse_lines(filename, BUFSIZE=20480):
    #f = open(filename, "r")
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
            for L in buf[i+1:].split("\n")[::-1]:
                yield L
            remainder = buf[:i]
        if p == 0:
            break
    yield remainder

def clean_number(number):
    return np.real_if_close(number)


#Function to get unique values
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
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result


#Extract column from matrix
def column(matrix, i):
    return [row[i] for row in matrix]

#From Knarr
def printDate():
    import datetime
    dagur = datetime.datetime.now()
    string1 = dagur.strftime('%d.%m.%Y')
    string2 = dagur.strftime('%H:%M')
    print('Time:%s Date:%s' % (string2, string1))
    return None