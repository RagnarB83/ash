#Module for calculating PhotoElectron/PhotoIonization Spectra
#####################################

import glob
import numpy as np
import os
import sys
import subprocess as sp
import struct
#import functions_coords
#from functions_coords import *
from functions_ORCA import *
from functions_general import *
from ash import *
import constants
from elstructure_functions import *

class bcolors:
    HEADER = '\033[95m' ; OKBLUE = '\033[94m'; OKGREEN = '\033[92m'; OKMAGENTA= '\033[95m'; WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'

eldict={'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Mo':42,'W':74,'Ru':44,'I':53}


#Cleanup function. Delete MO-files, det files, etc.
def cleanup():
    """
    Cleanup function for PES calculations. Deletes result and temporary files
    :return:
    """
    files=['AO_overl', 'dets_final', 'dets_init', 'memlog', 'wfovl.inp', 'wfovl.out', 'mos_final', 'mos_init']
    print("Cleaning up files: ", files)
    for file in files:
        try:
            os.remove(file)
        except:
            pass

#Readfile function
def readfile(filename):
  try:
    f=open(filename)
    out=f.readlines()
    f.close()
  except IOError:
    print('File {} does not exist!'.format(filename))
    sys.exit(12)
  return out

#Simple write long-string to file
def writestringtofile(string,file):
    with open(file, 'w') as f:
        f.write(string)

#Get Atomic overlap matrix from GBW file
#From SHARC
def saveAOmatrix(file, orcadir=None):
    NAO,Smat=get_smat_from_gbw(file, orcadir=orcadir)
    string='%i %i\n' % (NAO,NAO)
    for irow in range(NAO):
        for icol in range(NAO):
            string+='% .7e ' % (Smat[icol][irow])
        string+='\n'
    outfile='AO_overl'
    with open(outfile, 'w') as ofile:
        ofile.write(string)

#Get smat from GBW. Converted to Python3 From Python2 function in SHARC
def get_smat_from_gbw(file1, file2='', orcadir=None):

    if not file2:
      file2=file1

    # run orca_fragovl
    string=orcadir+'/orca_fragovl %s %s' % (file1,file2)
    try:
      proc=sp.Popen(string,shell=True,stdout=sp.PIPE,stderr=sp.PIPE)
    except OSError:
      print('Call has had some serious problems:',OSError)
      sys.exit(89)
    comm=proc.communicate()
    #Python 3 decoding necessary
    comm=comm[0].decode('utf-8')
    #print(comm)
    #exit()
    out=comm.split('\n')

    # get size of matrix
    for line in reversed(out):
      #print line
      s=line.split()
      if len(s)>=1:
        NAO=int(line.split()[0])+1
        break

    # read matrix
    #Python3 conversion necessary here
    nblock=6
    ao_ovl=[ [ 0. for i in range(NAO) ] for j in range(NAO) ]
    for x in range(NAO):
      for y in range(NAO):
        block=int(x/nblock)
        xoffset=x%nblock+1
        yoffset=block*(NAO+1)+y+10
        #Python3 issue with floats vs indices for block
        ao_ovl[x][y]=float( out[yoffset].split()[xoffset])

    return NAO,ao_ovl

#Get MO coefficients from GBW file. Converted to Python3 From Python2 function in SHARC
def get_MO_from_gbw(filename,restr,frozencore):

    # run orca_fragovl
    string='orca_fragovl %s %s' % (filename,filename)
    try:
      proc=sp.Popen(string,shell=True,stdout=sp.PIPE,stderr=sp.PIPE)
    except OSError:
      print('Call have had some serious problems:',OSError)
      sys.exit(80)
    comm=proc.communicate()
    #Python 3 decoding necessary
    comm=comm[0].decode('utf-8')
    #print(comm)
    data=comm.split('\n')
    #print(data)
    # get size of matrix
    for line in reversed(data):
      #print(line)
      s=line.split()
      if len(s)>=1:
        NAO=int(line.split()[0])+1
        break

    #job=QMin['IJOB']
    #restr=QMin['jobs'][job]['restr']

    # find MO block
    iline=-1
    while True:
      iline+=1
      if len(data)<=iline:
        print('MOs not found!')
        sys.exit(81)
      line=data[iline]
      if 'FRAGMENT A MOs MATRIX' in line:
        break
    iline+=3
    # formatting
    nblock=6
    npre=11
    ndigits=16
    # get coefficients for alpha
    NMO_A=NAO
    MO_A=[ [ 0. for i in range(NAO) ] for j in range(NMO_A) ]
    for imo in range(NMO_A):
      #RB. Changed to floor division here
      jblock=imo//nblock
      jcol =imo%nblock
      for iao in range(NAO):
        shift=max(0,len(str(iao))-3)
        jline=iline + jblock*(NAO+1) + iao
        line=data[jline]
        val=float( line[npre+shift+jcol*ndigits : npre+shift+ndigits+jcol*ndigits] )
        MO_A[imo][iao]=val
    #Rb. changed to floor division here
    iline+=(NAO//nblock+1)*(NAO+1)
    #print("iline:", iline)
    # coefficients for beta
    if not restr:
      #RB. New definition of iline due to bug in original version of formatting change??
      iline=jline+2
      NMO_B=NAO
      MO_B=[ [ 0. for i in range(NAO) ] for j in range(NMO_B) ]
      for imo in range(NMO_B):
        #print("imo:", imo)
        #Changed to floor division here for Python3
        jblock=imo//nblock
        #print("jblock", jblock)
        jcol =imo%nblock
        #print("jcol:", jcol)
        #print("NAO:", NAO)
        #print("range(NAO)", range(NAO))
        #print("data[184]:", data[184])
        for iao in range(NAO):
          #print("iao:", iao)
          shift=max(0,len(str(iao))-3)
          jline=iline + jblock*(NAO+1) + iao
          line=data[jline]
          val=float( line[npre+shift+jcol*ndigits : npre+shift+ndigits+jcol*ndigits] )
          MO_B[imo][iao]=val


    NMO=NMO_A      -  frozencore
    if restr:
        NMO=NMO_A      -  frozencore
    else:
        NMO=NMO_A+NMO_B-2*frozencore

    # make string
    string='''2mocoef
header
 1
MO-coefficients from Orca
 1
 %i   %i
 a
mocoef
(*)
''' % (NAO,NMO)
    x=0
    for imo,mo in enumerate(MO_A):
        if imo<frozencore:
            continue
        for c in mo:
            if x>=3:
                string+='\n'
                x=0
            string+='% 6.12e ' % c
            x+=1
        if x>0:
            string+='\n'
            x=0
    if not restr:
        x=0
        for imo,mo in enumerate(MO_B):
            if imo<frozencore:
                continue
            for c in mo:
                if x>=3:
                    string+='\n'
                    x=0
                string+='% 6.12e ' % c
                x+=1
            if x>0:
                string+='\n'
                x=0
    string+='orbocc\n(*)\n'
    x=0
    for i in range(NMO):
        if x>=3:
            string+='\n'
            x=0
        string+='% 6.12e ' % (0.0)
        x+=1

    return string


#RB. New function
#get determinant-string output for single-determinant case
def get_dets_from_single(logfile,restr,gscharge,gsmult,totnuccharge,frozencore):
    print("Inside get_dets_from_single")
    # get infos from logfile
    data=readfile(logfile)
    infos={}
    for iline,line in enumerate(data):
      if '# of contracted basis functions' in line:
        infos['nbsuse']=int(line.split()[-1])
      if 'Orbital ranges used for CIS calculation:' in line:
        s=data[iline+1].replace('.',' ').split()
        infos['NFC']=int(s[3])
        infos['NOA']=int(s[4])-int(s[3])+1
        infos['NVA']=int(s[7])-int(s[6])+1
        if restr:
          infos['NOB']=infos['NOA']
          infos['NVB']=infos['NVA']
        else:
          s=data[iline+2].replace('.',' ').split()
          infos['NOB']=int(s[4])-int(s[3])+1
          infos['NVB']=int(s[7])-int(s[6])+1

    if not 'NOA' in  infos:
      charge=gscharge
      #charge=QMin['chargemap'][gsmult]
      nelec=float(totnuccharge-charge)
      infos['NOA']=int(nelec/2. + float(gsmult-1)/2. )
      infos['NOB']=int(nelec/2. - float(gsmult-1)/2. )
      infos['NVA']=infos['nbsuse']-infos['NOA']
      infos['NVB']=infos['nbsuse']-infos['NOB']
      infos['NFC']=0


    # get ground state configuration
    # make step vectors (0:empty, 1:alpha, 2:beta, 3:docc)
    if restr:
        occ_A=[ 3 for i in range(infos['NFC']+infos['NOA']) ]+[ 0 for i in range(infos['NVA']) ]
    if not restr:
        occ_A=[ 1 for i in range(infos['NFC']+infos['NOA']) ]+[ 0 for i in range(infos['NVA']) ]
        occ_B=[ 2 for i in range(infos['NFC']+infos['NOB']) ]+[ 0 for i in range(infos['NVB']) ]
    occ_A=tuple(occ_A)
    if not restr:
        occ_B=tuple(occ_B)

    # get eigenvectors
    eigenvectors={}
    eigenvectors[gsmult]=[]
    if restr:
        key=tuple(occ_A[frozencore:])
    else:
        key=tuple(occ_A[frozencore:]+occ_B[frozencore:])
    eigenvectors[gsmult].append( {key:1.0} )
    strings={}
    print("Final (single-det case) eigenvectors:", eigenvectors)
    print("format_ci_vectors(eigenvectors[gsmult] :", format_ci_vectors(eigenvectors[gsmult]))
    strings["dets."+str(gsmult)] = format_ci_vectors(eigenvectors[gsmult])
    print("strings: ", strings)
    return strings



#Get determinants from ORCA cisfile.
#Converted to Python3 from function in SHARC
def get_dets_from_cis(logfile,cisfilename,restr,mults,gscharge,gsmult,totnuccharge,nstates_to_extract,nstates_to_skip,no_tda,frozencore,wfthres):
    print("Inside get_dets_from_cis")
    print("")
    print("logfile", logfile)
    print("cisfilename:", cisfilename)
    print("restr:", restr)
    print("mults:", mults)
    print("gscharge:", gscharge)
    print("gsmult:", gsmult)
    print("totnuccharge:", totnuccharge)
    print("nstates_to_extract:", nstates_to_extract)
    print("nstates_to_skip:", nstates_to_skip)
    print("no_tda:", no_tda)
    print("frozencore:", frozencore)
    print("wfthres", wfthres)
    print("nstates_to_extract:", nstates_to_extract)

    #print restr,mults,gsmult,nstates_to_extract
    #print "RB.....b"
    # get infos from logfile
    #logfile=os.path.join(os.path.dirname(filename),'ORCA.log')
    data=readfile(logfile)
    infos={}
    for iline,line in enumerate(data):
      if '# of contracted basis functions' in line:
        infos['nbsuse']=int(line.split()[-1])
      if 'Orbital ranges used for CIS calculation:' in line:
        s=data[iline+1].replace('.',' ').split()
        infos['NFC']=int(s[3])
        infos['NOA']=int(s[4])-int(s[3])+1
        infos['NVA']=int(s[7])-int(s[6])+1
        if restr:
          infos['NOB']=infos['NOA']
          infos['NVB']=infos['NVA']
        else:
          s=data[iline+2].replace('.',' ').split()
          infos['NOB']=int(s[4])-int(s[3])+1
          infos['NVB']=int(s[7])-int(s[6])+1
    if not 'NOA' in  infos:
      charge=gscharge
      nelec=float(totnuccharge-charge)
      infos['NOA']=int(nelec/2. + float(gsmult-1)/2. )
      infos['NOB']=int(nelec/2. - float(gsmult-1)/2. )
      infos['NVA']=infos['nbsuse']-infos['NOA']
      infos['NVB']=infos['nbsuse']-infos['NOB']
      infos['NFC']=0
    else:
      # get all info from cis file
      CCfile=open(cisfilename,'rb')
      nvec  =struct.unpack('i', CCfile.read(4))[0]
      header=[ struct.unpack('i', CCfile.read(4))[0] for i in range(8) ]
      print(infos)
      print(header)
      if infos['NOA']!=header[1]-header[0]+1:
        print('Number of orbitals in %s not consistent' % filename)
        sys.exit(82)
      if infos['NVA']!=header[3]-header[2]+1:
        print('Number of orbitals in %s not consistent' % filename)
        sys.exit(83)
      if not restr:
        if infos['NOB']!=header[5]-header[4]+1:
          print('Number of orbitals in %s not consistent' % filename)
          sys.exit(84)
        if infos['NVB']!=header[7]-header[6]+1:
          print('Number of orbitals in %s not consistent' % filename)
          sys.exit(85)
      if no_tda:
        nstates_onfile=nvec/2
      else:
        nstates_onfile=nvec


    # get ground state configuration
    # make step vectors (0:empty, 1:alpha, 2:beta, 3:docc)
    if restr:
        occ_A=[ 3 for i in range(infos['NFC']+infos['NOA']) ]+[ 0 for i in range(infos['NVA']) ]
    if not restr:
        occ_A=[ 1 for i in range(infos['NFC']+infos['NOA']) ]+[ 0 for i in range(infos['NVA']) ]
        occ_B=[ 2 for i in range(infos['NFC']+infos['NOB']) ]+[ 0 for i in range(infos['NVB']) ]
    occ_A=tuple(occ_A)
    if not restr:
        occ_B=tuple(occ_B)

    # get infos
    nocc_A=infos['NOA']
    nvir_A=infos['NVA']
    nocc_B=infos['NOB']
    nvir_B=infos['NVB']

    # get eigenvectors
    eigenvectors={}
    for imult,mult in enumerate(mults):
        eigenvectors[mult]=[]
        if mult==gsmult:
            # add ground state
            if restr:
                key=tuple(occ_A[frozencore:])
            else:
                key=tuple(occ_A[frozencore:]+occ_B[frozencore:])
            eigenvectors[mult].append( {key:1.0} )
        #print("struct.unpack('d', CCfile.read(8))[0]:", struct.unpack('d', CCfile.read(8))[0])
        #for istate in range(nstates_to_extract[mult-1]):
        for istate in range(nstates_to_extract[-1] - 1):
            CCfile.read(40)
            dets={}
            #print("dets:", dets)
            #print("header:", header)
            #print("--------")
            #print("range(header[0],header[1]+1):", range(header[0],header[1]+1))
            for iocc in range(header[0],header[1]+1):
              #print("--------")
              #print("range(header[2],header[3]+1):", range(header[2],header[3]+1))
              for ivirt in range(header[2],header[3]+1):
                #print("istate, iocc, ivirt", istate, iocc, ivirt)
                #print("here1")
                dets[ (iocc,ivirt,1) ]=struct.unpack('d', CCfile.read(8))[0]
            #print("1 case done!!")
            #pprint.pprint(dets)
            if not restr:
              #print("here2")
              #print("range(header[4],header[5]+1):", range(header[4],header[5]+1))
              for iocc in range(header[4],header[5]+1):
                #print("range(header[6],header[7]+1):", range(header[6],header[7]+1))
                for ivirt in range(header[6],header[7]+1):
                  #print("istate, iocc, ivirt", istate, iocc, ivirt)
                  #print("struct.unpack('d', CCfile.read(8))[0]:", struct.unpack('d', CCfile.read(8))[0])
                  dets[ (iocc,ivirt,2) ]=struct.unpack('d', CCfile.read(8))[0]
            if no_tda:
              CCfile.read(40)
              for iocc in range(header[0],header[1]+1):
                for ivirt in range(header[2],header[3]+1):
                  dets[ (iocc,ivirt,1) ]+=struct.unpack('d', CCfile.read(8))[0]
                  dets[ (iocc,ivirt,1) ]/=2.
              if not restr:
                for iocc in range(header[4],header[5]+1):
                  for ivirt in range(header[6],header[7]+1):
                    #print("rb here...")
                    dets[ (iocc,ivirt,2) ]+=struct.unpack('d', CCfile.read(8))[0]
                    dets[ (iocc,ivirt,2) ]/=2.

            #pprint.pprint(dets)
            # truncate vectors
            norm=0.
            for k in sorted(dets,key=lambda x: dets[x]**2,reverse=True):
                factor=1.
                if norm>factor*wfthres:
                    del dets[k]
                    continue
                norm+=dets[k]**2
            #pprint.pprint(dets)
            # create strings and expand singlets
            dets2={}
            if restr:
                for iocc,ivirt,dummy in dets:
                    # singlet
                    if mult==1:
                        # alpha excitation
                        key=list(occ_A)
                        key[iocc]=2
                        key[ivirt]=1
                        dets2[tuple(key)]=dets[ (iocc,ivirt,dummy) ]*math.sqrt(0.5)
                        # beta excitation
                        key[iocc]=1
                        key[ivirt]=2
                        dets2[tuple(key)]=dets[ (iocc,ivirt,dummy) ]*math.sqrt(0.5)
                    # triplet
                    elif mult==3:
                        key=list(occ_A)
                        key[iocc]=1
                        key[ivirt]=1
                        dets2[tuple(key)]=dets[ (iocc,ivirt,dummy) ]
            else:
                for iocc,ivirt,dummy in dets:
                    if dummy==1:
                        key=list(occ_A+occ_B)
                        key[iocc]=0
                        key[ivirt]=1
                        dets2[tuple(key)]=dets[ (iocc,ivirt,dummy) ]
                    elif dummy==2:
                        key=list(occ_A+occ_B)
                        key[infos['NFC']+nocc_A+nvir_A + iocc]=0
                        key[infos['NFC']+nocc_A+nvir_A + ivirt]=2
                        dets2[tuple(key)]=dets[ (iocc,ivirt,dummy) ]
            #pprint.pprint(dets2)
            # remove frozen core
            dets3={}
            for key in dets2:
                problem=False
                if restr:
                    if any( [key[i]!=3 for i in range(frozencore) ] ):
                        problem=True
                else:
                    if any( [key[i]!=1 for i in range(frozencore) ] ):
                        problem=True
                    if any( [key[i]!=2 for i in range(nocc_A+nvir_A+frozencore, nocc_A+nvir_A + 2*frozencore) ] ):
                        problem=True
                if problem:
                    print('WARNING: Non-occupied orbital inside frozen core! Skipping ...')
                    continue
                    #sys.exit(86)
                if restr:
                    key2=key[frozencore:]
                else:
                    key2=key[frozencore:frozencore+nocc_A+nvir_A] + key[nocc_A+nvir_A+2*frozencore:]
                dets3[key2]=dets2[key]
            #pprint.pprint(dets3)
            # append
            eigenvectors[mult].append(dets3)
        # skip extra roots
        #RB Removing mult in range. Weird
        for istate in range(nstates_to_skip[-1]):
            CCfile.read(40)
            for iocc in range(header[0],header[1]+1):
              for ivirt in range(header[2],header[3]+1):
                CCfile.read(8)
            if not restr:
              for iocc in range(header[4],header[5]+1):
                for ivirt in range(header[6],header[7]+1):
                  CCfile.read(8)
            if no_da:
              CCfile.read(40)
              for iocc in range(header[0],header[1]+1):
                for ivirt in range(header[2],header[3]+1):
                  CCfile.read(8)
              if not restr:
                for iocc in range(header[4],header[5]+1):
                  for ivirt in range(header[6],header[7]+1):
                    CCfile.read(8)


    strings={}
    #print("Final (CIS) eigenvectors:", eigenvectors)
    for imult,mult in enumerate(mults):
        filename='dets.%i' % mult
        strings[filename]=format_ci_vectors(eigenvectors[mult])
    return strings

def format_ci_vectors(ci_vectors):

    # get nstates, norb and ndets
    alldets=set()
    for dets in ci_vectors:
        for key in dets:
            alldets.add(key)
    ndets=len(alldets)
    nstates=len(ci_vectors)
    norb=len(next(iter(alldets)))

    string='%i %i %i\n' % (nstates,norb,ndets)
    for det in sorted(alldets,reverse=True):
        for o in det:
            if o==0:
                string+='e'
            elif o==1:
                string+='a'
            elif o==2:
                string+='b'
            elif o==3:
                string+='d'
        for istate in range(len(ci_vectors)):
            if det in ci_vectors[istate]:
                string+=' %11.7f ' % ci_vectors[istate][det]
            else:
                string+=' %11.7f ' % 0.
        string+='\n'
    return string

#Run wfoverlap program
def run_wfoverlap(wfoverlapinput,path_wfoverlap,memory):
    wfoverlapfilefile = open('wfovl.inp', 'w')
    for l in wfoverlapinput:
        wfoverlapfilefile.write(l)
    wfoverlapfilefile.close()
    wfcommand="{} -m {} -f wfovl.inp".format(path_wfoverlap,memory)
    print("Running wfoverlap program:")
    print("may take a while...")
    print(wfcommand)
    print("Using memory: {} MB".format(memory))
    print("OMP num threads available to WFoverlap: ", os.environ["OMP_NUM_THREADS"])
    try:
        proc=sp.Popen(wfcommand,shell=True,stdout=sp.PIPE,stderr=sp.PIPE)
        wfoverlapout=proc.communicate()
        wfoverlapout=wfoverlapout[0].decode('utf-8')
        wfoverlapout=wfoverlapout.split('\n')
        with open("wfovl.out", 'w') as f:
            for b in wfoverlapout:
                f.write(b+'\n')
    except OSError:
        print("Problem calling wfoverlap program.")
    print("Wfoverlap done! See outputfile: wfovl.out")
    return

#Get Dysonnorms from output of wfoverlap
def grabDysonnorms():
    with open("wfovl.out") as wout:
        out=wout.readlines()
    #Getting Dyson norms from output
    dysonnorms=[]
    dysonorbs=[]
    dysonorbitalgrab=False
    for line in out:
        if dysonorbitalgrab==True:
            if 'MO' in line:
                dysonorbs.append(float(line.split()[-1]))
        if '<PsiA ' in line and dysonorbitalgrab==False:
            dysonnorms.append(float(line.split()[-1]))
        if 'Dyson orbitals in reference' in line:
            dysonorbitalgrab=True
    return dysonnorms

#Calculate HOMO number from nuclear charge from XYZ-file and total charge
def HOMOnumbercalc(file,charge,mult):
    el=[]
    with open(file) as f:
        for count,line in enumerate(f):
            if count >1:
                el.append(line.split()[0])
    totnuccharge=0
    for e in el:
        atcharge=eldict[e]
        totnuccharge+=atcharge
    numel=totnuccharge-charge
    HOMOnum_a="unset";HOMOnum_b="unset"
    orcaoffset=-1
    if mult == 1:
        #RHF case. HOMO is numel/2 -1
        HOMOnum_a=(numel/2)+orcaoffset
        HOMOnum_b=(numel/2)+orcaoffset
    elif mult > 1:
        #UHF case.
        numunpel=mult-1
        Doubocc=(numel-numunpel)/2
        HOMOnum_a=Doubocc+numunpel+orcaoffset
        HOMOnum_b=Doubocc+orcaoffset
    return int(HOMOnum_a),int(HOMOnum_b)

def orbitalgrab(file):
    occorbsgrab=False
    virtorbsgrab=False
    endocc="unset"
    tddftgrab="unset"
    tddft="unset"
    bands_alpha=[]
    bands_beta=[]
    virtbands_a=[]
    virtbands_b=[]
    f=[]
    virtf=[]
    spinflag="unset"
    hftyp="unset"

    with open(file) as f:
        for line in f:
            if '%tddft' in line:
                tddft="yes"
            if 'Hartree-Fock type      HFTyp' in line:
                hftyp=line.split()[4]
                #if hftyp=="UHF":
            if hftyp == "RHF":
                spinflag="alpha"
            if 'SPIN UP ORBITALS' in line:
                spinflag="alpha"
            if 'SPIN DOWN ORBITALS' in line:
                spinflag="beta"
            if occorbsgrab==True:
                endocc=line.split()[1]
                if endocc == "0.0000" :
                    occorbsgrab=False
                    virtorbsgrab=True
                else:
                    if spinflag=="alpha":
                        bands_alpha.append(float(line.split()[3]))
                    if spinflag=="beta":
                        bands_beta.append(float(line.split()[3]))
            if virtorbsgrab==True:
                if '------------------' in line:
                    break
                if line == '\n':
                    virtorbsgrab=False
                    spinflag="unset"
                    continue
                if spinflag=="alpha":
                    virtbands_a.append(float(line.split()[3]))
                if spinflag=="beta":
                    virtbands_b.append(float(line.split()[3]))
                endvirt=line.split()[1]
            if 'NO   OCC          E(Eh)            E(eV)' in line:
                occorbsgrab=True
    return bands_alpha, bands_beta, hftyp

def checkORCAfinished(file):
    with open(file) as f:
        for line in f:
            if 'TOTAL RUN TIME:' in line:
                return True

def scfenergygrab(file):
    string='Total Energy       :'

    with open(file) as f:
        for line in f:
            if string in line:
                Energy=float(line.split()[-4])
    return Energy

#CASSCF: Grabbing first root energy. Simplified because of problem
def casscfenergygrab(file):
    #grab=False
    string='Final CASSCF energy       :'
    with open(file) as f:
        for line in f:
            if string in line:
                Energy=float(line.split()[4])
                return Energy
                #Changing from 5 to -2
                #CIPSI: 5 Regular CASSCF: -2  ?
                #Pretty ugly. to be fixed. TODO
                #Energy=float(line.split()[-2])
                #if Energy == 0.0:
                #    Energy=float(line.split()[5])
            #if 'CAS-SCF STATES FOR BLOCK' in line:
            #    grab=True

#CASSCF: Grabbing all root energies
#Slightly tricky funciton because output differs for ICE-CASSCF and regular CASSCF.
#Should be good now.
def casscf_state_energies_grab(file):
    Finished=False
    grab=False
    mult_dict={}
    state_energies=[];Energy=0.0
    #string='STATE '
    #string2='ROOT '
    with open(file) as f:
        for line in f:
            #Stop grabbing lines once we have reached end of table
            if 'SA-CASSCF TRANSITION ENERGIES' in line:
                grab=False
            if 'Spin-Determinant CI Printing' in line:
                grab=False
            if 'Extended CI Printing (values > TPrintWF)' in line:
                grab=False
            #Grabbing STATE lines
            if grab is True and 'STATE ' in line:
                print("Xline: ", line)
                Energy=float(line.split()[5])
                print("Energy :", Energy)
                state_energies.append(Energy)
                print("state_energies :", state_energies)
                if len(state_energies) == roots:
                    mult_dict[mult] = state_energies
                    grab=False
            if grab is True and 'ROOT ' in line:
                print("Yline: ", line)
                Energy=float(line.split()[3])
                print("Energy :", Energy)
                state_energies.append(Energy)
                print("state_energies :", state_energies)
                if len(state_energies) == roots:
                    mult_dict[mult] = state_energies
                    grab=False
            if Finished is True and 'CAS-SCF STATES FOR BLOCK' in line:
                print("line :", line)
                #New mult block. Resetting state-energies.
                state_energies=[];Energy=0.0
                mult=int(line.split()[6])
                print("mult : ", mult)
                #roots=int(line.split()[8])
                roots=int(line.split()[7:9][-1].replace('NROOTS=',''))
                print("roots :", roots)
                grab=True
            #Only grabbing lines once CASSCF calc has converged
            if 'Final CASSCF energy' in line:
                Finished=True
    return mult_dict

#MRCI: Grabbing all root energies
def mrci_state_energies_grab(file):
    grab=False
    blockgrab=False
    grab_blockinfo=False
    block_dict={}
    mult_dict={}
    state_energies=[];Energy=0.0
    string='STATE '
    prev_grabbed_blockinfo=False
    with open(file) as f:
        for line in f:
            #print("line:", line)
            #print("prev_grabbed_blockinfo:", prev_grabbed_blockinfo)
            #print("grab_blockinfo:", grab_blockinfo)
            #Note. Grabbing block info from CASSCF output
            if '<<<<<<<<<<<<<<<<<<INITIAL CI STATE CHECK>>>>>>>>>>>>>>>>>>' in line:
                if prev_grabbed_blockinfo is False:
                    grab_blockinfo = True
                    prev_grabbed_blockinfo=True
                    continue
                else:
                    grab_blockinfo=False
            if grab_blockinfo is True:
                if 'BLOCK' in line:
                    blocknum = int(line.split()[1])
                    mult = int(line.split()[3])
                    roots = int(line.split("=")[-1])
                    block_dict[blocknum] = (mult,roots)
                    #print("block_dict:", block_dict)
                #Only reading 2 blocks (two multiplicities)
                #Unncessary?
                if len(block_dict) == 2:
                    grab_blockinfo = False
            #Grabbing actual MRCI state energies
            if grab is True and string in line:
                Energy=float(line.split()[3])
                state_energies.append(Energy)
                if len(state_energies) == current_roots:
                    mult_dict[currentmult] = state_energies
                    #print("mult_dict:", mult_dict)
                    state_energies=[]
            #Getting info about what block we are currently reading in the output
            if '*              CI-BLOCK' in line:
                blockgrab=True
                currentblock=int(line.split()[-2])
                currentmult=block_dict[currentblock][0]
                current_roots = block_dict[currentblock][1]
            if 'TRANSITION ENERGIES' in line:
                grab = False
            if blockgrab is True:
                if 'Davidson type correc' in line:
                    grab=True
    return mult_dict

#CASSCF: Grab orbital ranges
def casscf_orbitalranges_grab(file):
    grab=False
    with open(file) as f:
        for line in f:
            if grab is True:
                if 'Internal' in line:
                    internal=int(line.split()[-2])
                if 'Active' in line:
                    active=int(line.split()[-2])
                if 'External' in line:
                    external=int(line.split()[-2])
            if 'Determined orbital ranges:' in line:
                grab=True
            if 'Number of rotation parameters' in line:
                grab=False

    return internal,active,external


def tddftgrab(file):
    tddftstates=[]
    with open(file) as f:
        for line in f:
            if 'STATE ' in line:
                tddftstates.append(float(line.split()[5]))
    return tddftstates

# MO-DOS PLOT
def modosplot(occorbs_alpha,occorbs_beta,hftyp):
    #Defining sticks as -1 times MO energy (eV)
    stk_alpha=[]
    stk_beta=[]
    for j in occorbs_alpha:
        stk_alpha.append(-1*j)
    if hftyp == "UHF":
        for k in occorbs_beta:
            stk_beta.append(-1*k)
        stk_beta.sort()
    stk_alpha.sort()
    return stk_alpha,stk_beta

def Gaussian(x, mu, strength, sigma):
    "Produces a Gaussian curve"
    bandshape = (strength)  * np.exp(-1*((x-mu))**2/(2*(sigma**2)))
    return bandshape


#Grab determinants from CASSCF-ORCA output with option PrintWF det
def grab_dets_from_CASSCF_output(file):

    class state_dets():
        def __init__(self, root,energy,mult):
            self.mult = mult
            self.root = root
            self.energy = energy
            self.determinants = {}
            self.configurations = {}
    list_of_states=[]
    detgrab=False
    grabrange=False
    with open(file) as f:
        for line in f:
            #Getting orbital ranges
            # Internal (doubly occ)and external orbitals (empty)
            if grabrange is True:

                if 'Internal' in line:
                    internal=int(line.split()[-2])
                    internal_tuple = tuple([3] * internal)
                if 'Active' in line:
                    active=int(line.split()[-2])
                if 'External' in line:
                    external=int(line.split()[-2])
                    external_tuple = tuple([0] * external)
            if 'Determined orbital ranges:' in line:
                grabrange=True
            if 'Number of rotation parameters' in line:
                grabrange=False

            if 'SA-CASSCF TRANSITION ENERGIES' in line:
                detgrab=False
            if 'DENSITY MATRIX' in line:
                detgrab=False
            if detgrab is True:

                if '[' in line and 'CFG' not in line:
                    det = line.split()[0]
                    #print("det:", det)
                    detlist=[i for i in det.replace('[','').replace(']','')]
                    detlist2=[]
                    #print("detlist:", detlist)
                    #Sticking with labelling: 3: doubly occ, 0: empty, 1 for up-alpha, 2 for down-beta
                    for j in detlist:
                        if j == '2':
                            detlist2.append(3)
                        elif j == '0':
                            detlist2.append(0)
                        elif j == 'u':
                            detlist2.append(1)
                        elif j == 'd':
                            detlist2.append(2)
                    #print("detlist2:", detlist2)
                    #combining
                    det_tuple=internal_tuple+tuple(detlist2)+external_tuple
                    #print("det_tuple : ", det_tuple)
                    coeff = float(line.split()[-1])
                    state.determinants[det_tuple] = coeff
                if '[' in line and 'CFG' in line:
                    cfg = line.split()[0]
                    coeff = float(line.split()[-1])
                    state.configurations[cfg] = coeff

                    #CASE: CFG contains only 2 and 0s. That means a situation where CFG and Det is same thing
                    # But det info is not printed so we need to add it
                    #Removed after Vijay update
                    #if '1' not in cfg:
                    #    #print("cfg : ", cfg)
                    #    print("Found CFG without Det info. Adding to determinants")
                    #    #print("line:", line)
                    #    bla = cfg.replace('[','').replace(']','').replace('CFG','')
                    #    #print("bla:", bla)
                    #    det = bla.replace(str(2),str(3))
                    #    #print("det:", det)
                    #    det2 = [int(i) for i in det]
                    #    det_tuple = internal_tuple + tuple(det2) + external_tuple
                    #    #print("det_tuple: ", det_tuple)
                    #    state.determinants[det_tuple] = coeff

                if 'ROOT ' in line:
                    print("line:", line)
                    root=int(line.split()[1][0])
                    energy = float(line.split()[3])
                    state = state_dets(root, energy, mult)
                    list_of_states.append(state)
            if 'CAS-SCF STATES FOR BLOCK' in line:
                print("CAS LINE: ", line)
                mult =int(line.split()[6])
                print("Setting mult to: ", mult)
                detgrab = False
                print("Det grab set to False")
            if '  Extended CI Printing (values > TPrintWF)' in line:
                print("Det grab set to True")
                detgrab=True
            if '  Spin-Determinant CI Printing' in line:
                print("Det grab set to True")
                detgrab=True

    #print("list_of_states:", list_of_states)
    #print(list_of_states[0])
    #print(list_of_states[0].determinants)
    #print(list_of_states[0].configurations)


    #Going through
    for n,state in enumerate(list_of_states):
        print("------------------------")
        print("This is state {}  with mult {} and energy {} and root {}".format(n,state.mult, state.energy, state.root))
        print("length of state CFGs :", len(state.configurations))
        print("length of state determinants :", len(state.determinants))
        if len(state.determinants) == 0:
            print("WARNING!!! No determinant output found.")
            print("THIS should go away. Disabling for now...")
            exit()
            print("Must be because CFG and det is the same. Using CFG info ")
            print("WARNING!!!")
            print("state.configurations : ", state.configurations)
            for cfg in state.configurations.items():
                bla = cfg[0].replace('[','').replace(']','').replace('CFG','')
                det = bla.replace(str(2),str(3))
                det2 = [int(i) for i in det]
                #det_tuple = tuple(det2)
                det_tuple = internal_tuple + tuple(det2) + external_tuple
                coeff = cfg[1]
                state.determinants[det_tuple] = coeff
            #print("state.determinants: ", state.determinants)

    #print("list_of_states:", list_of_states)

    mults = list(set([state.mult for state in list_of_states]))
    #Return a dictionary with all mults and all states
    final = {}
    for mult in mults:
        final[mult] = [state.determinants for state in list_of_states if state.mult == mult ]
    #print("final :", final)
    return final



#Grab determinants from MRCI-ORCA output with option PrintWF det
def grab_dets_from_MRCI_output(file):


    class state_dets():
        def __init__(self, root,energy,mult):
            self.mult = mult
            self.root = root
            self.energy = energy
            self.determinants = {}
            self.configurations = {}
            self.ciblock = None
    list_of_states=[]
    detgrab=False
    grabrange=False
    dummycount=0
    with open(file) as f:
        for line in f:
            if 'Program Version 4.2.1 -  RELEASE' in line:
                sys.exit('MRCI-determinant read will not work for ORCA 4.2.1 and older!')
            if 'Total number of orbitals            ...' in line:
                totorbitals=int(line.split()[-1])
                #print("totorbitals:", totorbitals)
            #Getting orbital ranges
            # Internal (doubly occ)and external orbitals (empty)
            if grabrange is True:

                if 'Internal' in line:
                    internal=int(line.split()[-2])
                    internal_tuple = tuple([3] * internal)
                if 'Active' in line:
                    active=int(line.split()[-2])
                if 'External' in line:
                    external=int(line.split()[-2])
                    external_tuple = tuple([0] * external)
                    #First index of external list
                    external_first=int(line.split()[1])
                    
            if 'Determined orbital ranges:' in line:
                grabrange=True
            if 'Number of rotation parameters' in line:
                grabrange=False

            if 'SA-CASSCF TRANSITION ENERGIES' in line:
                detgrab=False
            if 'DENSITY MATRIX' in line or 'DENSITY GENERATION' in line:
                #print("here. density line. Setting detgrab to false")
                detgrab=False
            if 'TRANSITION ENERGIES' in line:
                detgrab=False
                #print("here. transitio energies line. Setting detgrab to false")
            #Determining CI BLOCK
            #if 'CI BLOCK' in line:


            #What block we are reading through
            if '          CI-BLOCK' in line:
                #Setting detgrab to False for each new CI-block. Prevents us from grabbing State-lines for Reference-space CI
                detgrab = False
                ciblock = int(line.split()[-2])
                #print("Inside CI Block : ", ciblock)
            if 'Building a CAS' in line:
                #Setting mult here. mult will be used when creating state
                mult = int(line.split()[-1])

            if detgrab is True:

                #Here reading CFG line. Grabbing configuration
                #Also
                if '[' in line and 'CFG' in line:
                    
                    hole_indices=[]
                    particle_indices=[]
                    #print("----------------------------------------------------------------------------------------")
                    #print("line:", line)
                    cfg = line.split()[-1]
                    #This is the weight of this configuration, not CI coefficient
                    weight = float(line.split()[0])
                    #print("weight:", weight)
                    state.configurations[cfg] = weight
                    #Reading CFG line and determining hole/particle excitations outside active space
                    if 'h---h---' in line and line.count('p')==0:
                        #CAS excitation
                        #print("Assignment: 0 HOLE  0 PARTICLE")
                        hole_indices=[]
                        particle_indices=[]
                        #print("hole_indices:", hole_indices)                         
                        #print("particle_indices:", particle_indices)    
                    elif 'h---h---' in line and line.count('p')==1:
                        #0-hole 1-particle
                        hole_indices=[]
                        #print("Assignment: 0 HOLE  1 PARTICLE")
                        particle_index = int(find_between(line,']p','\n'))
                        particle_indices.append(particle_index)
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)    
                    elif 'h---h---' in line and line.count('p')==2:
                        #0-hole 2-particle
                        hole_indices=[]
                        #print("Assignment: 0 HOLE  2 PARTICLE")
                        particle_indices = [int(i) for i in find_between(line,']p','\n').replace('p','').split()]
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)                
                    elif 'h---h ' in line and line.count('p')==0:
                        #1-hole 0-particle
                        particle_indices=[]
                        #print("Assignment: 1 HOLE  0 PARTICLE")                             
                        hole_index=int(find_between(line,'h---h','[').strip())
                        hole_indices.append(hole_index)
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)  
                    elif 'h---h ' in line and line.count('p')==1:
                        #1-hole 1-particle
                        #print("Assignment: 1 HOLE  1 PARTICLE")
                        hole_index=int(find_between(line,'h---h','[').strip())
                        hole_indices.append(hole_index)
                        particle_index = int(find_between(line,']p','\n'))
                        particle_indices.append(particle_index)                         
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)  
                    elif 'h---h ' in line and line.count('p')==2:
                        #1-hole 2-particle
                        #print("Assignment: 1 HOLE  2 PARTICLE")
                        hole_index=int(find_between(line,'h---h','[').strip())
                        hole_indices.append(hole_index)
                        particle_indices = [int(i) for i in find_between(line,']p','\n').replace('p','').split()]
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)
                    elif 'CFG h ' in line and line.count('p')==0:
                        # 2-hole 0-particle
                        #print("Assignment: 2 HOLE  0 PARTICLE")
                        hole_indices=[int(i) for i in find_between(line,'CFG h','[').replace('h','').split()]
                        particle_indices=[]
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)
                    elif 'CFG h ' in line and line.count('p')==1:
                        # 2-hole 1-particle
                        #print("Assignment: 2 HOLE  1 PARTICLE")
                        hole_indices=[int(i) for i in find_between(line,'CFG h','[').replace('h','').split()]
                        particle_index = int(find_between(line,']p','\n'))
                        particle_indices.append(particle_index)   
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)
                    elif 'CFG h ' in line and line.count('p')==2:
                        # 2-hole 2-particle
                        #print("Assignment: 2 HOLE  2 PARTICLE")
                        hole_indices=[int(i) for i in find_between(line,'CFG h','[').replace('h','').split()]
                        particle_indices = [int(i) for i in find_between(line,']p','\n').replace('p','').split()]
                        #print("hole_indices:", hole_indices) 
                        #print("particle_indices:", particle_indices)                                                    
                    else:
                        print("Bad line. exiting")
                        exit()    
                        
                if '[' in line and 'CFG' not in line:
                    dummycount+=1
                    #print("dummycount:", dummycount)
                    #print("Determinant line:", line)
                    det = line.split()[0]
                    #print("det:", det)
                    detlist=[i for i in det.replace('[','').replace(']','')]
                    detlist2=[]
                    #print("detlist:", detlist)
                    #Sticking with labelling: 3: doubly occ, 0: empty, 1 for up-alpha, 2 for down-beta
                    for j in detlist:
                        if j == '2':
                            detlist2.append(3)
                        elif j == '0':
                            detlist2.append(0)
                        elif j == 'u':
                            detlist2.append(1)
                        elif j == 'd':
                            detlist2.append(2)
                    #print("detlist2:", detlist2)
                    
                    
                    
                    #Modifying internal_tuple for possible holes
                    #print("internal_tuple :", internal_tuple)
                    #CASE: 1 HOLES  0 PARTICLES:
                    if len(hole_indices) == 1 and len(particle_indices) == 0:
                        holeindex=hole_indices[0]
                        #print("Modifying internal_tuple")                        
                        lst_internaltuple=list(internal_tuple)
                        #Getting spinlabel of internal electron where hole was created from detlist (first index in bracket)
                        spinlabelh1p0int=detlist2[0]
                        lst_internaltuple[holeindex] = spinlabelh1p0int
                        modinternal_tuple=tuple(lst_internaltuple)
                        #print("Mod internal_tuple :", modinternal_tuple)
                        #Removing hole orb from detlist
                        moddetlist2=detlist2[1:]
                        #print("Mod active tuple :", detlist2)
                        #Unmodified external
                        modexternal_tuple=external_tuple
                    #CASE: 2 HOLES  0 PARTICLES:
                    elif len(hole_indices) == 2 and len(particle_indices) == 0:
                        moddetlist2=detlist2
                        holeindex1=hole_indices[0]
                        holeindex2=hole_indices[1]
                        
                        #SubCase: Double internal hole. Means no spin-label in bracket for this
                        if holeindex1 == holeindex2:
                            #print("Same holeindex")
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = 0
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #No modification to detlist2 needed
                        #Subcase: Not doubly internal hole.
                        else:
                            #print("Not same holeindex")
                            #print("moddetlist2:", moddetlist2)
                            spinlabelh2p0int_1=detlist2[0]
                            spinlabelh2p0int_2=detlist2[1]
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = spinlabelh2p0int_1
                            lst_internaltuple[holeindex2] = spinlabelh2p0int_2
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #Modification to detlist
                            moddetlist2=detlist2[2:]
                        #Unmodified external
                        modexternal_tuple=external_tuple
                    #CASE: 0 HOLE  1 PARTICLE:
                    elif len(hole_indices) == 0 and len(particle_indices) == 1:
                        particleindex1=particle_indices[0]
                        #Particleposition in external list
                        particleposition=particleindex1-external_first
                        #print("particleposition in external list:", particleposition)
                        #print("Modifying external")
                        spinlabelh0p1ext=detlist2[-1]
                        lst_externaltuple=list(external_tuple)
                        lst_externaltuple[particleposition] = spinlabelh0p1ext
                        modexternal_tuple=tuple(lst_externaltuple)
                        #print("Mod external tuple :", modexternal_tuple)
                        #Removed particle spin from detlist
                        moddetlist2=detlist2[:-1]
                        
                        #Unmodified internal
                        modinternal_tuple=internal_tuple
                    #CASE: 1 HOLE  1 PARTICLE:
                    elif len(hole_indices) == 1 and len(particle_indices) == 1:
                        holeindex=hole_indices[0]
                        particleindex1=particle_indices[0]
                        #Particleposition in external list
                        particleposition=particleindex1-external_first
                        
                        #Modifying internal list
                        spinlabelh1p1int=detlist2[0]
                        lst_internaltuple=list(internal_tuple)
                        lst_internaltuple[holeindex] = spinlabelh1p1int
                        modinternal_tuple=tuple(lst_internaltuple)
                        #print("Mod internal_tuple :", modinternal_tuple)
                        #Modifying external list
                        spinlabelh1p1ext=detlist2[-1]
                        lst_externaltuple=list(external_tuple)
                        lst_externaltuple[particleposition] = spinlabelh1p1ext
                        modexternal_tuple=tuple(lst_externaltuple)
                        #print("Mod external tuple :", modexternal_tuple)
                        #Modifying detlist
                        moddetlist2=detlist2[1:-1]
                    #CASE: 0 HOLE  2 PARTICLES:                        
                    elif len(hole_indices) == 0 and len(particle_indices) == 2:
                        particleindex1=particle_indices[0]
                        particleindex2=particle_indices[1]
                        #Particleposition in external list
                        particleposition1=particleindex1-external_first
                        particleposition2=particleindex2-external_first                        
                        #Modifying external list
                        #SubCase: Double particle position. Means no spin-label in bracket for this
                        if particleindex1 == particleindex2:
                            #print("Particle indices the same")
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = 3
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            
                            #Modifying active detlist
                            moddetlist2=detlist2
                            
                        else:
                            #print("Particle indices NOT the same")
                            spinlabelh0p2_1ext=detlist2[-2]
                            spinlabelh0p2_2ext=detlist2[-1]
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = spinlabelh0p2_1ext
                            lst_externaltuple[particleposition2] = spinlabelh0p2_2ext
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            #Modifying active detlist
                            moddetlist2=detlist2[:-2]                        
                        
                        #Unmodified internal
                        modinternal_tuple=internal_tuple
                        
                        
                    #CASE: 1 HOLE  2 PARTICLES:                        
                    elif len(hole_indices) == 1 and len(particle_indices) == 2:
                        #Grabbing immediately
                        moddetlist2=detlist2
                        holeindex=hole_indices[0]
                        particleindex1=particle_indices[0]
                        particleindex2=particle_indices[1]
                        #Particleposition in external list
                        particleposition1=particleindex1-external_first
                        particleposition2=particleindex2-external_first
                        
                        #Modifying internal list
                        spinlabelh1p2_1int=detlist2[0]
                        lst_internaltuple=list(internal_tuple)
                        lst_internaltuple[holeindex] = spinlabelh1p2_1int
                        modinternal_tuple=tuple(lst_internaltuple)
                        #print("Mod internal_tuple :", modinternal_tuple)
                        #Modifying active detlist for hole
                        moddetlist2=moddetlist2[1:]
                        
                        #Modifying external list
                        #SubCase: Double particle position. Means no spin-label in bracket for this
                        if particleindex1 == particleindex2:
                            #print("Particle indices the same")
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = 3
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            

                            
                        else:
                            #print("Particle indices NOT the same")
                            spinlabelh1p2_1ext=detlist2[-2]
                            spinlabelh1p2_2ext=detlist2[-1]
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = spinlabelh1p2_1ext
                            lst_externaltuple[particleposition2] = spinlabelh1p2_2ext
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            #Modifying active detlist
                            moddetlist2=moddetlist2[:-2]
                        

                    #CASE: 2 HOLES  2 PARTICLES:                       
                    elif len(hole_indices) == 2 and len(particle_indices) == 2:
                        #Grab this immediately
                        moddetlist2=detlist2
                        holeindex1=hole_indices[0]
                        holeindex2=hole_indices[1]
                        particleindex1=particle_indices[0]
                        particleindex2=particle_indices[1]
                        #Particleposition in external list
                        particleposition1=particleindex1-external_first
                        particleposition2=particleindex2-external_first
                        
                        #Modifying internal list
                        #SubCase: Double internal hole. Means no spin-label in bracket for this
                        if holeindex1 == holeindex2:
                            #print("Same holeindex")
                            spinlabelh2p2int=0
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = spinlabelh2p2int
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #No modification to detlist2 needed
                        #Subcase: Not doubly internal hole.
                        else:
                            #print("Not same holeindex")
                            spinlabelh2p2_1int=detlist2[0]
                            spinlabelh2p2_2int=detlist2[1]
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = spinlabelh2p2_1int
                            lst_internaltuple[holeindex2] = spinlabelh2p2_2int
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #Modification to detlist
                            moddetlist2=moddetlist2[2:]
                        
                        #Modifying external list
                        #SubCase: Double particle position. Means no spin-label in bracket for this
                        if particleindex1 == particleindex2:
                            #print("Particle indices the same")
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = 3
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            #No modification to detlist2 needed
                        else:
                            #print("Particle indices NOT the same")
                            spinlabelh2p2_1ext=detlist2[-2]
                            spinlabelh2p2_2ext=detlist2[-1]
                            lst_externaltuple=list(external_tuple)
                            lst_externaltuple[particleposition1] = spinlabelh2p2_1ext
                            lst_externaltuple[particleposition2] = spinlabelh2p2_2ext
                            modexternal_tuple=tuple(lst_externaltuple)
                            #print("Mod external tuple :", modexternal_tuple)
                            #Modifying active detlist
                            moddetlist2=moddetlist2[:-2]                        
                        
                        
                    #CASE: 2 HOLE  1 PARTICLE:   
                    elif len(hole_indices) == 2 and len(particle_indices) == 1:
                        moddetlist2=detlist2
                        holeindex1=hole_indices[0]
                        holeindex2=hole_indices[1]
                        particleindex1=particle_indices[0]
                        #Particleposition in external list
                        particleposition1=particleindex1-external_first
                        
                        #Modifying internal list
                        #SubCase: Double internal hole. Means no spin-label in bracket for this
                        if holeindex1 == holeindex2:
                            #print("Same holeindex")
                            spinlabelh2p1int=0
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = spinlabelh2p1int
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #No modification to detlist2 needed
                        #Subcase: Not doubly internal hole.
                        else:
                            #print("Not same holeindex")
                            spinlabelh2p1_1int=detlist2[0]
                            spinlabelh2p1_2int=detlist2[1]
                            lst_internaltuple=list(internal_tuple)
                            lst_internaltuple[holeindex1] = spinlabelh2p1_1int
                            lst_internaltuple[holeindex2] = spinlabelh2p1_2int
                            #print("lst_internaltuple:", lst_internaltuple)
                            modinternal_tuple=tuple(lst_internaltuple)
                            #print("Mod internal_tuple :", modinternal_tuple)
                            #Modification to detlist
                            moddetlist2=moddetlist2[2:]

                        #Modifying external
                        lst_externaltuple=list(external_tuple)
                        #print("external_tuple:", external_tuple)
                        spinlabelh2p1_1ext=detlist2[-1]
                        lst_externaltuple[particleposition1] = spinlabelh2p1_1ext
                        modexternal_tuple=tuple(lst_externaltuple)
                        #print("Mod external tuple :", modexternal_tuple)
                        
                        #Modifying detlist
                        moddetlist2=moddetlist2[:-1]                        
                        

                    #CASE: NO HOLES, NO PARTICLES                     
                    else:
                        modinternal_tuple=internal_tuple
                        modexternal_tuple=external_tuple
                        moddetlist2=detlist2
                        
                    #combining
                    det_tuple=modinternal_tuple+tuple(moddetlist2)+modexternal_tuple
                    #print("det_tuple ({}): {}".format(len(det_tuple),det_tuple))
                    
                    assert len(det_tuple) == totorbitals, "Orbital tuple ({}) not matching total number of orbitals ({})".format(len(det_tuple),totorbitals)
                    #if len(det_tuple) == 22:
                    #    print("problem")
                    #    exit()
                    #if len(det_tuple) != totorbitals:
                    #    print("XXXXXXXXX")
                    
                    #This is the CI coeffient
                    coeff = float(line.split()[-1])
                    #print("coeff : ", coeff)
                    state.determinants[det_tuple] = coeff
                    #print("state.determinants :", state.determinants)
                    #if dummycount == 7416:
                    #    exit()


                    #CASE: CFG contains only 2 and 0s. That means a situation where CFG and Det is same thing
                    # But det info is not printed so we need to add it
                    #DISABLING after Vijay update
                    #if '1' not in cfg:
                    #    print("cfg : ", cfg)
                    #    print("Found CFG without Det info. Adding to determinants")
                    #    print("line:", line)
                    #    bla = cfg.replace('[','').replace(']','').replace('CFG','')
                    #    print("bla:", bla)
                    #    det = bla.replace(str(2),str(3))
                    #    print("det:", det)
                    #    det2 = [int(i) for i in det]
                    #    det_tuple = internal_tuple + tuple(det2) + external_tuple
                    #    #print("det_tuple: ", det_tuple)
                    #    state.determinants[det_tuple] = coeff

                #Now creating state. Taking energy, root and mult (found earlier in beginning of CI block).
                if 'STATE' in line:
                    #print("STATE in line. Creating state")
                    #print("line:", line)
                    root=int(line.split()[1].replace(':',''))
                    #print("root:", root)
                    energy = float(line.split()[3])
                    state = state_dets(root,energy,mult)
                    list_of_states.append(state)
            #if 'CAS-SCF STATES FOR BLOCK' in line:
            #    mult =int(line.split()[6])
            #Now PT2-selection and CI-problem is solved. Final states coming next.
            if 'Davidson type correction:' in line:
                detgrab=True

    #print("list_of_states:", list_of_states)
    #print(list_of_states[0])
    #print(list_of_states[0].determinants)
    #print(list_of_states[0].configurations)


    #Going through
    #print("list_of_states[0].__dict__", list_of_states[0].__dict__)
    #for n,state in enumerate(list_of_states):
    #    print("------------------------")
    #    print("This is state {}  with mult {} and energy {} and root {}".format(n,state.mult, state.energy, state.root))
    #    print("length of state CFGs :", len(state.configurations))
    #    print("length of state determinants :", len(state.determinants))
    #    print("state.configurations : ", state.configurations)
    #    print("state.determinants : ", state.determinants)

    #print("list_of_states:", list_of_states)

    mults = list(set([state.mult for state in list_of_states]))
    #Return a dictionary with all mults and all states
    final = {}
    for mult in mults:
        final[mult] = [state.determinants for state in list_of_states if state.mult == mult ]
    #print("final :", final)
    return final



########################
# MAIN program
########################

# Calculate PES spectra using the Dyson orbital approach.
# Memory for WFoverlap in MB. Hardcoded
def PhotoElectronSpectrum(theory=None, fragment=None, InitialState_charge=None, Initialstate_mult=None,
                          Ionizedstate_charge=None, Ionizedstate_mult=None, numionstates=None, path_wfoverlap=None, tda=True,
                          brokensym=False, HSmult=None, atomstoflip=None, initialorbitalfiles=None, Densities='SCF', densgridvalue=100,
                          CAS=False, CAS_Initial=None, CAS_Final = None, memory=40000, numcores=1, noDyson=False, CASCI=False, MRCI=False,
                          MRCI_Initial=None, MRCI_Final = None, tprintwfvalue=1e-16, MRCI_CASCI_Final=True, EOM=False):
    blankline()
    print(bcolors.OKGREEN,"-------------------------------------------------------------------",bcolors.ENDC)
    print(bcolors.OKGREEN,"PhotoElectronSpectrum: Calculating PES spectra via TDDFT/CAS/MRCI/EOM and Dyson-norm approach",bcolors.ENDC)
    print(bcolors.OKGREEN,"-------------------------------------------------------------------",bcolors.ENDC)
    blankline()
    print("Numcores used for WFoverlap: ", numcores)
    os.environ["OMP_NUM_THREADS"] = str(numcores)
    print("OMP_NUM_THREADS : ", os.environ["OMP_NUM_THREADS"])

    #Numionstates can be number or list of numbers (states for each multiplicity for CAS/MRCI)
    if isint(numionstates):
        numionstates_A=numionstates
        numionstates_B=numionstates
    elif islist(numionstates):
        #A,B: first and second multiplicity in Ionizedstate_mult
        numionstates_A=numionstates[0]
        numionstates_B=numionstates[1]
        numionstates=numionstates_A+numionstates_B

    if EOM is True:
        print("EOM is True. Will do EOM-IP-CCSD calculations to calculate IPs directly.")


    if CAS is True and MRCI is True:
        print("Both CAS and MRCI can not both be True!")
        print("You must previously optimize orbitals (e.g. with CASSCF) and feed into MRCI")
        exit(1)

    if CAS is True:
        print("CASSCF option active!")
        if CASCI is True:
            print("CASCI option on! Initial state will be done with CASSCF while Final ionized states will do CAS-CI")

    if MRCI is True:
        print("MRCI option active!")
        print("Will do CASSCF orbital optimization for initial-state, followed by MRCI")
        if MRCI_CASCI_Final is True:
            print("Will do CAS-CI reference (using initial-state orbitals) for final-states")


    if InitialState_charge is None or Initialstate_mult is None or Ionizedstate_charge is None or Ionizedstate_mult is None:
        print("Provide charge and spin multiplicity of initial and ionized state: InitialState_charge, InitialState_mult, Ionizedstate_charge,Ionizedstate_mult ")
        exit(1)

    print("Densities option is: ", Densities, "(options are: SCF, All, None)")
    if Densities == 'SCF':
        print("Will do densities (and difference densities) for Inital-state and Final-state SCF wavefunctions only.")
        shutil.rmtree('Calculated_densities', ignore_errors=True)
        os.mkdir('Calculated_densities')
    elif Densities=='All':
        print("Will do densities (and difference densities) for all states: SCF and TDDFT states")
        shutil.rmtree('Calculated_densities', ignore_errors=True)
        os.mkdir('Calculated_densities')
    else:
        Densities=None
        print("Will not calculate densities")

    #Getting charge/mult of states from function argument
    totnuccharge=fragment.nuccharge
    fragment.print_coords()
    blankline()

    # new class for state (Initial, Final etc.) that may differ in charge or spin
    #Will contain energies, MOs, transition energies, IPs etc.
    class MolState:
        def __init__(self,charge,mult,numionstates):

            self.charge=charge
            self.mult=mult
            self.tddftstates=[]
            self.dysonnorms=[]
            self.energy=0.0
            self.occorbs_alpha = []
            self.occorbs_beta= []
            self.hftyp = None
            self.TDtransitionenergies=[]
            self.restricted=None
            self.GSIP=None
            self.IPs=[]
            #Energy
            self.ionstates=[]
            #Number of calculated states for each mult
            self.numionstates=numionstates
            self.gbwfile=None
            self.outfile=None
            self.cisfile=None

    # Always just one StateI object with one charge and one spin multiplicity
    stateI = MolState(charge=InitialState_charge, mult=Initialstate_mult,numionstates=1)
    print(bcolors.OKBLUE, "StateI: Charge=", stateI.charge, "Multiplicity", stateI.mult, bcolors.ENDC)

    if brokensym is True:
        print("Brokensym True. Will find BS-solution for StateI via spin-flipping. HSMult: ", HSmult)


    if type(Ionizedstate_mult) is int:
        #Only one mult for ionized-state. Using numionstates
        stateF1 = MolState(charge=Ionizedstate_charge, mult=Ionizedstate_mult,numionstates=numionstates)
        MultipleSpinStates = False
        Finalstates=[stateF1]
        print(bcolors.OKBLUE, "StateF_1: Charge=", Finalstates[0].charge, "Multiplicity", Finalstates[0].mult,
          bcolors.ENDC)
        print(bcolors.OKBLUE, "StateF_1: Numionstates=", Finalstates[0].numionstates, bcolors.ENDC)        
    #Case list provided for ionized state. Could mean multiple spin states: e.g.  Ionizedstate_mult=[5,7]
    elif type(Ionizedstate_mult) is list:

        if len(Ionizedstate_mult) == 1:
            MultipleSpinStates = False
            #Only one mult for ionized-state. Using numionstates
            stateF1 = MolState(charge=Ionizedstate_charge, mult=Ionizedstate_mult[0],numionstates=numionstates)
            Finalstates = [stateF1]
            print(bcolors.OKBLUE, "StateF_1: Charge=", Finalstates[0].charge, "Multiplicity", Finalstates[0].mult,
                  bcolors.ENDC)
            print(bcolors.OKBLUE, "StateF_1: Numionstates=", numionstates_A, bcolors.ENDC)
        elif len(Ionizedstate_mult) == 2:
            MultipleSpinStates = True
            stateF1 = MolState(charge=Ionizedstate_charge, mult=Ionizedstate_mult[0],numionstates=numionstates_A)
            stateF2 = MolState(charge=Ionizedstate_charge, mult=Ionizedstate_mult[1],numionstates=numionstates_B)
            Finalstates = [stateF1,stateF2]
            print("Multiple spin states for Final State:")
            print(bcolors.OKBLUE, "StateF_1: Charge=", Finalstates[0].charge, "Multiplicity", Finalstates[0].mult,
                  bcolors.ENDC)
            print(bcolors.OKBLUE, "StateF_2: Charge=", Finalstates[1].charge, "Multiplicity", Finalstates[1].mult,
                  bcolors.ENDC)
            print(bcolors.OKBLUE, "StateF_1: Numionstates=", Finalstates[0].numionstates, bcolors.ENDC)
            print(bcolors.OKBLUE, "StateF_2: Numionstates=", Finalstates[1].numionstates, bcolors.ENDC)
        else:
            print("More than Two spin multiplicities are now allowed in Ionizedstate_mult argument")
            exit(1)

    else:
        print("Unknown type for Ionizedstate_mult value. Should be integer or list of integers")



    print("")
    print("CAS:", CAS)
    print("MRCI:", MRCI)
    print("EOM:", EOM)
    if CAS is False and MRCI is False and EOM is False:
        print("TDDFT option chosen:")
        print(bcolors.OKBLUE,"Total ion states:", numionstates, bcolors.ENDC)
        print(bcolors.OKBLUE,"TDDFT-calculated ion states:", numionstates-1, bcolors.ENDC)



    #ORCA-theory
    if theory.__class__.__name__ == "ORCATheory":
        #########################
        #INITIAL STATE
        ########################
        theory.charge=stateI.charge
        theory.mult=stateI.mult
        theory.extraline=theory.extraline+"%method\n"+"frozencore FC_NONE\n"+"end\n"

        if CAS is True:
            print("Using TprintWF value of ", tprintwfvalue)
            print("Modifying CASSCF block for initial state, CAS({},{})".format(CAS_Initial[0],CAS_Initial[1]))
            print("{} electrons in {} orbitals".format(CAS_Initial[0],CAS_Initial[1]))

            #Removing nel/norb/nroots lines if user added
            for line in theory.orcablocks.split('\n'):
                if 'nel' in line:
                    theory.orcablocks=theory.orcablocks.replace(line,'')
                if 'norb' in line:
                    theory.orcablocks=theory.orcablocks.replace(line,'')
                if 'nroots' in line:
                    theory.orcablocks=theory.orcablocks.replace(line,'')
            theory.orcablocks = theory.orcablocks.replace('\n\n','\n')

            #Add nel,norb and nroots lines back in. Also determinant printing option
            theory.orcablocks = theory.orcablocks.replace('%casscf', '%casscf\n' + "printwf det\nci TPrintwf {} end\n".format(tprintwfvalue) + "nel {}\n".format(CAS_Initial[0]) +
                                                          "norb {}\n".format(
                                                              CAS_Initial[1]) + "nroots {}\n".format(1))
            theory.orcablocks = theory.orcablocks.replace('\n\n','\n')
            theory.orcablocks = theory.orcablocks.replace('\n\n','\n')
        if MRCI is True:
            print("Using TprintWF value of ", tprintwfvalue)
            print("Modifying MRCI block for initial state, CAS({},{})".format(MRCI_Initial[0],MRCI_Initial[1]))
            print("{} electrons in {} orbitals".format(MRCI_Initial[0],MRCI_Initial[1]))
            print("WARNING: MRCI determinant-printing read will only work for ORCA-current or ORCA 5.0, not older ORCA versions like ORCA 4.2")


            #USING CASSCF block to define reference
            #Add nel,norb and nroots lines back in. Also determinant printing option
            print("theory.orcablocks :", theory.orcablocks)
            
            #If CASSCF block present, trim and replace
            if '%casscf' in theory.orcablocks:
                            #Removing nel/norb/nroots lines if user added
                for line in theory.orcablocks.split('\n'):
                    if 'nel' in line:
                        theory.orcablocks=theory.orcablocks.replace(line,'')
                    if 'norb' in line:
                        theory.orcablocks=theory.orcablocks.replace(line,'')
                    if 'nroots' in line:
                        theory.orcablocks=theory.orcablocks.replace(line,'')
                    if 'maxiter' in line:
                        theory.orcablocks=theory.orcablocks.replace(line,'')
                theory.orcablocks = theory.orcablocks.replace('\n\n','\n')    
                theory.orcablocks = theory.orcablocks.replace('%casscf', '%casscf\n'  + "nel {}\n".format(MRCI_Initial[0]) +
                                                          "norb {}\n".format(MRCI_Initial[1]) + "nroots {}\n".format(1))
            else:
                 theory.orcablocks= theory.orcablocks + '%casscf\n'  + "nel {}\n".format(MRCI_Initial[0]) + "norb {}\n".format(MRCI_Initial[1]) + "nroots {}\nend\n".format(1)
            print("theory.orcablocks :", theory.orcablocks)
            #Enforcing CAS-CI
            #if 'noiter' not in theory.orcasimpleinput.lower():
            #    theory.orcasimpleinput = theory.orcasimpleinput + ' noiter '

            #Defining simple MRCI block. States defined
            theory.orcablocks = theory.orcablocks + "%mrci\n" + "printwf det\nTPrintwf {}\n".format(tprintwfvalue) + "end"
            #theory.orcablocks = "%mrci\n" + "printwf det\nTPrintwf 1e-16\n" + "newblock {} *\n refs cas({},{}) end\n".format(stateI.mult,MRCI_Initial[0],MRCI_Initial[1])+ "nroots {}\n end\n".format(1) + "end"
            theory.orcablocks = theory.orcablocks.replace('\n\n','\n')
            theory.orcablocks = theory.orcablocks.replace('\n\n','\n')

            #Adding MRCI+Q to simpleinputline
            #TODO: Remove as we may want another MRCI method
            if 'MRCI+Q' not in theory.orcasimpleinput:
                theory.orcasimpleinput = theory.orcasimpleinput + ' MRCI+Q' 

        # For orbital analysis
        if 'NORMALPRINT' not in theory.orcasimpleinput.upper():
            theory.orcasimpleinput = theory.orcasimpleinput + ' Normalprint'
            
        if brokensym is True:
            theory.brokensym=True
            theory.HSmult=HSmult
            theory.atomstoflip=atomstoflip
            #Making sure UKS always present if brokensym feature active. Important for open-shell singlets
            if 'UKS' not in theory.orcasimpleinput.upper():
                theory.orcasimpleinput = theory.orcasimpleinput + ' UKS'


        if initialorbitalfiles is not None:
            print("initialorbitalfiles keyword provided.")
            print("Will use file {} as guess GBW file for Initial state".format(initialorbitalfiles[0]))
            shutil.copyfile(initialorbitalfiles[0], theory.inputfilename + '.gbw')

        if EOM is not True:
            print(bcolors.OKGREEN, "Calculating Initial State SCF.",bcolors.ENDC)
            finalsinglepointenergy = Singlepoint(fragment=fragment, theory=theory)

        #Create Cube file of electron/spin density using orca_plot for INITIAL STATE
        if Densities == 'SCF' or Densities =='All':
            os.chdir('Calculated_densities')
            print("Density option active. Calling orca_plot to create Cube-file for Initial state SCF.")
            shutil.copyfile('../' + theory.inputfilename + '.gbw', './'+theory.inputfilename + '.gbw')
            #Electron density
            run_orca_plot(orcadir=theory.orcadir,filename=theory.inputfilename + '.gbw', option='density', gridvalue=densgridvalue)
            os.rename(theory.inputfilename+'.scfp','Init_State.scfp')
            shutil.copyfile(theory.inputfilename + '.eldens.cube', './' + 'Init_State' + '.eldens.cube')

            # Read Initial-state-SCF density Cube file into memory
            rlowx, dx, nx, orgx, rlowy, dy, ny, orgy, rlowz, dz, nz, \
            orgz, elems, molcoords, molcoords_ang, numatoms, filebase, initial_values = read_cube('Init_State.eldens.cube')

            #Spin density (only if UHF). Otherwise orca_plot gets confused (takes difference between alpha-density and nothing)
            if stateI.hftyp == "UHF":
                run_orca_plot(orcadir=theory.orcadir,filename=theory.inputfilename + '.gbw', option='spindensity', gridvalue=densgridvalue)
                os.rename(theory.inputfilename + '.scfr', 'Init_State.scfr')
                shutil.copyfile(theory.inputfilename + '.spindens.cube', './' + 'Init_State' + '.spindens.cube')
            os.chdir('..')
        #Note: Using SCF energy and not Final Single Point energy (does not work for TDDFT)
        if CAS is True:
            print("here")
            stateI.energy=casscfenergygrab("orca-input.out")
            print("stateI.energy: ", stateI.energy)

            #Get orbital ranges (stateI is sufficient)
            internal_orbs,active_orbs,external_orbs = casscf_orbitalranges_grab("orca-input.out")
        elif MRCI is True:
            stateI.energy=finalsinglepointenergy
            print("stateI.energy: ", stateI.energy)

            #Get orbital ranges (stateI is sufficient)
            internal_orbs,active_orbs,external_orbs = casscf_orbitalranges_grab("orca-input.out")
        elif EOM is True:
            #No separate initial-state calc when doing EOM
            pass
        else:
            stateI.energy=scfenergygrab("orca-input.out")


        #Saveing GBW/out/in files
        if EOM is not True:
            shutil.copyfile(theory.inputfilename + '.gbw', './' + 'Init_State' + '.gbw')
            shutil.copyfile(theory.inputfilename + '.out', './' + 'Init_State' + '.out')
            shutil.copyfile(theory.inputfilename + '.inp', './' + 'Init_State' + '.inp')

            stateI.gbwfile="Init_State"+".gbw"
            stateI.outfile="Init_State"+".out"

        # Initial state orbitals for MO-DOSplot
        if CAS is True or MRCI is True:
            stateI.hftyp='CASSCF'

            #CASSCF wavefunction interpreted as restricted
            stateI.restricted = True
            for fstate in Finalstates:
                fstate.restricted = True

        elif EOM is True:
            pass
        else:
            stateI.occorbs_alpha, stateI.occorbs_beta, stateI.hftyp = orbitalgrab(theory.inputfilename+'.out')
            print("stateI.occorbs_alpha:", stateI.occorbs_alpha)
            print("stateI.hftyp:", stateI.hftyp)

            # need to specify whether Initial/Final states are restricted or not.
            if stateI.hftyp == "UHF":
                stateI.restricted = False
            elif stateI.hftyp == "RHF":
                stateI.restricted = True
            else:
                print("hmmm")
                exit()


        #########################
        #FINAL STATES
        ########################
        #Final-state  calc. TDDFT or CASSCF
        #Adding TDDFT block to inputfile
        ##CAS option: State-averaged calculation for both spin multiplicities.
        if EOM is True:
            
            #Preserve old
            orig_orcablocks=copy.copy(theory.orcablocks)
            
            #Will calculate IPs directly
            print("Adding MDCI block for initial state")
            
            if 'IP-EOM-CCSD' not in theory.orcasimpleinput:
                theory.orcasimpleinput =  theory.orcasimpleinput + ' IP-EOM-CCSD '
            
            FinalIPs=[]
            fstates_dict={}
            for fstate in Finalstates:
                print(bcolors.OKGREEN, "Calculating Final State (mult {}) via IP-EOM-CCSD: ".format(fstate.mult), bcolors.ENDC)
                if fstate.mult > stateI.mult:
                    print("Final state mult {}, setting DoBeta true".format(fstate.mult))
                    Electron_ion_line='DoBeta true'
                else:
                    print("Final state mult {}, setting DoAlpha true".format(fstate.mult))
                    Electron_ion_line='DoAlpha true'
                #Add nel,norb and nroots lines back in. Also determinant printing option
                theory.orcablocks = orig_orcablocks + '\n%mdci\n' + 'nroots {}\n'.format(fstate.numionstates) + Electron_ion_line+'\n' + 'maxiter 200\n'+'end\n'
                theory.orcablocks = theory.orcablocks.replace('\n\n','\n')
                theory.orcablocks = theory.orcablocks.replace('\n\n','\n')        
            

                if initialorbitalfiles is not None:
                    print("not tested for IP-EOM-CCSD...")
                    print("initialorbitalfiles keyword provided.")
                    print("Will use file {} as guess GBW file for this Final state.".format(initialorbitalfiles[findex + 1]))
                    shutil.copyfile(initialorbitalfiles[findex + 1], theory.inputfilename + '.gbw')

                energy = Singlepoint(fragment=fragment, theory=theory)
                stateI.energy= energy


                #Grab EOM-IPs and dominant singles amplitudes
                IPs, amplitudes = grabEOMIPs("orca-input.out")
                print("IPs:", IPs)
                print("Dominant singles EOM amplitudes:", amplitudes)
                
                fstate.IPs=IPs
                fstate.dysonnorms=amplitudes
                
                #Collecting list of all IPs
                FinalIPs=FinalIPs+IPs
                
                
                #State_energies are Inititial-state energy + transition energy
                state_energies=[IP/constants.hartoeV + stateI.energy for IP in IPs]
                
                #Equating the dominant singles amplitudes with dysonnorms.
                
                fstates_dict[fstate.mult] = state_energies
                print("fstates_dict:", fstates_dict)
                
                fstate.ionstates = fstates_dict[fstate.mult]


                # Saveing GBW and CIS file
                shutil.copyfile(theory.inputfilename + '.gbw', './' + 'Final_State' + '.gbw')
                shutil.copyfile(theory.inputfilename + '.out', './' + 'Final_State' + '.out')
                shutil.copyfile(theory.inputfilename + '.inp', './' + 'Final_State' + '.inp')

                #Each fstate linked with same GBW file and outfile
                fstate.gbwfile = "Final_State" + ".gbw"
                fstate.outfile = "Final_State" + ".out"
                print("")
        
        
        elif CAS is True:
            print("Modifying CASSCF block for final state, CAS({},{})".format(CAS_Final[0],CAS_Final[1]))
            print("{} electrons in {} orbitals".format(CAS_Final[0],CAS_Final[0]))
            #Making sure multiplicties are sorted in ascending order and creating comma-sep string
            CAS_mults=','.join(str(x) for x in sorted([f.mult for f in Finalstates]))
            print("CAS_mults:", CAS_mults)
            #Removing nel/norb/nroots lines
            for line in theory.orcablocks.split('\n'):
                if 'nel' in line:
                    theory.orcablocks=theory.orcablocks.replace(line,'')
                if 'norb' in line:
                    theory.orcablocks=theory.orcablocks.replace(line,'')
                if 'nroots' in line:
                    theory.orcablocks=theory.orcablocks.replace(line,'')
                if 'mult' in line:
                    theory.orcablocks=theory.orcablocks.replace(line,'')
            theory.orcablocks = theory.orcablocks.replace('\n\n','\n')

            #Add nel,norb and nroots lines back in.
            # And both spin multiplicities. Nroots for each
            #numionstates_string = ','.join(str(numionstates) for x in [f.numionstates for f in Finalstates])
            numionstates_string = ','.join(str(f.numionstates) for f in Finalstates)
            theory.orcablocks = theory.orcablocks.replace('%casscf', '%casscf\n' + "nel {}\n".format(CAS_Final[0]) +
                                                          "norb {}\n".format(
                                                              CAS_Final[1]) + "nroots {}\n".format(numionstates_string) + "mult {}\n".format(CAS_mults))
            theory.orcablocks = theory.orcablocks.replace('\n\n','\n')
            theory.orcablocks = theory.orcablocks.replace('\n\n','\n')
            #CAS-CI option for Ionized FInalstate. CASSCF orb-opt done on Initial state but then CAS-CI using Init-state orbs on Final-states
            if CASCI is True:
                print("CASCI option on! Final ionized states performed at CAS-CI level using Initial-state orbitals.")
                theory.orcasimpleinput = theory.orcasimpleinput + ' noiter'


            print(bcolors.OKGREEN, "Calculating Final State CASSCF Spin Multiplicities: ", [f.mult for f in Finalstates], bcolors.ENDC)
            theory.charge = Finalstates[0].charge
            #Changing to first Finalstate-mult just to satisfy ORCA.
            theory.mult = Finalstates[0].mult

            if initialorbitalfiles is not None:
                print("not tested for CASSCF...")
                print("initialorbitalfiles keyword provided.")
                print("Will use file {} as guess GBW file for this Final state.".format(initialorbitalfiles[findex + 1]))
                shutil.copyfile(initialorbitalfiles[findex + 1], theory.inputfilename + '.gbw')

            Singlepoint(fragment=fragment, theory=theory)

            #Getting state-energies of all states for each spin multiplicity (state-averaged calculation)
            fstates_dict = casscf_state_energies_grab("orca-input.out")
            print("fstates_dict: ", fstates_dict)

            # Saveing GBW and CIS file
            shutil.copyfile(theory.inputfilename + '.gbw', './' + 'Final_State' + '.gbw')
            shutil.copyfile(theory.inputfilename + '.out', './' + 'Final_State' + '.out')
            shutil.copyfile(theory.inputfilename + '.inp', './' + 'Final_State' + '.inp')

            #Each fstate linked with same GBW file and outfile
            for fstate in Finalstates:
                fstate.gbwfile = "Final_State" + ".gbw"
                fstate.outfile = "Final_State" + ".out"

            #TODO: Saving files for density Cube file creation for CASSCF

        elif MRCI is True:
            print("Modifying MRCI block for Final state, MRCI({},{})".format(MRCI_Initial[0], MRCI_Initial[1]))
            print("{} electrons in {} orbitals".format(MRCI_Initial[0], MRCI_Initial[1]))
            # Making sure multiplicties are sorted in ascending order and creating comma-sep string
            MRCI_mults = ','.join(str(x) for x in sorted([f.mult for f in Finalstates]))

            print("MRCI_mults:", MRCI_mults)
            for line in theory.orcablocks.split('\n'):
                if 'nel' in line:
                    theory.orcablocks=theory.orcablocks.replace(line,'')
                if 'norb' in line:
                    theory.orcablocks=theory.orcablocks.replace(line,'')
                if 'nroots' in line:
                    theory.orcablocks=theory.orcablocks.replace(line,'')
                if 'mult' in line:
                    theory.orcablocks=theory.orcablocks.replace(line,'')
            theory.orcablocks = theory.orcablocks.replace('\n\n','\n')

            #Add nel,norb and nroots lines back in.
            # And both spin multiplicities. Nroots for each
            #numionstates_string = ','.join(str(numionstates) for x in [f.mult for f in Finalstates])
            numionstates_string = ','.join(str(f.numionstates) for f in Finalstates)
            print("numionstates_string:", numionstates_string)
            theory.orcablocks = theory.orcablocks.replace('%casscf', '%casscf\n' + "nel {}\n".format(MRCI_Final[0]) +
                                                          "norb {}\n".format(
                                                              MRCI_Final[1]) + "nroots {}\n".format(numionstates_string) + "mult {}\n".format(MRCI_mults))

            #In Final-state MRCI we would typically use the previous CASSCF-orbitals. Hence CAS-CI and noiter
            if MRCI_CASCI_Final is True:
                if 'noiter' not in theory.orcasimpleinput.lower():
                    theory.orcasimpleinput = theory.orcasimpleinput + ' noiter '


            #Creating newblock blocks for each multiplicity
            #newblockstring=""
            #for mult in [f.mult for f in Finalstates]:
            #    newblockstring = newblockstring + "  newblock {} *\n".format(mult)+"  refs cas({},{}) end\n".format(MRCI_Final[0],MRCI_Final[1] )+ "nroots {}\n".format(mult)+"end\n"
            #theory.orcablocks = theory.orcablocks + "%mrci\n" + "printwf det\nTPrintwf 1e-16\nend"
            theory.orcablocks = theory.orcablocks.replace('\n\n', '\n')
            theory.orcablocks = theory.orcablocks.replace('\n\n', '\n')

            print(bcolors.OKGREEN, "Calculating Final State MRCI Spin Multiplicities: ", [f.mult for f in Finalstates], bcolors.ENDC)
            theory.charge = Finalstates[0].charge
            #Changing to first Finalstate-mult just to satisfy ORCA.
            theory.mult = Finalstates[0].mult

            if initialorbitalfiles is not None:
                print("not tested for MRCI...")
                print("initialorbitalfiles keyword provided.")
                print("Will use file {} as guess GBW file for this Final state.".format(initialorbitalfiles[findex + 1]))
                shutil.copyfile(initialorbitalfiles[findex + 1], theory.inputfilename + '.gbw')

            Singlepoint(fragment=fragment, theory=theory)

            #Getting state-energies of all states for each spin multiplicity
            fstates_dict = mrci_state_energies_grab("orca-input.out")
            # Saveing GBW and CIS file
            shutil.copyfile(theory.inputfilename + '.gbw', './' + 'Final_State' + '.gbw')
            shutil.copyfile(theory.inputfilename + '.out', './' + 'Final_State' + '.out')
            shutil.copyfile(theory.inputfilename + '.inp', './' + 'Final_State' + '.inp')

            #Each fstate linked with same GBW file and outfile
            for fstate in Finalstates:
                fstate.gbwfile = "Final_State" + ".gbw"
                fstate.outfile = "Final_State" + ".out"

            #TODO: Saving files for density Cube file creation for MRCI

        else:
            #TDDFT-option SCF+TDDFT for each spin multiplicity
            #################################################
            if tda==False:
                # Boolean for whether no_tda is on or not
                no_tda = True
                tddftstring="%tddft\n"+"tda false\n"+"nroots " + str(numionstates-1) + '\n'+"maxdim 15\n"+"end\n"+"\n"
            else:
                tddftstring="%tddft\n"+"tda true\n"+"nroots " + str(numionstates-1) + '\n'+"maxdim 15\n"+"end\n"+"\n"
                # Boolean for whether no_tda is on or not
                no_tda = False
            theory.extraline=theory.extraline+tddftstring
            blankline()

            for findex,fstate in enumerate(Finalstates):
                print(bcolors.OKGREEN, "Calculating Final State SCF + TDDFT. Spin Multiplicity: ", fstate.mult, bcolors.ENDC)
                theory.charge=fstate.charge
                theory.mult=fstate.mult
                if initialorbitalfiles is not None:
                    print("initialorbitalfiles keyword provided.")
                    print("Will use file {} as guess GBW file for this Final state.".format(initialorbitalfiles[findex+1]))
                    shutil.copyfile(initialorbitalfiles[findex+1], theory.inputfilename + '.gbw')


                Singlepoint(fragment=fragment, theory=theory)
                fstate.energy = scfenergygrab("orca-input.out")
                #Saveing GBW and CIS file
                shutil.copyfile(theory.inputfilename + '.gbw', './' + 'Final_State_mult' + str(fstate.mult) + '.gbw')
                shutil.copyfile(theory.inputfilename + '.cis', './' + 'Final_State_mult' + str(fstate.mult) + '.cis')
                shutil.copyfile(theory.inputfilename + '.out', './' + 'Final_State_mult' + str(fstate.mult) + '.out')
                shutil.copyfile(theory.inputfilename + '.inp', './' + 'Final_State_mult' + str(fstate.mult) + '.inp')

                fstate.gbwfile="Final_State_mult"+str(fstate.mult)+".gbw"
                fstate.outfile="Final_State_mult"+str(fstate.mult)+".out"
                fstate.cisfile="Final_State_mult"+str(fstate.mult)+".cis"


                #Grab TDDFT states from ORCA file:
                fstate.TDtransitionenergies = tddftgrab(theory.inputfilename+'.out')
                # Final state orbitals for MO-DOSplot
                fstate.occorbs_alpha, fstate.occorbs_beta, fstate.hftyp = orbitalgrab(theory.inputfilename+'.out')

                print(fstate.__dict__)
                if fstate.hftyp == "UHF":
                    fstate.restricted = False
                elif fstate.hftyp == "RHF":
                    fstate.restricted = True
                else:
                    print("hmmm")
                    exit()


                #Create Cube file of electron/spin density using orca_plot for FINAL STATE
                if Densities == 'SCF' or Densities == 'All':
                    print("Density option active. Calling orca_plot to create Cube-file for Final state SCF.")
                    os.chdir('Calculated_densities')
                    shutil.copyfile('../' + theory.inputfilename + '.gbw', './' + theory.inputfilename + '.gbw')
                    #Electron density
                    run_orca_plot(orcadir=theory.orcadir, filename=theory.inputfilename + '.gbw', option='density',
                                 gridvalue=densgridvalue)
                    os.rename(theory.inputfilename + '.scfp', 'Final_State_mult' + str(fstate.mult) + '.scfp')
                    shutil.copyfile(theory.inputfilename + '.eldens.cube', './' + 'Final_State_mult' + str(fstate.mult) + '.eldens.cube')
                    #Spin density
                    if fstate.hftyp == "UHF":
                        run_orca_plot(orcadir=theory.orcadir, filename=theory.inputfilename + '.gbw', option='spindensity',
                                 gridvalue=densgridvalue)
                        os.rename(theory.inputfilename + '.scfr', 'Final_State_mult' + str(fstate.mult) + '.scfr')
                        shutil.copyfile(theory.inputfilename + '.spindens.cube', './' + 'Final_State_mult' + str(fstate.mult) + '.spindens.cube')
                    os.chdir('..')
            blankline()
            blankline()
            print("All SCF and TDDFT calculations done (unless Densities=All)!")
    else:
        print("Theory not supported for PhotoElectronSpectrum")
        exit(1)

    blankline()
    blankline()

    if CAS is True or MRCI is True:
        FinalIPs = []
        Finalionstates = []
        FinalTDtransitionenergies =[]
        print(bcolors.OKBLUE,"Initial State energy:", stateI.energy, "au",bcolors.ENDC)
        print(bcolors.OKBLUE,"Final State energies:", fstates_dict, bcolors.ENDC)
        
        for fstate in Finalstates:
            fstate.ionstates = fstates_dict[fstate.mult]
            for ionstate in fstate.ionstates:
                fstate.IPs.append((ionstate-stateI.energy)*constants.hartoeV)
            print("Mult: {} IPs: {}".format(fstate.mult,fstate.IPs))
            FinalIPs = FinalIPs + fstate.IPs
            Finalionstates = Finalionstates + fstate.ionstates
    elif EOM is True:
        print(bcolors.OKBLUE,"Initial State CCSD energy:", stateI.energy, "au",bcolors.ENDC)
        print(bcolors.OKBLUE,"Final State CCSD+EOM-IP energies:", fstates_dict, bcolors.ENDC)
        #IPs already calculated
        print("FinalIPs:", FinalIPs)
        Finalionstates = [];finaldysonnorms=[]
        for fstate in Finalstates:
            print("Mult: {} IPs: {}".format(fstate.mult,fstate.IPs))  
            Finalionstates = Finalionstates + fstate.ionstates      
            finaldysonnorms = finaldysonnorms + fstate.dysonnorms 
    else:

        FinalIPs = []
        Finalionstates = []
        FinalTDtransitionenergies =[]
        print(bcolors.OKBLUE,"Initial State SCF energy:", stateI.energy, "au",bcolors.ENDC)
        print("")
        for fstate in Finalstates:
            print("---------------------------------------------------------------------------")
            print("Now going through SCF energy and TDDFT transitions for FinalState mult: ", fstate.mult)
            # 1st vertical IP via deltaSCF
            GSIP=(fstate.energy-stateI.energy)*constants.hartoeV
            fstate.GSIP=GSIP
            print(bcolors.OKBLUE,"Initial Final State SCF energy:", fstate.energy, "au", bcolors.ENDC)
            print(bcolors.OKBLUE,"1st vertical IP (delta-SCF):", fstate.GSIP,bcolors.ENDC)
            print("")
            # TDDFT states
            print(bcolors.OKBLUE, "TDDFT transition energies (eV) for FinalState (mult: {}) : {}\n".format(fstate.mult, fstate.TDtransitionenergies), bcolors.ENDC, )

            # Adding GS-IP to IP-list and GS ion to ionstate
            fstate.IPs.append(fstate.GSIP)
            fstate.ionstates.append(fstate.energy)
            for e in fstate.TDtransitionenergies:
                fstate.ionstates.append(e / constants.hartoeV + fstate.energy)
                fstate.IPs.append((e / constants.hartoeV + fstate.energy - stateI.energy) * constants.hartoeV)
            print("")
            print(bcolors.OKBLUE, "TDDFT-derived IPs (eV), delta-SCF IP plus TDDFT transition energies:\n", bcolors.ENDC, fstate.IPs)
            print(bcolors.OKBLUE, "Ion-state energies (au):\n", bcolors.ENDC, fstate.ionstates)
            print("")
            FinalIPs = FinalIPs + fstate.IPs
            Finalionstates = Finalionstates + fstate.ionstates
            FinalTDtransitionenergies = FinalTDtransitionenergies + fstate.TDtransitionenergies

    blankline()
    blankline()
    print("All combined Final IPs:", FinalIPs)
    blankline()
    print("All combined Ion-state energies (au):", Finalionstates)

    if noDyson is True:
        print("NoDyson is True. Exiting...")
        return


    if CAS is not True and MRCI is not True and EOM is not True:
        #MO IP spectrum:
        stk_alpha,stk_beta=modosplot(stateI.occorbs_alpha,stateI.occorbs_beta,stateI.hftyp)
        moips=sorted(stk_alpha+stk_beta)
        print(bcolors.OKBLUE,"MO IPs (negative of MO energies of State I):", bcolors.ENDC)
        print(moips)
        print("")
        print("MO-IPs (alpha), eV : ", stk_alpha)
        print("MO-IPs (beta), eV : ", stk_beta)
        print("")
        print("") 
    else:
        stk_alpha=[]
        stk_beta=[]



    ###########################################
    # Dyson orbitals for TDDFT STATES/CAS STATES/MRCI STATES
    ###########################################
    if EOM is not True:
        if theory.__class__.__name__ == "ORCATheory":
            # Todo: Need to preserve this one as it has been deleted
            #gbwfile_init = glob.glob('Init_State.gbw')[0]
            # gbwfile_final = glob.glob('Final_State_mult.gbw')[0]
            # cisfile_final = glob.glob('Final_State_mult.cis')[0]
            print(bcolors.OKGREEN, "Grabbing AO matrix, MO coefficients and excited states from ORCA GBW file and CIS file",
            bcolors.ENDC)

            print(bcolors.OKGREEN, "Grabbing AO matrix, MO coefficients and excited states from ORCA GBW file and CIS file",
            bcolors.ENDC)
            print("")
            # Get AO matrix from init state calculation
            saveAOmatrix(stateI.gbwfile, orcadir=theory.orcadir)
            # Specify frozencore or not.
            frozencore = 0

            # Grab MO coefficients and write to files mos_init and mos_final
            if os.path.isfile('./mos_init') == True:
                print(bcolors.WARNING, "mos_init file already exists in dir! Using (is this what you want?!)", bcolors.ENDC)
            else:
                print("stateI.gbwfile: ", stateI.gbwfile)
                print("stateI.restricted :", stateI.restricted)
                print("frozencore: ", frozencore)
                mos_init = get_MO_from_gbw(stateI.gbwfile, stateI.restricted, frozencore)
                writestringtofile(mos_init, "mos_init")

            for fstate in Finalstates:
                print("here")
                print("fstate.restricted:", fstate.restricted)
                print("fstate.gbwfile:", fstate.gbwfile)
                mos_final = get_MO_from_gbw(fstate.gbwfile, fstate.restricted, frozencore)
                writestringtofile(mos_final, "mos_final-mult"+str(fstate.mult))
                #os.rename("mos_final","mos_final-mult"+str(fstate.mult))

            # Create determinant file for ionized TDDFT states
            # Needs Outputfile, CIS-file, restricted-option, XXX, GS multiplicity, number of ion states and states to skip
            # States per Initial and Final options
            #TDDFT-only:
            statestoextract = [1, numionstates]
            statestoskip = [0, 0]

            # Number of multiplicity blocks I think. Should be 2 in general, 1 for GS and 1 for ionized
            # Not correct, should be actual multiplicites. Finalstate mult. If doing TDDFT-triplets then I guess we have more



            # Threshold for WF. SHARC set it to 2.0
            wfthres = 2.0

            if CAS is True:
                #CASSCF: GETTING GETERMINANTS FROM DETERMINANT-PRINTING OPTION in OUTPUTFILE
                #Combining with internal and external orbitals: internal_orbs,active_orbs,external_orbs
                #Initial
                print("Grabbing determinants from Initial State output")
                init_state = grab_dets_from_CASSCF_output(stateI.outfile)
                print("init_state:", init_state)
                #init_state_dict = [i.determinants for i in init_state]
                #init_state_dict2 = {Initialstate_mult : init_state_dict}
                #print("init_state_dict:", init_state_dict)
                #print("init_state_dict2:", init_state_dict2)
                det_init = format_ci_vectors(init_state[Initialstate_mult])
                #print("det_init:", det_init)
                # Printing to file
                writestringtofile(det_init, "dets_init")
                print("")
                #Final state. Just need to point to the one outputfile
                print("Grabbing determinants from Final State output")
                final_states = grab_dets_from_CASSCF_output(Finalstates[0].outfile)
                #print("final_states:", final_states)
                #final_states_dict = [i.determinants for i in final_states]
                #print("final_states_dict:", final_states_dict)
                #final_states_dict2 = {Initialstate_mult : final_states_dict}
                #print("final_states_dict2:", final_states_dict2)
                for fstate in Finalstates:
                    #print("fstate: ", fstate)
                    #print("fstate.mult :", fstate.mult)
                    det_final = format_ci_vectors(final_states[fstate.mult])
                    #print("det_final : ", det_final)
                    # Printing to file
                    writestringtofile(det_final, "dets_final_mult" + str(fstate.mult))
            elif MRCI is True:

                print("Grabbing determinants from Initial State output")
                init_state = grab_dets_from_MRCI_output(stateI.outfile)
                det_init = format_ci_vectors(init_state[Initialstate_mult])

                writestringtofile(det_init, "dets_init")

                print("Grabbing determinants from Final State output")
                final_states = grab_dets_from_MRCI_output(Finalstates[0].outfile)
                for fstate in Finalstates:
                    print("fstate: ", fstate)
                    print("fstate.mult :", fstate.mult)
                    det_final = format_ci_vectors(final_states[fstate.mult])
                    #print("det_final : ", det_final)
                    # Printing to file
                    writestringtofile(det_final, "dets_final_mult" + str(fstate.mult))
            else:
                #TDDFT: GETTING DETERMINANTS FROM CIS FILE
                # Final state. Create detfiles
                for fstate in Finalstates:
                    # mults = [stateFmult]
                    mults = [fstate.mult]
                    det_final = get_dets_from_cis(fstate.outfile, fstate.cisfile, fstate.restricted, mults, fstate.charge, fstate.mult,
                                                totnuccharge, statestoextract, statestoskip, no_tda, frozencore, wfthres)
                    # Printing to file
                    for blockname, string in det_final.items():
                        writestringtofile(string, "dets_final_mult"+str(fstate.mult))


                # Now doing initial state. Redefine necessary here.
                #det_init = get_dets_from_cis("Init_State1.out", "dummy", restricted_I, mults, stateIcharge, stateImult, totnuccharge,
                #                             statestoextract, statestoskip, no_tda, frozencore, wfthres)
                # RB simplification. Separate function for getting determinant-string for Initial State where only one.
                det_init = get_dets_from_single(stateI.outfile, stateI.restricted, stateI.charge, stateI.mult, totnuccharge, frozencore)
                print("det_init: ", det_init)
                # Printing to file
                for blockname, string in det_init.items():
                    writestringtofile(string, "dets_init")

            print(bcolors.OKGREEN, "AO matrix, MO coefficients and excited state determinants have been written to files:",
                bcolors.ENDC)
            # TODO
            print(bcolors.OKGREEN, "AO_overl, mos_init, mos_final, dets_final_multX", bcolors.ENDC)

            finaldysonnorms = []
            for fstate in Finalstates:
                if Densities == 'SCF' or Densities == 'All':
                    os.chdir('Calculated_densities')
                    # Create difference density between Initial-SCF and Finalstate-SCFs
                    rlowx2, dx2, nx2, orgx2, rlowy2, dy2, ny2, orgy2, rlowz2, dz2, \
                    nz2, orgz2, elems2, molcoords2, molcoords_ang2, numatoms2, filebase2, finalstate_values = read_cube('Final_State_mult' + str(fstate.mult) + '.eldens.cube')
                    print("init value 0:  ", initial_values[0])
                    print("finalstate_values 0:  ", finalstate_values[0])
                    write_cube_diff(numatoms, orgx, orgy, orgz, nx, dx, ny, dy, nz, dz, elems, molcoords, initial_values, finalstate_values,"Densdiff_SCFInit-SCFFinalmult"+str(fstate.mult))
                    print("Wrote Cube file containing density difference between Initial State and Final State.")
                    os.chdir('..')
                ###################
                # Run Wfoverlap to calculate Dyson norms. Will write to wfovl.out.  Will take a while for big systems.
                print("")

                # Check if binary exists
                if os.path.exists(path_wfoverlap) is False:
                    print("Path {} does NOT exist !".format(path_wfoverlap))
                    exit()

                print("Running WFOverlap to calculate Dyson norms for Finalstate with mult: ", fstate.mult)
                # WFOverlap calculation needs files: AO_overl, mos_init, mos_final, dets_final, dets_init
                #Poing to file in inputfile
                wfoverlapinput = """
                mix_aoovl=AO_overl
                a_mo=mos_final
                b_mo=mos_init
                a_det=dets_final
                b_det=dets_init
                a_mo_read=0
                b_mo_read=0
                ao_read=0
                moprint=1
                """
                wfoverlapinput = wfoverlapinput.replace("dets_final", "dets_final_mult"+str(fstate.mult))
                wfoverlapinput = wfoverlapinput.replace("mos_final", "mos_final-mult"+str(fstate.mult))

                run_wfoverlap(wfoverlapinput,path_wfoverlap,memory)

                #This grabs Dyson norms from wfovl.out file
                dysonnorms=grabDysonnorms()
                print("")
                print(bcolors.OKBLUE,"Dyson norms ({}):".format(len(dysonnorms)),bcolors.ENDC)
                print(dysonnorms)
                if len(dysonnorms) == 0:
                    print("List of Dyson norms is empty. Something went wrong with WfOverlap calculation.")
                    print("Setting Dyson norms to zero and continuing.")
                    dysonnorms=len(fstate.IPs)*[0.0]
                print("")
                finaldysonnorms=finaldysonnorms+dysonnorms
            print("")
            print(bcolors.OKBLUE, "Final combined Dyson norms ({}):".format(len(finaldysonnorms)), bcolors.ENDC)
            print(finaldysonnorms)
            print("")
    else:
        print("")
        print(bcolors.WARNING,"WARNING: Dyson norms not calculated for IP-EOM-CCSD. Instead using dominant singles amplitudes as an approximation",bcolors.ENDC)
        print("Approximate Dyson norms: ", finaldysonnorms)

        assert len(FinalIPs) == len(finaldysonnorms), "List of Dysonnorms not same size as list of IPs."

        #Print table with info
        print("-------------------------------------------------------------------------")
        print("FINAL RESULTS for fragment (Label: {}  Formula: {})".format(fragment.label, fragment.prettyformula))
        print("-------------------------------------------------------------------------")
        print("Initial state:")
        print("{:>6} {:>7} {:^20} {:^5}".format("State no.", "Mult", "TotalE (Eh)", "State-type"))
        if EOM is True:
            print("{:>6d} {:>7d} {:20.11f} {:>8}".format(0, stateI.mult, stateI.energy, "CCSD"))            
        else:
            print("{:>6d} {:>7d} {:20.11f} {:>8}".format(0, stateI.mult, stateI.energy, "SCF"))
        print("")
        print("Final ionized states:")
        if CAS is True or MRCI is True:
            stype='SCF'
            print("{:>6} {:>7} {:^20} {:8} {:10} {:>7}".format("State no.", "Mult", "TotalE (Eh)", "IE (eV)", "Dyson-norm", "State-type"))
            for i, (E, IE, dys) in enumerate(zip(Finalionstates,FinalIPs,finaldysonnorms)):
                #Getting spinmult
                if MultipleSpinStates is True:
                    #Change test. what mult we are in.. TODO: Check this for correctness
                    if i < Finalstates[0].numionstates:
                        spinmult=Finalstates[0].mult
                    else:
                        spinmult=Finalstates[1].mult
                else:
                    spinmult=stateF1.mult
                print("{:>6d} {:>7d} {:20.11f} {:>10.3f} {:>10.5f} {:>10}".format(i, spinmult, E, IE, dys,stype))
        elif EOM is True:
            stype='EOM'
            print("{:>6} {:>7} {:^20} {:8} {:10} {:>7}".format("State no.", "Mult", "TotalE (Eh)", "IE (eV)", "Dyson-norm", "State-type"))
            for i, (E, IE, dys) in enumerate(zip(Finalionstates,FinalIPs,finaldysonnorms)):
                #Getting spinmult
                if MultipleSpinStates is True:
                    #Change test. what mult we are in.. TODO: Check this for correctness
                    if i < Finalstates[0].numionstates:
                        spinmult=Finalstates[0].mult
                    else:
                        spinmult=Finalstates[1].mult
                else:
                    spinmult=stateF1.mult
                print("{:>6d} {:>7d} {:20.11f} {:>10.3f} {:>10.5f} {:>10}".format(i, spinmult, E, IE, dys,stype))        
        
        else:
            print("{:>6} {:>7} {:^20} {:8} {:10} {:>7} {:>15}".format("State no.", "Mult", "TotalE (Eh)", "IE (eV)", "Dyson-norm", "State-type", "TDDFT Exc.E. (eV)"))
            for i, (E, IE, dys) in enumerate(zip(Finalionstates,FinalIPs,finaldysonnorms)):
                #Getting type of state
                if i == 0:
                    stype='SCF'
                else:
                    if tda is True:
                        stype = 'TDA'
                    else:
                        stype = 'TDDFT'
                    if MultipleSpinStates is True:
                        if i == numionstates:
                            stype='SCF'
                #Getting spinmult
                if MultipleSpinStates is True:
                    if i < numionstates:
                        spinmult=Finalstates[0].mult
                    else:
                        spinmult=Finalstates[1].mult
                else:
                    spinmult=stateF1.mult
                #Getting TDDFT transition energy
                if stype == "TDA" or stype == "TDDFT":
                    if i < numionstates:
                        TDtransenergy = Finalstates[0].TDtransitionenergies[i-1]
                    else:
                        #print("i:", i)
                        index=i-numionstates-1
                        #print("index:", index)
                        #print("Finalstates[1].TDtransitionenergies: ", Finalstates[1].TDtransitionenergies)
                        TDtransenergy = Finalstates[1].TDtransitionenergies[index]

                else:
                    TDtransenergy=0.0
                print("{:>6d} {:>7d} {:20.11f} {:>10.3f} {:>10.5f} {:>10} {:>17.3f}".format(i, spinmult, E, IE, dys,stype, TDtransenergy))


        #Here doing densities for each TDDFT-state. SCF-states already done.
        if Densities == 'All':
            print("")
            print(bcolors.OKMAGENTA, "Densities option: All . Will do TDDFT-gradient calculation for each TDDFT-state (expensive)", bcolors.ENDC)
            os.chdir('Calculated_densities')

            #Adding Keepdens and Engrad to do TDDFT gradient
            theory.orcasimpleinput=theory.orcasimpleinput+' KeepDens Engrad'

            if '/C' not in theory.orcasimpleinput:
                theory.orcasimpleinput = theory.orcasimpleinput + ' AutoAux'

            for findex, fstate in enumerate(Finalstates):
                print(bcolors.OKGREEN, "Calculating Final State SCF + TDDFT DENSITY CALCULATION. Spin Multiplicity: ", fstate.mult, bcolors.ENDC)
                theory.charge = fstate.charge
                theory.mult = fstate.mult
                shutil.copyfile('../'+'Final_State_mult' + str(fstate.mult) + '.gbw','Final_State_mult' + str(fstate.mult) + '.gbw')
                os.rename('Final_State_mult' + str(fstate.mult) + '.gbw', theory.inputfilename + '.gbw')


                #Looping over each TDDFT-state and doing TDDFT-calc
                for tddftstate in range(1,numionstates):
                    print("-------------------------------------------------")
                    print("Running TDDFT-gradient for State: ", tddftstate)
                    #Adding Iroot to get state-specific gradient+density                                              ''
                    if tda == False:
                        # Boolean for whether no_tda is on or not
                        print("Not sure if Full-TDDFT density is available. TO BE CHECKED.")
                        no_tda = True
                        tddftstring = "%tddft\n" + "tda false\n" + "nroots " + str(
                            numionstates - 1) + '\n' + "maxdim 15\n" + "Iroot {}\n".format(tddftstate) + "end\n" + "\n"
                    else:
                        tddftstring = "%tddft\n" + "tda true\n" + "nroots " + str(
                            numionstates - 1) + '\n' + "maxdim 15\n" + "Iroot {}\n".format(tddftstate) + "end\n" + "\n"
                        # Boolean for whether no_tda is on or not
                        no_tda = False
                    theory.extraline = "%method\n"+"frozencore FC_NONE\n"+"end\n" + tddftstring

                    #Turning off stability analysis. Not available for gradient run.
                    if 'stabperform true' in theory.orcablocks:
                        print("Turning off stability analysis.")
                        theory.orcablocks=theory.orcablocks.replace("stabperform true", "stabperform false")


                    Singlepoint(fragment=fragment, theory=theory)
                    # TDDFT state done. Renaming cisp and cisr files
                    os.rename('orca-input.cisp', 'Final_State_mult' + str(fstate.mult)+'TDDFTstate_'+str(tddftstate)+'.cisp')
                    os.rename('orca-input.cisr', 'Final_State_mult' + str(fstate.mult)+'TDDFTstate_'+str(tddftstate)+'.cisr')
                    print("Calling orca_plot to create Cube-file for Final state TDDFT-state.")

                    #Doing spin-density Cubefilefor each cisr file
                    run_orca_plot(orcadir=theory.orcadir, filename=theory.inputfilename + '.gbw', option='cisspindensity',gridvalue=densgridvalue,
                                  densityfilename='Final_State_mult' + str(fstate.mult)+'TDDFTstate_'+str(tddftstate)+'.cisr' )
                    os.rename(theory.inputfilename + '.spindens.cube', 'Final_State_mult' + str(fstate.mult)+'TDDFTstate_'+str(tddftstate)+'.spindens.cube')
                    #Doing eldensity Cubefile for each cisp file and then take difference with Initstate-SCF cubefile

                    run_orca_plot(orcadir=theory.orcadir, filename=theory.inputfilename + '.gbw', option='cisdensity',gridvalue=densgridvalue,
                                  densityfilename='Final_State_mult' + str(fstate.mult)+'TDDFTstate_'+str(tddftstate)+'.cisp' )
                    os.rename(theory.inputfilename + '.eldens.cube', 'Final_State_mult' + str(fstate.mult)+'TDDFTstate_'+str(tddftstate)+'.eldens.cube')

                    final_dens = 'Final_State_mult' + str(fstate.mult)+'TDDFTstate_'+str(tddftstate)+'.eldens.cube'
                    rlowx2, dx2, nx2, orgx2, rlowy2, dy2, ny2, orgy2, rlowz2, dz2, \
                    nz2, orgz2, elems2, molcoords2, molcoords_ang2, numatoms2, filebase2, finalstate_values = read_cube(final_dens)
                    write_cube_diff(numatoms, orgx, orgy, orgz, nx, dx, ny, dy, nz, dz, elems, molcoords, initial_values, finalstate_values,
                            "Densdiff_SCFInit-TDDFTFinalmult" + str(fstate.mult)+'TDState'+str(tddftstate))
                    print("Wrote Cube file containing density difference between Initial State and Final TDDFT State: ", tddftstate)

            os.chdir('..')


        #Writing stuff to file. Useful for separate plotting of IPs and Dysonnorms
        print("")
        print("Printing IPs, Dyson-norms, MOs to file: PES-Results.txt")
        print("Can be read by PES.Read_old_results() function")
        #Writing file in Configparser format for easy read-in below
        with open("PES-Results.txt", 'w') as resultfile:
            resultfile.write("[Results]\n")
            resultfile.write("IPs : {}\n".format(FinalIPs))
            resultfile.write("Dyson-norms : {}\n".format(finaldysonnorms))
            resultfile.write("MOs_alpha : {}\n".format(stk_alpha))
            resultfile.write("MOs_beta : {}\n".format(stk_beta))

        return FinalIPs, finaldysonnorms, stk_alpha, stk_beta

def Read_old_results():
    print("Reading file PES-Results.txt ...")
    # Parsing of files
    import json
    import configparser
    #from configparser import ConfigParser
    parser = configparser.ConfigParser()

    parser.read('PES-Results.txt')
    #Using JSON to load
    #From: https://stackoverflow.com/questions/335695/lists-in-configparser
    IPs = json.loads(parser.get("Results", "IPs"))
    dysonnorms = json.loads(parser.get("Results", "Dyson-norms"))
    mos_alpha = json.loads(parser.get("Results", "MOs_alpha"))
    mos_beta = json.loads(parser.get("Results", "MOs_beta"))

    #with open("PES-Results.txt") as file:
    #    for line in file:
    #        if 'IPs' in line:
    #            IPs = [float(i) for i in line.split(':')[1].replace('[', '').replace(']', '').split(',')]
    #        if 'Dyson' in line:
    #            dysonnorms = [float(i) for i in line.split(':')[1].replace('[', '').replace(']', '').split(',')]
    #        if 'MOs_alpha' in line:
    #            mos_alpha = [float(i) for i in line.split(':')[1].replace('[', '').replace(']', '').split(',')]
    #        if 'MOs_beta' in line:
    #            mos_beta = [float(i) for i in line.split(':')[1].replace('[', '').replace(']', '').split(',')]

    return IPs, dysonnorms, mos_alpha, mos_beta


def plot_PES_Spectrum(IPs=None, dysonnorms=None, mos_alpha=None, mos_beta=None, plotname='PES-plot',
                          start=None, finish=None, broadening=0.1, points=10000, hftyp_I=None, MOPlot=False):
    if IPs is None or dysonnorms is None:
        print("plot_PES_Spectrum requires IPs and dysonnorms variables")
        exit(1)

    assert len(IPs) == len(dysonnorms), "List of Dysonnorms not same size as list of IPs." 

    if mos_alpha is None:
        MOPlot=False
        print("mos_alpha and mos_beta not provided. Skipping MO-DOS plot.")
    else:
        MOPlot=True

    blankline()
    print(bcolors.OKGREEN,"-------------------------------------------------------------------",bcolors.ENDC)
    print(bcolors.OKGREEN,"plot_PES_Spectrum: Plotting TDDFT-Dyson-norm spectrum and MO-spectrum",bcolors.ENDC)
    print(bcolors.OKGREEN,"-------------------------------------------------------------------",bcolors.ENDC)
    blankline()
    print("IPs ({}): {}".format(len(IPs),IPs))
    print("Dysonnorms ({}): {}".format(len(dysonnorms),dysonnorms))

    if start is None:
        start = IPs[0] - 8
        finish = IPs[-1] + 8

    #########################
    # Plot spectra.
    ########################
    print(bcolors.OKGREEN, "Plotting-range chosen:", start, "-", finish, "eV", "with ", points, "points and ",
              broadening, "eV broadening.", bcolors.ENDC)

    # X-range is electron binding energy
    x = np.linspace(start, finish, points)
    stkheight = 0.5
    strength = 1.0

    ######################
    # MO-dosplot
    ######################
    if MOPlot is True:
        if hftyp_I is None:
            print("hftyp_I not set (value: RHF or UHF). Assuming hftyp_I=RHF and ignoring beta MOs.")
            blankline()
        # Creates DOS out of electron binding energies (negative of occupied MO energies)
        # alpha
        occDOS_alpha = 0
        for count, peak in enumerate(mos_alpha):
            occdospeak = Gaussian(x, peak, strength, broadening)
            #virtdospeak = Gaussian(x, peak, strength, broadening)
            occDOS_alpha += occdospeak
        # beta
        if hftyp_I == "UHF":
            occDOS_beta = 0
            for count, peak in enumerate(mos_beta):
                occdospeak = Gaussian(x, peak, strength, broadening)
                #virtdospeak = Gaussian(x, peak, strength, broadening)
                occDOS_beta += occdospeak

        # Write dat/stk files for MO-DOS
        if MOPlot is True:
            datfile = open('MO-DOSPLOT' + '.dat', 'w')
            stkfile_a = open('MO-DOSPLOT' + '_a.stk', 'w')
            if hftyp_I == "UHF":
                stkfile_b = open('MO-DOSPLOT' + '_b.stk', 'w')

            for i in range(0, len(x)):
                datfile.write(str(x[i]) + " ")
                datfile.write(str(occDOS_alpha[i]) + " \n")
                if hftyp_I == "UHF":
                    datfile.write(str(occDOS_beta[i]) + "\n")
            datfile.close()
            # Creating stk file for alpha. Only including sticks for plotted region
            stk_alpha2 = []
            stk_alpha2height = []
            for i in mos_alpha:
                if i > x[-1]:
                    # print("i is", i)
                    continue
                else:
                    stkfile_a.write(str(i) + " " + str(stkheight) + "\n")
                    stk_alpha2.append(i)
                    stk_alpha2height.append(stkheight)
            stkfile_a.close()
            stk_beta2 = []
            stk_beta2height = []
            if hftyp_I == "UHF":
                for i in mos_beta:
                    if i > x[-1]:
                        continue
                    else:
                        stkfile_b.write(str(i) + " " + str(stkheight) + "\n")
                        stk_beta2.append(i)
                        stk_beta2height.append(stkheight)
                stkfile_b.close()

    ######################
    # TDDFT states DOS
    ######################
    tddftDOS = 0
    for peak, strength in zip(IPs, dysonnorms):
        tddospeak = Gaussian(x, peak, strength, broadening)
        tddftDOS += tddospeak

    #Save dat file
    with open("TDDFT-DOS.dat", 'w') as tdatfile:
        for i,j in zip(x,tddftDOS):
            tdatfile.write("{:13.10f} {:13.10f} \n".format(i,j))
    #Save stk file
    with open("TDDFT-DOS.stk", 'w') as tstkfile:
        for b,c in zip(IPs,dysonnorms):
            tstkfile.write("{:13.10f} {:13.10f} \n".format(b,c))


    ##################################
    # Plot with Matplotlib
    ####################################
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    if MOPlot is True:
        # MO-DOSPLOT for initial state. Here assuming MO energies of initial state to be good approximations for IPs
        ax.plot(x, occDOS_alpha, 'C2', label='alphaMO')
        ax.stem(stk_alpha2, stk_alpha2height, label='alphaMO', basefmt=" ", markerfmt=' ', linefmt='C2-', use_line_collection=True)
        if hftyp_I == "UHF":
            ax.plot(x, occDOS_beta, 'C2', label='betaMO')
            ax.stem(stk_beta2, stk_beta2height, label='betaMO', basefmt=" ", markerfmt=' ', linefmt='C2-', use_line_collection=True)


    ##############
    # TDDFT-STATES
    ###############
    ax.plot(x, tddftDOS, 'C3', label='TDDFT')
    ax.stem(IPs, dysonnorms, label='TDDFT', markerfmt=' ', basefmt=' ', linefmt='C3-', use_line_collection=True)
    plt.xlabel('eV')
    plt.ylabel('Intensity')
    #################################
    plt.xlim(start, finish)
    plt.legend(shadow=True, fontsize='small')
    plt.savefig(plotname + '.png', format='png', dpi=200)
    # plt.show()
    print(BC.OKGREEN,"ALL DONE!", BC.END)
