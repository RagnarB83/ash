#Module for calculating PhotoElectron Spectra

# Calculate PES spectra using the Dyson orbital approach.
path_to_wfoverlap='/home/bjornsson/sharc-master/bin/wfoverlap.x'

#####################################3

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

class bcolors:
    HEADER = '\033[95m' ; OKBLUE = '\033[94m'; OKGREEN = '\033[92m'; WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'

eldict={'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Mo':42,'W':74,'Ru':44,'I':53}

def create_ORCA_inputfiles():
    inpfile_I = open('State_init.inp', 'w')
    for l in orcatemplate:
        inpfile_I.write(l)
    inpfile_I.write("*xyzfile"+' '+str(stateIcharge)+' '+str(stateImult)+' '+xyzfile)
    inpfile_I.write("\n")
    inpfile_I.close()
    inpfile_F = open('State_final.inp', 'w')
    if stateFmult > stateImult:
        ElectronSpinRemoval='Beta'
    else:
        ElectronSpinRemoval='Alpha'
    for l in orcatemplate:
        inpfile_F.write(l)
    inpfile_F.write("%method\n")
    inpfile_F.write("frozencore FC_NONE\n")
    inpfile_F.write("end\n")
    inpfile_F.write("%tddft\n")
    inpfile_F.write("tda true\n")
    inpfile_F.write("nroots "+str(numionstates-1)+'\n')
    inpfile_F.write("maxdim 10\n")
    inpfile_F.write("end\n")
    inpfile_F.write("\n")
    inpfile_F.write("*xyzfile"+' '+str(stateFcharge)+' '+str(stateFmult)+' '+xyzfile)
    inpfile_F.write("\n")
    inpfile_F.close()
    print(bcolors.OKBLUE,"ORCA Inputfiles created:",bcolors.ENDC)
    print(bcolors.OKBLUE,"State_init.inp",bcolors.ENDC)
    print(bcolors.OKBLUE,"State_final.inp",bcolors.ENDC)

#Write logfile for script for 2nd run
def write_logfile():
    with open("PEScalclogfile.py", 'w') as logfile:
        logfile.write("xyzfile="+"\'"+str(xyzfile)+"\'\n")
        logfile.write("stateIcharge="+str(stateIcharge)+"\n")
        logfile.write("stateImult="+str(stateImult)+"\n")
        logfile.write("stateFcharge="+str(stateFcharge)+"\n")
        logfile.write("stateFmult="+str(stateFmult)+"\n")
        logfile.write("numionstates="+str(numionstates)+"\n")

#Calculate nuclear charge from XYZ-file
def nuccharge(file):
    el=[]
    with open(file) as f:
        for count,line in enumerate(f):
            if count >1:
                el.append(line.split()[0])
    totnuccharge=0
    for e in el:
        atcharge=eldict[e]
        totnuccharge+=atcharge
    return totnuccharge

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
def saveAOmatrix(file):
    NAO,Smat=get_smat_from_gbw(file)
    string='%i %i\n' % (NAO,NAO)
    for irow in range(NAO):
        for icol in range(NAO):
            string+='% .7e ' % (Smat[icol][irow])
        string+='\n'
    outfile='AO_overl'
    with open(outfile, 'w') as ofile:
        ofile.write(string)

#Get smat from GBW. Converted to Python3 From Python2 function in SHARC
def get_smat_from_gbw(file1, file2=''):

    if not file2:
      file2=file1

    # run orca_fragovl
    string='orca_fragovl %s %s' % (file1,file2)
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
      #print line
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

#Get determinants from ORCA cisfile.
#Converted to Python3 from function in SHARC
def get_dets_from_cis(logfile,cisfilename,restr,mults,gscharge,gsmult,totnuccharge,nstates_to_extract,nstates_to_skip,no_tda,frozencore,wfthres):
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
    import pprint
    # get general infos
    #job=QMin['IJOB']
    #restr=QMin['jobs'][job]['restr']
    #mults=QMin['jobs'][job]['mults']
    #gsmult=QMin['multmap'][-job][0]
    #nstates_to_extract=deepcopy(QMin['states'])
    #nstates_to_skip=[ QMin['states_to_do'][i]-QMin['states'][i] for i in range(len(QMin['states'])) ]
    #print "mults:", mults
    #print "gsmult:", gsmult
    #print "RBX2. nstates_to_extract:", nstates_to_extract
    for i in range(len(nstates_to_extract)):
        if not i+1 in mults:
            nstates_to_extract[i]=0
            nstates_to_skip[i]=0
        elif i+1==gsmult:
            nstates_to_extract[i]-=1
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
    #print "RB. here 7"
    if not 'NOA' in  infos:
      nstates_onfile=0
      charge=gscharge
      #charge=QMin['chargemap'][gsmult]
      nelec=float(totnuccharge-charge)
      infos['NOA']=int(nelec/2. + float(gsmult-1)/2. )
      infos['NOB']=int(nelec/2. - float(gsmult-1)/2. )
      infos['NVA']=infos['nbsuse']-infos['NOA']
      infos['NVB']=infos['nbsuse']-infos['NOB']
      infos['NFC']=0
    else:
      # get all info from cis file
      CCfile=open(cisfilename,'rb')
     # with open("CCfilewritten", 'w') as ccfilex:
       # bla=struct.iter_unpack('i', CCfile.read(4))
       # for b in bla:
       #     ccfilex.write(str(b))
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
        for istate in range(nstates_to_extract[mult-1]):
            CCfile.read(40)
            dets={}
            print("dets:", dets)
            print("header:", header)
            print("--------")
            print("range(header[0],header[1]+1):", range(header[0],header[1]+1))
            for iocc in range(header[0],header[1]+1):
              print("--------")
              print("range(header[2],header[3]+1):", range(header[2],header[3]+1))
              for ivirt in range(header[2],header[3]+1):
                print("istate, iocc, ivirt", istate, iocc, ivirt)
                #print("here1")
                dets[ (iocc,ivirt,1) ]=struct.unpack('d', CCfile.read(8))[0]
            print("1 case done!!")
            pprint.pprint(dets)
            if not restr:
              print("here2")
              print("range(header[4],header[5]+1):", range(header[4],header[5]+1))
              for iocc in range(header[4],header[5]+1):
                print("range(header[6],header[7]+1):", range(header[6],header[7]+1))
                for ivirt in range(header[6],header[7]+1):
                  print("istate, iocc, ivirt", istate, iocc, ivirt)
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
                    print("rb here...")
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
        for istate in range(nstates_to_skip[mult-1]):
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
def run_wfoverlap(wfoverlapinput,path_to_wfoverlap):
    wfoverlapfilefile = open('wfovl.inp', 'w')
    for l in wfoverlapinput:
        wfoverlapfilefile.write(l)
    wfoverlapfilefile.close()
    wfcommand='%s -m 2000 -f wfovl.inp' % (path_to_wfoverlap)
    print("Running wfoverlap program:")
    print("may take a while...")
    print(wfcommand)
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
    with open(file) as f:
        for line in f:
            if 'Total Energy       :' in line:
                Energy=float(line.split()[-4])
    return Energy

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



########################
# MAIN program
########################

#PhotoElectronSpectrum function
#Provide theory

def PhotoElectronSpectrum(theory=theory, fragment=fragment, Ionstatemult=None, numstates=50, numcores=1 ):
    print(bcolors.OKGREEN,"PES-calc: Calculated PES spectra via TDDFT and Dyson-norm approach",bcolors.ENDC)
    if Ionstatemult==None:
        print("Provide spin multiplicity of ionized state to PhotoElectronSpectrum: e.g. Ionstatemult=2 ")
        exit(1)
    stateIcharge=theory.charge
    stateImult=theory.mult
    stateFcharge=theory.charge-1
    stateFmult=Ionstatemult
    print("")
    print("Coordinates: ")
    fragment.print_coords()

    print(bcolors.OKBLUE,"StateI: Charge=", stateIcharge, "Multiplicity", stateImult,bcolors.ENDC)
    print(bcolors.OKBLUE,"StateF: Charge=", stateFcharge, "Multiplicity", stateFmult,bcolors.ENDC)
    print("")
    print(bcolors.OKBLUE,"Calculated ion states:", numstates,bcolors.ENDC)
    print(bcolors.OKBLUE,"TDDFT-calculated ion states:", numstates-1,bcolors.ENDC)
    print("CPU Cores available: ", numcores)
    print("")

    #Create inputfiles
    create_inputfiles()
    #Run calculations

    print(bcolors.OKGREEN,"Reading outputfiles...",bcolors.ENDC)
    #Initial state orbitals for MO-DOSplot
    occorbsI_alpha,occorbsI_beta,hftyp_I=orbitalgrab(outfile_init)
    occorbsF_alpha,occorbsF_beta,hftyp_F=orbitalgrab(outfile_final)

    #SCF energies
    init_E=scfenergygrab(outfile_init)
    final_E=scfenergygrab(outfile_final)
    #1st vertical IP via deltaSCF=
    GSIP=(final_E-init_E)*hartoev
    print(bcolors.OKBLUE,"Initial State SCF energy:", init_E, "au",bcolors.ENDC)
    print(bcolors.OKBLUE,"Initial Final State SCF energy:", final_E, "au", bcolors.ENDC)
    print(bcolors.OKBLUE,"1st vertical IP:", GSIP,bcolors.ENDC)
    print("")

    #MO IP spectrum:
    stk_alpha,stk_beta=modosplot(occorbsI_alpha,occorbsI_beta,hftyp_I)
    moips=sorted(stk_alpha+stk_beta)
    print(bcolors.OKBLUE,"MO IPs (negative of MO energies of State I):", bcolors.ENDC)
    print(moips)
    print("")

    # TDDFT states
    TDtransitionenergies = tddftgrab(outfile_final)
    print(bcolors.OKBLUE, "TDDFT transition energies (eV):\n", bcolors.ENDC, TDtransitionenergies)

    ionstates = [];
    IPs = []
    # Adding GS-IP to IP-list and GS ion to ionstate
    IPs.append(GSIP);
    ionstates.append(final_E)
    for e in TDtransitionenergies:
        ionstates.append(e / hartoev + final_E)
        IPs.append((e / hartoev + final_E - init_E) * hartoev)
    print("")
    print(bcolors.OKBLUE, "Final IPs (eV):\n", bcolors.ENDC, IPs)
    print(bcolors.OKBLUE, "Ion-state energies (au):\n", bcolors.ENDC, ionstates)
    print("")

    ###########################################
    # Dyson orbitals for TDDFT STATES
    ###########################################

    print(bcolors.OKGREEN, "Grabbing AO matrix, MO coefficients and excited states from ORCA GBW file and CIS file",
          bcolors.ENDC)

    print(bcolors.OKGREEN, "Grabbing AO matrix, MO coefficients and excited states from ORCA GBW file and CIS file",
          bcolors.ENDC)
    print("")
    # Get AO matrix from init state calculation
    saveAOmatrix(gbwfile_init)
    # need to specify Initial/Final states are restricted or not.
    if hftyp_I == "UHF":
        restricted_I = False
    elif hftyp_I == "RHF":
        restricted_I = True
    else:
        print("hmmm")
        exit()
    if hftyp_F == "UHF":
        restricted_F = False
    elif hftyp_F == "RHF":
        restricted_F = True
    else:
        print("hmmm")
        exit()

    # Specify frozencore or not.
    frozencore = 0
    # Grab MO coefficients and write to files mos_init and mos_final

    if os.path.isfile('./mos_init') == True:
        print(bcolors.WARNING, "mos_init file already exists in dir! Using (is this what you want?!)", bcolors.ENDC)
    else:
        mos_init = get_MO_from_gbw(gbwfile_init, restricted_I, frozencore)
        writestringtofile(mos_init, "mos_init")
    if os.path.isfile('./mos_final') == True:
        print(bcolors.WARNING, "mos_final file exists in dir! Using (is this what you want?!)", bcolors.ENDC)
    else:
        mos_final = get_MO_from_gbw(gbwfile_final, restricted_F, frozencore)
        writestringtofile(mos_final, "mos_final")

    # Create determinant file for ionized TDDFT states

    # Needs Outputfile, CIS-file, restricted-option, XXX, GS multiplicity, number of ion states and states to skip
    # States per Initial and Final options
    statestoextract = [1, numionstates]
    statestoskip = [0, 0]
    # Number of multiplicity blocks I think. Should be 2 in general, 1 for GS and 1 for ionized
    mults = [2]
    # Boolean for whether no_tda is on or not
    no_tda = False
    # Threshold for WF. SHARC set it to 2.0
    wfthres = 2.0
    # Final state.

    # Skip slow determinant file creation if file already exists
    if os.path.isfile('./dets_final') == True:
        print(bcolors.WARNING, "dets_final file already exists in dir! Using (is this what you want?!)", bcolors.ENDC)
    else:
        # Final state
        det_final = get_dets_from_cis(outfile_final, cisfile_final, restricted_F, mults, stateFcharge, stateFmult,
                                      totnuccharge, statestoextract, statestoskip, no_tda, frozencore, wfthres)
        # Now doing initial state. Redefine necessary here.
        statestoextract = [1, numionstates]
        mults = [1]
        det_init = get_dets_from_cis(outfile_init, "dummy", restricted_I, mults, stateIcharge, stateImult, totnuccharge,
                                     statestoextract, statestoskip, no_tda, frozencore, wfthres)
        # Printing to file
        for blockname, string in det_init.items():
            writestringtofile(string, "dets_init")
        for blockname, string in det_final.items():
            writestringtofile(string, "dets_final")
        print(bcolors.OKGREEN, "AO matrix, MO coefficients and excited state determinants have been written to files:",
              bcolors.ENDC)
        print(bcolors.OKGREEN, "AO_overl, mos_init, mos_final, dets.1, dets.2", bcolors.ENDC)


    ###################
    # WFOverlap calculation
    # Needs files: AO_overl, mos_init, mos_final, dets_final, dets_init
    wfoverlapinput="""
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

    # Run Wfoverlap to calculate Dyson norms. Will write to wfovl.out.  Will take a while for big systems.
    print("")
    if os.path.isfile('./wfovl.out')==True:
        print(bcolors.WARNING, "wfovl.out file exists in dir!",bcolors.ENDC)
        print(bcolors.WARNING,"Using Dyson norms from file (is this what you want?!)",bcolors.ENDC)
    else:
        run_wfoverlap(wfoverlapinput,path_to_wfoverlap)

    #This grabs Dyson norms from wfovl.out file
    dysonnorms=grabDysonnorms()
    print("")
    print(bcolors.OKBLUE,"Dyson norms:",bcolors.ENDC)
    print(dysonnorms)
    print("")

    #########################
    # Plot spectra.
    ########################
    print(bcolors.OKGREEN, "------------------------------------", bcolors.ENDC)
    print(bcolors.OKGREEN, "Now in plotting mode", bcolors.ENDC)
    print(bcolors.WARNING, "You can give plot options as command-line arguments here:", bcolors.ENDC)
    print(bcolors.WARNING, "PES-calc.py start end points broadening (assuming eV)", bcolors.ENDC)
    print(bcolors.WARNING, "Example: PES-calc.py 3 8 10000 0.2", bcolors.ENDC)
    try:
        start = float(sys.argv[1])
        finish = float(sys.argv[2])
        points = int(sys.argv[3])
        broad = float(sys.argv[4])
        print("")
        print(bcolors.OKGREEN, "Plotting-range chosen:", start, "-", finish, "eV", "with ", points, "points and ",
              broad, "eV broadening.", bcolors.ENDC)
    except IndexError:
        start = IPs[0] - 8
        finish = IPs[-1] + 8
        points = 10000
        broad = 0.1
        print(bcolors.OKGREEN, "Using values for plotting:", bcolors.ENDC)
        print(bcolors.OKGREEN, "Plotting-range used:", start, "-", finish, "eV. Numpoints:", points, "and broadening:",
              broad, "eV", bcolors.ENDC)

    # X-range is electron binding energy
    x = np.linspace(start, finish, points)

    stkheight = 0.5
    strength = 1.0
    ######################
    # MO-dosplot
    ##################
    # Creates DOS out of electron binding energies (negative of occupied MO energies)
    # alpha
    occDOS_alpha = 0
    for count, peak in enumerate(stk_alpha):
        occdospeak = Gaussian(x, peak, strength, broad)
        virtdospeak = Gaussian(x, peak, strength, broad)
        occDOS_alpha += occdospeak
    # beta
    if hftyp_I == "UHF":
        occDOS_beta = 0
        for count, peak in enumerate(stk_beta):
            occdospeak = Gaussian(x, peak, strength, broad)
            virtdospeak = Gaussian(x, peak, strength, broad)
            occDOS_beta += occdospeak

    # TDDFT states DOS
    tddftDOS = 0
    for peak, strength in zip(IPs, dysonnorms):
        tddospeak = Gaussian(x, peak, strength, broad)
        tddftDOS += tddospeak

    # Write dat/stk files for MO-DOS
    datfile = open('MO-DOSPLOT' + '.dat', 'w')
    stkfile_a = open('MO-DOSPLOT' + '_a.stk', 'w')
    if hftyp_I == "UHF":
        stkfile_b = open('MO-DOSPLOT' + '_b.stk', 'w')
    for i in range(0, len(x)):
        datfile.write(str(x[i]) + " ")
        datfile.write(str(occDOS_alpha[i]) + " ")
        if hftyp_I == "UHF":
            datfile.write(str(occDOS_beta[i]) + "\n")
    datfile.close()
    # Creating stk file for alpha. Only including sticks for plotted region
    stk_alpha2 = []
    stk_alpha2height = []
    for i in stk_alpha:
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
        for i in stk_beta:
            if i > x[-1]:
                print("i is", i)
                continue
            else:
                stkfile_b.write(str(i) + " " + str(stkheight) + "\n")
                stk_beta2.append(i)
                stk_beta2height.append(stkheight)
        stkfile_b.close()

    ##################################
    # Plot with Matplotlib
    ####################################
    import matplotlib.pyplot as plt

    # MO-DOSPLOT for initial state. Here assuming MO energies of initial state to be good approximations for IPs
    fig, ax = plt.subplots()
    ax.plot(x, occDOS_alpha, 'C2', label='alphaMO')
    ax.stem(stk_alpha2, stk_alpha2height, label='alphaMO', basefmt=" ", markerfmt=' ', linefmt='C2-')
    if hftyp_I == "UHF":
        ax.plot(x, occDOS_beta, 'C2', label='betaMO')
        ax.stem(stk_beta2, stk_beta2height, label='betaMO', basefmt=" ", markerfmt=' ', linefmt='C2-')
    ########################

    ##############
    # TDDFT-STATES
    ###############
    ax.plot(x, tddftDOS, 'C3', label='TDDFT')
    ax.stem(IPs, dysonnorms, label='TDDFT', markerfmt=' ', basefmt=' ', linefmt='C3-')
    plt.xlabel('eV')
    plt.ylabel('Intensity')
    #################################
    plt.xlim(start, finish)
    plt.legend(shadow=True, fontsize='small')
    plt.savefig('PES-plot' + '.png', format='png', dpi=200)
    # plt.show()

