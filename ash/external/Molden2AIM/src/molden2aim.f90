!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Molden2AIM: a utility to convert the format from MOLDEN to AIM-WFN, AIM-WFX, and NBO-47.
!
! Webpage: https://github.com/zorkzou/Molden2AIM
!
! E-mail: qcband@gmail.com
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
program Molden2AIM
 implicit real(kind=8) (a-h,o-z)
 parameter(nprog=10)

 allocatable   :: iza(:), icore(:), xyz(:,:), ene(:), ispin(:), occup(:), carmo(:), sphmo(:), lprtmo(:),  &
                  s2cd(:), s2cf(:), s2cg(:), s2ch(:),  &  ! coefficients for sph. --> Car. GTF
                  mapatm(:),        &  ! atomic center of each contracted shell
                  lqnm(:),          &  ! L-quantum number of each contracted shell
                  nshlls(:),        &  ! starting primitive shell in each contracted shell
                  nshlln(:),        &  ! number of primitive shells in each contracted shell
                  expgto(:),        &  ! primitive exponent of each primitive shell
                  congto(:),        &  ! contraction coeffients of each primitive shell
                  scalmocar(:)         ! scaling factor of each contracted Cartesian function

 ! Cartesian NC-/C-GTFs; Spherical NC-/C-GTFs
 dimension         :: ncar(2),nsph(2)

 dimension         :: ICntrl(8)
 logical           :: doit,ifopen,ifwbo

 character*157     :: fwfn,fwfx,fnbo
 character*164     :: fmdn
 character*10      :: dt
 character*5       :: ver
 character*1       :: yn,L2U
 character*120     :: stline

!=================================================================================================================================
!  head
!=================================================================================================================================
 ver = "5.0.2"
 dt  = "10/09/2020"
 call headprt(ver,dt)

!=================================================================================================================================
!  Initialization
!=================================================================================================================================
 ICntrl(1)=1         ! Generating a standard Molden file in Cartesian functions
 ICntrl(2)=1         ! Generating a WFN file
 ICntrl(3)=0         ! Generating a WFX file
 ICntrl(4)=0         ! Generating an NBO .47 file
 ICntrl(5)=0         ! Checking normalization for WFN
 ICntrl(6)=0         ! Checking normalization for WFX
 ICntrl(7)=0         ! Checking normalization for NBO's .47
 ICntrl(8)=0         ! Calculate Generalized Wiberg bond order when ICntrl(4) >= 0
                     !
                     ! >0:  always performs the operation without asking the user
                     ! =0:  asks the user whether to perform the operation
                     ! <0:  always neglect the operation without asking the user

 IAllMO=0            ! for new-MOLDEN, WFN, and WFX
                     ! =0: print only the orbitals with occ. number > 5.0d-8
                     ! <0: print only the orbitals with occ. number > 0.1 (debug)
                     ! >0: print all the orbitals

 ICln=1              ! delete temporary files (1) or not (0)

 iprog=0             ! 1: ORCA, 2: CFOUR, 3: TURBOMOLE, 4: JAGUAR, ... (see Sub. countmo)
                     ! 0: other programs, or read [Program] xxx from MOLDEN.
                     ! If one of the above program is always used, you can provide iprog here without defining
                     ! [Program] xxx in MOLDEN any more.

 nosupp=0            ! Print (< 0) or do not print (> 0) supporting information; =0: ask the user

 ledt=0              ! Type of EDF. X2C/HF (=0) or X2C/PBE0 (=1)

 lpspin=1            ! 0/1: print the $MOSPIN block in wfn (=1)

 iunknw=1            ! 0/1: print 0.0 for Energy and 2.0 for Virial Ratio

!=================================================================================================================================
! Port numbers. Do not modify them.
!
! Port numbers 4x: important input and output files
! Port numbers 6x: temporary files used in sub. backupgto
! Port numbers 7x: temporary files
!=================================================================================================================================
 iini=40             ! m2a.ini
 imod=44             ! original molden file
 inmd=45             ! new molden file in Cartesian functions
 iwfn=46             ! wfn file
 iwfx=47             ! wfx file
 inbo=48             ! NBO 47 file
 iwbo=49             ! Generalized Wiberg bond order

 imtm=64             ! a temporary Molden file
 itmp=65             ! see backupgto
 igin=66             ! see backupgto
 igol=67             ! see backupgto

 iatm=70             ! Cartesian coordinates
 igto=71             ! basis functions
 iedf=73             ! EDF data for ECP
 isym=74             ! Symm for Molden

!=================================================================================================================================
! The following integers will be determined later. Do not modify them here.
!=================================================================================================================================
 lfc4=1              ! a scaling factor for occ. numbers: x 1 (1) or x 2 (2)
 lecp=0              ! PP (ECP or MCP) is used (> 0; = #core_electron) or not (0)
 lsca=0              ! MO factors will be scaled if lsca=1
 lsph=0              ! MOs from Molden are in Cartesian (0) or spherical (1) basis functions
 lrdecp=0            ! for core information in Molden
 lsymm=0             ! 0/1: without / with symmetry in Molden
 lalph=0             ! 0/1: without / with spin
 lbeta=0             ! 0/1: without / with beta spin
 lspout=0            ! 0/1: saved MOs are in Cartesian or spherical basis functions
 tolocc=0.0d0        ! tolerance of occupation number
 nbopro=0            ! 0/1: more data will be printed in NBO-47 if nbopro=1

!=================================================================================================================================
!  read user's parameters from m2a.ini
!=================================================================================================================================
 call crtini(iini)
 call usrini(iini,nprog,ICntrl,ICln,IAllMO,iprog,nosupp,ledt,lpspin,lspout,iunknw,tolocc,nbopro,stline)

!=================================================================================================================================
!  program list which can save MOLDEN file
!=================================================================================================================================
 doit = .false.
 if(nosupp == 0) then
   write(*,"(/,' Do you want to see the information about supported programs? (Yes / [No])',/,' > ',$)")
   read(*,"(a1)")yn
   yn=L2U(yn)
   if(yn == 'Y') doit = .true.
 else if(nosupp > 0) then
   write(*,"(/,' nosupp > 0: the program list will not be shown.')")
 else if(nosupp < 0) then
   doit = .true.
 end if
 if(doit) then
   call SuppInf
 else
   write(*,8000)
 end if

!=================================================================================================================================
!  define file names
!=================================================================================================================================
 call filename(imod,fmdn,fwfn,fwfx,fnbo)

 open(imtm,file='mtm123456789.tmp')
 open(itmp,file='tmp123456789.tmp')
 open(igin,file='gin123456789.tmp')
 open(igol,file='god123456789.tmp')
 open(iatm,file='atm123456789.tmp')
 open(igto,file='gto123456789.tmp')
 open(iedf,file='edf123456789.tmp')
 open(isym,file='sym123456789.tmp')

!=================================================================================================================================
!  check the molden file, and search the [PSEUDO] (lrdecp = 1) or [CORE] (lrdecp = 2) block
!=================================================================================================================================
 call ChkMolden(imod,lrdecp,stline,ierr)
   if(ierr == 1) goto 9910

!=================================================================================================================================
!  Program
!=================================================================================================================================
 call RdProg(imod,nprog,iprog,stline,ierr)
   if(ierr == 1) goto 9910

!=================================================================================================================================
!  Save molden to a temporary file, and reorder atoms in [Atoms] or [GTO] if necessary.
!=================================================================================================================================
 call ROADrv(imod,imtm,itmp,iprog,stline,ierr)
   if(ierr == 1) goto 9910

!=================================================================================================================================
!  nat (number of atoms) & nchar [= sum(iza)] from [Atoms], nbasmo (Car. or sph) & nmotot from [MO]
!=================================================================================================================================
 call countmo(imtm,iatm,nat,nchar,nbasmo,nmotot,stline,ierr)
   if(ierr == 1) goto 9910

!=================================================================================================================================
!  backup GTFs
!=================================================================================================================================
 call backupgto(imtm,igto,itmp,igin,igol,nat,iprog,stline,ierr)
   if(ierr == 1) goto 9910

!=================================================================================================================================
!  ncar(2), nsph(2), MaxL, nshell, and ngto from [GTO] / [BASIS]
!=================================================================================================================================
 call npgau(igto,iprog,ncar,nsph,MaxL,nshell,ngto,stline,ierr)
   if(ierr == 1) goto 9910
 write(*,"(' >>> SUMMARY',//,                               &
   5x,'Number of atoms: ',25x,i8,/,                         &
   5x,'Maximum angular momentum: ',16x,i8,/,                &
   5x,'Number of primitive GTF shells: ',10x,i8,/,          &
   5x,'Number of contracted GTF shells: ',9x,i8,/,          &
   5x,'Number of primitive Cartesian functions: ',1x,i8,/,  &
   5x,'Number of contracted Cartesian functions: ',i8,/,    &
   5x,'Number of primitive spherical functions: ',1x,i8,/,  &
   5x,'Number of contracted spherical functions: ',i8)") nat,MaxL,ngto,nshell,ncar(1),ncar(2),nsph(1),nsph(2)

!=================================================================================================================================
!  check: Cartesian vs. spherical functions
!=================================================================================================================================
 call checkcar(iprog,nbasmo,ncar,nsph,lsph,ierr)
   if(ierr == 1) goto 9910
 if(lsph == 0) then
   write(*,"(5x,'GTFs being used:',31x,'Car.')")
 else if(lsph == 1) then
   write(*,"(5x,'GTFs being used:',31x,'Sph.')")
 end if

!=================================================================================================================================
!  check compatibility
!=================================================================================================================================
 if(iprog /= 0) call chkbstyp(lsph,iprog,MaxL,ierr)
   if(ierr == 1) goto 9910

 allocate(iza(nat), icore(nat), s2cd(30), s2cf(70), s2cg(135), s2ch(231),  &
   xyz(3,nat), ene(nmotot), ispin(nmotot), occup(nmotot), lprtmo(nmotot),  &
   carmo(ncar(2)*nmotot), sphmo(nsph(2)*nmotot), scalmocar(ncar(2)), mapatm(nshell), lqnm(nshell),  &
   nshlls(nshell), nshlln(nshell), expgto(ngto), congto(ngto),  &
   stat=ierr)
   if(ierr /= 0) then
     write(iout,"(' ### Insufficient Memory!')")
     goto 9910
   end if

!=================================================================================================================================
!  read atomic coordinates from iatm
!=================================================================================================================================
 call RdAtoms(iatm,nat,iza,xyz)

!=================================================================================================================================
!  read basis functions
!=================================================================================================================================
 call RdBasis(igto,nat,nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,stline)

!=================================================================================================================================
!  compute scaling factors of MO. carmo is used as scratch.
!  lsca=1 if MO coeffients need to be scaled.
!=================================================================================================================================
 call carmoscale(iprog,nshell,lqnm,lsca,ncar(2),scalmocar,carmo,ierr)
   if(ierr == 1) goto 9910

!=================================================================================================================================
!  read MO coefficients
!=================================================================================================================================
 call RdMOs(imtm,isym,nbasmo,nmotot,sumocc,lsymm,lalph,lbeta,ispin,ene,occup,carmo,stline,ierr)
   if(ierr == 1) goto 9910
 ! reorder Cartesian H functions of CFour's Molden file
 if(iprog == 2 .and. lsph == 0 .and. MaxL > 4) call C4HFun(nshell,lqnm,ncar(2),nmotot,carmo)

!=================================================================================================================================
!  Will new Molden & NBO-47 be saved in spherical GTFs?
!=================================================================================================================================
 if(lspout == 1) then
   if(MaxL < 2) then
     lspout = 0
   else if(lbeta == 0 .and. nmotot /= nsph(2)) then
     lspout = 0
   else if(lbeta == 1 .and. nmotot /= nsph(2)*2) then
     lspout = 0
   end if
 end if

!=================================================================================================================================
!  MOs to be printed in new Molden, wfn, and wfx
!=================================================================================================================================
 call PrtListMO(nmotot,nmoprt,IAllMO,tolocc,occup,lprtmo)
 write(*,"(5x,'Total number of MOs:',22x,i8)") nmotot
 if(IAllMO > 0) then
   write(*,"(5x,'Number of MOs to be printed:',14x,i8)") nmoprt
 else
   write(*,"(5x,'Number of MOs to be printed:',14x,i8,' (tol = ',d8.2,')')") nmoprt, tolocc
 end if

!=================================================================================================================================
!  read core data
!=================================================================================================================================
 chanet = 0.d0
 call RdCore(imtm,iprog,chanet,ntote,nchar,sumocc,lrdecp,nat,lbeta,lfc4,lecp,iza,icore,stline,yn,ierr)
   if(ierr == 1) goto 9910
 ! check occup
 call ChkOcc(lbeta,lfc4,sumocc,nmotot,occup,ierr)
   if(ierr == 1) goto 9910
 write(*,8000)

!=================================================================================================================================
!  Spherical GTOs: convert MOs from spherical functions into Cartesian functions. Here sphmo is used as scratch.
!  Cartesian GTOs: scale MOs
!=================================================================================================================================
 if(lsph == 1) then
   ! if(MaxL > 1) call s2c_coef_old(MaxL,s2cd,s2cf,s2cg,ierr)
   if(MaxL > 1) call s2c_coef(MaxL,s2cd,s2cf,s2cg,s2ch,ierr)
     if(ierr == 1) goto 9910
   sphmo =  carmo(1:nsph(2)*nmotot)
   call S2CMO(nshell,iprog,nsph(2),ncar(2),nmotot,lqnm,s2cd,s2cf,s2cg,s2ch,sphmo,carmo)
 else
   call ScaleCarMO(ncar(2),nmotot,scalmocar,carmo)
 end if

!=================================================================================================================================
!  MOs in spherical GTOs
!=================================================================================================================================
 if(lspout == 1) then
   call c2s_coef(MaxL,s2cd,s2cf,s2cg,s2ch,ierr)
     if(ierr == 1) goto 9910
   call C2SMO(nshell,nsph(2),ncar(2),nmotot,lqnm,s2cd,s2cf,s2cg,s2ch,carmo,sphmo)
 end if

!=================================================================================================================================
!  save a standard Molden file in Cartesian basis functions
!=================================================================================================================================
 if(ICntrl(1) > 0)then
   doit = .true.
 else if(ICntrl(1) == 0)then
   write(*,"(////,' Do you want to generate a new Molden file? ([Yes] / No)',/,' > ',$)")
   read(*,"(a1)")yn
   yn=L2U(yn)
   if(yn == 'N')then
     doit = .false.
   else
     doit = .true.
   end if
 else
   doit = .false.
 end if

 if(doit) then
   if(lspout == 1) then
     call genmdn(fmdn,inmd,isym,ver,dt,nat,MaxL,lecp,iza,icore,xyz, nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  &
       lspout,nmotot,nsph(2),lsymm,ispin,ene,occup,lprtmo,sphmo,stline)
   else
     call genmdn(fmdn,inmd,isym,ver,dt,nat,MaxL,lecp,iza,icore,xyz, nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  &
       lspout,nmotot,ncar(2),lsymm,ispin,ene,occup,lprtmo,carmo,stline)
   end if
   write(*,8000)
 end if

!=================================================================================================================================
!  save a wfn file
!=================================================================================================================================
 if(ICntrl(2) > 0)then
   doit = .true.
 else if(ICntrl(2) == 0)then
   write(*,"(////,' Do you want to generate a *.WFN file? ([Yes] / No)',/,' > ',$)")
   read(*,"(a1)")yn
   yn=L2U(yn)
   if(yn == 'N')then
     doit = .false.
   else
     doit = .true.
   end if
 else
   doit = .false.
 end if

 if(doit) then
   call genwfn(iwfn,fwfn,ver,dt,lpspin, nat,lecp,iza,icore,xyz,  MaxL,nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  &
     ncar(1),ncar(2),nmotot,nmoprt,lalph,lbeta,ispin,ene,occup,lprtmo,carmo,  stline)

   ! Check the AIM-WFN file
   if(ICntrl(5) > 0)then
     doit = .true.
   else if(ICntrl(5) == 0)then
     write(*,"(/,' Do you want to check the *.WFN file? ([Yes] / No)')")
     ! WFN with ECP is supported only by MultiWFN at present
     if(lecp == 0) write(*,"(' (Omit it if you are using AIMALL because AIMALL will do it much faster.)')")
     write(*,"(' > ',$)")
     read(*,"(a1)")yn
     yn=L2U(yn)
     if(yn == 'N')then
       doit = .false.
     else
       doit = .true.
     end if
   else
     doit = .false.
   end if

   if(doit) then
     call CheckWFN(iwfn,nat,ncar(1),MaxL,ierr)
     if(ierr /= 0) goto 9910
   end if

   write(*,8000)
 end if

!=================================================================================================================================
!  save a wfx file
!=================================================================================================================================
 if(ICntrl(3) > 0)then
   doit = .true.
 else if(ICntrl(3) == 0)then
   write(*,"(////,' Do you want to generate a *.WFX file? ([Yes] / No)',/,' > ',$)")
   read(*,"(a1)")yn
   yn=L2U(yn)
   if(yn == 'N')then
     doit = .false.
   else
     doit = .true.
   end if
 else
   doit = .false.
 end if

 if(doit) then
   call genwfx(iwfx,fwfx,ver,dt,  iedf,lecp,ledt,chanet,ntote,nat,iza,icore,xyz,  &
     MaxL,nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  &
     ncar(1),ncar(2),nmotot,nmoprt,lalph,lbeta,ispin,ene,occup,lprtmo,carmo,  iunknw,stline)

   ! Check the AIM-WFX file
   if(ICntrl(6) > 0)then
     doit = .true.
   else if(ICntrl(6) == 0)then
     write(*,"(/,' Do you want to check the *.WFX file? ([Yes] / No)',/,  &
       ' (Omit it if you are using AIMALL because AIMALL will do it much faster.)',/,' > ',$)")
     read(*,"(a1)")yn
     yn=L2U(yn)
     if(yn == 'N')then
       doit = .false.
     else
       doit = .true.
     end if
   else
     doit = .false.
   end if

   if(doit) then
     call CheckWFX(iwfx,nat,ncar(1),nmotot,MaxL,ierr,stline)
     if(ierr /= 0) goto 9910
   end if

   write(*,8000)
 end if

!=================================================================================================================================
!  save NBO *.47
!=================================================================================================================================
 if(ICntrl(4) > 0)then
   doit = .true.
 else if(ICntrl(4) == 0)then
   write(*,"(////,' Do you want to generate NBO *.47 file? ([Yes] / No)',/,' > ',$)")
   read(*,"(a1)")yn
   yn=L2U(yn)
   if(yn == 'N')then
     doit = .false.
   else
     doit = .true.
   end if
 else
   doit = .false.
 end if

 if(doit) then

   ! Generalized Wiberg bond order (i.e. Mayer's bond order for RHF/UHF/RKS/UKS)?
   if(nat < 2) then
   	 doit = .false.
   else
     if(ICntrl(8) > 0)then
       doit = .true.
     else if(ICntrl(8) == 0)then
       write(*,"(/,' Do you want to calculate Wiberg bond order? ([Yes] / No)')")
       write(*,"(' > ',$)")
       read(*,"(a1)")yn
       yn=L2U(yn)
       if(yn == 'N')then
         doit = .false.
       else
         doit = .true.
       end if
     else
       doit = .false.
     end if
   end if
   ifwbo = doit
   if(doit) then
     lenth=Len_trim(fnbo)-3
     OPEN(iwbo,FILE=(fnbo(1:lenth)//'_wbo.out'))
   end if

   call DrvNBO(inbo,fnbo,ver,dt,nbopro,  nat,iza,icore,xyz,  MaxL,lspout,nshell,ngto,ncar(2),nsph(2),mapatm,lqnm,  &
     nshlls,nshlln,expgto,congto,  nmotot,lalph,lbeta,ispin,ene,occup,carmo,sphmo,  iwbo,ifwbo,  ierr,stline)
   if(ierr /= 0) goto 9910

   ! Check the NBO .47 file
   if(ICntrl(7) > 0)then
     doit = .true.
   else if(ICntrl(7) == 0)then
     write(*,"(/,' Do you want to check the *.47 file? ([Yes] / No)')")
     write(*,"(' > ',$)")
     read(*,"(a1)")yn
     yn=L2U(yn)
     if(yn == 'N')then
       doit = .false.
     else
       doit = .true.
     end if
   else
     doit = .false.
   end if

   if(doit) then
     call CheckNBO(inbo,nbopro,nat,sumocc,ierr,stline)
     if(ierr /= 0) goto 9910
   end if
   write(*,8000)
 end if

 !================================================================================================================================
 deallocate(iza, icore, s2cd, s2cf, s2cg, s2ch, xyz, ene, ispin, occup, lprtmo, carmo, sphmo, scalmocar, mapatm, lqnm,  &
   nshlls, nshlln, expgto, congto)
 9910  continue

 inquire(unit=iini,opened=ifopen)
 if(ifopen) close(iini)
 close(imod)
 close(imtm,status='delete')
 inquire(unit=inmd,opened=ifopen)
 if(ifopen) close(inmd)
 inquire(unit=iwfn,opened=ifopen)
 if(ifopen) close(iwfn)
 inquire(unit=iwfx,opened=ifopen)
 if(ifopen) close(iwfx)
 inquire(unit=inbo,opened=ifopen)
 if(ifopen) close(inbo)
 inquire(unit=iwbo,opened=ifopen)
 if(ifopen) close(iwbo)
 if(ICln == 0)then
   close(itmp)
   close(igin)
   close(igol)
   close(iatm)
   close(igto)
   close(iedf)
   close(isym)
 else
   close(itmp,status='delete')
   close(igin,status='delete')
   close(igol,status='delete')
   close(iatm,status='delete')
   close(igto,status='delete')
   close(iedf,status='delete')
   close(isym,status='delete')
 end if

 call estop(0)

 8000  format(1x,77('='))
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Check the NBO's .47 file
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine CheckNBO(inbo,nbopro,NAtom,TotE1,info,ctmp)
 implicit real(kind=8) (a-h,o-z)
 parameter(tole=5.d-5,tola=1.d-6)
 character*100     :: ctmp
 allocatable       :: S(:),P(:),CMO(:),SC1(:),SC2(:),SC3(:)
 logical           :: found1

 info=1

 ! read NBAS and (optional) OPEN
 rewind(inbo)
 NBAS = 0
 found1 = .false.
 iuhf = 1
 do while(.true.)
   read(inbo,"(a100)",err=5000,end=5000)ctmp

   if(index(ctmp,"$GENNBO") /= 0) found1=.true.
   if(found1 .and. index(ctmp,"NBAS") /= 0) then
     ip1 = index(ctmp,"NBAS") + 4
     ip1 = index(ctmp(ip1:),"=") + ip1
     read(ctmp(ip1:),*) NBAS
   end if
   if(found1 .and. index(ctmp,"OPEN") /= 0) iuhf = 2
   if(found1 .and. index(ctmp,"$END") /= 0) exit
 end do
   if(NBAS < 1) then
     write(iout,"(' ### NBAS < 1 in sub. CheckNBO!')")
     goto 5010
   end if

 NTT=NBAS*(NBAS+1)/2
 NSS=NBAS*NBAS
 allocate(S(NTT), P(NTT*iuhf), stat=info)
   if(info /= 0) then
     write(iout,"(' ### Insufficient Memory in sub. CheckNBO (1)!')")
     goto 5010
   end if

 info=1
 ! read S & P from the *.47 file
 rewind(inbo)
 do while(.true.)
   read(inbo,"(a100)",err=5000,end=5000)ctmp
   if(index(ctmp,"$OVERLAP") /= 0) then
     write(*,"(/,'  Reading overlap matrix...')")
     read(inbo,*,err=5000,end=5000)(S(i),i=1,NTT)
   else if(index(ctmp,"$DENSITY") /= 0) then
     write(*,"('  Reading density matrix...')")
     read(inbo,*,err=5000,end=5000)(P(i),i=1,NTT*iuhf)
     exit
   end if
 end do

 if(iuhf == 2) call AAdd(NTT,P,P(1+NTT),P)

 TotE2 = TracLL(NBAS,S,P)

 DifE=abs(TotE1-TotE2)
 DifA=DifE/dble(NAtom)
 write(*,"(/,'  Sum of MO Occupancies',22x,' = ',f18.10)")TotE1
 write(*,"('  Analytically integrated number of electrons = ',f18.10)")TotE2
 write(*,"('  Difference',33x,' = ',f18.10,/,'  Difference per atom',24x,' = ',f18.10)")DifE,DifA
 if(NAtom < 30) then
   if(DifE > tole) write(*,"(/,' ### Warning! Normalization check failed.')")
 else
   if(DifA > tola) write(*,"(/,' ### Warning! Normalization check failed.')")
 end if

 ! check C' S C = I
 if(nbopro == 1) then
   ! search $FOCK
   found1 = .false.
   rewind(inbo)
   do while(.true.)
     read(inbo,"(a100)",err=4000,end=4000)ctmp
     if(index(ctmp,"$FOCK") /= 0) then
       found1 = .true.
       exit
     end if
   end do
   if(.NOT. found1) goto 4000

   allocate(CMO(NSS*iuhf), SC1(NSS), SC2(NSS), SC3(NSS), stat=info)
   if(info /= 0) then
     write(iout,"(' ### Insufficient Memory in sub. CheckNBO (2)!')")
     goto 5000
   end if

   ! read MOs
   info = 0
   rewind(inbo)
   do while(.true.)
     read(inbo,"(a100)",err=5000,end=5000)ctmp
     if(index(ctmp,"$LCAOMO") /= 0) then
       ! write(*,"('  Reading MO coefficients...')")
       read(inbo,*,err=5000,end=5000)(CMO(i),i=1,NSS*iuhf)
       exit
     end if
   end do

   call LT2Sqr(NBAS,S,SC1)
   call ChkOrtho(NBAS,iuhf,SC1,CMO,SC2,SC3)

   deallocate(CMO, SC1, SC2, SC3)
 end if

 4000  info=0

 5000  deallocate(S,P)
 5010  return

 !---------------------------------------------------------------------- private subroutines
 contains

 ! Check orthogonality of MOs: C' * S * C = I
 subroutine ChkOrtho(n,iuhf,smat,cmo,SC1,SC2)
  implicit real(kind=8) (a-h,o-z)
  parameter(tol=1.d-4)
  dimension         :: smat(n*n), cmo(n*n,iuhf), SC1(n*n), SC2(n,n)
  logical           :: iffail

  ! write(*,"(/,'  Checking orthogonality of MOs...')")
  do k=1,iuhf
    call MatMult(2,n,n,n,cmo(1,k),smat,SC1)
    call MatMult(1,n,n,n,SC1,cmo(1,k),SC2)
    erro = 0.0d0
    errd = 0.0d0
    do i = 1, n
      do j= 1, i-1
        erro = max(erro, abs(SC2(i,j)), abs(SC2(j,i)))
      end do
      errd = max(errd, abs(SC2(i,j)-1.0d0))
    end do
    if(k == 1 .and. iuhf == 2) then
      write(*,"('  Error of orthogonality in C^T * S * C - I   = ',f18.10,' (o,alpha)',/,48x,f18.10,' (d,alpha)')") erro,errd
    else if(k == 2) then
      write(*,"(48x,f18.10,' (o,beta)',/,48x,f18.10,' (d,beta)')") erro,errd
    else
      write(*,"('  Error of orthogonality in C^T * S * C - I   = ',f18.10,' (o)',/,48x,f18.10,' (d)')") erro,errd
    end if
    iffail = erro > tol .or. errd > tol
  end do

  if(iffail) write(*,"(' ### Warning! ChkOrtho fails.',/,  &
    '  Do not use the $FOCK data for Second-Order Perturbation Theory analysis.')")

  return
 end subroutine ChkOrtho

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Check the AIM-WFX file
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine CheckWFX(iwfx,MaxAtm,maxpg,NMO,MaxL,info,ctmp)
 implicit real(kind=8) (a-h,o-z)
 parameter(tole=5.d-5,tola=1.d-6)
 allocatable       :: smat(:), Occ(:), r(:,:), Expon(:), CMO(:), FNor(:), ICent(:), IType(:), scr1(:), scr2(:,:)
 character*100     :: ctmp
 character*42      :: tag

 allocate(smat(maxpg*(1+maxpg)/2), Occ(NMO), r(3,MaxAtm), Expon(maxpg), CMO(maxpg), FNor(maxpg), ICent(maxpg), IType(maxpg),  &
   scr1(3), scr2(MaxL+1,7), stat=info)
   if(info /= 0) then
     write(iout,"(' ### Insufficient Memory in sub. CheckWFX!')")
     goto 5010
   end if

 info=0
 ! read basis function from the *.WFX file
 write(*,"(/,'  Reading basis functions...')")
 call RdBsx(iwfx,MaxAtm,maxpg,NMO,NGauss,NAtom,r,Expon,ICent,IType,FNor,ctmp,tag,info)
   if(info /= 0) goto 5000

 ! compute the overlap matrix
 write(*,"('  Computing the overlap matrix...')")
 call OvDriver(smat,NGauss,IType,Expon,r,ICent,MaxL,scr1,scr2,info)
   if(info /= 0) goto 5000

 ! Occ
 rewind(iwfx)
 tag = "<Molecular Orbital Occupation Numbers>"
 do while(.true.)
   read(iwfx,"(a100)",end=5020)ctmp
   if(index(ctmp,tag(1:38)) /= 0) exit
 end do
 read(iwfx,*,err=5020,end=5020) (Occ(i),i=1,NMO)

 ! MO block
 rewind(iwfx)
 tag = "<Molecular Orbital Primitive Coefficients>"
 do while(.true.)
   read(iwfx,"(a100)",end=5020)ctmp
   if(index(ctmp,tag(1:42)) /= 0) exit
 end do

 ! compute the number of electrons
 write(*,"('  Computing the integrated number of electrons...')")
 TotE1=0.d0
 TotE2=0.d0
 do imo=1,NMO
   call RdMOx(iwfx,NGauss,CMO,FNor,ctmp,tag,info)
   if(info /= 0) goto 5000
   TotE1=TotE1+Occ(imo)
   TotE2=TotE2+Occ(imo)*VSV(NGauss,CMO,smat)
 end do

 DifE=abs(TotE1-TotE2)
 DifA=DifE/dble(NAtom)
 write(*,"(/,'  Sum of MO Occupancies',22x,' = ',f18.10)")TotE1
 write(*,"('  Analytically integrated number of electrons = ',f18.10)")TotE2
 write(*,"('  Difference',33x,' = ',f18.10,/,'  Difference per atom',24x,' = ',f18.10)")DifE,DifA

 if(NAtom < 30) then
   if(DifE > tole) write(*,"(/,' ### Warning! Normalization check failed.')")
 else
   if(DifA > tola) write(*,"(/,' ### Warning! Normalization check failed.')")
 end if

 5000  deallocate(smat, Occ, r, Expon, CMO, FNor, ICent, IType, scr1, scr2)
 5010  return

 5020  write(*,"(/,' ### Error when reading MO in the WFX file!')")
 info=1
 goto 5000

 !---------------------------------------------------------------------- private subroutines
 contains

 ! read basis function from the *.WFX file
 subroutine RdBsx(iwfx,MaxAtm,MaxPG,NMO,NGauss,NAtom,r,Expon,ICent,IType,FNor,ctmp,tag,info)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: r(3,*),Expon(*),ICent(*),IType(*),FNor(*)
  character*100     :: ctmp
  character*42      :: tag

  info=0

  ! NMO
  rewind(iwfx)
  tag = "<Number of Occupied Molecular Orbitals>"
  do while(.true.)
    read(iwfx,"(a100)",end=9010)ctmp
    if(index(ctmp,tag(1:39)) /= 0) exit
  end do
  read(iwfx,*,err=9010,end=9010) NMO

  ! NAtom
  rewind(iwfx)
  tag = "<Number of Nuclei>"
  do while(.true.)
    read(iwfx,"(a100)",end=9010)ctmp
    if(index(ctmp,tag(1:18)) /= 0) exit
  end do
  read(iwfx,*,err=9010,end=9010) NAtom

  ! NGauss
  rewind(iwfx)
  tag = "<Number of Primitives>"
  do while(.true.)
    read(iwfx,"(a100)",end=9010)ctmp
    if(index(ctmp,tag(1:22)) /= 0) exit
  end do
  read(iwfx,*,err=9010,end=9010) NGauss

  ! check
  if(NAtom /= MaxAtm) goto 9920
  if(NGauss /= MaxPG) goto 9930

  ! coordinates
  rewind(iwfx)
  tag = "<Nuclear Cartesian Coordinates>"
  do while(.true.)
    read(iwfx,"(a100)",end=9010)ctmp
    if(index(ctmp,tag(1:31)) /= 0) exit
  end do
  read(iwfx,*,err=9010,end=9010) ((r(i,j), i=1,3), j=1,NAtom)

  ! GTO Center
  rewind(iwfx)
  tag = "<Primitive Centers>"
  do while(.true.)
    read(iwfx,"(a100)",end=9010)ctmp
    if(index(ctmp,tag(1:19)) /= 0) exit
  end do
  read(iwfx,*,err=9010,end=9010) (ICent(i),i=1,NGauss)

  ! GTO Type
  rewind(iwfx)
  tag = "<Primitive Types>"
  do while(.true.)
    read(iwfx,"(a100)",end=9010)ctmp
    if(index(ctmp,tag(1:17)) /= 0) exit
  end do
  read(iwfx,*,err=9010,end=9010) (IType(i),i=1,NGauss)

  ! GTO Exponent
  rewind(iwfx)
  tag = "<Primitive Exponents>"
  do while(.true.)
    read(iwfx,"(a100)",end=9010)ctmp
    if(index(ctmp,tag(1:21)) /= 0) exit
  end do
  read(iwfx,*,err=9010,end=9010) (Expon(i),i=1,NGauss)

  ! compute normalization factors
  do i=1,NGauss
    FNor(i)=fnorm_lmn(Expon(i),IType(i))
  end do

  1000  return

  9010  write(*,"(/,' ### Error when reading the WFX file!')")
  info=1
  return
  9920  write(*,"(/,' ### Wrong! NAtom /= MaxAtm:',2i8)")NAtom,MaxAtm
  info=1
  return
  9930  write(*,"(/,' ### Wrong! NGauss /= MaxPG:',2i8)")NGauss,MaxPG
  info=1
  return
 end subroutine RdBsx

 ! read MO coefficients from the *.WFX file
 subroutine RdMOx(iwfx,NGauss,CMO,FNor,ctmp,tag,info)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: CMO(*),FNor(*)
  character*100     :: ctmp
  character*12      :: tag

  tag = "</MO Number>"
  info= 0
  do while(.true.)
    read(iwfx,"(a100)",end=9010)ctmp
    if(index(ctmp,tag(1:12)) /= 0) exit
  end do
  read(iwfx,*,err=9010,end=9010)(CMO(i),i=1,NGauss)

  do i=1,NGauss
    CMO(i)=CMO(i)/FNor(i)
  end do
  return

  9010  write(*,"(/,' ### Error when reading MO in the WFX file!')")
  info=1
  return
 end subroutine RdMOx

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Check the AIM-WFN file
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine CheckWFN(iwfn,MaxAtm,maxpg,MaxL,info)
 implicit real(kind=8) (a-h,o-z)
 parameter(tole=5.d-5,tola=1.d-6)
 allocatable       :: r(:,:),Expon(:),CMO(:),FNor(:),ICent(:),IType(:),smat(:),scr1(:),scr2(:,:)

 allocate(r(3,MaxAtm), Expon(maxpg), CMO(maxpg), FNor(maxpg), ICent(maxpg), IType(maxpg), smat(maxpg*(1+maxpg)/2),  &
   scr1(3), scr2(MaxL+1,7), stat=info)
   if(info /= 0) then
     write(iout,"(' ### Insufficient Memory in sub. CheckWFN!')")
     goto 5010
   end if

 info=0
 ! read basis function from the *.WFN file
 write(*,"(/,'  Reading basis functions...')")
 call RdBas(iwfn,MaxAtm,maxpg,NMO,NGauss,NAtom,r,Expon,ICent,IType,FNor,info)
   if(info /= 0) goto 5000
 ! compute the overlap matrix
 write(*,"('  Computing the overlap matrix...')")
 call OvDriver(smat,NGauss,IType,Expon,r,ICent,MaxL,scr1,scr2,info)
   if(info /= 0) goto 5000

 ! compute the number of electrons
 write(*,"('  Computing the integrated number of electrons...')")
 TotE1=0.d0
 TotE2=0.d0
 do imo=1,NMO
   call RdMO(iwfn,NGauss,Occ,CMO,FNor)
   TotE1=TotE1+Occ
   TotE2=TotE2+Occ*VSV(NGauss,CMO,smat)
 end do

 DifE=abs(TotE1-TotE2)
 DifA=DifE/dble(NAtom)
 write(*,"(/,'  Sum of MO Occupancies',22x,' = ',f18.10)")TotE1
 write(*,"('  Analytically integrated number of electrons = ',f18.10)")TotE2
 write(*,"('  Difference',33x,' = ',f18.10,/,'  Difference per atom',24x,' = ',f18.10)")DifE,DifA
 if(NAtom < 30) then
   if(DifE > tole) write(*,"(/,' ### Warning! Normalization check failed.')")
 else
   if(DifA > tola) write(*,"(/,' ### Warning! Normalization check failed.')")
 end if

 5000  deallocate(r, Expon, CMO, FNor, ICent, IType, smat, scr1, scr2)
 5010  return

 !---------------------------------------------------------------------- private subroutines
 contains

 ! read basis function from the *.WFN file
 subroutine RdBas(iwfn,MaxAtm,MaxPG,NMO,NGauss,NAtom,r,Expon,ICent,IType,FNor,info)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: r(3,*),Expon(*),ICent(*),IType(*),FNor(*)

  info=0
  rewind(iwfn)
  read(iwfn,"(/,16x,i7,13x,i7,11x,i9)")NMO,NGauss,NAtom
  if(NAtom /= MaxAtm)then
    write(*,"(/,' ### Wrong! NAtom /= MaxAtm:',2i8)")NAtom,MaxAtm
    info=1
    goto 1000
  end if
  if(NGauss /= MaxPG)then
    write(*,"(/,' ### Wrong! NGauss /= MaxPG:',2i8)")NGauss,MaxPG
    info=1
    goto 1000
  end if

  do i=1,NAtom
    read(iwfn,"(24x,3f12.8)")r(1,i),r(2,i),r(3,i)
  end do

  read(iwfn,"(20x,20i3)")(ICent(i),i=1,NGauss)
  read(iwfn,"(20x,20i3)")(IType(i),i=1,NGauss)
  read(iwfn,"(10x,5d14.7)")(Expon(i),i=1,NGauss)

  ! compute normalization factors
  do i=1,NGauss
    FNor(i)=fnorm_lmn(Expon(i),IType(i))
  end do

  1000  return
 end subroutine RdBas

 ! read MO coefficients from the *.WFN file
 subroutine RdMO(iwfn,NGauss,Occ,CMO,FNor)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: CMO(*),FNor(*)

  read(iwfn,"(34x,f13.7)")Occ
  read(iwfn,"(5d16.8)")(CMO(i),i=1,NGauss)
  do i=1,NGauss
    CMO(i)=CMO(i)/FNor(i)
  end do

  return
 end subroutine RdMO

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! driver of overlap matrix
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine OvDriver(s,NGauss,IType,Expon,r,ICent,MaxL,scr1,scr2,info)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: s(*),IType(*),Expon(*),r(3,*),ICent(*),scr1(*),scr2(MaxL+1,*)

 info=0
 k=0
 loop1: do ig=1,NGauss
   ia = ICent(ig)
   do jg=1,ig
     ja = ICent(jg)
     k = k + 1
     call overlap(s(k),IType(ig),Expon(ig),r(1,ia),IType(jg),Expon(jg),r(1,ja),MaxL,scr1,scr2(1,1),scr2(1,2),scr2(1,5),info)
     if(info /= 0) exit loop1
   end do
 end do loop1

 return

 !---------------------------------------------------------------------- private subroutines
 contains

 ! calculate the overlap matrix element
 ! s = <itype1, a1, r1 | itype2, a2, r2>
 !   = <l1, m1, n1, a1, r1 | l2, m2, n2, a2, r2>
 subroutine overlap(s,itype1,a1,r1,itype2,a2,r2,MaxL,p,dgn,pan,pbn,info)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: r1(3),r2(3),p(3),dgn(0:MaxL),pan(0:MaxL,3),pbn(0:MaxL,3)

  pi=acos(-1.0d0)

 ! production of the normalization factors
  s=fnorm_lmn(a1,itype1)*fnorm_lmn(a2,itype2)

 ! gamma = as + bs and (2*gamma)^-n
  g=a1+a2
  dg=g+g
  dgn(0)=1.0d0
  do i=1,MaxL
    dgn(i)=dgn(i-1)/dg
  end do

  tmp1=dist(r1,r2)
  s=s*exp(-a1*a2*tmp1*tmp1/g)
  tmp1=sqrt(pi/g)
  s=s*tmp1*tmp1*tmp1

  ! P point
  p(1)=(a1*r1(1)+a2*r2(1))/g
  p(2)=(a1*r1(2)+a2*r2(2))/g
  p(3)=(a1*r1(3)+a2*r2(3))/g

  ! pa^n and pb^n
  do j=1,3
    pan(0,j)=1.0d0
    pbn(0,j)=1.0d0
  end do
  if(MaxL > 0)then
    do j=1,3
      pan(1,j)=p(j)-r1(j)
      pbn(1,j)=p(j)-r2(j)
    end do
    do i=2,MaxL
      do j=1,3
        pan(i,j)=pan(i-1,j)*pan(1,j)
        pbn(i,j)=pbn(i-1,j)*pbn(1,j)
      end do
    end do
  end if

  call pattwf(itype1,l1,m1,n1,info)
    if(info /= 0) goto 9000
  call pattwf(itype2,l2,m2,n2,info)
    if(info /= 0) goto 9000

  call integ(sx,MaxL,l1,l2,dgn,pan(0,1),pbn(0,1))
  call integ(sy,MaxL,m1,m2,dgn,pan(0,2),pbn(0,2))
  call integ(sz,MaxL,n1,n2,dgn,pan(0,3),pbn(0,3))
  s=s*sx*sy*sz

  9000  return
 end subroutine overlap

 !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 ! Obtain the xyz pattern for a given Cartesian type number. It is used in WFN and WFX.
 !
 ! 1  S     | 11 FXXX  | 21 GXXXX | 31 GXXZZ |
 ! 2  PX    | 12 FYYY  | 22 GYYYY | 32 GYYZZ |
 ! 3  PY    | 13 FZZZ  | 23 GZZZZ | 33 GXXYZ |
 ! 4  PZ    | 14 FXXY  | 24 GXXXY | 34 GXYYZ |
 ! 5  DXX   | 15 FXXZ  | 25 GXXXZ | 35 GXYZZ |
 ! 6  DYY   | 16 FYYZ  | 26 GXYYY |
 ! 7  DZZ   | 17 FXYY  | 27 GYYYZ |
 ! 8  DXY   | 18 FXZZ  | 28 GXZZZ |
 ! 9  DXZ   | 19 FYZZ  | 29 GYZZZ |
 ! 10 DYZ   | 20 FXYZ  | 30 GXXYY |
 !
 !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 subroutine pattwf(itype,l,m,n,info)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: PATDAT(3,56)
  data PATDAT/  &
 !              1          2          3          4          5              6          7          8          9         10
          0, 0, 0,   1, 0, 0,   0, 1, 0,   0, 0, 1,   2, 0, 0,       0, 2, 0,   0, 0, 2,   1, 1, 0,   1, 0, 1,   0, 1, 1,  &
 !             11         12         13         14         15             16         17         18         19         20
          3, 0, 0,   0, 3, 0,   0, 0, 3,   2, 1, 0,   2, 0, 1,       0, 2, 1,   1, 2, 0,   1, 0, 2,   0, 1, 2,   1, 1, 1,  &
 !             21         22         23         24         25             26         27         28         29         30
          4, 0, 0,   0, 4, 0,   0, 0, 4,   3, 1, 0,   3, 0, 1,       1, 3, 0,   0, 3, 1,   1, 0, 3,   0, 1, 3,   2, 2, 0,  &
 !             31         32         33         34         35             36         37         38         39         40
          2, 0, 2,   0, 2, 2,   2, 1, 1,   1, 2, 1,   1, 1, 2,       0, 0, 5,   0, 1, 4,   0, 2, 3,   0, 3, 2,   0, 4, 1,  &
 !             41         42         43         44         45             46         47         48         49         50
          0, 5, 0,   1, 0, 4,   1, 1, 3,   1, 2, 2,   1, 3, 1,       1, 4, 0,   2, 0, 3,   2, 1, 2,   2, 2, 1,   2, 3, 0,  &
 !             51         52         53         54         55             56
          3, 0, 2,   3, 1, 1,   3, 2, 0,   4, 0, 1,   4, 1, 0,       5, 0, 0/
  save PATDAT

  info = 0
  if(itype < 1 .or. itype > 56)then
    info = 1
    write(*,"(/,' ### Wrong!',' The Cartesian type number is out of range in sub. pattwf:',i3)")itype
    return
  end if

  l=PATDAT(1,itype)
  m=PATDAT(2,itype)
  n=PATDAT(3,itype)

  return
 end subroutine pattwf

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! generate an NBO-47 file.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine DrvNBO(inbo,fnbo,ver,dt,nbopro,  nat,iza,icore,xyz,  MaxL,lsph,nshell,ngto,ncar,nsph,mapatm,lqnm,  &
 nshlls,nshlln,expgto,congto,  nmotot,lalph,lbeta,ispin,ene,occup,carmo,sphmo,  iwbo,ifwbo,  ierr,ctmp) !
 implicit real(kind=8) (a-h,o-z)
 character*157     :: fnbo
 character*5       :: ver
 character*10      :: dt
 dimension         :: iza(*), icore(*), xyz(*), mapatm(*), lqnm(*), nshlls(*), nshlln(*), expgto(*), congto(*),  &
                      ispin(nmotot), ene(nmotot), occup(nmotot), carmo(ncar,nmotot), sphmo(nsph,nmotot)
 logical           :: ifwbo
 character*120     :: ctmp

 open(inbo,file=fnbo)

 if(lsph == 0) then
   call nbomain(inbo,ver,dt,nbopro,lalph,lbeta,  nat,iza,icore,xyz,  MaxL,lsph,nshell,ngto,ncar,mapatm,lqnm,nshlls,nshlln,  &
     expgto,congto,  nmotot,ispin,ene,occup,carmo,  iwbo,ifwbo,  ctmp,ierr)
 else
   call nbomain(inbo,ver,dt,nbopro,lalph,lbeta,  nat,iza,icore,xyz,  MaxL,lsph,nshell,ngto,nsph,mapatm,lqnm,nshlls,nshlln,  &
     expgto,congto,  nmotot,ispin,ene,occup,sphmo,  iwbo,ifwbo,  ctmp,ierr)
 end if
 if(ierr /= 0) return

 ! final step
 call finalnbo(fnbo,lsph,MaxL)

 return

 !---------------------------------------------------------------------- private subroutines
 contains

 !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 !
 ! generate NBO's *.47 data file.
 !
 !     Paremeters
 !     inbo           : port of 47 file
 !     ver,dt         : version number and date
 !     natm           : # atoms
 !     nshell         : # contracted shells
 !     nexp           : # primitive exponents
 !     nbas           : # contracted Cartesian (lsph=0) or spherical (lsph=1) basis functions
 !     MaxL           : max L (<=4 at present)
 !     ispin          : 1 (alpha) or 2 (beta) of each MO
 !     cmo            : MO oefficients
 !     ifwbo          : (.F./.T.) also calculate GWBO (MBO)
 !
 ! NOTE:
 !
 ! 1) It requires overlap and density in contracted Cartesian (lsph=0) or spherical (lsph=1) functions.
 !
 ! 2) T and V: optional but not calculated here. They are not defined in the case of all-electron scalar relativistic calculation.
 !
 !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 subroutine nbomain(inbo,ver,dt,nbopro,lalph,lbeta,  natm,iza,icore,xyz,  MaxL,lsph,nshell,nexp,nbas,mapatm,lqnm,nshlls,  &
  nshlln,expgto,congto,  nmo,ispin,ene,occup,cmo,  iwbo,ifwbo,  ctmp,ierr)
  implicit real(kind=8) (a-h,o-z)
  parameter(au2ang=0.529177249d0)
  logical           :: ifwbo, iffock, ifprmo
  character*5       :: ver
  character*10      :: dt
  character*120     :: ctmp
  dimension         :: iza(*), icore(*), xyz(3,*), mapatm(*), lqnm(*), nshlls(*), nshlln(*), expgto(*), congto(*), ispin(*),  &
                       ene(*), occup(*), cmo(nbas,nmo)
  allocatable       :: ncomp(:), nc(:), label(:), s2cd(:), s2cf(:), s2cg(:), s2ch(:), smat(:), pmat(:), scr1(:), scr2(:)

  iopen = 1         ! UHF or UKS?
  iffock = .false.  ! print Fock matrix?
  ifprmo = .false.  ! print MOs?
  if(nbopro == 1) then
    if(lbeta == 0 .and. nbas == nmo) then
      ifprmo = .true.
      call ChkSCF(iffock,nmo,iopen,occup,ene)
    else if(lalph == 1 .and. lbeta == 1 .and. nbas*2 == nmo) then
      na = 0
      nb = 0
      do imo=1,nmo
        if(ispin(imo) == 1) then
          na = na + 1
        else if(ispin(imo) == 2) then
          nb = nb + 1
        end if
      end do
      if(na == nbas .and. nb == nbas) then
        ifprmo = .true.
        iopen = 2
        call ChkSCF(iffock,nmo,iopen,occup,ene)
      end if
    end if
  end if

  rewind(inbo)

  ! $GENNBO
  if(iopen == 2) then
    write(inbo,"(' $GENNBO  NATOMS=',i4.4,' NBAS=',i5.5,' UPPER  BODM  OPEN  $END')") natm, nbas
  else
    write(inbo,"(' $GENNBO  NATOMS=',i4.4,' NBAS=',i5.5,' UPPER  BODM  $END')") natm, nbas
  end if
  ! $NBO
  write(inbo,"(' $NBO BNDIDX NLMO $END')")

  ! $COORD
  write(inbo,"(' $COORD')")
  call writitle(inbo,ver,dt,ctmp)
  ! About NBO6:
  ! 1. It may lead to numerical errors of about 1.0d-6 in the overlap matrix, which cannot pass the examination of NBO6. More
  !    digits should be printed.
  ! 2. au2ang: in NBO3, 0.529177249 is used instead of 0.529177. This may be also true in NBO6.
  do i=1,natm
    write(inbo,"(1x,2i5,3f18.9)")iza(i),iza(i)-icore(i),(xyz(j,i)*au2ang,j=1,3)
  end do
  write(inbo,"(' $END')")

  ntt=nbas*(nbas+1)/2
  nll = (MaxL+1)*(MaxL+2)/2
  nss = max(nll,nbas)
  nss = nss*nss
  allocate(ncomp(nshell), nc(nbas), label(nbas), s2cd(30), s2cf(70), s2cg(135), s2ch(231), smat(ntt), pmat(ntt*iopen),  &
    scr1(nss), scr2(nll*nll), stat=ierr)
    if(ierr /= 0) then
      write(iout,"(' ### Insufficient Memory in sub. nbomain!')")
      return
    end if

  ! set basis function centers and labels
  call GTFLab(nshell,lsph,mapatm,lqnm,ncomp,nc,label)

  ! $BASIS
  call wrbas(inbo,nbas,nc,label,ctmp)

  ! $CONTRACT
  call wrctr(inbo,nshell,nexp,MaxL,lsph,ncomp,nshlls,nshlln,expgto,congto,ctmp,scr1)

  ! $OVERLAP
  write(*,"(/,'  Calculating overlap matrix...')")
  call OvDriv2(inbo,nshell,nbas,MaxL,lsph,xyz,mapatm,lqnm,ncomp,nshlls,nshlln,expgto,congto,s2cd,s2cf,s2cg,s2ch,smat,  &
    scr1,scr2,ierr)
    if(ierr /= 0) return

  ! $DENSITY
  write(*,"('  Calculating density matrix...')")
  if(iopen == 2) then
    call DenDriO(inbo,nbas,ntt,nmo,ispin,occup,cmo,pmat)
  else
    call DenDriv(inbo,nbas,ntt,nmo,occup,cmo,pmat)
  end if

  ! $FOCK
  if(iffock) then
    call DrvFck(inbo,nbas,nmo,iopen,ispin,ene,cmo,smat,scr1,ierr)
    if(ierr /= 0) return
  end if

  ! $LCAOMO
  if(ifprmo) then
    call wrmos(inbo,nbas,nmo,iopen,ispin,cmo,scr1,ierr)
    if(ierr /= 0) return
  end if

  ! Generalized Wiberg bond order. S (in smat) and P (in pmat) will be destroyed.
  if(ifwbo) then
    write(*,"('  Calculating GWBO...')")
    if(iopen == 2) call AAdd(ntt,pmat,pmat(1+ntt),pmat)
    call GWBO(iwbo,nbas,natm,nc,smat,pmat,scr1,smat)
  end if

  deallocate(ncomp, nc, label, s2cd, s2cf, s2cg, s2ch, smat, pmat, scr1, scr2)

  return
 end subroutine nbomain

 ! print messages after saving the nbo-47 file
 subroutine finalnbo(fnbo,lsph,MaxL)
  implicit real(kind=8) (a-h,o-z)
  character*157     :: fnbo

  if(lsph == 1) then
    write(*,"(//,'  An NBO 47 file in spherical functions is generated successfully!',/,'  File Name = ',a)") trim(fnbo)
  else
    write(*,"(//,'  An NBO 47 file in Cartesian functions is generated successfully!',/,'  File Name = ',a)") trim(fnbo)
  end if

  if(MaxL > 4) then
    write(*,"(/,'  Warning: H-functions are not supported by NBO 3.0.')")
  else if(MaxL > 3) then
    write(*,"(/,'  Warning: G-functions are not fully supported by NBO 3.0.',/,  &
      '  Delete the $CONTRACT data block manually if NBO 3.0 is used.')")
  end if

  return
 end subroutine finalnbo

 ! Check whether the MOs are computed by R-/U-SCF
 subroutine ChkSCF(ifscf,nmo,iopen,occ,ene)
  implicit real(kind=8) (a-h,o-z)
  parameter(tol=1.0d-8)
  dimension         :: occ(nmo), ene(nmo)
  logical           :: ifscf

  ! iopen is 1 (RHF/RKS) or 2 (UHF/UKS)
  focc = dble(2/iopen)

  ifscf = .false.
  do imo = 1, nmo
    if(abs(occ(imo)) > tol .and. abs(occ(imo)-focc) > tol) return
  end do
  do imo = 1, nmo
    if(abs(ene(imo)) < tol) return
  end do
  ifscf = .true.

  return
 end subroutine ChkSCF

 ! Set basis function labels for NBO-47. The sp shells have been saved separately.
 subroutine GTFLab(nshell,lsph,mapatm,lqnm,ncomp,nc,label)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: mapatm(nshell), lqnm(nshell), ncomp(nshell), nc(*), label(*)
  ! Pattern in MOLDEN format:
  !  5D: D 0, D+1, D-1, D+2, D-2
  !      255  252  253  254  251
  !  6D: xx, yy, zz, xy, xz, yz
  !      201 204 206 202 203 205
  !  7F: F 0, F+1, F-1, F+2, F-2, F+3, F-3
  !      351  352  353  354  355  356  357
  ! 10F: xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
  !      301  307  310  304  302  303  306  309  308  305
  !  9G: G 0, G+1, G-1, G+2, G-2, G+3, G-3, G+4, G-4
  !      451  452  453  454  455  456  457  458  459
  ! 15G: xxxx yyyy zzzz xxxy xxxz yyyx yyyz zzzx zzzy xxyy xxzz yyzz xxyz yyxz zzxy
  !      401  411  415  402  403  407  412  410  414  404  406  413  405  408  409
  ! 11H: H 0, H+1, H-1, H+2, H-2, H+3, H-3, H+4, H-4, H+5, H-5
  !      551  552  553  554  555  556  557  558  559  560  561
  ! 21H: zzzzz yzzzz yyzzz yyyzz yyyyz yyyyy xzzzz xyzzz xyyzz xyyyz xyyyy xxzzz xxyzz xxyyz
  !      521   520   519   518   517   516   515   514   513   512   511   510   509   508
  !      xxyyy xxxzz xxxyz xxxyy xxxxz xxxxy xxxxx
  !      507   506   505   504   503   502   501

  !
  ! Labels used in NBO (see Section B.7.5 in the NBO manual)
  !
  ! s     x     y     z                                               !  4 * cart s, p
  ! 001   101   102   103
  !
  ! s     px    py    pz                                              !  4 * pure s, p
  ! 001   101   102   103
  !
  ! xx    xy    xz    yy    yz    zz                                  !  6 * cart d
  ! 201   202   203   204   205   206
  !
  ! d2-   d1+   d1-   d2+   d0                                        !  5 * pure d
  ! 251   252   253   254   255
  !
  ! xxx   xxy   xxz   xyy   xyz   xzz   yyy   yyz   yzz   zzz         ! 10 * cart f
  ! 301   302   303   304   305   306   307   308   309   310
  !
  ! f0    f1+   f1-   f2+   f2-   f3+   f3-                           !  7 * pure f
  ! 351   352   353   354   355   356   357
  !
  ! xxxx  xxxy  xxxz  xxyy  xxyz  xxzz  xyyy  xyyz  xyzz  xzzz        ! 15 * cart g
  ! yyyy  yyyz  yyzz  yzzz  zzzz
  ! 401   402   403   404   405   406   407   408   409   410
  ! 411   412   413   414   415
  !
  ! g0    g1+   g1-   g2+   g2-   g3+   g3-   g4+   g4-               !  9 * pure g
  ! 451   452   453   454   455   456   457   458   459
  !
  ! xxxxx xxxxy xxxxz xxxyy xxxyz xxxzz xxyyy xxyyz xxyzz xxzzz xyyyy ! 21 * cart h
  ! xyyyz xyyzz xyzzz xzzzz yyyyy yyyyz yyyzz yyzzz yzzzz zzzzz
  ! 501   502   503   504   505   506   507   508   509   510   511
  ! 512   513   514   515   516   517   518   519   520   521
  !
  ! h0    h1+   h1-   h2+   h2-   h3+   h3-   h4+   h4-   h5+   h5-   ! 11 * pure h
  ! 551   552   553   554   555   556   557   558   559   560   561
  !
  ! The real spherical harmonics are defined in
  ! http://en.wikipedia.org/wiki/Table_of_spherical_harmonics

  ibas = 0
  do ishell = 1, nshell
    if(lqnm(ishell) == 0)then
      ncomp(ishell) = 1
      label(ibas + 1) = 001
    else if(lqnm(ishell) == 1)then
      ncomp(ishell) = 3
      label(ibas + 1) = 101
      label(ibas + 2) = 102
      label(ibas + 3) = 103
    else if(lqnm(ishell) == 2)then
      if(lsph == 0) then
        ncomp(ishell) = 6
        label(ibas + 1) = 201
        label(ibas + 2) = 204
        label(ibas + 3) = 206
        label(ibas + 4) = 202
        label(ibas + 5) = 203
        label(ibas + 6) = 205
      else
        ncomp(ishell) = 5
        label(ibas + 1) = 255
        label(ibas + 2) = 252
        label(ibas + 3) = 253
        label(ibas + 4) = 254
        label(ibas + 5) = 251
      end if
    else if(lqnm(ishell) == 3)then
      if(lsph == 0) then
        ncomp(ishell) = 10
        label(ibas + 1) = 301
        label(ibas + 2) = 307
        label(ibas + 3) = 310
        label(ibas + 4) = 304
        label(ibas + 5) = 302
        label(ibas + 6) = 303
        label(ibas + 7) = 306
        label(ibas + 8) = 309
        label(ibas + 9) = 308
        label(ibas +10) = 305
      else
        ncomp(ishell) = 7
        label(ibas + 1) = 351
        label(ibas + 2) = 352
        label(ibas + 3) = 353
        label(ibas + 4) = 354
        label(ibas + 5) = 355
        label(ibas + 6) = 356
        label(ibas + 7) = 357
      end if
    else if(lqnm(ishell) == 4)then
      if(lsph == 0) then
        ncomp(ishell) = 15
        label(ibas + 1) = 401
        label(ibas + 2) = 411
        label(ibas + 3) = 415
        label(ibas + 4) = 402
        label(ibas + 5) = 403
        label(ibas + 6) = 407
        label(ibas + 7) = 412
        label(ibas + 8) = 410
        label(ibas + 9) = 414
        label(ibas +10) = 404
        label(ibas +11) = 406
        label(ibas +12) = 413
        label(ibas +13) = 405
        label(ibas +14) = 408
        label(ibas +15) = 409
      else
        ncomp(ishell) = 9
        label(ibas + 1) = 451
        label(ibas + 2) = 452
        label(ibas + 3) = 453
        label(ibas + 4) = 454
        label(ibas + 5) = 455
        label(ibas + 6) = 456
        label(ibas + 7) = 457
        label(ibas + 8) = 458
        label(ibas + 9) = 459
      end if
    else if(lqnm(ishell) == 5)then
      if(lsph == 0) then
        ncomp(ishell) = 21
        label(ibas + 1) = 521
        label(ibas + 2) = 520
        label(ibas + 3) = 519
        label(ibas + 4) = 518
        label(ibas + 5) = 517
        label(ibas + 6) = 516
        label(ibas + 7) = 515
        label(ibas + 8) = 514
        label(ibas + 9) = 513
        label(ibas +10) = 512
        label(ibas +11) = 511
        label(ibas +12) = 510
        label(ibas +13) = 509
        label(ibas +14) = 508
        label(ibas +15) = 507
        label(ibas +16) = 506
        label(ibas +17) = 505
        label(ibas +18) = 504
        label(ibas +19) = 503
        label(ibas +20) = 502
        label(ibas +21) = 501
      else
        ncomp(ishell) = 11
        label(ibas + 1) = 551
        label(ibas + 2) = 552
        label(ibas + 3) = 553
        label(ibas + 4) = 554
        label(ibas + 5) = 555
        label(ibas + 6) = 556
        label(ibas + 7) = 557
        label(ibas + 8) = 558
        label(ibas + 9) = 559
        label(ibas +10) = 560
        label(ibas +11) = 561
      end if
    else if(lqnm(ishell) == 6)then
      write(*,"(/,' *** Error! LQ > 5 in sub. GTFLab.')")
      call estop(1)
    end if
    nc(ibas+1:ibas+ncomp(ishell)) = mapatm(ishell)
    ibas = ibas + ncomp(ishell)
  end do

  return
 end subroutine GTFLab

 ! count the number of lines
 subroutine lines(n1,n2,nline,last)
  implicit real(kind=8) (a-h,o-z)

  nline=n1/n2
  last=n2
  if(mod(n1,n2) /= 0)then
    last=mod(n1,n2)
    nline=nline+1
  end if

  return
 end subroutine lines

 ! write integer array in the format a, n*i
 subroutine iwrite(tag1,tag2,io,nline,last,nstep,ivec)
  implicit real(kind=8) (a-h,o-z)
  character*10      :: tag1,tag2,ac
  dimension         :: ivec(*)

  idx=0
  do i=1,nline
    if(i == 1)then
      ac=tag1
    else
      ac=tag2
    end if
    if(i < nline)then
      write(io,1110)ac,(ivec(j),j=idx+1,idx+nstep)
      idx=idx+nstep
    else
      write(io,1110)ac,(ivec(j),j=idx+1,idx+last)
    end if
  end do

  return
  1110  format(a10,10i7)
 end subroutine iwrite

 ! write $BASIS in *.47
 subroutine wrbas(inbo,nbas,nc,label,tag)
  implicit real(kind=8) (a-h,o-z)
  character*10      :: tag(3)
  dimension         :: nc(nbas),label(nbas)

  tag = (/'  CENTER =','   LABEL =','          '/)

  write(inbo,"(' $BASIS')")

  nstep=10
  call lines(nbas,nstep,nline,last)
  ! CENTER
  call iwrite(tag(1),tag(3),inbo,nline,last,nstep,nc)
  ! LABEL
  call iwrite(tag(2),tag(3),inbo,nline,last,nstep,label)

  write(inbo,"(' $END')")

  return
 end subroutine wrbas

 ! write $CONTRACT in *.47
 subroutine wrctr(inbo,nshell,nexp,MaxL,lsph,ncomp,nshlls,nshlln,expgto,congto,tag,tmp)
  implicit real(kind=8) (a-h,o-z)
  character*10      :: tag(11)
  dimension ncomp(nshell),nshlls(nshell),nshlln(nshell),expgto(nexp),congto(nexp),tmp(nexp)

  tag = (/'   NCOMP =','   NPRIM =','    NPTR =','          ','     EXP =',  &
          '      CS =','      CP =','      CD =','      CF =','      CG =','      CH ='/)

  write(inbo,"(' $CONTRACT')")
  write(inbo,"('  NSHELL =',i7)")nshell
  write(inbo,"('    NEXP =',i7)")nexp

  nstep=10
  call lines(nshell,nstep,nline,last)
  ! NCOMP
  call iwrite(tag(1),tag(4),inbo,nline,last,nstep,ncomp)
  ! NPRIM
  call iwrite(tag(2),tag(4),inbo,nline,last,nstep,nshlln)
  ! NPTR
  call iwrite(tag(3),tag(4),inbo,nline,last,nstep,nshlls)

  nstep=4
  call lines(nexp,nstep,nline,last)
  ! EXP
  call fwrite(tag(5),tag(4),inbo,nline,last,nstep,expgto)
  ! CS
  nl = 1
  call cpcon('S',nl,nshell,nexp,ncomp,nshlln,expgto,congto,tmp)
  call fwrite(tag(6),tag(4),inbo,nline,last,nstep,tmp)
  ! CP
  if(MaxL >= 1)then
    nl = 3
    call cpcon('P',nl,nshell,nexp,ncomp,nshlln,expgto,congto,tmp)
    call fwrite(tag(7),tag(4),inbo,nline,last,nstep,tmp)
  end if
  ! CD
  if(MaxL >= 2)then
    nl = 5
    if(lsph == 0) nl = 6
    call cpcon('D',nl,nshell,nexp,ncomp,nshlln,expgto,congto,tmp)
    call fwrite(tag(8),tag(4),inbo,nline,last,nstep,tmp)
  end if
  ! CF
  if(MaxL >= 3)then
    nl = 7
    if(lsph == 0) nl = 10
    call cpcon('F',nl,nshell,nexp,ncomp,nshlln,expgto,congto,tmp)
    call fwrite(tag(9),tag(4),inbo,nline,last,nstep,tmp)
  end if
  ! CG
  if(MaxL >= 4)then
    nl = 9
    if(lsph == 0) nl = 15
    call cpcon('G',nl,nshell,nexp,ncomp,nshlln,expgto,congto,tmp)
    call fwrite(tag(10),tag(4),inbo,nline,last,nstep,tmp)
  end if
  ! CH
  if(MaxL >= 5)then
    nl =11
    if(lsph == 0) nl = 21
    call cpcon('H',nl,nshell,nexp,ncomp,nshlln,expgto,congto,tmp)
    call fwrite(tag(11),tag(4),inbo,nline,last,nstep,tmp)
  end if
  if(MaxL >= 6)then
    nl =11
    write(*,"(/,' *** Error! LQ > 5 in sub. wrctr.')")
    call estop(1)
  end if

  write(inbo,"(' $END')")

  return
 end subroutine wrctr

 !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 !
 ! Obtain the xyz pattern for a given Cartesian type number. It is used in MOLDEN.
 ! Note: the ordering 14-19 of F functions is different from the one in subroutine pattwf.
 ! 1  S     | 11 FXXX  | 21 GXXXX | 31 GXXZZ |
 ! 2  PX    | 12 FYYY  | 22 GYYYY | 32 GYYZZ |
 ! 3  PY    | 13 FZZZ  | 23 GZZZZ | 33 GXXYZ |
 ! 4  PZ    | 14 FXYY  | 24 GXXXY | 34 GXYYZ |
 ! 5  DXX   | 15 FXXY  | 25 GXXXZ | 35 GXYZZ |
 ! 6  DYY   | 16 FXXZ  | 26 GXYYY |
 ! 7  DZZ   | 17 FXZZ  | 27 GYYYZ |
 ! 8  DXY   | 18 FYZZ  | 28 GXZZZ |
 ! 9  DXZ   | 19 FYYZ  | 29 GYZZZ |
 ! 10 DYZ   | 20 FXYZ  | 30 GXXYY |
 !
 ! Ordering of G functions in Gaussian. For debug only.
 ! 21 GZZZZ | 31 GXXYZ |
 ! 22 GYZZZ | 32 GXXYY |
 ! 23 GYYZZ | 33 GXXXZ |
 ! 24 GYYYZ | 34 GXXXY |
 ! 25 GYYYY | 35 GXXXX |
 ! 26 GXZZZ |
 ! 27 GXYZZ |
 ! 28 GXYYZ |
 ! 29 GXYYY |
 ! 30 GXXZZ |
 !
 ! Ordering of H functions in Gaussian, WFX, and MultiWFN are the same.
 !
 !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 subroutine pattml(itype,l,m,n,info)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: PATDAT(3,56)
  data PATDAT/  &
  !              1          2          3          4          5              6          7          8          9         10
           0, 0, 0,   1, 0, 0,   0, 1, 0,   0, 0, 1,   2, 0, 0,       0, 2, 0,   0, 0, 2,   1, 1, 0,   1, 0, 1,   0, 1, 1,  &
  !             11         12         13         14         15             16         17         18         19         20
           3, 0, 0,   0, 3, 0,   0, 0, 3,   1, 2, 0,   2, 1, 0,       2, 0, 1,   1, 0, 2,   0, 1, 2,   0, 2, 1,   1, 1, 1,  &
  !             21         22         23         24         25             26         27         28         29         30
           4, 0, 0,   0, 4, 0,   0, 0, 4,   3, 1, 0,   3, 0, 1,       1, 3, 0,   0, 3, 1,   1, 0, 3,   0, 1, 3,   2, 2, 0,  &
  !             31         32         33         34         35             36         37         38         39         40
           2, 0, 2,   0, 2, 2,   2, 1, 1,   1, 2, 1,   1, 1, 2,       0, 0, 5,   0, 1, 4,   0, 2, 3,   0, 3, 2,   0, 4, 1,  &
  !             41         42         43         44         45             46         47         48         49         50
           0, 5, 0,   1, 0, 4,   1, 1, 3,   1, 2, 2,   1, 3, 1,       1, 4, 0,   2, 0, 3,   2, 1, 2,   2, 2, 1,   2, 3, 0,  &
  !             51         52         53         54         55             56
           3, 0, 2,   3, 1, 1,   3, 2, 0,   4, 0, 1,   4, 1, 0,       5, 0, 0/
  !<<< Gaussian (see also function fnorm_lmn)
  !             21         22         23         24         25             26         27         28         29         30
  !        0, 0, 4,   0, 1, 3,   0, 2, 2,   0, 3, 1,   0, 4, 0,       1, 0, 3,   1, 1, 2,   1, 2, 1,   1, 3, 0,   2, 0, 2,  &
  !             31         32         33         34         35
  !        2, 1, 1,   2, 2, 0,   3, 0, 1,   3, 1, 0,   4, 0, 0/
  !>>>
  save PATDAT

  info = 0
  if(itype < 1 .or. itype > 56)then
    info = 1
    write(*,"(/,' ### Wrong!',' The Cartesian type number is out of range in sub. pattml:',i3)")itype
    return
  end if

  l=PATDAT(1,itype)
  m=PATDAT(2,itype)
  n=PATDAT(3,itype)

  return
 end subroutine pattml

 ! overlap integrals for primitive Cartesian GTO_i and GTO_j
 subroutine poverlap(s6d, as,npta,ioffst,ra, bs,nptb,joffst,rb,pi,MaxL,rab,info)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: s6d(nptb,npta),ra(3),rb(3)

  do icat=1,npta
    ipat=icat+ioffst
    ! normalization factor
    fna=fnorm_lmn(as,ipat)
    call pattml(ipat,l1,m1,n1,info)
      if(info /= 0) goto 9000
    do jcat=1,nptb
      jpat=jcat+joffst
      ! normalization factor
      fnb=fnorm_lmn(bs,jpat)
      call pattml(jpat,l2,m2,n2,info)
        if(info /= 0) goto 9000
      call overlap2(s6d(jcat,icat),  l1,m1,n1,as,ra,fna,  l2,m2,n2,bs,rb,fnb,  pi,MaxL,rab)
    end do
  end do

  9000  return
 end subroutine poverlap

 ! calculate an overlap matrix element  <as(ra);l1,m1,n1 | bs(rb);l2,m2,n2>
 ! another version
 subroutine overlap2(s,  l1,m1,n1,as,ra,fna,  l2,m2,n2,bs,rb,fnb,  pi,MaxL,rab)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: ra(3),rb(3),p(3),dgn(0:MaxL),pan(0:MaxL,3),pbn(0:MaxL,3)

  ! gamma = as + bs and (2*gamma)^-n
  g=as+bs
  dg=g+g
  dgn(0)=1.d0
  do i=1,MaxL
    dgn(i)=dgn(i-1)/dg
  end do
  tmp1=sqrt(pi/g)

  s=fna*fnb*exp(-as*bs*rab*rab/g)*tmp1*tmp1*tmp1

 ! P point
  p(1)=(as*ra(1)+bs*rb(1))/g
  p(2)=(as*ra(2)+bs*rb(2))/g
  p(3)=(as*ra(3)+bs*rb(3))/g

 ! pa^n and pb^n
  do j=1,3
    pan(0,j)=1.d0
    pbn(0,j)=1.d0
  end do
  if(MaxL > 0)then
    do j=1,3
      pan(1,j)=p(j)-ra(j)
      pbn(1,j)=p(j)-rb(j)
    end do
    do i=2,MaxL
      do j=1,3
        pan(i,j)=pan(i-1,j)*pan(1,j)
        pbn(i,j)=pbn(i-1,j)*pbn(1,j)
      end do
    end do
  end if

  call integ(sx,MaxL,l1,l2,dgn,pan(0,1),pbn(0,1))
  call integ(sy,MaxL,m1,m2,dgn,pan(0,2),pbn(0,2))
  call integ(sz,MaxL,n1,n2,dgn,pan(0,3),pbn(0,3))
  s=s*sx*sy*sz

  9000  return
 end subroutine overlap2

 ! put overlap integrals of ci*cj*<IGTO|JGTO> into overlap matrix S.
 ! IBS and JBS are row and column indices which have been finished already.
 subroutine PackLT(S,IBS,JBS,si,nspa,nspb,LT)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: S(*),si(nspb,nspa)
  logical           :: LT

  do i=1,nspa
    ii=IBS+i
    jj=JBS
    kk=IdxNTT(jj,ii)
    nspb2=nspb
    if(LT) nspb2=i
    do j=1,nspb2
      kk=kk+1
      S(kk)=S(kk)+si(j,i)
    end do
  end do

  return
 end subroutine PackLT

 ! driver of overlap matrix (contracted and spherical)
 subroutine OvDriv2(inbo,nshell,nbas,MaxL,lsph,xyz,mapatm,lqnm,ncomp,nshlls,nshlln,expgto,congto,s2cd,s2cf,s2cg,s2ch,SMat,  &
  sblk,scr,info)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: xyz(3,*),mapatm(nshell),lqnm(nshell),ncomp(nshell),nshlls(nshell),nshlln(nshell),expgto(*),congto(*),  &
                       s2cd(*),s2cf(*),s2cg(*),s2ch(*),SMat(*),sblk(*),scr(*)

  pi=acos(-1.d0)
  NTTS=nbas*(nbas+1)/2
  call AClear(NTTS,SMat)

  if(lsph /= 0 .and. MaxL > 1) call s2c_coef(MaxL,s2cd,s2cf,s2cg,s2ch,ierr)
    if(ierr == 1) goto 9000

  IBS=0
  do ishell = 1, nshell
    ityp=lqnm(ishell)              ! ityp & jtyp: 0 (s), 1 (p), 2 (d), ...
    nspa=ncomp(ishell)             ! nspa & nspb: #Cart./sph. functions (depending on lsph) in a shell
    npta=(ityp+1)*(ityp+2)/2       ! npta & nptb: #Cart. functions in a shell
    nofa=ityp*(ityp+1)*(ityp+2)/6  ! nofa & nofb: offset of ityp/jtyp, = 0 (s), 1 (p), 4 (d),10 (f), 20(g)
    i1 = nshlls(ishell)
    i2 = i1 + nshlln(ishell) - 1
    JBS=0
    do jshell = 1, ishell
      jtyp=lqnm(jshell)
      nspb=ncomp(jshell)
      nptb=(jtyp+1)*(jtyp+2)/2
      nofb=jtyp*(jtyp+1)*(jtyp+2)/6
      j1 = nshlls(jshell)
      j2 = j1 + nshlln(jshell) - 1

      rab = dist(xyz(1,mapatm(ishell)),xyz(1,mapatm(jshell)))
      ! for contracted shells ishell and jshell
      do igto = i1,i2
        ags = expgto(igto)
        acn = congto(igto)
        do jgto = j1,j2
          bgs = expgto(jgto)
          bcn = congto(jgto)
          ! for primitive Cartesian GTO_i and GTO_j: sblk(nptb,npta)
          call poverlap(sblk,ags,npta,nofa,xyz(1,mapatm(ishell)), bgs,nptb,nofb,xyz(1,mapatm(jshell)),pi,MaxL,rab,info)
            if(info /= 0) goto 9000
          ! for primitive spherical GTO_i and GTO_j: sblk(nspb,nspa)
          if(lsph /= 0 .and. (ityp > 1 .or. jtyp > 1)) call car2sph_blk(sblk,scr,npta,nptb,nspa,nspb,s2cd,s2cf,s2cg,s2ch)
          ! do contraction for sblk(nspb,nspa)
          call AScale(nspa*nspb,acn*bcn,sblk,sblk)
          call PackLT(SMat,IBS,JBS,sblk,nspa,nspb,ishell==jshell)
        end do
      end do

      JBS=JBS+nspb
    end do
    IBS=IBS+nspa
  end do

  call noiserm(SMat,NTTS)
  write(inbo,"(' $OVERLAP')")
  call wrarray(inbo,NTTS,SMat)
  write(inbo,"(' $END')")

  9000  return
 end subroutine OvDriv2

 ! driver of density matrix calculation (UHF): alpha=1 and beta=2
 subroutine DenDriO(inbo,nbas,NTTS,nmo,ispin,occ,cmo,Den)
  implicit real(kind=8) (a-h,o-z)
  parameter(occtol=1.d-12)
  dimension         :: ispin(nmo), occ(nmo), cmo(nbas,nmo), Den(NTTS,2)

  call AClear(NTTS*2,Den)

  do i=1,nmo
    if(abs(occ(i)) < occtol) cycle

    ! density matrix
    j=0
    do ibs=1,nbas
      do jbs=1,ibs
        j=j+1
        Den(j,ispin(i))=Den(j,ispin(i))+occ(i)*cmo(ibs,i)*cmo(jbs,i)
      end do
    end do
  end do

  call noiserm(Den,NTTS*2)
  write(inbo,"(' $DENSITY')")
  call wrarray(inbo,NTTS,Den(1,1))
  call wrarray(inbo,NTTS,Den(1,2))
  write(inbo,"(' $END')")

  return
 end subroutine DenDriO

 ! driver of density matrix calculation (RHF)
 subroutine DenDriv(inbo,nbas,NTTS,nmo,occ,cmo,Den)
  implicit real(kind=8) (a-h,o-z)
  parameter(occtol=1.d-12)
  dimension         :: occ(nmo), cmo(nbas,nmo), Den(*)

  call AClear(NTTS,Den)

  do i=1,nmo
    if(abs(occ(i)) < occtol) cycle

    ! density matrix
    j=0
    do ibs=1,nbas
      do jbs=1,ibs
        j=j+1
        Den(j)=Den(j)+occ(i)*cmo(ibs,i)*cmo(jbs,i)
      end do
    end do
  end do

  call noiserm(Den,NTTS)
  write(inbo,"(' $DENSITY')")
  call wrarray(inbo,NTTS,Den)
  write(inbo,"(' $END')")

  return
 end subroutine DenDriv

 ! Driver of Fock matrix calculation. F = (S C) E (S C)'
 subroutine DrvFck(inbo,nbas,nmo,iopen,ispin,ene,cmo,slt,smat,ierr)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: ispin(nmo), ene(nmo), cmo(nbas,nmo), slt(*), smat(nbas,nbas)
  allocatable       :: scr1(:,:), scr2(:,:), scr3(:)

  ntt = nbas*(nbas+1)/2
  nss = nbas*nbas

  allocate(scr1(nbas,nbas), scr2(nbas,nbas), scr3(nss), stat=ierr)
    if(ierr /= 0) then
      write(iout,"(' ### Insufficient Memory in sub. DrvFck!')")
      return
    end if

  ierr = 0
  call LT2Sqr(nbas,slt,smat)

  write(inbo,"(' $FOCK')")

  do k = 1, iopen
    scr1 = 0.0d0
    j = 0
    do i=1,nmo
      if(ispin(i) /= k) cycle
      j = j + 1
      scr1(:,j) = cmo(:,i)
      scr3(j) = ene(j)
    end do
    if(j /= nbas) then
      write(*,"(/,' ### Error! NBas is wrong in sub. DrvFck.')")
      ierr = 1
      return
    end if
    ! S * C --> scr2
    call MatMult(1,nbas,nbas,nbas,smat,scr1,scr2)
    ! S * C * E --> scr1
    do i = 1, nbas
      scr1(:,i) = scr2(:,i) * scr3(i)
    end do
    call MatMult(3,nbas,nbas,nbas,scr1,scr2,scr3)
    call Sqr2Tr(nbas,scr3,scr1)
    call wrarray(inbo,ntt,scr1)
  end do

  write(inbo,"(' $END')")
  deallocate(scr1, scr2, scr3)

  return
 end subroutine DrvFck

 ! write an array in the NBO-47 format
 subroutine wrarray(inbo,n,a)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: a(n)

  write(inbo,"(2x,5e15.7e2)") (a(idx),idx=1,n)

  return
 end subroutine wrarray

 ! Print MOs. It requires nbas = nmo (iopen=1) or 2*nbas = nmo (iopen=2).
 subroutine wrmos(inbo,nbas,nmo,iopen,ispin,cmo,scr,ierr)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: ispin(nmo), cmo(nbas,nmo), scr(nbas,nbas)

  ierr = 0

  write(inbo,"(' $LCAOMO')")

  do k = 1, iopen
    scr = 0.0d0
    j = 0
    do i=1,nmo
      if(ispin(i) /= k) cycle
      j = j + 1
      scr(:,j) = cmo(:,i)
    end do
    if(j /= nbas) then
      write(*,"(/,' ### Error! NBas is wrong in sub. wrmos.')")
      ierr = 1
      return
    end if
    call wrarray(inbo,nbas*nbas,scr)
  end do

  write(inbo,"(' $END')")

  return
 end subroutine wrmos

 ! Generalized Wiberg bond order index, which is Mayer's bond order in the case of closed-shell system.
 subroutine GWBO(iwbo,N,Natm,icent,S,P,D,bo)
  implicit real(kind=8) (a-h,o-z)
  parameter(NCol=10)
  dimension         :: icent(*),S(*),P(*),D(N,N),bo(Natm,Natm)

  ! D = P * S
  call AClear(N*N,D)
  call LTxLT(N,P,S,D)

  ! D = D .* D^T
  do i = 1, N
    do j = 1, i-1
      D(j,i) = D(i,j) * D(j,i)
      ! D(i,j) = D(j,i)    ! not used
    end do
  end do

  call AClear(Natm*Natm,bo)
  do i = 1, N
    ii = icent(i)
    do j = 1, i-1
      jj = icent(j)
      if(ii /= jj)then
        bo(jj,ii) = bo(jj,ii) + D(j,i)
        ! bo(ii,jj) = bo(jj,ii)    ! not used
        bo(ii,ii) = bo(ii,ii) + D(j,i)
        bo(jj,jj) = bo(jj,jj) + D(j,i)
      end if
    end do
  end do

  rewind(iwbo)
  write(iwbo,"(  &
    ' Generalized Wiberg bond order indices in MO.',//,  &
    '   E(i,i) = Total WBO of atom i, and',/,  &
    '   E(i,j) = WBO between atoms i and j.',/)")

  NBlock=(Natm-1)/NCol+1
  do i=1,NBlock
    iv1=(i-1)*NCol+1
    iv2=min(i*NCol,Natm)
    write(iwbo,"(/,' I_atm        ',10i8)") (k,k=iv1,iv2)
    write(iwbo,*)
    do j=iv1,Natm
      write(iwbo,"(i6,8x,10f8.4)")j,(bo(k,j),k=iv1,min(iv2,j))
    end do
  end do

  return
 end subroutine GWBO

 ! write real array in the format a, n*f
 subroutine fwrite(tag1,tag2,io,nline,last,nstep,fvec)
  implicit real(kind=8) (a-h,o-z)
  character*10      :: tag1,tag2,ac
  dimension         :: fvec(*)

  idx=0
  do i=1,nline
    if(i == 1)then
      ac=tag1
    else
      ac=tag2
    end if
    if(i < nline)then
      write(io,1110)ac,(fvec(j),j=idx+1,idx+nstep)
      idx=idx+nstep
    else
      write(io,1110)ac,(fvec(j),j=idx+1,idx+last)
    end if
  end do

  return
  1110  format(a10,4e16.7)
 end subroutine fwrite

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! generate a wfx file.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine genwfx(iwfx,fwfx,ver,dt,  iedf,lecp,iedftyp,chanet,ntote,nat,iza,icore,xyz,  &
 MaxL,nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  &
 ncarp,ncarc,nmotot,nmo,lalph,lbeta,ispin,ene,occup,lprtmo,carmo,  iunknw,ctmp)
 implicit real(kind=8) (a-h,o-z)
 character*157     :: fwfx
 character*5       :: ver
 character*10      :: dt
 dimension         :: iza(*), icore(*), xyz(*), mapatm(*), lqnm(*), nshlls(*), nshlln(*), expgto(*), congto(*),  &
                      ispin(*), ene(*), occup(*), lprtmo(*), carmo(ncarc,*)
 character*100     :: ctmp

 call edfmain(iedf,lecp,iedftyp,nat,nedf,iza,icore)

 open(iwfx,file=fwfx)

 call wfxmain(iwfx,ver,dt,  nat,chanet,ntote,iedf,lecp,nedf,iza,icore,xyz,  &
   nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  &
   ncarp,ncarc,nmotot,nmo,lalph,lbeta,ispin,ene,occup,lprtmo,carmo,  iunknw,ctmp)

 ! final step
 call finalwfx(fwfx,lecp,iunknw,MaxL)

 return

 !---------------------------------------------------------------------- private subroutines
 contains

 !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 !
 ! Save a wavefunction file in wfx format.
 !
 !     Note:
 !
 !     1. Fortran D (or d) descriptor for numbers is NOT allowed.
 !
 !     2. All data must be in atomic units.
 !
 !     3. Comment lines start with the # character must NOT be used inside
 !        data sections, i.e., between opening and closing tags.
 !
 !     4. MO should be in the order doubly, singly Alpha, singly Beta.
 !        For RO-SCF, this means all doubly occupied MOs, then all singly
 !        occupied Alpha MOs.
 !        For U-SCF or post-SCF, this means all Alpha MOs then all Beta
 !        MOs.
 !        Within a block of MOs, the SCF MOs should be in the order of
 !        increasing MO energy
 !        Within a block of post-SCF natural MOs, they should be in the
 !        order of decreasing occupancy.
 !
 !        >>>>>>>>>>>>>>>>>>>>>>>>>> Not considered at present!
 !
 !     5. Non-nuclear attractors can be added to the .wfx file as nuclei
 !        with atomic number 0, nuclear charge 0.0 and with names begin-
 !        ning with NNA (e.g., NNA8).
 !        and
 !        Ghost atoms can be added to the .wfx file as nuclei with atomic
 !        number 0, nuclear charge 0.0 and with names beginning with Bq
 !        (e.g., Bq12).  The coordinates of ghost nuclei must appear in
 !        the appropriate position of the <Nuclear Cartesian Coordinates>
 !        section of the .wfx file.
 !
 !        >>>>>>>>>>>>>>>>>>>>>>>>>> Not considered at present!
 !
 !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 subroutine wfxmain(iwfx,ver,dt,  nat,chanet,ntote,iedf,lecp,nedf,iza,icore,xyz,  &
   nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  &
   ncarp,ncarc,nmotot,nmo,lalph,lbeta,ispin,ene,occup,lprtmo,carmo,  iunknw,ctmp)
  parameter(enemax=9999.d0)
  implicit real(kind=8) (a-h,o-z)
  character*10      :: dt
  character*5       :: ver
  character*100     :: ctmp
  dimension         :: iza(*), icore(*), xyz(3,*), mapatm(*), lqnm(*), nshlls(*), nshlln(*), expgto(*), congto(*),  &
                       ispin(*), ene(*), occup(*), lprtmo(*), carmo(ncarc,*)
  allocatable       :: expg(:), conf(:), ityp(:), icmo(:), cn(:)
  logical           :: prtspn

  rewind(iwfx)

  ! Title
  call wfxlab(iwfx,0,"Title")
  call writitle(iwfx,ver,dt,ctmp)
  call wfxlab(iwfx,1,"Title")

  ! Keywords
  call wfxlab(iwfx,0,"Keywords")
  write(iwfx,"(' GTO')")
  call wfxlab(iwfx,1,"Keywords")

  ! Number of Nuclei
  call wfxlab(iwfx,0,"Number of Nuclei")
  write(iwfx,"(i8)")nat
  call wfxlab(iwfx,1,"Number of Nuclei")

  ! Number of Occupied Molecular Orbitals
  call wfxlab(iwfx,0,"Number of Occupied Molecular Orbitals")
  write(iwfx,"(i8)")nmo
  call wfxlab(iwfx,1,"Number of Occupied Molecular Orbitals")

  ! Number of Perturbations
  call wfxlab(iwfx,0,"Number of Perturbations")
  write(iwfx,"(i8)")0
  call wfxlab(iwfx,1,"Number of Perturbations")

  ! Net Charge
  call wfxlab(iwfx,0,"Net Charge")
  write(iwfx,"(i8)")nint(chanet)
  call wfxlab(iwfx,1,"Net Charge")

  ! Number of Electrons (Core Electrons by ECP are excluded)
  call wfxlab(iwfx,0,"Number of Electrons")
  write(iwfx,"(i8)")ntote
  call wfxlab(iwfx,1,"Number of Electrons")

  ! it works only when Beta exists
  prtspn=.false.
  if(lalph == 1 .and. lbeta == 1) then
    elea = 0.d0
    eleb = 0.d0
    do i=1,nmotot
      if(lprtmo(i) < 1) cycle
      if(ispin(i) == 1) then
        elea = elea + occup(i)
      else if(ispin(i) == 2) then
        eleb = eleb + occup(i)
      end if
    end do
    if(abs(elea+eleb-dble(ntote)) < 0.01d0 .and.  &
       abs(elea-dble(nint(elea))) < 0.01d0 .and.  &
       abs(eleb-dble(nint(eleb))) < 0.01d0) prtspn = .true.
  end if

  if(prtspn) then
    nelea = nint(elea)
    neleb = nint(eleb)
    MS = nelea - neleb + 1
  else
    ! Read in Spin Multiplicity, Numbers of Alpha and Beta electrons
    do while(.true.)
      write(*,"(/,'  Type in the Spin Multiplicity:',/,  &
        '  (default: 1 for even- and 2 for odd-number of electron system)',/,' > ',$)")
      read(*,"(a10)",err=50) ctmp
      if(len_trim(ctmp(1:10)) == 0) then
      	MS = 1
      	if(mod(ntote,2) == 1) MS = 2
      else
        read(ctmp(1:10),*,err=50) MS
      end if

      if(MS < 1 .or. MS > ntote+1) then
        write(*,"(/,'   MS is out of range! Try again.')")
        cycle
      else if(mod(ntote,2) == 0 .and. mod(MS,2) == 0) then
        write(*,"(/,'   MS must be an odd number! Try again.')")
        cycle
      else if(mod(ntote,2) == 1 .and. mod(MS,2) == 1) then
        write(*,"(/,'   MS must be an even number! Try again.')")
        cycle
      else
        exit
      end if

      50  write(*,"(/,'   Error when reading MS! Try again.')")
      cycle
    end do
    neleb = (ntote + 1 - MS) / 2
    nelea = ntote - neleb
  end if
  write(*,"(/,'  Spin Multiplicity:', i4)") MS

  ! Number of Alpha Electrons
  call wfxlab(iwfx,0,"Number of Alpha Electrons")
  write(iwfx,"(i8)") nelea
  call wfxlab(iwfx,1,"Number of Alpha Electrons")

  ! Number of Beta Electrons
  call wfxlab(iwfx,0,"Number of Beta Electrons")
  write(iwfx,"(i8)") neleb
  call wfxlab(iwfx,1,"Number of Beta Electrons")

  ! Electronic Spin Multiplicity (optional)
  call wfxlab(iwfx,0,"Electronic Spin Multiplicity")
  write(iwfx,"(i8)") MS
  call wfxlab(iwfx,1,"Electronic Spin Multiplicity")

  ! Number of Core Electrons
  call wfxlab(iwfx,0,"Number of Core Electrons")
  write(iwfx,"(i8)")lecp
  call wfxlab(iwfx,1,"Number of Core Electrons")

  ! Nuclear Names
  call wfxlab(iwfx,0,"Nuclear Names")
  do i=1,nat
    write(ctmp(11:100),*)i
    call trulen(ctmp(11:100),len2,len3,len1)
    call ElemZA(1,ctmp,iza(i),ctmp)
    len1=len_trim(ctmp(1:3))
    write(iwfx,"(a)")ctmp(1:len1)//ctmp(len2+10:len3+10)
  end do
  call wfxlab(iwfx,1,"Nuclear Names")

  ! Atomic Numbers (Z)
  call wfxlab(iwfx,0,"Atomic Numbers")
  write(iwfx,"(i8)")(iza(i),i=1,nat)
  call wfxlab(iwfx,1,"Atomic Numbers")

  ! Nuclear Charges (Z-#core)
  call wfxlab(iwfx,0,"Nuclear Charges")
  do i=1,nat
    write(iwfx,"(e21.12e3)")dble(iza(i)-icore(i))
  end do
  call wfxlab(iwfx,1,"Nuclear Charges")

  ! Nuclear Cartesian Coordinates (in a.u.)
  call wfxlab(iwfx,0,"Nuclear Cartesian Coordinates")
  do i=1,nat
    write(iwfx,"(3e21.12e3)") xyz(:,i)
  end do
  call wfxlab(iwfx,1,"Nuclear Cartesian Coordinates")

  ! Number of Primitives
  call wfxlab(iwfx,0,"Number of Primitives")
  write(iwfx,"(i8)")ncarp
  call wfxlab(iwfx,1,"Number of Primitives")

  ! print basis functions
  allocate(expg(ncarp), conf(ncarp), ityp(ncarp), icmo(ncarp), cn(ncarp))
  call writecnt(iwfx,1,  nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  ncarp,expg,conf,ityp,icmo,cn)

  !<<<<<<<<<<<<<<<<<< ECP >>>>>>>>>>>>>>>>>>
  ! see sub. edfmain for the format of the lecp data file
  if(lecp > 0)then
    call wfxlab(iwfx,0,"Additional Electron Density Function (EDF)")
      ! Number of EDF Primitives
      call wfxlab(iwfx,0,"Number of EDF Primitives")
      write(iwfx,"(i8)")nedf
      call wfxlab(iwfx,1,"Number of EDF Primitives")
      ! EDF Primitive Centers
      call wfxlab(iwfx,0,"EDF Primitive Centers")
      rewind(iedf)
      do i=1,nedf
        read(iedf,"(i5)") ix
        write(iwfx,"(i8)",advance='no') ix
        if(mod(i,5) == 0) write(iwfx,*)
      end do
      if(mod(nedf,5) /= 0) write(iwfx,*)
      call wfxlab(iwfx,1,"EDF Primitive Centers")
      ! EDF Primitive Types
      call wfxlab(iwfx,0,"EDF Primitive Types")
      write(iwfx,"(5i8)") (1, i=1, nedf)
      call wfxlab(iwfx,1,"EDF Primitive Types")
      ! EDF Primitive Exponents
      call wfxlab(iwfx,0,"EDF Primitive Exponents")
      rewind(iedf)
      do i=1,nedf
        read(iedf,"(5x,e21.12e3)") x
        write(iwfx,"(e21.12e3)",advance='no') x
        if(mod(i,5) == 0) write(iwfx,*)
      end do
      if(mod(nedf,5) /= 0) write(iwfx,*)
      call wfxlab(iwfx,1,"EDF Primitive Exponents")
      ! EDF Primitive Coefficients
      call wfxlab(iwfx,0,"EDF Primitive Coefficients")
      rewind(iedf)
      do i=1,nedf
        read(iedf,"(26x,e21.12e3)") x
        write(iwfx,"(e21.12e3)",advance='no') x
        if(mod(i,5) == 0) write(iwfx,*)
      end do
      if(mod(nedf,5) /= 0) write(iwfx,*)
      call wfxlab(iwfx,1,"EDF Primitive Coefficients")
    call wfxlab(iwfx,1,"Additional Electron Density Function (EDF)")
  end if

  ! Molecular Orbital Occupation Numbers
  call wfxlab(iwfx,0,"Molecular Orbital Occupation Numbers")
  do i=1,nmotot
    if(lprtmo(i) < 1) cycle
    write(iwfx,"(e21.12e3)")occup(i)
  end do
  call wfxlab(iwfx,1,"Molecular Orbital Occupation Numbers")

  ! Molecular Orbital Energies
  call wfxlab(iwfx,0,"Molecular Orbital Energies")
  do i=1,nmotot
    if(lprtmo(i) < 1) cycle
    write(iwfx,"(e21.12e3)")min(ene(i),enemax)
  end do
  call wfxlab(iwfx,1,"Molecular Orbital Energies")

 ! Molecular Orbital Spin Types
  call wfxlab(iwfx,0,"Molecular Orbital Spin Types")
  do i=1,nmotot
    if(lprtmo(i) < 1) cycle
    if(prtspn) then
      if(ispin(i) == 1) then
        write(iwfx,"(' Alpha')")
      else if(ispin(i) == 2) then
        write(iwfx,"(' Beta')")
      end if
    else
      write(iwfx,"(' Alpha and Beta')")
    end if
  end do
  call wfxlab(iwfx,1,"Molecular Orbital Spin Types")

  ! MO
  call wfxlab(iwfx,0,"Molecular Orbital Primitive Coefficients")
  call writemol(iwfx,nmotot,ncarp,ncarc,lprtmo,carmo,expg,conf,ityp,icmo,cn)
  call wfxlab(iwfx,1,"Molecular Orbital Primitive Coefficients")
  deallocate(expg, conf, ityp, icmo, cn)

  ! Energy = T + Vne + Vee + Vnn
  write(iwfx,"('# The total energy of the molecule.')")
  write(iwfx,"('# For HF and KSDFT, this is the SCF energy.')")
  write(iwfx,"('# For MP2, this is the MP2 total energy.')")
  write(iwfx,"('# For CCSD, this is the CCSD total energy.')")
  write(iwfx,"('# etc.')")
  call wfxlab(iwfx,0,"Energy = T + Vne + Vee + Vnn")
  if(iunknw == 0) then
    write(iwfx,"(' UNKNOWN')")
  else
    write(iwfx,"(e21.12e3)") 0.d0
  end if
  call wfxlab(iwfx,1,"Energy = T + Vne + Vee + Vnn")

  ! Virial Ratio (-V/T)
  call wfxlab(iwfx,0,"Virial Ratio (-V/T)")
  if(iunknw == 0) then
    write(iwfx,"(' UNKNOWN')")
  else
    write(iwfx,"(e21.12e3)") 2.d0
  end if
  call wfxlab(iwfx,1,"Virial Ratio (-V/T)")

  return
 end subroutine wfxmain

 ! write MO.s to the *.wfx file
 subroutine writemol(iwfx,nmo,ng,ngc,lprtmo,carmo,expg,conf,ityp,icmo,cn)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: lprtmo(*), carmo(ngc,*), expg(*), conf(*), ityp(*), icmo(*), cn(*)

  ! normalization factor
  do i=1,ng
    cn(i)=fnorm_lmn(expg(i),ityp(i))
  end do

  ix=0
  do i=1,nmo
    if(lprtmo(i) < 1) cycle

    ix = ix + 1
    ! MO index
    call wfxlab(iwfx,0,"MO Number")
    write(iwfx,"(i8)") ix
    call wfxlab(iwfx,1,"MO Number")
    ! N*cgto*cmo
    write(iwfx,"(5e21.12e3)") (cn(j)*conf(j)*carmo(icmo(j),i),j=1,ng)
  end do

  return
 end subroutine writemol

 !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 !
 ! main subroutine of EDF library.
 !
 !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 subroutine edfmain(iedf,lecp,iedftyp,nat,nedf,iza,icore)
  implicit real(kind=8) (a-h,o-z)
  parameter(MxEDF=40)  ! it's assumed that nfun <= 40
  dimension         :: iza(*), icore(*)
  allocatable       :: edfa(:), edfc(:)

  nedf = 0
  if(lecp <= 0) return

  pi=acos(-1.d0)
  alf=pi * 4.d0

  rewind(iedf)
  allocate(edfa(MxEDF), edfc(MxEDF))

  write(*,"(/,'  Generate EDF data:')")
  do i=1,nat
    if(icore(i) > 0)then
      if(iedftyp == 0) then
        call EDFLIB(iza(i),icore(i),nfun,edfa,edfc)
      else
        call EDFPBE0(iza(i),icore(i),nfun,edfa,edfc)
      end if

      if(nfun > 0) then
        nedf = nedf + nfun
        acore = 0.d0
        do j=1, nfun
          write(iedf,"(i5,2e21.12e3)") i, edfa(j), edfc(j)
          x = sqrt(pi/edfa(j))
          acore = acore + edfc(j)*x*x*x
        end do
        if(iedftyp == 0) then
          write(*,"('  Iatm=',i4,', ZA=',i4,', NCore=',i4,', ACore=',f14.9,': from the EDF library by X2C/HF.')") &
            i, iza(i), icore(i), acore
        else
          write(*,"('  Iatm=',i4,', ZA=',i4,', NCore=',i4,', ACore=',f14.9,': from the EDF library by X2C/PBE0.')") &
            i, iza(i), icore(i), acore
        end if
      else
        ! Generate EDF using the tight core density function, ie. alpha = 4pi, c = 8Ncore
        ! It is good for small-core or medium-core ECP but worse for large-core ECP. See Eq. 9 in JPCA 115, 12879 (2011).
        c=dble(8*icore(i))
        nedf = nedf + 1
        write(iedf,"(i5,2e21.12e3)") i, alf, c
        ! acore = c*sqrt(pi/alf)^3 = c/8 = icore(i); see Eq. 2 in JCC, 39, 1697 (2018).
        acore = dble(icore(i))
        write(*,"('  Iatm=',i4,', ZA=',i4,', NCore=',i4,', ACore=',f14.9,': Tight Core Function')") i, iza(i), icore(i), acore
      end if
    end if
  end do
  deallocate(edfa, edfc)

  return
 end subroutine edfmain

 ! print messages after saving the wfx file
 subroutine finalwfx(fwfx,lecp,iunknw,MaxL)
  implicit real(kind=8) (a-h,o-z)
  character*157     :: fwfx

  write(*,"(//,'  A WFX file is generated successfully!',/,'  File Name = ',a)")trim(fwfx)
  ! with ECP/MCP
  if(lecp > 0) then
    if(MaxL < 4)then
      write(*,"(/,2x,'ECPs/MCPs are found! Please use',/,3x,'AIMALL, Critic2, DensToolKit, IGMPlot, MultiWFN, or ORBKIT',/)")
    else
      write(*,"(/,2x,'ECPs/MCPs and G-functions are found! Please use',/,3x,'AIMALL, Critic2, MultiWFN, or ORBKIT',/)")
    end if
  ! without ECP/MCP
  else
    if(MaxL < 4)then
      write(*,"(/,2x,'Please use',/,3x,'AIMALL, Critic2, DensToolKit, GPView, IGMPlot, MultiWFN, or ORBKIT',/)")
    else if(MaxL < 5)then
      write(*,"(/,2x,'G-functions are found! Please use',/,3x,'AIMALL, Critic2, GPView, MultiWFN, or ORBKIT',/)")
    else
      write(*,"(/,2x,'H-functions are found! Please use',/,3x,'AIMALL or MultiWFN',/)")
    end if
  end if
  write(*,"('  to analyse the electron density distribution.')")
  if(iunknw == 0) then
    write(*,"(//,6x,65('-'),/, 6x,'>>> Please correct the UNKNOWN terms in the WFX file manually <<<',/, 6x,65('-'),/)")
  else
    write(*,"(//,6x,65('-'),/, 6x,'>>> Please modify Energy and Virial Ratio in WFX if necessary <<<',/, 6x,65('-'),/)")
  end if

  return
 end subroutine finalwfx

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! generate a wfn file.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine genwfn(iwfn,fwfn,ver,dt,lpspin,  nat,lecp,iza,icore,xyz,  MaxL,nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  &
 ncarp,ncarc,nmotot,nmo,lalph,lbeta,ispin,ene,occup,lprtmo,carmo,  tmp)
 implicit real(kind=8) (a-h,o-z)
 character*5       :: ver
 character*10      :: dt
 character*157     :: fwfn
 character*100     :: tmp
 dimension         :: iza(*), icore(*), xyz(3,*), mapatm(*), lqnm(*), nshlls(*), nshlln(*), expgto(*), congto(*),  &
                      ispin(*), ene(*), occup(*), lprtmo(*), carmo(ncarc,*)
 allocatable       :: expg(:), conf(:), ityp(:), icmo(:), cn(:)

 OPEN(iwfn,FILE=fwfn)
 rewind(iwfn)

 ! title
 call writitle(iwfn,ver,dt,tmp)

 write(iwfn,"('GAUSSIAN',8x,i7,' MOL ORBITALS',i7,' PRIMITIVES',i9,' NUCLEI')")nmo, ncarp, nat

 call writeatm(iwfn,nat,iza,icore,xyz,tmp)

 allocate(expg(ncarp), conf(ncarp), ityp(ncarp), icmo(ncarp), cn(ncarp))
 call writecnt(iwfn,0,  nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  ncarp,expg,conf,ityp,icmo,cn)
 call writemol(iwfn,nmotot,ncarp,ncarc,ene,occup,lprtmo,carmo,expg,conf,ityp,icmo,cn)
 deallocate(expg, conf, ityp, icmo, cn)

 write(iwfn,"('END DATA')")
 write(iwfn,"(' THE  HF ENERGY =',f20.12,' THE VIRIAL(-V/T)=',f13.8)")0.d0,2.d0

 ! Spin of MO. These data can be used by MultiWFN.
 ! if lalph=0, "Spin" cannot be found in Molden
 if(lpspin ==1 .and. lalph == 1) call writespn(iwfn,lbeta,nmotot,ispin,lprtmo)

 ! final step
 call finalwfn(fwfn,lecp,MaxL)

 return

 !---------------------------------------------------------------------- private subroutines
 contains

 ! print coordinates to WFN.
 subroutine writeatm(iwfn,nat,iza,icore,xyz,am)
  implicit real(kind=8) (a-h,o-z)
  dimension         :: iza(*), icore(*), xyz(3,*)
  character*3       :: am

  do i=1,nat
    call ElemZA(1,am,iza(i),am)
    write(iwfn,"(2x,A3,i3,4x,'(CENTRE',i3,') ',3f12.8,'  CHARGE =',f5.1)")am,i,i,xyz(:,i),dble(iza(i)-icore(i))
  end do

  return
 end subroutine writeatm

 ! write MO.s to the *.wfn file
 subroutine writemol(iwfn,nmo,ng,ngc,ene,occup,lprtmo,carmo,expg,conf,ityp,icmo,cn)
  implicit real(kind=8) (a-h,o-z)
  parameter(enemax=9999.d0)
  dimension         :: ene(*), occup(*), lprtmo(*), carmo(ngc,*), expg(*), conf(*), ityp(*), icmo(*), cn(*)

  ! normalization factor
  do i=1,ng
    cn(i)=fnorm_lmn(expg(i),ityp(i))
  end do

  do i=1,nmo
    if(lprtmo(i) < 1) cycle

    write(iwfn,"('MO',i5,5x,'MO 0.0',8x,'OCC NO =',f13.7,'  ORB. ENERGY =',f12.6)")i,occup(i),min(ene(i),enemax)

    ! N*cgto*cmo
    write(iwfn,"(5d16.8)")(cn(j)*conf(j)*carmo(icmo(j),i),j=1,ng)
  end do

  return
 end subroutine writemol

 ! print spin of MO
 ! in ispin,   0: n.a.     1: alpha or alpha+beta    2: beta
 ! in iwfn,    1: alpha    2: beta    3: alpha+beta
 subroutine writespn(iwfn,lbeta,nmo,ispin,lprtmo)
  implicit real(kind=8) (a-h,o-z)
  parameter(ncol=40)
  dimension         :: ispin(nmo), lprtmo(nmo)

  write(iwfn,"(//,' $MOSPIN $END')")

  imo = 0
  do i=1,nmo
    if(lprtmo(i) < 1) cycle
    imo = imo + 1
    ls = 3
    if(lbeta == 1) ls = ispin(i)
    if(imo < ncol) then
      write(iwfn,"(i2)",advance='no')ls
    else
      imo = 0
      write(iwfn,"(i2)")ls
    end if
  end do
  write(iwfn,"(//)")

  return
 end subroutine writespn

 ! print messages after saving the wfn file
 subroutine finalwfn(fwfn,lecp,MaxL)
  implicit real(kind=8) (a-h,o-z)
  character*157     :: fwfn

  write(*,"(//,'  A WFN file is generated successfully!',/,'  File Name = ',a)")trim(fwfn)
  if(lecp > 0)then
    write(*,"(/,2x,'Because of PP (ECP or MCP), please use',/,3x,'MultiWFN 3.2.1 or higher versions',/)")
  else if(MaxL < 4)then
    write(*,"(/,2x,'Please use',/,3x,'AIM2000, AIMALL, AIMPAC, AIMPAC2, AIM-UC, CheckDen, Critic2, DensToolKit,',/,  &
      3x,'DGrid, IGMPlot, MORPHY, MultiWFN, ORBKIT, PAMoC, ProMolden, TopChem,',/,3x,'TopMoD, or XAIM',/)")
  else if(MaxL < 5)then
    write(*,"(/,2x,'G-functions are found! Please use',/,3x,  &
      'AIM2000 (Ver. 2013), AIMALL, AIM-UC, Critic2, DGrid, MultiWFN, ORBKIT,',/,3x,'or TopChem',/)")
  else
    write(*,"(/,2x,'H-functions are found! Please use',/,3x,  &
      'AIMALL or MultiWFN',/)")
  end if
  write(*,"('  to analyse the electron density distribution.')")
  if(lecp > 0) write(*,"(/,9x,'>>> Please consult MultiWFN manual (sec. 5.7) for details <<<')")

  return
 end subroutine finalwfn

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! write atoms, coordinates, and basis functions to the *.wfn (mode=0) or *.wfx (mode /= 0) file
!
! See the WFX format for ITyp.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine writecnt(iwfn,mode,  nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  npg,expg,conf,ityp,icmo,icnt)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: mapatm(nshell), lqnm(nshell), nshlls(nshell), nshlln(nshell), expgto(nshell), congto(nshell),  &
                      expg(npg), conf(npg), ityp(npg), icmo(npg), icnt(npg)

 igc = 0
 ipt2= 0
 do ish = 1, nshell
   select case(lqnm(ish))
     case(0)
       nfun=1
       it=0
     case(1)
       nfun=3
       it=1
     case(2)
       nfun=6
       it=4
     case(3)
       nfun=10
       it=10
     case(4)
       nfun=15
       it=20
     case(5)
       nfun=21
       it=35
     case default
       write(*,"(/,' *** Error! LQ > 5 in sub. writecnt.')")
       call estop(1)
   end select
   do i=1,nfun
     igc = igc + 1
     ipt1=ipt2+1
     ipt2=ipt1+nshlln(ish)-1
     it=it+1
     if(it >= 14 .and. it <= 19)then
       ! f: xyy, xxy, xxz, xzz, yzz, yyz (MOLDEN or Gaussian-out) -->
       !    xxy, xxz, yyz, xyy, xzz, yzz (Gaussian-WFN or GAMESS-WFN)
       ityp(ipt1:ipt2)=iorder(it)
     else
       ityp(ipt1:ipt2)=it
     end if
     call acopy(nshlln(ish),expgto(nshlls(ish)),expg(ipt1))
     call acopy(nshlln(ish),congto(nshlls(ish)),conf(ipt1))
     icmo(ipt1:ipt2)=igc
     icnt(ipt1:ipt2)=mapatm(ish)
   end do
 end do

 if(mode == 0)then
   !  wfn
   write(iwfn,"('CENTRE ASSIGNMENTS  ',20i3)")(icnt(i),i=1,npg)
   write(iwfn,"('TYPE ASSIGNMENTS    ',20i3)")(ityp(i),i=1,npg)
   write(iwfn,"('EXPONENTS ',5d14.7)")(expg(i),i=1,npg)
 else
   !  wfx
   call wfxlab(iwfn,0,"Primitive Centers")
   write(iwfn,"(5i20)")(icnt(i),i=1,npg)
   call wfxlab(iwfn,1,"Primitive Centers")
   call wfxlab(iwfn,0,"Primitive Types")
   write(iwfn,"(5i20)")(ityp(i),i=1,npg)
   call wfxlab(iwfn,1,"Primitive Types")
   call wfxlab(iwfn,0,"Primitive Exponents")
   write(iwfn,"(5e21.12e3)")(expg(i),i=1,npg)
   call wfxlab(iwfn,1,"Primitive Exponents")
 end if

 return

 !---------------------------------------------------------------------- private subroutines
 contains

 ! reorder f-functions from MOLDEN to WFN.
 function iorder(n)
  implicit real(kind=8) (a-h,o-z)

  select case(n)
    case(14)
      iorder=17
    case(15)
      iorder=14
    case(16)
      iorder=15
    case(17)
      iorder=18
    case(18)
      iorder=19
    case(19)
      iorder=16
  end select

  return
 end function iorder

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! generate a standard Molden file with Cartrsian basis functions.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine genmdn(fmdn,inmd,isym,ver,dt,nat,MaxL,lecp,iza,icore,xyz, nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,  &
 lspout,nmotot,ngc,lsymm,ispin,ene,occup,lprtmo,carmo,tmp)
 implicit real(kind=8) (a-h,o-z)
 character*5       :: ver
 character*10      :: dt
 character*164     :: fmdn
 character*100     :: tmp
 dimension         :: iza(*), icore(*), xyz(3,*), mapatm(*), lqnm(*), nshlls(*), nshlln(*), expgto(*), congto(*),  &
                      ispin(*), ene(*), occup(*), lprtmo(*), carmo(ngc,*)
 character*1       :: ALQ(0:5)

 OPEN(inmd,FILE=fmdn)
 rewind(inmd)
 if(lsymm == 1) rewind(isym)

 ! title
 write(inmd,"('[Molden Format]')")
 write(inmd,"('[Title]')")
 call writitle(inmd,ver,dt,tmp)
 write(inmd,*)

 ! coordinates
 write(inmd,"('[Atoms] AU')")
 do i=1,nat
   call ElemZA(1,tmp,iza(i),tmp)
   write(inmd,"(2x,a3,2i5,3f20.10)") tmp(1:3), i, iza(i)-icore(i), xyz(:,i)
 end do

 ! #core
 if(lecp > 0)then
   write(inmd,"('[PSEUDO]')")
   do i=1,nat
     call ElemZA(1,tmp,iza(i),tmp)
     write(inmd,"(a3,1x,2i6)") tmp(1:3), i, iza(i)-icore(i)
   end do
 end if

 ! the blank line before [GTO] is not allowed by MOLDEN!!!

 ! basis functions
 ALQ = (/'S','P','D','F','G','H'/)
 write(inmd,"('[GTO]')")
 ia = 0
 do i = 1, nshell
   if(ia < mapatm(i)) then
     ia = mapatm(i)
     if(ia > 1) write(inmd,*)
     write(inmd,"(2i5)") ia, 0
   end if

   write(inmd,"(1x,a1,i5,2x,'1.0')") ALQ(lqnm(i)), nshlln(i)

   do j = nshlls(i), nshlls(i)+nshlln(i)-1
     write(inmd,"(2d20.10)") expgto(j), congto(j)
   end do
 end do
 write(inmd,*)

 if(lspout == 1) then
   if(MaxL == 2) then
     write(inmd,"('[5D]')")
   else if(MaxL > 2) then
     write(inmd,"('[5D7F]')")
     if(MaxL > 3) write(inmd,"('[9G]')")
     if(MaxL > 4) write(inmd,"('[11H]')")
   end if
 end if

 ! MO in Cartesian basis functions
 write(inmd,"('[MO]')")
 do i=1,nmotot
   ! read "Sym="
   if(lsymm == 1) read(isym,"(a100)") tmp

   if(lprtmo(i) < 1) cycle
   ! print "Sym=", "Ene=", "Spin=", and "Occup="
   ! Gabedit doesn't recognize UPPERCASE keywords!
   if(lsymm == 1) write(inmd,"(' Sym= ',a)") trim(tmp)
   write(inmd,"(' Ene=',f20.8)") ene(i)
   if(ispin(i) == 2)then
     write(inmd,"(' Spin= Beta')")
   else
     write(inmd,"(' Spin= Alpha')")
   end if
   write(inmd,"(' Occup=',f18.8)") occup(i)
   do j=1,ngc
     write(inmd,"(i5,f20.10)") j, carmo(j,i)
   end do
 end do
 write(inmd,*)

 if(lspout == 1) then
   write(*,"(//,'  A new Molden file in spherical functions is generated successfully!',/,'  File Name = ',a)") trim(fmdn)
 else
   write(*,"(//,'  A new Molden file in Cartesian functions is generated successfully!',/,'  File Name = ',a)") trim(fmdn)
 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! print a title
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine writitle(idat,ver,dt,cdate)
 implicit real(kind=8) (a-h,o-z)
 character*10      :: dt
 character*5       :: ver
 character*50      :: cdate

! if fdate doesn't work, just use this line
! write(idat,"(' Molden2AIM, Version ',a5,' (',a10,')')")ver,dt

 call fdate(cdate)
 call trulen(cdate,L1,L2,length)
 write(idat,"(' Molden2AIM, Version ',a5,' (',a10,')',11x,'Time: ',a24)")ver,dt,cdate(L1:L2)

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! MOs to be printed in the new-MOLDEN, WFN, and WFX files.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine PrtListMO(nmotot,nmoprt,IAllMO,tolocc,occup,lprtmo)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: occup(nmotot),lprtmo(nmotot)

 if(IAllMO > 0) then
   nmoprt = nmotot
   lprtmo = 1
 else
   lprtmo = 0
   nmoprt = 0
   do i = 1, nmotot
     if(occup(i) >= tolocc) then
       lprtmo(i) = 1
       nmoprt = nmoprt + 1
     end if
   end do
 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Scale MOs in Cartesian functions
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine ScaleCarMO(ncar,nmo,scal,carmo)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: scal(ncar), carmo(ncar,nmo)

 do idxmo=1, nmo
   do i=1,ncar
     carmo(i,idxmo) = carmo(i,idxmo) * scal(i)
   end do
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! convert MOs from Cartesian functions into spherical functions
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine C2SMO(nshell,nsph,ncar,nmo,lqnm,c2sd,c2sf,c2sg,c2sh,carmo,sphmo)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: lqnm(nshell),c2sd(6,5),c2sf(10,7),c2sg(15,9),c2sh(21,11),carmo(ncar,nmo),sphmo(nsph,nmo)
 allocatable       :: npure(:),puremo(:),ncart(:),cartmo(:)

 allocate(npure(6), puremo(11), ncart(6), cartmo(21))

 npure = (/ 1, 3, 5, 7, 9,11/)
 ncart = (/ 1, 3, 6,10,15,21/)

 do idxmo=1, nmo
   ips = 1
   ipc = 1
   do i=1,nshell
     ltyp=lqnm(i)
     msph=npure(ltyp+1)
     mcar=ncart(ltyp+1)
     call acopy(mcar,carmo(ipc,idxmo),cartmo)
     call car2sph(ltyp,c2sd,c2sf,c2sg,c2sh,cartmo,puremo)
     call acopy(msph,puremo,sphmo(ips,idxmo))
     ips = ips + msph
     ipc = ipc + mcar
   end do
 end do

 deallocate(npure, puremo, ncart, cartmo)

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! convert MOs from spherical functions into Cartesian functions
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine S2CMO(nshell,iprog,nsph,ncar,nmo,lqnm,s2cd,s2cf,s2cg,s2ch,sphmo,carmo)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: lqnm(nshell),s2cd(6,5),s2cf(10,7),s2cg(15,9),s2ch(21,11),sphmo(nsph,nmo),carmo(ncar,nmo)
 allocatable       :: npure(:),puremo(:),ncart(:),cartmo(:),scal(:)

 allocate(npure(6), puremo(11), ncart(6), cartmo(21), scal(11))

 npure = (/ 1, 3, 5, 7, 9,11/)
 ncart = (/ 1, 3, 6,10,15,21/)

 ! Scaling factors for Orca
 !        class                           factor
 !   1    s                             * 1.0
 !        p
 !        d
 !        f(0,+1,-1,+2,-2)
 !        g(0,+1,-1,+2,-2)
 !        h(0,+1,-1,+2,-2,+5,-5)
 !   2    f(+3,-3)                      *-1.0
 !        g(+3,-3,+4,-4)
 scal=1.d0
 if(iprog == 1)scal(6:9)=-1.d0

 do idxmo=1, nmo
   ips = 1
   ipc = 1
   do i=1,nshell
     ltyp=lqnm(i)
     msph=npure(ltyp+1)
     mcar=ncart(ltyp+1)
     ! Scaling factor for Crystal
     !   1    s,p,d                         * 1.0
     !   2    f                             *-1.0
     if(iprog ==10) then
       scal = 1.d0
       if(ltyp == 3) scal =-1.d0
     end if
     call acopy(msph,sphmo(ips,idxmo),puremo)
     do j=1,msph
       puremo(j)=puremo(j)*scal(j)
     end do
     ! There is a bug in NBO6: d(-1) and d(+1) should be exchanged!
     if(iprog == 9 .and. ltyp == 2) call swap(puremo(2),puremo(3))
     call sph2car(ltyp,s2cd,s2cf,s2cg,s2ch,puremo,cartmo)
     call acopy(mcar,cartmo,carmo(ipc,idxmo))
     ips = ips + msph
     ipc = ipc + mcar
   end do
 end do

 deallocate(npure, puremo, ncart, cartmo, scal)

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Cartesian MO --> spherical MO
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine car2sph(lq,c2sd,c2sf,c2sg,c2sh,fi,fo)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: c2sd(6,5),c2sf(10,7),c2sg(15,9),c2sh(21,11),fi(*),fo(*)

 select case(lq)
   case(0)    ! S
     fo(1)=fi(1)
   case(1)    ! P
     fo(1:3)=fi(1:3)
   case(2)    ! D
     fo(1:5)=0.0d0
     do j=1,5
       do i=1,6
         fo(j)=fo(j)+c2sd(i,j)*fi(i)
       end do
     end do
   case(3)    ! F
     fo(1:7)=0.0d0
     do j=1,7
       do i=1,10
         fo(j)=fo(j)+c2sf(i,j)*fi(i)
       end do
     end do
   case(4)    ! G
     fo(1:9)=0.0d0
     do j=1,9
       do i=1,15
         fo(j)=fo(j)+c2sg(i,j)*fi(i)
       end do
     end do
   case(5)    ! H
     fo(1:11)=0.0d0
     do j=1,11
       do i=1,21
         fo(j)=fo(j)+c2sh(i,j)*fi(i)
       end do
     end do
   case default
     write(*,"(/,' *** Error! LQ > 5 in sub. car2sph.')")
     call estop(1)
 end select

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! spherical MO --> Cartesian MO
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine sph2car(lq,s2cd,s2cf,s2cg,s2ch,fi,fo)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: s2cd(6,5),s2cf(10,7),s2cg(15,9),s2ch(21,11),fi(*),fo(*)

 select case(lq)
   case(0)    ! S
     fo(1)=fi(1)
   case(1)    ! P
     fo(1:3)=fi(1:3)
   case(2)    ! D
     fo(1:6)=0.0d0
     do j=1,5
       do i=1,6
         fo(i)=fo(i)+s2cd(i,j)*fi(j)
       end do
     end do
   case(3)    ! F
     fo(1:10)=0.0d0
     do j=1,7
       do i=1,10
         fo(i)=fo(i)+s2cf(i,j)*fi(j)
       end do
     end do
   case(4)    ! G
     fo(1:15)=0.0d0
     do j=1,9
       do i=1,15
         fo(i)=fo(i)+s2cg(i,j)*fi(j)
       end do
     end do
   case(5)    ! H
     fo(1:21)=0.0d0
     do j=1,11
       do i=1,21
         fo(i)=fo(i)+s2ch(i,j)*fi(j)
       end do
     end do
   case default
     write(*,"(/,' *** Error! LQ > 5 in sub. sph2car.')")
     call estop(1)
 end select

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! converts Cartesian matrix s to spherical matrix from left and right, and save to s:
! amap^T * s(npta x nptb) * bmap --> s(nspa x nspb)
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine car2sph_blk(s,tmp,npta,nptb,nspa,nspb,dmap,fmap,gmap,hmap)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: s(nptb,npta),tmp(*),dmap(6,5),fmap(10,7),gmap(15,9),hmap(21,11)

 ! First half of transformation:
 if(nptb <= 3)then        ! S, P
   call acopy(nptb*npta,s,tmp)
 else if(nptb == 6)then   ! Cart. D
   call MatMult(2,nspb,nptb,npta,dmap,s,tmp)
 else if(nptb == 10)then  ! Cart. F
   call MatMult(2,nspb,nptb,npta,fmap,s,tmp)
 else if(nptb == 15)then  ! Cart. G
   call MatMult(2,nspb,nptb,npta,gmap,s,tmp)
 else if(nptb == 21)then  ! Cart. H
   call MatMult(2,nspb,nptb,npta,hmap,s,tmp)
 end if

 ! Second half of transformation:
 if(npta <= 3)then        ! S, P
   call acopy(nspa*nspb,tmp,s)
 else if(npta == 6)then   ! Cart. D
   call MatMult(1,nspb,npta,nspa,tmp,dmap,s)
 else if(npta == 10)then  ! Cart. F
   call MatMult(1,nspb,npta,nspa,tmp,fmap,s)
 else if(npta == 15)then  ! Cart. G
   call MatMult(1,nspb,npta,nspa,tmp,gmap,s)
 else if(npta == 21)then  ! Cart. H
   call MatMult(1,nspb,npta,nspa,tmp,hmap,s)
 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! coefficients for Cartesian GTF --> spherical GTF. See sub. s2c_coef_old for the ordering.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine c2s_coef(MaxL,c2sd,c2sf,c2sg,c2sh,ierr)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: c2sd(6,5),c2sf(10,7),c2sg(15,9),c2sh(21,11)

 ierr = 0

 if(MaxL > 5) then
   write(*,"(/,' *** Error! MaxL > 5 in sub. c2s_coef.')")
   ierr = 1
   return
 end if

 if(MaxL > 1) then
   c2sd = 0.0D0
   c2sd( 1, 1) = -0.3333333333333333D+00
   c2sd( 2, 1) = -0.3333333333333333D+00
   c2sd( 3, 1) =  0.6666666666666666D+00
   c2sd( 5, 2) =  1.0D0
   c2sd( 6, 3) =  1.0D0
   c2sd( 1, 4) =  0.5773502691896257D+00
   c2sd( 2, 4) = -0.5773502691896257D+00
   c2sd( 4, 5) =  1.0D0
 end if

 if(MaxL > 2) then
   c2sf = 0.0D0
   c2sf( 3, 1) =  0.4D0
   c2sf( 6, 1) = -0.4472135954999578D+00
   c2sf( 9, 1) = -0.4472135954999578D+00
   c2sf( 1, 2) = -0.2449489742783178D+00
   c2sf( 4, 2) = -0.1825741858350554D+00
   c2sf( 7, 2) =  0.7302967433402215D+00
   c2sf( 2, 3) = -0.2449489742783178D+00
   c2sf( 5, 3) = -0.1825741858350554D+00
   c2sf( 8, 3) =  0.7302967433402214D+00
   c2sf( 6, 4) =  0.5773502691896257D+00
   c2sf( 9, 4) = -0.5773502691896257D+00
   c2sf(10, 5) =  1.0D0
   c2sf( 1, 6) =  0.3162277660168379D+00
   c2sf( 4, 6) = -0.7071067811865475D+00
   c2sf( 2, 7) = -0.3162277660168379D+00
   c2sf( 5, 7) =  0.7071067811865475D+00
 end if

 if(MaxL > 3) then
   c2sg = 0.0D0
   c2sg( 1, 1) =  0.8571428571428570D-01
   c2sg( 2, 1) =  0.8571428571428572D-01
   c2sg( 3, 1) =  0.2285714285714286D+00
   c2sg(10, 1) =  0.9759000729485329D-01
   c2sg(11, 1) = -0.3903600291794133D+00
   c2sg(12, 1) = -0.3903600291794133D+00
   c2sg( 5, 2) = -0.3585685828003183D+00
   c2sg( 8, 2) =  0.4780914437337576D+00
   c2sg(14, 2) = -0.2672612419124243D+00
   c2sg( 7, 3) = -0.3585685828003183D+00
   c2sg( 9, 3) =  0.4780914437337576D+00
   c2sg(13, 3) = -0.2672612419124243D+00
   c2sg( 1, 4) = -0.1277753129999880D+00
   c2sg( 2, 4) =  0.1277753129999880D+00
   c2sg(11, 4) =  0.4364357804719848D+00
   c2sg(12, 4) = -0.4364357804719847D+00
   c2sg( 4, 5) = -0.1690308509457035D+00
   c2sg( 6, 5) = -0.1690308509457035D+00
   c2sg(15, 5) =  0.7559289460184542D+00
   c2sg( 5, 6) =  0.3162277660168382D+00
   c2sg(14, 6) = -0.7071067811865474D+00
   c2sg( 7, 7) = -0.3162277660168382D+00
   c2sg(13, 7) =  0.7071067811865475D+00
   c2sg( 1, 8) =  0.1690308509457033D+00
   c2sg( 2, 8) =  0.1690308509457032D+00
   c2sg(10, 8) = -0.5773502691896257D+00
   c2sg( 4, 9) =  0.4472135954999581D+00
   c2sg( 6, 9) = -0.4472135954999581D+00
 end if

 if(MaxL > 4) then
   c2sh = 0.0D0
   c2sh( 1, 1) =  0.1269841269841268D+00
   c2sh( 3, 1) = -0.2909571869813233D+00
   c2sh( 5, 1) =  0.1428571428571428D+00
   c2sh(12, 1) = -0.2909571869813233D+00
   c2sh(14, 1) =  0.1626500121580887D+00
   c2sh(19, 1) =  0.1428571428571429D+00
   c2sh( 7, 2) =  0.2950844454253272D+00
   c2sh( 9, 2) = -0.2519763153394846D+00
   c2sh(11, 2) =  0.3688555567816591D-01
   c2sh(16, 2) = -0.3380617018914065D+00
   c2sh(18, 2) =  0.5634361698190118D-01
   c2sh(21, 2) =  0.6147592613027647D-01
   c2sh( 2, 3) =  0.2950844454253272D+00
   c2sh( 4, 3) = -0.3380617018914065D+00
   c2sh( 6, 3) =  0.6147592613027650D-01
   c2sh(13, 3) = -0.2519763153394846D+00
   c2sh(15, 3) =  0.5634361698190118D-01
   c2sh(20, 3) =  0.3688555567816593D-01
   c2sh( 3, 4) = -0.2981423969999721D+00
   c2sh( 5, 4) =  0.1951800145897066D+00
   c2sh(12, 4) =  0.2981423969999720D+00
   c2sh(19, 4) = -0.1951800145897067D+00
   c2sh( 8, 5) =  0.5163977794943222D+00
   c2sh(10, 5) = -0.2581988897471612D+00
   c2sh(17, 5) = -0.2581988897471612D+00
   c2sh( 9, 6) = -0.5443310539518175D+00
   c2sh(11, 6) =  0.1195228609334393D+00
   c2sh(16, 6) =  0.2434322477800738D+00
   c2sh(18, 6) =  0.6085806194501837D-01
   c2sh(21, 6) = -0.6640158940746627D-01
   c2sh( 4, 7) = -0.2434322477800739D+00
   c2sh( 6, 7) =  0.6640158940746628D-01
   c2sh(13, 7) =  0.5443310539518175D+00
   c2sh(15, 7) = -0.6085806194501844D-01
   c2sh(20, 7) = -0.1195228609334393D+00
   c2sh( 5, 8) =  0.1690308509457033D+00
   c2sh(14, 8) = -0.5773502691896257D+00
   c2sh(19, 8) =  0.1690308509457032D+00
   c2sh(10, 9) = -0.4472135954999579D+00
   c2sh(17, 9) =  0.4472135954999580D+00
   c2sh(11,10) =  0.2672612419124244D+00
   c2sh(18,10) = -0.4082482904638632D+00
   c2sh(21,10) =  0.8908708063747461D-01
   c2sh( 6,11) =  0.8908708063747466D-01
   c2sh(15,11) = -0.4082482904638631D+00
   c2sh(20,11) =  0.2672612419124244D+00
 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! coefficients for spherical GTF --> Cartesian GTF. See sub. s2c_coef_old for the ordering.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine s2c_coef(MaxL,s2cd,s2cf,s2cg,s2ch,ierr)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: s2cd(6,5),s2cf(10,7),s2cg(15,9),s2ch(21,11)

 ierr = 0

 if(MaxL > 5) then
   write(*,"(/,' *** Error! MaxL > 5 in sub. s2c_coef.')")
   ierr = 1
   return
 end if

 if(MaxL > 1) then
   s2cd = 0.0D0
   s2cd( 1, 1) = -0.5000000000000000D+00
   s2cd( 2, 1) = -0.5000000000000000D+00
   s2cd( 3, 1) =  1.0D0
   s2cd( 5, 2) =  1.0D0
   s2cd( 6, 3) =  1.0D0
   s2cd( 1, 4) =  0.8660254037844387D+00
   s2cd( 2, 4) = -0.8660254037844387D+00
   s2cd( 4, 5) =  1.0D0
 end if

 if(MaxL > 2) then
   s2cf = 0.0D0
   s2cf( 3, 1) =  1.0D0
   s2cf( 6, 1) = -0.6708203932499368D+00
   s2cf( 9, 1) = -0.6708203932499368D+00
   s2cf( 1, 2) = -0.6123724356957946D+00
   s2cf( 4, 2) = -0.2738612787525830D+00
   s2cf( 7, 2) =  0.1095445115010332D+01
   s2cf( 2, 3) = -0.6123724356957946D+00
   s2cf( 5, 3) = -0.2738612787525830D+00
   s2cf( 8, 3) =  0.1095445115010332D+01
   s2cf( 6, 4) =  0.8660254037844386D+00
   s2cf( 9, 4) = -0.8660254037844386D+00
   s2cf(10, 5) =  1.0D0
   s2cf( 1, 6) =  0.7905694150420949D+00
   s2cf( 4, 6) = -0.1060660171779821D+01
   s2cf( 2, 7) = -0.7905694150420949D+00
   s2cf( 5, 7) =  0.1060660171779821D+01
 end if

 if(MaxL > 3) then
   s2cg = 0.0D0
   s2cg( 1, 1) =  0.3750000000000000D+00
   s2cg( 2, 1) =  0.3750000000000000D+00
   s2cg( 3, 1) =  1.0D0
   s2cg(10, 1) =  0.2195775164134200D+00
   s2cg(11, 1) = -0.8783100656536799D+00
   s2cg(12, 1) = -0.8783100656536799D+00
   s2cg( 5, 2) = -0.8964214570007953D+00
   s2cg( 8, 2) =  0.1195228609334394D+01
   s2cg(14, 2) = -0.4008918628686366D+00
   s2cg( 7, 3) = -0.8964214570007953D+00
   s2cg( 9, 3) =  0.1195228609334394D+01
   s2cg(13, 3) = -0.4008918628686366D+00
   s2cg( 1, 4) = -0.5590169943749475D+00
   s2cg( 2, 4) =  0.5590169943749475D+00
   s2cg(11, 4) =  0.9819805060619659D+00
   s2cg(12, 4) = -0.9819805060619659D+00
   s2cg( 4, 5) = -0.4225771273642583D+00
   s2cg( 6, 5) = -0.4225771273642583D+00
   s2cg(15, 5) =  0.1133893419027682D+01
   s2cg( 5, 6) =  0.7905694150420950D+00
   s2cg(14, 6) = -0.1060660171779821D+01
   s2cg( 7, 7) = -0.7905694150420950D+00
   s2cg(13, 7) =  0.1060660171779821D+01
   s2cg( 1, 8) =  0.7395099728874520D+00
   s2cg( 2, 8) =  0.7395099728874520D+00
   s2cg(10, 8) = -0.1299038105676658D+01
   s2cg( 4, 9) =  0.1118033988749895D+01
   s2cg( 6, 9) = -0.1118033988749895D+01
 end if

 if(MaxL > 4) then
   s2ch = 0.0D0
   s2ch( 1, 1) =  1.0D0
   s2ch( 3, 1) = -0.1091089451179962D+01
   s2ch( 5, 1) =  0.625D+00
   s2ch(12, 1) = -0.1091089451179962D+01
   s2ch(14, 1) =  0.3659625273557000D+00
   s2ch(19, 1) =  0.625D+00
   s2ch( 7, 2) =  0.1290994448735806D+01
   s2ch( 9, 2) = -0.5669467095138407D+00
   s2ch(11, 2) =  0.1613743060919757D+00
   s2ch(16, 2) = -0.1267731382092775D+01
   s2ch(18, 2) =  0.2112885636821291D+00
   s2ch(21, 2) =  0.4841229182759271D+00
   s2ch( 2, 3) =  0.1290994448735806D+01
   s2ch( 4, 3) = -0.1267731382092775D+01
   s2ch( 6, 3) =  0.4841229182759271D+00
   s2ch(13, 3) = -0.5669467095138407D+00
   s2ch(15, 3) =  0.2112885636821291D+00
   s2ch(20, 3) =  0.1613743060919757D+00
   s2ch( 3, 4) = -0.1118033988749895D+01
   s2ch( 5, 4) =  0.8539125638299665D+00
   s2ch(12, 4) =  0.1118033988749895D+01
   s2ch(19, 4) = -0.8539125638299665D+00
   s2ch( 8, 5) =  0.1290994448735806D+01
   s2ch(10, 5) = -0.6454972243679028D+00
   s2ch(17, 5) = -0.6454972243679028D+00
   s2ch( 9, 6) = -0.1224744871391589D+01
   s2ch(11, 6) =  0.5229125165837972D+00
   s2ch(16, 6) =  0.9128709291752769D+00
   s2ch(18, 6) =  0.2282177322938192D+00
   s2ch(21, 6) = -0.5229125165837972D+00
   s2ch( 4, 7) = -0.9128709291752769D+00
   s2ch( 6, 7) =  0.5229125165837972D+00
   s2ch(13, 7) =  0.1224744871391589D+01
   s2ch(15, 7) = -0.2282177322938192D+00
   s2ch(20, 7) = -0.5229125165837972D+00
   s2ch( 5, 8) =  0.7395099728874520D+00
   s2ch(14, 8) = -0.1299038105676658D+01
   s2ch(19, 8) =  0.7395099728874520D+00
   s2ch(10, 9) = -0.1118033988749895D+01
   s2ch(17, 9) =  0.1118033988749895D+01
   s2ch(11,10) =  0.1169267933366857D+01
   s2ch(18,10) = -0.1530931089239487D+01
   s2ch(21,10) =  0.7015607600201140D+00
   s2ch( 6,11) =  0.7015607600201140D+00
   s2ch(15,11) = -0.1530931089239487D+01
   s2ch(20,11) =  0.1169267933366857D+01
 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! coefficients for spherical GTF --> Cartesian GTF
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!Subroutine s2c_coef_old(MaxL,s2cd,s2cf,s2cg,ierr)
! implicit real(kind=8) (a-h,o-z)
! parameter(                   &
!      c375 = 0.375d0,         &
!      r3_4 = 0.8660254038d0,  &     ! sqrt(3/4)
!      r5_4 = 1.1180339887d0,  &     ! sqrt(5/4)
!      r6_5 = 1.0954451150d0,  &     ! sqrt(6/5)
!      r9_7 = 1.1338934190d0,  &     ! sqrt(9/7)
!     r10_7 = 1.1952286093d0,  &     ! sqrt(10/7)
!      r5_8 = 0.7905694150d0,  &     ! sqrt(5/8)
!      r9_8 = 1.0606601718d0,  &     ! sqrt(9/8)
!     r5_16 = 0.5590169944d0,  &     ! sqrt(5/16)
!    r27_28 = 0.9819805061d0,  &     ! sqrt(27/28)
!    r35_64 = 0.7395099729d0,  &     ! sqrt(35/64)
!   r27_560 = 0.2195775164d0,  &     ! sqrt(27/560)
!      v3_4 =-0.8660254038d0,  &     !-sqrt(3/4)
!      v5_4 =-1.1180339887d0,  &     !-sqrt(5/4)
!      v3_8 =-0.6123724357d0,  &     !-sqrt(3/8)
!      v5_8 =-0.7905694150d0,  &     !-sqrt(5/8)
!      v9_8 =-1.0606601718d0,  &     !-sqrt(9/8)
!     v5_16 =-0.5590169944d0,  &     !-sqrt(5/16)
!    v27_16 =-1.2990381057d0,  &     !-sqrt(27/16)
!     v9_20 =-0.6708203932d0,  &     !-sqrt(9/20)
!     v5_28 =-0.4225771274d0,  &     !-sqrt(5/28)
!    v27_28 =-0.9819805061d0,  &     !-sqrt(27/28)
!    v27_35 =-0.8783100657d0,  &     !-sqrt(27/35)
!     v3_40 =-0.2738612788d0,  &     !-sqrt(3/40)
!     v9_56 =-0.4008918629d0,  &     !-sqrt(9/56)
!    v45_56 =-0.8964214570d0)        !-sqrt(45/56)
! dimension         :: s2cd(6*5),s2cf(10*7),s2cg(15*9)
!
! ierr = 0
!
! if(MaxL > 4) then
!   write(*,"(/,' *** Error! MaxL > 4 in sub. s2c_coef_old.')")
!   ierr = 1
!   return
! end if
!
! ! In pure basis functions, MOLDEN ordering is used.
! ! In Cartesian basis functions, MOLDEN ordering is used and will be re-ordered in subroutine writecnt.
!
! if(MaxL > 1) s2cd = (/  &
! !     xx      yy      zz      xy      xz      yz
!   -0.5d0, -0.5d0,  1.0d0,  0.0d0,  0.0d0,  0.0d0,  &  ! d0  = zz-(xx+yy)/2
!    0.0d0,  0.0d0,  0.0d0,  0.0d0,  1.0d0,  0.0d0,  &  ! d+1 = xz
!    0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  1.0d0,  &  ! d-1 = yz
!     r3_4,   v3_4,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  &  ! d+2 = r3_4*(xx-yy)
!    0.0d0,  0.0d0,  0.0d0,  1.0d0,  0.0d0,  0.0d0/)    ! d-2 = xy
!
! if(MaxL > 2) s2cf = (/  &
! !    xxx     yyy     zzz     xyy     xxy     xxz     xzz     yzz     yyz     xyz
!    0.0d0,  0.0d0,  1.0d0,  0.0d0,  0.0d0,  v9_20,  0.0d0,  0.0d0,  v9_20,  0.0d0,  &  ! f0  = zzz-(xxz+yyz)*sqrt(9/20)
!     v3_8,  0.0d0,  0.0d0,  v3_40,  0.0d0,  0.0d0,   r6_5,  0.0d0,  0.0d0,  0.0d0,  &  ! f+1 = -xxx*sqrt(3/8)-xyy*sqrt(3/40)+xzz*sqrt(6/5)
!    0.0d0,   v3_8,  0.0d0,  0.0d0,  v3_40,  0.0d0,  0.0d0,   r6_5,  0.0d0,  0.0d0,  &  ! f-1 = -yyy*sqrt(3/8)-xxy*sqrt(3/40)+yzz*sqrt(6/5)
!    0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,   r3_4,  0.0d0,  0.0d0,   v3_4,  0.0d0,  &  ! f+2 = (xxz-yyz)*sqrt(3/4)
!    0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  1.0d0,  &  ! f-2 = xyz
!     r5_8,  0.0d0,  0.0d0,   v9_8,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  &  ! f+3 = xxx*sqrt(5/8)-xyy*sqrt(9/8)
!    0.0d0,   v5_8,  0.0d0,  0.0d0,   r9_8,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0/)    ! f-3 = yyy*sqrt(5/8)-xxy*sqrt(9/8)
!
! if(MaxL > 3) s2cg = (/  &
! !   xxxx    yyyy    zzzz    xxxy    xxxz    yyyx    yyyz    zzzx    zzzy    xxyy    xxzz    yyzz    xxyz    yyxz    zzxy
!     c375,   c375,  1.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,r27_560, v27_35, v27_35,  0.0d0,  0.0d0,  0.0d0,  &  ! g0  = (xxxx+yyyy)*c375+zzzz+xxyy*r27_560+(xxzz+yyzz)*v27_35
!    0.0d0,  0.0d0,  0.0d0,  0.0d0, v45_56,  0.0d0,  0.0d0,  r10_7,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  v9_56,  0.0d0,  &  ! g+1 = xzzz*r10_7+xxxz*v45_56+yyxz*v9_56
!    0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0, v45_56,  0.0d0,  r10_7,  0.0d0,  0.0d0,  0.0d0,  v9_56,  0.0d0,  0.0d0,  &  ! g-1 = yzzz*r10_7+yyyz*v45_56+xxyz*v9_56
!    v5_16,  r5_16,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0, r27_28, v27_28,  0.0d0,  0.0d0,  0.0d0,  &  ! g+2 = -(xxxx-yyyy)*r5_16+(xxzz-yyzz)*r27_28
!    0.0d0,  0.0d0,  0.0d0,  v5_28,  0.0d0,  v5_28,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,   r9_7,  &  ! g-2 = (xxxy+xyyy)*v5_28+zzxy*r9_7
!    0.0d0,  0.0d0,  0.0d0,  0.0d0,   r5_8,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,   v9_8,  0.0d0,  &  ! g+3 = xxxz*r5_8+yyxz*v9_8
!    0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,   v5_8,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,   r9_8,  0.0d0,  0.0d0,  &  ! g-3 = yyyz*v5_8+xxyz*r9_8
!   r35_64, r35_64,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0, v27_16,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  &  ! g+4 = (xxxx+yyyy)*r35_64+xxyy*v27_16
!    0.0d0,  0.0d0,  0.0d0,   r5_4,  0.0d0,   v5_4,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0,  0.0d0/)    ! g-4 = (xxxy-xyyy)*r5_4
!
! H-functions are in the Gaussian (MultiWFN) ordering:
! 11h:  0, +1, -1, +2, -2, +3, -3, +4, -4, +5, -5
! 21h: zzzzz, yzzzz, yyzzz, yyyzz, yyyyz, yyyyy, xzzzz, xyzzz, xyyzz, xyyyz, xyyyy, xxzzz,
!      xxyzz, xxyyz, xxyyy, xxxzz, xxxyz, xxxyy, xxxxz, xxxxy, xxxxx
!
! return
!end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! calculate the normalization factor for GTO(l,m,n)
! it = 1,...,35
! Ordering type:
! S,P,D: MOLDEN, Gaussian, GAMESS, WFN, ...
! F: WFN (for MOLDEN and Gaussian, 14~19 are different)
! G: WFN, MOLDEN (not for Gaussian)
! H: WFX, Gaussian
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fnorm_lmn(a,it)
 implicit real(kind=8) (a-h,o-z)

 pi=acos(-1.d0)
 pi3=pi**3.d0

 select case(it)
   case(1)            ! 1S
     n1=3
     n2=3
     nf=1
   case(2:4)          ! 3P: x, y, z
     n1=7
     n2=5
     nf=1
   case(5:10)         ! 6D
     n1=11
     n2=7
     select case(it)
       case(5:7)        ! xx, yy, zz
         nf=9
       case(8:10)       ! xy, xz, yz
         nf=1
     end select
   case(11:20)        ! 10F
     n1=15
     n2=9
     select case(it)
       case(11:13)      ! xxx, yyy, zzz
         nf=225
       case(14:19)      ! xxy, xxz, yyz, xyy, xzz, yzz
         nf=9
       case(20)         ! xyz
         nf=1
     end select
   case(21:35)        ! 15G
     n1=19
     n2=11
     select case(it)
       case(21:23)      ! xxxx yyyy zzzz
         nf=11025
       case(24:29)      ! xxxy xxxz yyyx yyyz zzzx zzzy
         nf=225
       case(30:32)      ! xxyy xxzz yyzz
         nf=81
       case(33:35)      ! xxyz yyxz zzxy
         nf=9
     end select
 !<<< Gaussian (subroutine pattml should also be modified)
 !    select case(it)
 !      case(21,25,35)               ! xxxx yyyy zzzz
 !        nf=11025
 !      case(22,24,26,29,33,34)      ! xxxy xxxz yyyx yyyz zzzx zzzy
 !        nf=225
 !      case(23,30,32)               ! xxyy xxzz yyzz
 !        nf=81
 !      case(27,28,31)               ! xxyz yyxz zzxy
 !        nf=9
 !    end select
 !>>>
   case(36:56)        ! 21H
     n1=23
     n2=13
     select case(it)
       case(36,41,56)            ! zzzzz yyyyy xxxxx
         nf=893025
       case(37,40,42,46,54,55)   ! yzzzz yyyyz xzzzz xyyyy xxxxz xxxxy
         nf=11025
       case(38,39,47,50,51,53)   ! yyzzz yyyzz xxzzz xxyyy xxxzz xxxyy
         nf=2025
       case(43,45,52)            ! xyzzz xyyyz xxxyz
         nf=225
       case(44,48,49)            ! xyyzz xxyzz xxyyz
         nf=81
     end select
 end select

 ! Normal^4 = 2^n1 * a^n2 / (pi^3 * nf)
 !   n1=3+4*(l+m+n)
 !   n2=3+2*(l+m+n)
 !   nf=[(2l-1)!!(2m-1)!!(2n-1)!!]^2
 f = (2.d0**dble(n1)) * (a**dble(n2)) / (pi3 * dble(nf))
 fnorm_lmn=sqrt(sqrt(f))

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! renormalize a contracted GTF
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine renorm(al,ngauss,a,ci,c)
 implicit real(kind=8) (a-h,o-z)
 parameter (tol=1.0d-10)
 dimension         :: a(ngauss),ci(ngauss),c(ngauss)
 character*1       :: al

 pi=acos(-1.d0)
 pi3=pi*pi*pi

 ! unnormalize primitives
 ! Normal^4 = 2^n1 * a^n2 / (pi^3 * nf)
 !   n1=3+4*L; n2=3+2*L, nf=[(2L-1)!!]^2
 call power(al,n1,n2,nf)
 fc = (2.d0**dble(n1)) / (pi3 * dble(nf))
 do i = 1,ngauss
   f = fc * (a(i)**dble(n2))
   f = sqrt(sqrt(f))
   c(i) = ci(i)*f
 end do

 fsum = 0.d0
 do i = 1,ngauss
   do j = 1,i
     a2 = (a(i)+a(j)) * 0.5d0
     f = fc * (a2**dble(n2))
     f = sqrt(f)
     f = c(i)*c(j)/f
     if (i /= j) f = f + f
     fsum = fsum+f
   end do
 end do

 if (fsum > tol) fsum = 1.d0 / sqrt(fsum)
 ci = ci * fsum

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! calculate the integral Ix, Iy, or Iz
!   (l1+l2)/2
! I = sum [fcoef * (2i-1)!! / (2g)^i]
!     i=0
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine integ(s,MaxL,l1,l2,dgn,pan,pbn)
 implicit real(kind=8) (a-h,o-z)
 ! Tlm1ff = (2N -1)!!, N=0~5; (2g)^-i; pa^n; pb^n
 dimension         :: Tlm1ff(0:5),dgn(0:MaxL),pan(0:MaxL),pbn(0:MaxL)
 data Tlm1ff/1.D0, 1.D0, 3.D0, 15.D0, 105.D0, 945.D0/
 save Tlm1ff

 s=0.d0
 do i=0,(l1+l2)/2
   call fcoef(tmp,MaxL,i+i,l1,l2,pan,pbn)
   tmp=tmp*Tlm1ff(i)*dgn(i)
   s=s+tmp
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! calculate the coefficients
!           l1   l2
! fcoef =  sum  sum [clm(l1,k1) * clm(l2,k2) * a^(l1-k1) * b^(l2-k2)]
!         k1=0 k2=0
!          k1+k2=i
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine fcoef(fc,MaxL,i,l1,l2,pan,pbn)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: pan(0:MaxL),pbn(0:MaxL)

 fc=0.d0
 do k1=0,l1
   k2=i-k1
   if(k2 < 0 .or. k2 > l2) cycle
   fc = fc + pan(l1-k1) * pbn(l2-k2) * clm(l1,k1) * clm(l2,k2)
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! calculate the normalization factor for GTO(L)
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fnorm(ex,al)
 implicit real(kind=8) (a-h,o-z)
 character*1       :: al

 pi=acos(-1.d0)
 pi3=pi**3.d0

 ! unnormalize primitives
 ! Normal^4 = 2^n1 * a^n2 / (pi^3 * nf), where n1=3+4*L; n2=3+2*L, nf=[(2L-1)!!]^2
 call power(al,n1,n2,nf)
 f = (2.d0**dble(n1)) * (ex**dble(n2)) / (pi3 * dble(nf))
 fnorm=sqrt(sqrt(f))

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! get power(n1,n2,nf) for GTO(L) normalization
! n1=3+4*L; n2=3+2*L, nf=[(2L-1)!!]^2
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine power(al,n1,n2,nf)
 implicit real(kind=8) (a-h,o-z)
 character*1       :: al

 n1=0
 n2=0
 nf=0
 select case(al)
   case('S')
     n1=3
     n2=3
     nf=1
   case('P')
     n1=7
     n2=5
     nf=1
   case('D')
     n1=11
     n2=7
     nf=9
   case('F')
     n1=15
     n2=9
     nf=225
   case('G')
     n1=19
     n2=11
     nf=11025
   case('H')
     n1=23
     n2=13
     nf=893025
   case default
     write(*,"(/,' *** Error! LQ > 5 in sub. power.')")
     call estop(1)
 end select

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Check occupation numbers
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine ChkOcc(ifbeta,lfc4,sumocc,nmotot,occup,ierr)
 implicit real(kind=8) (a-h,o-z)
 parameter(occtol=5.0d-4)
 dimension  :: occup(nmotot)
 logical    :: ifprt

 ierr = 0

 if(lfc4 == 2) then
   sumocc = sumocc * 2.0d0
   occup = occup * 2.0d0
 end if

 otolmin = -occtol
 otolmax = 2.0d0 + occtol
 if(ifbeta == 1) then
   otolmin = otolmin * 0.5d0
   otolmax = otolmax * 0.5d0
 end if

 ifprt = .false.
 do i = 1, nmotot
   if(occup(i) < otolmin  .or. occup(i) > otolmax) then
     if(.not. ifprt) write(*,"(/,' Warning! The occupation numbers are out of range!')")
     ifprt = .true.
     write(*,"('  OCC[',i5,'] = ',f12.6,',  TolMin = ',f12.6,',  TolMax = ',f12.6)") i, occup(i), otolmin, otolmax
     !ierr = 1
     !exit
   end if
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! read core data
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 subroutine RdCore(imtm,iprog,chanet,ntote,nchar,sumocc,lrdecp,natm,ifbeta,lfc4,lecp,iza,icore,ctmp,yn,ierr)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: iza(natm), icore(natm)
 character*100     :: ctmp
 character*1       :: yn, L2U

 ierr = 0

 if(lrdecp == 1) then
   write(*,"(//,' The [PSEUDO] data block has been found in the Molden file.')")
 else if(lrdecp == 2) then
   write(*,"(//,' The [CORE] data block has been found in the Molden file.')")
 ! else if(abs(dble(nchar)-sumocc) > 0.001d0)then
 else if(abs(dble(nchar)-sumocc) > 1.001d0 .and. (dble(nchar)-sumocc) > -8.001d0)then
   write(*,"(//,' Is ECP or MCP used? ([Yes] / No)',/,' > ',$)")
   read(*,"(a1)")yn
   yn=L2U(yn)
   lrdecp = 0
   if(yn /= 'N') then
     lrdecp = 3
     write(*,"(/,' No [CORE] or [PSEUDO] data found. You have to type them in this terminal.',/, &
       ' Please consult the format of [Core]. Use an empty line to end the input.')")
   end if
 end if

 icore = 0
 if(lrdecp == 1 .or. lrdecp == 2)then
   call RdCore1(imtm,natm,lecp,lrdecp,iza,icore,ctmp,ierr)
     if(ierr /= 0)goto 1000
 else if(lrdecp == 3) then
   call RdCore2(natm,lecp,iza,icore,ctmp,ierr)
     if(ierr /= 0)goto 1000
 end if

 chanet = dble(nchar-lecp)-sumocc
 if(abs(chanet) > 0.001d0)then
    ! #e > 2, charge =+/-1, and #core=0
    if(sumocc > 2.0d0-0.001d0 .and. abs(chanet) < 1.0d0+0.001d0 .and. lecp >= 0) then
      call chkcharge1(sumocc,chanet,ierr)
    else
      call chkcharge(nchar,sumocc,iprog,ifbeta,lfc4,lecp,chanet,ierr)
    end if
    if(ierr /= 0)goto 1000
 end if
 ! total number of electrons: core electrons by ECP should be excluded!
 ntote = nchar - lecp - nint(chanet)

 1000  return

 !---------------------------------------------------------------------- private subroutines
 contains

 ! check special cases of molecular charge: +/-1
 subroutine chkcharge1(sumocc,chanet,ierr)
  implicit real(kind=8) (a-h,o-z)

  ierr=0

  docc=abs(ANINT(sumocc)-sumocc)
  if(docc < 1.d-4)then
    if(chanet >= 0.0d0) then
      write(*,"(5x,'Net charge of the ionic system:',16x,f8.4)") chanet
    else
      write(*,"(5x,'Net charge of the anionic system:',14x,f8.4)") chanet
    end if
  else if(docc < 1.0d-2)then
    write(*,"(/,' Warning! Strange occupation: ',f10.4)")sumocc
    write(*,"(/,' Please check your AIM results carefully.')")
    call xcontinue
  else
    write(*,"(/,' Error! Strange occupation: ',f10.4)")sumocc
    ierr=1
    call xcontinue
  end if

  return
 end subroutine chkcharge1

 ! check molecular charge
 subroutine chkcharge(nchar,sumocc,iprog,ifbeta,lfc4,lecp,chanet,ierr)
  implicit real(kind=8) (a-h,o-z)
  character*1       :: ioc

  ierr=1

  ! CFour: RHF with dble(nchar-lecp)-2*sumocc = 0
  !chanet=dble(nchar-lecp)-sumocc
  if(iprog == 2 .and. ifbeta == 0 .and. abs(sumocc-chanet) <= 1.0d-10) then
    lfc4=2
    ierr=0
    return
  end if

  write(*,"(//,' Warning: the total electron is different from the sum of occupations!',//,4x,  &
    '#Electron        =',f10.4,/,4x,'Sum_Occupation   =',f10.4,/,4x,  &
    '#Core Electron   =',f10.4,/,4x,'Net Charge       =',f10.4,//,' The reasons may be',/,  &
    ' 1) semi-empirical Hamiltonian is used,')") dble(nchar),sumocc,dble(lecp),chanet
  if(chanet >= 0) then
    write(*,"(' 2) ionic system with net charge',f8.2,',')")chanet
  else
    write(*,"(' 2) anionic system with net charge',f8.2,',')")chanet
  end if
  ! C4: RHF; Q-Chem: RHF, RDFT, ROGF, RODFT
  write(*,"(' 3) beta MOs of R-/RO-SCF are not printed by CFOUR or Q-Chem, and therefore',/,  &
    '    the occupation numbers should be multiplied by 2.0,',/,  &
    ' 4) other reasons.',//,  &
    ' Which one corresponds to your case?',/,' > ',$)")

  read(*,*)ioc

  select case(ioc)
    case("1")
      write(*,"(/,' semi-empirical Hamiltonian is not supported.')")
      goto 9910
    case("2")
      docc=abs(ANINT(sumocc)-sumocc)
      if(docc < 1.d-4)then
        if(chanet >= 0.0d0) then
          write(*,"(/,' This is an ionic system with net charge',f10.4)") chanet
        else
          write(*,"(/,' This is an anionic system with net charge',f10.4)") chanet
        end if
      else if(docc < 1.0d-2)then
        write(*,"(/,' Warning! Strange occupation: ',f10.4)")sumocc
        write(*,"(/,' Please check your AIM results carefully.')")
      else
        write(*,"(/,' Error! Strange occupation: ',f10.4)")sumocc
        goto 9910
      end if
    case("3")
      if(ifbeta == 1) then
        write(*,"(/,' *** Error! This is not true since there are beta MOs.')")
        goto 9910
      end if
      write(*,"(///,' Occupations are multiplied by 2.0...')")
      lfc4=2
      chanet=dble(nchar-lecp)-sumocc*dble(lfc4)
      write(*,"(/,4x,                     &
        '#Electron        =',f10.4,/,4x,  &
        'Modified Sum_Occ =',f10.4,/,4x,  &
        '#Core Electron   =',f10.4,/,4x,  &
        'Net Charge       =',f10.4,/)") dble(nchar),sumocc*dble(lfc4),dble(lecp),chanet
      ! for Q-Chem
      if(iprog == 0)write(*,"(' *** WARNING ***',/,'  There is a BUG in Q-Chem RO-SCF. In this case, please correct the',/,  &
        '  occupation numbers of singly occupied MOs manually.')")
    case default
      write(*,"(/,' Unknown reason. Please report the problem.')")
      goto 9910
  end select
  call xcontinue
  ierr=0

  9910  return
 end subroutine chkcharge

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! read core information from the terminal
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine RdCore2(nat,ncor,iza,ico,ctmp,ierr)
 implicit real(kind=8) (a-h,o-z)
 character*100     :: ctmp
 character*3       :: atom
 dimension         :: iza(nat), ico(nat)
 logical           :: ifind

 ierr = 1
 ncor = 0

 ! read core information in the [CORE] format
 do while(.true.)
   write(*,"(' > ',$)")
   read(*,"(a100)",iostat=irdfin)ctmp
   if(LEN_TRIM(ctmp) == 0 .or. irdfin /= 0) exit

   k=index(ctmp,":")
   if(k <= 1) goto 9010

   read(ctmp(k+1:),*,err=9010,end=9010) icore
   ! check: icore
   if(icore < 0 .or. icore > 120) goto 9035

   read(ctmp(1:k-1),*,err=9010,end=9010) atom
   call charl2u(atom)
   call trulen(atom,I,J,K)
   K=ichar(atom(I:I))
   if( (K >= 65) .and. (K <= 90) ) then
     do L=I,J
       K=ichar(atom(L:L))
       if( (K < 65) .or. (K > 90) ) goto 9020
     end do
     call ElemZA(0,atom,za,za)
     IZ = nint(za)
     ! check: ZA > icore
     if(IZ <= icore) goto 9040
     ifind = .false.
     ! search all atoms with ZA = IZ
     do L=1,nat
       if(iza(L) == IZ) then
         ifind = .true.
         ico(L) = icore
       end if
     end do
     if(.not. ifind) goto 9050

   else if( (K >= 48) .and. (K <= 57) ) then
     do L=I,J
       K=ichar(atom(L:L))
       if( (K < 48) .or. (K > 57) ) goto 9020
     end do
     read(atom,*)IA
     IZ = iza(IA)

     ! check: IA <= NAtom
     if(IA < 1 .or. IA > nat) goto 9030
     ! check: ZA(IA) > icore
     if(IZ <= icore) goto 9040

     ico(IA) = icore

   else
     goto 9020
   end if

   ! check ZA vs. icore: icore must be an even number (4f & 5f metals are excluded)
   ! if(IZ < 57 .or. IZ > 103 .or. (IZ > 71 .and. IZ < 89) ) then
   if(mod(icore,2) /= 0) goto 9060
   ! end if
 end do

 ncor = 0
 do i=1,nat
   ncor = ncor + ico(i)
 end do

! print core information
 if(ncor > 0) then
   write(*,"(//,'  Core information',//,'    I    Atom       ZA    NCore',/)")
   do i=1,nat
     if(ico(i) > 0) then
       call ElemZA(1,atom,iza(i),ctmp)
       write(*,"(i5,5x,a3,2i9)")i,atom,iza(i),ico(i)
     end if
   end do
   write(*,*)
 end if

 ierr = 0
 return

 9010  write(*,"(//,' ### Error when reading core data! The format is',/,5x,'Iatom: Ncore  or  Element: Ncore')")
 write(*,"(2x,a)")trim(ctmp)
 return

 9020  write(*,"(//,' ### Wrong! Unknown element or atom index in the core data!',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

 9030  write(*,"(//,' ### Error in core data: IA is out of range!',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

 9035  write(*,"(//,' ### Error in core data: Ncore is out of range!',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

 9040  write(*,"(//,' ### Error in core data: ZA <= NCore!',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

 9050  write(*,"(//,' ### Error in core data: No such an element in the molecule!',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

 9060  write(*,"(//,' ### Error in core data: odd Ncore has not been programmed!',/,' Please contact the author.',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! read core information from the [PSEUDO] (lrdecp = 1) or [CORE] (lrdecp = 2) block in MOLDEN
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine RdCore1(imtm,nat,ncor,lrdecp,iza,ico,ctmp,ierr)
 implicit real(kind=8) (a-h,o-z)
 character*100     :: ctmp
 character*3       :: atom
 dimension         :: iza(nat), ico(nat)
 logical           :: ifind

 ierr = 1
 ncor = 0

 ! read core information
 rewind(imtm)
 do while(.true.)
   read(imtm,"(a100)")ctmp
   call charl2u(ctmp)
   if(index(ctmp,'[PSEUDO]') /= 0 .or. index(ctmp,'[CORE]') /= 0) exit
 end do

 if(lrdecp == 1) then
   ! [PSEUDO]
   do while(.true.)
     read(imtm,"(a100)",iostat=irdfin)ctmp
     if(LEN_TRIM(ctmp) == 0 .or. index(ctmp,'[') /= 0 .or. irdfin /= 0) exit

     read(ctmp,*,err=9015,end=9015) atom,IA,icore

     IZ = iza(IA)
     icore = IZ - icore
     ! check: icore
     if(icore < 0 .or. icore > 120) goto 9035

     ! check: IA <= NAtom
     if(IA < 1 .or. IA > nat) goto 9030
     ! check: ZA(IA) > icore
     if(IZ <= icore) goto 9040

     ico(IA) = icore

     ! check ZA vs. icore: icore must be an even number (4f & 5f metals are excluded)
     ! if(IZ < 57 .or. IZ > 103 .or. (IZ > 71 .and. IZ < 89) ) then
     if(mod(icore,2) /= 0) goto 9060
     ! end if
   end do

 else if(lrdecp == 2) then
   ! [CORE]
   do while(.true.)
     read(imtm,"(a100)",iostat=irdfin)ctmp
     if(LEN_TRIM(ctmp) == 0 .or. index(ctmp,'[') /= 0 .or. irdfin /= 0) exit

     k=index(ctmp,":")
     if(k <= 1) goto 9010

     read(ctmp(k+1:),*,err=9010,end=9010) icore
     ! check: icore
     if(icore < 0 .or. icore > 120) goto 9035

     read(ctmp(1:k-1),*,err=9010,end=9010) atom
     call charl2u(atom)
     call trulen(atom,I,J,K)
     K=ichar(atom(I:I))
     if( (K >= 65) .and. (K <= 90) ) then
       do L=I,J
         K=ichar(atom(L:L))
         if( (K < 65) .or. (K > 90) ) goto 9020
       end do
       call ElemZA(0,atom,za,za)
       IZ = nint(za)
       ! check: ZA > icore
       if(IZ <= icore) goto 9040
       ifind = .false.
       ! search all atoms with IZA = IZ
       do L=1,nat
         if(iza(L) == IZ) then
           ifind = .true.
           ico(L) = icore
         end if
       end do
       if(.not. ifind) goto 9050

     else if( (K >= 48) .and. (K <= 57) ) then
       do L=I,J
         K=ichar(atom(L:L))
         if( (K < 48) .or. (K > 57) ) goto 9020
       end do
       read(atom,*)IA
       IZ = iza(IA)

       ! check: IA <= NAtom
       if(IA < 1 .or. IA > nat) goto 9030
       ! check: ZA(IA) > icore
       if(IZ <= icore) goto 9040

       ico(IA) = icore

     else
       goto 9020
     end if

     ! check ZA vs. icore: icore must be an even number (4f & 5f metals are excluded)
     ! if(IZ < 57 .or. IZ > 103 .or. (IZ > 71 .and. IZ < 89) ) then
     if(mod(icore,2) /= 0) goto 9060
     ! end if
   end do

 end if

 ncor = 0
 do i=1,nat
   ncor = ncor + ico(i)
 end do

 ! print core information
 if(ncor > 0) then
   write(*,"(//,'  Core information',//,'    I    Atom       ZA    NCore',/)")
   do i=1,nat
     if(ico(i) > 0) then
       call ElemZA(1,atom,iza(i),ctmp)
       write(*,"(i5,5x,a3,2i9)")i,atom,iza(i),ico(i)
     end if
   end do
   write(*,*)
 end if

 1100  continue

 ierr = 0
 return

 9010  write(*,"(//,' ### Error when reading core data! The format is',/,5x,'Iatom: Ncore  or  Element: Ncore')")
 write(*,"(2x,a)")trim(ctmp)
 return

 9015  write(*,"(//,' ### Error when reading core data! The format is',/,5x,'Name  IAtom  ZA-Ncore')")
 write(*,"(2x,a)")trim(ctmp)
 return

 9020  write(*,"(//,' ### Wrong! Unknown element or atom index in the core data!',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

 9030  write(*,"(//,' ### Error in core data: IA is out of range!',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

 9035  write(*,"(//,' ### Error in core data: Ncore is out of range!',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

 9040  write(*,"(//,' ### Error in core data: ZA <= NCore!',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

 9050  write(*,"(//,' ### Error in core data: No such an element in the molecule!',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

 9060  write(*,"(//,' ### Error in core data: odd Ncore has not been programmed!',/,' Please contact the author.',/)")
 write(*,"(2x,a)")trim(ctmp)
 return

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! make sure that iprog matches with lsph
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine chkbstyp(lsph,iprog,MaxL,ierr)
 implicit real(kind=8) (a-h,o-z)

 ierr=0
 if(lsph == 0 .and. MaxL > 1 .and. iprog == 1)then
   ! orca: spherical b.s. is used; for Cartesian b.s. (in a future version?), you should do some tests.
   write(*,"(' ### Wrong! ORCA does not use Cartesian b.s.!')")
   ierr=1
 else if(lsph /= 0 .and. MaxL > 1 .and. iprog == 2)then
   ! cfour: Cartesian b.s. is used; for spherical b.s. (in a future version?), you should do some tests.
   write(*,"(' ### Wrong! CFOUR does not print MOs in spherical b.s.!')")
   ierr=1
 else if(lsph /= 0 .and. MaxL > 1 .and. iprog == 3)then
   ! turbomole: Cartesian b.s. is used; for spherical b.s. (in a future version?), you should do some tests.
   write(*,"(' ### Wrong! TURBOMOLE does not print MOs in spherical b.s.!')")
   ierr=1
 else if(lsph /= 0 .and. MaxL > 1 .and. iprog == 4)then
   ! jaguar: Cartesian b.s. is used; for spherical b.s. (in a future version?), you should do some tests.
   write(*,"(' ### Wrong! Jaguar does not print MOs in spherical b.s.!')")
   ierr=1
 else if(lsph /= 0 .and. MaxL > 1 .and. iprog == 5)then
   ! aces2: Cartesian b.s. is used; for spherical b.s. (in a future version?), you should do some tests.
   write(*,"(' ### Wrong! ACES2 does not print MOs in spherical b.s.!')")
   ierr=1
 else if(iprog == 10)then
   ! CRYSTAL: only spherical spdf functions may be used.
   if((lsph == 0 .and. MaxL > 1) .or. MaxL > 3) then
     write(*,"(' ### Wrong! CRYSTAL can use only spherical spdf functions!')")
     ierr=1
   end if
 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! read ENE, SPIN (see sub. MOspin), OCCUP, and CMO from the [MO] block. SYM is not used.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine RdMOs(imtm,isym,nbas,nmo,sumocc,lsymm,ifspin,ifbeta,ispin,ene,occup,cmo,ctmp,ierr)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: ispin(nmo), ene(nmo), occup(nmo), cmo(nbas,nmo)
 character*100     :: ctmp
 logical           :: istar

 ierr = 0

 rewind(imtm)
 do while(.true.)
   read(imtm,"(a100)")ctmp
   call charl2u(ctmp)
   if(index(ctmp,'[MO]') /= 0) exit
 end do

 rewind(isym)
 Isp = 0
 nsym = 0
 idxmo = 0
 ispin = 0
 ene = 0.0d0
 cmo = 0.0d0
 ene0 = 0.0d0
 occ0 = 0.0d0
 occup = 0.0d0
 do while(.true.)
   read(imtm,"(a100)",iostat=ird)ctmp
   if(ird /= 0 .or. len_trim(ctmp) == 0) exit
   if(index(ctmp,'[') /= 0 .and. index(ctmp,']') /= 0) exit
   call charl2u(ctmp)

   if(index(ctmp,'SYM') /= 0)then
     lstr=index(ctmp,'SYM')
     lend=index(ctmp,'=')
     if(lstr < lend) then
       lstr=lend + nonspace(ctmp(lend+1:100))
       lend=LEN_TRIM(ctmp)
       if(lstr <= lend) then
         nsym = nsym + 1
         write(isym,"(a)") ctmp(lstr:lend)
       end if
     end if
   else if(index(ctmp,'ENE') /= 0)then
     is=index(ctmp,'=')
     ! in MOLDEN by ACES2-F/TeraChem, there may be " Ene=   ********" or a huge value
     if(index(ctmp(is+1:),'***') == 0) then
       read(ctmp(is+1:),*) ene0
       ene0 = sign(min(abs(ene0), 1.0d6), ene0)
     else
       ene0 = 1.0d6
     end if
   else if(index(ctmp,'SPIN') /= 0)then
     Isp = MOspin(ctmp)
   else if(index(ctmp,'OCCUP') /= 0)then
     is=index(ctmp,'=')
     read(ctmp(is+1:),*) occ0
     idxmo=idxmo+1
     if(nmo < idxmo) goto 8100
     ispin(idxmo) = Isp
     ene(idxmo) = ene0
     occup(idxmo) = occ0
     ene0=0.d0
     occ0=0.d0
     Isp=0
   else
     ! MO coefficients
     c0 = 0.0d0
     istar = index(ctmp,'*') /= 0
     if(istar) then
       if(occup(idxmo) < 1.0d-6 .or. (occup(idxmo) < 1.0d-4 .and. ene(idxmo) > 1.0d2)) then
         read(ctmp,*) ibs
       else
         goto 8000
       end if
     else
       read(ctmp,*) ibs, c0
     end if
     cmo(ibs,idxmo) = c0
   end if
 end do

 if(nmo /= idxmo) goto 8100

 ! sumocc,ifspin,ifbeta
 sumocc = 0.0d0
 ifspin=1    ! 0/1: without / with spin
 ifbeta=0    ! 0/1: without / with beta spin
 do idxmo = 1, nmo
   sumocc = sumocc + occup(idxmo)
   ifspin=min(ifspin,ispin(idxmo))
   ifbeta=max(ifbeta,ispin(idxmo)/2)
 end do

 ! lsymm
 lsymm = 0
 if(nsym > 0 .and. nsym == nmo) lsymm = 1

 return

 8000  write(*,"(/,' *** Error! A star (*) has been found in the MO coefficients.')")
 ierr = 1
 return

 8100  write(*,"(/,' *** Error! nmo < idxmo in sub. RdMOs.')")
 ierr = 1
 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! spin of MO:
!   0: unknown
!   1: Alpha, or Alpha + Beta
!   2: Beta
! NOTE: tmp must be in upper case and contains 'SPIN'
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MOspin(tmp)
 implicit real(kind=8) (a-h,o-z)
 character*100     :: tmp

 if(index(tmp,'ALPHA') /= 0)then
   MOspin=1
 else if(index(tmp,'BETA') /= 0)then
   MOspin=2
 else
   MOspin=0
 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Read Cartesian coordinates from iatm
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine RdAtoms(iatm,natm,iza,xyz)
 implicit real(kind=8) (a-h,o-z)
 parameter(ang2au=1.0d0/0.52917720859d0)
 dimension         :: iza(natm), xyz(3,natm)

 rewind(iatm)

 read(iatm,*) iang
 do i = 1, natm
   read(iatm,*) iza(i), xyz(:,i)
 end do
 if(iang == 1) xyz = xyz * ang2au

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Check the molden file, and search the [PSEUDO] (lrdecp = 1) or [CORE] (lrdecp = 2) block
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine ChkMolden(imod,lrdecp,tmp1,ierr)
 implicit real(kind=8) (a-h,o-z)
 character*100     :: tmp1
 logical           :: start

 ierr = 0
 ipp  = 0
 iatm = 0
 igto = 0
 imo  = 0
 start = .false.

 rewind(imod)
 do while(.true.)
   read(imod,"(a100)",iostat=irdfin)tmp1
   if(irdfin /= 0) exit
   call charl2u(tmp1)

   ! The scratch information (e.g. in Molpro's Molden file) will be deleted
   if(.NOT. start) then
     if(index(tmp1,'[MOLDEN FORMAT]') /= 0 .or. index(tmp1,'[GABEDIT FORMAT]') /= 0) then
       start = .true.
     end if
     cycle
   end if

   ! [ATOMS] block
   if(index(tmp1,'[ATOMS]') /= 0) then
     iatm = iatm + 1
   end if

   ! [PSEUDO] block
   if(index(tmp1,'[PSEUDO]') /= 0) then
     ipp = ipp + 1
     lrdecp = 1
   end if

   ! [CORE] block
   if(index(tmp1,'[CORE]') /= 0) then
     ipp = ipp + 1
     lrdecp = 2
   end if

   ! [GTO] block
   if(index(tmp1,'[GTO]') /= 0) then
     igto = igto + 1
   end if

   ! [BASIS] block (Gabedit)
   if(index(tmp1,'[BASIS]') /= 0) then
     igto = igto + 1
   end if

   ! [MO] block
   if(index(tmp1,'[MO]') /= 0) then
     imo = imo + 1
   end if

 end do

 if(.NOT. start) then
   write(*,"(' *** Error! This is not a standard MOLDEN / GABEDIT file!')")
   ierr = 1
   goto 9999
 end if

 if(ipp > 1) then
   write(*,"(' *** Error! [PSEUDO] and [CORE] can not be used together!')")
   ierr = 1
   goto 9999
 end if

 if(iatm < 1) then
   write(*,"(' *** Error! [ATOMS] can not be found!')")
   ierr = 1
   goto 9999
 else if(iatm > 1) then
   write(*,"(' *** Error! [ATOMS] can not be repeated!')")
   ierr = 1
   goto 9999
 end if

 if(igto < 1) then
   write(*,"(' *** Error! [GTO]/[BASIS] can not be found!')")
   ierr = 1
   goto 9999
 else if(igto > 1) then
   write(*,"(' *** Error! [GTO]/[BASIS] can not be repeated!')")
   ierr = 1
   goto 9999
 end if

 if(imo < 1) then
   write(*,"(' *** Error! [MO] can not be found!')")
   ierr = 1
   goto 9999
 else if(imo > 1) then
   write(*,"(' *** Error! [MO] can not be repeated!')")
   ierr = 1
   goto 9999
 end if

 9999  return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! compute scaling factors for Cartesian MO coefficients
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine carmoscale(iprog,nshell,lqnm,isclmo,ncg,scalmo,itypc,ierr)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: lqnm(nshell), scalmo(ncg), itypc(ncg)
 character*1       :: al

 ierr = 0
 ip1=0
 ip2=0
 if(iprog == 2 .or. iprog == 5 .or. iprog == 6)then
   ! Cartesian CGTO types of cfour, aces2, or molcas (cart.)
   do ish = 1, nshell
     select case(lqnm(ish))
       case(0)
         ip1=ip2+1
         ip2=ip2+1
         itypc(ip1:ip2)=1
       case(1)
         ip1=ip2+1
         ip2=ip2+3
         itypc(ip1:ip2)=1
       case(2)
         ip1=ip2+1
         ip2=ip2+6
         itypc(ip1:(ip1+2))=2
         itypc((ip1+3):ip2)=1
       case(3)
         ip1=ip2+1
         ip2=ip2+10
         itypc(ip1:(ip1+2))=4
         itypc((ip1+3):(ip2-1))=2
         itypc(ip2:ip2)=1
       case(4)
         ip1=ip2+1
         ip2=ip2+15
         itypc(ip1:(ip1+2))=5
         itypc((ip1+3):(ip1+8))=4
         itypc((ip1+9):(ip1+11))=3
         itypc((ip2-2):ip2)=2
       case(5)
         if(iprog /= 2) then
           write(*,"(' *** Error: LQ > 4 in Sub. moscale.')")
           ierr = 1
           goto 9999
         end if
         ! H-function by CFour in the ordering of Gaussian/WFX.
         ! MO coefficients of H-functions have to be reordered before using these scaling factors.
         ip1=ip2+1
         ip2=ip2+21
         itypc(ip1+ 0)= 7    ! zzzzz
         itypc(ip1+ 1)= 5    ! yzzzz
         itypc(ip1+ 2)= 6    ! yyzzz
         itypc(ip1+ 3)= 6    ! yyyzz
         itypc(ip1+ 4)= 5    ! yyyyz
         itypc(ip1+ 5)= 7    ! yyyyy
         itypc(ip1+ 6)= 5    ! xzzzz
         itypc(ip1+ 7)= 4    ! xyzzz
         itypc(ip1+ 8)= 3    ! xyyzz
         itypc(ip1+ 9)= 4    ! xyyyz
         itypc(ip1+10)= 5    ! xyyyy
         itypc(ip1+11)= 6    ! xxzzz
         itypc(ip1+12)= 3    ! xxyzz
         itypc(ip1+13)= 3    ! xxyyz
         itypc(ip1+14)= 6    ! xxyyy
         itypc(ip1+15)= 6    ! xxxzz
         itypc(ip1+16)= 4    ! xxxyz
         itypc(ip1+17)= 6    ! xxxyy
         itypc(ip1+18)= 5    ! xxxxz
         itypc(ip1+19)= 5    ! xxxxy
         itypc(ip1+20)= 7    ! xxxxx
       case(6:)
         write(*,"(' *** Error: LQ > 5 in Sub. moscale.')")
         ierr = 1
         goto 9999
     end select
   end do
 else if(iprog == 3)then
   ! Cartesian CGTO types of turbomole
   do ish = 1, nshell
     select case(lqnm(ish))
       case(0)
         ip1=ip2+1
         ip2=ip2+1
         itypc(ip1:ip2)=1
       case(1)
         ip1=ip2+1
         ip2=ip2+3
         itypc(ip1:ip2)=1
       case(2)
         ip1=ip2+1
         ip2=ip2+6
         itypc(ip1:ip2)=2
       case(3)
         ip1=ip2+1
         ip2=ip2+10
         itypc(ip1:ip2)=3
       case(4)
         ip1=ip2+1
         ip2=ip2+15
         itypc(ip1:ip2)=4
       case(5:)
         write(*,"(' *** Error: LQ > 4 in Sub. moscale.')")
         ierr = 1
         goto 9999
     end select
   end do
 else if(iprog == 8)then
  ! Cartesian CGTO types of mrcc
   do ish = 1, nshell
     select case(lqnm(ish))
       case(0)
         ip1=ip2+1
         ip2=ip2+1
         itypc(ip1:ip2)=1
       case(1)
         ip1=ip2+1
         ip2=ip2+3
         itypc(ip1:ip2)=1
       case(2)
         ip1=ip2+1
         ip2=ip2+6
         itypc(ip1:(ip1+2))=1
         itypc((ip1+3):ip2)=2
       case(3)
         ip1=ip2+1
         ip2=ip2+10
         itypc(ip1:(ip1+2))=1
         itypc((ip1+3):(ip2-1))=3
         itypc(ip2:ip2)=4
       case(4)
         ip1=ip2+1
         ip2=ip2+15
         itypc(ip1:(ip1+2))=1
         itypc((ip1+3):(ip1+8))=5
         itypc((ip1+9):(ip1+11))=6
         itypc((ip2-2):ip2)=7
       case(5:)
         write(*,"(' *** Error: LQ > 4 in Sub. moscale.')")
         ierr = 1
         goto 9999
     end select
   end do
 end if

 if(iprog == 2 .or. iprog == 5 .or. iprog == 6)then
   !  cfour, aces2 (Cart.), or molcas (Cart.)
   do i=1,ncg
     scalmo(i)=obscalc4(itypc(i))
   end do
   isclmo = 1
 else if(iprog == 3)then
   !  turbomole (Cart.)
   do i=1,ncg
     scalmo(i)=obscaltm(itypc(i))
   end do
   isclmo = 1
 else if(iprog == 8)then
   !  mrcc (Cart.)
   do i=1,ncg
     scalmo(i)=obscalmr(itypc(i))
   end do
   isclmo = 1
 else
   isclmo = 0
   scalmo=1.d0
 end if

 9999  return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! obtain scaling factors for mrcc (Cart.)
! class                                factor
! 1    s                             * 1.0
!      p(x,y,z)                      * 1.0
!      d(xx,yy,zz)
!      f(xxx,yyy,zzz)
!      g(x4,y4,z4)
! 2    d(xy,xz,yz)                   * 1/sqrt(3)
! 3    f(xyy,xzz,yxx,yzz,zxx,zyy)    * 1/sqrt(5)
! 4    f(xyz)                        * 1/sqrt(15)
! 5    g(x3y,x3z,y3x,y3z,z3x,z3y)    * 1/sqrt(7)
! 6    g(x2y2,x2z2,y2z2)             * sqrt(3/35)
! 7    g(x2yz,y2xz,z2xy)             * 1/sqrt(35)
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function obscalmr(iclass)
 implicit real(kind=8) (a-h,o-z)

 select case(iclass)
   case(1)
     a=1.d0
   case(2)
     a=sqrt(1.d0/3.d0)
   case(3)
     a=sqrt(1.d0/5.d0)
   case(4)
     a=sqrt(1.d0/15.d0)
   case(5)
     a=sqrt(1.d0/7.d0)
   case(6)
     a=sqrt(3.d0/35.d0)
   case(7)
     a=sqrt(1.d0/35.d0)
 end select

 obscalmr=a

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! obtain scaling factors for turbomole
! class                                factor
! 1    s                             * 1.0
!      p
! 2    d                             * sqrt(3)
! 3    f                             * sqrt(15)
! 4    g                             * sqrt(105)
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function obscaltm(iclass)
 implicit real(kind=8) (a-h,o-z)

 select case(iclass)
   case(1)
     a=1.d0
   case(2)
     a=sqrt(3.d0)
   case(3)
     a=sqrt(15.d0)
   case(4)
     a=sqrt(105.d0)
 end select

 obscaltm=a

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! obtain scaling factors for cfour
! class                                   factor
! 1    s                                  * 1.0
!      p(x,y,z)
!      d(xy,xz,yz)
!      f(xyz)
! 2    d(xx,yy,zz)                        * sqrt(3)
!      f(xyy,xzz,yxx,yzz,zxx,zyy)
!      g(x2yz,y2xz,z2xy)
! 3    g(x2y2,x2z2,y2z2)                  * 3.0
!      h(xy2z2,x2yz2,x2y2z)
! 4    f(xxx,yyy,zzz)                     * sqrt(15)
!      g(x3y,x3z,y3x,y3z,z3x,z3y)
!      h(xyz3,xy3z,x3yz)
! 5    g(x4,y4,z4)                        * sqrt(105)
!      h(yz4,y4z,xz4,xy4,x4z,x4y)
! 6    h(y2z3,y3z2,x2z3,x2y3,x3z2,x3y2)   * sqrt(45)
! 7    h(z5,y5,x5)                        * sqrt(945)
!
! H-functions may be used only for CFour.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function obscalc4(iclass)
 implicit real(kind=8) (a-h,o-z)

 select case(iclass)
   case(1)
     a=1.d0
   case(2)
     a=sqrt(3.d0)
   case(3)
     a=3.d0
   case(4)
     a=sqrt(15.d0)
   case(5)
     a=sqrt(105.d0)
   case(6)
     a=sqrt(45.d0)
   case(7)
     a=sqrt(945.d0)
 end select

 obscalc4=a

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! reorder Cartesian H functions in MO coefficients of CFour's Molden file, i.e. an inverted Gaussian/WFX Cartesian ordering.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine C4HFun(nshell,lqnm,ncar,nmo,carmo)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: lqnm(nshell),carmo(ncar,nmo)
 allocatable       :: nq(:), tmp(:)

 allocate(nq(0:5), tmp(21))

 nq = (/1,3,6,10,15,21/)

 do i=1,nmo
   k = 0
   do j=1,nshell
     lq = lqnm(j)
     if(lq == 5) then
       tmp = carmo(k+1:k+21,i)
       do l = 1, 21
         carmo(k+l,i) = tmp(22-l)
       end do
     end if
     k = k + nq(lq)
   end do
 end do

 deallocate(nq, tmp)

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! clm = l! / (m! * (l-m)!)
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function clm(l,m)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: factorial(0:5)
 data factorial/1.D0,1.D0,2.D0,6.D0,24.D0,120.D0/
 save factorial

 if(l > 5) then
   write(*,"(//,' *** Error! L > 5 in function clm.')")
   call estop(1)
 end if

 clm=factorial(l)/(factorial(m)*factorial(l-m))

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! copy contraction coefficients of a give LQ from con1 to con2 and then multiply the normalization factors
! nl: # Cartesian or spherical functions of LQ
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine cpcon(al,nl,nshell,nexp,ncomp,nprim,ex,con1,con2)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: ncomp(nshell),nprim(nshell),ex(nexp),con1(nexp),con2(nexp)
 character*1       :: al

 con2 = 0.0d0
 ip=0
 do i=1,nshell
   if(ncomp(i) == nl)then
     do j=1,nprim(i)
       ip = ip + 1
       con2(ip) = con1(ip) * fnorm(ex(ip),al)
     end do
   else
     ip = ip + nprim(i)
   end if
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! read basis functions
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine RdBasis(igto,natm,nshell,mapatm,lqnm,nshlls,nshlln,expgto,congto,al)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: mapatm(nshell), lqnm(nshell), nshlls(nshell), nshlln(nshell), expgto(*), congto(*)
 character*1       :: al

 rewind(igto)

 iatm = 1
 ishell = 0
 nshlls(1) = 1
 igtf = 0
 do while(.true.)
   read(igto,*)al, np
   if(al == 'E' .and. np == 0) iatm = iatm + 1
   if(iatm > natm) exit

   if(np > 0) then
     ishell = ishell + 1
     mapatm(ishell) = iatm
     if(ishell > 1) nshlls(ishell) = nshlls(ishell-1) + nshlln(ishell-1)
     nshlln(ishell) = np
     select case(al)
       case('S')
         lq = 0
       case('P')
         lq = 1
       case('D')
         lq = 2
       case('F')
         lq = 3
       case('G')
         lq = 4
       case('H')
         lq = 5
     end select
     lqnm(ishell) = lq
     do i=1,np
       igtf = igtf + 1
       read(igto,*) expgto(igtf), congto(igtf)
     end do
   end if
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! check: Cartesian vs. spherical functions
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine checkcar(iprog,nbasmo,ncar,nsph,lsph,ierr)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: ncar(2), nsph(2)

 ierr = 0

 if(nbasmo == ncar(2))then
   !write(*,"(//,' >>> Cartesian basis functions are used.',/)")
   lsph=0
 else if(nbasmo < ncar(2) .and. nbasmo == nsph(2))then
   !write(*,"(//,' >>> Spherical basis functions are used.',/)")
   lsph=1
   ! molcas with spherical functions: do nothing
   if(iprog == 6) iprog=-6
   ! mrcc with spherical functions: do nothing
   if(iprog == 8) iprog=-8
 else if(nbasmo < ncar(2) .and. nbasmo > nsph(2))then
   write(*,"(//,' *** Wrong! nbasmo=',i5,', NCGTO=',i5,/,' Sperical and Cartesian functions are mixed together.',/,  &
     ' This is not supported.')") nbasmo,ncar(2)
   ierr = 1
 else
   write(*,"(//,' *** Wrong! nbasmo=',i5,', NCGTO=',i5,/)") nbasmo,ncar(2)
   ierr = 1
 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! count the numbers of Cartesian and Spherical basis functions from GTO.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine npgau(igto,iprog,ncar,nsph,MaxL,nshell,ngto,tmp,ierr)
 implicit real(kind=8) (a-h,o-z)
 ! Cartesian NC-/C-GTO; Spherical NC-/C-GTO
 dimension         :: ncar(2), nsph(2)
 character*100     :: tmp
 character*2       :: al

 nshell=0
 ngto=0
 ns=0
 np=0
 nd=0
 nf=0
 ng=0
 nh=0
 nsc=0
 npc=0
 ndc=0
 nfc=0
 ngc=0
 nhc=0
 MaxL=0
 Ierr=0

 rewind(igto)
 do while(.true.)
   read(igto,"(a100)",iostat=irdfin)tmp
   if(irdfin /= 0) exit

   if(index(tmp,'S ') /= 0)then
     call readng(tmp,al,npg)
     ns=ns+npg
     nsc=nsc+1
     nshell=nshell+1
     ngto=ngto+npg
   else if(index(tmp,'P ') /= 0)then
     call readng(tmp,al,npg)
     np=np+npg
     npc=npc+1
     nshell=nshell+1
     ngto=ngto+npg
   else if(index(tmp,'D ') /= 0)then
     call readng(tmp,al,npg)
     nd=nd+npg
     ndc=ndc+1
     nshell=nshell+1
     ngto=ngto+npg
   else if(index(tmp,'F ') /= 0)then
     call readng(tmp,al,npg)
     nf=nf+npg
     nfc=nfc+1
     nshell=nshell+1
     ngto=ngto+npg
   else if(index(tmp,'G ') /= 0)then
     call readng(tmp,al,npg)
     ng=ng+npg
     ngc=ngc+1
     nshell=nshell+1
     ngto=ngto+npg
   else if(index(tmp,'H ') /= 0)then
     ! For MultiWFN, CFour, and Orca only!
     if(iprog /=0 .and. iprog /= 1 .and. iprog /= 2) then
       write(*,"(' *** Error! Only S,P,D,F,G functions are supported!',/,' Please check your MOLDEN file.')")
       Ierr=1
       goto 9999
     end if
     call readng(tmp,al,npg)
     nh=nh+npg
     nhc=nhc+1
     nshell=nshell+1
     ngto=ngto+npg
   else if(index(tmp,'I ') /= 0)then
     write(*,"(' *** Error! Only S,P,D,F,G,H functions are supported!',/,' Please check your MOLDEN file.')")
     Ierr=1
     goto 9999
   end if
 end do

 ncar(1) = ns  + np *3 + nd *6 + nf *10+ng *15+nh *21
 ncar(2) = nsc + npc*3 + ndc*6 + nfc*10+ngc*15+nhc*21
 nsph(1) = ns  + np *3 + nd *5 + nf *7 +ng *9 +nh *11
 nsph(2) = nsc + npc*3 + ndc*5 + nfc*7 +ngc*9 +nhc*11

 if(ncar(1) < 1)then
   write(*,*)"*** Error! No Primitive Basis Functions found!"
   write(*,*)"Please check your MOLDEN file."
   Ierr=1
   goto 9999
 end if

! MaxL
 if(nh > 0)then
   MaxL=5
 else if(ng > 0)then
   MaxL=4
 else if(nf > 0)then
   MaxL=3
 else if(nd > 0)then
   MaxL=2
 else if(np > 0)then
   MaxL=1
 else
   MaxL=0
 end if

 9999  return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! backup GTOs
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine backupgto(imtm,igto,itmp,igtoin,igtold,nat,iprog,tmp,ierr)
 implicit real(kind=8) (a-h,o-z)
 character*100     :: tmp
 character*2       :: al
 logical           :: found

 ispace=0
 ifirst=0
 ierr=0

 ! step 1: backup original GTO
 rewind(imtm)
 rewind(igtoin)

 do while(.true.)
   read(imtm,"(a100)",end=8100)tmp
   call charl2u(tmp)
   ! [GTO]: MOLDEN file; [BASIS]: GABEDIT file
   found = index(tmp,'[GTO]') /= 0 .or. index(tmp,'[BASIS]') /= 0
   if(found) exit
 end do

 found = .false.
 do while(.true.)
   read(imtm,"(a100)",iostat=irdfin)tmp
   if(irdfin /= 0) exit

   ! the first several blank lines are skipped
   if(.not. found) then
     if(len_trim(tmp) == 0) then
       cycle
     else
       found = .true.
     end if
   end if

   call charl2u(tmp)

   if(index(tmp,'[') /= 0) then
     if(ispace == 0) write(igtoin,"(' E 0')")    ! E = END
     exit
   else
     if(len_trim(tmp) > 0)then
       if(index(tmp,'S') /= 0 .or. index(tmp,'P') /= 0 .or. index(tmp,'D') /= 0 .or. index(tmp,'F') /= 0 .or.  &
          index(tmp,'G') /= 0 .or. index(tmp,'H') /= 0) then
         call truncate(imtm,itmp,igtoin,al,tmp)
       else
         write(igtoin,"(a)")trim(tmp)
       end if
       ispace=0
     else
       if(ispace == 0)then
         write(igtoin,"(' E 0')")    ! E = END
         ispace=1                    ! the second blank line is skipped
       end if
     end if
   end if
 end do

 ! step 2: re-normalize GTFs and copy them to igtold because the GTFs in the MOLDEN file may be unnormalized
 call bknorm(igtoin,igtold,iprog,nat,al,ierr)
   if(ierr == 1) goto 9999

 ! setp 3: reorder, and dump igtold to igto
 call dumpbs(igto,igtold,nat,tmp,ierr)

 9999  return

 8100  write(*,*)"### Wrong! [GTO] or [BASIS] can not be found!"
 ierr=1
 return

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! reorder, and dump igtold to igto
!   NOTE: because of some uncertainties in the MOLDEN format, atoms in {Atoms} and [GTO] must be ordered ascendingly.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine dumpbs(igto,igtold,nat,tmp,ierr)
 implicit real(kind=8) (a-h,o-z)
 character*100     :: tmp

 rewind(igto)

 do i=1,nat
   call locatm(igtold,i,tmp,ierr)
     if(ierr == 1) goto 9999
   write(igto,"(i5,' 0')")i

   do while(.true.)
     read(igtold,"(a100)",end=9999)tmp
     if(tmp(1:1) == '&') exit
     write(igto,"(a)")trim(tmp)
   end do
 end do

 9999  return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! locate atom-i in igtold
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine locatm(igtold,i,tmp,ierr)
 implicit real(kind=8) (a-h,o-z)
 character*100     :: tmp

 ierr=0
 rewind(igtold)
 do while(.true.)
   read(igtold,"(a100)",end=1000)tmp
   if(tmp(1:1) /= '&') cycle

   read(tmp(2:100),*)it
   if(it == i) exit
 end do

 return

 1000  write(*,"(/,' ### Wrong! No basis for atom ',i5,/)") i
 ierr=1
 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! renormalize the basis functions, and check compatibility: because of some uncertainties in the MOLDEN format, the atomic indices
! must be 1,2,3,4,...
!
! In Molden 5.0, it's found that if the ordering of GTO centers is, for example, 3,1,2, then the MO coefficients in [MO] are also
! in the same order 3,1,2. This means that the atoms in [Atoms] should be reordered according to [GTO]. Example:
!
! < original MOLDEN file >
!
! [Atoms] Angs
! F     1    9    ...
! H     2    1    ...
! [GTO]
!   2 0
! ... (basis functions of H)
!   1 0
! ... (basis functions of F)
!
! < modified MOLDEN file >
!
! [Atoms] Angs
! H     1    1    ...
! F     2    9    ...
! [GTO]
!   1 0
! ... (basis functions of H)
!   2 0
! ... (basis functions of F)
!
! However in CFour's MOLDEN, the atomic numbers in [GTO] should be corrected.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine bknorm(igtoin,igtold,iprog,nat,al,ierr)
 implicit real(kind=8) (a-h,o-z)
 parameter(maxgau=200)
 character*1       :: al
 allocatable       :: expgc(:), confc(:), scr(:)

 allocate(expgc(maxgau), confc(maxgau), scr(maxgau))

 rewind(igtold)
 rewind(igtoin)

 do iat = 1, nat
   read(igtoin,*) ic1

   ! this should have been corrected in sub. ROADrv
   if(iat /= ic1)then
     write(*,"(' *** Error: the atoms in [GTO] are not ordered ascendingly.')")
     ! write(*,"(' Please correct the MOLDEN file using ReOrdAtm in util by')")
     ! if(iprog == 2)then
     !   write(*,"(/,' roa.exe -m 1 < old_molden > new_molden')")
     ! else
     !   write(*,"(/,' roa.exe < old_molden > new_molden')")
     ! end if
     ierr=1
     goto 9999
   end if

   write(igtold,"('&',i5,' 0')") ic1

   do while(.true.)
     read(igtoin,*) al, np
     if(np > maxgau) then
       write(*,"(' *** Error: np > maxgau!')")
       write(*,"(' Please increase maxgau to at least ',i4,' in Sub. bknorm.')") np
       ierr=1
       goto 9999
     end if

     if(al == 'E' .and. np == 0)then
       write(igtold,"(1x,a1,1x,i1)")al,np
       exit
     else
       write(igtold,"(1x,a1,i4,' 1.0')")al,np
     end if

     do i=1,np
       read(igtoin,*) expgc(i), confc(i)
       cbs=1.d0
       ! Orca, PSI4, NBO6, CRYSTAL: transform the basis set into input format
       if(iprog == 1 .or. iprog == 7 .or. iprog == 9 .or. iprog ==10) cbs=fnorm(expgc(i),al)
       confc(i)=confc(i)/cbs
     end do

     ! For uncontracted basis functions, the contraction coefficient can be renormalized to +1.0 or -1.0,
     ! depending on its sign. Do not change the sign!
     call renorm(al,np,expgc,confc,scr)
     do i=1,np
       write(igtold,"(d20.10,2x,d20.10)") expgc(i),confc(i)
     end do
   end do

 end do

 9999  continue
 deallocate(expgc, confc, scr)

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Write basis functions. S and P basis functions in SP are saved separately.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine truncate(imtm,itmp,igtoin,al,tmp)
 implicit real(kind=8) (a-h,o-z)
 character*100     :: tmp
 character*2       :: al

 call readng(tmp,al,ng)
 rewind(itmp)

 ng1=ng
 do i=1,ng
   if(al == 'SP')then
     read(imtm,*)exp,fc1,fc2
   else
     read(imtm,*)exp,fc1
     fc2=1.d0
   end if
   !if(abs(fc1*fc2) > 1.d-8)then
   if(abs(fc1*fc2) > 1.0d-12)then
     if(al == 'SP')then
       write(itmp,"(E20.10,2x,E20.10,2x,E20.10)")exp,fc1,fc2
     else
       write(itmp,"(E20.10,2x,E20.10)")exp,fc1
     end if
   else
     ng1=ng1-1
     cycle
   end if
 end do

 if(al == 'SP')then
   write(igtoin,*)"S ",ng1,"1.0"
 else
   write(igtoin,*)al,ng1,"1.0"
 end if

 rewind(itmp)
 do i=1,ng1
   read(itmp,*)exp,fc1
   write(igtoin,"(E20.10,2x,E20.10)")exp,fc1
 end do

 ! SP functions are saved separately
 if(al == 'SP')then
   write(igtoin,*)"P ",ng1,"1.0"
   rewind(itmp)
   do i=1,ng1
     read(itmp,*)exp,fc1,fc2
     write(igtoin,"(E20.10,2x,E20.10)")exp,fc2
   end do
 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! get the number of contracted basis functions
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine readng(tmp,al,ng)
 implicit real(kind=8) (a-h,o-z)
 character*100     :: tmp
 character*2       :: al

 read(tmp,*)al,ng

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Get iname if it is not known
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine RdProg(imod,nprog,iname,tmp1,ierr)
 implicit real(kind=8) (a-h,o-z)
 character*10,allocatable :: pname(:)
 character*100     :: tmp1
 100  format(' >>> This MOLDEN file was generated by ',a,/)

 ierr = 0
 allocate(pname(nprog))

 ! MOLDEN files generated by the following programs require special modifications
 ! (The same length is needed by modern compilers!)
 pname = (/'ORCA      ',    &  !  1
           'CFOUR     ',    &  !  2
           'TURBOMOLE ',    &  !  3
           'JAGUAR    ',    &  !  4
           'ACES2     ',    &  !  5
           'MOLCAS    ',    &  !  6
           'PSI4      ',    &  !  7
           'MRCC      ',    &  !  8
           'NBO6      ',    &  !  9
           'CRYSTAL   '/)      ! 10
 ! < 0: same as 0 except the name of QC program is known.
 ! If MOLCAS uses spherical functions, iname will be reset to -6 later.

 ! iname is known
 if(iname >= 1 .and. iname <= nprog) then
   write(*,100)trim(pname(iname))
 else
   ! iname is unknown
   iname = 0
   rewind(imod)
   do while(.true.)
     read(imod,"(a100)",iostat=irdfin)tmp1
     if(irdfin /= 0) exit
     call charl2u(tmp1)

     ! [PROGRAM] block
     if(index(tmp1,'[PROGRAM]') /= 0) then
       ! FORMAT:
       ! [PROGRAM] pname
       !      or
       ! [PROGRAM]
       !   pname
       call trulen(tmp1,L1,L2,length)
       if(length == 9)then
         read(imod,"(a100)",end=8100)tmp1
         call charl2u(tmp1)
       else
         tmp1(L1:L1+8) = "         "
       end if
       call trulen(tmp1,L1,L2,length)
       do i=1,nprog
         if(index(tmp1(L1:L2),trim(pname(i))) /= 0)then
           iname=i
           write(*,100)trim(pname(iname))
           exit
         end if
       end do
       if(iname == 0)  goto 8200
     end if

     ! [TITLE] block
     if(index(tmp1,'[TITLE]') /= 0) then
       ! FORMAT:
       ! [TITLE]
       !  Molden file created by orca_2mkl for BaseName=...
       read(imod,"(a100)",end=8300)tmp1
       if(index(tmp1,"Molden file created by orca_2mkl for BaseName=") /= 0)then
         iname=1
         write(*,100)trim(pname(iname))
         exit
       end if
     end if
   end do
 end if

 deallocate(pname)
 return

 8100 write(*,"(' *** Error! Cannot get the name from [PROGRAM]!')")
 ierr = 1
 return
 8200 write(*,"(' *** Error! Unknown program ',a)") tmp1(L1:L2)
 ierr = 1
 return
 8300 write(*,"(' *** Error! Cannot get the title from [TITLE]!')")
 ierr = 1
 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Driver of ReOrdAtm
!
! This utility reorders atoms in the [Atoms] section (imode=0) or in the [GTO] section (imode=1).
! imode=0 has been tested only for MOLPRO's MOLDEN file, whereas
! imode=1 has been tested only for CFour's MOLDEN file (iprog=2)
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine ROADrv(imod1,imod2,igto,iprog,ctmp,ierr)
 implicit real(kind=8) (a-h,o-z)
 character*100 :: ctmp
 character*8   :: starint
 integer(kind=4),allocatable :: iord(:)

 ierr = 0
 imode = 0
 if(iprog == 2) imode = 1

 rewind(imod1)
 rewind(imod2)

 ! #atoms without and with dummy atoms
 call CountAtm(imod1,NAtm,NAtmX,ctmp,ierr)
  if(ierr == 1) return

 ! save new MOLDEN file
 write(imod2,"('[Molden Format]')")
 if(iprog == 2) write(imod2,"('[Program] CFour')")

 call searchar(imod1,7,"[ATOMS]",ctmp,ierr)
  if(ierr == 1) return
 ! NOTE: some programs do not recognize ANGS; Angs must be used instead
 if(index(ctmp,'ANGS')/=0)then
   write(imod2,"('[Atoms] Angs')")
 else
   write(imod2,"('[Atoms] AU')")
 end if

 if(imode == 0)then
   allocate(iord(NAtm))
   ! reference atomic ordering in [GTO]
   call RefOrd(imod1,NAtm,iord,ctmp,ierr)
    if(ierr == 1) return
   do i=1,NAtm
     call RdAtmI(imod1,imod2,NAtmX,iord(i),i,ctmp,ierr)
      if(ierr == 1) return
   end do
   deallocate(iord)
 else if(imode == 1)then
   call RdAtm(imod1,imod2,NAtm,ctmp,ierr)
    if(ierr == 1) return
 end if

 write(imod2,"('[GTO]')")
 call searchar(imod1,5,"[GTO]",ctmp,ierr)
  if(ierr == 1) return
 if(imode == 0)then
   i=0
   do while(.true.)
     read(imod1,"(100a)")ctmp
     if(len_trim(ctmp) == 0) cycle
     i = i +1
     write(imod2,"(i4,'  0')")i

     ! search the next empty line
     do while(.true.)
       read(imod1,"(100a)")ctmp
       write(imod2,"(a)")trim(ctmp)
       if(len_trim(ctmp) == 0) exit
     end do
     if(i == NAtm) exit
   end do
 else if(imode == 1)then
   call gtocpy(NAtm,imod1,igto)
   do i=1,NAtm
    write(starint,"('***',i5)") i
    call searchar(igto,8,starint,ctmp,ierr)
     if(ierr == 1) return
     write(imod2,"(i4,'  0')")i
     do while(.true.)
       read(igto,"(100a)")ctmp
       if(index(ctmp,"***") == 1) exit
       write(imod2,"(a)")trim(ctmp)
     end do
     write(imod2,*)
   end do
 end if

 write(imod2,"('[MO]')")
 call searchar(imod1,4,"[MO]",ctmp,ierr)
  if(ierr == 1) return
 do while(.true.)
   read(imod1,"(100a)",err=100,end=100)ctmp
   if(index(ctmp,'[') /= 0 .and. index(ctmp,']') /= 0) exit
   write(imod2,"(a)")trim(ctmp)
 end do

 100 return

 !---------------------------------------------------------------------- private subroutines
 contains

 subroutine chl2u(cha,length)
  implicit real(kind=8) (a-h,o-z)
  character*(*)     :: cha
  character*1       :: L2U

  do i=1, length
    cha(i:i)=L2U(cha(i:i))
  end do

  return
 end subroutine chl2u

 ! make a copy of basis functions (imod = 1)
 subroutine gtocpy(NAtm,imod1,igto)
  implicit none
  integer(kind=4) :: i,ia,NAtm,imod1,igto
  character*100 :: ctmp

  i=0
  do while(.true.)
    read(imod1,"(100a)")ctmp
    if(len_trim(ctmp) == 0) cycle
    i = i +1
    read(ctmp,*) ia
    write(igto,"('***',i5)") ia

  ! search the next empty line
    do while(.true.)
      read(imod1,"(100a)")ctmp
      if(len_trim(ctmp) == 0) exit
      write(igto,"(a)")trim(ctmp)
    end do
    if(i == NAtm) exit
  end do
  write(igto,"('***',i5)") 0

  return
 end subroutine gtocpy

 ! read & write coordinates
 subroutine RdAtm(imod1,imod2,NAtm,ctmp,ierr)
  implicit none
  integer(kind=4) :: imod1, imod2, NAtm, i, ia, iz, ierr
  real(kind=8) :: xyz(3)
  character*100 :: ctmp
  character*1,external :: L2U

  call searchar(imod1,7,"[ATOMS]",ctmp,ierr)
   if(ierr == 1) return

  do i=1,NAtm
    read(imod1,*)ctmp, ia, iz, xyz(1), xyz(2), xyz(3)
    if(ia /= i) goto 100
    ctmp(1:1) = L2U(ctmp(1:1))
    if(ctmp(1:1) == "X" .or. ctmp(1:1) == "Q" .or. iz <= 0) goto 200

    write(imod2,"(a4,2i5,3f20.10)")trim(ctmp), i, iz, xyz
  end do
  return

  100   write(*,"(' *** Error! This MOLDEN file is not supported!')")
  ierr = 1
  return
  200   write(*,"(' *** Error! Dummy & ghost atoms are not supported!')")
  ierr = 1
  return
 end subroutine RdAtm

 ! read coordinates of the iatm-th atom
 ! note that the atoms may be not ordered ascendingly.
 subroutine RdAtmI(imod1,imod2,NAtm,iold,inew,ctmp,ierr)
  implicit none
  integer(kind=4) :: imod1, imod2, NAtm, iold, inew, i, ia, iz, ierr
  real(kind=8) :: xyz(3)
  character*100 :: ctmp

  call searchar(imod1,7,"[ATOMS]",ctmp,ierr)
   if(ierr == 1) return

  do i=1,NAtm
    read(imod1,*)ctmp, ia, iz, xyz(1), xyz(2), xyz(3)
    if(ia == iold)then
      write(imod2,"(a4,2i5,3f20.10)")trim(ctmp), inew, iz, xyz
      return
    end if
  end do

  write(*,"(' *** Error! Atom(',i4,') cannot be found.')") iold
  ierr = 1
  return
 end subroutine RdAtmI

 ! atomic ordering in [GTO] (imod = 0)
 subroutine RefOrd(imod1,NAtm,iord,ctmp,ierr)
  implicit none
  integer(kind=4) :: imod1, NAtm, iord(*), iatm=0, ierr
  character*100 :: ctmp

  call searchar(imod1,5,"[GTO]",ctmp,ierr)
   if(ierr == 1) return

  do while(.true.)
    read(imod1,"(100a)",err=100,end=100)ctmp
    if(len_trim(ctmp) == 0) cycle
    iatm = iatm +1
    read(ctmp,*) iord(iatm)
    if(iatm == NAtm) return

  ! search the next empty line
    do while(.true.)
      read(imod1,"(100a)",err=100,end=100)ctmp
      if(len_trim(ctmp) == 0) exit
    end do
  end do
  return

  100 write(*,"(/,' *** Error! See [GTO]!')")
  ierr = 1
  return
 end subroutine RefOrd

 ! search [Atoms] and count the number of atoms
 subroutine CountAtm(imod1,NAtm,NAtmX,ctmp,ierr)
  implicit none
  integer(kind=4) :: imod1, NAtm, NAtmX, i1, i2, ierr
  character*100 :: ctmp
  character*3 :: Elm

  call searchar(imod1,7,"[ATOMS]",ctmp,ierr)
   if(ierr == 1) return

  NAtm=0
  NAtmX=0
  do while(.true.)
    read(imod1,"(100a)",err=100,end=100)ctmp
    if(index(ctmp,'[')/=0 .and. index(ctmp,']')/=0) exit
    if(len_trim(ctmp) == 0) cycle
    NAtmX = NAtmX +1
    ! Dummy atom or ghost atom?
    read(ctmp,*) Elm, i1, i2
    call chl2u(Elm,len_trim(Elm))
    if(Elm(1:1) == "X" .or. Elm(1:1) == "Q" .or. i2 == 0) cycle

    NAtm = NAtm +1
  end do

  100  if(NAtm < 1)then
    write(*,"(' *** Error! NAtm < 1.')")
    ierr = 1
  end if

  return
 end subroutine CountAtm

 ! search LOGO
 subroutine searchar(imod1,lenth,LOGO,ctmp,ierr)
  implicit none
  integer(kind=4) :: imod1, lenth, ierr
  character*(*) :: LOGO
  character*100 :: ctmp

  rewind(imod1)

  do while(.true.)
    read(imod1,"(100a)",err=100,end=100)ctmp
    call charl2u(ctmp)
    if(index(ctmp,LOGO(1:lenth))/=0) return
  end do

  return

  100 write(*,"(' *** Error! ',a,/,' was not found!')") trim(LOGO(1:lenth))
  ierr = 1
  return
 end subroutine searchar

end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Get the following numbers
! 1. the number of atoms (natm) and the number of nuclear charges (nchar) in the [Atoms] block
! 2. the number of basis functions (nbas) and the number of MOs (nmo) in the [MO] block
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine countmo(imtm,iatm,natm,nchar,nbas,nmo,tmp1,ierr)
 implicit real(kind=8) (a-h,o-z)
 character*100     :: tmp1

 ierr = 0

 !---------------------------------------------------------------------- natm & nchar
 rewind(imtm)
 rewind(iatm)

 do while(.true.)
   read(imtm,"(a100)",end=7000)tmp1
   call charl2u(tmp1)

   ! [ATOMS] block
   if(index(tmp1,'[ATOMS]') /= 0) then
     if(index(tmp1,'ANGS') /= 0) then
       write(iatm,"('1')")
     else if(index(tmp1,'AU') /= 0) then
       write(iatm,"('0')")
     else
       goto 7100
     end if
     exit
   end if
 end do

 natm = 0
 nchar= 0
 do while(.true.)
   read(imtm,"(a100)",iostat=irdfin)tmp1
   if(len_trim(tmp1) == 0 .or. irdfin /= 0) exit
   if(index(tmp1,'[') /= 0 .and. index(tmp1,']') /= 0) exit
   natm = natm + 1
   call RdCoord(tmp1,ia,iz,x,y,z,ierr)
     if(ierr == 1) goto 9999
   write(iatm,"(i4,3f22.12)") iz,x,y,z
   nchar=nchar+iz
   ! check compatibility: because of some uncertainties in the MOLDEN format, the atomic indices must be 1,2,3,4,...
   ! (this should have been corrected in sub. ROADrv)
   if(ia /= natm) goto 7200
 end do

 if(natm < 1) goto 7300

 !---------------------------------------------------------------------- nbas & nmo
 rewind(imtm)
 do while(.true.)
   read(imtm,"(a100)",end=8000)tmp1
   call charl2u(tmp1)

   ! [MO] block
   if(index(tmp1,'[MO]') /= 0) exit
 end do

 nbas = 0
 ibas = 0
 nmo = 0
 do while(.true.)
   read(imtm,"(a100)",iostat=irdfin)tmp1
   if(len_trim(tmp1) == 0 .or. irdfin /= 0) exit
   if(index(tmp1,'[') /= 0 .and. index(tmp1,']') /= 0) exit
   call charl2u(tmp1)

   if(index(tmp1,'=') /= 0)then
     if(index(tmp1,'OCCUP') /= 0) then
       nmo = nmo + 1
       nbas = max(nbas,ibas)
     end if
     cycle
   end if

   read(tmp1,*) ibas
 end do
 nbas = max(nbas,ibas)
 if(nbas < 1) goto 8100
 if(nmo < 1) goto 8200

 9999  return

 7000 write(*,"(' *** Error! [Atoms] can not be found!')")
 ierr = 1
 return
 7100 write(*,"(' *** Error! The unit of the coordinates is not defined!')")
 ierr = 1
 return
 7200 write(*,"(' *** Error: atoms in [Atoms] are not ordered ascendingly.')")
 ierr=1
 return
 7300 write(*,"(' *** Error! No atoms found!',/,' Please check your MOLDEN file.')")
 ierr = 1
 return
 8000 write(*,"(' *** Error! [MO] can not be found!')")
 ierr = 1
 return
 8100 write(*,"(' *** Error! NBas < 1 in the [MO] block!')")
 ierr = 1
 return
 8200 write(*,"(' *** Error! NMO < 1 in the [MO] block!')")
 ierr = 1
 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! b(*) = a(*)
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine acopy(n,a,b)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: a(n),b(n)

 b = a

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! swap a and b
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine swap(a,b)
 implicit real(kind=8) (a-h,o-z)

 scr=a
 a=b
 b=scr

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Routine to clear N elements in array A.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine AClear(N,A)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: A(N)

 A = 0.0d0

 Return
End

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! C(*) = A(*) + B(*)
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine AAdd(N,A,B,C)
 implicit real(kind=8) (a-h,o-z)
 real(kind=8) :: A(N),B(N),C(N)

  C = A + B

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Removal of numerical noise before printing array A.
! Some compilers print stars (***) if ABS(A(i)) <= 1.0d-100.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine noiserm(A,N)
 implicit real(kind=8) (a-h,o-z)
 parameter(eps=1.0d-20)
 dimension         :: A(N)

 do i=1,N
   if(abs(A(i)) < eps) A(i)=0.0d0
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! vout = c * vin
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine ascale(n,c,vin,vout)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: vin(n),vout(n)

 vout = c * vin

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! distance between two points
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dist(p1,p2)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: p1(3),p2(3)

 dist=0.d0
 do i=1,3
   dist=dist+(p1(i)-p2(i))**2
 end do
 dist=sqrt(dist)

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! It returns the position of element (J,I) in a L.T. matrix.
! J must be .LE. I. It doesn't work if J > I.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function IdxNTT(J,I)
 implicit real(kind=8) (a-h,o-z)

 IdxNTT=J+(I-1)*I/2

 Return
End

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! symmetric square matrix --> lower triangular matrix
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine Sqr2Tr(N,S,T)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: S(N,N),T(*)

 ii=0
 Do i=1,N
   Do j=1,i
     ii=ii+1
     T(ii)=(S(j,i)+S(i,j))*0.5d0
   end Do
 end Do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! lower triangular matrix --> symmetric square matrix
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subroutine LT2Sqr(N,T,S)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: T(*),S(N,N)

 k=0
 do i=1,N
   do j=1,i-1
     k=k+1
     S(j,i)=T(k)
     S(i,j)=T(k)
   end do
   k=k+1
   S(i,i)=T(k)
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! v * S * v', S is a symmetric L.T. matrix
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function VSV(N,v,s)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: v(N),s(*)

 p=0.d0
 do i=1,N
   k=IdxNTT(1,i)-1
   do j=1,i-1
     k=k+1
     q=v(j)*s(k)*v(i)
     p=p+q+q
   end do
   k=k+1
   q=v(i)*s(k)*v(i)
   p=p+q
 end do
 VSV=p

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Trace of the product of lower triangular matrices A and B.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function TracLL(N,A,B)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: A(*), B(*)

 Su = 0.0d0
 Do I = 1, N
   II = (I*(I-1))/2
   Do J = 1, I-1
     Su = Su + (A(II+J)+A(II+J))*B(II+J)
   End Do
   Su = Su + A(II+I)*B(II+I)
 End Do

 TracLL = Su

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! C = C + A * B, where A and B are symmetric in L.T. and C is in Sq. The following relationship is adopted:
! M_sq(j,i) = M_lt(j+(i-1)*i/2) if j <= i
!           = M_lt(i+(j-1)*j/2) if j > i
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine LTxLT(N,A,B,C)
 implicit real(kind=8) (a-h,o-z)
 dimension         :: A(*),B(*),C(N,N)

 do i = 1,N
   i0 = i*(i-1)/2
   X = A(i+i0)
   do j = 1,i
     C(j,i) = X * B(j+i0)
   end do
   do j = i + 1,N
     j0 = j*(j-1)/2
     C(j,i) = X * B(i+j0)
   end do
   do j = 1,i - 1
     j0 = j*(j-1)/2
     X = A(j+i0)
     do k = 1,j
       C(k,i) = C(k,i) + X * B(k+j0)
     end do
     do k = j+1,N
       k0 = k*(k-1)/2
       C(k,i) = C(k,i) + X * B(j+k0)
     end do
   end do
   do j = i + 1,N
     j0 = j*(j-1)/2
     X = A(i+j0)
     do k = 1,j
       C(k,i) = C(k,i) + X * B(k+j0)
     end do
     do k = j+1,N
       k0 = k*(k-1)/2
       C(k,i) = C(k,i) + X * B(j+k0)
     end do
   end do
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Mode = 1: C = A * B
!        2: C = A^T * B
!        3: C = A * B^T
!        4: C = A^T * B^T
! where C(MxN), op(A)(MxL), and op(B)(LxN)
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine MatMult(Mode,M,L,N,A,B,C)
 implicit real(kind=8) (a-h,o-z)
 parameter(Zero=0.d0,One=1.d0)
 dimension         :: A(*),B(*),C(*)

 LDC=M
 if(Mode == 1)then
   LDA=M
   LDB=L
   call DGEMM('N','N',M,N,L,One,A,LDA,B,LDB,Zero,C,LDC)
 else if(Mode == 2)then
   LDA=L
   LDB=L
   call DGEMM('T','N',M,N,L,One,A,LDA,B,LDB,Zero,C,LDC)
 else if(Mode == 3)then
   LDA=M
   LDB=N
   call DGEMM('N','T',M,N,L,One,A,LDA,B,LDB,Zero,C,LDC)
 else if(Mode == 4)then
   LDA=L
   LDB=N
   call DGEMM('T','T',M,N,L,One,A,LDA,B,LDB,Zero,C,LDC)
 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! This subroutine is taken from BLAS (version 3.4.0, November 2011).
! BLAS is a software package provided by Univ. of Tennessee, Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
! Scalar Arguments ..
 DOUBLE PRECISION ALPHA,BETA
 INTEGER K,LDA,LDB,LDC,M,N
 CHARACTER TRANSA,TRANSB
! Array Arguments ..
 DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
! Intrinsic Functions ..
 INTRINSIC MAX
! Local Scalars ..
 DOUBLE PRECISION TEMP
 INTEGER I,INFO,J,L,NCOLA,NROWA,NROWB
 LOGICAL NOTA,NOTB
! Parameters ..
 DOUBLE PRECISION ONE,ZERO
 PARAMETER (ONE=1.0D+0,ZERO=0.0D+0)

! Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not transposed and set  NROWA, NCOLA and  NROWB
!  as the number of rows and  columns of  A  and the  number of  rows  of  B  respectively.
 NOTA = TRANSA == 'N'
 NOTB = TRANSB == 'N'
 IF (NOTA) THEN
     NROWA = M
     NCOLA = K
 ELSE
     NROWA = K
     NCOLA = M
 END IF
 IF (NOTB) THEN
     NROWB = K
 ELSE
     NROWB = N
 END IF

! Test the input parameters.
 INFO = 0
 IF ((.NOT.NOTA) .AND. (.NOT.(TRANSA == 'C')) .AND. (.NOT.(TRANSA == 'T'))) THEN
     INFO = 1
 ELSE IF ((.NOT.NOTB) .AND. (.NOT.(TRANSB == 'C')) .AND. (.NOT.(TRANSB == 'T'))) THEN
     INFO = 2
 ELSE IF (M < 0) THEN
     INFO = 3
 ELSE IF (N < 0) THEN
     INFO = 4
 ELSE IF (K < 0) THEN
     INFO = 5
 ELSE IF (LDA < MAX(1,NROWA)) THEN
     INFO = 8
 ELSE IF (LDB < MAX(1,NROWB)) THEN
     INFO = 10
 ELSE IF (LDC < MAX(1,M)) THEN
     INFO = 13
 END IF
 IF (INFO /= 0) THEN
     WRITE( *, FMT = 9999 ) INFO
     call estop(1)
 END IF

! Quick return if possible.
 IF ((M == 0) .OR. (N == 0) .OR. (((ALPHA == ZERO).OR. (K == 0)).AND. (BETA == ONE))) RETURN
! And if  alpha == zero.
 IF (ALPHA == ZERO) THEN
     IF (BETA == ZERO) THEN
         DO J = 1,N
             DO I = 1,M
                 C(I,J) = ZERO
             end do
         end do
     ELSE
         DO J = 1,N
             DO I = 1,M
                 C(I,J) = BETA*C(I,J)
             end do
         end do
     END IF
     RETURN
 END IF

! Start the operations.
 IF (NOTB) THEN
     IF (NOTA) THEN

!        Form  C := alpha*A*B + beta*C.
         DO J = 1,N
             IF (BETA == ZERO) THEN
                 DO I = 1,M
                     C(I,J) = ZERO
                 end do
             ELSE IF (BETA /= ONE) THEN
                 DO I = 1,M
                     C(I,J) = BETA*C(I,J)
                 end do
             END IF
             DO L = 1,K
                 IF (B(L,J) /= ZERO) THEN
                     TEMP = ALPHA*B(L,J)
                     DO I = 1,M
                         C(I,J) = C(I,J) + TEMP*A(I,L)
                     end do
                 END IF
             end do
         end do
     ELSE

!        Form  C := alpha*A**T*B + beta*C
         DO J = 1,N
             DO I = 1,M
                 TEMP = ZERO
                 DO L = 1,K
                     TEMP = TEMP + A(L,I)*B(L,J)
                 end do
                 IF (BETA == ZERO) THEN
                     C(I,J) = ALPHA*TEMP
                 ELSE
                     C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                 END IF
             end do
         end do
     END IF
 ELSE
     IF (NOTA) THEN
!        Form  C := alpha*A*B**T + beta*C
         DO J = 1,N
             IF (BETA == ZERO) THEN
                 DO I = 1,M
                     C(I,J) = ZERO
                 end do
             ELSE IF (BETA /= ONE) THEN
                 DO I = 1,M
                     C(I,J) = BETA*C(I,J)
                 end do
             END IF
             DO L = 1,K
                 IF (B(J,L) /= ZERO) THEN
                     TEMP = ALPHA*B(J,L)
                     DO I = 1,M
                         C(I,J) = C(I,J) + TEMP*A(I,L)
                     end do
                 END IF
             end do
         end do
     ELSE
!        Form  C := alpha*A**T*B**T + beta*C
         DO J = 1,N
             DO I = 1,M
                 TEMP = ZERO
                 DO L = 1,K
                     TEMP = TEMP + A(L,I)*B(J,L)
                 end do
                 IF (BETA == ZERO) THEN
                     C(I,J) = ALPHA*TEMP
                 ELSE
                     C(I,J) = ALPHA*TEMP + BETA*C(I,J)
                 END IF
             end do
         end do
     END IF
 END IF

 RETURN
 9999 FORMAT( ' ** On entry to DGEMM parameter number ', I2, ' had an illegal value' )
END

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Read an atom and its coordinates
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine RdCoord(tmp,ia,iz,x,y,z,ierr)
 implicit real(kind=8) (a-h,o-z)
 character*100     :: tmp
 character*3       :: element

 ! read: atomic_name number atomic_number x y z
 ! NOTE: core electrons by ECP may be not inclued in iz (eg. in Molpro 2015), so iz should be obtained from atomic name
 read(tmp,*,err=1020,end=1020)element,ia,iz0,x,y,z
 ! eliminate non-letter characters for a special case, for example, O_1 (Dalton), C1 (Cadpac)
 call rmnumb(3,element)
 call ElemZA(0,element,za,za)
 ! In the Molden program, atomic_name can be any characters and is omitted, so iz can be 0. In this case read iz from iz0.
 iz = nint(za)
 if(iz == 0) iz = iz0

 if(iz <= 0) then
   write(*,"(' *** Error! Dummy atom and ghost atom are not supported!')")
   ierr=1
 end if

 !write(*,"(a3,1x,2i6,3f24.12)")element,ia,iz,x,y,z
 return

 ! In the geometry optimization of deMon2k, the format is "atom, x, y, z" by mistake
 1020  write(*,*)"*** Error when reading the [ATOMS] section!"
 ierr=1
 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! define file names
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine filename(imod,fmdn,fwfn,fwfx,fnbo)
 implicit real(kind=8) (a-h,o-z)
 character*157     :: fwfn,fwfx,fnbo,fmod(2)
 character*164     :: fmdn
 character*7       :: exten(8)
 data exten/'.mol   ','.MOL   ','.mold  ','.MOLD  ','.molden','.MOLDEN','.gab   ','.GAB   '/

 write(*,"(/)")
 100  write(*,"(' Type in the MOLDEN/GABEDIT file name within 150 characters:',/,  &
        ' (extension mol/mold/molden/gab can be omitted; default: MOLDEN)',/,' > ',$)")
 read(*,"(a50)")fmod(1)(:)
 lstr=nonspace(fmod(1)(:))
 lend=LEN_TRIM(fmod(1)(:))
 if(lend == 0)then                 ! use default file name
   lstr=1
   lend=6
   fmod(1)(1:6)='MOLDEN'
 end if
 open(imod,file=fmod(1)(lstr:lend),status='old',err=110)
 iinp=1
 goto 300
 110  if(fmod(1)(lend:lend) == '.')lend=lend-1
 iinp=2
 do i=1,8
   fmod(2)(:)=fmod(1)(lstr:lend)//trim(exten(i))
   open(imod,file=fmod(2)(:),status='old',err=120)
   goto 300
 120  continue
 end do
 write(*,"(//,' ### Wrong! These MOLDEN/GABEDIT files do not exist!')")
 write(*,"(1x,a)")fmod(1)
 do i=1,8
   write(*,"(1x,a)")fmod(1)(lstr:lend)//trim(exten(i))
 end do
 write(*,"(/,' Please try again.',/)")
 goto 100

 300  write(*,"(/,' The MOLDEN/GABEDIT file ',a,' has been found.',/,1x,77('_'),/)") trim(fmod(iinp))
! define the *.wfn/wfx/47 file name
 lend2=index(fmod(iinp),'.',.true.)
 if(lend2 > 1) lend = lend2-1
 fmdn=fmod(iinp)(lstr:lend)//'_new.molden'
 fwfn=fmod(iinp)(lstr:lend)//'.wfn'
 fwfx=fmod(iinp)(lstr:lend)//'.wfx'
 fnbo=fmod(iinp)(lstr:lend)//'.47'

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Read user's initialization parameters from m2a.ini
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine usrini(iini,nprog,ICntrl,ICln,IAllMO,iprog,nosupp,ledt,lpspin,lspout,iunknw,tolocc,nbopro,ctmp)
 implicit real(kind=8) (a-h,o-z)
 parameter(nkey=18)
 Dimension         :: ICntrl(8)
 character*100     :: ctmp
 character*9       :: keyword(nkey)
 data keyword/"MOLDEN=","WFN=","WFX=","NBO=","WFNCHECK=","WFXCHECK=","NBOCHECK=","WBO=","PROGRAM=","CLEAR=","ALLMO=",  &
   "NOSUPP=","RDCORE=","EDFTYP=","PRSPIN=","CARSPH=","NBOPRO=","UNKNOWN="/

 open(iini,file='m2a.ini',status='old',err=9000)
 write(*,"(/,' m2a.ini has been loaded.')")
 rewind(iini)

 do while(.true.)
   read(iini,"(a100)",iostat=irdfin)ctmp
   if(irdfin /= 0) exit

   if(ctmp(1:1) == "$" .or. ctmp(1:1) == "%" .or. ctmp(1:1) == "!" .or. len_trim(ctmp) == 0) cycle

   call charl2u(ctmp)
   ikey=0
   do i=1,nkey
     if(index(ctmp,trim(keyword(i))) /= 0) then
       ikey=i
       istart=index(ctmp,"=")+1
       read(ctmp(istart:100),*)keyvalue
       exit
     end if
   end do

   if(ikey == 0) cycle

   select case(ikey)
     case(1:8)
       if(keyvalue > 0) then
         keyvalue=1
       else if(keyvalue < 0) then
         keyvalue=-1
       end if
       ICntrl(ikey)=keyvalue
     case(9)
       if(keyvalue < 0 .or. keyvalue > nprog) keyvalue=0
       iprog=keyvalue
     case(10)
       if(keyvalue /= 0) keyvalue=1
       ICln=keyvalue
     case(11)
       if(keyvalue > 0)then
         keyvalue=1
       else if(keyvalue < 0)then
         keyvalue=-1
       end if
       IAllMO=keyvalue
     case(12)
       nosupp = 0
       if(keyvalue /= 0) nosupp = 1
     case(13)
       ! do nothing
       !lrdecp = 0
       !if(keyvalue /= 0) lrdecp = 1
     case(14)
       ledt = 0
       if(keyvalue == 1) ledt = 1
     case(15)
       lpspin = 1
       if(keyvalue == 0) lpspin = 0
     case(16)
       lspout = 0
       if(keyvalue == 1) lspout = 1
     case(17)
       nbopro = 1
       if(keyvalue /= 1) nbopro = 0
     case(18)
       iunknw = 0
       if(keyvalue /= 0) iunknw = 1
   end select
 end do
 close(iini)

 9000  continue
 if(IAllMO == 0)then
   tolocc=5.0d-8
 else if(IAllMO > 0)then
   tolocc=-1.0d2
 else
   tolocc=1.0d-1 - 5.0d-8
 end if

 if(nbopro == 1) lspout = 1

 return
End

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! Mode = 0 : returns nuclear charge zchar for an element symbol "el"
!            iza is not used.
!   /= 0 : returns element symbol "el" for nuclear charge iza
!            zchar is not used.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine ElemZA(Mode,el,iza,zchar)
 implicit real(kind=8) (a-h,o-z)
 parameter (maxza=120)
 character*3       :: el,atomlib(maxza)
 data (atomlib(i),i=1,maxza) /  &
   'H  ','HE ','LI ','BE ','B  ','C  ','N  ','O  ','F  ','NE ',   'NA ','MG ','AL ','SI ','P  ','S  ','CL ','AR ','K  ','CA ',  &
   'SC ','TI ','V  ','CR ','MN ','FE ','CO ','NI ','CU ','ZN ',   'GA ','GE ','AS ','SE ','BR ','KR ','RB ','SR ','Y  ','ZR ',  &
   'NB ','MO ','TC ','RU ','RH ','PD ','AG ','CD ','IN ','SN ',   'SB ','TE ','I  ','XE ','CS ','BA ','LA ','CE ','PR ','ND ',  &
   'PM ','SM ','EU ','GD ','TB ','DY ','HO ','ER ','TM ','YB ',   'LU ','HF ','TA ','W  ','RE ','OS ','IR ','PT ','AU ','HG ',  &
   'TL ','PB ','BI ','PO ','AT ','RN ','FR ','RA ','AC ','TH ',   'PA ','U  ','NP ','PU ','AM ','CM ','BK ','CF ','ES ','FM ',  &
   'MD ','NO ','LR ','RF ','DB ','SG ','BH ','HS ','MT ','DS ',   'RG ','CN ','NH ','FL ','MC ','LV ','TS ','OG ','UUE','UBN'/
 save atomlib

 if (Mode == 0) then

   call charl2u(el)
   zchar = 0.d0
   do i=1,maxza
     if(index(el,atomlib(i)) /= 0)then
       zchar = dble(i)
       exit
     end if
   end do

 else

   el = "???"
   if(iza > 0 .and. iza <= maxza) el = adjustl(atomlib(iza))
   call charu2l(el(2:3))

 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! print supporting information
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine SuppInf
 implicit real(kind=8) (a-h,o-z)

 write(*,"(' Supported programs:',/,                                                   &
   '  1) ACES-II 2.9 (fix reorder.F, and insert [PROGRAM] ACES2 into MOLDEN file)',/,  &
   '  2) BDF-G (thanks to Dr. Bingbing Suo for testing)',/,                            &
   '  3) CADPAC',/,                                                                    &
   '  4) CFour (fix reorderdf.F, and insert [PROGRAM] CFOUR into MOLDEN file)',/,      &
   '  5) Columbus (Natural orb. from MCSCF or CI with Cartesian SPDF functions;',/,    &
   '     thanks to Dr. Marat Talipov for testing)',/,                                  &
   '  6) Crystal (0D-Molecule with SPDF GTOs; insert [PROGRAM] Crystal into MOLDEN)',/,&
   '  7) Dalton (> 2013; HF/DFT/MP2/MCSCF with spherical functions)',/,                &
   '  8) deMon2k',/,                                                                   &
   '  9) Gabedit (the GAB file is compatible)',/,                                      &
   ' 10) MOLCAS (for Cart. functions, insert [PROGRAM] MOLCAS into MOLDEN file)',/,    &
   ' 11) Molden (the Molden program can read MOs from the output file of some',/,      &
   '     QC programs, and save a MOLDEN file)',/,                                      &
   ' 12) MOLPRO',/,                                                                    &
   ' 13) MRCC (for Cart. functions, insert [PROGRAM] MRCC into MOLDEN file)',/,        &
   ' 14) MultiWFN (it can read the fchk file of Gaussian and Q-Chem, and save',/,      &
   '     a MOLDEN file)',/,                                                            &
   ' 15) NBO6 (> May.2014, insert [PROGRAM] NBO6 into MOLDEN file)',/,                 &
   ' 16) NWChem (>= 6.8) by MOLDEN_NORM JANPA or NONE',/,                              &
   ' 17) ORCA (insert [PROGRAM] ORCA into MOLDEN file)',/,                             &
   ' 18) Priroda (thanks to Dr. Evgeniy Pankratyev for testing)',/,                    &
   ' 19) PSI4 (spherical functions only; insert [PROGRAM] PSI4 into MOLDEN file)',/,   &
   ' 20) PySCF',/,                                                                     &
   ' 21) Q-Chem (spherical SPDF or Cartesian SPD basis functions)',/,                  &
   ' 22) StoBe',/,                                                                     &
   ' 23) TeraChem (SPDF basis functions)',/,                                           &
   ' 24) Turbomole (insert [PROGRAM] TURBOMOLE into MOLDEN file)'                      &
   )")

 write(*,"(/,' Programs to be tested:      1) Bagel      2) NRLMOL     3) SeqQuest')")

 write(*,"(/,' Unsupported programs:       1) ADF        2) Jaguar')")

 call xcontinue

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! create m2a.ini
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine crtini(iini)
 implicit real(kind=8) (a-h,o-z)

 open(iini,file='m2a.ini',status='old',err=100)
 close(iini)
 return

 100  continue
 open(iini,file='m2a.ini',status='new')
 rewind(iini)

 write(iini,"('########################################################################')")
 write(iini,"('#  In the following 8 parameters,')")
 write(iini,"('#     >0:  always performs the operation without asking the user')")
 write(iini,"('#     =0:  asks the user whether to perform the operation')")
 write(iini,"('#     <0:  always neglect the operation without asking the user')")
 write(iini,"('molden= 0           ! Generating a standard Molden file in Cart. function')")
 write(iini,"('wfn= 0              ! Generating a WFN file')")
 write(iini,"('wfncheck= 0         ! Checking normalization for WFN')")
 write(iini,"('wfx= 0              ! Generating a WFX file (not implemented)')")
 write(iini,"('wfxcheck= 0         ! Checking normalization for WFX (not implemented)')")
 write(iini,"('nbo= 0              ! Generating an NBO .47 file')")
 write(iini,"('nbocheck= 0         ! Checking normalization for NBO .47 file')")
 write(iini,"('wbo= 0              ! GWBO after the .47 file being generated')")
 write(iini,*)
 write(iini,"('########################################################################')")
 write(iini,"('#  Which quantum chemistry program is used to generate the MOLDEN file?')")
 write(iini,"('#  1: ORCA, 2: CFOUR, 3: TURBOMOLE, 4: JAGUAR (not supported),')")
 write(iini,"('#  5: ACES2, 6: MOLCAS, 7: PSI4, 8: MRCC, 9: NBO 6 (> ver. 2014),')")
 write(iini,"('#  10:CRYSTAL (0D only),')")
 write(iini,"('#  0: other programs, or read [Program] xxx from MOLDEN.')")
 write(iini,"('#')")
 write(iini,"('#  If a non-zero value is given, [Program] xxx in MOLDEN will be ignored.')")
 write(iini,"('#')")
 write(iini,"('program=0')")
 write(iini,*)
 !write(iini,"('########################################################################')")
 !write(iini,"('#  For ECP: read core information from Molden file')")
 !write(iini,"('#<=0: if the total_occupation_number is smaller than the total_Za, ask')")
 !write(iini,"('#     the user whether to read core information')")
 !write(iini,"('# >0: always search and read core information')")
 !write(iini,"('rdcore=0')")
 !write(iini,*)
 write(iini,"('########################################################################')")
 write(iini,"('#  Type of EDF')")
 write(iini,"('# =0: EDF by X2C/HF')")
 write(iini,"('# =1: EDF by X2C/PBE0')")
 write(iini,"('edftyp=0')")
 write(iini,*)
 write(iini,"('########################################################################')")
 write(iini,"('#  Which orbirals will be printed in the new-MOLDEN, WFN, and WFX files?')")
 write(iini,"('# =0: print only the orbitals with occ. number > 5.0d-8')")
 write(iini,"('# <0: print only the orbitals with occ. number > 0.1 (debug only)')")
 write(iini,"('# >0: print all the orbitals')")
 write(iini,"('allmo=1')")
 write(iini,*)
 write(iini,"('########################################################################')")
 write(iini,"('#  Used for WFN only')")
 write(iini,"('# .ne. 0: print $MOSPIN at the end of the file')")
 write(iini,"('prspin=1')")
 write(iini,*)
 write(iini,"('########################################################################')")
 write(iini,"('#  Used for WFX only')")
 write(iini,"('# =0: print UNKNOWN for Energy and Virial Ratio')")
 write(iini,"('# .ne. 0: print 0.0 for Energy and 2.0 for Virial Ratio')")
 write(iini,"('unknown=1')")
 write(iini,*)
 write(iini,"('########################################################################')")
 write(iini,"('#  Used for new Molden and NBO-47 only. Cartesian vs spherical functions')")
 write(iini,"('# =0: (default) print MOs in Cartesian functions')")
 write(iini,"('# =1: determined by the program')")
 write(iini,"('carsph=0')")
 write(iini,*)
 write(iini,"('########################################################################')")
 write(iini,"('#  Used for NBO-47 only.')")
 write(iini,"('# =1: More data will be printed if possible (carsph will be reset to 1)')")
 write(iini,"('nbopro=0')")
 write(iini,*)
 write(iini,"('########################################################################')")
 write(iini,"('#  Close supporting information or not')")
 write(iini,"('# <0: print; =0: asks the user; >0: do not print')")
 write(iini,"('nosupp=1')")
 write(iini,*)
 write(iini,"('########################################################################')")
 write(iini,"('#  The following parameters are used only for debugging.')")
 write(iini,"('clear=1            ! delete temporary files (1) or not (0)')")
 write(iini,*)
 write(iini,"('########################################################################')")
 write(iini,*)

 close(iini)
 write(*,"(/,' m2a.ini has been created.')")

 return
End

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! wfx: print a label
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine wfxlab(iwfx,mode,label)
 implicit real(kind=8) (a-h,o-z)
 character*(*)     :: label

 if(mode == 0) then
   write(iwfx,"('<',a,'>')")label
 else
   write(iwfx,"('</',a,'>')")label
 end if

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! print a head
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine headprt(ver,dt)
 implicit real(kind=8) (a-h,o-z)
 character*10      :: dt
 character*5       :: ver

 write(*,"(//,1x,77('*'),/,31x,            '*  Molden2AIM  *',/,26x,       'Version ',a5,',  ',a10,/,  &
   6x,'It converts the format from MOLDEN to AIM-WFN, AIM-WFX, and NBO-47.',/,1x,77('*'),/)")ver,dt
 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! replace non-letter characters by space
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine rmnumb(N,cha)
 implicit real(kind=8) (a-h,o-z)
 character*(*)     :: cha

 do i=1,N
   if((ichar(cha(i:i)) >= 65) .and. (ichar(cha(i:i)) <= 90)) cycle     ! A-Z
   if((ichar(cha(i:i)) >= 97) .and. (ichar(cha(i:i)) <= 122)) cycle    ! a-z
   cha(i:i)=' '
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! cha --> CHA
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine charl2u(cha)
 implicit real(kind=8) (a-h,o-z)
 character*(*)     :: cha
 character*1       :: L2U

 do i=1, len_trim(cha)
   cha(i:i)=L2U(cha(i:i))
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! CHA --> cha
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine charu2l(cha)
 implicit real(kind=8) (a-h,o-z)
 character*(*)     :: cha
 character*1       :: U2L

 do i=1, len_trim(cha)
   cha(i:i)=U2L(cha(i:i))
 end do

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! l --> L
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L2U(letter)
 implicit real(kind=8) (a-h,o-z)
 character*1       :: letter,L2U

 if((ichar(letter) >= 97).and.(ichar(letter) <= 122))then
   L2U=char(ichar(letter)-32)
 else
   L2U=letter
 endif

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! L --> l
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function U2L(letter)
 implicit real(kind=8) (a-h,o-z)
 character*1       :: letter,U2L

 if((ichar(letter) >= 65).and.(ichar(letter) <= 90))then
   U2L=char(ichar(letter)+32)
 else
   U2L=letter
 endif

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! position of the first non-space character
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function nonspace(string)
 implicit real(kind=8) (a-h,o-z)
 character*(*)     :: string
 character*1       :: space

 space=' '
 length=LEN_TRIM(string)
 if(length == 0) then
  i=1
 else
  do i=1,length
    if(string(i:i) /= space) goto 20
  end do
 endif

 20  nonspace=i

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! lenchar: length of a string without tail spaces.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lenchar(string)
 implicit real(kind=8) (a-h,o-z)
 character*(*)     :: string
 character*1       :: space

 space=' '
 i=len(string)

 do while(i > 0 .and. string(i:i) == space)
   i=i-1
 end do

 lenchar=i

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! trulen: true length of a string
! len1: pointor to the first non-space character
! len2: pointor to the last non-space character
! length: true length of a string without the initial and tail spaces
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine trulen(cha,len1,len2,length)
 implicit real(kind=8) (a-h,o-z)
 character*(*)     :: cha

 len1=nonspace(cha)
 len2=lenchar(cha)
 length=len2-len1+1

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! read an <ENTER> and (if imode /= 0) stop.
! stop can trigger a message "Warning: ieee_inexact is signaling" by pgf90.
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine estop(imode)
 implicit real(kind=8) (a-h,o-z)

 write(*,"(//,' Press <ENTER> to exit',/)")
 read(*,*)

 if(imode /= 0) stop

 return
end

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!
! read an <ENTER> to continue
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Subroutine xcontinue
 implicit real(kind=8) (a-h,o-z)

 write(*,"(/,' Press <ENTER> to continue')")
 read(*,*)

 return
end

!--- END
