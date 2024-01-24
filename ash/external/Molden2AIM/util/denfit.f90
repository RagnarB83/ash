!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%%%
!%%% DenFit: a program to fit atomic density using Gaussian s-functions. (2019.04.02)
!%%%
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%%%
!%%% Reference
!%%%
!%%% [1] E. J. Baerends, D. E. Ellis, P. Ros, Chem. Phys. 2, 41 (1973).
!%%%
!%%% [2] B. I. Dunlap, J. W. D. Connolly, J. R. Sabin, J. Chem. Phys. 71, 3396 (1979).
!%%%
!%%% [3] C. Fonseca Guerra, O. Visser, J. G. Snijders, G. te Velde, E. J. Baerends, in Methods and
!%%%     Techniques in Computational Chemistry , (Eds.: E. Clementi, G. Corongiu), STEF, Cagliari,
!%%%     1995, p. 305-395
!%%%
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%%%
!%%% Parameters are defined in the namelist $control. For example:
!%%%   $control
!%%%       nz=4 nc=2 npt=59000 igtf=3
!%%%   $end
!%%% Then the density data are provided.
!%%%
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!%%%
!%%% Parameters:
!%%% nz      = nuclear charge number of the element. For example, nz=30 for Zn
!%%% nc      = number of core electrons. 1 < nc < nz
!%%% npt     = number of 1D-density points
!%%% igtf    = formula to generate Gaussian exponents (1~4; default: 4)
!%%% dr      = step size (optional; it also defines the format of density data)
!%%% iunit   = (0/1, i.e. in Bohr or Angstrom) unit of dr and the r(:) array (optional; default: 0)
!%%%
!%%%
!%%% Density data (dr <=0 or dr is not set):
!%%% r(1), rho(1)
!%%% r(2), rho(2)
!%%% ...
!%%%
!%%% Density data (dr > 0):
!%%% rho(1), rho(2), ...
!%%%
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
program DenFit
implicit real(kind=8) (a-h,o-z)
parameter(MaxGau = 60,au2ang=0.5291772086d0)
real(kind=8) :: alf(MaxGau)
real(kind=8),allocatable :: r(:), rho(:)
real(kind=8),allocatable :: qn(:), as(:,:), ax(:,:), al(:), an(:), coef(:), scr(:)
real(kind=8),allocatable :: alf0(:), as0(:,:), al0(:), an0(:), qn0(:)
character(8) logo
logical ifnrm

! normalized (.T.) s-functions or not(.F.)
!ifnrm =.false.
ifnrm =.true.

iinp = 5
iedf = 8
write(*,"(//,1x,41('='),/,1x,10('='),'  Results of DenFit  ',10('='),/,1x,41('='))")

! read parameters
call rdpara(iinp,IGTF,NZ,NC,Npt,drval,iunit)
if(iunit == 1) drval=drval/au2ang

logo="E000C000"
write(logo(2:4),"(i3.3)")NZ
write(logo(6:8),"(i3.3)")NC

! Gaussian s-functions alpha for fitting
call GenGau(IGTF,NGau,alf)
if (NGau > MaxGau) then
  write(*,"(//,' NGau > MaxGau!')")
  stop
else
  write(*,"(//,' Starting #S-Fun =',i8)") NGau
end if

! sort alpha in descending order
call ShellSort(NGau,alf,1)

! read density data
allocate(r(Npt), rho(Npt))
call rddata(iinp,Npt,iunit,drval,r,rho,Info)
if (Info == 0) then
  write(*,"(' ### Error when reading density data!')")
  stop
end if

! r0 ~ 1.0d-5
r0=0.d0
rho0=0.d0
drho=1.0d5
do i=1, Npt
  if(drho >= abs(r(i) - 1.0d-5) ) then
    drho=abs(r(i) - 1.0d-5)
    cycle
  else
    j = i -1
    if(r(j) < 0.5d0 * 1.0d-5) j = i
    r0 = r(j)
    rho0=rho(j)
    exit
  end if
end do
rr0 = r0 * r0

! There may be an artificial peak after r = 1.8 a.u.
i18=0
do i=1, Npt
  if(r(i) >= 1.8d0) then
    i18 = i
    exit
  end if
  if(i .eq. Npt) then
    write(*,"(' r(Npt) is too small!')")
    stop
  end if
end do

write(*,"(/,' R0  = ',d12.5,',  Rho0  = ',d20.14)") r0, rho0

allocate(qn(NGau), as(NGau,NGau), ax(NGau,NGau), al(NGau), an(NGau), coef(NGau), scr(NGau))
allocate(alf0(NGau), as0(NGau,NGau), al0(NGau), an0(NGau), qn0(NGau))

! normalization factors of s-ARDF functions
call norms(NGau,ifnrm,alf,qn)

! Calculate the matrices/arrays for fitting
call EqFit(NGau,alf,qn,Npt,r,rho,as,al,an,Acore,scr)
write(*,"(/,' Integrated Ncore: ',f16.10)") Acore
if(abs(Acore-dble(NC)) > 1.d-2) then
  write(*,"(' Accuracy is too low!')")
  stop
end if

! Optimize NGau; get rid of the redundant steepest functions (alpha > 1.0d4)
NGau0 = NGau
call acopy(NGau,alf,alf0)
call acopy(NGau*NGau,as,as0)
call acopy(NGau,al,al0)
call acopy(NGau,an,an0)
call acopy(NGau,qn,qn0)
do I = 1, NGau0
  if (alf0(1) < 1.0d4) exit

  call GauFit(NGau0,as0,ax,al0,an0,qn0,coef,NC,Acore,Info,scr)
  if (Info == 0) then
    write(*,"(' Inverse calculation fails.')")
    stop
  end if

! delta_rho at the first point
  call DltRho(NGau0,I,alf0,coef,rr0,rho0,idxneg,drho)

! delete the first function
  call GauRm(NGau0,1,alf0,as0,al0,an0,qn0)
end do

! the first idxneg-1 functions are redundant and should be deleted
if  (idxneg > 1) then
  write(*,"(' Delete the first ',i3,' redundant functions with min[dRho0] = ',d20.14)") idxneg-1, drho
  call GauRm(NGau,-(idxneg-1),alf,as,al,an,qn)
end if

! Do fitting
do while(.true.)

  call GauFit(NGau,as,ax,al,an,qn,coef,NC,Acore,Info,scr)
  if (Info == 0) then
    write(*,"(' Inverse calculation fails.')")
    exit
  end if

! check
  if(nint(Acore) /= NC) then
    write(*,"(' The fitted Ncore is wrong!')")
    exit
  end if
  if( (NC >= 10 .and. NGau < 10) .or. (NC < 10 .and. NGau < 5) ) then
    write(*,"(' The fitting fails!')")
    exit
  end if
  call ChkPos(NGau,Npt,r,alf,coef,idxneg)
  if(idxneg > 0) then
    write(*,"(' Negative density found!   Delete function-',i3,' with alpha = ',d20.14)") idxneg, alf(idxneg)
    call GauRm(NGau,idxneg,alf,as,al,an,qn)
    cycle
  end if
  call ChkRed(NGau,alf,coef,idxneg)
  if(idxneg > 0) then
    write(*,"(' Redundant function found! Delete function-',i3,' with alpha = ',d20.14)") idxneg, alf(idxneg)
    call GauRm(NGau,idxneg,alf,as,al,an,qn)
    cycle
  end if
  call ChkAPk(NGau,Npt,i18,r,alf,coef,idxneg)
  if(idxneg > 0) then
    write(*,"(' Artificial peak found!    Delete function-',i3,' with alpha = ',d20.14)") idxneg, alf(idxneg)
    call GauRm(NGau,idxneg,alf,as,al,an,qn)
    cycle
  end if
  ! to get a smaller error at dRho(1), ChkHss must be done at the last step
  call ChkHss(NGau,alf,coef,idxneg)
  if(idxneg > 0) then
    write(*,"(' Hessian(r=0) > 0 found!   Delete function-',i3,' with alpha = ',d20.14)") idxneg, alf(idxneg)
    call GauRm(NGau,idxneg,alf,as,al,an,qn)
    cycle
  end if

!================================================= Fitting finished

! delta(rho) at the first point
  call DltRho(NGau,1,alf,coef,rr0,rho0,idxneg,drho)
  write(*,"(/,' Fitting finished successfully with',/,5x,'Ncore(analytic) =',f16.10,' and dRho0  = ',d20.14)") Acore, drho

! print results
  write(*,"(//,' Final results:',//,' Element=',i8,/,' Ncore  =',i8,/,' #S-Fun =',i8, //,19x,'Alpha',13x,'Coefficient',/)") &
    NZ, NC, NGau
  do i = 1, NGau
    write(*,"(2e24.14)") alf(i), coef(i)
  end do

  open(iedf,file="EDF",iostat=Info)
  if (Info > 0) then
    write(*,"(' ### Error when creating EDF file!')")
    exit
  end if
  rewind(iedf)
!  write(iedf,"('**',a8)")logo
!  write(iedf,"(i4)") NGau
!  write(iedf,"(5e22.14)") (alf(i), i=1,NGau)
!  write(iedf,"(5e22.14)") (coef(i), i=1,NGau)

  write(iedf,"(//,2x,'EDF in FORTRAN90',/,2x,'NZA   = ',i4,/,2x,'NCore = ',i4)") NZ, NC
  write(iedf,"(/,4x,'nfun =',i3)") NGau
  Nline=(NGau+3)/4
  Nlast=mod(NGau,4)
!  alf
  write(iedf,"(4x,'alf(1:nfun)=(/',4(d22.14,','),'&')") (alf(j),j=1,4)
  do i = 2, Nline-1
    write(iedf,"(18x,4(d22.14,','),'&')")(alf(j),j=4*i-3,4*i)
  end do
  if(Nlast .eq. 0) then
    write(iedf,"(18x,3(d22.14,','),d22.14,'/)')") (alf(j),j=NGau-3,NGau)
  else if(Nlast .eq. 3) then
    write(iedf,"(18x,2(d22.14,','),d22.14,'/)')") (alf(j),j=NGau-2,NGau)
  else if(Nlast .eq. 2) then
    write(iedf,"(18x,d22.14,',',d22.14,'/)')") (alf(j),j=NGau-1,NGau)
  else
    write(iedf,"(18x,d22.14,'/)')") alf(NGau)
  end if
!  coe
  write(iedf,"(4x,'coe(1:nfun)=(/',4(d22.14,','),'&')") (coef(j),j=1,4)
  do i = 2, Nline-1
    write(iedf,"(18x,4(d22.14,','),'&')")(coef(j),j=4*i-3,4*i)
  end do
  if(Nlast .eq. 0) then
    write(iedf,"(18x,3(d22.14,','),d22.14,'/)')") (coef(j),j=NGau-3,NGau)
  else if(Nlast .eq. 3) then
    write(iedf,"(18x,2(d22.14,','),d22.14,'/)')") (coef(j),j=NGau-2,NGau)
  else if(Nlast .eq. 2) then
    write(iedf,"(18x,d22.14,',',d22.14,'/)')") (coef(j),j=NGau-1,NGau)
  else
    write(iedf,"(18x,d22.14,'/)')") coef(NGau)
  end if

  exit

end do

deallocate(r, rho)
deallocate(qn, as, ax, al, an, coef, scr)
deallocate(alf0, as0, al0, an0, qn0)

end


!%%%
!%%% read parameters from input
!%%%
subroutine rdpara(iinp,IGTF,NZ,NC,Npt,dr,iunit)
implicit real(kind=8) (a-h,o-z)
namelist/control/NZ,NC,Npt,IGTF,dr,iunit

NZ=0
NC=0
Npt=0
IGTF=4
dr=-1.d0

rewind(iinp)
read(iinp,control,err=100,end=200)

! default: 4
if (IGTF < 1 .or. IGTF > 4) IGTF = 4

if (NZ < 2 .or. NZ > 120) then
  write(*,"(' NZ is out of range!')")
  stop
end if
if (NC < 0 .or. NC >= NZ) then
  write(*,"(' NC is out of range!')")
  stop
end if
if (Npt < 50) then
  write(*,"(' Npt is too small!')")
  stop
end if
if (dr > 0.02d0) then
  write(*,"(' dR is too big!')")
  stop
end if
if(iunit /= 1) iunit = 0

Return

100  write(*,"(//,' Unknown parameters found!')")
  stop
200  write(*,"(//,' No parameters defined!')")
  stop
End


!%%%
!%%% read density data
!%%%
subroutine rddata(iinp,Npt,iunit,drval,r,rho,l)
implicit real(kind=8) (a-h,o-z)
parameter(au2ang=0.5291772086d0)
real(kind=8) :: r(Npt), rho(Npt)

l=1
if(drval > 0) then
  read(iinp,*,err=100) (rho(i), i=1,Npt)
  r(1) = 0.d0
  do i=2,Npt
    r(i) = r(i-1) + drval
  end do
else
  do i=1,Npt
    read(iinp,*,err=100) r(i), rho(i)
  end do
  if(iunit == 1) r = r/au2ang
end if
Return

100  l=0
Return
End


!%%%
!%%% delete redundant functions
!%%%
subroutine ChkRed(NGau,alf,coef,idxneg)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: alf(NGau), coef(NGau)

idxneg = 0

Nhalf=NGau/2

! for steep functions with alpha > 1.0d5
do i=1,Nhalf
  if(alf(i) > 1.0d5) then
    if(abs(coef(i)) < 5.0d0) then
      idxneg = i
      goto 1000
    end if
  else
    exit
  end if
end do

! for flat functions with alpha < 3
do i=NGau, Nhalf+1, -1
  if(alf(i) < 3.0d0) then
    if(abs(coef(i)) < 1.0d-5) then
      idxneg = i
      goto 1000
    end if
  else
    exit
  end if
end do

1000 return
end


!%%%
!%%% check artificial peak in ARDF for r>1.8
!%%%
subroutine ChkAPk(NGau,Npt,i18,r,alf,coef,idxneg)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: r(Npt), alf(NGau), coef(NGau)

idxneg = 0
! alf(idxneg) should be < 1
if(alf(NGau) > 1.d0) return

ardfold = 0.d0
do i= i18, Npt, 10
  ! ardf/(4pi)
  rr = r(i)*r(i)
  ardf = 0.d0
  do j=1,NGau
    ardf = ardf + coef(j)*rr*exp(-alf(j)*rr)
  end do
  if(i == i18 .or. ardf <= ardfold) then
    ardfold = ardf
    cycle
  else
  	idxneg = NGau
  	exit
  end if
end do

return
end


!%%%
!%%% Check Hessian at r=0, which should be negative, i.e. there is a maximum at r=0.
!%%%
!%%% Hessian(0) = -2 * hess, where hess = sum[coe(i)*alf(i)]
!%%%
subroutine ChkHss(NGau,alf,coef,idxneg)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: alf(NGau), coef(NGau)

idxneg = 0
hess = 0.0d0
do i= 1, NGau
  hess = hess + coef(i)*alf(i)
end do
if(hess <= 0.0d0) idxneg = 1

return
end


!%%%
!%%% check positive definiteness of rho
!%%%
subroutine ChkPos(NGau,Npt,r,alf,coef,idxneg)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: r(Npt), alf(NGau), coef(NGau)

rhomin = 1.d-10
idxneg = 0
do i=1,Npt
  rho=0.d0
  do j=1,NGau
    rho = rho + coef(j) * exp(-alf(j) * r(i) * r(i))
  end do
  if(i == 1 .and. rho < 0.0d0) then
    rhomin = rho
    idxneg = 1
    exit
  end if
!  if(i < 5) write(*,"(d24.12)") rho
  if(rhomin > rho) then
    rhomin = rho
    idxneg = i
  end if
end do

if(rhomin < 0.d0)then
  r0 = r(idxneg)
  rhomin = 1.d-10
  idxneg = 0
  do i=1,NGau
    rho = coef(i) * exp(-alf(i) * r0 * r0)
    if(rhomin > rho) then
      rhomin = rho
      idxneg = i
    end if
  end do
else
  idxneg = 0
end if

return
end


!%%%
!%%% Irm > 0: Remove the Irm-th row/column in an array/matrix
!%%% Irm < 0: Remove the first |Irm| rows/columns in an array/matrix
!%%%
Subroutine GauRm(NGau,Irm,alf,as,al,an,qn)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: alf(NGau), as(NGau,NGau), al(NGau), an(NGau), qn(NGau)
logical Lmod

Iab = abs(Irm)

if (Iab == 0) return

J = 0
Do I = 1, NGau
  if (Irm > 0) then
    Lmod = (I == Iab)
  else
    Lmod = (I <= Iab)
  end if
  if (Lmod) cycle
  J = J + 1
  alf(J) = alf(I)
  al(J) = al(I)
  an(J) = an(I)
  qn(J) = qn(I)
end do

call DelRC(NGau,Irm,as,as)

if (Irm > 0) then
  NGau = NGau - 1
else
  NGau = NGau - Iab
end if

Return
End


!%%%
!%%% Irm > 0: Remove the Irm-th row & column in a matrix
!%%% Irm < 0: Remove the first |Irm| rows & columns in a matrix
!%%%
Subroutine DelRC(N,Irm,a,b)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: a(N,N), b(*)
logical Imod, Jmod

Iab = abs(Irm)

if (Iab == 0) return

K = 0
Do I = 1, N
  if (Irm > 0) then
    Imod = (I == Iab)
  else
    Imod = (I <= Iab)
  end if
  if (Imod) cycle
  Do J = 1, N
    if (Irm > 0) then
      Jmod = (J == Iab)
    else
      Jmod = (J <= Iab)
    end if
    if (Jmod) cycle
    K = K + 1
    b(K) = a(J,I)
  end do
end do

Return
End


!%%%
!%%% Calculate the matrices/arrays for fitting
!%%%
subroutine EqFit(NGau,alf,qn,Npt,r,rho,as,al,an,Acore,scr)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: alf(NGau), qn(NGau), r(Npt), rho(Npt), as(NGau,NGau), al(NGau), an(NGau), scr(NGau)

! For normalized s-ARDF functions, the overlap matrix element
! as(j,i)
! = N(i) * N(j) * Int{(4*pi*r*r)^3 * [exp(-a_i*r*r)] * [exp(-a_j*r*r)]} dr
! = N(i) * N(j) * 60 * (pi/(a_i+a_j))^3.5
!
! an(i)
! = Int{[4*pi*r*r*exp(-a_i*r*r)]} dr
! = (pi/a_i)^1.5
pi=acos(-1.d0)
cons1 = power(sqrt(pi),3)
cons2 = pi * 4.d0
cons3 = cons2*cons2*cons2
do i=1,NGau
  do j=1,i
    as(j,i) = pi / (alf(i) + alf(j))
    as(j,i) = qn(i) * qn(j) * 60.d0 * as(j,i)**3.5d0
    if(j < i) as(i,j) = as(j,i)
  end do
  an(i) = sqrt(alf(i))
  an(i) = cons1 / power(an(i),3)
end do

! calculate Ncore and al numerically
Acore=0.d0
al=0.d0
do i=1,Npt-1
  dr=r(i+1)-r(i)
  rr=r(i)*r(i)
  dc=rr*rho(i)
  d6=rr*rr*dc
  do j=1,NGau
    scr(j) = d6 * exp(-alf(j) * rr)
  end do
  Acore = Acore + dc * dr
  call AccAB(NGau,dr,scr,al,al)
end do
Acore=Acore*cons2
al=al*cons3

return
end


!%%%
!%%% Do fitting
!%%%
subroutine GauFit(NGau,as,ax,al,an,qn,coef,NC,Acore,Info,scr)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: as(NGau,NGau), ax(NGau,NGau), al(NGau), an(NGau), qn(NGau), coef(NGau), scr(NGau)

! as^-1 --> ax
call acopy(NGau*NGau,as,ax)
call bssgj(NGau,Info,ax,scr)
if (Info == 0) return

! mat(qn) * ax * mat(qn) --> ax
call dmd(NGau,qn,ax)

! lambda = (Ncore-an'*ax*al)/(an'*ax*an)
call MatxL(NGau,ax,an,scr)
alam = (dble(NC) - dotx(NGau,scr,al)) / dotx(NGau,scr,an)

! coefficients: ax * (al + lambda*an)
call AccAB(NGau,alam,an,al,scr)
call MatxL(NGau,ax,scr,coef)

! recalculate Ncore analytically
Acore=0.d0
do i=1,NGau
  Acore=Acore+an(i)*coef(i)
end do

return
end


!%%%
!%%% Inverse of a symmetric positive definite matrix a
!%%% Taken from Shiliang Xu's Fortran77 book, SS2.5
!%%%
subroutine bssgj(n,l,a,b)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: a(n,n), b(n)

l=1
do k=1,n
  m=n-k+1
  w=a(1,1)
  if (w+1.d0 == 1.d0) then
    l=0
    return
  end if
  do i=2,n
    g=a(i,1)
    b(i)=g/w
    if (i <= m) b(i)=-b(i)
    do j=2,i
      a(i-1,j-1)=a(i,j)+g*b(j)
    end do
  end do
  a(n,n)=1.d0/w
  do i=2,n
    a(n,i-1)=b(i)
  end do
end do

do i=1,n-1
  do j=i+1,n
    a(i,j)=a(j,i)
  end do
end do

return
end


!%%%
!%%% delta_rho at the first point
!%%%
!%%% rr0 = r0*r0
!%%%
subroutine DltRho(N,Iter,a,c,rr0,rho0,imin,dmin)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: a(*), c(*)

x = 0.d0
do i = 1, N
  x = x + c(i) * exp(-a(i) * rr0)
end do

! delta_rho
x = abs(x - rho0)

if(Iter == 1) then
  imin = 1
  dmin = x
else
  if(dmin > x) then
    imin = Iter
    dmin = x
  end if
end if

return
end


!%%%
!%%% D(N,N) * A(N,N) * D(N,N) --> A(N,N)
!%%% where D is a diagonal matrix with elements in d(N)
!%%%
subroutine dmd(N,d,a)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: d(N), a(N,N)

do i = 1, N
  do j = 1, i
    a(j,i) = a(j,i) * d(i) * d(j)
    if(j < i) a(i,j) = a(j,i)
  end do
end do

return
end


!%%%
!%%% B(*) = A(*)
!%%%
subroutine acopy(n,a,b)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: a(*), b(*)

do i = 1, n
  b(i) = a(i)
end do

return
end


!%%%
!%%% a**N
!%%%
function power(a,N)
implicit real(kind=8) (a-h,o-z)

if (N == 0) then
  power = 1.d0
  return
end if

power = a
Do I = 2, abs(N)
  power = power * a
end do
if (N < 0) power = 1.d0 / power

Return
End


!%%%
!%%% normalization factors of s-ARDF functions: q_i = (2*a_i/pi)^(7/4) / sqrt(60)
!%%%
Subroutine norms(N,ifnrm,a,q)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: a(N), q(N)
logical ifnrm

q = 1.d0
if(ifnrm) then
  pi=acos(-1.d0)
  Do I = 1, N
    q(I) = ((a(I)+a(I))/pi)**1.75d0
  end do
end if

q=q/sqrt(60.d0)

Return
End


!%%%
!%%% A(N,N) * B(N) = C(N)
!%%%
Subroutine MatxL(N,A,B,C)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: A(N,N), B(N), C(*)

Do I = 1,N
  C(I) = dotx(N,A(1,I),B)
end do

Return
End


!%%%
!%%% vector A dot_product vector B
!%%%
function dotx(N,A,B)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: A(N), B(N)

dotx = 0.d0

Do I = 1,N
  dotx = dotx + A(I)*B(I)
end do

Return
End


!%%%
!%%% C(*) = c0 * A(*) + B(*)
!%%%
Subroutine AccAB(N,c0,A,B,C)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: A(N), B(N), C(N)

Do I = 1, N
  C(I) = B(I) + c0*A(I)
end do

Return
End


!%%%
!%%%  sort a in ascending (Mode = 0) or descending (Mode /= 0) order using D. L. Shell's method
!%%%
subroutine ShellSort(N,a,Mode)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: a(N)
logical :: AD

K = N /2

do while (K > 0)
  do I = K + 1, N
    J = I - K
    do while (J > 0)
      if (Mode == 0) then
        AD = a(J) > a(J+K)
      else
        AD = a(J) < a(J+K)
      end if
      if (AD) then
        t = a(J)
        a(J) = a(J + K)
        a(J + K) = t
        J = J - K
      else
        exit
      end if
    end do
  end do
  K = K / 2
end do

Return
End


!%%%
!%%% Gaussian s-functions
!%%%
subroutine GenGau(icase,NGau,alf)
implicit real(kind=8) (a-h,o-z)
real(kind=8) :: alf(*)

write(*,"(//,' Type of functions: ',i6)") icase

select case(icase)

  case(1)
!   Alpha_i = exp[a + b * (i - 1)], a=-3.21885281944033, b=0.672
!   Parameters used by G09
    NGau = 40
    a =-3.21885281944033d0
    b = 0.672d0
    do i = 1, NGau
      j = i - 1
      alf(i) = exp(a + b * j)
    end do
    write(*,"(&
    '  Alpha_i = exp[a + b * (i - 1)], a=-3.21885281944033, b=0.672',/,&
    ' Reference:',/,&
    '  E. V. R. de Castro and F. E. Jorge, J. Chem. Phys. 108 5225 (1998).',/,&
    '  and modified in',/,&
    '  T. A. Keith and M. J. Frisch, J. Phys. Chem. A 115, 12879 (2011).')")

  case(2)
!   Alpha_i = exp[a + b * (i - 1)], a=-4.584, b=0.672
!   Ref.
!   E. V. R. de Castro and F. E. Jorge, J. Chem. Phys. 108 5225 (1998).
    NGau = 40
    a =-4.584d0
    b = 0.672d0
    do i = 1, NGau
      j = i - 1
      alf(i) = exp(a + b * j)
    end do
    write(*,"(&
    '  Alpha_i = exp[a + b * (i - 1)], a=-4.584, b=0.672',/,&
    ' Reference:',/,&
    '  E. V. R. de Castro and F. E. Jorge, J. Chem. Phys. 108 5225 (1998).')")

  case(3)
!   Alpha_i = exp[a + b * (i - 1)], a=-3.84, b=0.72
!   Ref.
!   G. L. Malli, A. B. F. DaSilva, and Y. Ishikawa, Phys. Rev. A 47, 143 (1993).
!   A. Wolf, M. Reiher, and B. A. Hess, J. Chem. Phys. 117, 9215 (2002).
    NGau = 38
    a =-3.84d0
    b = 0.72d0
    do i = 1, NGau
      j = i - 1
      alf(i) = exp(a + b * j)
    end do
    write(*,"(&
    '  Alpha_i = exp[a + b * (i - 1)], a=-3.84, b=0.72',/,&
    ' Reference:',/,&
    '  G. L. Malli, A. B. F. DaSilva, and Y. Ishikawa, Phys. Rev. A 47, 143 (1993).',/,&
    '  A. Wolf, M. Reiher, and B. A. Hess, J. Chem. Phys. 117, 9215 (2002).')")

!  case(4)
  case default
!   Even-tempered universal Gaussian exponents
!   Alpha_i = 0.001 * 1.65^(i-1)
!           ~ exp[a + b * (i - 1)], a=-6.907755, b=0.500775
!   Ref.
!   M. Reiher and A. Wolf, J. Chem. Phys. 121, 10945 (2004).
    NGau = 60
    a = 1.0d-3
    b = 1.65d0
    alf(1) = a;
    do i=2,NGau
      alf(i) = alf(i-1) * b
    end do
    write(*,"(&
    '  Even-tempered universal Gaussian exponents',/,&
    '  Alpha_i = 0.001 * 1.65^(i-1)',/,&
    '          ~ exp[a + b * (i - 1)], a=-6.907755, b=0.500775',/,&
    ' Reference:',/,&
    '  M. Reiher and A. Wolf, J. Chem. Phys. 121, 10945 (2004).')")

end select

return
end

