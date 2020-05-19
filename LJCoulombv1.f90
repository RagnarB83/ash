!LJ+Coulomb routine in F90. Based on...
subroutine LJCoulEgrad(Pos, rc, epsij, sigmaij, charges, PEnergy, LJenergy, coulenergy, Grad, Dim, NAtom)
implicit none
integer, intent(in) :: Dim, NAtom
real(8), intent(in), dimension(0:NAtom-1, 0:Dim-1) :: Pos
real(8), intent(in), dimension(0:NAtom-1, 0:NAtom-1) :: epsij,sigmaij
real(8), dimension(0:NAtom-1) :: charges
real(8), intent(in) :: rc
real(8), intent(out) :: PEnergy, LJenergy, coulenergy
real(8), intent(inout), dimension(0:NAtom-1, 0:Dim-1) :: Grad
!f2py intent(in,out) :: Grad
real(8), dimension(Dim) :: rij, Gij
real(8) :: d,r,ri,ri3,kc,kLJ, bohr2ang,har2kcal
integer :: i, j

bohr2ang=0.52917721067
har2kcal=627.50946900
PEnergy = 0.
coulenergy = 0.
LJenergy = 0.
Grad = 0.
do i = 0, NAtom - 1
    do j = i + 1, NAtom - 1
        rij = Pos(j,:) - Pos(i,:)
        r = sum(rij * rij)
        d = sqrt(r)
        ri=1/r
        ri3=ri*ri*ri
        if (d > rc) then
          cycle
        end if
        coulenergy=coulenergy+charges(i)*charges(j)/(d*1/bohr2ang)
        kC=charges(i)*charges(j)*sqrt(ri3)*bohr2ang
        LJenergy=LJenergy+(1/har2kcal)*epsij(i,j)*4. * ((d/sigmaij(i,j))**(-12) - (d/sigmaij(i,j))**(-6))
        kLJ=-1*(1/har2kcal) * (24*epsij(i,j)*((sigmaij(i,j)/d)**6-2*(sigmaij(i,j)/d)**12))*(1/(d**2))
        !slightly odd unitconversion done here in end
        Gij=(kLJ+kC)*rij*bohr2ang
        Grad(i,:) = Grad(i,:) + Gij
        Grad(j,:) = Grad(j,:) - Gij
     enddo
enddo
PEnergy = coulenergy + LJenergy
end subroutine