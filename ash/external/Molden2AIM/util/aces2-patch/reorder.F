      Subroutine reorder(Vecin, Vecout, Iang, nao)

      Implicit Double Precision (A-H, O-Z)
      Double Precision Vecin(nao), Vecout(nao)
      Integer Iang(nao)
c
      Call Dcopy(nao, Vecin, 1, Vecout, 1)

      index = 1
      Do while (index .lt. nao)
         If (iang(index) .Le. 1) then
C
c handle s and p functions (ie, do nothing)
C
             index = index + 1

         Else if (iang(index) .eq. 2) Then
c
c handle d functions
C
             vecout(index)   = Vecin(index)
             vecout(index+1) = Vecin(index+3)
             vecout(index+2) = Vecin(index+5)
             vecout(index+3) = Vecin(index+1)
c<<<
             vecout(index+4) = Vecin(index+2)
             vecout(index+5) = Vecin(index+4)
c>>>
             index = index + 6

         Else if (iang(index) .EQ. 3) Then
c
c handle f functions

             vecout(index)   = Vecin(index)
             vecout(index+1) = Vecin(index+6)
             vecout(index+2) = Vecin(index+9)
             vecout(index+3) = Vecin(index+3)
             vecout(index+4) = Vecin(index+1)
             vecout(index+5) = Vecin(index+2)
             vecout(index+6) = Vecin(index+5)
             vecout(index+7) = Vecin(index+8)
             vecout(index+8) = Vecin(index+7)
             vecout(index+9) = Vecin(index+4)
             Index = Index + 10
c<<<
         Else if (iang(index) .EQ. 4) Then
c
c handle g functions

             vecout(index)   = Vecin(index)
             vecout(index+1) = Vecin(index+10)
             vecout(index+2) = Vecin(index+14)
             vecout(index+3) = Vecin(index+1)
             vecout(index+4) = Vecin(index+2)
             vecout(index+5) = Vecin(index+6)
             vecout(index+6) = Vecin(index+11)
             vecout(index+7) = Vecin(index+9)
             vecout(index+8) = Vecin(index+13)
             vecout(index+9) = Vecin(index+3)
             vecout(index+10)= Vecin(index+5)
             vecout(index+11)= Vecin(index+12)
             vecout(index+12)= Vecin(index+4)
             vecout(index+13)= Vecin(index+7)
             vecout(index+14)= Vecin(index+8)
             Index = Index + 15

         Else if (iang(index) .gt. 4) Then
             Write(6, "(a,a,a)") " MOLDEN display of orbitals for",
     &                           " angluar momentum higher",
     &                           " than g functions are incorrect."
c>>>
             Return
         Endif
C
      Enddo

      Return
      End
