Part of Fragment class
def calc_connectivity(self, conndepth=99, scale=None, tol=None):
    self.atomlist = list(range(0, self.numatoms))
    self.connectivity = []
    # Going through each atom and getting recursively connected atoms
    testlist = self.atomlist
    # Removing atoms from atomlist until empty
    while len(testlist) > 0:
        for index in testlist:
            wholemol = get_molecule_members_loop_np2(self.coords, self.elems, conndepth, scale, tol, atomindex=index)
            if wholemol in self.connectivity:
                continue
            else:
                self.connectivity.append(wholemol)
                for i in wholemol:
                    testlist.remove(i)
    # Calculate number of atoms in connectivity list of lists
    conn_number_sum = 0
    for l in self.connectivity:
        conn_number_sum += len(l)
    if self.numatoms != conn_number_sum:
        print("Connectivity problem")
        exit()
    self.connected_atoms_number = conn_number_sum
    print("self.connected_atoms_number:", self.connected_atoms_number)
    print("self.connectivity:", self.connectiv