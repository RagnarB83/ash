from pyscf.lib import logger
from pyscf import lib
from pyscf import scf,gto
from pyscf import df
import pyscf
import numpy as np


# Modifications to pyscf code for GPU support
# NOTE: Temporary qmmm_for_scf function that is compatible with gpu4pyscf
def qmmm_for_scf(method, mm_mol, platform="CPU"):
    print("Inside qmmm_for_scf")
    print("method:", method)
    print("QM/MM. Case: normal MF object")
    # Avoid to initialize QMMM twice
    if isinstance(method, pyscf.qmmm.QMMM):
        method.mm_mol = mm_mol
        return method
    cls = QMMMSCF
    return pyscf.lib.set_class(cls(method, mm_mol,platform=platform), (cls, method.__class__))

#Option only if we first convert the method to GPU object
def qmmm_for_scf2(method, mm_mol, platform="CPU"):
    print("Inside qmmm_for_scf")
    print("method:", method)
    #assert (isinstance(method, (pyscf.scf.hf.SCF, pyscf.mcscf.casci.CASBase)))
    if isinstance(method, pyscf.scf.hf.SCF):
        print("QM/MM. Case: normal MF object")
        # Avoid to initialize QMMM twice
        if isinstance(method, pyscf.qmmm.QMMM):

            method.mm_mol = mm_mol
            return method

        cls = QMMMSCF
        print("cls:", cls)
        print(cls.__dict__)
    else:
        print("method classname:", method.__class__)
        print("platform:", platform)
        if platform == 'GPU':
            import gpu4pyscf
            if isinstance(method, (gpu4pyscf.scf.hf.RHF, gpu4pyscf.scf.uhf.UHF)):
                print("Method is GPU object")
                # Avoid to initialize QMMM twice
                if isinstance(method, pyscf.qmmm.QMMM):
                    print("a")
                    print("method:", method)
                    print("method.mm_mol:", method.mm_mol)
                    method.mm_mol = mm_mol
                    return method
                cls = QMMMSCF
                print("cls:", cls)
                print(cls.__dict__)
        else:
            print("Some post-HF method")
            # post-HF methods
            if isinstance(method._scf, pyscf.qmmm.QMMM):
                method._scf.mm_mol = mm_mol
                return method

            cls = pyscf.qmmm.QMMMPostSCF

    return pyscf.lib.set_class(cls(method, mm_mol), (cls, method.__class__))


class QMMM:
    __name_mixin__ = 'QMMM'

_QMMM = QMMM

class QMMMSCF(QMMM):
    _keys = {'mm_mol'}

    def __init__(self, method, mm_mol=None, platform="CPU"):
        self.__dict__.update(method.__dict__)
        if mm_mol is None:
            mm_mol = gto.Mole()
        self.mm_mol = mm_mol
        self.platform=platform

    def undo_qmmm(self):
        print("Inside undo QM/MM")
        obj = lib.view(self, lib.drop_class(self.__class__, QMMM))
        del obj.mm_mol
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, '** Add background charges for %s **',
                    self.__class__.__name__)
        if self.verbose >= logger.DEBUG:
            logger.debug(self, 'Charge      Location')
            coords = self.mm_mol.atom_coords()
            charges = self.mm_mol.atom_charges()
            for i, z in enumerate(charges):
                logger.debug(self, '%.9g    %s', z, coords[i])
        return self

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        mm_mol = self.mm_mol

        h1e = super().get_hcore(mol)
        print("h1e type:", type(h1e))
        #RB mode
        if self.platform == "CPU":
            print("cpu type")
            einsumfunc=np.einsum
            coords = mm_mol.atom_coords()
            charges = mm_mol.atom_charges()
            def dummy(f): return f
            array_mod=dummy
        elif self.platform == "GPU":
            print("gpu type")
            import cupy
            einsumfunc=cupy.einsum
            #h1e=cupy.asnumpy(h1e)
            coords = mm_mol.atom_coords()
            charges = cupy.asarray(mm_mol.atom_charges())
            array_mod=cupy.asarray

        print("h1e type:", type(h1e))
        print("coords type:", type(coords))
        print("charges type", type(charges))

        nao = mol.nao
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(min(max_memory*1e6/8/nao**2, 200))
        blksize = max(blksize, 1)
        if mm_mol.charge_model == 'gaussian':
            expnts = mm_mol.get_zetas()

            if mol.cart:
                intor = 'int3c2e_cart'
            else:
                intor = 'int3c2e_sph'
            cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                                 mol._env, intor)
            v = 0
            for i0, i1 in lib.prange(0, charges.size, blksize):
                fakemol = gto.fakemol_for_charges(coords[i0:i1], expnts[i0:i1])
                j3c = df.incore.aux_e2(mol, fakemol, intor=intor,
                                       aosym='s2ij', cintopt=cintopt)
                v += einsumfunc('xk,k->x', j3c, -charges[i0:i1])
            v = lib.unpack_tril(v)
            h1e += v
        else:
            for i0, i1 in lib.prange(0, charges.size, blksize):
                j3c = mol.intor('int1e_grids', hermi=1, grids=coords[i0:i1])
                h1e += einsumfunc('kpq,k->pq', j3c, -charges[i0:i1])
        h1e = array_mod(h1e)
        return h1e

    def energy_nuc(self):
        # interactions between QM nuclei and MM particles
        nuc = self.mol.energy_nuc()
        coords = self.mm_mol.atom_coords()
        charges = self.mm_mol.atom_charges()
        for j in range(self.mol.natm):
            q2, r2 = self.mol.atom_charge(j), self.mol.atom_coord(j)
            r = lib.norm(r2-coords, axis=1)
            nuc += q2*(charges/r).sum()
        return nuc

    def to_gpu(self):
        print("Inside to_gpu")
        obj = self.undo_qmmm().to_gpu()
        print("obj1:", obj)
        obj = qmmm_for_scf(obj, self.mm_mol)
        print("obj2:", obj)
        return lib.to_gpu(obj)

    def nuc_grad_method(self):
        scf_grad = super().nuc_grad_method()
        return qmmm_grad_for_scf(scf_grad)

    Gradients = nuc_grad_method


def qmmm_grad_for_scf(scf_grad):
    '''Add the potential of MM particles to SCF (HF and DFT) object and then
    generate the corresponding QM/MM gradients method for the QM system.
    '''
    if getattr(scf_grad.base, 'with_x2c', None):
        raise NotImplementedError('X2C with QM/MM charges')

    # Avoid to initialize QMMMGrad twice
    if isinstance(scf_grad, QMMMGrad):
        return scf_grad

    assert (isinstance(scf_grad.base, scf.hf.SCF) and
           isinstance(scf_grad.base, QMMM))

    return scf_grad.view(lib.make_class((QMMMGrad, scf_grad.__class__)))

class QMMMGrad:
    __name_mixin__ = 'QMMM'

    def __init__(self, scf_grad):
        self.__dict__.update(scf_grad.__dict__)

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, '** Add background charges for %s **',
                    self.__class__.__name__)
        if self.verbose >= logger.DEBUG1:
            logger.debug1(self, 'Charge      Location')
            coords = self.base.mm_mol.atom_coords()
            charges = self.base.mm_mol.atom_charges()
            for i, z in enumerate(charges):
                logger.debug1(self, '%.9g    %s', z, coords[i])
        return self

    def get_hcore(self, mol=None):
        ''' (QM 1e grad) + <-d/dX i|q_mm/r_mm|j>'''

        #RB
        einsumfunc=np.einsum

        if mol is None:
            mol = self.mol
        mm_mol = self.base.mm_mol
        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()

        nao = mol.nao
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(min(max_memory*1e6/8/nao**2/3, 200))
        blksize = max(blksize, 1)
        g_qm = super().get_hcore(mol)
        if mm_mol.charge_model == 'gaussian':
            expnts = mm_mol.get_zetas()
            if mol.cart:
                intor = 'int3c2e_ip1_cart'
            else:
                intor = 'int3c2e_ip1_sph'
            cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                                 mol._env, intor)
            v = 0
            for i0, i1 in lib.prange(0, charges.size, blksize):
                fakemol = gto.fakemol_for_charges(coords[i0:i1], expnts[i0:i1])
                j3c = df.incore.aux_e2(mol, fakemol, intor, aosym='s1',
                                       comp=3, cintopt=cintopt)
                v += einsumfunc('ipqk,k->ipq', j3c, charges[i0:i1])
            g_qm += v
        else:
            for i0, i1 in lib.prange(0, charges.size, blksize):
                j3c = mol.intor('int1e_grids_ip', grids=coords[i0:i1])
                g_qm += einsumfunc('ikpq,k->ipq', j3c, charges[i0:i1])
        return g_qm

    def grad_hcore_mm(self, dm, mol=None):
        r'''Nuclear gradients of the electronic energy
        with respect to MM atoms:

        ... math::
            g = \sum_{ij} \frac{\partial hcore_{ij}}{\partial R_{I}} P_{ji},

        where I represents MM atoms.

        Args:
            dm : array
                The QM density matrix.
        '''
        #RB
        einsumfunc=np.einsum

        if mol is None:
            mol = self.mol
        mm_mol = self.base.mm_mol

        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        expnts = mm_mol.get_zetas()

        intor = 'int3c2e_ip2'
        nao = mol.nao
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(min(max_memory*1e6/8/nao**2/3, 200))
        blksize = max(blksize, 1)
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                             mol._env, intor)

        g = np.empty_like(coords)
        for i0, i1 in lib.prange(0, charges.size, blksize):
            fakemol = gto.fakemol_for_charges(coords[i0:i1], expnts[i0:i1])
            j3c = df.incore.aux_e2(mol, fakemol, intor, aosym='s1',
                                   comp=3, cintopt=cintopt)
            g[i0:i1] = einsumfunc('ipqk,qp->ik', j3c * charges[i0:i1], dm).T
        return g

    contract_hcore_mm = grad_hcore_mm # for backward compatibility

    def grad_nuc(self, mol=None, atmlst=None):
        #RB
        einsumfunc=np.einsum

        if mol is None: mol = self.mol
        coords = self.base.mm_mol.atom_coords()
        charges = self.base.mm_mol.atom_charges()

        g_qm = super().grad_nuc(atmlst)
# nuclei lattice interaction
        g_mm = np.empty((mol.natm,3))
        for i in range(mol.natm):
            q1 = mol.atom_charge(i)
            r1 = mol.atom_coord(i)
            r = lib.norm(r1-coords, axis=1)
            g_mm[i] = -q1 * einsumfunc('i,ix,i->x', charges, r1-coords, 1/r**3)
        if atmlst is not None:
            g_mm = g_mm[atmlst]
        return g_qm + g_mm

    def grad_nuc_mm(self, mol=None):
        '''Nuclear gradients of the QM-MM nuclear energy
        (in the form of point charge Coulomb interactions)
        with respect to MM atoms.
        '''
        #RB
        einsumfunc=np.einsum
        if mol is None:
            mol = self.mol
        mm_mol = self.base.mm_mol
        coords = mm_mol.atom_coords()
        charges = mm_mol.atom_charges()
        g_mm = np.zeros_like(coords)
        for i in range(mol.natm):
            q1 = mol.atom_charge(i)
            r1 = mol.atom_coord(i)
            r = lib.norm(r1-coords, axis=1)
            g_mm += q1 * einsumfunc('i,ix,i->ix', charges, r1-coords, 1/r**3)
        return g_mm

    def to_gpu(self):
        obj = self.undo_qmmm().to_gpu()
        obj = qmmm_grad_for_scf(obj)
        return lib.to_gpu(self, obj)

_QMMMGrad = QMMMGrad
