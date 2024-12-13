from cs_entropy import index_to_paulistring, paulistring_to_index
from stim import PauliString
import stim
import numpy as np
from lattice_definitions import Lattice2D, compass_to_surface

class StabilizerCode:
    """
    Represents a stabilizer code

    logicals_generators: the logical operators of the code
    stabilizer_generators: the stabilizer generators of the code
    gauge_generators: the gauge generators of the code


    all operators are represented as stim.PauliString objects
    """
    def __init__(self, logicals_generators, stabilizer_generators, gauge_generators=None):
        self.num_qubits = len(stabilizer_generators[0])
        self.logical_Xs = logicals_generators[0]
        self.logical_Zs = logicals_generators[1]
        self.stabilizer_generators = stabilizer_generators
        if gauge_generators is None:
            gauge_generators = [[], []]
        self.gauge_Xs = gauge_generators[0]
        self.gauge_Zs = gauge_generators[1]

        self.tableau = self._make_stim_tableau()
        self.logical_bits = np.array(range(len(self.logical_Zs)))
        self.stabilizer_bits = np.array(range(len(self.stabilizer_generators))) + len(self.logical_Zs)
        self.gauge_bits = np.array(range(len(self.gauge_Zs))) + len(self.logical_Zs) + len(self.stabilizer_generators)
        self.destabilizers = self._make_destabilizers()

        # run the checks and raise warnings if the checks fail
        self._check_group_dimensions()
        self._check_operator_support()

    def __str__(self):
        return   (f'Code [num_qubits:{self.num_qubits}, num stabs:{len(self.stabilizer_generators)},' +
                 f'num logicals:{len(self.logical_Xs)},{len(self.logical_Zs)},' + 
                 f'num_gauge:{len(self.gauge_Xs)},{len(self.gauge_Zs)}]')
    
    @property
    def logical_autocommutation_matrix(self):
        """
        Returns the commutation matrix of the X and Z logical operators
        """
        commutation_matrix = np.zeros((len(self.logical_Xs), len(self.logical_Zs)))
        for i in range(len(self.logical_Xs)):
            for j in range(len(self.logical_Zs)):
                if self.logical_Xs[i].commutes(self.logical_Zs[j]):
                    commutation_matrix[i, j] = 0
                else:
                    commutation_matrix[i, j] = 1
        return commutation_matrix
    
    def _make_stim_tableau(self):
        # create a list of all the Z logicals, stabilizers, and Z gauges
        zops = self.logical_Zs + self.stabilizer_generators + self.gauge_Zs
        # make the sitme tableau 
        stim_tableau = stim.Tableau.from_stabilizers(zops, allow_underconstrained=True)
        return stim_tableau
    
    def _make_destabilizers(self):
        # calculate the destabilizers
        destabilizers = []
        for sbit in self.stabilizer_bits:
            x = ['_']*self.num_qubits
            x[sbit] = 'X'
            x_pauli = stim.PauliString(''.join(x))
            destab = self.tableau(x_pauli)
            destabilizers.append(destab)
        return destabilizers
    
    def syndrome_tableau(self, error):
        """
        
        """
        return self.tableau(error).to_numpy()[0][self.stabilizer_bits]
    
    def syndrome(self, error):
        """
        Returns the syndrome of the error
        """
        syndrome = np.zeros(len(self.stabilizer_generators))
        for stabilizer in self.stabilizer_generators:
            if not stabilizer.commutes(error):
                syndrome[self.stabilizer_generators.index(stabilizer)] = 1
        return syndrome

    
    def logical_effect(self, error):
        """
        Returns the effect of the error on the logical operators after recovery
        """
        syndrome = self.syndrome(error)
        syn_recovery = self.syndrome_recovery(syndrome)
        # ensure the recovery returns to the ground state
        assert np.all(self.syndrome(syn_recovery*error) == np.zeros(len(self.stabilizer_generators))), (self.syndrome(syn_recovery), syndrome)
        logical_x_effects = np.zeros((len(self.logical_Xs)))
        logical_z_bits = np.zeros(len(self.logical_Zs))
        for idx, lx in enumerate(self.logical_Xs):
            if not lx.commutes(error):
                logical_x_effects[idx] = 1
        for idx, lz in enumerate(self.logical_Zs):
            if not lz.commutes(error):
                logical_z_bits[idx] = 1
        return np.hstack([logical_x_effects, logical_z_bits])
    
    def gauge_effect(self, error):
        """
        Returns the effect of the error on the gauge operators
        """
        return np.vstack([self.tableau(error).to_numpy()[0][self.gauge_bits] , self.tableau(error).to_numpy()[1][self.gauge_bits]])

    def syndrome_recovery(self, syndrome):
        """
        Returns a recovery operator for the given syndrome
        """
        # return the product of the destabilizers in the syndrome
        recovery = PauliString('_'*self.num_qubits)
        for i in range(len(syndrome)):
            if syndrome[i] == 1:
                recovery *= self.destabilizers[i]
        #assert np.all(self.syndrome(recovery) == syndrome), (self.syndrome(recovery), syndrome)
        return recovery


    def make_joint_distribution(self, pauli_distribution):
        """
        calculate P(Y, Z) under the given Pauli distribution
        """
        pvec = pauli_distribution.pvec_joint()
        joint_distribution = {}
        for i in range(len(pvec)):
            pstr = index_to_paulistring(i, self.num_qubits)
            error = stim.PauliString(pstr)
            syndrome = self.syndrome(error)
            logical_effect = self.logical_effect(error)
            pair = (tuple(syndrome.flatten()), tuple(logical_effect.flatten()))
            if pair in joint_distribution:
                joint_distribution[pair] += pvec[i]
            else:
                joint_distribution[pair] = pvec[i]
        return joint_distribution

    def _check_group_dimensions(self):
        if len(self.logical_Xs) != len(self.logical_Zs):
            raise Warning('The number of logical Xs and Zs are not equal')
        if len(self.gauge_Xs) != len(self.gauge_Zs):
            raise Warning('The number of gauge Xs and Zs are not equal')

    def _check_operator_support(self):
        """
        Check that the operators are supported on the correct number of qubits
        """
        for op in self.logical_Xs + self.logical_Zs + self.stabilizer_generators + self.gauge_Xs + self.gauge_Zs:
            if len(op) != self.num_qubits:
                raise Warning(f'The operator {op} is not supported on the correct number of qubits')



class FiveQubitCode(StabilizerCode):
    def __init__(self):
        stabilizer_generators = [
            PauliString('XZZXI'), 
            PauliString('IXZZX'),
            PauliString('XIXZZ'),
            PauliString('ZXIXZ'),
        ]
        Lx = PauliString('XXXXX')
        Lz = PauliString('ZZZZZ')
        super().__init__([[Lx], [Lz]], stabilizer_generators)


class SteaneCode(StabilizerCode):
    def __init__(self):
        stabilizer_generators = [
            PauliString('IIIXXXX'), 
            PauliString('IXXIIXX'), 
            PauliString('XIXIXIX'),
            PauliString('IIIZZZZ'),
            PauliString('IZZIIZZ'),
            PauliString('ZIZIZIZ'),
        ]

        Lx = PauliString('XXXXXXX')
        Lz = PauliString('ZZZZZZZ')
        super().__init__([[Lx], [Lz]], stabilizer_generators)

class SurfaceCode(StabilizerCode):
    def __init__(self, dimX, dimZ):
        """
        Make a surface stabilizer code from compass code lattice logic
        """
        lat = compass_to_surface(dimX, dimZ)
        Lx = PauliString(lat.Lx)
        Lz = PauliString(lat.Lz)
        stabilizer_generators = [PauliString(stab) for stab in lat.getS()]
        super().__init__([[Lx], [Lz]], stabilizer_generators)

class ShorRedCode(StabilizerCode):
    def __init__(self, dimX, dimZ):
        """
        Make a red (X bias) shor code from compass code lattice logic
        """
        lat = Lattice2D(dimX, dimZ)
        lat.color_lattice([-1]*(dimX-1)*(dimZ-1))
        Lx = PauliString(lat.Lx)
        Lz = PauliString(lat.Lz)
        stabilizer_generators = [PauliString(stab) for stab in lat.getS()]
        super().__init__([[Lx], [Lz]], stabilizer_generators)


class ShorBlueCode(StabilizerCode):
    def __init__(self, dimX, dimZ):
        """
        Make a blue (Z bias) shor code from compass code lattice logic
        """
        lat = Lattice2D(dimX, dimZ)
        lat.color_lattice([1]*(dimX-1)*(dimZ-1))
        Lx = PauliString(lat.Lx)
        Lz = PauliString(lat.Lz)
        stabilizer_generators = [PauliString(stab) for stab in lat.getS()]
        super().__init__([[Lx], [Lz]], stabilizer_generators)