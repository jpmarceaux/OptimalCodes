import numpy as np
import matplotlib.pyplot as plt
import stim
import qutip
from stim import PauliString


def random_1qb_paulistring(pvec=None):
    if pvec is None:
        return np.random.choice(['_', 'X', 'Y', 'Z'])
    else:
        # pick with probability pvec   
        return np.random.choice(['_', 'X', 'Y', 'Z'], p=pvec)

def index_to_paulistring(index, num_qubits):
    """
    Converts the index to a Pauli string
    first converts to base-4 then to Pauli string
    """
    base4 = np.base_repr(index, 4)
    base4 = base4.zfill(num_qubits)
    paulistring = ''
    for i in range(num_qubits):
        if base4[i] == '0':
            paulistring += '_'
        elif base4[i] == '1':
            paulistring += 'X'
        elif base4[i] == '2':
            paulistring += 'Y'
        elif base4[i] == '3':
            paulistring += 'Z'
    return paulistring

def paulistring_to_index(paulistring):
    """
    Converts the Pauli string to an index
    first converts to base-4 then to index
    """
    index = 0
    for i in range(len(paulistring)):
        if paulistring[i] == '_':
            index += 0 * 4**(len(paulistring)-i-1)
        elif paulistring[i] == 'X':
            index += 1 * 4**(len(paulistring)-i-1)
        elif paulistring[i] == 'Y':
            index += 2 * 4**(len(paulistring)-i-1)
        elif paulistring[i] == 'Z':
            index += 3 * 4**(len(paulistring)-i-1)
    return index

def random_nqb_paulistring(pvec=None, num_qubits=1):
    if pvec is None:
        ridx = np.random.randint(4**num_qubits)
        return index_to_paulistring(ridx, num_qubits)
    else:
        # pick with probability pvec
        p0 = 1 - np.sum(pvec)
        ridx = np.random.choice(4**num_qubits, p=np.concatenate(([p0], pvec)))
        return index_to_paulistring(ridx, num_qubits)


def pvec_entropy(pvec):
    p0 = 1 - np.sum(pvec)
    p = np.concatenate(([p0], pvec))
    return -np.sum([p[i]*np.log2(p[i]) for i in range(len(p)) if p[i] > 0])


def estimate_channel_entropy(pvec, num_samples):
    """
    Estimates the entropy of the channel using num_samples samples.
    This function samples a random Pauli 

    pvec: the probability vector of the channel
    num_samples: the number of samples

    returns: the estimated entropy of the channel

    """
    # the number of qubits is log4(len(pvec)+1)
    num_qubits = int(np.log2(len(pvec)+1)/2)

    # sample num_samples random Pauli strings
    samples = [random_nqb_paulistring(pvec, num_qubits) for _ in range(num_samples)]

    # compute the empirical distribution
    empirical_pvec = np.zeros(len(pvec)+1)
    for sample in samples:
        idx = paulistring_to_index(sample)
        empirical_pvec[idx] += 1
    empirical_pvec /= num_samples
    return pvec_entropy(empirical_pvec)

def tensor_pauli_pvecs(p1, p2):
    p1_0 = 1 - np.sum(p1)
    p2_0 = 1 - np.sum(p2)
    p1_ext = np.concatenate(([p1_0], p1))
    p2_ext = np.concatenate(([p2_0], p2))
    p1p2 = np.kron(p1_ext, p2_ext)
    return p1p2[1:] # remove the first element


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

        # run the checks and raise warnings if the checks fail
        self._check_group_dimensions()
        self._check_operator_support()
        self._check_logical_commute_stabilizers(self.logical_Xs, self.logical_Zs)
        self._check_logical_commute_gauge(self.logical_Xs, self.logical_Zs)
        self._check_logical_commutation(self.logical_Xs, self.logical_Zs)

        

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
    
    def stabilizer_tableau(self):
        """
        Returns the stabilizer tableau of the code
        """
        return stim.Tableau.from_stabilizers(self.stabilizer_generators, allow_underconstrained=True)
    
    def parity_check_tensor(self):
        """
        Returns the parity check matrix of the code

        the parity check matrix is a tensor of shape (k, 2, n) where k is the number of stabilizers
        and n is the number of qubits
        the 1st index of the tensor is the X part and the 2nd index is the Z part

        """
        pten = np.zeros((len(self.stabilizer_generators), 2, self.num_qubits))
        for i in range(len(self.stabilizer_generators)):
            bits = self.stabilizer_generators[i].to_numpy()
            pten[i, 0, :] = bits[0]
            pten[i, 1, :] = bits[1]
        return pten
    
    def syndrome(self, error):
        """
        
        """
        pten = self.parity_check_tensor()
        error_bits = error.to_numpy()
        syndrome = np.zeros(len(self.stabilizer_generators))
        for i in range(len(self.stabilizer_generators)):
            syndrome[i] = np.sum(pten[i, 0, :] @ error_bits[1]) % 2
            syndrome[i] += np.sum(pten[i, 1, :] @ error_bits[0]) % 2
        return syndrome
    
    def logical_effect(self, error):
        """
        Returns the effect of the error on the logical operators
        """
        xpart = []
        zpart = []
        for idx, l in enumerate(self.logical_Xs):
            if l.commutes(error):
                xpart.append(0)
            else:
                xpart.append(1)
        for idx, l in enumerate(self.logical_Zs):
            if l.commutes(error):
                zpart.append(0)
            else:
                zpart.append(1)

        return xpart, zpart



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

    def _check_logical_commute_stabilizers(self, logical_Xs, logical_Zs ):
        """
        Check if the logical operators commute with the stabilizers
        """
        for stabilizer in self.stabilizer_generators:
            for logical in logical_Xs + logical_Zs:
                if not stabilizer.commutes(logical):
                    raise Warning(f'The logical operator {logical} does not commute with the stabilizer {stabilizer}')
    
    def _check_logical_commute_gauge(self, logical_Xs, logical_Zs ):
        """
        Check if the logical operators commute with the gauge operators
        """
        for gauge in self.gauge_Xs + self.gauge_Zs:
            for logical in logical_Xs + logical_Zs:
                if not gauge.commutes(logical):
                    raise Warning(f'The logical operator {logical} does not commute with the gauge {gauge}')
                
    def _check_logical_commutation(self, logical_Xs, logical_Zs):
        """
        Check that all Xs commute with Xs, all Zs commute with Zs, and the Xs and Zs anticommute in pairs
        """
        
        # check that Xs commute with Xs
        for i in range(len(logical_Xs)):
            for j in range(i+1, len(logical_Xs)):
                if not logical_Xs[i].commutes(logical_Xs[j]):
                    raise Warning(f'Xs do not commute: {logical_Xs[i]} {logical_Xs[j]}')
                
        # check that Zs commute with Zs
        for i in range(len(logical_Zs)):
            for j in range(i+1, len(logical_Zs)):
                if not logical_Zs[i].commutes(logical_Zs[j]):
                    raise Warning(f'Zs do not commute: {logical_Zs[i]} {logical_Zs[j]}')

        # check that Xs and Zs anticommute in pairs
        lcom_mat = self.logical_autocommutation_matrix
        # check that each row has exactly one 1
        if np.sum(lcom_mat, axis=1).all() != 1:
            raise Warning('Xs and Zs do not anticommute in pairs')



def make_codespace_error_distribution(code, pvec):
    """
    Calculates the distribution of errors on the codespace 

    code: the stabilizer code
    pvec: the probability vector of the channel

    returns: the entropy of the code space
    """
    error_effect_dist = {}
    p0 = 1 - np.sum(pvec)
    pvec_ext = np.concatenate(([p0], pvec))
    for idx, rate in enumerate(pvec_ext):
        error = index_to_paulistring(idx, code.num_qubits)
        effect = code.logical_effect(PauliString(error))
        effect_idx = tuple(effect[0] + effect[1])
        if effect_idx in error_effect_dist:
            error_effect_dist[effect_idx] += rate
        else:
            error_effect_dist[effect_idx] = rate
    return error_effect_dist
        

def calc_codespace_entropy(code, pvec):
    """
    Calculates the entropy of the code space

    code: the stabilizer code
    pvec: the probability vector of the channel

    returns: the entropy of the code space
    """
    error_effect_dist = make_codespace_error_distribution(code, pvec)
    entropy = 0
    for key, prob in error_effect_dist.items():
        if prob > 0:
            entropy += - prob * np.log2(prob)
    return entropy

def make_codespace_joint_distribution(code, pvec):
    """
    Calculates the joint distribution of errors on the codespace and syndromes

    code: the stabilizer code
    pvec: the probability vector of the channel

    returns: the entropy of the code space
    """
    error_effect_dist = {}
    p0 = 1 - np.sum(pvec)
    pvec_ext = np.concatenate(([p0], pvec))
    for idx, rate in enumerate(pvec_ext):
        error = index_to_paulistring(idx, code.num_qubits)
        effect = code.logical_effect(PauliString(error))
        syndrome = code.syndrome(PauliString(error))
        effect_key = (tuple(effect[0] + effect[1]), tuple(syndrome))
        if effect_key in error_effect_dist:
            error_effect_dist[effect_key] += rate
        else:
            error_effect_dist[effect_key] = rate
    return error_effect_dist

def calc_codespace_conditional_entropy(code, pvec):
    """
    Calculates the conditional entropy of the code space given the syndrome

    code: the stabilizer code
    pvec: the probability vector of the channel

    returns: the entropy of the code space
    """
    error_effect_dist = make_codespace_joint_distribution(code, pvec)
    entropy = 0
    # first identify the syndromes and calculate their marginals
    syndrome_marginals = {}
    for key, prob in error_effect_dist.items():
        syndrome = key[1]
        if syndrome in syndrome_marginals:
            syndrome_marginals[syndrome] += prob
        else:
            syndrome_marginals[syndrome] = prob

    # calculate the conditional entropy
    for key, prob in error_effect_dist.items():
        logical_class = key[0]
        syndrome = key[1]
        if prob > 0 and syndrome_marginals[syndrome] > 0:
            entropy +=  -prob * np.log2(prob / syndrome_marginals[syndrome])
    return entropy
    

def logical_effect(paulistring, code):
    """
    Computes the logical effect of a Pauli string on a code
    """
    # the logical effect is the stabilizer of the Pauli string
    return code.stabilizer(paulistring)