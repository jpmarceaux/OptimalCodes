import numpy as np
import matplotlib.pyplot as plt
import stim
from stim import PauliString
from scipy.stats import entropy

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

class SeparablePauliNoise:
    def __init__(self, p_x, p_y, p_z, num_qubits=1):
        self.p_x = p_x
        self.p_y = p_y
        self.p_z = p_z
        self.num_qubits = num_qubits

    @property 
    def pvec_1q(self):
        px = self.p_x
        py = self.p_y
        pz = self.p_z
        return np.array([1-px-py-pz, px, py, pz])

    def pvec_joint(self):
        # tensor the pvec_1 for each qubit
        pvec = self.pvec_1q
        pvec_joint = np.array([1])
        for i in range(self.num_qubits):
            pvec_joint = np.kron(pvec_joint, pvec)
        return pvec_joint

    def sample_error(self):
        pvec = self.pvec_joint()
        return random_nqb_paulistring(pvec=pvec[1:], num_qubits=self.num_qubits)
    
    def entropy(self):
        pvec = self.pvec_joint()
        return entropy(pvec, base=2)
    
    def entropy_1q(self):
        pvec = self.pvec_1q
        return entropy(pvec, base=2)
    

class BiasedPauliNoise(SeparablePauliNoise):
    def __init__(self, px, pz, num_qubits=1):
        pZ = pz
        pX = px
        pY = pX*pZ
        super().__init__(pX, pY, pZ, num_qubits=num_qubits)

class DepolarizingChannel(SeparablePauliNoise):
    def __init__(self, p, num_qubits):
        pZ = p/3 
        pX = p/3
        pY = p/3
        super().__init__(pX, pY, pZ, num_qubits)

def calculate_logical_marginal_distribution(joint_distribution):
    """
    calculate the marginal distribution of the logical operators
    """
    logical_distribution = {}
    for key, val in joint_distribution.items():
        logical_effect = key[1]
        if logical_effect in logical_distribution:
            logical_distribution[logical_effect] += val
        else:
            logical_distribution[logical_effect] = val
    return logical_distribution

def calculate_syndrome_marginal_distribution(joint_distribution):
    """
    calculate the marginal distribution of the syndrome operators
    """
    syndrome_distribution = {}
    for key, val in joint_distribution.items():
        syndrome = key[0]
        if syndrome in syndrome_distribution:
            syndrome_distribution[syndrome] += val
        else:
            syndrome_distribution[syndrome] = val
    return syndrome_distribution


def calculate_logical_entropy(joint_distribution):
    logical_marginal = calculate_logical_marginal_distribution(joint_distribution)
    logical_pvec = np.array(list(logical_marginal.values()))
    return entropy(logical_pvec, base=2)

def calculate_conditional_entropy(joint_distribution):
    # calculate the entropy of the logical operators conditioned on the syndrome
    conditional_entropy = 0
    joint_entropy = entropy(np.array(list(joint_distribution.values())), base=2)
    syndrome_marginal = calculate_syndrome_marginal_distribution(joint_distribution)
    syndrome_entropy = entropy(np.array(list(syndrome_marginal.values())), base=2)
    return joint_entropy - syndrome_entropy

def calculate_mutual_information(joint_distribution):
    return calculate_logical_entropy(joint_distribution) - calculate_conditional_entropy(joint_distribution)
