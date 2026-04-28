""" from a set of relative phase bias measurements, compute a set of physical ones. """

import numpy as np

def make_phi_matrix(Nrows):
    dim = int(Nrows*(Nrows-1)/2 )
    mat = np.zeros((dim,dim))
    Ndiags = Nrows - 1

    j = 0
    len_diags = [Nrows-j-1 for j in range(Nrows-1)]
    for idiag in range(Ndiags):
        for i in range(len_diags[idiag]):
            start = i
            for k in range(idiag+1):
                if i > 0:
                    mat[j,start] = 1.0
                    mat[j,start-1] = -1.0
                else:
                    mat[j,start] = 1.0
                start += len_diags[k]
            j+=1
    return mat

def solve_for_phi0s(phi_rels):
    """ solve for one set of values for the phases biases of phi shifters, given 
        a set of relative calibration measurements.
    ARGS:
        phi_rels: list of relative phi bias measurements in diagonal order.
    RETURNS:
        (np.ndarray): array of physical bias values in same diag order, in the range [-pi,pi].
    """
    N = len(phi_rels)
    Nrows = int((np.sqrt(8*N+1)+1)/2)
    return np.mod(np.linalg.inv(make_phi_matrix(Nrows)) @ np.array(phi_rels) + np.pi , 2*np.pi) - np.pi



