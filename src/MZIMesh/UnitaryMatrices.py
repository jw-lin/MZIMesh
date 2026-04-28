""" some functions to generate useful unitary matrices that we might want to upload for wavefront sensing,
    coronagraphy, etc, and to solve for the necessary phase shifts on a mesh.
 """

import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import dft,null_space

#region matrix decomposition

def MZ(phi,omega):
    """ return 2x2 MZ interferometer matrix """
    return np.array([[np.exp(1.j*phi)*np.sin(omega),np.exp(1.j*phi)*np.cos(omega)],[np.cos(omega),-np.sin(omega)]],dtype=np.complex128).T

def expand(m,N,i,j):
    """ expand 2x2 matrix to NxN"""
    out = np.identity(N,dtype=np.complex128)
    out[i,i] = m[0,0]
    out[i,j] = m[0,1]
    out[j,i] = m[1,0]
    out[j,j] = m[1,1]
    return out

def solve_angles(m1,m2):
    def equations(p):
        phi,omega = p
        c = m1 * np.exp(1.j*phi)*np.cos(omega) - m2*np.sin(omega)
        re,im = np.real(c),np.imag(c)
        return(re,im)
    p,o = fsolve(equations,(np.pi,np.pi))
    rem_p, rem_0 = p%(2*np.pi),o%(np.pi)
    if np.isclose(rem_p,2*np.pi) or np.isclose(rem_p,0.):
        rem_p = 0.
    if np.isclose(rem_0,np.pi) or np.isclose(rem_0,0.):
        rem_0 = 0.
    return rem_p,rem_0

def decompose(u,verbose=False):
    """ given an arbitrary unitary matrix u, express its INVERSE in terms of 2x2 MZI transformations
        in a full triangular arrangement, with some diagonal phase corrections at the end. pass the 
        inverse of u to get a decomposition of u. uses Reck's method.
    
    Returns:
        pairs (list[tuple]): list of pairs which are to be sequentially interfered, giving the triangular structure
        phis (list): list of phi angles % 2 pi. phi is the shifter at the MZI input
        omegas (list): list of omega angles % 2 pi; omega is theta/2; theta is the shifter between the couplers
        phase_cors (list): list of phase corrections needed to be applied at the end
        Ms (list): list of MZI transformations corresponding to above channel pairs and shifter angles
    """
    _u = np.copy(u)
    N = u.shape[0]-1
    pairs = []
    # make interfered pairs for full tri mesh
    for i in range(N):
        for j in range(N-1,i-1,-1):
            pairs.append((j+1,j))
    
    zero_order = []
    for i in range(N):
        for j in range(N,i,-1):
            zero_order.append((j,i))
    phis = []
    omegas = []
    Ms = []
    for k in range(len(pairs)):
        p = pairs[k]
        i,j = p
        # for each step, the goal is to zero out 
        # matrix elmnt U[zi,zj] by interfering
        # channels i,j
        zi,zj = zero_order[k]
        if np.isclose(_u[zi,zj],0.):
            phi,omega = 0.,np.pi/2 # bar state
        else:
            phi,omega = solve_angles(_u[zi-1,zj],_u[zi,zj])
        m = MZ(phi,omega)*1.j*np.exp(1.j*omega)
        #print(m)
        M = expand(m,N+1,j,i)
        #print("M: ",np.round(np.abs(M),decimals=2))
        _u = np.dot(M,_u)
        if verbose:
            print("reduced |u|: ")
            print(np.round(np.abs(_u),decimals=2))
        Ms.append(M)
        phis.append(phi)
        omegas.append(omega)

    # phase correction
    phase_cors = []    
    for k in range(_u.shape[0]):
        cor = -np.angle(_u[k,k])
        phase_cors.append(cor)
    Ms.append(np.diag(np.exp(1.j*np.array(phase_cors))))

    return pairs,phis,omegas,phase_cors,Ms

def is_unitary(m):
    return np.allclose(np.eye(m.shape[0]), np.conj(m.T) @ m)

def get_mesh_config(U):
    assert is_unitary(U), "matrix is not unitary"
    pairs,phis,omegas,phase_cors,Ms = decompose(np.conj(U).T)
    phis,thetas = np.array(phis),np.array(omegas)*2
    return phis,thetas

#endregion

#region matrices

def circ(N):
    """ optimal WFS matrix when coupling a spatially discretized pupil-plane e-field.
        it's called 'circ' because the matrix is circulant.
    """
    U = 1/np.sqrt(N) * np.exp(2*np.pi*1.j * np.arange(N)[None,:]*np.arange(N)[:,None]  / N)
    eigs = np.ones(N,dtype=np.complex128)*1.j
    eigs[0] = -1.
    out = np.dot(np.conj(U.T),np.dot(np.diag(eigs),U))
    phase0 = np.arctan(N-1)
    return out*np.exp(1.j * -phase0)

def DFT(N):
    """ 1D DFT of dimension N """
    return dft(N,scale="sqrtn")

def DFT2(N):
    """ matrix rep of 2D DFT for an NxN image, assuming the image has been vectorized to 1 x (NxN) """
    d = DFT(N)
    return np.kron(d,d)

def WFS_unbalanced(ref_field):
    """ an extension of circ to e-fields with non-uniform intensity. """
    
    N = ref_field.shape[0]

    eigs = np.ones(N,dtype=np.complex128)
    eigs[0] = 1.j

    kern = null_space([ref_field]).T
    K = np.empty((N,N),dtype=np.complex128)
    K[0,:] = ref_field
    K[1:,:] = kern

    return K.T @ np.diag(eigs) @ K

#endregion