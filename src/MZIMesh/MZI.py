import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
from itertools import combinations
from matplotlib import rcParams
from MZIMesh.UnitaryMatrices import get_mesh_config
from MZIMesh.PhiBiasCalculator import solve_for_phi0s
import csv

if 'left' in rcParams['keymap.back']:
    rcParams['keymap.back'].remove('left')
if 'right' in rcParams['keymap.forward']:
    rcParams['keymap.forward'].remove('right')

rng = np.random.default_rng(seed=123)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

#region aux stuff

class index_tracker:
    def __init__(self,update_func=None):
        self._idx = 0
        self._update_func = update_func
    @property
    def idx(self):
        return self._idx
    @idx.setter
    def idx(self,val):
        self._idx = val
        if self._update_func is not None:
            self._update_func(val)

def make_simple_XT_matrix(N,XT,dtype=np.float64):
    """ create a crosstalk matrix of size NxN, where the off-diagonal terms are XT * the diagonal value """
    B = np.full((N,N),XT,dtype=dtype)
    np.fill_diagonal(B,1)
    return B

def make_random_XT_matrix(N,XT,dtype=np.float64):
    """ create a symmetric positive crosstalk matrix where the elements are (pseudo) normally distributed with stdev XT """
    var = 0.5*XT**2
    dB_sample = np.random.normal(0,np.sqrt(var),(N,N))
    dB_sym = np.abs(dB_sample + dB_sample.T)
    np.fill_diagonal(dB_sym,0.)
    B = np.identity(N) + dB_sym
    return B.astype(dtype)

def make_uniform_random_XT_matrix(N,XT, dtype=np.float64):
    """ create a symmetric positive crosstalk matrix where the elements are uniformly distributed from 0 to XT"""
    
    B = np.identity(N,dtype=dtype)
    samp = rng.uniform(0,XT,size=int(N*(N-1)/2))
    k = 0
    for i in range(N):
        for j in range(i+1,N):
            B[i,j] = B[j,i] = samp[k]
            k+=1
    return B

def _max(connectivity):
    """ get max of a nested list of tuples """
    if isinstance(connectivity, tuple):
        return max(connectivity)
    elif isinstance(connectivity, list):
        return max([_max(c) for c in connectivity])

def trimesh_connectivity(Nchannels,repeat=1):
    """ make a connectivity list for a full triangular mesh with Nchannels """
    bottom_channel = 0
    connectivity = []
    start_idx = 0
    # build first half
    while bottom_channel < Nchannels-2:
        col = []
        for j in range(start_idx,bottom_channel+1,2):
            col.append((j,j+1))
        if len(col) == 1:
            col = col[0]
        connectivity.append(col)
        bottom_channel += 1
        start_idx = 1 - start_idx
    # build second half
    while bottom_channel >= 0:
        col = []
        for j in range(start_idx,bottom_channel+1,2):
            col.append((j,j+1))
        if len(col) == 1:
            col = col[0]
        connectivity.append(col)
        bottom_channel -= 1
        start_idx = 1 - start_idx
    # repeat
    return [col for col in connectivity for k in range(repeat)]

def rectmesh_connectivity(Nchannels,repeat=1):
    """ make connectivity list for a full rectangular mesh """
    total_mzis = int(Nchannels * (Nchannels-1) / 2)
    num_mzis = 0
    start_idx = 0
    connectivity = []
    while num_mzis < total_mzis:
        col = []
        for j in range(start_idx,Nchannels-1,2):
            col.append((j,j+1))
            num_mzis += 1
        connectivity.append(col)
        start_idx = 1 - start_idx
    return [col for col in connectivity for k in range(repeat)]

#endregion

class MZI:
    """ class for a 2x2 MZI. takes into account the following: 
        - couplers that aren't perfectly 50:50
        - nonzero offsets for phase shifters at 0 V
        - linear crosstalk between phase shifters
    
        to actuate, pass voltages to MZI.Vphi and MZI.Vtheta
        to get the output intensity for a given input_vector
        call MZI.measure(input_vector)
    """
    # conversion ratio between squared voltage to phase delay
    voltagesq_to_phase = 1.0

    # number of input photons, for simulating photon noise. if 0, ignore photon noise.
    Nphot = 0.
    
    # DAC simulation params
    # the allowable voltages are assumed to be in discrete steps
    # of size max_voltage / 2^DAC_bit_depth
    # idk how real DACs work lol
    DAC_bit_depth = 12
    max_voltage = np.sqrt(4 * np.pi) # 2 cycles for voltage_sq_to_phase = 1
    actuation_noise = 0.0 # volts, corresponds to gaussian sigma on output voltage

    # simulate DAC noise/discretization error
    DACsim = False

    # simulate crosstalk within the MZI
    # when MZI objects are incorporated into 
    # needs to be computed over the whole mesh
    # so this should be false
    localXT = False

    # datatype for complex calcs
    dtype = np.complex128

    # configs: 
    # 1 = phi - coupler - theta - coupler
    # 2 = coupler - phi - coupler - theta
    shifterconfig = 1

    # voltage to phase mapping
    realistic_volts2rads = True
    v2r_B = 197.3384152262288 # linear
    v2r_C = -365.0693897652933 # quad
    v2r_R = 818.1419041173418 # resistance
    maxV = 6.0 # maximum voltage

    modulo = True

    def __init__(self,XT=0,phi0=0,theta0=0,dg1=0,dg2=0,port0ix=0,port1ix=1,colix=0):
        """ a more realistic 2x2 mzi.
        
        ARGS:
            XT: crosstalk between phase shifters, parameterized
                as a fraction of voltagesq_to_phase
            phi0: phi phase shift at 0 volts
            theta0: theta phase shift at 0 volts
            dgi: split ratio imperfection for coupler i,
                 the actual split ratio is 0.5 + dgi
            port0ix: index of channel w/ phase shifters, default 0
            port1ix: index of other channel, default 1
            colix: index of the mesh column to insert this MZI; ignored for single MZIs
        """
        # modification flags
        self._modifiedA1 = False
        self._modifiedA2 = False
        self._modifiedA3 = False
        self._modifiedA4 = False

        # property stuff
        self._phi0 = phi0
        self._theta0 = theta0
        self._XT = XT
        self._dg1,self._dg2 = dg1,dg2
        self._Vphi = 0.
        self._Vtheta = 0.
        self._phi = phi0
        self._theta = theta0
        self._phase0 = np.array([phi0,theta0],dtype=np.float64)
        self._V = np.zeros(2,dtype=np.float64)

        if self.realistic_volts2rads:
            self.voltagesq_to_phase = 1.0

        # crosstalk matrix
        B = np.identity(2,dtype=np.float64)
        B[0,1] = B[1,0] = XT
        self._B = B*self.voltagesq_to_phase

        # coupler matrices
        self.A1 = self.compute_A1()
        self.A3 = self.compute_A3()
        
        # phase delay matrices, assuming 0 voltage
        self.A2 = self.compute_A2()
        self.A4 = self.compute_A4()

        self.port0ix = port0ix # default channel indices to interfere
        self.port1ix = port1ix # only the 0 port has the phase shifters
        self.colix = colix
        self.ix = np.ix_([self.colix],[port0ix,port1ix],[port0ix,port1ix])

        # bar and cross trackers
        self._Vthetabar = None
        self._Vthetacross = None

    def set_bar(self):
        assert self._Vthetabar is not None
        self.Vtheta = self._Vthetabar

    def set_cross(self):
        assert self._Vthetacross is not None
        self.Vtheta = self._Vthetacross

    def set_0V(self):
        self.Vphi = 0.
        self.Vtheta = 0.

    def get_state(self):
        _state = (self.phi0,self.theta0,self.Vphi,self.Vtheta,self.dg1,self.dg2)
        return _state

    def load_state(self,state):
        phi0,theta0,Vphi,Vtheta,dg1,dg2 = state
        self.phi0 = phi0
        self.theta0 = theta0
        self.Vphi = Vphi
        self.Vtheta = Vtheta
        self.dg1 = dg1
        self.dg2 = dg2
    
    def realistic_phase(self,V):
        # with crosstalk
        x = np.power(V,2)/self.v2r_R
        return self._B @ (self.v2r_B * x) + (self.v2r_C * np.power(x,2)) + self.phase0
    
    def realistic_phi(self,Vphi):
        # without crosstalk
        x = np.power(Vphi,2)/self.v2r_R
        return self.v2r_B * x + (self.v2r_C * np.power(x,2)) + self.phase0[0]

    def realistic_theta(self,Vtheta):
        # without crosstalk
        x = np.power(Vtheta,2)/self.v2r_R
        return self.v2r_B * x + (self.v2r_C * np.power(x,2)) + self.phase0[1]

    def compute_phase(self,_set=True):
        if self.realistic_volts2rads:
            x = np.power(self.V,2)/self.v2r_R
            # is the nonlinear term also supposed to go into crosstalk?? for now it's not
            # seems like that would be fourth order or something
            phi,theta = self.realistic_phase(self.V)
        else:
            phi,theta = self._B @ np.power(self.V,2) + self.phase0
        if _set:
            self.phi = phi
            self.theta = theta
        return phi,theta

    def compute_A1(self):
        dg1 = self.dg1
        return np.array([[np.sqrt(0.5+dg1),1.j*np.sqrt(0.5-dg1)],[1.j*np.sqrt(0.5-dg1),np.sqrt(0.5+dg1)]],dtype=self.dtype)
    
    def compute_A2(self):
        return np.array([[np.exp(1.j*self.phi),0.],[0.,1.]],dtype=self.dtype)

    def compute_A3(self):
        dg2 = self.dg2
        return np.array([[np.sqrt(0.5+dg2),1.j*np.sqrt(0.5-dg2)],[1.j*np.sqrt(0.5-dg2),np.sqrt(0.5+dg2)]],dtype=self.dtype)
    
    def compute_A4(self):
        return np.array([[np.exp(1.j*self.theta),0.],[0.,1.]],dtype=self.dtype)
    
    def zero(self):
        self.dg1 = 0.
        self.dg2 = 0.
        self.phi0 = 0.
        self.theta0 = 0.
        self.Vphi = 0.
        self.Vtheta = 0.
        self.phi = 0.
        self.theta = 0.

    #region mzi properties

    @property # phase offset for phi shifter at 0 V
    def phi0(self):
        return self._phi0
    @phi0.setter
    def phi0(self,val):
        self._phi0 = val
        self._phase0[0] = val
        self._modifiedA2 = True

    @property # phase offset for theta shifter at 0 V
    def theta0(self):
        return self._theta0
    @theta0.setter
    def theta0(self,val):
        self._theta0 = val
        self._phase0[1] = val
        self._modifiedA4 = True
    
    @property # read-only; array of phase offsets
    def phase0(self):
        return self._phase0
        
    @property # split ratio imperfection for first coupler
    def dg1(self):
        return self._dg1
    @dg1.setter
    def dg1(self,val):
        self._dg1 = val
        self._modifiedA1 = True

    @property  # split ratio imperfection for second coupler
    def dg2(self):
        return self._dg2
    @dg2.setter
    def dg2(self,val):
        self._dg2 = val
        self._modifiedA3 = True

    @property # crosstalk (off diagonal terms of B matrix)
    def XT(self):
        return self._XT
    @XT.setter
    def XT(self,val):
        if not self.localXT:
            print("because localXT=False, this will have no effect. set XT through the mesh, not the MZIs.")
        self._XT = val
        self._B[0,1] = self._B[1,0] = val

    @property # voltage setting for phi shifter
    def Vphi(self):
        return self._Vphi
    @Vphi.setter
    def Vphi(self,val):
        if self._Vphi == val:
            return
        self._Vphi = val
        self._V[0] = val
    
    @property # voltage setting for theta shifter
    def Vtheta(self):
        return self._Vtheta
    @Vtheta.setter
    def Vtheta(self,val):
        if self._Vtheta == val:
            return
        self._Vtheta = val 
        self._V[1] = val

    @property # phase setting for phi shifter
    def phi(self):
        return self._phi
    @phi.setter
    def phi(self,val):
        if self._phi == val:
            return
        self._phi = val
        self._modifiedA2 = True

    @property # phase setting for theta shifter
    def theta(self):
        return self._theta
    @theta.setter
    def theta(self,val):
        if self._theta == val:
            return
        self._theta = val
        self._modifiedA4= True

    @property # read-only; output voltage sent to phase shifters, which can include DAC effects
    def V(self):
        if self.DACsim:
            _Vdiscrete = (self._V/self.max_voltage * 2**self.DAC_bit_depth ).astype(int) * (self.max_voltage / 2**self.DAC_bit_depth)
            return rng.normal(_Vdiscrete,self.actuation_noise)       
        return self._V

    @property # read-only; transfer matrix of the MZI
    def A(self):
        if self.localXT:
            self.compute_phase()
        if self._modifiedA1:
            self.A1 = self.compute_A1()
        if self._modifiedA3:
            self.A3 = self.compute_A3()
        if self._modifiedA2:
            self.A2 = self.compute_A2()
        if self._modifiedA4:
            self.A4 = self.compute_A4()
        self.modified = False
        if self.shifterconfig == 1:
            return self.A3@self.A4@self.A1@self.A2
        else:
            return self.A4@self.A3@self.A2@self.A1
    
    @property # checks is transfer matrix needs to be recomputed
    def modified(self):
        return self._modifiedA1 or self._modifiedA2 or self._modifiedA3 or self._modifiedA4
    @modified.setter
    def modified(self,val:bool):
        self._modifiedA1 = self._modifiedA2 = self._modifiedA3 = self._modifiedA4 = val

    #endregion

    def prop(self,u):
        """ propagate the complex electric field, represented by the two-element vector u, through the MZI 
        Args:
            u (np.ndarray): electric field vector at mesh input, may be complex
        Returns:
            (np.ndarray): complex-valued output electric field vector
        """
        return self.A @ u
    
    def measure(self,u):
        """ measure the output intensity for a given input field vector u. accounts for 
        photon noise if MZI.Nphot > 0

        Args:
            u (np.ndarray): electric field vector at mesh input, may be complex
        Returns:
            (np.ndarray): real nonnegative output intensity vector
        """
        amps = self.prop(u)
        pows = np.power(np.abs(amps),2)
        if self.Nphot != 0:
            pows *= self.Nphot
            return rng.poisson(pows)
        else:
            return pows
    
    def phi2Vphi(self,phi):
        """ compute voltage value to achieve phi phase shift (modulo 2 pi), ignoring crosstalk 
            sln via quadratic formula
        """
        phi -= np.floor((phi-self.phi0)/(2*np.pi))*2*np.pi
        if self.realistic_volts2rads:
            x = (np.sqrt(self.v2r_B**2 + 4*self.v2r_C*(phi-self.phi0)) - self.v2r_B)/(2*self.v2r_C)
            if x < 0:
                print("no sln found")
                return np.nan
            return np.sqrt(self.v2r_R*x)
        else:
            while phi-self.phi0 <0:
                phi += 2*np.pi
            return np.sqrt((phi-self.phi0)/self.voltagesq_to_phase)

    def theta2Vtheta(self,theta,theta0=None):
        """ compute voltage value to achieve phi phase shift (modulo 2 pi), ignoring crosstalk 
            sln via quadratic formula
        """
        theta0 = self.theta0 if theta0 is None else theta0
        if self.modulo:
            theta -= np.floor((theta-theta0)/(2*np.pi))*2*np.pi
        if self.realistic_volts2rads:
            x = (np.sqrt(self.v2r_B**2 + 4*self.v2r_C*(theta-theta0)) - self.v2r_B)/(2*self.v2r_C)
            if x < 0:
                print("no sln found")
                return np.nan
            return np.sqrt(x*self.v2r_R)
        else:
            while theta-theta0 <0:
                theta += 2*np.pi
            return np.sqrt((theta-theta0)/self.voltagesq_to_phase)

    def expected_null_voltage(self,order=0):
        """ compute the voltage setting corresponding to 
            the null of a given order, assuming no crosstalk.
        """
        return np.sqrt((order*np.pi-self.phi0)/self.voltagesq_to_phase)

class Mesh(MZI):
    """ class for simulating meshes of MZIs. the mesh structure is saved 
        as a nested list of (i,j) tuples, where each tuple corresponds to an MZI
        interfering channels i and j. individual MZIs may be actuated or tuned, 
        this class will compute the corresponding global optical transformation.
    """

    mziorder = "column" # can also be "diagonal" for triangular meshes

    def __init__(self,connectivity:list,XT=0.0):
        """ 
        ARGS:
            connectivity: a potentially nested list of 2-element tuples.
                each tuple gives the channel indices which are interfered by an MZI.
                each element of connectivity is a column in the mesh, which may be a
                single MZI or multiple (but a channel should not be interfered more than 
                once per column.)
            XT: crosstalk amount, as in the MZI class. for now, all pairs of shifters
                are assumed to have the same crosstalk.
        """

        self.mzimeshflat = []

        self.connectivity = connectivity
        self.Ncols = len(connectivity)
        self.Nchannels = _max(connectivity) + 1
        self.construct_mesh_from_connectivity(connectivity)
        self.Nmzis = len(self.mzimeshflat)
        self.Nshifters = len(self.mzimeshflat) * 2

        self.diagonal_order = None

        if self.mziorder == "diagonal":
            self.diagonal_order = self.get_diagonal_order()
            # resort
            self.mzimeshflat = [self.mzimeshflat[i] for i in self.diagonal_order]
        
        self._XT = XT
        self._B = make_simple_XT_matrix(self.Nshifters,XT) * self.voltagesq_to_phase

        # array of transfer matrices for column of the mesh
        self.Acols = np.repeat(np.identity(self.Nchannels,dtype=self.dtype)[None,:,:],self.Ncols,axis=0)
        for m in self:
            self.Acols[m.ix] = m.A # initialize

        self._voltages = np.empty(self.Nshifters,dtype=np.float64)
        self._phase0 = np.zeros(self.Nshifters,dtype=np.float64)
        self._truephases = np.empty(self.Nshifters,dtype=np.float64)

        self._scannedchannel = None
        self._savedstate = None

    def get_state(self):
        global_state = []
        for m in self:
            global_state.append(m.get_state())
        global_state.append(self.B)
        return global_state
    
    def save_state(self):
        self._savedstate = self.get_state()

    def load_state(self,global_state):
        for m,state in zip(self,global_state[:-1]):
            m.load_state(state)
        self.B = global_state[-1]

    def get_column_order(self):
        idxs = []
        j=0
        for c in self.connectivity:
            if isinstance(c,list):
                _idx = []
                for _c in c:
                    _idx.append(j)
                    j+=1
                idxs.append(_idx)
            else:
                idxs.append([j])
                j+=1
        return idxs

    def get_diagonal_order(self):
        """ get the index permutation which goes through the MZIs in diagonal order """
        Ndiags = self.Nchannels - 1

        colorder = self.get_column_order()
        diag_order_idx = 0
        diag_order_idxs = []
        
        for j in range(Ndiags):
            cur_col = 2*j
            cur_row = 0
            # ascend the diagonal
            for k in range(Ndiags-j):
                if k%2 == 0 and k>0:
                    cur_row += 1
                diag_order_idx = colorder[cur_col][cur_row]
                diag_order_idxs.append(diag_order_idx)
                cur_col += 1
        return diag_order_idxs

    def get_top_row_ixs(self,Nrows:int):
        ixs = []
        for i,m in enumerate(self):
            if m.port1ix <= Nrows:
                ixs.append(i)
        return ixs

    #region mesh properties

    @property
    def realistic_volts2rads(self):
        return self[0].realistic_volts2rads
    @realistic_volts2rads.setter
    def realistic_volts2rads(self,val:bool):
        for m in self:
            m.realistic_volts2rads = val

    @property
    def modulo(self):
        return self[0].modulo
    @modulo.setter
    def modulo(self,val:bool):
        for m in self:
            m.modulo = val

    @property
    def shifterconfig(self):
        return MZI.shifterconfig
    @shifterconfig.setter
    def shifterconfig(self,val):
        MZI.shifterconfig = val

    @property
    def DACsim(self):
        return self[0].DACsim
    @DACsim.setter
    def DACsim(self,val:bool):
        for m in self:
            m.DACsim = val

    @property # crosstalk (assumed to be the same between all phase shifters)
    def XT(self):
        return self._XT
    @XT.setter
    def XT(self,val):
        self._XT = val
        self._B = make_simple_XT_matrix(self.Nshifters,val)*self.voltagesq_to_phase
    
    @property # use this to set a more general crosstalk matrix
    def B(self):
        return self._B
    @B.setter
    def B(self,val):
        self._B = val

    @property # read-only; vector of phase offsets
    def phase0(self):
        for i,m in enumerate(self):
            self._phase0[2*i:2*i+2] = m.phase0
        return self._phase0

    @property # vector of applied voltages. readout may include DAC effects
    def V(self):
        for i,m in enumerate(self):
            self._voltages[2*i:2*i+2] = m.V
        return self._voltages
    @V.setter
    def V(self,val):
        for i,m in enumerate(self):
            m.Vphi = val[2*i]
            m.Vtheta = val[2*i+1]

    def set_V(self,val,ps):
        if ps%2 == 0:
            self[int(ps/2)].Vphi = val
        else:
            self[int(ps/2)].Vtheta = val

    @property # read-only; global transfer matrix
    def A(self):
        self.compute_phase()
        for m in self:
            if m.modified:
                self.Acols[m.ix] = m.A
        if self.Acols.shape[0]==1:
            return self.Acols[0]
        return np.linalg.multi_dot(self.Acols)
    
    #endregion

    def noiseless_measure(self,u):
        if self.DACsim:
            self.DACsim = False
            amps = self.prop(u)
            self.DACsim = True
        else:
            amps = self.prop(u)
        pows = np.power(np.abs(amps),2)
        return pows

    def compute_phase(self):
        """ read all the voltages on the mesh and compute the corresponding phases """
        
        #technically this should only be called when V or B are changed to save time
        if self.realistic_volts2rads:
            x = np.power(self.V,2)/self.v2r_R
            self._truephases[:] = self.B @ (self.v2r_B * x) + (self.v2r_C * np.power(x,2)) + self.phase0
        else:
            self._truephases[:] = self.B @ np.power(self.V,2) + self.phase0

        # set phases
        for i,mzi in enumerate(self):
            mzi.phi = self._truephases[2*i]
            mzi.theta = self._truephases[2*i+1]
        
        return self._truephases

    def construct_mesh_from_connectivity(self,connectivity,flip = True):
        """ make a list of mzis with the same structure as connectivity 
        ARGS:
            connectivity: a potentially nested list of (i,j) tuples, each corresponding
                to an MZI
        """
        mzimeshflat = []

        for i,col in enumerate(connectivity):
            if isinstance(col,list):
                for c in col:
                    if flip:
                        m = MZI(port1ix=self.Nchannels-1-c[0],port0ix=self.Nchannels-1-c[1],colix=self.Ncols-i-1)
                    else:
                        m = MZI(port0ix=c[0],port1ix=c[1],colix=self.Ncols-i-1)
                    mzimeshflat.append(m)
            else:
                if flip:
                    m = MZI(port1ix=self.Nchannels-1-col[0],port0ix=self.Nchannels-1-col[1],colix=self.Ncols-i-1)
                else:
                    m = MZI(port0ix=col[0],port1ix=col[1],colix=self.Ncols-i-1)
                mzimeshflat.append(m)
        
        self.mzimeshflat = mzimeshflat

    def set_random_XT(self,XT=0.1):
        """ sample a value for the crosstalk between shifters from a uniform distribution with range [0,XT]. """
        self.B = make_uniform_random_XT_matrix(self.Nshifters,XT)

    def set_random_params(self,sdg=0.01,sphase0=np.pi/2):
        """ sample a value for dg and phas0 params for every mzi in the mesh from a 0-centered normal distribution with 
            the corresponding standard deviations sdg and sphase0.
        """
        for m in self:
            if sdg>0 :
                m.dg1 = rng.normal(0,sdg)
                m.dg2 = rng.normal(0,sdg)
            if sphase0>0:
                m.phi0 = rng.normal(0,sphase0)
                m.theta0 = rng.normal(0,sphase0)

    def set_random_phi0(self,s=1.0):
        for m in self:
            m.phi0 = rng.normal(0,s)

    def set_random_theta0(self,s=1.0):
        for m in self:
            m.theta0 = rng.normal(0,s)

    def show(self,fig=None,ax=None,mzitracker=None):
        """ plot the mesh layout """
        show=False
        if fig is None:
            show=True
            fig,ax = plt.subplots(1,1)
        
        self.compute_phase()
        Ncols = len(self.connectivity)
        channel_lines = []
        for i in range(self.Nchannels):
            channel_line = ax.axhline(y=i,xmin=-1,xmax=Ncols+1,color=color_cycle[i],picker=True)
            channel_line.channelidx = i
            channel_lines.append(channel_line)
            ax.annotate("ch. "+ str(i),(-1,i),xytext=(-30,0),textcoords="offset points")
            
        i = 0
        rects = []
        annotations = []

        for mzi in self:
            i0 = mzi.port0ix
            i1 = mzi.port1ix
            xcur = self.Ncols - mzi.colix - 1
            mzirect = patches.FancyBboxPatch(
                (xcur-0.4,min(i0,i1)-0.4), 
                width=0.8, 
                height=max(i0,i1)-min(i0,i1) + 0.8, 
                fc='lightblue',
                ec='none',
                boxstyle='round,pad=0.05,rounding_size=0.2',
                fill=True,
                alpha=0.5,
                picker=True
            )
            mzirect.idx = i
            ax.add_patch(mzirect)
            rects.append(mzirect)
            fontsize = 24/np.sqrt(self.Nchannels)
            ax.annotate(str(i),(xcur,(i0+i1)/2+0.2),ha='center',va='center',fontsize=fontsize*1.4)
            VPtxt = ax.annotate(r"$V_\phi=$"+str(round(mzi.Vphi,2)),(xcur-0.4,i0),xytext=(0,0.45*fontsize),fontsize=fontsize,textcoords="offset points")
            Vttxt = ax.annotate(r"$V_\theta=$"+str(round(mzi.Vtheta,2)),(xcur-0.4,i0),xytext=(0,-fontsize),fontsize=fontsize,textcoords="offset points")
            annotations.append([VPtxt,Vttxt])
            i+=1

        ax.set_xlim(-1,Ncols)
        ax.set_ylim(-1,self.Nchannels)
        ax.invert_yaxis()
        if not show:
            rects[0].set_edgecolor('k')
        else:
            plt.show()
            return

        def onpick(event):

            if hasattr(event.artist,"idx"):
                recti = event.artist.idx
                for i,r in enumerate(rects):
                    if i == recti:
                        r.set_edgecolor('k')
                    else:
                        r.set_edgecolor('None')

                if mzitracker is not None:
                    mzitracker.idx = recti

        fig.canvas.mpl_connect('pick_event', onpick)

        return annotations,rects,channel_lines

    def __getitem__(self,i):
        return self.mzimeshflat[i]

    @staticmethod
    def make_slider(fig,name,y,val0,lims,ax_left_lim=0.5):
        ax = fig.add_axes([ax_left_lim, y, 0.83-ax_left_lim, 0.03])
        if y == 0.4:
            ax.set_title("adjust mesh params",pad=20)
        return Slider(ax, name, lims[0], lims[1], valinit=val0,valfmt='%.6f')

    def make_sliders(self,fig,maxV,names,ax_left_lim=0.5):
        ys = np.arange(0.4,0.1,-0.05)
        
        lims = [(0,0.1),(-np.pi,np.pi),(-np.pi,np.pi),(-0.03,0.03),(-0.03,0.03),(0,maxV),(0,maxV)]
        
        vals = [self.XT,0,0,0,0,0,0]

        sliders = [self.make_slider(fig,name,y,val,lim,ax_left_lim=ax_left_lim) for name,y,val,lim in zip(names,ys,vals,lims)]
        for i,s in enumerate(sliders):
            s.track.slider_idx = i
            s.track.set_picker(10.0)
        return sliders

    def make_XT_selector(self,fig,XTtracker=None):
        ax = fig.add_axes([0.05,0.05,0.45,0.45])
        ax.axis("off")
        Nshifters = self.Nshifters
        ax.annotate("XT term selector",(0,Nshifters),xytext=(-0.1,0.3),rotation=90,fontsize="large",textcoords="axes fraction")
        for i in range(Nshifters):
            if i%2 == 0:
                ax.annotate(r"$V_{\phi," + rf"{{{int(i/2)}}}"+"}$",(Nshifters,i+0.5),fontsize="medium")
                ax.annotate(r"$V_{\phi," + rf"{{{int(i/2)}}}"+"}$",(i+0.5,0),xytext=(i+0.25,-0.25),fontsize="medium",rotation=45)
            else:
                ax.annotate(r"$V_{\theta," + rf"{{{int(i/2)}}}"+"}$",(Nshifters,i+0.5),fontsize="medium")
                ax.annotate(r"$V_{\theta," + rf"{{{int(i/2)}}}"+"}$",(i+0.5,0),xytext=(i+0.25,-0.25),fontsize="medium",rotation=45)
        ax.set_aspect("equal")
        ax.set_xlim(0,Nshifters)
        ax.set_ylim(Nshifters,0)
        boxes = []
        indices = []
        txtvals = []
        k = 0
        for i in range(Nshifters):
            for j in range(Nshifters):
                if j>i:
                    color = "steelblue"
                    box = patches.Rectangle((j,i),1,1,ec='k',fc=color,picker=True)
                    box.XTidx = (i,j)
                    box.XTidx_flat = k
                    indices.append((i,j))
                    txt = ax.annotate(np.format_float_positional(0, precision=2, unique=False, fractional=False, trim='k'),(j+0.2,i+0.5),fontsize=24/(self.Nchannels-1))
                    txtvals.append(txt)
                    boxes.append(box)
                    k += 1
                else:
                    color = '0.5'
                    box = patches.Rectangle((j,i),1,1,ec='k',fc=color)
                ax.add_patch(box)

        def onpick(event):
            rect = event.artist
            if hasattr(rect,"XTidx"):
                boxes[XTtracker.flat_idx].set_fc("steelblue")
                XTtracker.flat_idx = rect.XTidx_flat
                XTtracker.idx = rect.XTidx
                rect.set_facecolor("indianred")
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect('pick_event', onpick)
        boxes[0].set_fc("indianred") # first off-diag term will start selected
        return indices,txtvals

    def phi2Vphi(self,phi,i):
        """ compute the voltage assuming no crosstalk, perfect splitters, and perfect knowledge of phase bias
            in mzi i.
            assumes that phase ~ voltage^2 + bias, but this can be relaxed (e.g. to include linear term).
        """
        return self[i].phi2Vphi(phi)

    def theta2Vtheta(self,theta,i):
        """ compute the voltage assuming no crosstalk, perfect splitters, and perfect knowledge of phase bias
            in mzi i.
            assumes that phase ~ voltage^2 + bias, but this can be relaxed (e.g. to include linear term).
        """
        return self[i].theta2Vtheta(theta)

    def load_matrix(self,U,ignored_rows=[]):
        """ load a matrix onto the mesh, assuming the mesh is perfect. if U is smaller than the mesh
            the matrix will be loaded towards the top point
        """
        #assert self.shifterconfig == 1, "matrix decomp assumes shifter config 1"
        NU = U.shape[0]
        if NU < self.Nchannels:
            _U = np.identity(self.Nchannels,dtype=self.dtype)
            _U[np.ix_(np.arange(NU),np.arange(NU))] = U
        else:
            _U = U

        phis,thetas = get_mesh_config(_U) # these are arranged in diagonal order
        for i,(p,t) in enumerate(zip(phis,thetas)):
            if self.mziorder == "diagonal":
                m = self[i]
            else:
                m = self[self.diagonal_order[i]]
            if m.port0ix in ignored_rows:
                continue
            Vp,Vt = self.phi2Vphi(p,i),self.theta2Vtheta(t,i)
            m.Vphi = Vp
            m.Vtheta = Vt
            m.phi = p
            m.theta = t
        self.A # force computation
        return

    def make_launch_field(self,launch):
        # launch field
        if isinstance(launch,int):
            _input = np.zeros(self.Nchannels,dtype=self.dtype)
            _input[launch] = 1
        elif isinstance(launch,np.ndarray):
            _input = launch
        elif launch == "flat":
            _input = np.ones(self.Nchannels,dtype=self.dtype)/np.sqrt(self.Nchannels)
        return _input

    # probably the messiest function i've ever written lol
    def live_plot(self,launch="flat",XTmode="simple",scanmode="shifter",maxV=None,minphase=None,maxphase=None,yscale="log",xscale="linear",fullscreen=False):
        """ make an interactive plot, which shows the intensity output of the mesh as a function of different scans.
            the scan is customizable via gui. there are two scanning modes. 
            in <scanmode>="shifter", you may scan over phase shifter voltages, like is done in the lab.
            in <scanmode>="field" you may scan over the PHASE of the electric field for an individual channel.
            the starting launch field is set by <input>. for crosstalk, there are two options.
            in <XTmode>="simple", crosstalk is assumed uniform between all shifters.
            in <XTmode>="advanced", crosstalk may be different for each shifter pair. do not use this with
            with large meshes since the number of crosstalk terms scales as (number of channels)^4 (for
            a full mesh) and the GUI can't handle that.
        Args:
            launch (int or np.ndarray or str): complex-valued input vector (assumed normalized). 
                if an integer, launch all power in channel <input>.
                if "flat", light is split evenly (and cophased) between input channels.
            XTmode: default 'simple', which assumes all cross terms are equal. 'advanced' 
                also produces a XT matrix gui so cross terms can be adjusted individually
            scanmode: default "shifter", which lets you scan over any phase shifter (or multiple in parallel).
                if "field", you may scan the phase of the electric field in one input channel. you probably
                want <input>="flat" with this mode. parallel scanning and amplitude scanning are not supported.
            maxV: max voltage for scans over voltage params (min is assumed 0)
            minphase,maxphase: min and max phases for scans over electric field phase.
            yscale: scaling for vertical axis of sweep plot. "log" or "linear"
        """

        assert XTmode in ["simple","advanced"], "XTmode must be 'simple' or 'advanced'"
        assert scanmode in ["shifter","field"], "must scan over 'shifter' params or 'input' electric field"

        if XTmode == "advanced":
            assert self.Nmzis <= 10, "advanced XT gui only works for a small number of MZIs"

        # voltage scan range defaults
        maxV = self.maxV if maxV is None else maxV

        # phase scan range defaults
        minphase = -np.pi if minphase is None else minphase
        maxphase = np.pi if maxphase is None else maxphase

        # for tracking phase shifter scans
        whichmzi=None
        whichshifter=None

        # for tracking input electric field scans
        scannedinput = None

        # save starting state
        starting_state = self.get_state()

        minVstep = np.sqrt(1e-10 / self.voltagesq_to_phase) # approx resolves down to 1e-10 normalized intensity
        target_fractional_change = 0.03
        max_growth_rate = 2
        _compute = True # computation flag

        # launch field
        input = self.make_launch_field(launch)

        def adaptive_eval(maxV,Vstep0):
            """adaptively evaulate chip output as a function of a phase shifter scan"""
            norm = self.Nphot if self.Nphot != 0 else 1
            output = []
            if not self.DACsim and self.Nphot == 0:
                output_noiseless = output
            else:
                output_noiseless = []
            Vscan = []
            curV = 0.0
            Vstep = Vstep0

            if whichmzi is None:
                Vscan = np.array([0,maxV])
                _int = self.noiseless_measure(input)/norm
                output.append(_int)
                output.append(_int)
                return Vscan,np.array(output)

            while curV <= maxV:
                if isinstance(whichmzi,list):
                    for mix,shft in zip(whichmzi,whichshifter):
                        if shft == "phi":
                            self[mix].Vphi = curV
                        else:
                            self[mix].Vtheta = curV
                else:
                    if whichshifter == "phi":
                        self[whichmzi].Vphi = curV
                    else:
                        self[whichmzi].Vtheta = curV
                intensity = self.noiseless_measure(input)/norm
                if len(output_noiseless)>0:
                    diff = np.abs(intensity-output_noiseless[-1])
                    diff[intensity==0] = 0
                    diff[intensity!=0] /= intensity[intensity!=0]
                    if np.all(diff==0):
                        Vstep  *= max_growth_rate
                    elif np.max(diff)<1:
                        growth_rate = min(target_fractional_change / np.max(diff), max_growth_rate)
                        Vstep = max(Vstep * growth_rate,minVstep)
                    elif Vstep>minVstep:
                        curV -= Vstep
                        growth_rate = target_fractional_change / np.max(diff)
                        Vstep = max(Vstep * growth_rate,minVstep)
                        curV += Vstep
                        continue

                if not self.DACsim and self.Nphot == 0:
                    output.append(intensity)
                else:
                    output_noiseless.append(intensity)
                    output.append(self.measure(input)/norm)
                Vscan.append(curV)
                curV += Vstep
            # add the last point
            if isinstance(whichmzi,list):
                for mix,shft in zip(whichmzi,whichshifter):
                    if shft == "phi":
                        self[mix].Vphi = maxV
                    else:
                        self[mix].Vtheta = maxV
            else: 
                if whichshifter == "phi":
                    self[whichmzi].Vphi = maxV
                else:
                    self[whichmzi].Vtheta = maxV
            output.append(self.measure(input)/norm)
            Vscan.append(maxV)
            output = np.array(output)
            return Vscan,output

        minphasestep = np.sqrt(minVstep)
        def adaptive_eval_phase(minphase,maxphase,step0):
            "adaptively evaluate chip output as a funciton of input phase"
            scannedinput = self._scannedchannel
            _input = np.copy(np.array(input))
            norm = self.Nphot if self.Nphot != 0 else 1
            output = []
            if not self.DACsim and self.Nphot == 0:
                output_noiseless = output
            else:
                output_noiseless = []
            
            phasescan = []
            curphase = minphase
            step = step0

            if scannedinput is None:
                phasescan = np.array([minphase,maxphase])
                _int = self.noiseless_measure(_input)/norm
                output.append(_int)
                output.append(_int)
                return phasescan,np.array(output)

            while curphase <= maxphase:
                _input[scannedinput] = np.abs(_input[scannedinput])*np.exp(1.j*curphase)
                intensity = self.noiseless_measure(_input)/norm
                if len(output_noiseless)>0:
                    diff = np.abs(intensity-output_noiseless[-1])
                    diff[intensity==0] = 0
                    diff[intensity!=0] /= intensity[intensity!=0]
                    if np.all(diff==0):
                        step  *= max_growth_rate
                    elif np.max(diff)<1:
                        growth_rate = min(target_fractional_change / np.max(diff), max_growth_rate)
                        step = max(step * growth_rate,minphasestep)
                    elif step>minphasestep:
                        curphase -= step
                        growth_rate = target_fractional_change / np.max(diff)
                        step = max(step * growth_rate,minphasestep)
                        curphase += step
                        continue

                if not self.DACsim and self.Nphot == 0:
                    output.append(intensity)
                else:
                    output_noiseless.append(intensity)
                    output.append(self.measure(_input)/norm)
                phasescan.append(curphase)
                curphase += step

            # add the last point
            _input[scannedinput] = np.abs(_input[scannedinput])*np.exp(1.j*maxphase)
            output.append(self.measure(_input)/norm)
            phasescan.append(maxphase)
            output = np.array(output)
            return phasescan,output

        #fig,(axselector,ax) = plt.subplots(1,2,figsize=(12,8))
        fig = plt.figure(figsize=(12,8))
        if XTmode == "simple":
            axselector = fig.add_axes([0.05,0.55,0.9,0.4])
            ax = fig.add_axes([0.05,0.07,0.4,0.45])
        else:
            axselector = fig.add_axes([0.05,0.55,0.4,0.4])
            ax = fig.add_axes([0.55,0.55,0.4,0.4])

        axselector.set_title("mesh overview")
        if yscale == "log":
            ax.set_ylim(ymin=1e-10,ymax=2)
        else:
            ax.set_ylim(-0.1,1.1)

        if xscale=="quad":
            ax.set_xlim(0,maxV**2)

        axselector.axis("off")
        ax.set_ylabel("normalized flux")
        # initial data
        if scanmode == "shifter":
            scan,output = adaptive_eval(maxV,minVstep)
            ax.set_title("voltage scan result")
            if xscale=="linear":
                xlabel = ax.set_xlabel(r"$V$")
            else:
                xlabel = ax.set_xlabel(r"$V^2$")
        else:
            scan,output = adaptive_eval_phase(minphase,maxphase,minphasestep)
            ax.set_title("input scan result")
            xlabel = ax.set_xlabel("phase, channel "+str(scannedinput))
        fig.subplots_adjust(bottom=0.5)
        
        fluxes = []
        for i in range(self.Nchannels):
            if yscale == "log":
                f = ax.semilogy(scan,output[:,i])
            else:
                f = ax.plot(scan,output[:,i])
            fluxes += f
        
        def plot_update():
            """update the righthand plot which shows chip output."""
            if scanmode == "shifter":
                scan,output = adaptive_eval(maxV,minVstep)
                if xscale=="quad":
                    scan = np.power(scan,2)
            else:
                scan,output = adaptive_eval_phase(minphase,maxphase,minphasestep)

            for i in range(self.Nchannels):
                fluxes[i].set_xdata(scan)
                fluxes[i].set_ydata(output[:,i])
            
            fig.canvas.draw_idle()

        def XT_update(val):
            nonlocal _compute
            _compute = False
            sliders[0].set_val(self.B[val])
            _compute = True

        names = ["XT",r'$\phi_0$',r'$\theta_0$',r'$\delta \gamma_{1}$',r'$\delta \gamma_{2}$',r'$V_\phi$',r'$V_\theta$']
        if XTmode == "simple":
            sliders = self.make_sliders(fig,maxV,names)
            XTtracker = None
            XTselectorix = None
            XTtxtvals = None
            combs = None
        else:
            sliders = self.make_sliders(fig,maxV,names,ax_left_lim=0.55) # make space
            XTtracker = index_tracker(XT_update)
            XTtracker.idx = (0,1)
            XTtracker.flat_idx = 0
            XTselectorix,XTtxtvals = self.make_XT_selector(fig,XTtracker)
            combs = list(combinations(np.arange(self.Nshifters),2))

        def mzi_update(val):
            """this function controls the logic for when an MZI is selected in the top left GUI."""
            nonlocal _compute
            _compute = False
            dg1,dg2,p0,t0,vphi,vtheta = self[val].dg1,self[val].dg2,self[val].phi0,self[val].theta0,self[val].Vphi,self[val].Vtheta
            sliders[1].set_val(p0)
            sliders[2].set_val(t0)
            sliders[3].set_val(dg1)
            sliders[4].set_val(dg2)
            sliders[5].track.set_alpha(1)
            sliders[6].track.set_alpha(1)
            sliders[5].poly.set_alpha(1)
            sliders[6].poly.set_alpha(1)
            if isinstance(whichmzi,list):
                pairs = list(zip(whichmzi,whichshifter))
                if (val,"phi") in pairs and (val,"theta") in pairs:
                    sliders[5].track.set_alpha(0.2)
                    sliders[5].poly.set_alpha(0.2)
                    sliders[6].track.set_alpha(0.2)
                    sliders[6].poly.set_alpha(0.2)
                elif (val,"phi") in pairs:
                    sliders[5].track.set_alpha(0.2)
                    sliders[5].poly.set_alpha(0.2)
                    sliders[6].set_val(vtheta)
                elif (val,"theta") in pairs:
                    sliders[6].track.set_alpha(0.2)
                    sliders[6].poly.set_alpha(0.2)
                    sliders[5].set_val(vphi)
                else:
                    sliders[5].set_val(vphi)
                    sliders[6].set_val(vtheta)
            else:
                if whichshifter == "phi" and whichmzi == val:
                    sliders[6].set_val(vtheta)
                    sliders[5].track.set_alpha(0.2)
                    sliders[5].poly.set_alpha(0.2)
                elif whichshifter == "theta" and whichmzi == val:
                    sliders[5].set_val(vphi)
                    sliders[6].track.set_alpha(0.2)
                    sliders[6].poly.set_alpha(0.2)
                else:
                    sliders[5].set_val(vphi)
                    sliders[6].set_val(vtheta)     
            _compute = True

        mzitracker = index_tracker(mzi_update)
        slidertracker = index_tracker()

        def slideronpick(event):
            """this function which parameter slider is currently selected so it can be 
            fine tuned with arrow keys."""
            if hasattr(event.artist,"slider_idx"):
                sliders[slidertracker.idx].track.set_edgecolor("None")
                slidertracker.idx = event.artist.slider_idx
                event.artist.set_edgecolor('k')

        def on_press(event):
            cur = sliders[slidertracker.idx]
            step = (cur.valmax - cur.valmin)/1e4 # fine tuning step
            if event.key == "left":
                cur.set_val(cur.val - step)
            elif event.key == "right":
                cur.set_val(cur.val + step)

        fig.canvas.mpl_connect('pick_event', slideronpick)
        fig.canvas.mpl_connect('key_press_event', on_press)
        annotations,rects,channel_lines = self.show(fig,axselector,mzitracker)

        def channellineonpick(event):
            """this function lets you select channels to scan electric field phase over,
            by clicking the channel lines on the top left GUI."""
            if hasattr(event.artist,"channelidx") and scanmode=="field":
                for l in channel_lines:
                    l.set_linewidth(1.5)
                if self._scannedchannel == event.artist.channelidx:
                    self._scannedchannel = None
                else:
                    self._scannedchannel = event.artist.channelidx
                    channel_lines[event.artist.channelidx].set_linewidth(3.)
                xlabel.set_text("phase, channel "+str(self._scannedchannel))
                plot_update()

        fig.canvas.mpl_connect('pick_event', channellineonpick)
        buttonx = 0.9
        reset_button_ax = fig.add_axes([buttonx, 0.25, 0.04, 0.04])
        reset_button = Button(reset_button_ax, 'reset', hovercolor='0.975')

        markbutton_ax = fig.add_axes([buttonx, 0.4, 0.04, 0.04])
        markbutton = Button(markbutton_ax, 'mark', hovercolor='0.975')

        marked_fluxes = []
        def markfunc(mouse_event):
            """this function lets you mark a translucent copy of the current chip output plot
                so you can alter the chip output and compare with the marked copy.
            """
            nonlocal marked_fluxes
            for i,f in enumerate(fluxes):
                xd = f.get_xdata()
                yd = f.get_ydata()
                if yscale == "log":
                    marked_fluxes += ax.semilogy(xd,yd,color=color_cycle[i],alpha=0.5,zorder=-100)
                else:
                    marked_fluxes += ax.plot(xd,yd,color=color_cycle[i],alpha=0.5,zorder=-100)

        markbutton.on_clicked(markfunc)

        clearbutton_ax = fig.add_axes([buttonx, 0.35, 0.04, 0.04])
        clearbutton = Button(clearbutton_ax, 'clear', hovercolor='0.975')

        randbutton_ax = fig.add_axes([buttonx, 0.3, 0.04, 0.04])
        randbutton = Button(randbutton_ax, 'rando\nmize!', hovercolor='0.975')

        zero_button_ax = fig.add_axes([buttonx, 0.2, 0.04, 0.04])
        zero_button = Button(zero_button_ax, 'zero', hovercolor='0.975')

        def zerofunc(mouse_event):
            nonlocal _compute
            _compute = False

            _curidx = mzitracker.idx
            for i in range(self.Nmzis):
                self[i].zero()
                mzitracker.idx = i
                
            if XTmode=="advanced":
                self.B = np.identity(self.Nchannels,dtype=np.float64)
                _curXTidx = XTtracker.flat_idx
                for j in range(int(self.Nshifters*(self.Nshifters-1)/2)):
                    XTtracker.flat_idx = j
                    XTtracker.idx = XTselectorix[j]
                    if sliders[0].val != 0. :
                        sliders[0].set_val(0.)
                XTtracker.flat_idx = _curXTidx
                XTtracker.idx = XTselectorix[_curXTidx]
            else:
                if sliders[0].val != 0. :
                    sliders[0].set_val(0.)
            
            mzitracker.idx = _curidx
            _compute = True            
            plot_update()

        zero_button.on_clicked(zerofunc)

        def random_func(mouse_event):
            """this controls the "randomize!" button, which puts random param values onto the mesh."""
            self.set_random_params()
            if XTmode=="advanced":
                self.set_random_XT()
            else:
                self.XT = rng.uniform(0,0.1)
            # force reset the current sliders
            mzitracker.idx = mzitracker.idx
            
            if XTmode=="advanced":
                comb_cur = XTtracker.idx
                for i,c in enumerate(combs):
                    XTtracker.idx = c
                    XTtxtvals[i].set_text(np.format_float_positional(self.B[c]*100,unique=False,precision=2,trim='k',fractional=False))
                XTtracker.idx = comb_cur
            else:
                sliders[0].set_val(self.XT)

            plot_update()

        randbutton.on_clicked(random_func)

        def clearfunc(mouse_event):
            """this function lets you clear any marked plot lines."""
            nonlocal marked_fluxes
            for i,f in enumerate(marked_fluxes):
                f.remove()
            marked_fluxes.clear()
            fig.canvas.draw_idle()
        
        clearbutton.on_clicked(clearfunc)

        if scanmode=="shifter":
            scanphi_button_ax = fig.add_axes([buttonx, 0.15, 0.04, 0.04])
            scantheta_button_ax = fig.add_axes([buttonx, 0.1, 0.04, 0.04])
            scanphi_button = Button(scanphi_button_ax, 'scan', hovercolor='0.975')
            scanptheta_button = Button(scantheta_button_ax, 'scan', hovercolor='0.975')

        def reset_func(mouse_event):
            """this function resets all mesh parameters to their starting value"""
            nonlocal _compute
            self.load_state(starting_state)
            _compute=False
            _curidx = mzitracker.idx
            for i in range(self.Nmzis-1,-1,-1):
                mzitracker.idx = i

            if XTmode == "simple":
                if sliders[0].val != sliders[0].valinit:
                    sliders[0].reset()
            else:
                _curXTidx = XTtracker.flat_idx
                for j in range(int(self.Nshifters*(self.Nshifters-1)/2)):
                    XTtracker.flat_idx = j
                    XTtracker.idx = XTselectorix[j]
                    if sliders[0].val != sliders[0].valinit:
                        sliders[0].reset()
                XTtracker.flat_idx = _curXTidx
                XTtracker.idx = XTselectorix[_curXTidx]
            
            mzitracker.idx = _curidx
            _compute=True
            plot_update()
            clearfunc(mouse_event)

        def rescanphi_func(mouse_event):
            """this function lets you scan over a phi phase shifter with the "scan" button."""
            nonlocal whichshifter,whichmzi,_compute

            _compute = False
            _cur_idx = mzitracker.idx
            # unscan
            if isinstance(whichmzi,list) and (mzitracker.idx,"phi")  in list(zip(whichmzi,whichshifter)):
                i = list(zip(whichmzi,whichshifter)).index((mzitracker.idx,"phi"))
                self[mzitracker.idx].Vphi = 0.0
                del whichmzi[i]
                del whichshifter[i]
                if len(whichmzi) == 1:
                    whichmzi = whichmzi[0]
                    whichshifter = whichshifter[0]
                # annotations
                annotations[_cur_idx][0].set_text(r"$V_\phi=0.0$")
            # scan
            elif isinstance(whichmzi,list):
                whichmzi.append(_cur_idx)
                whichshifter.append("phi")
                annotations[_cur_idx][0].set_text(r"$V_\phi=$scan")
            else:
                # also scan
                if whichmzi is not None and (whichmzi,whichshifter) != (mzitracker.idx,"phi"):
                    whichmzi = [whichmzi,_cur_idx]
                    whichshifter = [whichshifter,"phi"]
                    annotations[_cur_idx][0].set_text(r"$V_\phi=$scan")
                elif whichmzi is None:
                    whichmzi = _cur_idx
                    whichshifter = "phi"
                    annotations[_cur_idx][0].set_text(r"$V_\phi=$scan")
                else:
                    # unscan
                    self[whichmzi].Vphi = 0.
                    annotations[_cur_idx][0].set_text(r"$V_\phi=0.0$")
                    whichmzi = None
                    whichshifter = None
                    for f in fluxes:
                        f.set_xdata(np.array(f.get_xdata())[[0,-1]])
                        f.set_ydata(np.array(f.get_ydata())[[0,0]])
                    
            mzitracker.idx = _cur_idx
            _compute = True
            plot_update()

        def rescantheta_func(mouse_event):
            """this function lets you scan over a theta phase shifter with the "scan" button."""
            nonlocal whichshifter,whichmzi,_compute

            _compute = False
            _cur_idx = mzitracker.idx

            # unscan
            if isinstance(whichmzi,list) and (mzitracker.idx,"theta")  in list(zip(whichmzi,whichshifter)):
                i = list(zip(whichmzi,whichshifter)).index((mzitracker.idx,"theta"))
                self[mzitracker.idx].Vtheta = 0.0
                del whichmzi[i]
                del whichshifter[i]
                if len(whichmzi) == 1:
                    whichmzi = whichmzi[0]
                    whichshifter = whichshifter[0]
                # annotations
                annotations[_cur_idx][1].set_text(r"$V_\theta=0.0$")
            
            # scan
            elif isinstance(whichmzi,list):
                whichmzi.append(_cur_idx)
                whichshifter.append("theta")
                annotations[_cur_idx][1].set_text(r"$V_\theta=$scan")
            else:
                # also scan
                if whichmzi is not None and (whichmzi,whichshifter) != (mzitracker.idx,"theta"):
                    whichmzi = [whichmzi,_cur_idx]
                    whichshifter = [whichshifter,"theta"]
                    annotations[_cur_idx][1].set_text(r"$V_\theta=$scan")
                elif whichmzi is None:
                    whichmzi = _cur_idx
                    whichshifter = "theta"
                    annotations[_cur_idx][1].set_text(r"$V_\theta=$scan")
                else:
                    # unscan
                    self[whichmzi].Vtheta = 0.
                    annotations[_cur_idx][1].set_text(r"$V_\theta=0.0$")
                    whichmzi = None
                    whichshifter = None
                    for f in fluxes:
                        f.set_xdata(np.array(f.get_xdata())[[0,-1]])
                        f.set_ydata(np.array(f.get_ydata())[[0,0]])
            mzitracker.idx = _cur_idx
            _compute = True
            plot_update()

        reset_button.on_clicked(reset_func)
        if scanmode=="shifter":
            scanphi_button.on_clicked(rescanphi_func)
            scanptheta_button.on_clicked(rescantheta_func)

        def supdate(slider,paramname):
            def _inner_(val):
                i = mzitracker.idx
                if paramname == "XT":
                    if XTmode == "simple":
                        setattr(self,paramname,slider.val)
                    else:
                        _B = np.copy(self.B) # idk if copy is necessary
                        _B[XTtracker.idx] = _B[XTtracker.idx[::-1]] = slider.val
                        self.B = _B
                        XTtxtvals[XTtracker.flat_idx].set_text(np.format_float_positional(slider.val*100,unique=False,precision=2,trim='k',fractional=False))
                else:
                    setattr(self[i],paramname,slider.val)
                if not isinstance(whichmzi,list):
                    if paramname == "Vphi" and not (i == whichmzi and whichshifter == "phi") :
                        annotations[i][0].set_text(r"$V_\phi=$"+str(round(self[i].Vphi,2)))
                    elif paramname == "Vtheta" and not (i == whichmzi and whichshifter == "theta"):
                        annotations[i][1].set_text(r"$V_\theta=$"+str(round(self[i].Vtheta,2)))
                else:
                    if paramname == "Vphi" and not (i,"phi") in zip(whichmzi,whichshifter):
                        annotations[i][0].set_text(r"$V_\phi=$"+str(round(self[i].Vphi,2)))
                    elif paramname == "Vtheta" and not (i,"theta") in zip(whichmzi,whichshifter):
                        annotations[i][1].set_text(r"$V_\theta=$"+str(round(self[i].Vtheta,2)))
                if _compute:
                    plot_update()
            return _inner_

        paramnames = ["XT","phi0","theta0","dg1","dg2","Vphi","Vtheta"]
        for i,s in enumerate(sliders):
            s.on_changed(supdate(s,paramnames[i]))

        mzitracker.idx = 0

        if whichshifter == "phi":
            annotations[whichmzi][0].set_text(r"$V_\phi=$"+"scan")
        elif whichshifter == "theta":
            annotations[whichmzi][1].set_text(r"$V_\theta=$"+"scan")
        
        if XTmode=="simple":
            sliders[0].set_val(self.XT)
        if fullscreen:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()     

        plt.show()

    def set_0V(self):
        vcommand = np.zeros_like(self.V)
        self.V = vcommand

    def multisweep(self,ps_list,scanres,vmin=0.,vmax=6.,_input = "flat"):
        if isinstance(_input,str) and _input == "flat":
            _input = np.ones(self.Nchannels,dtype=np.complex64)/np.sqrt(self.Nchannels)

        dim = len(ps_list)
        
        vsw = np.linspace(vmin,vmax,scanres)

        datasize = tuple(scanres for i in range(dim))

        out = np.zeros((*datasize,self.Nchannels))

        voltgrid = np.array(np.meshgrid(*tuple(vsw for i in range(dim)),indexing='ij'))
        vcommand = np.copy(self.V)
        for ix in np.ndindex(voltgrid[0].shape):
            ix = np.array(ix)
            vcommand[ps_list] = voltgrid[:,*ix]
            self.V = vcommand
            out[*ix,:] = self.measure(_input)
        
        #self.set_0V()
        return out

    def sweep(self,ps,scanres=25,vmin=0.,vmax=6.,launch="flat",vsw = None):

        _input = self.make_launch_field(launch)
        
        if vsw is None:
            vsw = np.linspace(vmin,vmax,scanres)
        out = np.zeros((vsw.shape[0],self.Nchannels))
        vcommand = np.copy(self.V)
        for i,v in enumerate(vsw):
            vcommand[ps] = v
            self.V = vcommand
            out[i] = self.measure(_input)
        return out
    
    def set_bar(self):
        """ load bar state, without touching phi phase shifters. this perfect knowledge of phase bias, but neglects XT """
        phis,thetas = get_mesh_config(np.identity(self.Nchannels)) # these are arranged in diagonal order
        for i,t in enumerate(thetas):
            if self.mziorder == "diagonal":
                m = self[i]
            else:
                m = self[self.diagonal_order[i]]
            Vt = m.theta2Vtheta(t)
            m.Vtheta = Vt
            m.Vphi = 0
        self.A # force computation
        return
    
    def set_cal_bar(self):
        """ load the calibrated bar state """
        for m in self:
            m.set_bar()
    
    def calibrate(self,mzi_ix:int,input_port = -1,monitor_port = 0,scanres=300,vmin=0.,vmax=6.):
        """ sweep Vtheta of the given MZI and extract the bar and cross voltages. 
        default behavior is to inject in the bottom port and monitor the top port.
        """
        vsw = np.linspace(vmin,vmax,scanres)
        _powers = self.sweep(2*mzi_ix+1,scanres,launch=input_port,vsw=vsw)[:,monitor_port]

        self[mzi_ix]._Vthetabar = vsw[np.argmin(_powers)]
        self[mzi_ix]._Vthetacross = vsw[np.argmax(_powers)]
        
    def gridscan(self,mzi_ix:int,scanres=25):
        """ sweep Vtheta and Vphi of a given MZI and return the output
        """
        #monitor_ports = [self[mzi_ix].port0ix,self[mzi_ix].port1ix]
        data = self.multisweep([2*mzi_ix,2*mzi_ix+1],scanres)
        return data

    def load_cal_top_Nrows(self,Nrows,allphirs,allthetas):
        ixs = self.get_top_row_ixs(Nrows)
        print("loading shifter bias data for mzis "+str(ixs)[1:-1])
        theta0s = allthetas[ixs]
        phi0s = solve_for_phi0s(allphirs[ixs])

        for j,i in enumerate(ixs):
            self[i].phi0 = phi0s[j]
            self[i].theta0 = theta0s[j]

    def save_voltages_csv(self,fname,skiprows = [],skipps = []):
        voltage_settings = np.array([self._voltages]).T
        # Open the file in write mode ('w') with newline=''

        for i in range(2*self.Nmzis):
            mziix = int(i/2)

            if i in skipps or self[mziix].port0ix in skiprows:
                voltage_settings[i] = np.nan

        with open(f'voltage_setting_files/{fname}.csv', 'w', newline='') as csvfile:
            # Create a CSV writer object
            writer = csv.writer(csvfile)
            writer.writerows(voltage_settings)

class TriMesh(Mesh):
    
    mziorder = "diagonal"

    def __init__(self,N):
        super().__init__(trimesh_connectivity(N))