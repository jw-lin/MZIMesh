""" 9x9 triangular mesh model which loads in phase shifter biases 
    from ShifterCals.py. 
"""

from MZIMesh.UnitaryMatrices import circ,DFT
from MZIMesh.MZI import TriMesh
from ShifterCals import allphirs,allthetas

########################
### matrix to upload ###
########################

NU = 6  # dim of desired optical transformation

#U = np.identity(NU)
#U = DFT(NU)
U = circ(NU)

#####################
### make 9x9 mesh ###
#####################

N = 9
mmesh = TriMesh(N)

##############################
### load shifter bias data ###
##############################

Nrows = NU - 1
mmesh.load_cal_top_Nrows(Nrows,allphirs,allthetas)

###############################
### compute & load voltages ###
###############################

if U is None:
    pass
else:
    mmesh.load_matrix(U)

#############################
### mesh simulator widget ###
#############################

mmesh.live_plot(yscale="linear",scanmode="shifter",launch="flat",xscale="quad")

############################################
### save computed voltage setting to CSV ###
############################################

#matrix_name = ""
#mmesh.save_voltages_csv(matrix_name,skiprows = np.arange(Nrows,8,dtype=int))