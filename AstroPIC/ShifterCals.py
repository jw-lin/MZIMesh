""" best-fit values for phase shifter biases of AstroPIC mesh. """

import numpy as np
from numpy import pi

# when fitting 2D phi-theta shifter sweeps, there is freedom
# freedom to translate any best-fit value by 2 pi, or add +/- pi to BOTH
# also need to shift phi values by pi to account for sign flips of bar states

#region cal arrays

allthetas = [
    0,              # 0
    0,              # 1
    0,              # 2
    -0.25745157,    # 3
    -2.09848235+pi, # 4
    0.80567973,     # 5
    1.14362729-pi,  # 6
    0.94345192,     # 7
    0,              # 8
    0,              # 9
    0,              # 10
    -0.20633229,    # 11
    1.64223197-pi,  # 12
    0.05281516,     # 13
    2.14704086-pi,  # 14
    0,              # 15
    0,              # 16
    0,              # 17
    0.89215347-pi,  # 18
    2.63746349-pi,  # 19
    -0.25282845,    # 20
    0,              # 21
    0,              # 22
    0,              # 23
    -0.11247303,    # 24
    -0.12166616,    # 25
    0,              # 26
    0,              # 27
    0,              # 28
    -0.69808917,    # 29
    0,              # 30
    0,              # 31
    0,              # 32
    0,              # 33
    0,              # 34
    0,              # 35
]

# the 'r' signifies that these are relative values;
# they need to be passed into PhiBiasCalculator 
# to solve for the real phi shifter biases.

allphirs = [
    0,              # 0
    0,              # 1
    0,              # 2
    -3.02261929+pi, # 3
    -2.74466721,    # 4
    3.00983983-pi,  # 5
    -0.63902337,    # 6
    2.23840237-pi,  # 7
    0,              # 8
    0,              # 9
    0,              # 10
    0.90192421-pi,  # 11
    -1.27556968,    # 12
    -0.43466206+pi, # 13
    2.06896403,     # 14
    0,              # 15
    0,              # 16
    0,              # 17
    -1.54584196,    # 18
    1.27502917,     # 19
    -2.05410981+pi, # 20
    0,              # 21
    0,              # 22
    0,              # 23
    -1.90215843+pi, # 24
    -1.27412877+pi, # 25
    0,              # 26
    0,              # 27
    0,              # 28
    -1.75833790+pi, # 29
    0,              # 30
    0,              # 31
    0,              # 32
    0,              # 33
    0,              # 34
    0,              # 35
]

allphirs = np.array(allphirs)
allthetas = np.array(allthetas)

#endregion