"""Fixed array geometry, copied from the archived experiment configuration.
No human survey coordinates or local capture paths are included here.
"""
import numpy as np
C=299792458.
F=5.23e9+np.arange(-58,59,4)*312500.
MAP={'G1': [1, 2, 3], 'G2': [4, 5, 6], 'G3': [7, 8, 9], 'G4': [10, 11, 12], 'C1': [1, 8, 6], 'C2': [10, 8, 6]}
STATES=list(MAP)
Y=np.array([0.154, 0.126, 0.098, 0.07, 0.042, 0.014, -0.014, -0.042, -0.07, -0.098, -0.126, -0.154],dtype=float)
