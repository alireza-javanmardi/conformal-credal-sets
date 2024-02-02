import sys
import os
import numpy as np
import src.helper as h 

resolution = sys.argv[1]

simplex = h.simplex_discretizer(step=float(resolution))
os.makedirs(os.path.join("data", "simplex"), exist_ok=True)  
np.save(os.path.join("data", "simplex", resolution), simplex)  