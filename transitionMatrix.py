import numpy as np
import pandas as pd

#propability of moving based on current position
transitionMatrix = [[0.5,0.5,0],[0,0.5,0.5],[0,0,1]]
transmatrix2 = [[0.8,0.2],[0.4,0.6]]

#day one we start in position 1
dayOne = [1,0,0]
start = [1,0]


step1 = np.dot(start,transmatrix2)
step2 = np.dot(step1,transmatrix2)
print(step2)