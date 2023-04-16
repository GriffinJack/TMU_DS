import numpy as np
import pandas as pd

s=1
t=1
u=1
v=0
z=0

S = [0.9, 0.1]
T = [0.95, 0.05]
U = [[[.9,.1],[.75,.25]],[[.4,.6],[.25,.75]]]
V = [[.8,.2],[.2,.8]]
Z = [[.85,.15],[.05,.95]]

jointProb = S[s] * T[t] * U[s][t][u] * V[u][v] * Z[u][z]

print(jointProb)

