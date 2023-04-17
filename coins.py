import pandas as pd
import numpy as np

#there are 3 coins in a bag with varying probs of fliping heads or 
#tails. Given a result of x heads and x tails, what coin was most likely chosen 

def coins(coinMatrix, resultsDict):
    probs = np.array([])
    for coin in coinMatrix:
        probability = pow(coin[0],resultsDict['tails']) * pow(coin[1], resultsDict['heads'])
        probs = np.append(probs, round(probability,3))

    coinProb = np.array([])
    for prob in probs:
        coinProb = np.append(coinProb, round((prob/probs.sum()),3))
    
    return coinProb


#0 for tails 1 for heads
coinMatrix = [[0.8,0.2],[0.4,0.6],[0.2,0.8]]

resultsDict = {'tails': 8, 'heads': 2}

probs = coins(coinMatrix, resultsDict)
print(probs)


    
#P(HHT) for each coin.

# Coin 1 P1=0.53
# Coin 2 P2=0.3⋅0.72
# Coin 3 P2=0.6⋅0.42
# P(HTT)=1/3*P1+1/3*P2+1/3*P3
# P(coin1|HHT)=P1/(P1+P2+P3)




