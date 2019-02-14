import numpy as np
import matplotlib.pyplot as plt
import time
#set q and p
q = np.arange(0,1, 0.01)
p = 0.35
p2 = 0.4
N = 100000 #number of bits

#transmit the bits trhough the channel
def bsc(txBits,p): #simulates a binary symmetric channel with transition probability p
    flips = np.zeros((N,),dtype='bool') #there are no flips at this point
    x = np.random.rand(N)
    flips[x<p] = True
    rxBits = np.logical_xor(txBits,flips)
    return rxBits

def bac(txBits,p1, p2): #simulates a binary symmetric channel with transition probability p
    flips = np.zeros((N,),dtype='bool') #there are no flips at this point
    x = np.random.rand(N)
    for bit in range(0, len(txBits)):
        if(txBits[bit] == 0):
            flips[x < p1] = True
        else:
            flips[x < p2] = True

    rxBits = np.logical_xor(txBits,flips)
    return rxBits

#Perform both ML and MAP detection at the reciever
def MLDetector(rxBits): #simulates an ML Detector, assumes p < 0.5
    return rxBits

def MAPDetector(rxBits,q,p): #simulates a MAP detector
    if q < p:
        return np.ones((N,),dtype='bool')
    if q > p and q <= 1-p:
        return rxBits
    if q > 1-p:
        return np.zeros((N,),dtype='bool')

def MAPDetectorB(rxBits,q,p1,p2): #simulates a MAP detector
    for x in range(0, len(rxBits)):
        if(rxBits[x] == 0):
            if q < p1:
                rxBits[x] = 1
            if q > p1 and q <= 1-p1:
                pass
            if q > 1-p1:
                rxBits[x] = 0
        else:
            if q < p2:
                rxBits[x] = 1
            if q > p2 and q <= 1-p2:
                pass
            if q > 1-p2:
                rxBits[x] = 0
    return rxBits


# Channel A

PEMLA = []
PEMLB = []
mapA = []
mapB = []

for x in np.nditer(q):
    # Setup
    txBits = np.zeros((N,),dtype='bool')
    txBits[np.random.rand(N)>x] = True

    rxBitsA = bsc(txBits,p)
    rxBitsB = bac(txBits,p, p2)

    # Channel A
    MLEstimates = MLDetector(rxBitsA)
    PEML = np.sum(np.logical_xor(txBits,MLEstimates))/N
    print('\nChannel A probability of error for ML estimator is %f'%PEML)
    PEMLA.append(PEML)

    # Channel B
    MLEstimates = MLDetector(rxBitsB)
    PEML = np.sum(np.logical_xor(txBits,MLEstimates))/N
    print('Channel B probability of error for ML estimator is %f'%PEML)
    PEMLB.append(PEML)

    # Channel A
    MAPEstimates = MAPDetector(rxBitsA,x,p)
    PEMAP = np.sum(np.logical_xor(txBits,MAPEstimates))/N
    print('\nChannel A probability of error for MAP estimator is %f'%PEMAP)
    mapA.append(PEMAP)

    # Channel B
    MAPEstimates = MAPDetectorB(rxBitsB,x,p, p2)
    PEMAP = np.sum(np.logical_xor(txBits,MAPEstimates))/N
    print('Channel B probability of error for MAP estimator is %f'%PEMAP)
    mapB.append(PEMAP)


fig1, ax = plt.subplots()
fig2, bx = plt.subplots()
ax.plot(q, PEMLA, label="PEMLA")
ax.plot(q, mapA, label="Map A")
ax.legend()
bx.plot(q, PEMLB, label="PEML B")
bx.plot(q, mapB, label="Map B")
bx.legend()
ax.set(xlabel='Q', ylabel='Probability of Error %',
       title='Channel A')
bx.set(xlabel='Q', ylabel='Probability of Error %',
       title='Channel B')

plt.show()
