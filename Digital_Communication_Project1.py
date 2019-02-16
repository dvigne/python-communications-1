import numpy as np
import matplotlib.pyplot as plt
import time
#set q and p
q = np.arange(0,1.01, 0.01)
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
    rxBits = np.zeros_like(txBits)
    flips = np.zeros_like(rxBits[txBits == 0], dtype='bool')
    x = np.random.rand(flips.shape[0])
    flips[x < p1] = True
    rxBits[txBits == 0] = np.logical_xor(rxBits[txBits == 0], flips)

    flips = np.zeros_like(rxBits[txBits > 0], dtype='bool')
    x = np.random.rand(flips.shape[0])
    flips[x < p2] = True
    rxBits[txBits > 0] = np.logical_xor(rxBits[txBits > 0], flips)
    rxBits = np.logical_xor(txBits, rxBits)

    return rxBits

#Perform both ML and MAP detection at the reciever
def MLDetector(rxBits): #simulates an ML Detector, assumes p < 0.5
    return rxBits

def MAPDetector(rxBits,q,p): #simulates a MAP detector
    if q <= p:
        return np.ones_like(rxBits)
    if q > p and q <= 1-p:
        return rxBits
    if q > 1-p:
        return np.zeros_like(rxBits)


# Channel A

PEMLA = []
PEMLB = []
mapA = []
mapB = []
start = time.time()

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
    MAPEstimates = np.zeros_like(rxBitsB, dtype='bool')
    MAPEstimates[rxBitsB == 0] = MAPDetector(rxBitsB[rxBitsB == 0],x,p)
    MAPEstimates[rxBitsB > 0] = MAPDetector(rxBitsB[rxBitsB > 0],x,p2)
    PEMAP = np.sum(np.logical_xor(txBits,MAPEstimates))/N
    print('Channel B probability of error for MAP estimator is %f'%PEMAP)
    mapB.append(PEMAP)

print("="*50, "Finished Execution in %.2f seconds" % (time.time()-start), sep="\n", end='\n\n')

# fig1, ax = plt.subplots()
fig2, bx = plt.subplots()
# ax.plot(q, PEMLA, label="PEMLA")
# ax.plot(q, mapA, label="Map A")
# ax.legend()
bx.plot(q, PEMLB, label="PEML B")
bx.plot(q, mapB, label="Map B")
bx.legend()
# ax.set(xlabel='Q', ylabel='Probability of Error %', title='Channel A With %d Bits' % N)
bx.set(xlabel='Q', ylabel='Probability of Error %', title='Channel B With %d Bits' % N)

plt.show()
