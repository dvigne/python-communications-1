import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#set q and p
q = 0.8
p = 0.3
N = 10000 #number of bits
#generate random bits
txBits = np.zeros((N,),dtype='bool')
txBits[np.random.rand(N)>q] = True
txBits
array([False, False, False, ..., False, False, False])
#transmit the bits trhough the channel
def bsc(txBits,p): #simulates a binary symmetric channel with transition probability p
    flips = np.zeros((N,),dtype='bool') #there are no flips at this point
    x = np.random.rand(N)
    flips[x<p] = True
    rxBits = np.logical_xor(txBits,flips)
    return rxBits

rxBits = bsc(txBits,p)
rxBits
array([ True, False,  True, ...,  True, False, False])
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

MLEstimates = MLDetector(rxBits)
MAPEstimates = MAPDetector(rxBits,q,p)
MAPEstimates
array([False, False, False, ..., False, False, False])
#calculate probability of error
PEML = np.sum(np.logical_xor(txBits,MLEstimates))/N
PEMAP = np.sum(np.logical_xor(txBits,MAPEstimates))/N
print('Probability of error for ML estimator is %f'%PEML)
print('Probability of error for MAP estimator is %f'%PEMAP)
