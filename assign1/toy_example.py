import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
X = np.array([
                [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]
])
Y = np.array([[0,0,1,1]]).T

'''
X = np.array([
                [0,1,1],
                [1,0,0],
                [0,1,0],
                [0,0,1],
                [1,1,0],
                [1,0,1]
            ])
Y = np.array([[1,0,0,0,1,1]]).T
'''
X_test = np.array(
                [[1,1,1],
                [0,0,0]
                ])

W = np.random.random((3,1))

loss_batch = np.zeros((100000,1))

num_samples = X[:,0].size
num_test_samples = X_test[:,0].size


for i in xrange(100000):
    dW = np.zeros((3,1))
    for j in xrange(num_samples):

        #forward propagation
        H = X[j].dot(W)
        S = 1/(1+np.exp(-H))
        loss = 0.5*np.square(Y[j] - S)
        loss_batch[i] += loss

        #backward pass
        dLdS = -1* (Y[j]-S)
        dLdH = dLdS*S*(1-S)
        dW += np.array([dLdH*X[j]]).transpose()
    W = W - 1*dW

    if i%1000==0:
        print 'loss after', i, 'iterations:', loss_batch[i]

for j in xrange(num_samples):
    print Y[j], 1/(1+np.exp(-X[j].dot(W)))

print W
#for j in xrange(num_test_samples):
#    print 1/(1+np.exp(-X_test[j].dot(W)))