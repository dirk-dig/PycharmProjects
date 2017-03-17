import numpy as np
import mnist1 as mnist

##### training data
#
mn = mnist.MNIST ( 'L' )
labL = mn.getLabel ( )
nL = labL.shape[ 0 ]
xL = mn.getImage ( ).reshape(nL, -1)
print '# xL: ', xL.shape

##### test data
#
mn = mnist.MNIST ( 'T' )
labT = mn.getLabel ( )
nT = labT.shape[ 0 ]
xT = mn.getImage ( ).reshape ( (nT, -1) )
print '# xT: ', xT.shape

##### nearest neighbor classification
#
xLsq = np.sum ( xL ** 2, axis=1 )
out = np.empty ( nT, dtype=int )
for i in range ( nT ):
    if i % 1000 == 0:
        print i
    d = -2 * np.dot ( xL, xT[ i, : ] ) + xLsq
    out[ i ] = labL[ np.argmin ( d ) ]

er = np.sum ( out != labT ) / float ( nT )

print '# test error rate = ', er * 100, '%'