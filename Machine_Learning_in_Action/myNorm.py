from numpy import array
import kNN
reload(kNN)
normMat, ranges, minVals = kNN.autoNorm('datingDataMat')
print normMat
