from numpy import array

import trees
reload(trees)
myDat,labels=trees.createDataSet()
print myDat
print trees.calcShannonEnt(myDat)
print trees.splitDataSet(myDat,0,1)
print trees.splitDataSet(myDat,0,0)
print trees.chooseBestFeatureToSplit(myDat)
myTree = trees.createTree(myDat,labels)
print myTree
