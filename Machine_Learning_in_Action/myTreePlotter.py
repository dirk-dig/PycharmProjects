import treePlotter
import trees
reload(treePlotter)
reload(trees)
treePlotter.retrieveTree (1)
myDat,labels=trees.createDataSet()
myTree = treePlotter.retrieveTree (0)
myTree['no surfacing'][3]='maybe'
trees.storeTree(myTree,'classifierStorage.txt')
print trees.grabTree('classifierStorage.txt')
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)
treePlotter.createPlot(lensesTree)
#treePlotter.createPlot(myTree)
