from numpy import array
import kNN
import matplotlib
import matplotlib.pyplot as plt
reload(kNN)
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2],15.0*array(datingLabels).astype(float), 15.0*array(datingLabels).astype(float))
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
plt.show()
