# coding: UTF-8
#コメント
import numpy as np
x = np.array ([[1,2,3],[4,5,6]], np.int32)
print x[0,0]
a = np.array([1.534,2.867,3.23,4.97,5.24,6.8788,7.234,8.9,9.342,10.000])
print a[0:8:2]
print np.round(a)

y = np.sin (np.radians(90))
print y
#import matplotlib.pylab as plt
#x = np.linspace(-np.pi, np.pi, 201)
#plt.plot(x, np.sin(x))
#plt.xlabel('Angle [rad]')
#plt.ylabel('sin(x)')
#plt.axis('tight')
#plt.show()

b = np.exp(0)
print b
c = np.log(1)
print c

