# coding: UTF-8
#コメント

import numpy as np
randMat = np.mat(np.random.rand(4,4))
# print randMat.I
IrandMat = randMat.I
myEye = randMat  * IrandMat
print myEye - np.eye(4)
