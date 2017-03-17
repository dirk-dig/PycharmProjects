import Image
import numpy as np
import struct
import operator
from os import listdir

infile = open('./t10k-images.idx3-ubyte','rb')
header = infile.read( 4 * 4 )
header_up = struct.unpack('>4i', header)   # > : big endian, 4i: 4 x int(32bit)
numOfPixelsIn1Data = header_up[2] * header_up[3]
print '# of image            : %d' % header_up[1]
print 'image width           : %d' % header_up[2]
print 'image height          : %d' % header_up[3]
print '# of pixels in a data : %d' % numOfPixelsIn1Data
for i in range(0,5):
    data = infile.read( numOfPixelsIn1Data )
    fmt  = '%dB' % numOfPixelsIn1Data
    data_up = struct.unpack(fmt, data)
    npData = np.asarray( data_up ).astype('uint8')
    imData = np.reshape(npData, (28,28),order='C')
    im = Image.fromarray( imData )
    im.show()

testFileList = listdir('./t10k-images.idx3-ubyte','rb')
errorCount = 0.0
mTest = len ( infile )
for i in range ( mTest ):
    fileNameStr = testFileList[ i ]
    fileStr = fileNameStr.split ( '.' )[ 0 ]  # take off .txt
    classNumStr = int ( fileStr.split ( '_' )[ 0 ] )
    vectorUnderTest = img2vector ( 'testDigits/%s' % fileNameStr )
    classifierResult = classify0 ( vectorUnderTest, trainingMat, hwLabels, 3 )
    print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
    if (classifierResult != classNumStr): errorCount += 1.0
print "\nthe total number of errors is: %d" % errorCount
print "\nthe total error rate is: %f" % (errorCount / float ( mTest ))