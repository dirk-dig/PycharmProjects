import struct
import numpy as np


class MNIST:
    def __init__(self, LT):

        if LT == 'L':
            self.fnLabel = 'train-labels.idx1-ubyte'
            self.fnImage = 'train-images.idx3-ubyte'
        else:
            self.fnLabel = 't10k-labels.idx1-ubyte'
            self.fnImage = 't10k-images.idx3-ubyte'

    def getLabel(self):

        return readLabel ( self.fnLabel )

    def getImage(self):
        return readImage(self.fnImage)


##### reading the label file
#
def readLabel(fnLabel):
    f = open ( fnLabel, 'rb')

    ### header (two 4B integers, magic number(2049) & number of items)
    #
    header = f.read ( 8 )
    mn, num = struct.unpack ( '>2i', header )  # MSB first (bigendian)
    assert mn == 2049
    # print mn, num

    ### labels (unsigned byte)
    #
    label = np.array ( struct.unpack ( '>%dB' % num, f.read ( ) ), dtype=int )

    f.close ( )

    return label


##### reading the image file
#
def readImage(fnImage):
    f = open ( fnImage, 'rb' )

    ### header (four 4B integers, magic number(2051), #images, #rows, and #cols
    #
    header = f.read ( 16 )
    mn, num, nrow, ncol = struct.unpack ( '>4i', header )  # MSB first (bigendian)
    assert mn == 2051
    # print mn, num, nrow, ncol

    ### pixels (unsigned byte)
    #
    pixel = np.empty ( (num, nrow, ncol) )
    npixel = nrow * ncol
    for i in range ( num ):
        buf=struct.unpack('>%dB'%npixel,f.read(npixel))
        pixel[ i, :, : ] = np.asarray ( buf ).reshape ( (nrow, ncol) )
        #pixel = np.memmap(filename, ">B", mode="r", offset=16, shape=(num, nrow, ncol))
    f.close ( )

    return pixel


if __name__ == '__main__':
    print '# MNIST training data'
    mnist = MNIST ( 'L' )
    lab = mnist.getLabel ( )
    dat = mnist.getImage ( )
    print lab.shape, dat.shape

    print '# MNIST test data'
    mnist = MNIST ( 'T' )
    lab = mnist.getLabel ( )
    dat = mnist.getImage ( )
    print lab.shape, dat.shape