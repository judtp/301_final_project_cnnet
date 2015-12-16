"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
import cPickle
import gzip
from theano import config
from PIL import Image
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper(samples = 4, trans = 4, angle = 20, scaling = 8):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784,)).astype(config.floatX) for x in tr_d[0]]
    training_results = [vectorized_result(y).astype(config.floatX) for y in tr_d[1]]
    training_data = (training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784,)).astype(config.floatX) for x in va_d[0]]
    validation_results = [vectorized_result(y).astype(config.floatX) for y in va_d[1]]
    validation_data = (validation_inputs, validation_results)
    test_inputs = [np.reshape(x, (784,)).astype(config.floatX) for x in te_d[0]]
    test_results = [vectorized_result(y).astype(config.floatX) for y in te_d[1]]
    test_data = (test_inputs, test_results)
    
    train_size = len(training_inputs)
    entries = range(train_size)
    rng = np.random.RandomState()
    for k in xrange(samples):
        rng.shuffle(entries)
        train_input_shuffle = [training_inputs[k] for k in entries]
        train_result_shuffle = [training_results[k] for k in entries]
        
        set1 = train_input_shuffle[:int(train_size*0.5)]
        set2 = train_input_shuffle[int(train_size*0.5):int(train_size*0.75)]
        set3 = train_input_shuffle[int(train_size*0.75):]

        out1 = shift(set1,trans) 
        out2 = rotate(set2,angle) 
        out3 = scale(set3,scaling)

        set11 = out1[:int(len(out1)*0.5)]
        set12 = out1[int(len(out1)*0.5):int(len(out1)*0.75)]
        set13 = out1[int(len(out1)*0.75):]
        set21 = out2[:int(len(out2)*0.5)]
        set22 = out2[int(len(out2)*0.5):int(len(out2)*0.75)]
        set23 = out2[int(len(out2)*0.75):]
        set31 = out3[:int(len(out3)*0.5)]
        set32 = out3[int(len(out3)*0.5):int(len(out3)*0.75)]
        set33 = out3[int(len(out3)*0.75):]

        out11 = set11
        out12 = scale(set12,scaling) 
        out13 = rotate(set13,angle)
        out21 = set21
        out22 = shift(set22,trans)
        out23 = scale(set23,scaling) 
        out31 = set31
        out32 = shift(set32,trans)
        out33 = rotate(set33,angle)
        
        new_inputs = out11 + out12 + out13 + out21 + out22 + out23 + out31 + out32 + out33
        training_inputs = training_inputs + new_inputs 
        training_results = training_results + train_result_shuffle

    training_data = (training_inputs, training_results)
    return (training_data, validation_data, test_data)
    
def rotate(data, ang):
    train_data_img = [Image.fromarray(255*np.subtract(1,data.reshape((28,28)))).convert('L') for data in data]
    total_data=[]
    for k in xrange(len(train_data_img)):
        imdata = train_data_img[k]
        angle = np.random.randint(-ang,ang)
        imdata = imdata.rotate(angle, resample=Image.BICUBIC)
        imdata = np.asarray(imdata)
        imdata = imdata.reshape(784).astype(config.floatX)
        imdata = np.subtract(255,imdata)
        imdata = imdata/255
        total_data += [imdata]
    return total_data
    
def scale(data, sc_fac):
    train_data_img = [Image.fromarray(255*np.subtract(1,data.reshape((28,28)))).convert('L') for data in data]
    total_data=[]
    for k in xrange(len(train_data_img)):
        imdata = train_data_img[k].convert('RGBA')
        size = imdata.size
        new = Image.new("RGBA", size, (255,)*4)
        scale_factor = np.random.randint(0,sc_fac)
        newsize = (size[0] - scale_factor, size[1] - scale_factor)
        imdata = imdata.resize(newsize, resample=Image.BICUBIC)
        new.paste(imdata,(0,0,newsize[0],newsize[1]),imdata)
        new= new.convert('L')
        data_out = np.asarray(new)
        data_out = data_out.reshape(784).astype(config.floatX)
        data_out = np.subtract(255,data_out)
        data_out = data_out/255
        total_data.append(data_out)
    return total_data
    
def shift(data, shiftfact):
    train_inputs_pad = [(np.pad(np.reshape(train_input, (28,28)), shiftfact, 'constant', constant_values = 0)) for train_input in data]
    train_inputs_temp = []
    for k in xrange(len(train_inputs_pad)):
        rand_pos1 = np.random.randint(0,2*shiftfact)
        rand_pos2 = np.random.randint(0,2*shiftfact)
        train_inputs_temp += [train_inputs_pad[k][rand_pos1:28+rand_pos1, rand_pos2:28+rand_pos2].reshape((784,))]
    return train_inputs_temp
        
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10,))
    e[j] = 1.0
    return e