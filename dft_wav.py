import numpy as np
import matplotlib.pyplot as p
from scipy.io import wavfile

def dft(file, rate=1):
    # discrete fourier transform
    # 'file' input takes filename of a .wav file or 1-D input vector
    if type(file) == str: # tests the input format
        assert file.endswith(".wav"), "file is not a wavefile"
        rate, v = wavfile.read(file)
    else:
        v = file
    assert len(np.shape(v)) == 1, "input array needs to be one-dimensional"
    N = len(v) # get length of vector
    xt = np.arange(0,N)*2*np.pi*rate/N # frequency vector in hertz
    vt = np.zeros(N,dtype="complex_") # initialize array
    # we only want to calculate the exponent N-times, evenly spaced along the unit circle
    # for this a vector of 0 to N-1 integers is generated
    n_vec = np.arange(N)
    # then the exponential is evaluated at those points
    exp_vec = np.exp(-1j*2*np.pi/N*n_vec)
    for k in np.arange(N):
        # now the exp_vec needs to be called at the right indices, this is done at n*k mod N
        vt[k] = np.dot(exp_vec[np.mod(n_vec*k,N, dtype=np.intp)],v)
    return xt, vt

def idft(v):
    # inverse discrete fourier transform
    # 'v' is the input vector on which the inverse fourier transform is performed
    # 'v' needs to be a 1-D array
    # for comments see the dft function, which is very similar
    assert len(np.shape(v)) == 1, "input array needs to be one-dimensional"
    N = len(v)
    vt = np.zeros(N,dtype="complex_")
    n_vec = np.arange(N)
    exp_vec = np.exp(1j*2*np.pi/N*n_vec) # notice the missing (-) in the argument
    # this makes this the inverse function
    for k in np.arange(N):
        # normalization by 1/N cannot be missing:
        vt[k] = 1/N*np.dot(exp_vec[np.mod(n_vec*k,N, dtype=np.intp)],v)
    return vt