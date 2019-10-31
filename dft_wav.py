import numpy as np
import matplotlib.pyplot as p
from scipy.io import wavfile

def dft(file):
    # discrete fourier transform
    if type(file) == str:
        assert file.endswith(".wav") "file is not a wavefile"
        rate, v = wavfile.read(file)
    else:
        rate = 1
        v = file
    assert len(np.shape(v)) == 1 "input array needs to be one-dimensional"
    N = len(v)
    xt = np.arange(0,N)*2*np.pi*rate/N # frequency vector in hertz
    n_vec = np.arange(N)
    n_mat, k_mat = np.meshgrid(np.arange(N),np.arange(N))
    vt = np.zeros(N,dtype="complex_")
    exp_mat = np.zeros((N,N),dtype="complex_")
    exp_vec = np.exp(-1j*2*np.pi/N*n_vec)
    ind_mat = np.mod(n_mat*k_mat, N, dtype=np.intp)
    exp_mat = exp_vec[ind_mat]
    vt = np.matmul(exp_mat,v)
    return xt, vt

def idft(v, rate):
    # inverse discrete fourier transform
    if type(file) == str:
        assert file.endswith(".wav") "file is not a wavefile"
        rate, v = wavfile.read(file)
    else:
        rate = 1
        v = file
    assert len(np.shape(v)) == 1 "input array needs to be one-dimensional"
    N = len(v)
    xt = np.arange(0,N)*2*np.pi*rate/N # frequency vector in hertz
    n_vec = np.arange(N)
    n_mat, k_mat = np.meshgrid(np.arange(N),np.arange(N))
    vt = np.zeros(N,dtype="complex_")
    exp_mat = np.zeros((N,N),dtype="complex_")
    exp_vec = np.exp(1j*2*np.pi/N*n_vec)
    ind_mat = np.mod(n_mat*k_mat, N, dtype=np.intp)
    exp_mat = exp_vec[ind_mat]
    vt = 1/N*np.matmul(exp_mat,v)
    return vt