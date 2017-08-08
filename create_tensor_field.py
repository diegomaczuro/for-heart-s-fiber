# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:32:02 2016

@author: svyatoslav
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def main():
    tensor_len = 10000
    sigma = 0.1
    x = np.linspace(0,1,tensor_len)
    normal_noise = np.random.normal(0,sigma,tensor_len)
    another_normal_noise = np.random.normal(0,sigma,tensor_len)
    rician_noise = np.sqrt(normal_noise**2+another_normal_noise**2)
    tensor = np.zeros((tensor_len,3))
    tensor[:tensor_len//2,:] = np.array([1,0,1])
    tensor[tensor_len//2:,:] = np.array([.5,.75,.225])
    for i in xrange(3):
        tensor[:,i] = tensor[:,i]+normal_noise[:]
   
    plt.figure(0)
    plt.plot(x,tensor[:,0])
#   # plt.plot(x,rician_noise)
#    n, bins, patches = plt.hist(rician_noise, 100, normed=1, facecolor='green', alpha=0.75)
#    plt.savefig('rician_sigma_0_15.eps')
    np.savetxt('tensor.dat',tensor)
#    plt.show()
if __name__ == "__main__":
    main()