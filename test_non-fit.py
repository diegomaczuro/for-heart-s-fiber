# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:02:35 2016

@author: svyatoslav
"""

import matplotlib.pyplot as plt
import numpy as np
from interfaces.bspline_vector_space import *
from scipy import integrate
def CV(lam,influence_matrix,y_n,num):
    return 0

def main():
    num = 1000
    cv_iter = 1000
    num_knots = 20
    sigma = 1.5
    d = np.random.normal(0,sigma,num)
    d_1 = np.random.normal(0,sigma,num)
    x = np.linspace(0,1,num) 
    basis = np.zeros((num,num_knots))
    basis_1 = np.zeros((num,num_knots))
    basis_deriative = np.zeros((num,num_knots))
    basis_deriative_1 = np.zeros((num,num_knots))
    x_n = np.linspace(0,1,num)    
    y = (6*x-2)*(6*x-2)*np.sin(12*x-4)
    z = np.sin(x*np.pi)*x*x
    z_n = z+d_1
    y_n = (6*x_n-2)*(6*x_n-2)*np.sin(12*x_n-4)+d
    newfun = y_n+z_n
    new_fun = z+y
    # define space and knots
    knots_array = np.linspace(0,1,num_knots-1)
    random_knots = np.random.normal(0,0.1/num_knots,num_knots-1)
    knots_array = knots_array[:]+random_knots[:]
    kn = np.zeros(num_knots+4)
    kn[2:21] = knots_array
    kn[20:] = 1
    kn[2] = 0
    kn1 = np.zeros(num_knots+2)
    kn1[1:20] = knots_array
    kn1[19:] = 1
    kn1[1] = 0
    print kn,kn1 
    knots = kn#r_[2*[0], linspace(0,1,num_knots-1), 3*[1]]
    knots_1 = kn1#r_[1*[0], linspace(0,1,num_knots-1), 2*[1]]
    
    vs = BsplineVectorSpace(2, knots)
    vs_1 = BsplineVectorSpace(1, knots_1)
    # compute basis and deriative
    lam = 0.02
    lam1 = lam #0.01
    D = np.eye(num_knots)
    D[0:2,:] = 0
    for i in xrange(0,num_knots):
        basis[:,i] = vs.basis_der(i,0)(np.linspace(0,1,num))
        basis_deriative[:,i] = vs.basis_der(i,1)(np.linspace(0,1,num))/num
        basis_1[:,i] = vs_1.basis_der(i,0)(np.linspace(0,1,num))
        basis_deriative_1[:,i] = vs_1.basis_der(i,1)(np.linspace(0,1,num))/num
    B = abs(basis_deriative-basis_1)
    S = np.zeros((num_knots,num_knots,num))
    k = np.zeros((num_knots,num_knots,num))
    for i in xrange(num_knots):
        for j in xrange(num_knots):
            S[i,j,:] = B[:,i]*B[:,j]
            k[i,j,:] =basis_deriative_1[:,i] * basis_deriative_1[:,j]
    S_int = np.zeros((num_knots,num_knots))
    k_int = np.zeros((num_knots,num_knots))
    for i in xrange(num_knots):
        for j in xrange(num_knots):
            S_int[i,j] = integrate.trapz(S[i,j,:])
            k_int[i,j] = integrate.trapz(k[i,j,:])
    x = np.linspace(0,1,num)
    pat = basis
    influence_matrix = np.dot(np.dot(pat,(np.linalg.inv(np.dot(np.transpose(pat),pat)+lam1*S_int+lam*k_int))),np.transpose(pat))
    I = np.eye(len(influence_matrix))

    GSV = np.zeros(cv_iter)
    lamb = np.linspace(0,0.1,cv_iter)
    tr = np.zeros(cv_iter)
    fun = np.zeros(cv_iter)
    znam = np.zeros(num_knots)
    for i in xrange(cv_iter):
        influence_matrix = np.dot(np.dot(pat,(np.linalg.inv(np.dot(np.transpose(pat),pat)+lamb[i]*S_int+lamb[i]*k_int))),np.transpose(pat))
        tr = np.trace(influence_matrix)
        fun = np.sum((y_n-np.dot(influence_matrix,y_n))**2)
        GSV[i] =fun/((num-tr)**2)
    print lamb[np.argmin(GSV)]
    lam = lamb[np.argmin(GSV)]
    mod_y = np.dot(np.dot(np.dot(pat,(np.linalg.inv(np.dot(np.transpose(pat),pat)+lam*S_int+lam*k_int))),np.transpose(pat)),y_n)
    mod_y1 = np.dot(np.dot(np.dot(pat,(np.linalg.inv(np.dot(np.transpose(pat),pat)+lam*D))),np.transpose(pat)),y_n)    
    plt.figure(0)
    y = (6*x-2)*(6*x-2)*np.sin(12*x-4)+z
    plt.plot(x, mod_y, color='b', linewidth=2,label = 'model_fit')
    plt.plot(x,y,color = 'r', linewidth=2, label = 'real function')
    plt.legend()
    plt.savefig('compare.eps')
    #plt.plot(x,y_n, 'ro',color = 'g', markersize = 4)
    #plt.plot(x, mod_y1,color='g', linewidth=2)
#    plt.figure(1)
#    plt.plot(x,basis_1)
    plt.figure(2)
    plt.plot(lamb,GSV)
    plt.savefig('GSV.eps')
    plt.figure(3)
    plt.plot(x, mod_y, color='b', linewidth=2)
    plt.plot(x,y,color = 'r', linewidth=2)
    plt.plot(x,y_n, 'ro',color = 'g', markersize = 4, label = 'noise data')
    plt.legend()
    plt.savefig('noise.eps')
    plt.figure(4)
    plt.plot(x, mod_y, color='b', linewidth=2, label = 'model fit')
    plt.plot(x,mod_y1,color = 'r', linewidth=2, label = 'model fit by other method')
    plt.legend()
    plt.savefig('comp.eps')
    #plt.plot(x, mod_y1,color='g', linewidth=2)
if __name__ == '__main__':
    main()