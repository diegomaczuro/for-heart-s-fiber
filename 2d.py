# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 01:07:06 2016

@author: svyatoslav
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from interfaces.bspline_vector_space import *
from scipy import integrate

def tensor():
    """this function open file, and compute log(tensor) for future computation
    return 2d array contains 4 element [a,b]"""
    tensor = np.loadtxt('tensor.dat')
    #tensor = np.reshape(tensor,len(tensor)/4)
    tensor_new = np.zeros((len(tensor),2,2))
    tensor_new[:,0,0] = tensor[:,0]
    tensor_new[:,0,1] = tensor[:,1]
    tensor_new[:,1,0] = tensor[:,1]
    tensor_new[:,1,1] = tensor[:,2]
    #  eigen-value decomposition
    S = np.zeros((len(tensor_new),2,2))
    for i in xrange(len(S)):
        R,D = np.linalg.eig(tensor_new[i])
        D_log = np.log(D)
        S[i] = np.dot(R.T,np.dot(D_log,R))
    S = np.reshape(S,(len(S),4))
    return S

def CV(lam,influence_matrix,y_n,num):
    return 0
def method(knots,y_n):
    cv_iter = 1000
    num = len(y_n)
    num_knots = len(knots)
    linear_knots = knots[1:num_knots-1]
    num_knots = num_knots-4
    basis = np.zeros((num,num_knots))
    basis_1 = np.zeros((num,num_knots))
    basis_deriative = np.zeros((num,num_knots))
    basis_deriative_1 = np.zeros((num,num_knots))
    vs = BsplineVectorSpace(2, knots)
    vs_1 = BsplineVectorSpace(1, linear_knots)
    for i in xrange(0,num_knots):
        basis[:,i] = vs.basis_der(i,0)(np.linspace(0,1,num))
        basis_deriative[:,i] = vs.basis_der(i,1)(np.linspace(0,1,num))/num
        basis_1[:,i] = vs_1.basis_der(i,0)(np.linspace(0,1,num))
        basis_deriative_1[:,i] = vs_1.basis_der(i,1)(np.linspace(0,1,num))/num
    B = abs(basis_deriative-basis_1)
    S = np.zeros((num_knots,num_knots,num))
    ss = np.eye(num_knots)
    ss[0:1,:] = 0
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
    pat = basis
    GSV = np.zeros(cv_iter)
    lamb = np.linspace(0,0.1,cv_iter)
    tr = np.zeros(cv_iter)
    fun = np.zeros(cv_iter)
    for i in xrange(cv_iter):
        influence_matrix = np.dot(np.dot(pat,(np.linalg.inv(np.dot(np.transpose(pat),pat)+lamb[i]*ss))),np.transpose(pat))
        tr = np.trace(influence_matrix)
        fun = np.sum((y_n-np.dot(influence_matrix,y_n))**2)
        GSV[i] =fun/((num-tr)**2)
    print lamb[np.argmin(GSV)]
    lam = lamb[np.argmin(GSV)]
    model_fit = np.dot(np.dot(np.dot(pat,(np.linalg.inv(np.dot(np.transpose(pat),pat)+lam*ss))),np.transpose(pat)),y_n)
    return model_fit
def test_function(num):
    x= np.linspace(-5,10,num)
    y = np.linspace(0,15,num)
    a = 1.0
    b = 5.1/(4*np.pi*np.pi)
    c = 5/np.pi
    r = 6
    s = 10
    t = 1/(8*np.pi)
    z = np.zeros((num,num))
    for i in xrange(num):
        for j in xrange(num):
            z[i,j] = a*(y[i]-b*x[j]*x[j]+c*x[j]-r)**2+s*(1-t)*np.cos(x[j])+s
    return np.reshape(z,(num*num)) 
def method_2d(knots,y_n,num):
    """ knots - 2d array [[xi,yi]]
        y_n - 1d array
        lam - vector contained 2 element"""
    cv_iter = 10 # number of iteration for cross-validation    
    GSV = np.zeros((cv_iter,cv_iter))
#    tr = np.zeros((cv_iter,cv_iter))
#    fun =np.zeros((cv_iter,cv_iter))
    lam_x = np.linspace(0,0.2,cv_iter)
    lam_y = np.linspace(0,0.2,cv_iter)
    num_knots = len(knots)
    linear_knots = knots[1:num_knots-1]
    num_knots = num_knots-4
    znam = np.zeros((num_knots))
    basis = np.zeros((num,num_knots))
    basis_1 = np.zeros((num,num_knots))
    basis_deriative = np.zeros((num,num_knots))
    basis_deriative_1 = np.zeros((num,num_knots))
    S = np.zeros((num_knots,num_knots,num))
    vs = BsplineVectorSpace(2, knots)
    vs_1 = BsplineVectorSpace(1, linear_knots)
    I_i = np.eye(num_knots)
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
    basis_product = np.kron(basis,basis)
    S_x = np.kron(S_int,I_i)
    S_y = np.kron(I_i,S_int)
    K_x = np.kron(k_int,I_i)
    K_y = np.kron(I_i,k_int)
    for i in xrange(cv_iter):
        for j in xrange(cv_iter):
            influence_matrix = np.dot(np.dot(basis_product,(np.linalg.inv(np.dot(np.transpose(
                basis_product),basis_product)+lam_x[i]*S_x+lam_y[j]*S_y+lam_x[i]* K_x+lam_y[j]*K_y))),np.transpose(basis_product))
            for k in xrange(num_knots):
                znam[k] =(1-influence_matrix[k,k])**2
            tr = np.sum(znam)
            fun = np.sum((y_n-np.dot(influence_matrix,y_n))**2)
            GSV[i,j] =fun/(num*tr)
            print i,j
    a,b = np.unravel_index(GSV.argmin(), GSV.shape)
#    a = np.argmin(np.argmin(GSV,axis = 0))
#    b = np.argmin(np.argmin(GSV,axis = 1))
    lamb_x = lam_x[a]
    lamb_y = lam_y[b]
    print lamb_x,lamb_y
    model_fit = np.dot(np.dot(np.dot(basis_product,(np.linalg.inv(np.dot(np.transpose(
                basis_product),basis_product)+lamb_x*S_x+lamb_y*S_y+lamb_x* K_x+lamb_y*K_y))),np.transpose(basis_product)),y_n)
    return model_fit,GSV
def main():
    S = tensor()
#    num = 100
    num = np.sqrt(len(S))
    num_knots = 15
    #sigma = 15.0
    #d = np.random.normal(0,sigma,num*num)
    x = np.linspace(0,1,num)    
    #y = (6*x-2)*(6*x-2)*np.sin(12*x-4)
    #y_n = (6*x_n-2)*(6*x_n-2)*np.sin(12*x_n-4)+d
    #lam = 0.03
    # define space and knots
    knots_array = np.linspace(0,1,num_knots-1)
    random_knots = np.random.normal(0,0.1/num_knots,num_knots-1)
    knots_array = knots_array[:]+random_knots[:]
    kn = np.zeros(num_knots+4)
    kn[2:num_knots+1] = knots_array
    kn[num_knots:] = 1
    kn[2] = 0
    knots = kn#r_[2*[0], linspace(0,1,num_knots-1), 3*[1]]
    #test = test_function(num)
    #test_n = test+d
    #print np.shape(test_n)
    S_fit = S
    S_fit[:,0], GSV = method_2d(knots,S[:,0],num)
    S_fit[:,1], GSV = method_2d(knots,S[:,1],num)
    S_fit[:,3], GSV = method_2d(knots,S[:,3],num)
    S_fit[:,2] = S_fit[:,1]
   #mean_err = (np.sum((test[:]-predict[:])**2))/(num*num)
    #print mean_err
    Z = np.reshape(S_fit[:,0],(num,num))
    #te = np.reshape(test,(num,num))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection ='3d')
    X,Y = np.meshgrid(x,x)
    ax.plot_surface(X,Y,Z, rstride =4, cstride =4, color ='b')
   # ax.plot_surface(X,Y,te, rstride =4, cstride =4, color ='r')
def main_2():
    num = 1000
    num_knots = 20
    sigma = 1.5
    sigma_1 = 1.0
    d = np.random.normal(0,sigma,num)
    d_1 = np.random.normal(0,sigma_1,num)
    x = np.linspace(0,1,num)    
    y = (6*x-2)*(6*x-2)*np.sin(12*x-4)
    z = np.sin(x*np.pi)*18*x*x
    z_n = z+d_1
    an = np.sqrt(d*d+d_1*d_1)
    y_n = (6*x-2)*(6*x-2)*np.sin(12*x-4)+an-sigma
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
    knots = kn#r_[2*[0], linspace(0,1,num_knots-1), 3*[1]]
    y_mod = method(knots,y_n)
    z_mod = method(knots,z_n)
    newfun_mod = method(knots,newfun)
    test = z_mod+y_mod
    plt.figure(0)
    plt.plot(x, z_mod, color='b', linewidth=2, label = 'model fit')
    plt.plot(x,z,color = 'r', linewidth=2)
    plt.legend()
    plt.figure(1)
    plt.plot(x, y_mod, color='b', linewidth=2,  label = 'model fit')
    plt.plot(x,y,color = 'r', linewidth=2)
    plt.legend()
    plt.figure(2)
    plt.plot(x, newfun_mod, color='b', linewidth=2,  label = 'model fit')
    plt.plot(x,new_fun,color = 'r', linewidth=2)
    plt.legend()
    plt.savefig('./gr/function.eps')
    plt.figure(3)
    plt.plot(x, newfun_mod, color='b', linewidth=2,  label = 'model fit')
    plt.plot(x,test,color = 'r', linewidth=2, label = 'function sum')
    plt.legend()
    plt.savefig('./gr/additive_fit.eps')
    plt.figure(4)
    plt.plot(x,newfun, 'ro',color = 'g', markersize = 4, label = 'noise data')
    plt.legend()
    plt.savefig('noise_fun.eps')
if __name__ == '__main__':
    main_2()