# -*- coding: utf-8 -*-
"""
Created on Sun May 28 23:08:41 2017

@author: svyatoslav
"""

import petsc4py
import sys
import math
#petsc4py.init(sys.argv)
from petsc4py import PETSc
from matplotlib import pylab
import numpy as np
from interfaces.bspline_vector_space import *
from scipy import sparse
from mpi4py import MPI
import scipy
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
#from matplotlib.collections import EllipseCollection
import line_profiler

num_iter = 2
n = 25
m = 5
slic = 6
comm = MPI.COMM_WORLD
filename = "./DTI080803/"

def real_data(filename):
    D,R = read_tensor(filename)
    #print D[100,:,:]
   # D,R = reduction(D,R)
    log_matrix = in_log_euclidian(D,R)
    return log_matrix
def test_data():
    D,R = read_test_tensor()
    log_matrix = in_log_euclidian_2d(D,R)
    return log_matrix
def read_test_tensor():
    T = np.load('tensor.npy')
    T = np.reshape(T,(n*n,2,2))
    DD = np.zeros((n*n,2))
    D = np.zeros((n*n,2,2))
    R = np.zeros((n*n,2,2))
    for i in xrange(n*n):
        DD[i,:],R[i,:,:] = np.linalg.eig(T[i,:,:])
        D[i,:,:] = np.diag(DD[i,:])
    return D,R
def reduction(D,R,k = 4):
    new_D = np.zeros((len(D)/(k*k),3,3))
    new_R = np.zeros((len(D)/(k*k),3,3))
    for i in xrange(len(new_D)):
        new_D[i,:,:] = D[i*k,:,:]
        new_R[i,:,:] = R[i*k,:,:]
    return new_D,new_R

def read_tensor(filename):
    koef = 100000
    e1 = scipy.io.loadmat(filename+'e1.mat')['e1']
    e2 = scipy.io.loadmat(filename+'e2.mat')['e2']
    e3 = scipy.io.loadmat(filename+'e3.mat')['e3']
    v11 = scipy.io.loadmat(filename+'v11.mat')['v11']
    v12 = scipy.io.loadmat(filename+'v12.mat')['v12']
    v13 = scipy.io.loadmat(filename+'v13.mat')['v13']
    v21 = scipy.io.loadmat(filename+'v21.mat')['v21']
    v22 = scipy.io.loadmat(filename+'v22.mat')['v22']
    v23 = scipy.io.loadmat(filename+'v23.mat')['v23']
    v31 = scipy.io.loadmat(filename+'v31.mat')['v31']
    v32 = scipy.io.loadmat(filename+'v32.mat')['v32']
    v33 = scipy.io.loadmat(filename+'v33.mat')['v33']
    D = np.zeros((n*n,3,3))
    R = np.zeros((n*n,3,3))
    D[:,0,0] = np.reshape(e1[0:n,0:n,slic],n*n)*koef
    D[:,1,1] = np.reshape(e2[0:n,0:n,slic],n*n)*koef
    D[:,2,2] = np.reshape(e3[0:n,0:n,slic],n*n)*koef
    R[:,0,0] = np.reshape(v11[0:n,0:n,slic],n*n)
    R[:,0,1] = np.reshape(v12[0:n,0:n,slic],n*n)
    R[:,0,2] = np.reshape(v13[0:n,0:n,slic],n*n)
    R[:,1,0] = np.reshape(v21[0:n,0:n,slic],n*n)
    R[:,1,1] = np.reshape(v22[0:n,0:n,slic],n*n)
    R[:,1,2] = np.reshape(v23[0:n,0:n,slic],n*n)
    R[:,2,0] = np.reshape(v31[0:n,0:n,slic],n*n)
    R[:,2,1] = np.reshape(v32[0:n,0:n,slic],n*n)
    R[:,2,2] = np.reshape(v33[0:n,0:n,slic],n*n)

    return D,R
def in_log_euclidian_2d(D,R):
    """
    This function transforms euclidian tensor into log_euclidian
    """
    log_D = np.zeros((len(D),2,2))
    log_matrix = np.zeros((len(D),2,2))
    for i in xrange(len(D)):
        if D[i,0,0] < 0 or D[i,0,0] == 0:
            D[i,0,0] = 0.0001
        if D[i,1,1] < 0 or D[i,1,1] == 0:
            D[i,1,1] = 0.0001
        log_D[i,0,0] = np.log(D[i,0,0])
        log_D[i,1,1] = np.log(D[i,1,1])
        log_matrix[i,:,:] = np.dot(np.dot(R[i,:,:],log_D[i,:,:]),R[i,:,:].T)
    return log_matrix
def in_log_euclidian(D,R):
    """
    This function transforms euclidian tensor into log_euclidian
    """
    log_D = np.zeros((len(D),3,3))
    log_matrix = np.zeros((len(D),3,3))
    for i in xrange(len(D)):
        if D[i,0,0] < 0 or D[i,0,0] == 0:
            D[i,0,0] = 0.0001
        if D[i,1,1] < 0 or D[i,1,1] == 0:
            D[i,1,1] = 0.0001
        if D[i,2,2] < 0 or D[i,2,2] == 0:
            D[i,2,2] = 0.0001
        log_D[i,0,0] = np.log(D[i,0,0])
        log_D[i,1,1] = np.log(D[i,1,1])
        log_D[i,2,2] = np.log(D[i,2,2])
        log_matrix[i,:,:] = np.dot(np.dot(R[i,:,:],log_D[i,:,:]),R[i,:,:].T)
    return log_matrix

def from_log_euclidian(log_matrix):
    """
    This function return matrix from log_matrix
    """
    matrix = np.exp(log_matrix[:,:])
    return matrix

def test_function(num):
    sigma = 15.0
    d = np.random.normal(0,sigma,num*num)
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
    return np.reshape(z,(num*num))+d

def basis(m,n):
    knots = np.zeros(m+2)
    knots[1:m+1] = np.linspace(0,1,m)
    knots[m:] = 1
    knots[1] = 0
    vs_1 = BsplineVectorSpace(1, knots)
    basis = np.zeros((n,m))
    for i in xrange(m):
        basis[:,i]= vs_1.basis_der(i,0)(np.linspace(0,1,n))
    return basis

def b_matrix():
    S = sparse.coo_matrix(basis(m,n))
    SS = sparse.coo_matrix(sparse.kron(S,S))
    return SS

def qr():
    q,r = np.linalg.qr(b_matrix().toarray())
    return sparse.coo_matrix(r),sparse.coo_matrix(q)

def decomposition(R,lamb):
    B = np.eye(m*m)
    B = sparse.coo_matrix(B).dot(lamb)
    B = scipy.sparse.vstack([R,B])
    U,D,V = scipy.sparse.linalg.svds(B)
    D = sparse.coo_matrix(np.diag(1/D))
    U1 = sparse.coo_matrix(V.transpose()).dot(D)
    U1 = sparse.coo_matrix(R).dot(U1)
    return U1

def function(lamb,y):
    b = PETSc.Vec().createSeq(m*m)
    b.setValues(range(m*m), b_matrix().transpose().dot(y).toarray())
    D = np.eye(m*m)
    D[:1,:] = 0
    D = sparse.coo_matrix(D).dot(lamb)
    B = b_matrix().transpose().dot(b_matrix())
    B = D + B
    A = PETSc.Mat()
    A.create(comm)
    A.setSizes([m*m, m*m])
    A.setType('mpidense')
    A.setUp()
    A.setValues(range(m*m),range(m*m),B.toarray())
    A.assemblyBegin()
    A.assemblyEnd()
    x = PETSc.Vec().createSeq(m*m)
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setType('cg')
    print 'Solving with:', ksp.getType()
    ksp.solve(b, x)
    #SS.setValues(range(m*m), range(m*m),B.toarray())
    #S = sparse.kron(S,SX)
    print 'Converged in', ksp.getIterationNumber(), 'iterations.'
    x = sparse.coo_matrix(x.getArray())
    fun = b_matrix().dot(x.transpose())
    return fun

def CV(Q,R,lamb,y):
    U1 = decomposition(R,lamb)
    trace = np.trace(U1.dot(U1.transpose()).toarray())
    print trace
#    infl = Q.dot(U1.dot(U1.transpose()))
#    d = Q.transpose().dot(y)
    #fun = infl.dot(d)
#    print np.shape(fun),np.shape(y)
    fun = function(lamb,y)
    CV = np.sum((fun.toarray()-y.toarray())**2)/np.sum(n-trace)**2
    return CV

def main(y):
#    lamb = 1.2
#    print y.toarray()
    lam = np.linspace(0,2,num_iter)
    GSV = np.zeros((num_iter))
#    print b_matrix().transpose().dot(y).toarray()
    R,Q = qr()
    print "end QR"
    for i in xrange(num_iter):
        GSV[i] = CV(Q,R,lam[i],y)
        print GSV[i],i
    a = np.unravel_index(GSV.argmin(), GSV.shape)
    lamb = lam[a]
    print b_matrix().transpose().dot(y).toarray()
    b = PETSc.Vec().createSeq(m*m)
    b.setValues(range(m*m), b_matrix().transpose().dot(y).toarray())
    D = np.eye(m*m)
    D[:1,:] = 0
    D = sparse.coo_matrix(D).dot(lamb)
    B = b_matrix().transpose().dot(b_matrix())
    B = D + B
    A = PETSc.Mat()
    A.create(comm)
    A.setSizes([m*m, m*m])
    A.setType('mpidense')
    A.setUp()
    A.setValues(range(m*m),range(m*m),B.toarray())
    A.assemblyBegin()
    A.assemblyEnd()
    x = PETSc.Vec().createSeq(m*m)
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setType('cg')
    #ksp.set
    print 'Solving with:', ksp.getType()
    ksp.solve(b, x)
    #SS.setValues(range(m*m), range(m*m),B.toarray())
    #S = sparse.kron(S,SX)
    print 'Converged in', ksp.getIterationNumber(), 'iterations.'
#    print x.getArray()
    x = sparse.coo_matrix(x.getArray())
    fun = b_matrix().dot(x.transpose()).toarray()
#    x = np.linspace(0,1,n)
#    Z = np.reshape(fun,(n,n))
#    fig = plt.figure()
#    ax = fig.add_subplot(111,projection ='3d')
#    X,Y = np.meshgrid(x,x)
#    ax.plot_surface(X,Y,Z, rstride =4, cstride =4, color ='b')
    return fun

def test_case():
    data = test_data()
    x = np.linspace(0,1,n)
    Z = np.reshape(data[:,0,0],(n,n))
    #te = np.reshape(test,(num,num))
#    fig = plt.figure()
#    ax = fig.add_subplot(111,projection ='3d')
#    X,Y = np.meshgrid(x,x)
#    ax.plot_surface(X,Y,Z, rstride =4, cstride =4, color ='b')
#    for i in xrange(n*n):
#        if data[i,0,0]<-20 or math.isnan(data[i,0,0]) == True:
#            data[i,0,0] = -13.5

    #print len(data[:,0,0])
   # y = sparse.coo_matrix(test_function(n)).transpose()
    y = sparse.coo_matrix(data[:,0,0]).transpose()
    f = main(y)
    d = np.zeros((n*n,2,2))
    d[:,0,0] = f[:,0]
    y = sparse.coo_matrix(data[:,0,1]).transpose()
    f = main(y)
    d[:,0,1] = f[:,0]
    d[:,1,0] = f[:,0]
    y = sparse.coo_matrix(data[:,1,1]).transpose()
    f = main(y)
    d[:,1,1] = f[:,0]
    res = from_log_euclidian(d)
    #plot_2d(res)
    return res

def real_case():
    filename = "./DTI080803/"
    data = real_data(filename)
    D,R = read_tensor(filename)
    T = np.zeros((len(D),3,3))
    for i in xrange(len(D)):
        T[i,:,:] = np.dot(np.dot(R[i,:,:],D[i,:,:]),R[i,:,:].T)
    del T,D,R
    print np.max(data[:,0,0]),np.max(data[:,1,1]),np.max(data[:,2,2])
    for i in xrange(n*n):
        if data[i,0,0]<-20 or math.isnan(data[i,0,0]) == True:
            data[i,0,0] = np.mean(data[:,0,0])
        if data[i,1,1]<-20 or math.isnan(data[i,0,0]) == True:
            data[i,1,1] = np.mean(data[:,1,1])
    y = sparse.coo_matrix(data[:,0,0]).transpose()
    f = main(y)
    d = np.zeros((n*n,2,2))
    d[:,0,0] = f[:,0]
    y = sparse.coo_matrix(data[:,0,1]).transpose()
    f = main(y)
    d[:,0,1] = f[:,0]
    d[:,1,0] = f[:,0]
    y = sparse.coo_matrix(data[:,1,1]).transpose()
    f = main(y)
    d[:,1,1] = f[:,0]
    res = from_log_euclidian(d)
    return res
#if __name__ == '__main__':
#    num_iter = 10
#    n = 256
#    m = 50
#    slic = 60
#    comm = MPI.COMM_WORLD
#    filename = "./DTI080803/"
#   # test_case()
#    data = real_case()
#   # data = data(filename)
#
#profile = line_profiler.LineProfiler(real_case)
#profile.runcall(real_case)
#profile.print_stats()

def test():
    profile = line_profiler.LineProfiler(real_case)
    profile.runcall(real_case)
    profile.print_stats()