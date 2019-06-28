import numpy as np
import time
from scipy import signal
from scipy import sparse
from scipy import optimize
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from scipy.io import savemat
import pickle


import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import torch.nn.init as init

class RNN(nn.Module):
    """
    -- RNN model class --
    Architecture:
    Input layer --> Hidden recurrent layer --> Output recurrent layer
    """

    def __init__(self, input_dim, hidden_dim, output_dim, use_cuda):
        super(RNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_cuda = use_cuda
        
        self.input_to_hidden = nn.Linear(input_dim + hidden_dim, hidden_dim, bias = True)
        self.hidden_to_output = nn.Linear(hidden_dim + output_dim, output_dim, bias = True)
        self.activation_function = nn.ReLU()
        
    
    def forward(self, inputs):
        
        N_batches, T = inputs.shape[0], inputs.shape[1]
        
        # Initialize the activity of the recurrent layers
        output_activity, hidden_activity = torch.rand(N_batches, self.output_dim), torch.rand(N_batches, self.hidden_dim)
        output_activity = np.sqrt(self.output_dim)*output_activity
        hidden_activity = np.sqrt(self.hidden_dim)*hidden_activity
        
        r = torch.zeros(N_batches, T, self.output_dim) # output layer activity
        h = torch.zeros(N_batches, T, self.hidden_dim) # hidden layer activity
        
        if self.use_cuda and torch.cuda.is_available():
            r = r.cuda()
            h = h.cuda()
            output_activity = output_activity.cuda()
            hidden_activity = hidden_activity.cuda()
        
        for t in range(T):
            
            combined_inputs_hiddenlayer = torch.cat((inputs[:,t,:], hidden_activity),1)
            hidden_activity = self.activation_function(self.input_to_hidden(combined_inputs_hiddenlayer))
            
            combined_inputs_outputlayer = torch.cat((hidden_activity, output_activity),1)
            output_activity = self.activation_function(self.hidden_to_output(combined_inputs_outputlayer))
            
            r[:,t,:] = output_activity
            h[:,t,:] = hidden_activity
        
        return r, h



def Create_J(Nx, sp, Jtype, SelfCoupling):
    
    """
    Generate a sparse, symmetric coupling matrix with desired kind of interactions

    Inputs: 
    Nx    : No. of x's
    sp    : degree of sparsity of J
    Jtype : coupling type - ferromagnetic (all positive), antiferr (all negative), nonferr (mixed)
    SelfCoupling: determines if J matrix has self coupling or not

    Output
    J     : coupling matrix
    """

    # Create the mask for zeros
    H = np.random.rand(Nx,Nx)
    H = np.tril(H,k=-1)
    H[H < sp] = 0
    H[H >= sp] = 1
    
    if (SelfCoupling == 1):
        H = H + H.T + np.eye(Nx)
    else:
        H = H + H.T
        
    # Create full coupling matrix with required kind of interaction
    
    if Jtype == 'ferr':
        J = np.tril(np.random.rand(Nx,Nx),-1)
        J = J + J.T + np.diag(np.random.rand(Nx))
        J = J/np.sqrt(Nx)
    elif Jtype == 'antiferr':
        J = -np.tril(np.random.rand(Nx,Nx),-1)
        J = J + J.T + np.diag(np.random.rand(Nx))
        J = J/np.sqrt(Nx)
    else:
        J = np.tril(0.5*np.random.randn(Nx,Nx),-1)
        J = J + J.T + np.diag(0.5*np.random.randn(Nx))
        J = J/np.sqrt(Nx)
        
    # Apply mask
    if sp != 0:
        J = J*H
        
    return J


def generateBroadH(Nx,T,Th,scaling):
    """
    Function to generate h(t), the input to the TAP dynamics
    Modeling h(t) such that it stays constant for every Nh time steps.
    """    

    # First generate only T/Nh independent values of h
    shape = 1 # gamma shape parameter
    Lh = T//Th + 1*(T%Th != 0)
    gsmScale = np.random.gamma(shape,scaling,(Nx,Lh))
    hInd = gsmScale*np.random.randn(Nx,Lh)
    hMat = np.zeros([Nx,T])

    # Then repeat each independent h for Nh time steps
    for t in range(T):
        hMat[:,t] = hInd[:,t//Th]
        
    return hMat



def nonlinearity(x,nltype):

    if nltype == 'sigmoid':
        y = 1/(1 + np.exp(-x))
        dy = y*(1-y)
    elif nltype == 'expsqrt':
        y   = np.sqrt(np.log(1 + np.exp(x)))
        dy  = np.exp(x)/(y*(1+np.exp(x)))
    elif nltype == 'dgauss':
        y   = .5 + x*np.exp(-x**2)
        dy  = np.exp(-x**2)*(1-2*x**2)
    elif nltype == 'xcauchytanh':
        y   = .5 + x/(1+x**2) + .05*np.tanh(x)
        dy  = 1/(1 + x**2) + x/((1 + x**2)**2) + 0.05*1/(np.cosh(x)**2)
    else:
        print('Nonlinearity unknown')
        
    return y, dy

def JVecToMat(JVec,Nx):
    JMat = np.zeros([Nx,Nx])
    for kk in range(Nx):
        JMat[kk:,kk] = JVec[0:Nx-kk]
        JMat[kk,kk:] = JVec[0:Nx-kk]
        JVec = np.delete(JVec,np.arange(Nx-kk))
        
    return JMat

def JMatToVec(JMat):
    Nx = np.shape(JMat)[0]
    JVec = np.zeros([1])
    
    for kk in range(Nx):
        JVec = np.concatenate((JVec,JMat[kk:,kk].flatten()),axis=0)
        
    JVec = JVec[1:]
        
    return JVec


def extractParams(theta, lG, Nx, Nh, Nr):
    # extract the parameters
    NJ = np.int(Nx*(Nx+1)/2)
    
    lam = theta[0]
    G = theta[1:1+lG]
    JVec = theta[1+lG:1+lG+NJ]
    J = JVecToMat(JVec,Nx)
    U = np.reshape(theta[1+lG+NJ:1+lG+NJ+Nr*Nx],[Nr,Nx],'F')
    V = np.reshape(theta[1+lG+NJ+Nr*Nx:],[Nx,Nh],'F')
    
    return lam, G, J, U, V
        
    
def runTAP(x0, hMat, Qpr, Qobs, theta, nltype):

    """
    % Function that generates the TAP dynamics

    % Inputs: 
    % x0    : latent variables at time t = 0
    % hMat  : of size Nx x T, specifies inputs h(t) for t = 1,..,T
    % lam   : low pass fitlering constant for TAP dynamics
    % Qpr   : covariance of process noise
    % Qobs  : covariance of measurement noise
    % U     : embedding matrix from latent space to neural activity
    % V     : emedding matrix from input space to latent variable space
    % J     : coupling matrix of the underlying distribution
    % G     : global hyperparameters

    % Outputs: 
    % xMat  : latent variables 
    % rMat  : neural activity. r = Ux + noise
    """

    Nh, T = hMat.shape # input dimensions, no. of time steps
    Nx = Qpr.shape[0]  # latent dimensions
    Nr = Qobs.shape[0] # output dimensions

    lG = 18 # hard coded for now
    lam, G, J, U, V = extractParams(theta, lG, Nx, Nh, Nr)

    x = x0 # initial value of x

    xMat = np.zeros([Nx,T+1])
    xMat[:,0] = x0

    J2 = J**2

    for tt in range(T):  
        
        ht = hMat[:,tt]

        x2      = x**2
        J1      = np.dot(J,np.ones([Nx]))
        Jx      = np.dot(J,x)
        Jx2     = np.dot(J,x2)
        J21     = np.dot(J2,np.ones([Nx]))
        J2x     = np.dot(J2,x)
        J2x2    = np.dot(J2,x2)

        argf = np.dot(V,ht) + G[0]*J1 + G[1]*Jx + G[2]*Jx2 + G[9]*J21 + G[10]*J2x + G[11]*J2x2 + x*( G[3]*J1 + G[4]*Jx + G[5]*Jx2 + G[12]*J21 + G[13]*J2x + G[14]*J2x2 ) + x2*(G[6]*J1 + G[7]*Jx + G[8]*Jx2 + G[15]*J21 + G[16]*J2x + G[17]*J2x2)
        
        TAPFn = nonlinearity(argf, nltype)[0]
        xnew = (1-lam)*x + lam*TAPFn + np.random.multivariate_normal(np.zeros(Nx),Qpr)
        xMat[:,tt+1] = xnew
        x = xnew

    rMat = np.dot(U,xMat[:,1:]) + np.random.multivariate_normal(np.zeros(Nr),Qobs,T).T  #Adding independent observation noise to each time step

    return xMat, rMat



def UhatICA(R, Nx):
	"""
	Function to recover initial estimate of the embedding matrix U
	from the latent dynamics using ICA
	"""

	ica = FastICA(n_components=Nx, algorithm='deflation',fun='cube')
	# R = np.reshape(rMatFull,[Nr,T*Ns],order='F').T
	Xe = ica.fit_transform(R)  # Reconstruct signals
	Uhat = ica.mixing_  # Get estimated mixing matrix
	m = ica.mean_

	Xe = Xe + np.dot(np.linalg.pinv(Uhat),m)

	minx = np.min(Xe,axis=0)
	maxx = np.max(Xe,axis=0)
	DW = np.zeros([Nx])
	for ii in range(Nx):
	    if abs(minx[ii]) > abs(maxx[ii]):
	        DW[ii] = minx[ii]
	    else:
	        DW[ii] = maxx[ii]

	Uhat = Uhat*DW
	Xe = Xe/DW
	
	return Uhat, Xe.T


def EstimatePermutation_ICA(U,U_1):
	"""
	Function that estimates the permutation matrix for ICA estimate of the embedding U
	Inputs:
		U 		ground truth embedding
		U_1 	ICA estimate of embedding
	Outputs:
		P 		permutation matrix such that U = U_1 x P
	"""
	Nx = U.shape[1]
	P = np.zeros([Nx,Nx])

	for i in range(Nx):
	    err = np.sum((np.expand_dims(U_1[:,i],1) - U)**2, axis=0)
	    idx = np.argsort(err)
	      
	    if i == 0:
	        taken = np.array([idx[0]])
	        P[i,idx[0]] = 1
	    else:
	        k = 0
	        while np.intersect1d(idx[k], taken).shape[0] != 0:
	            k += 1
	            
	        P[i,idx[k]] = 1
	        taken = np.append(taken, idx[k])

	return P



    
def resampleSystematic(w, N):
    """
    % [ indx ] = resampleSystematic( w, N)
    % Systematic resampling method for particle filtering. 
    % Author: Tiancheng Li,Ref:
    % T. Li, M. Bolic, P. Djuric, Resampling methods for particle filtering, 
    % submit to IEEE Signal Processing Magazine, August 2013

    % Input:
    %       w    the input weight sequence 
    %       N    the desired length of the output sequence(i.e. the desired number of resampled particles)
    % Output:
    %       indx the resampled index according to the weight sequence
    """

    M = len(w)
    w = w/sum(w)
    Q = np.cumsum(w)
    indx = np.zeros([N],dtype=int)
    T = np.linspace(0,1-1/N,N) + np.random.rand(1)/N;

    i = 0
    j = 0
    
    while (i<N) and (j<M):
        while Q[j] < T[i]:
            j = j + 1

        indx[i] = j
        i = i + 1
        
    return indx
    
def particlefilter(rMat, hMat, K, P, M, theta, nltype):
# function [ LL, xhat, ParticlesAll, WVec, ESSVec] = particlefilter(rMat, hMat, K, P, M, RG, theta, nltype)

    """
    % Particle filter function specific to the TAP dynamics
    % Type of particle filter: standard SIR filter

    % Inputs: 
    % rMat  : observations r(t)
    % hMat  : inputs h(t)
    % K     : No. of particles
    % lam   : low pass filtering constant for the TAP dynamics
    % P     : covariance of process noise
    % M     : covariance of observation noise
    % theta : parameter vector which contains G J and U
    % U     : embedding matrix, r = Ux + noise
    % V     : input embedding matrix
    % J     : coupling matrix
    % G     : global hyperparameters
    % nltype: nonlinearity used in the TAP dynamics

    % Ouputs:
    % LL    : data log likelihood
    % xhat  : decoded latent variables xhat(t)
    % ParticlesAll: set of particles for all time steps
    % WVec  : weights of the particles
    % ESSVec: Effective sample size at each time
    """

    Nr, T = rMat.shape
    Nx = P.shape[1]
    Nh = hMat.shape[0]
    
    lG = 18
    lam, G, J, U, V = extractParams(theta, lG, Nx, Nh, Nr)
    UT = U.T
    J2 = J**2

    ParticlesAll = np.zeros([Nx,K,T+1])
    x = np.reshape(np.dot(np.linalg.pinv(U),rMat[:,0]),[Nx,1]) + np.random.multivariate_normal(np.zeros([Nx]),P,K).T #old particles
    ParticlesNew = np.zeros([Nx,K])

    ParticlesAll[:,:,0] = x
    
    WVec = np.ones([K])/K
    ESSVec = np.zeros([T])
    
    Pinv = np.linalg.inv(P)
    Q_postinv  = Pinv + np.dot(UT,np.linalg.solve(M,U))
    Q_post = np.linalg.inv(Q_postinv)

    Q_post = (Q_post + Q_post.T)/2 #just to ensure it is perfectly symmetric (numerical errors creepy)


    LL = 0 #log likelihood log(p(R))

    for tt in range(T):

        ht = hMat[:,tt]
        rt = rMat[:,tt]
        Minvr = np.linalg.solve(M,rt)
        rMinvr = np.dot(rt.T,Minvr)
        UMinvr = np.dot(UT,Minvr)

        # sampling x(t) from the proposal distribution p(x(t)|x(t-1), r(t))
        # p(x(t)|x(t-1),r(t)) = 1/Z*p(x(t)|x(t-1))*p(r(t)|x(t))

        x2      = x**2
        J1      = np.expand_dims(np.dot(J,np.ones([Nx])),1)
        Jx      = np.dot(J,x)
        Jx2     = np.dot(J,x2)
        J21     = np.expand_dims(np.dot(J2,np.ones([Nx])),1)
        J2x     = np.dot(J2,x)
        J2x2    = np.dot(J2,x2)

        argf = np.expand_dims(np.dot(V,ht),1) + G[0]*J1 + G[1]*Jx + G[2]*Jx2 + G[9]*J21 + G[10]*J2x + G[11]*J2x2 + x*( G[3]*J1 + G[4]*Jx + G[5]*Jx2 + G[12]*J21 + G[13]*J2x + G[14]*J2x2 ) + x2*(G[6]*J1 + G[7]*Jx + G[8]*Jx2 + G[15]*J21 + G[16]*J2x + G[17]*J2x2)
        
        outmat = nonlinearity(argf, nltype)[0]
        f_tap = (1-lam)*x + lam*outmat
        Pinvf_tap = np.linalg.solve(P,f_tap)
        v = Pinvf_tap + UMinvr.reshape([Nx,1])
        #v = (Pinvf_tap.T + UMinvr).T
        mu_post = np.linalg.solve(Q_postinv,v) # mean of the proposal distribution

        # draw new particles from this proposal distribution
        ParticlesNew = np.random.multivariate_normal(np.zeros([Nx]),Q_post,K).T + mu_post

        # assigning weights to the particles proportional to p(r(t)|x(t-1))
        w_ii = np.exp(-0.5*( rMinvr + sum(f_tap*Pinvf_tap - v*mu_post) )) + 1e-128 #adding a small constant to avoid nan problem
        WVec = WVec*w_ii

        LL = LL + np.log(sum(WVec))

        ParticlesAll[:,:,tt+1] = ParticlesNew

        WVec = WVec/sum(WVec) # Normalize the weights

        # Resample the particles based on their weights

        ESS = 1/sum(WVec**2)
        ESSVec[tt] = ESS

        if ESS < K/2 and tt != T:
            idx = resampleSystematic(WVec,K)
            ParticlesAll[:,:,0:tt+1] = ParticlesAll[:,idx,0:tt+1]
            WVec = np.ones([K])/K

        x = ParticlesAll[:,:,tt+1]



    xhat = ParticlesAll*WVec.reshape([1,K,1])
    xhat = np.sum(xhat,axis=1)
    
    return LL, xhat, ParticlesAll, WVec

def NegLL(theta, rMat, hMat, P_S, WVec, P, M, nltype, computegrad, alpha_J, alpha_G):
    
    """
    % Function for computing the Log Likelihood cost for the probabilistic
    % model for the TAP dynamics
    % Inputs:
    % rMat  : observations r(t)
    % hMat  : inputs h(t)
    % P_S   : Particles trajectories
    % WVec  : Weights of the particles trajectories
    % P     : covariance of process noise
    % M     : covariance of measurement noise
    % RG    : indicates whether G is of reduced size or not
    % nltype: external nonlinearity for TAP dynamics 
    % theta : parameter vector with the following subcomponents
    % lam   : low pass filtering constant for the TAP dynamics
    % G     :global hyperparameters 
    % J     :coupling matrix
    % U     :embedding matrix, r = Ux + noise
    % V     :embedding of input
    % alpha_J       : scaling of L1 norm of J
    % alpha_G       : scaling of L1 norm of G


    % Output: 
    % Cost C and gradient w.r.t G 
    """

    Nr, T = rMat.shape     # No. of neurons and time steps
    Nx, K = P_S.shape[0:2] # No. of latent variables and no. of particles  
    Nh    = hMat.shape[0]  # input dimension

    lG = 18
    lam, G, J, U, V = extractParams(theta, lG, Nx, Nh, Nr)
    UT = U.T
    J2 = J**2

    # two components of the cost
    C1      = 0
    C2      = 0

    for t in range(T):

        r_t     = rMat[:,t]
        ht      = hMat[:,t]
        x       = P_S[:,:,t]
        x_curr  = P_S[:,:,t+1]

        x2      = x**2
        J1      = np.expand_dims(np.dot(J,np.ones([Nx])),1)
        Jx      = np.dot(J,x)
        Jx2     = np.dot(J,x2)
        J21     = np.expand_dims(np.dot(J2,np.ones([Nx])),1)
        J2x     = np.dot(J2,x)
        J2x2    = np.dot(J2,x2)

        argf = np.expand_dims(np.dot(V,ht),1) + G[0]*J1 + G[1]*Jx + G[2]*Jx2 + G[9]*J21 + G[10]*J2x + G[11]*J2x2 + x*( G[3]*J1 + G[4]*Jx + G[5]*Jx2 + G[12]*J21 + G[13]*J2x + G[14]*J2x2 ) + x2*(G[6]*J1 + G[7]*Jx + G[8]*Jx2 + G[15]*J21 + G[16]*J2x + G[17]*J2x2)

        fx      = nonlinearity(argf,nltype)[0]
        x_pred  = (1-lam)*x + lam*fx
        dx      = x_curr - x_pred
        dr      = r_t.reshape([Nr,1]) - np.dot(U,x_curr)

        # update the cost
        C1 = C1 + 0.5*np.dot(np.sum(dx*np.linalg.solve(P,dx),axis=0), WVec)
        C2 = C2 + 0.5*np.dot(np.sum(dr*np.linalg.solve(M,dr),axis=0), WVec)


    
    # Add the L1 norms of G and J
    C = C1 + C2 + alpha_G*sum(np.abs(G)) + alpha_J*sum(np.abs(JMatToVec(J)))
    
    return C

def NegLL_D(theta, rMat, hMat, P_S, WVec, P, M, nltype, computegrad, alpha_J, alpha_G):

    """
    % Function for computing the derivatives of Log Likelihood cost for the probabilistic
    % model for the TAP dynamics
    % Inputs:
    % rMat  : observations r(t)
    % hMat  : inputs h(t)
    % P_S   : Particles trajectories
    % WVec  : Weights of the particles trajectories
    % lam   : low pass filtering constant for the TAP dynamics
    % P     : covariance of process noise
    % M     : covariance of measurement noise
    % RG    : indicates whether G is of reduced size or not
    % nltype: external nonlinearity for TAP dynamics 
    % theta : parameter vector with the following subcomponents
    % G     :global hyperparameters 
    % J     :coupling matrix
    % U     :embedding matrix, r = Ux + noise
    % V     :embedding of input
    % computegrad: specifies which variables to compute gradient for
    % computegrad(1): G
    % computegrad(2): J
    % computegrad(3): U
    % computegrad(4): V
    % computegrad(5): lam
    % alpha_J       : scaling of L1 norm of J
    % alpha_G       : scaling of L1 norm of G


    % Output: 
    % Gradient w.r.t G
    """

    Nr, T = rMat.shape     # No. of neurons and time steps
    Nx, K = P_S.shape[0:2] # No. of latent variables and no. of particles  
    Nh    = hMat.shape[0]  # input dimension

    lG = 18
    lam, G, J, U, V = extractParams(theta, lG, Nx, Nh, Nr)
    UT = U.T
    J2 = J**2


    # Initialize the gradients
    dG = G*0
    dJ = J*0
    dU = U*0
    dV = V*0
    dlam = np.zeros([1])

    for t in range(T):

        r_t     = rMat[:,t]
        ht      = hMat[:,t]
        x       = P_S[:,:,t]
        x_curr  = P_S[:,:,t+1]

        x2      = x**2
        J1      = np.expand_dims(np.dot(J,np.ones([Nx])),1)
        Jx      = np.dot(J,x)
        Jx2     = np.dot(J,x2)
        J21     = np.expand_dims(np.dot(J2,np.ones([Nx])),1)
        J2x     = np.dot(J2,x)
        J2x2    = np.dot(J2,x2)

        argf = np.expand_dims(np.dot(V,ht),1) + G[0]*J1 + G[1]*Jx + G[2]*Jx2 + G[9]*J21 + G[10]*J2x + G[11]*J2x2 + x*( G[3]*J1 + G[4]*Jx + G[5]*Jx2 + G[12]*J21 + G[13]*J2x + G[14]*J2x2 ) + x2*(G[6]*J1 + G[7]*Jx + G[8]*Jx2 + G[15]*J21 + G[16]*J2x + G[17]*J2x2)

        fx, dfx = nonlinearity(argf,nltype)
        x_pred  = (1-lam)*x + lam*fx
        dx      = x_curr - x_pred
        dr      = r_t.reshape([Nr,1]) - np.dot(U,x_curr)

        
        Pinvdx  = np.linalg.solve(P,dx)
        Im1     = lam*Pinvdx*WVec.reshape([1,K])*dfx
        
        # gradient for U
        if computegrad[2] == 1:
            dU = dU - np.dot(np.linalg.solve(M,dr), x_curr.T*WVec.reshape([K,1]))              

        # gradient for V
        if computegrad[3] == 1:
            dV = dV - np.dot(Im1, ht.reshape([1,Nh])*np.ones([K,1]))
            
        # gradient for lam
        if computegrad[4] == 1:
            dlam = dlam - np.dot(np.sum(Pinvdx*(fx-x),axis=0),WVec)

        # gradient for G
        if computegrad[0] == 1:
            dG[0]   = dG[0] - np.sum(Im1*J1) 
            dG[1]   = dG[1] - np.sum(Im1*Jx)
            dG[2]   = dG[2] - np.sum(Im1*Jx2) 
            dG[3]   = dG[3] - np.sum(Im1*x*J1)
            dG[4]   = dG[4] - np.sum(Im1*x*Jx) 
            dG[5]   = dG[5] - np.sum(Im1*x*Jx2) 
            dG[6]   = dG[6] - np.sum(Im1*x2*J1)
            dG[7]   = dG[7] - np.sum(Im1*x2*Jx) 
            dG[8]   = dG[8] - np.sum(Im1*x2*Jx2)
            dG[9]   = dG[9] - np.sum(Im1*J21) 
            dG[10]   = dG[10] - np.sum(Im1*J2x)
            dG[11]   = dG[11] - np.sum(Im1*J2x2) 
            dG[12]   = dG[12] - np.sum(Im1*x*J21)
            dG[13]   = dG[13] - np.sum(Im1*x*J2x) 
            dG[14]   = dG[14] - np.sum(Im1*x*J2x2) 
            dG[15]   = dG[15] - np.sum(Im1*x2*J21)
            dG[16]   = dG[16] - np.sum(Im1*x2*J2x) 
            dG[17]   = dG[17] - np.sum(Im1*x2*J2x2)
            

        # gradient for J 
    
        if computegrad[1] == 1:
            for ii in range(Nx):
                for jj in range(ii + 1):
                    dA = np.zeros([Nx,K])
                    xi = x[ii,:]
                    xj = x[jj,:]
                    x2i = x2[ii,:]
                    x2j = x2[jj,:]
                    Jij = J[ii,jj]

                    if ii == jj:
                        dA[ii,:] = G[0] + G[1]*xj + G[2]*x2j + G[3]*xi + G[4]*xi*xj + G[5]*xi*x2j + G[6]*x2i + G[7]*x2i*xj + G[8]*x2i*x2j + 2*Jij*(G[9] + G[10]*xj + G[11]*x2j + G[12]*xi + G[13]*xi*xj + G[14]*xi*x2j + G[15]*x2i + G[16]*x2i*xj + G[17]*x2i*x2j)
                    else:
                        dA[ii,:] = G[0] + G[1]*xj + G[2]*x2j + G[3]*xi + G[4]*xi*xj + G[5]*xi*x2j + G[6]*x2i + G[7]*x2i*xj + G[8]*x2i*x2j + 2*Jij*(G[9] + G[10]*xj + G[11]*x2j + G[12]*xi + G[13]*xi*xj + G[14]*xi*x2j + G[15]*x2i + G[16]*x2i*xj + G[17]*x2i*x2j)

                        dA[jj,:] = G[0] + G[1]*xi + G[2]*x2i + G[3]*xj + G[4]*xj*xi + G[5]*xj*x2i + G[6]*x2j + G[7]*x2j*xi + G[8]*x2j*x2i + 2*Jij*(G[9] + G[10]*xi + G[11]*x2i + G[12]*xj + G[13]*xj*xi + G[14]*xj*x2i + G[15]*x2j + G[16]*x2j*xi + G[17]*x2j*x2i)

                    dJ[ii,jj] = dJ[ii,jj] - np.sum(Im1*dA)
                    

    # Add gradient of L2 norm of G
    #alpha_G = 0
    dG = dG + alpha_G*np.sign(G)
    
    # Add gradient of L1 norm of J
    #alpha_J = 0
    dJ = dJ + alpha_J*np.sign(J)
    dJ = JMatToVec(dJ)

    dtheta = np.concatenate([dlam, dG, dJ, dU.flatten('F'), dV.flatten('F') ])
    
    return dtheta
