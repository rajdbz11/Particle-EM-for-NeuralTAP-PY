from PF import *

"""
Parameter initialization
"""

# Set noise seed
noise_seed = np.random.randint(10000)
print('noise_seed =',noise_seed)
np.random.seed(seed=noise_seed)

#8933

# -------------------- Initialize the parameters -------------------- 
Nx  = 5    # No. of variables
Nr  = 2*Nx  # No. of neurons
Nh  = 1 + Nx  # No. of external input variables

T   = 1000  # No. of time steps
Th  = 2     # No. of time steps for which h is the same
Ns  = 1     # No. of batches
lam = np.array([0.25])  # low pass filtering constant for the TAP dynamics

nltype = 'sigmoid' # external nonlinearity in TAP dynamics

# Noise covariances 
Qpr = 1e-5*np.eye(Nx) # process noise
Qobs = 4e-4*np.eye(Nr) # measurement noise

# Initialize ground truth parameters

G = np.array([0,2,0,0,0,0,0,0,0,0,4,-4,0,-8,8,0,0,0]) # message passing parameters corresponding to the TAP equation

Jtype = 'nonferr'
sc_J = 1 # 1 : self coupling ON, 0: OFF
if Nx <= 4:
    sp_J = 0.1 # sparsity in J
else:
    sp_J = 0.3 # sparsity in J
    
J = 3*Create_J(Nx, sp_J, 'Jtype', sc_J) # Coupling matrix  


U = np.random.randn(Nr,Nx) # embedding matrix

V = np.linalg.svd(np.random.randn(Nx,Nh), full_matrices=False)[2] # input embedding matrix

# concatenate the parameters
theta = np.concatenate([lam, G, JMatToVec(J), U.flatten('F'), V.flatten('F') ])

"""
Generate the latent dynamics and observations
"""

# Inputs signal parameters
gh = 25/np.sqrt(Nx) # gain for inputs h
hMatFull = np.zeros([Nh,T,Ns]) # inputs h

# Initial conditions for latent dynamics
x0Full = np.random.rand(Nx,Ns) # initial x

# Initialize arrays to save the dynamics
xMatFull = np.zeros([Nx,T+1,Ns])
rMatFull = np.zeros([Nr,T,Ns])

# Filter used to smooth the inputs
b = signal.hamming(5,sym=True) 
b = b/sum(b)

for s in range(Ns):
    hMatFull[:,:,s] = signal.filtfilt(b,1,generateBroadH(Nh,T,Th,gh))
    xMatFull[:,:,s], rMatFull[:,:,s] = runTAP(x0Full[:,s], hMatFull[:,:,s], Qpr, Qobs, theta, nltype)

"""
Use ICA to get initial estimate of the embedding from the latent dynamics
"""

U_1 = UhatICA(np.reshape(rMatFull,[Nr,T*Ns],order='F').T, Nx, U)[0]

# Estimate the permutation matrix P (need this to resolve permutation ambiguity of ICA)
P = EstimatePermutation_ICA(U,U_1)


"""
Run the particle filter on the neural activity using the ground truth parameters 
to obtain the data log likelihood.
"""

# Pick only a subset of the data. 
# We used a lot more data to obtain the initial estimate of the embedding using ICA. 
# We will use a subset of the data for the particle filter - EM
TTotal = 500 # Total no. of time steps to use for analysis
T = TTotal//Ns

# Pick only the required subset and discard the rest
rMatFull = rMatFull[:,0:T,:]
hMatFull = hMatFull[:,0:T,:]
xMatFull = xMatFull[:,0:T+1,:]


# ---------  Run the particle filter with true values of (G, J, U, V) --------

K = 100                              # No. of particles

x_truedec = np.zeros([Nx,T+1,Ns])    # decoded latent dynamics using ground truth parameters
P_truedec = np.zeros([Nx,K,T+1,Ns])  # dynamics of individual particles
r_truedec = np.zeros([Nr,T,Ns])      # fit to measurements using ground truth parameters
W_truedec = np.zeros([K,Ns])         # weights of each particles
LL        = np.zeros([Ns])           # data log-likelihood for each session

for s in range(Ns):
    LL[s],x_truedec[:,:,s], P_truedec[:,:,:,s], W_truedec[:,s] = particlefilter(rMatFull[:,:,s], hMatFull[:,:,s], K, Qpr, Qobs, theta, nltype)
    r_truedec[:,:,s] = np.dot(U,x_truedec[:,1:,s])

print('Log likelihood with true params = ',LL.mean())

"""
Initialize the parameters for PF-EM
"""

lam_1 = np.array([0.25])                # use ground truth low pass filtering constant

G_1 = 0.1*np.random.randn(18)           # init for message passing parameters 
J_1 = Create_J(Nx, sp_J, 'Jtype', sc_J) # init for coupling matrix
                                        # init for embedding matrix was obtained using ICA 
V_1 = np.linalg.svd(np.random.randn(Nx,Nh), full_matrices=False)[2] # init for the input mapping V

# Initialize the arrays for latent dynamics, fits to meausurements, particles, 
# particle weights and data log-likelihoods obtained using current values of the parameters

x_1 = np.zeros([Nx,T+1,Ns])
r_1 = np.zeros([Nr,T,Ns])
P_1 = np.zeros([Nx,K,T+1,Ns])
W_1 = np.zeros([K,Ns])
L_1 = np.zeros([Ns])


theta_1 = np.concatenate([lam_1, G_1, JMatToVec(J_1), U_1.flatten('F'), V_1.flatten('F') ])

# Run the PF using initial estimates of the parameters
for s in range(Ns):
    L_1[s],x_1[:,:,s], P_1[:,:,:,s], W_1[:,s] = particlefilter(rMatFull[:,:,s], hMatFull[:,:,s], K, Qpr, Qobs, theta_1, nltype)
    r_1[:,:,s] = np.dot(U_1,x_1[:,1:,s])



# Record the initial values of the required parameters and arrays
xinit = x_1*1.0
rinit = r_1*1.0

Jinit = J_1*1.0
Ginit = G_1*1.0
Uinit = U_1*1.0
Vinit = V_1*1.0
lam_init = lam_1*1.0

#del x_truedec, r_truedec, P_truedec, W_truedec

print('Log likelihood before EM = ',L_1.mean())

"""
Run the particle EM algorithm
"""

# Run the PF-EM on mini-batches of the entire data. Each batch comprises a subset of sessions. 
# (Right now, each mini-batch can comprise just one individual session. I might have to update this to handle multiple sessions)

idx     = np.random.randint(Ns)
rB      = rMatFull[:,:,idx]      # pick the observations for the mini batch
hB      = hMatFull[:,:,idx]      # pick the input signals for the mini batch 
P_B     = P_1[:,:,:,idx]
W_B     = W_1[:,idx]

EMIters = 50                 # maximum no. of iterations to use for the EM algorithm
MStepMaxIter = 10            # Maximum no. of iterations used by the optimizer in the M step
LMat   = np.zeros([EMIters]) # record how the log-likelihood changes with iterations

computegrad = np.array([1,1,0,1,0],dtype=int) # Flags which indicate which variables are updated in the order: G, J, U, V, lam

# We keep the Uhat fixed for the first 20% of EM iterations and update the rest. After that, we update all the parameters together.
# Right now, I am just running the EM algorithm for EMIters no. of iterations. Instead I need to run till convergence.

for iterem in range(EMIters):
    
    if iterem == EMIters//5:
        computegrad = [1, 1, 1, 1, 0]

    if iterem % 50 == 49:
        print('iterem =', iterem + 1)  
    
    MStep = optimize.minimize(NegLL, theta_1, args = (rB, hB, P_B, W_B, Qpr, Qobs, nltype, computegrad), method='BFGS', jac = NegLL_D, options={'disp': False,'maxiter':MStepMaxIter})
    theta_1 = MStep.x
        
    # E step: Pick a new batch and run the particle filter with the updated parameters    
    idx     = np.random.randint(Ns)
    rB      = rMatFull[:,:,idx] # pick the observations for the mini batch
    hB      = hMatFull[:,:,idx]
    
    LMat[iterem], x_B, P_B, W_B = particlefilter(rB, hB, K, Qpr, Qobs, theta_1, nltype)
    

# Extract parameters
lam_1, G_1, J_1, U_1, V_1 = extractParams(theta_1, 18, Nx, Nh, Nr)

# Run the PF with new parameters on all the batches
for s in range(Ns):
    L_1[s],x_1[:,:,s], P_1[:,:,:,s], W_1[:,s] = particlefilter(rMatFull[:,:,s], hMatFull[:,:,s], K, Qpr, Qobs, theta_1, nltype)
    r_1[:,:,s] = np.dot(U_1,x_1[:,1:,s])


print('Log likelihood post EM = ', L_1.mean())

