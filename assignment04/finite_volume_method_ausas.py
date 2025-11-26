import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time
import matplotlib
import matplotlib.pyplot as plt
from uvw import RectilinearGrid, DataArray

# Creating coordinates
L1, N1 = 1.0, 51
L2, N2 = 1.0, 51
n1, n2 = N1-1, N2-1

flag_coo = True # Activate this for efficiency on large grids

def CreateMesh(L1, L2, N1, N2, refine_walls=False, alpha=0.12):

    x = np.linspace(0.0, L1, N1)
    y = np.linspace(0.0, L2, N2)

    if(False):
        for i1 in range(1,N1-1):
            x[i1] = alpha*(x[i1]-0.5)
            x[i1] = 0.5 * (np.cos(np.pi * (x[i1] - 1.) / 2.) + 1.)
        
        for i2 in range(1,N2-1):
            y[i2] = alpha*(y[i2]-0.5)
            y[i2] = 0.5 * (np.cos(np.pi * (y[i2] - 1.) / 2.) + 1.)

    if(refine_walls):
        for i1 in range(1,N1-1):
            x[i1] = x[i1] - alpha*np.sin(2*np.pi*x[i1])
        
        for i2 in range(1,N2-1):
            y[i2] = y[i2] - alpha*np.sin(2*np.pi*y[i2])

    return x, y

x, y = CreateMesh(L1, L2, N1, N2, False, 0.12) # Be careful with alpha

# Post-processing
def WriteSol(x, y, n1, n2, n, tn, U):
    filename = 'ref3FVsol'+str(n)+'.vtr'
    grid = RectilinearGrid(filename, (x, y), compression=True)
    grid.addCellData(DataArray(U.reshape(n1,n2), range(2), 'Velocity'))
    grid.write()

# Just for testing the mesh
U = np.random.rand(n1*n2)
WriteSol(x, y, n1, n2, 0, 0.0, U)

#raise Exception('lala')

Nc = n1*n2

mu  = 1.0
rho = 1.0

def jglob(i1,i2,n1):
    return i1 + i2*n1

def fh(i1,i2,n1):
    return i1 + i2*n1

def fv(i1,i2,n1,n2):
    return n1*(n2+1) + i1 + i2*(n1+1)

Nh = n1 * (n2+1)
Nv = n2 * (n1+1)
Nf = Nh + Nv

nunk = Nc + Nf

# Face-to-cell connectivity
FC = -np.ones(shape=(Nf,2), dtype=int)

for k1 in range(n1):
    for k2 in range(n2+1):
        f = fh(k1,k2,n1)
        if(k2 == 0):
            FC[f,0] = jglob(k1,k2,n1)
            FC[f,1] = -1
        elif(k2 == n2):
            FC[f,0] = jglob(k1,k2-1,n1)
            FC[f,1] = -3
        else:
            FC[f,0] = jglob(k1,k2-1,n1)
            FC[f,1] = jglob(k1,k2,n1)

for k1 in range(n1+1):
    for k2 in range(n2):
        f = fv(k1,k2,n1,n2)
        if(k1 == 0):
            FC[f,0] = jglob(k1,k2,n1)
            FC[f,1] = -4
        elif(k1 == n1):
            FC[f,0] = jglob(k1-1,k2,n1)
            FC[f,1] = -2
        else:
            FC[f,0] = jglob(k1-1,k2,n1)
            FC[f,1] = jglob(k1,k2,n1)

# Cell-to-face connectivity
CF = np.zeros(shape=(Nc,4), dtype=int)

for i1 in range(n1):
    for i2 in range(n2):
        g = jglob(i1,i2,n1)
        CF[g,:] = [fh(i1,i2,n1), fv(i1+1,i2,n1,n2), fh(i1,i2+1,n1), fv(i1,i2,n1,n2)]


#plt.spy(A.todense(),precision=0.1,markersize=1)
        
def BuildSystem(x, y, CF, FC, mu, rho, theta, Deltat):

    # Total number of unknowns
    N1, N2 = len(x), len(y)
    n1, n2 = N1-1, N2-1
    Nc = n1 * n2
    Nh = n1 * (n2+1)
    Nv = n2 * (n1+1)
    Nf = Nh + Nv
    nunk = Nc + Nf
    
    if(flag_coo):
        row, col, coefL, coefR = [], [], [], []

    #------------------------------------------------
    # Preliminaries

    t0 = time.time()
    vcells = np.zeros(Nc)
    Dx = np.zeros(Nc)
    Dy = np.zeros(Nc)
    xc = np.zeros(Nc)
    yc = np.zeros(Nc)
    for i1 in range(n1):
        for i2 in range(n2):
            g = jglob(i1,i2,n1)
            Dx[g] = (x[i1+1] - x[i1])
            Dy[g] = (y[i2+1] - y[i2])
            xc[g] = x[i1] + 0.5*Dx[g]
            yc[g] = y[i2] + 0.5*Dy[g]
    vcells = Dx * Dy

    #------------------------------------------------
    # Matrix K
    t0 = time.time()
    dgn = np.zeros(Nf)
    fs = np.zeros(Nf)
    for f in range(Nf):
        g, n = FC[f, :]
        fs[f] = Dx[g] if (f < Nh) else Dy[g]
        
        if(g >= 0 and n >= 0):
            dv = np.array([xc[g]-xc[n], yc[g]-yc[n]])
            dgn[f] = np.linalg.norm(dv) / mu
        else:
            dgn[f] = (0.5*Dy[g])/mu if(f < Nh) else (0.5*Dx[g])/mu
            
        if(flag_coo):
            row.append(f)
            col.append(f)
            coefL.append(dgn[f])
            coefR.append(0.0)

    Kinv = scipy.sparse.diags([dgn], [0], format='lil')
    print('Matrix K: ', time.time() - t0)
    #------------------------------------------------

    #------------------------------------------------
    # Matrix A
    t0 = time.time()
    A = scipy.sparse.lil_matrix((Nf,Nc), dtype=float)
    for f in range(Nf):
        g, n = FC[f, :]
        if(g >= 0 and n >= 0):
            A[f,g] = +1.0
            A[f,n] = -1.0

            if(flag_coo):
                row.append(f)
                row.append(f)
                col.append(Nf+g)
                col.append(Nf+n)
                coefL.append(-1.0*theta)
                coefL.append(1.0*theta)
                coefR.append(-1.0*(theta-1.0))
                coefR.append(1.0*(theta-1.0))
        else:
            A[f,g] = +1.0

            if(flag_coo):
                row.append(f)
                col.append(Nf+g)
                coefL.append(-1.0*theta)
                coefR.append(-1.0*(theta-1.0))

    print('Matrix A: ', time.time() - t0)
    #------------------------------------------------

    #------------------------------------------------
    # Matrix C
    t0 = time.time()
    C = scipy.sparse.lil_matrix((Nc,Nf), dtype=float)
    for g in range(Nc):
        for f in CF[g,:]:
            C[g,f] = fs[f] if(g == FC[f,0]) else -fs[f]

            if(flag_coo):
                row.append(Nf+g)
                col.append(f)
                coefL.append(C[g,f])
                coefR.append(0.0)
                
    print('Matrix C: ', time.time() - t0)
    #------------------------------------------------

    #------------------------------------------------
    # Matrix M
    d = (rho/Deltat) * vcells
    M = scipy.sparse.diags([d], [0], format='lil')
        
    if(flag_coo):
        for g in range(Nc):
            row.append(Nf+g)
            col.append(Nf+g)
            coefL.append(M[g,g])
            coefR.append(M[g,g])
            
    print('Matrix M: ', time.time() - t0)
    #------------------------------------------------

    #------------------------------------------------
    # Global matrix
    if(flag_coo):
        t0 = time.time()
        row = np.array(row)
        col = np.array(col)
        coefL = np.array(coefL)
        coefR = np.array(coefR)
        MatL = scipy.sparse.coo_matrix((coefL, (row, col)), shape=(nunk, nunk))
        MatR = scipy.sparse.coo_matrix((coefR, (row, col)), shape=(nunk, nunk))
        print('Matrix L and R (coo format): ', time.time() - t0)
    else:
        t0 = time.time()
        MatL = scipy.sparse.csr_matrix((nunk,nunk), dtype=float)
        MatL[ 0:Nf    , 0:Nf    ] =  Kinv
        MatL[ 0:Nf    , Nf:nunk ] = -A*theta
        MatL[ Nf:nunk , 0:Nf    ] =  C
        MatL[ Nf:nunk , Nf:nunk ] =  M

        MatR = scipy.sparse.csr_matrix((nunk,nunk), dtype=float)
        MatR[ 0:Nf    , Nf:nunk ] = -A*(theta - 1.0)
        MatR[ Nf:nunk , Nf:nunk ] =  M
        print('Matrix L and R (csr-slicing): ', time.time() - t0)
    #------------------------------------------------
        
    # Visualize sparsity pattern
    #plt.spy(B.todense(),precision=0.1,markersize=1)
    #plt.show()
    
    #------------------------------------------------
    # Vector gb
    gb = np.zeros(nunk)
    gb[Nf:nunk] = vcells[0:Nc]
        
    return MatL, MatR, gb

def PressGrad(t):
    T = 0.5
    return np.cos(2.0*np.pi*t/T)

#Time interval
t0, tf = 0.0, 0.1
nsteps = 100
theta = 1.0
freq_out = 1

#Initialization
t, Deltat = np.linspace(t0, tf, nsteps, retstep=True, endpoint=False)

# Build algebraic objects
MatL, MatR, gb = BuildSystem(x, y, CF, FC, mu, rho, theta, Deltat)

#raise Exception('Bye')

# Initial condition
JU = np.zeros(nunk)

print('Begin loop over time steps:')
t0 = time.time()
for n in range(nsteps):
    
    if(n % freq_out == 0 or n == nsteps-2):
        print('Writing solution file', n, 'at time= ', t[n])
        U = JU[Nf:nunk]
        WriteSol(x, y, n1, n2, n, t[n], U)

    G = PressGrad(t[n] + theta*Deltat)
    rhs = MatR @ JU - G * gb
    JU = scipy.sparse.linalg.spsolve(MatL, rhs)

print('Time stepping: ', time.time()-t0)

