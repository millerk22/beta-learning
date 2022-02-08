import numpy as np
import scipy.sparse as sparse
from sklearn.datasets import make_moons, make_blobs
import graphlearning as gl

def create_binary_clusters():
    np.random.seed(4)
    Xm, labelsm = make_moons(200, shuffle=False, noise=0.12)
    X1, labels1 = make_blobs([50,60, 40, 30, 40], 2, shuffle=False, centers=[[1.6,-1.3],[1.3,1.7], [0.5, 2.4], [0.2,-1.], [-1.7,2.2]], cluster_std=[.26, .23, .23, .26, .23])
    labels1 = labels1 % 2
    X2 = np.random.randn(100,2) @ np.array([[.4, 0.],[0.,.3]]) + np.array([-1.5,-.8])
    X3 = np.random.randn(70,2) @ np.array([[.4, 0.],[0.,.3]]) + np.array([2.5,2.8])
    x11, x12 = np.array([-2., 0.8])[np.newaxis, :], np.array([-.2,2.])[np.newaxis, :]
    l1 = (x11 + np.linspace(0,1, 80)[:, np.newaxis] @ (x12 - x11))  + np.random.randn(80, 2)*0.18
    x21, x22 = np.array([2.5, -1.5])[np.newaxis, :], np.array([2.5, 2.])[np.newaxis, :]
    l2 = (x21 + np.linspace(0,1, 90)[:, np.newaxis] @ (x22 - x21))  + np.random.randn(90, 2)*0.2


    X = np.concatenate((Xm, X1, X2, X3, l1, l2))
    labels = np.concatenate((labelsm, labels1, np.zeros(100), np.ones(70), np.ones(80), np.zeros(90)))

    return X, labels

def create_clusters(return_clusters=False):
    N, nc = 800, 8
    np.random.seed(5)
    X, clusters = make_blobs(N, 2, shuffle=False, centers=2.*np.hstack((np.cos(2.*np.pi*np.arange(nc)/float(nc))[:,np.newaxis],
                                                                     np.sin(2.*np.pi*np.arange(nc)/float(nc))[:,np.newaxis])), 
                                                                     cluster_std=.32)  # DECREASE std for better clustering
    # Divide into classes by alternating clusters
    labels = np.copy(clusters)
    labels[clusters % 2 == 0] = 0
    labels[clusters % 2 == 1] = 1
    if return_clusters:
        return X, labels, clusters
    return X, labels

def create_checkerboard(N):
    np.random.seed(2)
    X = np.random.rand(N,2)
    labels = np.zeros(N)
    mask = (X[:,0] > 0.25) & (X[:,0] <= 0.5) & (X[:,1] >= 0) & (X[:,1] <= 0.25)
    mask = mask | (X[:,0] > 0.75) & (X[:,0] <= 1.0) & (X[:,1] >= 0) & (X[:,1] <= 0.25)
    mask = mask | (X[:,0] > 0.25) & (X[:,0] <= 0.5) & (X[:,1] >= 0.5) & (X[:,1] <= 0.75)
    mask = mask | (X[:,0] > 0.75) & (X[:,0] <= 1.0) & (X[:,1] >= 0.5) & (X[:,1] <= 0.75)
    mask = mask | (X[:,0] > 0.5) & (X[:,0] <= .75) & (X[:,1] >= 0.25) & (X[:,1] <= 0.5)
    mask = mask | (X[:,0] > 0.) & (X[:,0] <= .25) & (X[:,1] >= 0.25) & (X[:,1] <= 0.5)
    mask = mask | (X[:,0] > 0.5) & (X[:,0] <= .75) & (X[:,1] >= 0.75) & (X[:,1] <= 1.0)
    mask = mask | (X[:,0] > 0.) & (X[:,0] <= .25) & (X[:,1] >= 0.75) & (X[:,1] <= 1.0)
    labels[mask] = 1
    return X, labels

def exp_weight(x):
    return np.exp(-x)

#Perform approximate nearest neighbor search, returning indices I,J of neighbors, and distance D
# Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot".
def knnsearch_annoy(X,k, similarity='euclidean', dataset=None, metric='raw'):

    from annoy import AnnoyIndex

    n = X.shape[0]  #Number of points
    dim = X.shape[1]#Dimension

    print('kNN search with Annoy approximate nearest neighbor package...')
    #printProgressBar(0, n, prefix = 'Progress:', suffix = 'Complete', length = 50)

    u = AnnoyIndex(dim, similarity)  # Length of item vector that will be indexed
    for i in range(n):
        u.add_item(i, X[i,:])

    u.build(10)  #10 trees
    
    D = []
    I = []
    J = []
    for i in range(n):
        #printProgressBar(i+1, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
        A = u.get_nns_by_item(i, k,include_distances=True,search_k=-1)
        I.append([i]*k)
        J.append(A[0])
        D.append(A[1])

    I = np.array(I)
    J = np.array(J)
    D = np.array(D)

    #If dataset name is provided, save permutations to file
    if not dataset is None:
        #data file name
        dataFile = dataset + '_' + metric + '.npz'

        #Full path to file
        dataFile_path = os.path.join(kNNData_dir(), dataFile)

        #Check if Data directory exists
        if not os.path.exists(kNNData_dir()):
            os.makedirs(kNNData_dir())

        np.savez_compressed(dataFile_path,I=I,J=J,D=D)

    return I,J,D


#Compute weight matrix from nearest neighbor indices I,J and distances D
#k = number of neighbors
def weight_matrix(I,J,D,k,f=exp_weight,symmetrize=True):

    #Restrict I,J,D to k neighbors
    k = np.minimum(I.shape[1],k)
    I = I[:,:k]
    J = J[:,:k]
    D = D[:,:k]

    n = I.shape[0]
    k = I.shape[1]

    D = f(D*D)

    #Flatten
    I = I.flatten()
    J = J.flatten()
    D = D.flatten()

    #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
    W = sparse.coo_matrix((D, (I,J)),shape=(n,n)).tocsr()

    if symmetrize:
        W = (W + W.transpose())/2;

    return W

#Compute weight matrix from dataset info
#k = number of neighbors
def knn_weight_matrix(k,data=None,dataset=None,metric='raw',f=exp_weight,symmetrize=True):

    if data is not None:
        I,J,D = knnsearch_annoy(data,k)
    elif dataset is not None:
        I,J,D = gl.load_kNN_data(dataset,metric=metric)
    else:
        sys.exit("Must provide data or a dataset name.")

    return weight_matrix(I,J,D,k,f=f,symmetrize=symmetrize)