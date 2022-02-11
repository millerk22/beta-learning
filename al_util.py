import numpy as np
import matplotlib.pyplot as plt
import graphlearning as gl
from scipy.sparse.linalg import eigsh
from copy import deepcopy
from scipy.special import softmax


def prop_alpha_beta_thresh(V, Lam, c0_ind, c1_ind, dt, kernel='heat', thresh=1e-9):
    '''
    Propagate from labeled points to get alpha and beta, the "amount of mass" of success/failure for each node in graph
    '''
    if kernel == 'heat':
        Sigma = V * np.exp(-dt*Lam)[np.newaxis, :]
    elif kernel == 'diffuse':
        Sigma = V / (1.+dt*Lam)[np.newaxis, :]
    else:
        raise ValueError(f"kernel = {kernel} not valid kernel option")
    
    beta = Sigma @ (V.T[:,c0_ind]).sum(axis=1)
    alpha = Sigma @ (V.T[:,c1_ind]).sum(axis=1)
#     print(alpha.min(), alpha.max(), beta.min(), beta.max())
    if thresh:
        beta[beta < thresh] = 0.0
        alpha[alpha < thresh] = 0.0
        beta /= beta.max()/c0_ind.size
        alpha /= alpha.max()/c1_ind.size
    return alpha, beta


def beta_look_ahead_acquisition(alpha, beta, V, Lam, candidate_ind, classifier='mode', method='vopt', dt=1., kernel='heat', thresh=1e-9, gamma=None, show=False, deg=None, X=None, c0_ind=None, c1_ind=None):
    
    assert method in ['vopt', 'ryan', 'var', 'deg-var', 'mc-avg', 'mc-map'] 
    assert classifier in ['mode', 'mean']
    
    if method == 'mc-avg':
        def beta_mc(k, p):
            alpha0, beta0 = prop_alpha_beta_thresh(V, Lam, np.append(c0_ind, k), c1_ind, dt=0.5, kernel='heat', thresh=1e-9)
            alpha1, beta1 = prop_alpha_beta_thresh(V, Lam, c0_ind, np.append(c1_ind, k), dt=0.5, kernel='heat', thresh=1e-9)

            avg = p[k]*(alpha1+1.)/(alpha1 + beta1 + 2.) + (1.-p[k])*(alpha0 + 1.)/(alpha0 + beta0 + 2.)

            return np.linalg.norm(p - avg)
        u = (alpha + 1.)/(alpha + beta + 2.)
        return np.array([beta_mc(k, u) for k in candidate_ind])


    if method == 'mc-map':
        def beta_mc_map(k, p, yk):
            if yk == 0:
                alpha, beta = prop_alpha_beta_thresh(V, Lam, np.append(c0_ind, k), c1_ind, dt=0.5, kernel='heat', thresh=1e-9)
            else:
                alpha, beta = prop_alpha_beta_thresh(V, Lam, c0_ind, np.append(c1_ind, k), dt=0.5, kernel='heat', thresh=1e-9)

            return np.linalg.norm(p - (alpha + 1.)/(alpha + beta + 2.))
        u = (alpha + 1.)/(alpha + beta + 2.)
        y = 1*(u > 0.5)
        return np.array([beta_mc_map(k, u, y[k]) for k in candidate_ind])
        
        
       
    
    if method == "deg-var":
        assert deg is not None
        var = (alpha + 1.)*(beta + 1.)/((alpha + beta + 2.)**2. * (alpha + beta + 3.))
        return deg[candidate_ind] * var[candidate_ind]
    if method == "var":
        var = (alpha + 1.)*(beta + 1.)/((alpha + beta + 2.)**2. * (alpha + beta + 3.))
        return var[candidate_ind]
    
    if gamma is not None and thresh is not None:
        print("Both gamma and thresh were specified, defaulting to threshold.")
        gamma = None
    
    if gamma is None and thresh is None:
        print("Both gamma and thresh were set as None, defaulting to threshold = 0.01")
        thresh = 0
    
    if kernel == 'heat':
        Sigma = V * np.exp(-dt*Lam)[np.newaxis, :]
    elif kernel == 'diffuse':
        Sigma = V / (1.+dt*Lam)[np.newaxis, :]
    else:
        raise ValueError(f"kernel = {kernel} not valid kernel option")
    
    if classifier == 'mode':
        probs_candidate = alpha/(alpha + beta) # note alpha and beta are without the prior 1,1
        probs_candidate[np.isnan(probs_candidate)] = 0.5
    else:
        probs_candidate = (alpha + 1.)/(alpha + beta + 2.)
    
    weights = Sigma @ V.T[:,candidate_ind] # pretty darn large, would need to distribute in larger problems for memory issues
    if thresh:
        weights[weights < thresh] = 0.0
        weights /= weights.max(axis=0)[np.newaxis, :]
        
    if gamma: # should not occur if thresh is specified, per checks previously done
        weights = (1. + gamma*dt)*weights
    
    
    # not the most efficient, need to redo
    if method == 'ryan':
        influence = np.linalg.norm(weights, axis=0)
        var = (alpha + 1.)*(beta + 1.)/((alpha + beta + 2.)**2. * (alpha + beta + 3.))
#         if show and X is not None:
#             fig, (ax1, ax2) = two_scatter_plots(X, influence, var[candidate_ind], train_ind, train_ind, cand=candidate_ind)
#             ax1.set_title(f"Influence, dt = {dt}")
#             ax2.set_title("Current Variance")
#             plt.show()
        return var[candidate_ind] * influence
        
    betaks = weights + beta[:, np.newaxis] 
    alphaks = weights + alpha[:, np.newaxis] 
    
    result1 = ((alphaks + 1.) * (beta[:,np.newaxis] + 1.)) / ((alphaks + beta[:, np.newaxis] + 2.)**2. * (alphaks + beta[:,np.newaxis] + 3.))
    result0 = ((alpha[:,np.newaxis] + 1.)*(betaks+1.)) / ((alpha[:,np.newaxis] + betaks + 2.)**2. * (alpha[:,np.newaxis] + betaks + 3.))
    
    return -(probs_candidate[candidate_ind]*(result1).sum(axis=0) + (1. - probs_candidate[candidate_ind])*(result0).sum(axis=0)) # negative for max ordering

def acquisition_function(C_a, V, candidate_inds, u, method='vopt', uncertainty_method = 'norm', gamma=0.1):
    '''
    Main function for computing acquisition function values. All available methods are callable from this function.
    Params:
        - C_a : (M x M) numpy array, covariance matrix of current Gaussian Regression model with spectral truncation (M is number of eigenvalues)
        - V : (N x M) numpy array, eigenvector matrix
        - candidate_inds: (N,) numpy array, indices to calculate the acquisition function on. (Usually chosen to be all unlabeled nodes)
        - u: (N x n_c) numpy array, current classifier, where the i^th row represents the prethresholded classifier on the i^th node
        - method: str, which acquisition function to compute
        - uncertainty_method: str, if method requires "uncertainty calculation" this string specifies which type of uncertainty measure to apply
        - gamma: float, value of weighting in spectral truncated Gaussian Regression model. gamma=0 recovers Laplace Learning, which is numerically unstable for covariance matrix calculations.

    Output:
        - acq_vals: (len(candidate_inds), ) numpy array, acquisition function values on the specified candidate_inds nodes
    '''
    assert method in ['uncertainty','vopt','mc','mcvopt']
    assert uncertainty_method in ["norm", "entropy", "least_confidence", "smallest_margin", "largest_margin"]

    if method != 'uncertainty':
        Cavk = C_a @ V[candidate_inds,:].T
        col_norms = np.linalg.norm(Cavk, axis=0)
        diag_terms = (gamma**2. + np.array([np.inner(V[k,:], Cavk[:, i]) for i,k in enumerate(candidate_inds)]))
        
    if method == 'vopt':
        return col_norms**2. / diag_terms


    # Calculate uncertainty terms based on string "uncertainty_method"
    if uncertainty_method == "norm":
        u_probs = softmax(u[candidate_inds], axis=1)
        one_hot_predicted_labels = np.eye(u.shape[1])[np.argmax(u[candidate_inds], axis=1)]
        unc_terms = np.linalg.norm((u_probs - one_hot_predicted_labels), axis=1)
    elif uncertainty_method == "entropy":
        u_probs = softmax(u[candidate_inds], axis=1)
        unc_terms = np.max(u_probs, axis=1) - np.sum(u_probs*np.log(u_probs +.00001), axis=1)
    elif uncertainty_method == "least_confidence":
        unc_terms = np.ones((u[candidate_inds].shape[0],)) - np.max(u[candidate_inds], axis=1)
    elif uncertainty_method == "smallest_margin":
        u_sort = np.sort(u[candidate_inds])
        unc_terms = 1.-(u_sort[:,-1] - u_sort[:,-2])
    elif uncertainty_method == "largest_margin":
        u_sort = np.sort(u[candidate_inds])
        unc_terms = 1.-(u_sort[:,-1] - u_sort[:,0])

    if method == 'uncertainty':
        return unc_terms

    if method == 'mc':
        return unc_terms * col_norms / diag_terms

    else:
        return unc_terms * col_norms **2. / diag_terms

    
def update_C_a(C_a, V, Q, gamma=0.1):
    '''
    Function to update spectral truncation covariance matrix
    Params:
        - C_a: (M x M) numpy array, covariance matrix of current Gaussian Regression model with spectral truncation (M is number of eigenvalues)
        - V : (N x M) numpy array, eigenvector matrix
        - Q: (-1, ) numpy array or Python list, indices of chosen query points. (Implementation allows for batch update)
        - gamma: float, value of weighting in spectral truncated Gaussian Regression model. gamma=0 recovers Laplace Learning, which is numerically unstable for covariance matrix calculations.

    Output:
        - C_a: updated covariance matrix
    '''
    for k in Q:
        vk = V[k]
        Cavk = C_a @ vk
        ip = np.inner(vk, Cavk)
        C_a -= np.outer(Cavk, Cavk)/(gamma**2. + ip)
    return C_a



def al_test(W, X, train_ind, labels, evecs, evals, al_iters=20, method='vopt', classifier='mean', \
            kernel='heat', dt=.5, thresh=1e-9, show=False, return_ll=False, deg=None):
    labeled_ind = deepcopy(train_ind)
    if return_ll:
        print("return_ll will return poisson learning")
        model = gl.ssl.poisson(W)
#         model = gl.ssl.laplace(W)
    
    if kernel == 'heat':
        Sigma = evecs * np.exp(-dt*evals)[np.newaxis, :]
    elif kernel == 'diffuse':
        Sigma = evecs / (1.+dt*evals)[np.newaxis, :]
    
    c0_ind, c1_ind = train_ind[labels[train_ind] == 0], train_ind[labels[train_ind] == 1]
    alpha, beta = prop_alpha_beta_thresh(evecs, evals, c0_ind, c1_ind, dt, kernel=kernel, thresh=thresh)
    if classifier == 'mode':
        p = alpha/(alpha + beta)
        p[np.isnan(p)] = 0.5
    else:
        p = (alpha + 1.)/(alpha + beta + 2.)
     
    
    acc = np.array([])
    p_calc = p.copy()
    p_calc[p_calc == 0.5] += 0.000001*np.random.randn((p_calc==0.5).sum())
    acc = np.append(acc, gl.ssl.ssl_accuracy(1.*(p_calc >= 0.5), labels, labeled_ind.size))
    if return_ll:
        acc_ll = np.array([])
        u_laplace = model.fit(labeled_ind, labels[labeled_ind])[:,1]
        acc_ll = np.append(acc_ll, gl.ssl.ssl_accuracy(1.*(u_laplace >= 0.), labels, labeled_ind.size))
    
    
    for it in range(al_iters):
        candidate_ind = np.setdiff1d(np.arange(X.shape[0]), labeled_ind)
        obj_vals = beta_look_ahead_acquisition(alpha, beta, evecs, evals, candidate_ind, dt=dt, \
                                             classifier=classifier, method=method, deg=deg, c0_ind=c0_ind, c1_ind=c1_ind)
        
        max_inds = np.where(np.isclose(obj_vals, np.max(obj_vals)))[0]
        k = candidate_ind[np.random.choice(max_inds)]
#         k = candidate_ind[np.argmax(obj_vals)]
        
        # add k's propagation based on oracle
        prop_k = Sigma @ evecs.T[:,k]
        if thresh:
            prop_k[prop_k < thresh] = 0.0
            prop_k /= prop_k.max()
            
        if labels[k] == 1:
            alpha += prop_k
        else:
            beta += prop_k
            
        if show:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
            ax1.scatter(X[:,0], X[:,1], c=p)
            ax1.set_title(f"Current Classifier, Iter {it}, Acc = {acc[-1]:.3f}")
            p2 = ax2.scatter(X[candidate_ind,0], X[candidate_ind,1], c=obj_vals)
            ax2.scatter(X[labeled_ind,0], X[labeled_ind,1], c='r', marker='^', s=80)
            ax2.scatter(X[k,0], X[k,1], c='magenta', marker='^', s=100)
            plt.colorbar(p2, ax=ax2)
            ax2.set_title(f"{method.upper()}, {classifier} Objective Values")
            plt.show()
            
            
        labeled_ind = np.append(labeled_ind, k)
        
        
        # update classifier
        if classifier == 'mode':
            p = alpha/(alpha + beta)
            p[np.isnan(p)] = 0.5
        else:
            p = (alpha + 1.)/(alpha + beta + 2.)
    
        
        # calculate new accuracy -- add random coin toss for the 0.5 values
        p_calc = p.copy()
        p_calc[p_calc == 0.5] + 0.000001*np.random.randn((p_calc==0.5).sum())
        acc = np.append(acc, gl.ssl.ssl_accuracy(1.*(p_calc >= 0.5), labels, labeled_ind.size))
        if return_ll:
            u_laplace = model.fit(labeled_ind, labels[labeled_ind])[:,-1]
            acc_ll = np.append(acc_ll, gl.ssl.ssl_accuracy(1.*(u_laplace >= 0), labels, labeled_ind.size))
    
    if return_ll:
        return acc, labeled_ind, acc_ll
    return acc, labeled_ind


def al_test_gl(W, X, train_ind, labels, evecs, evals, al_iters=20, method='vopt', algorithm='laplace', \
            show=False, gamma=0.1):
    labeled_ind = deepcopy(train_ind)
    
    if algorithm == 'laplace':
        model = gl.ssl.laplace(W)
        u = model.fit(labeled_ind, labels[labeled_ind])
    elif algorithm == 'poisson':
        model = gl.ssl.poisson(W)
        u = model.fit(labeled_ind, labels[labeled_ind])
        
#     u = gl.graph_ssl(W, labeled_ind, labels[labeled_ind], algorithm=algorithm, return_vector=True)
       
    acc = np.array([])
    acc = np.append(acc,  gl.ssl.ssl_accuracy(np.argmax(u, axis=1), labels, labeled_ind.size))
    
    C_a = np.linalg.inv(np.diag(evals) + evecs[labeled_ind,:].T @ evecs[labeled_ind,:] / gamma**2.) # M by M covariance matrix

    for it in range(al_iters):
        candidate_ind = np.setdiff1d(np.arange(X.shape[0]), labeled_ind)
        obj_vals = acquisition_function(C_a, evecs, candidate_ind, u, method=method, \
                                                uncertainty_method='smallest_margin', gamma=gamma)
        max_inds = np.where(np.isclose(obj_vals, np.max(obj_vals)))[0]
        k = candidate_ind[np.random.choice(max_inds)]
            
        if show:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
            ax1.scatter(X[:,0], X[:,1], c=p)
            ax1.set_title(f"Current Classifier, Iter {it}, Acc = {acc[-1]:.3f}")
            p2 = ax2.scatter(X[candidate_ind,0], X[candidate_ind,1], c=obj_vals)
            ax2.scatter(X[labeled_ind,0], X[labeled_ind,1], c='r', marker='^', s=80)
            ax2.scatter(X[k,0], X[k,1], c='magenta', marker='^', s=100)
            plt.colorbar(p2, ax=ax2)
            ax2.set_title(f"{method.upper()}, {algorithm} Objective Values")
            plt.show()
            
        # update classifier
        labeled_ind = np.append(labeled_ind, k)
        u = model.fit(labeled_ind, labels[labeled_ind])
        
        C_a = update_C_a(C_a, evecs, [k], gamma=gamma)
        
        # calculate new accuracy
        acc = np.append(acc, gl.ssl.ssl_accuracy(np.argmax(u, axis=1), labels, labeled_ind.size))
    
    return acc, labeled_ind


def two_scatter_plots(X, vals1, vals2, highlight1=None, highlight2=None, cand=None):
    if cand is None:
        cand = np.arange(X.shape[0])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    p1 = ax1.scatter(X[cand,0], X[cand,1], c=vals1)
    if highlight1 is not None:
        ax1.scatter(X[highlight1,0], X[highlight1,1], c='r', marker='^', s=100)
    plt.colorbar(p1, ax=ax1)
    p2 = ax2.scatter(X[cand,0], X[cand,1], c=vals2)
    if highlight2 is not None:
        ax2.scatter(X[highlight2,0], X[highlight2,1], c='r', marker='^', s=100)
    plt.colorbar(p2, ax=ax2)
    return fig, (ax1, ax2)