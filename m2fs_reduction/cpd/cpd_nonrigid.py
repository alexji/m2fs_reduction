from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import sys
import numpy as np
import numpy.matlib
import scipy.sparse
from .cpd_p import cpd_p


def register_nonrigid(x, y, w, lamb=3.0, beta=2.0, max_it=150, eps=1e-8, callback=None):
    """
    Registers Y to X using the Coherent Point Drift algorithm, in non-rigid fashion.
    Note: For affine transformation, t = y+g*wc(* is dot). 
    Parameters
    ----------
    x : ndarray
        The static shape that Y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for X and Y,
        but n_points does not need to match.
    w : float
        Weight for the outlier suppression. Value is expected to be in range [0.0, 1.0].
    lamb : float, optional
        lamb represents the trade-off between the goodness of maximum likelihood fit and regularization.
        Default value is 3.0.
    beta : float, optional
        beta defines the model of the smoothness regularizer(width of smoothing Gaussian filter in
        equation(20) of the paper).Default value is 2.0.
    max_it : int, optional
        Maximum number of iterations. Used to prevent endless looping when the algorithm does not converge.
        Default value is 150.
    tol : float

    Returns
    -------
    t : ndarray
        The transformed version of y. Output shape is [n_points_y, n_dims].
    """
    # Construct G:
    g = y[:, np.newaxis, :]-y
    g = g*g
    g = np.sum(g, 2)
    g = np.exp(-1.0/(2*beta*beta)*g)
    [n, d] = x.shape
    [m, d] = y.shape
    t = y
    # initialize sigma^2
    sigma20 = 1e8
    sigma2 = (m*np.trace(np.dot(np.transpose(x), x))+n*np.trace(np.dot(np.transpose(y), y)) -
              2*np.dot(sum(x), np.transpose(sum(y))))/(m*n*d)
#    sigma2 = 100*eps

    iter = 0
    while (iter < max_it) and (sigma2 > eps) and abs(sigma2-sigma20)/sigma2>1e-8:
        sigma20 = 1.*sigma2
        [p1, pt1, px] = cpd_p(x, t, sigma2, w, m, n, d)
        # precompute diag(p)
        dp = scipy.sparse.spdiags(p1.T, 0, m, m)
        # wc is a matrix of coefficients
        wc = np.dot(np.linalg.inv(dp*g+lamb*sigma2*np.eye(m)), (px-dp*y))
        t = y+np.dot(g, wc)
        Np = np.sum(p1)
        sigma2 = np.abs((np.sum(x*x*np.matlib.repmat(pt1, 1, d))+np.sum(t*t*np.matlib.repmat(p1, 1, d)) -
                         2*np.trace(np.dot(px.T, t)))/(Np*d))
        iter = iter+1
        if callback:
           if not iter or iter and (not iter%10):
              print(iter,sigma2,eps)
              callback(x,t)
    return t,iter,sigma2
