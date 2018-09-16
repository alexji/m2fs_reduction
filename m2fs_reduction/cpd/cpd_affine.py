from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import sys
import numpy as np
from .cpd_p import cpd_p


def register_affine(x, y, w, max_it=150, eps=1e-8, callback=None):
    """
    Registers Y to X using the Coherent Point Drift algorithm, in affine fashion.
    Note: For affine transformation, t = y*b'+1*t'(* is dot). b is any random matrix here.
    Parameters
    ----------
    x : ndarray
        The static shape that y will be registered to. Expected array shape is [n_points_x, n_dims]
    y : ndarray
        The moving shape. Expected array shape is [n_points_y, n_dims]. Note that n_dims should be equal for x and y,
        but n_points does not need to match.
    w : float
        Weight for the outlier suppression. Value is expected to be in range [0.0, 1.0].
    max_it : int
        Maximum number of iterations. The default value is 150.

    Returns
    -------
    t : ndarray
        The transformed version of y. Output shape is [n_points_y, n_dims].
    """
    [n, d] = x.shape
    [m, d] = y.shape
    # initialize t using y.
    t = y
    # initialize sigma^2
    sigma20 = 1e8
    sigma2 = (m*np.trace(np.dot(np.transpose(x), x)) + n*np.trace(np.dot(np.transpose(y), y)) -
              2*np.dot(sum(x), np.transpose(sum(y))))/(m*n*d)
    iter = 0
    # the epsilon
    #eps = np.spacing(1)
#    sigma2 = 100*eps

    while (iter < max_it) and (sigma2 > eps) and abs(sigma2-sigma20)/sigma2>1e-8:
        sigma20 = 1.*sigma2
        [p1, pt1, px] = cpd_p(x, t, sigma2, w, m, n, d)
        # precompute
        Np = np.sum(p1)
        mu_x = np.dot(np.transpose(x), pt1)/Np
        mu_y = np.dot(np.transpose(y), p1)/Np
        # solve for parameters
        b1 = np.dot(np.transpose(px), y)-Np*(np.dot(mu_x, np.transpose(mu_y)))
        b2 = np.dot(np.transpose(y*np.matlib.repmat(p1, 1, d)), y)-Np*np.dot(mu_y, np.transpose(mu_y))
        b = np.dot(b1, np.linalg.inv(b2))
        # ts is the translation
        ts = mu_x-np.dot(b, mu_y)
        sigma22 = np.abs(np.sum(np.sum(x*x*np.matlib.repmat(pt1, 1, d)))-Np *
                         np.dot(np.transpose(mu_x), mu_x) - np.trace(np.dot(b1, np.transpose(b))))/(Np*d)
        # get a float number here
        sigma2 = sigma22[0][0]
        # Update centroids positioins
        t = np.dot(y, np.transpose(b))+np.matlib.repmat(np.transpose(ts), m, 1)
        iter = iter+1
        if callback:
           if not iter or iter and (not iter%10):
              print(iter,sigma2,eps,b.ravel())
              callback(x,t)
    return t,iter,sigma2
