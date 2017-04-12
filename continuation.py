from __future__ import division
from __future__ import print_function
from six.moves import range
import numpy as np

import scipy
from scipy import linalg, matrix
def null(A, eps=1e-12):
    u, s, vh = scipy.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

def pseudo_arc_length_continuation_step(x, alpha, f, jacobi, delta, error_tol=1e-12):
    """continue solution in tangent direction for delta length
       
       Args:
       x: previous solution, numpy array
       alpha: previous parameter, float
       f: function maps (x, alpha) to vector of x.shape
       jacobi: (f_x, f_alpha)
       delta: float
       error_tol: error tolerance

       Returns
       x_new: 
       alpha_new:
    """
    #y_0 = x.append(alpha)

    y_0 = np.append(x, alpha)
    y = y_0
    jacobi_mat = jacobi(y)
    tangent = null(jacobi_mat).ravel()
    #augmented_jacobi = jacobi_mat.append(tangent)
    augmented_jacobi = np.vstack([jacobi_mat, tangent])
    #print(augmented_jacobi)
    #tangent = null(jacobi_mat)[:, 0]
    while np.sqrt(np.dot(f(y), f(y)) + np.power( np.dot(tangent, y - y_0) - delta, 2 )) > error_tol:
        rhs = f(y)
        rhs = np.append(rhs, ( np.dot(tangent, y - y_0) - delta ))
        y = y - np.linalg.solve(augmented_jacobi, rhs)
    return y[:-1], y[-1]

def natural_parameter_continuation_step(x, alpha, alpha_new, f, jacobi, error_tol=1e-12):
    """natural parameter continuation step
    update the solution of f(x) = 0 from x at parameter value alpha to new solution at parameter value alpha_new
    signiture similar to pseudo_arc_length_continuation_step
    """
    y_0 = np.append(x, alpha)
    y = y_0
    jacobi_mat = jacobi(y)
    tangent = null(jacobi_mat).ravel()
    # get new initial guess
    x_0 = (alpha_new - alpha) * tangent[:-1]/tangent[-1] 
    # update jacobi
    y = np.append(x_0, alpha_new)
    jacobi_mat = jacobi(y)
    reduced_jacobi_mat = jacobi_mat[:, :-1]
    while np.sqrt(np.dot(f(y), f(y))) > error_tol:
        y[:-1] = y[:-1] - np.linalg.solve(reduced_jacobi_mat, f(y))
    return y[:-1], y[-1]

def pseudo_arc_length_continuation(x_0, alpha_0, alpha_1, f, jacobi, delta, error_tol=1e-12):
    x = x_0
    alpha = alpha_0
    s = 0.0
    count = 0
    while alpha < alpha_1 and count < 100000:
        x_new, alpha_new = pseudo_arc_length_continuation_step(x, alpha, f, jacobi, delta, error_tol)
        s += np.sqrt( np.dot(x_new - x, x_new - x) + np.power(alpha_new - alpha, 2) )
        x = x_new
        alpha = alpha_new
        count += 1
    x_new, alpha_new = natural_parameter_continuation_step(x, alpha, alpha_1, f, jacobi, error_tol)
    s -= np.sqrt( np.dot(x_new - x, x_new - x) + np.power(alpha_new - alpha, 2) )
    x = x_new
    alpha = alpha_new
    return x, alpha, s






    


