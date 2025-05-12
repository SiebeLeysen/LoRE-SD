import numpy as np

from lore_sd.utils import SphericalHarmonics as sh

def non_negative_odf(Q):
    """
    Non-negative ODF constraint. ODF must be non-negative along the directions defined in Q (typically 300).
    :param Q: Transformation matrix between angular domain and SH
    :return: dictionary according to scipy.optimize's NonLinearConstraint
    """
    return {'type': 'ineq', 
            'fun': lambda x: Q @ x[:Q.shape[-1]],
            # 'jac': lambda x: np.hstack((Q, np.zeros((Q.shape[0], x.shape[-1]-Q.shape[-1])))),
            }

def sum_of_fractions_equals_one(lmax):
    """
    All fractions used in the respones functions representation must sum to unity.
    :param lmax: Max SH order
    :return: dictionary according to scipy.optimize's NonLinearConstraint
    """
    return {'type': 'eq', 
            'fun':lambda d: np.sum(d[sh.n4l(lmax):])-1,
            # 'jac':lambda d: np.hstack((np.zeros((1, sh.n4l(lmax))), np.ones((1, d.shape[-1]-sh.n4l(lmax))))),
            }