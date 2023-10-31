import opendp.prelude as dp
dp.enable_features("contrib")
import numpy as np
import math
from scipy.spatial import ConvexHull
from scipy.stats import norm
from scipy.special import logsumexp
from dp_accounting.pld.pld_pmf import create_pmf_pessimistic_connect_dots_fixed_gap
from _privacy_lib import PrivacyMechanism, SaddlePointAccountant, get_minima
import matplotlib.pyplot as plt

def get_lower_general(points):

    '''
    Input: points: [ [eps_1, delta_1], [eps_2, delta_2], ...]
    Assumes that all input epsilons are positive.

    Algorithm: Finds lower convex hull in e^eps space. This algortihm
    appends [eps_0 = 0, delta_0 = 1] to points, and removes it afterwards.
    This is to ensure compatability with CTD.

    Output: Lower convex hull. Points returned to eps space. The point at
    eps = -infinity is not outputed, but it is considered in the algorithm.
    '''

    if not np.all(points[:,0] > 0):
      raise ValueError(f'Input epsilons must be positive. '
                       f'Got: epsilons = {points[:,0]}')

    #turn eps into e^eps before taking Convex Hull
    points[:,0] = np.exp(points[:,0])
    #append e^eps = 0, delta = 1 to beginning
    points = np.vstack(( [0,1], points))

    #take Convex Hull
    hull = ConvexHull(points)

    #get polygon
    polygon = points[hull.vertices]

    #get lower Hull
    minx = np.argmin(polygon[:, 0])
    maxx = np.argmax(polygon[:, 0]) + 1
    if minx >= maxx:
        lower_curve = np.concatenate([polygon[minx:], polygon[:maxx]])
    else:
        lower_curve = polygon[minx:maxx]
    
    # Deal with non-decreasing input deltas
    slopes = np.diff(lower_curve[:,1])/np.diff(lower_curve[:,0])
    co = None
    for i in range(1,len(slopes)):
        if slopes[i] < slopes[i-1] or slopes[i]>0:
            co = i
            break
    if co:
        lower_curve = lower_curve[:co+1,:]
    if len(lower_curve) == 1:
        raise Exception("Invalid input")
    elif len(lower_curve) == 2:
        slope = np.diff(lower_curve[:,1])/np.diff(lower_curve[:,0])
        extrax = 1
        extray = float(np.abs(extrax - lower_curve[1,0]) * np.abs(slope) + lower_curve[1,1])
        lower_curve = np.vstack((lower_curve[0,:], np.array([extrax, extray]), lower_curve[1,:]))


    #remove e^eps = 0 , delta = 1 point, which is guarenteed
    #to be the left most point
    lower_curve = lower_curve[1:,:]

    #return to eps space
    lower_curve[:,0] = np.log(lower_curve[:,0])

    #return lower Convex Hull
    return lower_curve

def get_lower(points):

    '''
    Input: points: [ [eps_1, delta_1], [eps_2, delta_2], ...]
    Assumes that all input epsilons are positive.

    Algorithm: Finds lower convex hull in e^eps space. This algortihm
    appends [eps_0 = 0, delta_0 = 1] to points, and removes it afterwards.
    This is to ensure compatability with CTD.

    Output: Lower convex hull. Points returned to eps space. The point at
    eps = -infinity is not outputed, but it is considered in the algorithm.
    '''

    if not np.all(points[:,0] > 0):
      raise ValueError(f'Input epsilons must be positive. '
                       f'Got: epsilons = {points[:,0]}')

    #turn eps into e^eps before taking Convex Hull
    points[:,0] = np.exp(points[:,0])

    #append e^eps = 0, delta = 1 to beginning
    points = np.vstack(( [0,1], points))

    #take Convex Hull
    hull = ConvexHull(points)

    #get polygon
    polygon = points[hull.vertices]

    #get lower Hull
    minx = np.argmin(polygon[:, 0])
    maxx = np.argmax(polygon[:, 0]) + 1
    if minx >= maxx:
        lower_curve = np.concatenate([polygon[minx:], polygon[:maxx]])
    else:
        lower_curve = polygon[minx:maxx]

    #remove e^eps = 0 , delta = 1 point, which is guarenteed
    #to be the left most point
    lower_curve = lower_curve[1:,:]

    #return to eps space
    lower_curve[:,0] = np.log(lower_curve[:,0])

    #return lower Convex Hull
    return lower_curve


##### CODE FOR SPA ######
def get_pld_for_spa(lower_curve):
    '''

    Input: lower_curve: [ [eps_1, delta_1], [eps_2, delta_2], ...]
    the input is assumed to be the output of the get_lower() function.
    The code also concatenates the point [ eps_0 = - inf, delta_0 = 1 ] into
    the lower curve.

    Usage: This function does not assume that the eps_i values are on
    any grid, hence the support of the resulting PLD is not equally spaced.
    This means the FFT cannot be used, and only the saddlepoint can be.

    Description: Creates a dominating pair of distributions that represent the
    points given by lower_curve. In particular, it returns the privacy loss
    distribution of these points.

    Output: domain and probs, which correspond to the domain and PMF values
    of the PLD. Also returns probability at epsilon = infinity.

    Possible pitfall: Note that the returned PMF values are of the PLD, not the PLD
    conditioned on the PLD being finite. This means the probs do not add
    up to 1. The probs plus the probability at infinity sum to 1.

    Code adapted from
    https://github.com/google/differential-privacy/blob/main/python/dp_accounting/pld/pld_pmf.py#L561

    and from the equation above Section 4.1 in https://arxiv.org/pdf/2207.04380.pdf,
    which states that the "grid" that the PLD lives in is just the input
    epsilon values.
    '''

    epsilons = lower_curve[:,0]
    deltas = lower_curve[:,1]

    # epsilon_diffs = [eps_2 - eps_1, ... , eps_n - eps_{n-1}]
    epsilon_diffs = np.diff(epsilons)
    if np.any(epsilon_diffs <= 0):
      raise ValueError('epsilons are not in strictly increasing order: '
                      f'epsilons={epsilons}.')

    # Notation: deltas = [delta_1, ... , delta_n]
    if np.any(deltas < 0) or np.any(deltas > 1):
      raise ValueError(f'deltas are not between 0 and 1 : deltas={deltas}.')

    # delta_diffs = [delta_2 - delta_1, ... , delta_n - delta_{n-1}]
    delta_diffs = np.diff(deltas)
    if np.any(delta_diffs > 0):
      raise ValueError(f'deltas are not in non-increasing order: '
                      f'deltas={deltas}.')

    # delta_diffs_scaled_v1 = [y_0, y_1, ..., y_{n-1}]
    # where y_i = (delta_{i+1} - delta_i) / (exp(eps_i - eps_{i+1}) - 1)
    # and   y_0 = (1 - delta_1)
    delta_diffs_scaled_v1 = np.append(1 - deltas[0],
                                      delta_diffs / np.expm1(- epsilon_diffs))

    # delta_diffs_scaled_v2 = [z_1, z_2, ..., z_{n-1}, z_n]
    # where z_i = (delta_{i+1} - delta_i) / (exp(eps_{i+1} - eps_i) - 1)
    # and   z_n = 0
    delta_diffs_scaled_v2 = np.append(delta_diffs / np.expm1(epsilon_diffs), 0.0)

    # PLD contains eps_i with probability mass y_{i-1} + z_i, and infinity with
    # probability mass delta_n. Enforce that probabilities are non-negative.
    probs = np.maximum(0, delta_diffs_scaled_v1 + delta_diffs_scaled_v2)

    return epsilons, probs, deltas[-1]


def compute_delta_moments_accountant(prv,
                                     epsilon: float,
                                     compositions: int,
                                     ma_bounds = [1e-8,100]) -> float:
        '''
        Implemented to quickly compute the moments accountant epsilon.
        Note that "prv" is assumed to be the class used by the SaddlePointAccountant.
        '''
        def f_ma(t): return compositions*prv.cumulant(t) - epsilon * t

        t_ma = get_minima(f_ma, ma_bounds)
        delta_ma = math.exp(f_ma(t_ma))
        return delta_ma

##### CODE FOR FFT #####
def sample_evenly_on_convex_hull(lower_curve, discretization):

    '''
    Input: lower_curve: [ [eps_1, delta_1], [eps_2, delta_2], ...]
    the input is assumed to be the output of the get_lower() function.
    This means the points [[e^eps_1, delta_1], ...] form the lower
    convex hull of some collection of points.

    Description: Creates a new curve, i.e. a new list of epsilon, delta pairs
    [ [eps'_1, delta'_1], [eps'_2, delta'_2], ...] that are evenly spaced in eps space
    and lie on the piecewise-linear privacy curve spanned by the input lower_curve

    Output: integers  rounded_epsilon_lower, rounded_epsilon_upper and array
    new_deltas. This uniquely determines the eps' list such that the epsilons are
    equally spaced, i.e. eps'_{i+1} - eps'_i = discretization for all i.

    Details: The Connect the Dots code assumes the epsilons have a certain
    structure:

    eps'_i = int_i * discretization

    where int_i is some integer. In order to enforce this, this function
    finds the smallest and largest epsilon value in the convex hull,
    call it min_eps and max_eps. It then constructs the equally spaced epsilon
    grid by letting

    min_eps' = ceil( min_eps / discretization)
    max_eps' = floor( max_eps / discretization)

    this way, the resulting equally spaced grid is strictly inside the region
    spanned by the lower convex hull. It is worth mentioning that the Connect
    the Dots implementation flips the location of the floor and ceil:
    https://github.com/google/differential-privacy/blob/main/python/dp_accounting/pld/privacy_loss_distribution.py#L854C9-L857C74
    because in their implementation, they chose min_eps and max_eps to satisfy
    certain tail bounds. This means they want to make their equally spaced grid
    LARGER than their original grid. In contrast, we want to make it smaller,
    since we have no information on the privacy curve outside of the lower convex
    hull.
    '''

    original_deltas = lower_curve[:,1]
    original_epsilons= lower_curve[:,0]

    #create new alpha grid
    mini, maxy = np.min(original_epsilons), np.max(original_epsilons)

    rounded_epsilon_upper = math.floor(maxy/discretization)
    rounded_epsilon_lower = math.ceil(mini/discretization)

    new_epsilons = np.arange(rounded_epsilon_lower, rounded_epsilon_upper + 1) * discretization

    #digitize to find in which "bin" each alpha_i in new_alpha_grid belongs in,
    #where "bins" is each of the piecewise linear functions.
    #e.g. indices[7] = 2 means the point new_alpha_grid[7] is in the
    #second piecewise linear function spanned by old_alpha_grid.
    indices = np.digitize(new_epsilons, original_epsilons) - 1

    slopes = np.diff(original_deltas) / np.diff(original_epsilons)
    intercepts = original_deltas[1:] - slopes * original_epsilons[1:]

    new_deltas = slopes[indices] * new_epsilons + intercepts[indices]
    return rounded_epsilon_lower, rounded_epsilon_upper, new_deltas


def delta_exact(eps, sigma, compositions = 1):
    '''
    Input: list of epsilons [eps_1, ..., eps_n]
    Output: Exact deltas [delta_1, ..., delta_n]
    for the Gaussian mechaism with variacne sigma^2 under composition
    '''
    mu = np.sqrt(compositions)/sigma
    return norm.cdf(-eps/mu+mu/2)-np.exp(eps)*norm.cdf(-eps/mu-mu/2)