import math
from scipy.optimize import minimize_scalar, root_scalar
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import math
import numpy as np
from sympy import bell
from scipy.stats import norm
import mpmath as mp
from sympy import bell, symbols, lambdify
from scipy.special import logsumexp


## Code from SP Accountant https://github.com/Felipe-Gomez/saddlepoint_accountant/tree/main

ArrayLike = Union[np.ndarray, List[float]]

class PrivacyMechanism(ABC):

    def log_p(self, z: float) -> float:
        """
        Compute the log probability density function of the privacy random variable at point z
        conditioned on the value being finite.

        Args:
        z: point to evaluate the log probability density

        Returns:
        Log probability density
        """
        raise NotImplementedError(f"{type(self)} has not provided an implementation for a pdf.")

    def privacy_loss_without_subsampling(self, x: float) -> float:
        """
        Computes the privacy loss, i.e. P(x-sensitivity) / P(x)  at a given point without sub-sampling.

        Args:
        x: Point at which to evaluate the privacy loss at.

        Returns:
        Privacy loss at x
        """
        
        raise NotImplementedError(f"{type(self)} has not provided an implementation for privacy loss without subsampling.")

    def privacy_loss(self, x: float) -> float:
        """
        Computes the privacy loss at a given point.
        With sub-sampling probability of lambda < 1: the privacy loss at x is
        log(1 - lambda + lambda *exp(privacy_loss_without_subsampling(x)))

        Args:
        x: Point at which to evaluate the subsampled privacy loss at.

        Returns:
        Subsampled privacy loss at x
        """
        privacy_loss_without_subsampling = self.privacy_loss_without_subsampling(x)

        # For performance, the case of sampling probability = 1
        # is handled separately.
        if self._lambda == 1.0:
          return privacy_loss_without_subsampling
        
        return log_a_times_exp_b_plus_c(self._lambda,
                                                privacy_loss_without_subsampling,
                                                1 - self._lambda)
    
    def diff_j_cumulant(self,
                        t: float,
                        j: int) -> ArrayLike:
        '''
        Let K(t) be the cumulant and M(t) = e^{K(t)} be the moment generating function. 
        This function returns K(t), K'(t), K''(t), ... K^(j)(t)

        This function uses the abtract method self.diff_j_MGF to compute
        M(t), M'(t), ... M^(j)(t)
        then uses these values, along with Faà di Bruno's formula,
        which uses the Bell polynomials, to compute the value of the
        derivative of the cumulant.

        Args:
        t: Value at which to compute the cumulant and its j derivatives at
        j: Integer denoting how many derivatives of the cumulant to compute

        Returns:
        K(t), K'(t), K''(t), ... K^(j)(t)
        '''
        cumulanty = self.cumulant(t)
        mgf_derivatives = [self.diff_j_MGF(t,kk) for kk in range(1,j+1)]
    
        output_lst = [cumulanty]
        mgf = math.exp(cumulanty)

        for kk in range(1,j+1):

            output = 0
            for jj in range(1,kk+1):

                factor = (2*(jj%2)-1) * math.factorial(jj-1) / math.pow(mgf,jj)
                bell_term = bell(int(kk), jj, mgf_derivatives[0:int(kk-jj+1)])
                output += factor * bell_term
            output_lst.append(output)

        return np.array(output_lst, dtype = float)
    

    def get_integration_bounds(self) -> Tuple[float, float]:
        """
        Only used for mechanisms which use numerical integration to compute the
        moment generating function and its derivatives, e.g. for the Gaussian mechanism.
        """
        raise NotImplementedError(f"{type(self)} has not provided an implementation for getting the integration bounds.")


    @abstractmethod
    def cumulant(self, t):
        """
        Compute the cumulant of this privacy random variable at point t
        conditioned on the value being finite.
        """
        pass
    
    @abstractmethod
    def diff_j_MGF(self, t, j):
        """
        Compute the jth derivative of the moment generating function
        of this privacy random variable at point t conditioned on the value being finite.
        """
        pass
    
    @abstractmethod
    def Pt(self, t, norm, shift):
        """
        Compute the tensorized absolute third moment of this privacy
        random variable with tilting parameter t
        """



class SaddlePointAccountant:

    def __init__(self, 
                 mechanism: PrivacyMechanism) -> None:
        self.mechanism = mechanism

    def diff_j_F(self,
                 t: float,
                 j: int,
                 compositions: int,
                 epsilon: float) -> ArrayLike:
        '''
        Returns F_eps(t), F_eps'(t) , ..., F_eps^(j)(t), where
        F_eps is as defined in Section 3.1 of ICML paper. 

        Args:
        t: point at which to compute F_eps(t) and its first j derivatives
        j: interger denoting the number of derivatives of F_eps to take

        Returns:
        Python list of [F_eps(t), F_eps'(t) , ..., F_eps^(j)(t)]
        '''
        K_lst = self.mechanism.diff_j_cumulant(t,j) #K(t), K'(t), ... K^j(t)
        
        #handle the derivatives of log(t) and log(t+1)
        second_lst = [(-2*(kk%2)+1) * math.factorial(kk-1) * (1/t**kk + 1/(t+1)**kk) for kk in range(1,j+1)]
        
        #handle F_eps(t) and F_eps'(t), both of which have unique terms 
        second_lst = [-epsilon*t -math.log(t) - math.log1p(t)] + second_lst #F_eps(t) term
        second_lst[1] += -epsilon #add missing epsilon term to F_eps'(t)
        
        output = compositions*K_lst + np.array(second_lst)
        return output
    
    def compute_delta_msd(self,
                          epsilon: float,
                          compositions: int,
                          k: int = 1,
                          spa_bounds: Tuple[float,float] = [1e-3,100]) -> float:
        '''
        Computes the order-k method-of-steepest-descent saddle-point approximation
        as defined in Section 3.3 of ICML paper.

        Warning: Using larger values of k will yield better approximations
        only after a sufficiently high number of compositions. Running k > 2
        for a ~1000 compositions may lead to numerical error.   

        Args:
        
        epsilon: privacy parameter under (epsilon, delta)-DP
        
        compositions: number of compositions. In DP-SGD, this corresponds
        to the number of iterations in the optimization.

        k: The order of the saddlepoint approximation. Typcially either
        1, 2, or 3.

        spa_bounds: A guess on the location of the saddlepoint.
        The code uses bounded binary search over this region to
        to find the saddlepoint value.
        
        Returns:
        
        An approximation to the privacy paraemter delta using the
        method-of-steepest-descent
        '''
 
        def feps(t): return compositions*self.mechanism.cumulant(t) \
                            - epsilon*t  - math.log(t) - math.log1p(t)

        spa = get_minima(feps, spa_bounds)

        #compute [ F_eps(spa), F_eps'(spa), ... F_eps^{2k}(spa) ] 
        flst = self.diff_j_F(spa, 2*k, compositions, epsilon)
        
        #delta_correction corresponds to the term in parenthesis in equation 24 of ICML paper
        delta_correction = 1
        
        #update the spa_correction term if k != 1
        if k != 1:
            betas = np.array( [1] + [beta(flst[:2*m+1]) for m in range(2,k+1)])
            summed_betas = sum(betas)
            delta_correction = summed_betas
            
        delta_msd1 = math.exp(flst[0]) / math.sqrt(2 * np.pi * math.fabs(flst[2]))

        return delta_msd1 * delta_correction 
    
    def compute_epsilon_msd(self,
                            delta: float,
                            compositions: int,
                            k: int = 1,
                            spa_bounds: Tuple[float,float] = [1e-3, 100],
                            eps_lower: float = 1e-10) -> float:
        '''
        Use binary search to invert compute_delta_msd(). 
        The moments accountant epsilon is used as an upper bound on
        epsilon, and a small value eps_lower is used as a lower bound.

        Args:

        delta: privacy parameter under (epsilon, delta)-DP
        
        compositions: number of compositions. In DP-SGD, this corresponds
        to the number of iterations in the optimization.

        k: The order of the saddlepoint approximation. Typcially either
        1, 2, or 3.

        spa_bounds: A guess on the location of the saddlepoint.
        The code uses bounded binary search over this region to
        to find the saddlepoint value.

        eps_lower: A guess on a lower bound on epsilon. Can be
        arbitrarily small, as long as it is greater than 0. 
        '''
        eps_upper = self.compute_epsilon_moments_accountant(delta, compositions, ma_bounds = spa_bounds)
        def get_delta_msd(epsilon): return self.compute_delta_msd(epsilon, compositions, k, spa_bounds)
        return root_scalar(lambda epsilon: get_delta_msd(epsilon) - delta, bracket = [eps_lower, eps_upper]).root

    def compute_epsilon_moments_accountant(self,
                                           delta: float,
                                           compositions: int,
                                           ma_bounds: Tuple[float,float] = [1e-3,100]) -> float:
        
        def f_ma(t): return (compositions*self.mechanism.cumulant(t) + math.log(1/delta)) / t
        
        t_ma = get_minima(f_ma, ma_bounds)
        eps_ma = f_ma(t_ma)
        return eps_ma
    
    def compute_delta_clt(self,
                          epsilon: float,
                          compositions: int,
                          spa_bounds: Tuple[float,float] = [1e-3,100]) -> Tuple[float,float]:
        '''
        The CLT version of the saddle-point accountant
        
        Returns both a hybrid saddleppoint / tilted CLT approximation
        to delta along with the error associated with the approximation

        Args:
        
        epsilon: privacy parameter under (epsilon, delta)-DP
        
        compositions: number of compositions. In DP-SGD, this corresponds
        to the number of iterations in the optimization.

        spa_bounds: A guess on the location of the saddlepoint.
        The code uses bounded binary search over this region to
        to find the saddlepoint value.
        
        Returns:
        
        An approximation to the privacy parameter delta using a
        hybrid tilted CLT / saddlepoint approach, along with
        the error of the approximation.
        '''
        
        def feps(t): return compositions*self.mechanism.cumulant(t)\
                            - epsilon*t  - math.log(t) - math.log1p(t)
        
        spa = get_minima(feps, spa_bounds)
        
        #log_q_clt is the logarithm of q(z) as defined in equation 3 of ICML paper
        rv = norm() 
        log_q_clt = lambda z: 0.5 * math.log(2*np.pi) + rv.logsf(z) + z*z/2

        #grab K(spa), K'(spa), K''(spa)
        K0, K1, K2 = self.mechanism.diff_j_cumulant(spa,2)
        
        #compute constants alpha,beta,gamma as defined in Proposition 5.3
        gamma = (compositions*K1 - epsilon)/math.sqrt(compositions*K2)
        alpha = spa*math.sqrt(compositions*K2) - gamma
        beta = alpha + math.sqrt(compositions*K2)
        
        #compute log ( q(alpha) - q(beta) ) in a float point stable manner
        log_q_diff = _log_sub(log_q_clt(alpha), log_q_clt(beta))
        
        #compute delta clt
        delta_clt = math.exp(compositions*K0 - epsilon*spa - gamma*gamma/2 + log_q_diff) / math.sqrt(2*np.pi)
        
        #compute error term
        pt = self.mechanism.Pt(spa, math.exp(K0), K1) 
        err_clt = math.exp(compositions*K0 - epsilon*spa) * spa**spa / (1+spa)**(1+spa) *\
                    1.12 * compositions * pt / (compositions*K2)**(3/2)
        return (delta_clt, err_clt)
    

    def compute_epsilon_clt(self,
                            delta: float,
                            compositions: int,
                            spa_bounds: Tuple[float,float] = [1e-3, 100],
                            eps_lower: float = 1e-10) -> ArrayLike:
        
        '''
        Use binary search to find epsilon as a function of delta. 
        The moments accountant epsilon is used as an upper bound on
        epsilon, and a small value eps_lower is used as a lower bound.

        Args:
        
        delta: privacy parameter under (epsilon, delta)-DP
        
        compositions: number of compositions. In DP-SGD, this corresponds
        to the number of iterations in the optimization.


        spa_bounds: A guess on the location of the saddlepoint.
        The code uses bounded binary search over this region to
        to find the saddlepoint value.

        eps_lower: A guess on a lower bound on epsilon. Can be
        arbitrarily small, as long as it is greater than 0.

        Returns:
        
        A lower bound, approximation, and upper bound  to the
        privacy parameter epsilon using a hybrid tilted CLT / saddlepoint approach.

        This returns a python list [ eps_lower, eps_approximation, eps_upper]
        '''
        eps_upper = self.compute_epsilon_moments_accountant(delta, compositions, ma_bounds = spa_bounds)
        
        def get_delta_clt(epsilon, bound):
            delta_clt, err_clt = self.compute_delta_clt(epsilon, compositions, spa_bounds)
            if bound == 'lower': return delta_clt - err_clt
            elif bound == 'none': return delta_clt
            elif bound == 'upper': return delta_clt + err_clt
        
        epsilon_clt_lower = root_scalar(lambda epsilon: get_delta_clt(epsilon, 'lower') - delta, bracket = [eps_lower, eps_upper]).root
        epsilon_clt = root_scalar(lambda epsilon: get_delta_clt(epsilon, 'none') - delta, bracket = [eps_lower, eps_upper]).root
        epsilon_clt_upper = root_scalar(lambda epsilon: get_delta_clt(epsilon, 'upper') - delta, bracket = [eps_lower, eps_upper]).root
        return [epsilon_clt_lower, epsilon_clt, epsilon_clt_upper]


class CTD_Mechanism(PrivacyMechanism):

    '''
    Todo: Move this function to the SPA github

    Todo: Implement subsampling by using Gopi's
    approach: PLD_subsampled(l) = q PLD_Y(l) + (1-q) PLD_X(l)
    Need to figure out how to get both PLDs from the Connect the Dots
    approach.
    '''
    def __init__(self,
                 sampling_probability: float,
                 grid: ArrayLike,
                 probs: ArrayLike,
                 inf_mass: float) -> None:

        grid = np.asarray(grid)
        probs = np.asarray(probs)

        if grid.shape != probs.shape:
          raise ValueError('Length of grid and pmf values are unequal '
                     f'grid={grid}, '
                     f'probs={probs}.')

        if inf_mass > 1 or inf_mass < 0:
          raise ValueError(f'Probability mass at infinity is either larger than 1'
                           f' or smaller than 0: inf_mass={inf_mass}')

        #check if probs sum to 1. If they do not, but probs + inf mass sum to 1,
        #then
        if not np.isclose( math.fabs(1 - sum(probs) ) ,  np.finfo(float).eps) and np.isclose(math.fabs(1 - sum(probs) - inf_mass ) , np.finfo(float).eps):
          probs = probs / sum(probs)
        else:
          raise ValueError(f'Either probs + inf_mass do not sum to 1, or probs sum to 1'
                            f'sum probs = {sum(probs)}, sum probs + inf_mass = {sum(probs) + inf_mass}')


        self.grid = grid
        self.probs_for_finite_L = probs
        self.inf_mass = inf_mass


    def cumulant(self, t):
        """
        Compute the cumulant of this privacy random variable at point t
        conditioned on the value being finite.
        """
        return logsumexp(self.grid * t, b=self.probs_for_finite_L)

    def diff_j_MGF(self, t, j):
        """
        Compute the jth derivative of the moment generating function
        of this privacy random variable at point t conditioned on the value being finite.
        """
        log_diff_j_mgf = logsumexp( self.grid * t, b=self.probs_for_finite_L * np.power(self.grid,j) )
        return np.exp(log_diff_j_mgf)

    def Pt(self, t, norm, shift):
        """
        Compute the tensorized absolute third moment of this privacy
        random variable with tilting parameter t
        """

        log_unormed_pt = logsumexp(self.grid * t, b=self.probs_for_finite_L * np.abs(self.grid-shift)**3)
        return np.exp(log_unormed_pt) / norm



def _log_add(logx: float, logy: float) -> float:
    """Adds two numbers in the log space."""
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
    """Subtracts two numbers in the log space. Answer must be non-negative."""
    if logx < logy:
        raise ValueError('The result of subtraction must be non-negative.')
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx

def log_a_times_exp_b_plus_c(a: float, b: float, c: float) -> float:
    """Computes log(a * exp(b) + c)."""
    if a == 0:
        return math.log(c)
    if a < 0:
        if c <= 0:
            raise ValueError(f'a exp(b) + c must be positive: {a}, {b}, {c}.')
        return _log_sub(math.log(c), math.log(-a) + b)
    if b == 0:
        return math.log(a + c)
    d = b + math.log(a)
    if c == 0:
        return d
    elif c < 0:
        return _log_sub(d, math.log(-c))
    else:
        return _log_add(d, math.log(c))

def get_minima(function, bounds):
    '''
    Search for the minima of "function" via binary search over "bounds".

    This function is used to compute the saddlepoint of F_eps, as the saddlepoint 
    is a minima along the real axis, and to find the optimal "order" for the moments accountant. 

    '''
    minima = minimize_scalar(function, bounds = bounds, method = 'bounded').x

    #if the saddlepoint is close to the endpoints, throw an error
    if math.isclose(minima, bounds[0]):
        raise RuntimeError("saddlepoint is close to lower bound, try decreasing the default lower bound in spa_bounds")
    if math.isclose(minima, bounds[1]):
        raise RuntimeError("saddlepoint is close to upper bound, try increasing the default upper bound in spa_bounds")
    return minima

def complete_bell(n):
    '''
    computes the nth complete exponential Bell polynomial in sympy
    and outputs a numpy lambda function that evaluates it
    '''
    out = 0
    for k in range(1,n+1):
        stry = str(n-k+2)
        out += bell(n,k,symbols('x:'+stry)[1:])
    stry = str(n + 1)
    return lambdify(symbols('x:'+stry)[1:], out, 'numpy')

def beta(flst: ArrayLike) -> float:
    '''
    Computes equation 22 of ICML paper, i.e. beta_{epsilon, m}
    These can be thought of as the higher order corrections to the saddlepoint
        
    input:
        
    flst: output from diff_j_F evaluated at the saddlepoint, i.e.
    a numpy array of [F_eps(spa), F_eps'(spa), F_eps''(spa), F_eps^3(spa) ... F_eps^{2m}(spa)]
        
    output: beta_{epsilon, m} = (-1)^m Beta((0,0 F_eps^3(spa) ... F_eps^{2m}(spa)) / 2^m m! F_eps''(spa)^m
    '''
    
    assert float( ( len(flst) - 1 ) / 2).is_integer()
        
    m = int( ( len(flst) - 1 ) / 2 )
    input_lst = [0,0] + flst[3:].tolist() #0, 0, F^3(spa) ... F^{2m}(spa)

    return (-2*(m%2) + 1) * complete_bell(2*m)(*input_lst) / (math.pow(2,m) * math.factorial(m) * math.pow(flst[2],m))
