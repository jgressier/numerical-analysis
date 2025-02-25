import numpy as np

class _integrator():
    def __init__(self):
        pass
    
class _rungekutta(_integrator):
    """Explicit Runge Kutta integrators
    Butchers array is supposed to be strictly lower triangular
    """
    _butcher = np.array([[0.]])
    _weights = np.array([1.])

    def __init__(self):
        super().__init__()
        self._nstage = len(self._weights)
        assert self._butcher.shape == (self._nstage, self._nstage)
        # check explicit
        assert np.all(np.all(np.triu(self._butcher, k=0) == 0.))
        # check consistency
        assert np.isclose(np.sum(self._weights), 1.)

    def propagator(self, zarray):
        z = zarray.flatten()
        R = 1 + z[:, None] * np.dot(self._weights, np.linalg.inv(np.eye(self._nstage)-z[:, None, None]*self._butcher)@np.ones((self._nstage, 1)))
        return R.reshape(zarray.shape)

class rk3ssp(_rungekutta):
    _butcher = np.array([ 
        [0.,  0.,  0.],
        [1.,  0.,  0.],
        [.25, .25, 0.] ] )
    _weights = np.array([1.0, 1.0, 4.0]) / 6.0

class rk4(_rungekutta):
    """Classical 4th order RK"""
    _butcher = np.array([ 
        [0., 0., 0., 0.],
        [.5, 0., 0., 0.],
        [0., .5, 0., 0.],
        [0., .0, 1., 0.] ] )
    _weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0

# class rk2heun(_rungekutta):
#     """RK 2nd order Heun's method (or trapezoidal)"""
#     _butcher = [np.array([1.0]), np.array([0.5, 0.5])]

# class rk3heun(_rungekutta):
#     """RK 3rd order Heun's method """
#     _butcher = [
#             np.array([1.0 / 3.0]),
#             np.array([0, 2.0 / 3.0]),
#             np.array([0.25, 0, 0.75])     ]

# --------------------------------------------------------------------
# LOW STORAGE RUNGE KUTTA MODELS
# --------------------------------------------------------------------

class _LSrungekutta(_rungekutta):
    """generic implementation of LOW-STORAGE Runge-Kutta method

    Hu and Hussaini (JCP, 1996) method needs p-1 coefficients (_beta)
    needs specification of Butcher array from derived class
        $ for 1<=s<=p, Qs = Q0 + dt * _beta_s RHS(Q_{s-1}) $
    """
    _beta = np.array([1.])

    def __init__(self):
        self._butcher = np.diag(self._beta[:-1], k=-1)
        self._weights = np.zeros(len(self._beta))
        self._weights[-1] = self._beta[-1]
        super().__init__()

class lsrk25bb(_LSrungekutta):
    """Low Storage implementation of Bogey Bailly (JCP 2004) 2nd order 5 stages Runge Kutta """
    _beta = [ 0.1815754863270908, 0.238260222208392, 0.330500707328, 0.5, 1. ]

class lsrk26bb(_LSrungekutta):
    """Low Storage implementation of Bogey Bailly (JCP 2004) 2nd order 6 stages Runge Kutta """
    _beta = [  0.11797990162882 , 0.18464696649448 , 0.24662360430959 , 0.33183954253762 , 0.5, 1. ]

class lsrk4(_LSrungekutta):
    """RK4 to check"""
    _beta = [ 1./4. , 1./3. , 0.5, 1. ]


all_lsrk = [ lsrk25bb, lsrk26bb, lsrk4 ]
all_erk = [ rk3ssp, rk4] + all_lsrk
all_rk = all_erk
all_explicit = all_erk
