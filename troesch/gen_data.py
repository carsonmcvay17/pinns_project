# imports
import numpy as np
from scipy.optimize import root_scalar

class GenData:
    """
    Class that generates data to solve the troesch problem
    """ 

    def true_sol(self, x, lam):
        """
        Reutrns the solution to the Troesch problem according to the paper 
        Solution approximated by polynomials
        y_5(x)= sx + 1/6 lambda^2sx^3 + (lambda^4s)/120(1+s^2)x^5
        Find parameter s by solving y_5(1)=1 for fixed lambda
        """
        x = np.array(x)
        
        def f(s):
            return s + (1/6)*lam**2*s + (lam**4*s/120)*(1+s**2) - 1
        
        sol = root_scalar(f, bracket=[1e-6, 2], method='brentq')
        s = sol.root

        y5 = s * x + (1/6) * (lam**2 * s * x**3) + ((lam**4 * s) / 120) * (1 + s**2) * x**5
        return y5
