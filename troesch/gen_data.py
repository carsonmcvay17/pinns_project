# imports
import numpy as np

class GenData:
    """
    Class that generates data to solve the troesch problem
    """ 

    def true_sol(self, x_colloc, s, lam):
        """
        Reutrns the solution to the Troesch problem according to the paper 
        Solution approximated by polynomials
        y_5(x)= sx + 1/6 lambda^2sx^3 + (lambda^4s)/120(1+s^2)x^5
        Find parameter s by solving y_5(1)=1 for fixed lambda
        """
        return s * x_colloc + 1/6 * (lam**2 * s * x_colloc**3) + ((lam**4 * s) / 120) * (1 + s**2) * x_colloc**5
