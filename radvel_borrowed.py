## Radvel functionality that I need for completeness work
## I don't want the occurrence code to depend on RVSearch or radvel, so I'm bringing this code here

# Normalization.
# RV m/s of a 1.0 Jupiter mass planet tugging on a 1.0
# solar mass star on a 1.0 year orbital period

import numpy as np
from astropy import constants as c
from astropy import units as u
from scipy.optimize import root
import warnings

K_0 = 28.4329

def Msini(K, P, Mstar, e, Msini_units='earth'):
    """Calculate Msini

    Calculate Msini for a given K, P, stellar mass, and e

    Args:
        K (float or array: Doppler semi-amplitude [m/s]
        P (float or array): Orbital period [days]
        Mstar (float or array): Mass of star [Msun]
        e (float or array): eccentricity
        Msini_units (Optional[str]): Units of Msini {'earth','jupiter'}
            default: 'earth'

    Returns:
        float or array: Msini [units = Msini_units]

    """

    # convert inputs to array so they work with units
    P = np.array(P)
    Mstar = np.array(Mstar)
    K = np.array(K)
    e = np.array(e)
    G = c.G.value                # added gravitational constant
    Mjup = c.M_jup.value         # added Jupiter's mass
    Msun = c.M_sun.value         # added sun's mass
    Mstar = Mstar*Msun
    Mstar = np.array(Mstar)

    P_year = (P * u.d).to(u.year).value
    P = (P * u.d).to(u.second).value

    # First assume that Mp << Mstar
    Msini = K / K_0 * np.sqrt(1.0 - e ** 2.0) * (Mstar/Msun) ** (2.0 / 3.0) * P_year ** (1 / 3.0)

    # Use correct calculation if any elements are >10% of the stellar mass
    
    if (np.array(((Msini * u.Mjup).to(u.M_sun) / (Mstar/Msun)).value > 0.10)).any():
        # with warnings.catch_warnings():
        #     warnings.simplefilter("always")
        #     warnings.warn("Mpsini << Mstar assumption broken, correcting Msini calculation.", stacklevel=2)

        a = K*(((2*(np.pi)*G)/P)**(-1/3.))*np.sqrt(1-(e**2))
        Msini = []
        if isinstance(P, float):
            n_elements = 1
        else:
            assert type(K) == type(P) == type(Mstar) == type(e), "All input data types must match."
            assert K.size == P.size == Mstar.size == e.size, "All input arrays must have the same length."
            n_elements = len(P)
        for i in range(n_elements):
            def func(x):
                try:
                    return x - a[i]*((Mstar[i]+x)**(2/3.))
                except IndexError:
                    return x - a * ((Mstar + x) ** (2 / 3.))

            sol = root(func, Mjup)
            Msini.append(sol.x[0])

        Msini = np.array(Msini)
        Msini = Msini/Mjup
    
    if Msini_units.lower() == 'jupiter':
        pass
    elif Msini_units.lower() == 'earth':
        Msini = (Msini * u.M_jup).to(u.M_earth).value
    else:
        raise Exception("Msini_units must be 'earth', or 'jupiter'")

    return Msini


def semi_major_axis(P, Mtotal):
    """Semi-major axis

    Kepler's third law

    Args:
        P (float): Orbital period [days]
        Mtotal (float): Mass [Msun]

    Returns:
        float or array: semi-major axis in AU
    """

    # convert inputs to array so they work with units
    P = np.array(P)
    Mtotal = np.array(Mtotal)

    Mtotal = Mtotal*c.M_sun.value
    P = (P * u.d).to(u.second).value
    G = c.G.value
    
    a = ((P**2)*G*Mtotal/(4*(np.pi)**2))**(1/3.)
    a = a/c.au.value

    return a