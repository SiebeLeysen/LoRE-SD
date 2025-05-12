import numpy as np
from lore_sd.utils.SphericalHarmonics import sphconv, modshbasis



def getOrientedApodizedDirac(theta, phi, lmax=8):
    """
    See https://onlinelibrary.wiley.com/doi/10.1002/mrm.23058
    To get an apodized dirac delta along an arbitrary direction, the apodized dirac along the
    z-direction is spherically convolved with a standard dirac pointing along
    the orientation of interest.
    """
    oriented_dirac = genPSF(theta, phi, lmax)
    apodized_dirac = getapocoefs(lmax)
    res = sphconv(np.expand_dims(apodized_dirac, axis=0), oriented_dirac)
    return res

def genPSF(theta, phi, lmax=8):
    """
    Point Distribution Function: Dirac Delta in a specified direction.
    """
    theta_arr = np.array(np.radians(theta))
    phi_arr = np.array(np.radians(phi))
    return modshbasis(lmax, theta_arr, phi_arr)[0]

def getapocoefs(L):
    # Hard coded apodized delta coefficients. Let's pray that cython makes these static.
    if L == 2:
        return np.array([0.2820947918, 0.2645459007])
    elif L == 4:
        return np.array([0.2820947918, 0.4012319618, 0.1564533344])
    elif L == 6:
        return np.array([0.2820947918, 0.4761803401, 0.3141952797, 0.09779180595])
    elif L == 8:
        return np.array([0.2820947918, 0.5196695259, 0.4338197298, 0.2282445903, 0.06505347349])
    elif L == 10:
        return np.array([0.2820947918, 0.547052636, 0.5206293826, 0.3516207639, 0.166328382, 0.04586653939])
    elif L == 12:
        return np.array([0.2820947918, 0.5660506469, 0.5862931541, 0.4602398011, 0.2811219643, 0.1270249528, 0.03529685691])
    elif L == 14:
        return np.array([0.2820947918, 0.5785407495, 0.6320529393, 0.5438245935, 0.3841957934, 0.2199412472, 0.09605546497, 0.02665962304])
    elif L == 16:
        return np.array([0.2820947918, 0.5882758938, 0.6691135457, 0.6159894424, 0.4821624075, 0.3216386667, 0.1785835337, 0.07757696932, 0.02212700756])
    elif L == 18:
        return np.array([0.2820947918, 0.5950010962, 0.6954831457, 0.6698973538, 0.5607479358, 0.4118518254, 0.2630266557, 0.1420872623, 0.06099824937, 0.01753724896])
    elif L == 20:
        return np.array([0.2820947918, 0.5998215552, 0.7147702826, 0.710641902, 0.6230508551, 0.4882538936, 0.3413011721, 0.2099196543, 0.1102854613, 0.04647214808, 0.01325299126])
    else:
        return None
