import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.legendre import legfit
from scipy.special import sph_harm


def calcsig(dwi, mask, grad, lmax=8):
    """
    Multi-shell Diffusion Weighted Imaging (DWI) fit in the Spherical Harmonics (SH) basis.

    Parameters:
        dwi (numpy.ndarray): The input DWI data.
        mask (numpy.ndarray): A binary brain mask.
        grad (numpy.ndarray): The gradient directions and magnitudes (b-values) (gradient table).
        lmax (int): The maximum order of the spherical harmonics.

    Returns:
        numpy.ndarray: The fitted signal in the SH basis.
    """

    # If DWI is a single voxel, expand to (1, #dirs)
    if len(dwi.shape) == 1:
        dwi_masked = np.expand_dims(dwi, axis=0)
    else:
        dwi_masked = dwi[mask, :]

    # Round the gradient table to the nearest multiple of 10
    grad_round = np.concatenate((grad[:, :3], np.expand_dims(np.round(grad[:, 3], -1), axis=1)), axis=1)
    bvals = np.unique(grad_round[:, 3])

    nlmax = n4l(lmax)

    # Transformation matrix between polar coordinates and SH
    Q = modshbasiscart(lmax, grad_round[:, 0], grad_round[:, 1], grad_round[:, 2])
    n = [n4l(l) for l in range(0, lmax + 1, 2)]
    S = np.zeros((dwi_masked.shape[0], len(bvals), nlmax))

    # We fit every shell individually
    for k, b in enumerate(bvals):
        bidx = grad_round[:, 3] == b
        nn = min([nlmax] + [a for a in n if a >= np.sum(bidx)]) if b > 10.0 else 1
        pinvQ = np.linalg.pinv(Q[bidx, :nn])
        S[..., k, :nn] = np.dot(dwi_masked[..., bidx], pinvQ.T)

    res = np.zeros(dwi.shape[:-1] + (len(bvals), nlmax,))

    if len(dwi.shape) == 1:
        return S

    res[mask] = S
    return res


import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.legendre import legfit
from scipy.special import sph_harm


def calcsig(dwi, mask, grad, lmax=8):
    """
    Multi-shell Diffusion Weighted Imaging (DWI) fit in the Spherical Harmonics (SH) basis.

    Parameters:
        dwi (numpy.ndarray): The input DWI data.
        mask (numpy.ndarray): A binary brain mask.
        grad (numpy.ndarray): The gradient directions and magnitudes (b-values) (gradient table).
        lmax (int): The maximum order of the spherical harmonics.

    Returns:
        numpy.ndarray: The fitted signal in the SH basis.
    """

    # If DWI is a single voxel, expand to (1, #dirs)
    if len(dwi.shape) == 1:
        dwi_masked = np.expand_dims(dwi, axis=0)
    else:
        dwi_masked = dwi[mask, :]

    # Round the gradient table to the nearest multiple of 10
    grad_round = np.concatenate((grad[:, :3], np.expand_dims(np.round(grad[:, 3], -1), axis=1)), axis=1)
    bvals = np.unique(grad_round[:, 3])

    nlmax = n4l(lmax)

    # Transformation matrix between polar coordinates and SH
    Q = modshbasiscart(lmax, grad_round[:, 0], grad_round[:, 1], grad_round[:, 2])
    n = [n4l(l) for l in range(0, lmax + 1, 2)]
    S = np.zeros((dwi_masked.shape[0], len(bvals), nlmax))

    # We fit every shell individually
    for k, b in enumerate(bvals):
        bidx = grad_round[:, 3] == b
        nn = max([a for a in n if a < np.sum(bidx)]) if b > 10.0 else 1
        pinvQ = np.linalg.pinv(Q[bidx, :nn])
        S[..., k, :nn] = np.dot(dwi_masked[..., bidx], pinvQ.T)

    res = np.zeros(dwi.shape[:-1] + (len(bvals), nlmax,))

    if len(dwi.shape) == 1:
        return S

    res[mask] = S
    return res


def calcdwi(sh, grad):
    """
    This function performs a multi-shell Diffusion Weighted Imaging (DWI) fit in the Spherical Harmonics (SH) basis.

    Parameters:
        sh (numpy.ndarray): The input data in the SH basis.
        grad (numpy.ndarray): The gradient directions and b-values.

    Returns:
        numpy.ndarray: The fitted signal in the original space.
    """
    nlmax = sh.shape[-1]
    lmax = l4n(nlmax)

    # Round the gradient table to the nearest multiple of 10
    grad_round = np.concatenate((grad[:, :3], np.expand_dims(np.round(grad[:, 3], -1), axis=1)), axis=1)
    bvals = np.unique(grad_round[:, 3])

    n = [n4l(l) for l in range(0, lmax + 1, 2)]

    flat_sh = sh.reshape((-1, sh.shape[-2], sh.shape[-1]))

    Q = modshbasiscart(lmax, grad_round[:, 0], grad_round[:, 1], grad_round[:, 2])

    flat_S = np.zeros((flat_sh.shape[0], grad_round.shape[0]))

    # We fit every shell individually
    for k, b in enumerate(bvals):
        bidx = grad_round[:, 3] == b
        nn = max([a for a in n if a < np.sum(bidx)]) if b > 10.0 else 1
        flat_S[..., bidx] = np.dot(flat_sh[..., k, :nn], Q[bidx, :nn].T)
    return flat_S.reshape(sh.shape[:-2] + (grad_round.shape[0],))


def c2s(x, y, z):
    '''
    Converts cartesian to spherical coordinates.
    '''
    r = np.hypot(x, y)
    np.hypot(r, z, out=r)
    # Ignore any runtime warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        theta = np.arccos(z / r)
        phi = np.pi + np.arctan2(-y, -x)
    return (theta, phi, r)


def s2c(theta, phi, r):
    '''
    Converts spherical to cartesian coordinates.
    '''
    x = r * np.cos(phi) * np.sin(theta);
    y = r * np.sin(phi) * np.sin(theta);
    z = r * np.cos(theta);
    return (x, y, z)


def n4l(L):
    '''
    Returns the number of components in the SH basis of order L.
    '''
    return (L + 1) * (L + 2) // 2


def l4n(R):
    '''
    Returns the order of the SH basis, given the number of coefficients.
    '''
    return int(np.sqrt(1 + 8 * R) - 3) // 2


def zh2rh(h):
    '''
    Convert Zonal Harmonics to Rotational Harmonics.
    '''
    lmax = 2 * (h.shape[-1] - 1)
    z2r = np.zeros((h.shape[-1], n4l(lmax)))
    for l in range(0, lmax + 1, 2):
        j1, j2 = n4l(l - 2), n4l(l)
        z2r[l // 2, j1:j2] = np.sqrt(4 * np.pi / (2 * l + 1))
    return np.einsum('...i,ij->...j', h, z2r)


def rh2zh(kernel):
    '''
    Convert Rotational Harmonics to Zonal Harmonics.
    '''
    nlmax = kernel.shape[-1]
    lmax = l4n(nlmax)
    idx = 0
    idx_list = []
    scaling = [1 / np.sqrt(4 * np.pi / (2 * l + 1)) for l in range(0, lmax + 1, 2)]
    for l in range(0, lmax + 1, 2):
        idx_list.append(idx)
        idx += 2 * l + 1
    return scaling * kernel[:, idx_list]


def modshbasis(L, theta, phi):
    '''
    Modified SH basis for spherical coordinates (theta, phi).
    Only even order SH functions are returned.
    '''
    assert theta.size == phi.size;
    out = np.zeros((theta.size, n4l(L)))
    rt2 = np.sqrt(2.0)
    for l in range(0, L + 1, 2):
        c = l * (l + 1) // 2
        out[:, c] = np.real(sph_harm(0, l, phi, theta))
        for m in range(1, l + 1):
            sh = sph_harm(m, l, phi, theta)
            out[:, c + m] = rt2 * np.real(sh)
            out[:, c - m] = rt2 * np.imag(sh)
    return out


def modshbasiscart(L, x, y, z):
    '''
    Modified SH basis for cartesian coordinates (x, y, z).
    '''
    theta, phi, r = c2s(x, y, z)
    return modshbasis(L, theta, phi)


def sphconv(h, f):
    '''
    Perform spherical convolution of response function and odf.
    The response function is assumed to be zonal.
    '''
    return np.einsum('...ij,...j->...ij', zh2rh(h), f)

def angularCorrelation(sh1, sh2):
    """
    Angular correlation coefficient (Anderson, 2005) between two SH representations.
    :param sh1:
    :param sh2:
    :return:
    """
    assert sh1.shape[-1] == sh2.shape[-1], f'Last dimension of SH functions do not match: sh1 has shape {sh1.shape}, sh2 has shape {sh2.shape}'
    # Suppress the runtime warning
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.einsum('...b, ...b->...', sh1[...,1:], sh2[...,1:])
        denominator_1 = np.linalg.norm(sh1[...,1:], axis=-1)
        denominator_2 = np.linalg.norm(sh2[...,1:], axis=-1)
        res = np.true_divide(numerator, denominator_1*denominator_2)
        res = np.nan_to_num(res, nan=0, posinf=0, neginf=0)
    return res

def expcoefs(a, lmax):
    """
    Computes the exponential coefficients for a given array 'a' and maximum degree 'lmax'.

    Parameters:
    a (array-like): Input array of values.
    lmax (int): Maximum degree.

    Returns:
    array: Exponential coefficients computed using the input array and maximum degree.

    """
    x = np.linspace(-1, 1, 10 * (lmax // 2 + 1) + 1)[1:]
    y = np.exp(np.outer(-a, x ** 2))
    return legfit(x, y.T, np.arange(0, lmax + 1, 2))[::2].T


def zhgaussian(bvals, Da, Dr, lmax=8):
    """
    Computes the zonal harmonic Gaussian coefficients for the given parameters.

    Parameters:
    bvals (array-like): Input array of b-values.
    Da (float): Axial Diffusivity.
    Dr (float): Radial Diffusivity.
    lmax (int, optional): Maximum degree of the coefficients. Default is 8.

    Returns:
    array: zonal harmonics coefficients computed using the input parameters.

    """
    b = bvals.reshape((-1, 1))
    gauss = (np.sqrt(4 * np.pi / (2 * np.arange(0, lmax + 1, 2) + 1)) *
             np.exp(-b * Dr)
            * expcoefs(b * (Da - Dr), lmax))
    return gauss / gauss[0,0]

def plot_wmr(wmr, bvals, c='blue'):
    '''
    Plots the white matter response, defined by the SH coordinates in wmr.
    '''
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.zeros((200,))
    L = 2*(wmr.shape[1]-1)
    shb = modshbasis(L, theta, phi)
    shb0 = shb[:,[(l+1)*(l+2)//2-l-1 for l in range(0,L+1,2)]]
    for k in range(wmr.shape[0]):
        r = np.dot(shb0, wmr[k,:].T)
        plt.polar(theta - np.pi/2, r, label='b'+str(bvals[k]), c=c)
        xlocs, xlabels = plt.xticks()
        plt.xticks(xlocs, [''] * len(xlocs))
        plt.grid(False)
        plt.yticks([],[])

def plot_wmr_on_axis(wmr, bvals, ax, **kwargs):
    '''
    Plots the white matter response, defined by the SH coordinates in wmr on an already existing axis.
    '''
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.zeros((200,))
    L = 2*(wmr.shape[1]-1)
    shb = modshbasis(L, theta, phi)
    shb0 = shb[:,[(l+1)*(l+2)//2-l-1 for l in range(0,L+1,2)]]
    for k in range(wmr.shape[0]):
        r = np.dot(shb0, wmr[k,:].T)
        ax.plot(theta - np.pi/2, r, label='b'+str(bvals[k]), **kwargs)
        xlocs = np.arange(0, 2*np.pi, np.pi/4)
        ax.grid(False)
        ax.set_xticks(xlocs, [''] * len(xlocs))
        ax.set_yticklabels([])

def plot_odf(odf):
    """
    Plot the ODF using matplotlib. Note that the ODF is 3D, while this is just a 2D visualisation.
    :param odf: ODF is spherical harmonics
    :return: None
    """
    lmax = l4n(odf.shape[-1])
    theta = np.linspace(0, np.pi, 200)
    phi = np.zeros(200)
    Q = modshbasis(lmax, theta, phi)
    odf_dwi_half = Q @ odf.T
    odf_dwi_full = np.concatenate((odf_dwi_half, odf_dwi_half))

    plt.polar(np.linspace(0, 2*np.pi, 400)-np.pi/2, odf_dwi_full, c='blue')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

def plot_odf_on_axis(odf, ax, **kwargs):
    """
    Plot the ODF using matplotlib on an existing axis. Note that the ODF is 3D, while this is just a 2D visualisation.
    :param odf: ODF is spherical harmonics
    :param ax: Matplotlib pyplot axis
    :param kwargs: Additional keyword arguments for ax.plot
    :return: None
    """
    lmax = l4n(odf.shape[-1])
    theta = np.linspace(0, np.pi, 200)
    phi = np.zeros(200)
    Q = modshbasis(lmax, theta, phi)
    odf_dwi_half = Q @ odf.T
    odf_dwi_full = np.concatenate((odf_dwi_half, odf_dwi_half))

    ax.plot(np.linspace(0, 2 * np.pi, 400) - np.pi / 2, odf_dwi_full, **kwargs)
    # Hide grid lines
    ax.grid(False)
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
