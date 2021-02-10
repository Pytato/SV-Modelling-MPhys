import numpy as np


def force(h, yin, phi, mu, sigma_eta):
    """
    Compute the force term: - pdot_i = dH/dh_i.
    Returns negative to behave nicely in integrator??!?!??!

    :param h:
    :param yin:
    :param phi:
    :param mu:
    :param sigma_eta:
    """
    sigma_eta = np.sqrt(sigma_eta)
    T = len(yin)
    res = np.zeros(T)
    i = np.arange(1, T - 1)
    ip = np.arange(2, T)
    im = np.arange(0, T - 2)

    res[0] = 0.5 * (1. - yin[0] ** 2. * np.exp(-h[0])) + (1 - phi ** 2) * (-mu + h[0]) / sigma_eta ** 2 - phi * (
            -mu - phi * (-mu + h[0]) + h[1]) / sigma_eta ** 2
    res[i] = 0.5 * (1. - yin[i] ** 2. * np.exp(-h[i])) + ((-mu - phi * (-mu + h[im]) + h[i]) - phi *
                                                          (-mu - phi * (-mu + h[i]) + h[ip])) / sigma_eta ** 2
    res[-1] = 0.5 * (1. - yin[-1] ** 2 * np.exp(-h[-1])) + (-mu - phi * (-mu + h[-2]) + h[-1]) / sigma_eta ** 2

    return res


def H(h, p, yin, phi, mu, sigma_eta):
    """
    Compute entire Hamiltonian.

    :param p:
    :param h:
    :param yin:
    :param phi:
    :param mu:
    :param sigma_eta:
    """
    sigma_eta = np.sqrt(sigma_eta)
    A = np.sum(0.5 * (h + np.square(yin) * np.exp(-h)))
    B = (h[0] - mu) ** 2 / 2. / sigma_eta ** 2 * (1 - phi ** 2)
    C = np.sum((h[1:] - mu - phi * (h[0:-1] - mu)) ** 2) / (2. * (sigma_eta ** 2))
    return A + B + C + 0.5 * np.sum(np.square(p))
