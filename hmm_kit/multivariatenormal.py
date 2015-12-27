import numpy as np


class MultivariateNormal(object):
    """
    Normal distribution for multidimensional data
    @param mean: mean array
    @param cov: covariance matrix
    """

    def __init__(self, mean, cov):
        mean = np.array(mean)
        if len(mean.shape) == 1:
            mean = mean[None, :]
        self.mean = mean
        cov = np.array(cov)  # in case cov is list not np array
        if cov.ndim == 0:
            cov = cov[np.newaxis, np.newaxis]
        elif cov.ndim == 1:
            cov = cov[np.newaxis]

        d = cov.shape[0]
        s, u = np.linalg.eigh(cov)
        s = np.maximum(s, 0)
        s_pinv = np.array([0 if abs(x) < 1e-5 else 1 / x for x in s], dtype=float)
        self.prec_U = np.multiply(u, np.sqrt(s_pinv))
        pdet = np.prod(s[s > 1e-5])
        self.norm_p = d * np.log(2 * np.pi) + np.log(pdet)
        self.inv_cov = np.linalg.inv(cov)
        try:
            self.my_norm_p = (((2 * np.pi) ** d) * np.linalg.det(cov)) ** -0.5
        except FloatingPointError:
            # create semi positive definite matrix for the covariance... (numrical issue...)
            s = np.maximum(s, np.finfo(float).tiny * 2)
            cov = np.dot(np.dot(u, np.diag(s)), np.linalg.inv(u))
            self.inv_cov = np.linalg.inv(cov)
            almost_det = np.prod(s[s > 1e-5])
            self.my_norm_p = (((2 * np.pi) ** d) * almost_det) ** -0.5

    def log_pdf(self, x):
        """
        probability density function
        @param x: d-point or n X d-points array
        @return:
        """
        if len(x.shape) == 1:
            x = x[None, :]

        maha = np.sum(np.square(np.dot(x.T - self.mean, self.prec_U)), axis=-1)
        maha = -0.5 * (self.norm_p + maha)
        return maha

    def pdf(self, x):
        x = x.T - self.mean
        x_cov = np.dot(x, self.inv_cov)
        p = np.sum(x_cov * x, 1)

        try:
            p = np.exp(-0.5 * p)
            return (p * self.my_norm_p).T
        except FloatingPointError:
            log_norm = np.log(self.my_norm_p)
            minimum_val = np.ceil(np.log(np.finfo(float).tiny))
            p = np.maximum(p, 1e-300)
            p = np.maximum(-0.5 * p + log_norm, minimum_val)
            p = np.minimum(p, 0)
            return np.exp(p).T


class MixtureModel(object):
    """
    Mixture model for multiple distributions
    @param distributions: list of distribution objects (that expose pdf function)
    @param weights: array of weights such that sum(weights)=1
    """

    def __init__(self, distributions, weights):
        self.distributions = distributions
        self.weights = weights

    def components_pdf(self, x):
        return np.array([w * dist.pdf(x) for w, dist in zip(self.weights, self.distributions)])

    def pdf(self, x):
        return np.dot(self.weights, np.array([dist.pdf(x) for dist in self.distributions]))


def test_mtuidim_multivariate_normal():
    from matplotlib.pylab import plt
    x, y = np.mgrid[-5:5:.1, -5:5:.1]
    pos = np.array([x.ravel(), y.ravel()])
    rv = MultivariateNormal([0, 0], np.sqrt(np.array([[0.25, 0.3], [0.3, 1.0]])))

    z = rv.pdf(pos)

    cs = plt.contour(x, y, z.reshape((100, 100)))
    CB = plt.colorbar(cs, shrink=0.8, extend='both')
    plt.show()


if __name__ == "__main__":
    test_mtuidim_multivariate_normal()
