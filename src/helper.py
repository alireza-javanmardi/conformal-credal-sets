import numpy as np

def runif_in_simplex(n):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)



def dicrete_simplex(n_samples=5e5, n_classes=3):
    """descretize simplex space with n_samples

    Args:
        n_samples (int, optional): number of random samples. Defaults to 5e5.
        n_classes (int, optional): number of classes. Defaults to 3.

    Returns:
        np.array: array of size n_samples*n_classes that descretize the simplex space
    """
    
    simplex = []
    for i in range(n_samples):
        simplex.append(runif_in_simplex(n_classes))

    return np.array(simplex)


def tv(p,q):
    """total variation distance of two discrete distribution

    Args:
        p (_type_): first distribution
        q (_type_): second distribution

    Returns:
        float: a number between 0 and 1
    """
    return 0.5*np.sum(np.abs(p-q))


def compute_quantile(scores, alpha):
    """compute quantile from the scores

    Args:
        scores (list or np.array): scores of calibration data
        alpha (float): error rate in conformal prediction
    """
    n = len(scores)

    return np.quantile(scores, np.ceil((n+1)*(1-alpha))/n, method="inverted_cdf")
