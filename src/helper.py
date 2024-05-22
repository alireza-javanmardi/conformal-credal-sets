import numpy as np

def simplex_discretizer(step=0.001):
    vectors = []
    for x in np.arange(0, 1 + step, step):
        for y in np.arange(0, 1 + step, step):
            for z in np.arange(0, 1 + step, step):
                if x + y + z == 1:  # Ensure the sum is one
                    vectors.append([x, y, z])
    return np.array(vectors)


def prob_rounder(probs, decimals=3):
    a = probs.copy()*0
    a[:, 1:] = np.round(probs[:,1:], decimals)
    a[:,0] = 1 - np.sum(a[:,1:], axis=1)
    return a

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

def prob_projector(p, axis): 
    """distribute the probability mass of given axis equally between the other two 

    Args:
        p (array): the input 3-d categorical probability distribution
        axis (int): 0, 1, or 2. 

    Returns:
        array: the resulting distribution
    """
    new_p = p.copy()
    for i in range(len(p)): 
        new_p[i] = new_p[i] + p[axis]/2
    new_p[axis] = 0
    return new_p 

def tu_au_eu(entropies):
    """calculate TU, AU, and EU for a given vector of entropies (of a credal set distributions)

    Args:
        entropies (array): entropies (of a credal set distributions)

    Returns:
        TU, AU, and EU
    """
    
    tu = np.max(entropies)
    au = np.min(entropies)
    eu = tu - au 

    return tu, au, eu 