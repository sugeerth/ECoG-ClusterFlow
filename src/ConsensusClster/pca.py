"""

PCA and other normalisation methods.


Copyright 2008 Michael Seiler
Rutgers University
miseiler@gmail.com

This file is part of ConsensusCluster.

ConsensusCluster is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ConsensusCluster is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ConsensusCluster.  If not, see <http://www.gnu.org/licenses/>.


"""

import numpy

try:
    import scipy.stats as st
    PVAL = 1
except:
    PVAL = 0

def get_pca_genes(M, pca_fraction=0.85, eigenvector_weight=0.15):
    """
    Convenience function.  Expects a matrix with samples on the rows and genes on the columns

    pca_fraction        - Fraction of eigenvalues which explain pca_fraction of the variance to accept
    eigenvector_weight  - The top eigenvector_weight (by absolute value) fraction of those genes which occur with high weights
                          in those eigenvectors which correspond to the eigenvalues explained by pca_fraction
    
    Returns a tuple: (pca_fraction eigenvectors, eigenvector_weight gene indices)

    """

    V = pca(M, pca_fraction)    #From SVD
    gene_indices = select_genes(V, eigenvector_weight)

    return V, gene_indices

def pca(M, frac = 1.):
    """Takes a matrix M and returns those eigenvectors which explain frac of the variance"""
    
    avg = numpy.average(M, 0)
    M -= avg    #PCA requires a centered matrix

    u, s, v = numpy.linalg.svd(M, 0) #Will run out of memory from U otherwise
    
    M += avg    #M is a reference...shouldn't touch user's toys
    
    i = get_var_fractions(s, frac)

    #return numpy.transpose(numpy.dot(v[:i], numpy.transpose(M))) #The transformed data
    return v[:i]

def mds(M, frac = 1.):
    """
    
    Takes a matrix M and returns the classical multidimensional scaling
    
    If a lower dimensional representation is needed (e.g., for a plot), frac is the accuracy of reconstruction

    """

    avgrow = M.mean(1)
    avgcol = M.mean(0)

    N = (M.T - avgrow).T - avgcol + M.mean() #Row and column centred

    u, s, v = numpy.linalg.svd(N, 0)

    i = get_var_fractions(s, frac)
    
    S = numpy.identity(s.shape[0]) * s

    return numpy.dot(u, numpy.sqrt(S))[:,:i]
    
def get_var_fractions(s, frac):
    """
    Get the element that, when summed up to this element, frac
    of the percentage of the variance is explained by the elements
    summed in this way.

    s is the singular matrix (in 1-D)
    frac is the fraction to explain
    
    """

    variances = s**2/len(s)
    total_variances = numpy.sum(variances, 0)

    variance_fractions = numpy.divide(variances, total_variances)

    for i in range(1, len(variance_fractions) + 1):
        if numpy.sum(variance_fractions[:i], 0) >= frac:
            break

    if i < 2:
        i = 2   #Minimum 2 for plotting!

    return i

def select_genes(v, weight):
    """Returns a tuple of the indices of those genes which comprise the top weight% in each eigenvector"""

    genes = [0] * len(v[0])
    gene_indices = []

    for vec in v:
        min_value = (1 - weight) * numpy.min(vec)
        max_value = (1 - weight) * numpy.max(vec)
 
        for i in xrange(len(vec)):
            if vec[i] <= min_value or vec[i] >= max_value:
                genes[i] = 1

    for i in xrange(len(genes)):
        if genes[i]:
            gene_indices.append(i)

    if len(gene_indices) < 2:
        raise TypeError, "Not enough genes at %s%% weight in eigenvectors" % (weight)

    return tuple(gene_indices)

def normalise(M, log2=True, sub_medians=True, center=True, scale=True):
    """
    Perform a number of normalisation routines on M
    
    log2                - log2 transform the data (Yes if this is raw gene data)
    sub_medians         - subtract the median of medians from each data value
    center              - subtract the data by its average, making the overall mean 0
    scale               - subtract the root-mean-square of the data AFTER centering

    """

    if log2:
        M = log2_transform(M)

    if sub_medians:
        M = subtract_medians(M)

    if scale or center:
        M = center_matrix(M)

    if scale:
        M = scale_matrix(M)

    return M

def center_matrix(M):
    """Subtract the mean from matrix M, resulting in a matrix with mean 0"""

    return (M - numpy.average(M, 0))

def scale_matrix(M):
    """Subtract the root-mean-square from each data member"""

    numpy.seterr(all='raise')   #Catch divide-by-zero, otherwise SVD won't converge

    T = numpy.transpose(M)
    for i in range(M.shape[1]):
        if T[i].any():
            T[i] = T[i] / numpy.sqrt(numpy.sum(T[i]**2) / (M.shape[0] - 1))

    return M

def log2_transform(M):
    """Take the log2 of each value in M"""

    return numpy.log(M)/numpy.log(2)

def subtract_medians(M):
    """Subtract each value in M by the median of medians"""

    return M - numpy.median(M)
    
def subtract_feature_medians(M):
    """Subtract each column in M by the median over all rows"""

    return M - numpy.median(M, 0)

def row_normalise_mean(M):
    """Normalise rows to mean 0"""

    return (M.T - numpy.average(M, 1)).T

def row_normalise_median(M):
    """Normalise rows to median 0"""

    return (M.T - numpy.median(M, 1)).T

def snr(M, list1, list2, threshold = None, significance = False):
    """

    Performs a signal-to-noise ratio test on M, assuming samples are in rows and genes are in columns

        list1       - List of row indices for first group
        list2       - List of row indices for second group
        threshold   - Minimum SNR ratio to report
        significance - Run kruskal ttest (requires scipy)

    Returns a reverse-ordered list of (ratio, index, mean1, mean2, pvalue) tuples, where index is the column index of the gene,
    and mean1 and mean2 correspond to the mean for that particular gene in list1 and list2, respectively.  pvalue is blank if significance
    is False.

    If signifance is true (and scipy is installed) a pvalue will be assigned. Be ware this increases processing
    time significantly (ha).

    """

    ratios = []

    N1 = M.take(tuple(list1), 0)
    N2 = M.take(tuple(list2), 0)

    N1mean, N2mean = N1.mean(0), N2.mean(0)
    means = numpy.abs(N1mean - N2mean)
    stds  = N1.std(0) + N2.std(0)

    if stds.all():
        rats = means / stds
    else:
        rats = numpy.zeros((len(means),), dtype=numpy.float32)
        for i in xrange(len(stds)):
            if stds[i]:
                rats[i] = means[i] / stds[i]

    for i in xrange(M.shape[1]):

        rat = rats[i]
        mean1, mean2 = N1mean[i], N2mean[i]

        if threshold is None or rat >= threshold:

            if PVAL and significance:
                pval = st.kruskal(N1[:,i], N2[:,i])[1]
            else:
                pval = ''
    
            ratios.append( (rat, i, mean1, mean2, pval) )

    ratios.sort(reverse=True)

    return ratios

def binary_classifier(M, list1, list2, threshold = None, prior_prob = None):
    """

    Create a bayesian linear-discrimination function based on two clusters being list1 and list2

        list1   - List of row indices for first group
        list2   - List of row indices for second group
        threshold - Minimum SNR ratio to report

    WARNING: This function makes some assumptions of which you should be aware:
        
        Genes are independent events (likely false)
        Your prior probabilities are indeed indicative of a true population
        Your data is indicative of a true population
        Your clusters are definitively known, and perfect
        Up/down status determined by gene average IN YOUR DATA SET truly represents average and upgregulation/downregulation (definitely false!)

    NOTE: Prior probabilities unimplemented

    """
    #FIXME: Broken currently, fix column shit
    
    w = []
    w0 = 0.0

    ran = numpy.random.random

    cols = get_columns(M, list1, list2)

    snr_genes = []
    for ratio in snr(M, list1, list2, threshold):
        snr_genes.append(ratio[1])

    for i in snr_genes:
        p_array = cols[i][0]
        q_array = cols[i][1]

        p = (p_array > 0.0).sum() / float(len(p_array))  #array([True, False, False, True... etc
        q = (q_array > 0.0).sum() / float(len(q_array))

        #hack
        if p > 0.999:
            p = 0.999
        if q > 0.999:
            q = 0.999
        if p < 0.001:
            p = 0.001
        if q < 0.001:
            q = 0.001

        w.append( (numpy.log( (p * (1 - q)) / (q * (1 - p)) ), i) ) #(wi, i) pairs
        w0 += numpy.log( (1 - p) / (1 - q) )

    return w, w0 #g(x) = w*xi for x in w + w0
