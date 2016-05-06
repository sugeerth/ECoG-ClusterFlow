#Random utilities

import parsers, numpy, os, log_analyse, pca, display

from itertools import combinations as comb

def list_or_files(*args):
    """
    
    Return the contents of *args as dict[name] = list pairs regardless of whether the user provided a filename or dict
    If a list is provided, it will be appended to the dict with key ''

    Returns a dict of name->list pairs

    WARNING: This function uses base names, not the full path!  If the base names are the same, one will be ignored!

    """

    ndict = {}

    for lst in args:
        if isinstance(lst, list):
            ndict.setdefault('', []).extend(lst)
            print("WARNING: List received! Multiple lists are concatenated!")

        elif isinstance(lst, dict):
            ndict.update(lst)
        
        else:
            name = os.path.basename(lst)
            ndict[name] = parsers.get_list_from_file(lst)

    return ndict

def union(list1, list2):
    """
    Remove the indices which make up the union between two lists in log n time
    
    Returns a tuple, where tuple[0] is the list of indices in list1 which is in common with list2, and tuple[1]
    is the same list for list2

    """

    swapped = False
    if len(list1) > len(list2):         #Make list2 the longer one
        list1, list2 = list2, list1
        swapped = True

    indices_list1 = numpy.argsort(list1)
    indices_list2 = numpy.argsort(list2)

    union_indices_list1 = []
    union_indices_list2 = []
    
    breakpoint = 0

    for i in indices_list1:    
        for j in range(len(indices_list2))[breakpoint:]:    #Ugly, but reduces complexity
            idx = indices_list2[j]

            if list1[i] == list2[idx]:
                union_indices_list1.append(i)
                union_indices_list2.append(idx)
                breakpoint = j
                break

    if not swapped:
        return union_indices_list1, union_indices_list2

    return union_indices_list2, union_indices_list1

def scale_to_set(sdata, *filenames):
    """

    scale_to_set(filename)

        Removes all but those sample_ids you specifiy.

        filenames    - filenames or dicts
                       each file containing list of sample ids to use
                       or each dict containing name->list of sample ids

    Returns: modified sdata object, dict of cluster->indices

    """

    print "I am in scripts",sdata, defined_clusters
    
    defined_clusters = list_or_files(*filenames)

    sample_id_list = [ x.sample_id for x in sdata.samples ]
    samples_to_keep = sum([ defined_clusters[x] for x in defined_clusters ], [])
    sample_indices = union(sample_id_list, samples_to_keep)[0]

    #Adjustment
    sdata.samples = [ sdata.samples[x] for x in sample_indices ]
    sdata.M = sdata.M.take(tuple(sample_indices), 0)

    sample_id_list = [ x.sample_id for x in sdata.samples ] #This is different!
    
    for name in defined_clusters: #If samples aren't in the main, ignore them
        sample_list = defined_clusters[name]
        def_indices = union(sample_list, sample_id_list)[0]
        defined_clusters[name] = [ sample_list[x] for x in def_indices ]

    print "I am in scripts",sdata, defined_clusters

    return sdata, defined_clusters

def scale_probes(sdata, *filenames):
    """

    scale_probes(sdata, filename)
        
        Removes all gene probes except those you specify

        filename    - File(s) containing a list of probes, one on each line
                      Also accepted: Lists, dicts.  Only the values will be used in the dicts.

    Returns: modified sdata object

    NOTE: Currently there is no way to call this from common.py!  This will change in the future.
          For now, stick this in your _preprocess(self) subclass.

    """

    plist = sum(list_or_files(*filenames).values(), [])

    probes_to_keep = tuple(union(sdata.gene_names, plist)[0])

    sdata.M = sdata.M.take(probes_to_keep, 1)
    sdata.gene_names = sdata.gene_names.take(probes_to_keep)

    return sdata

def new_defined_clusters(sdata, conv):
    """

    Define different clusters than the ones specified by your Defined Clusters, whether
    through the GUI, modification of keep_list, or through the command line

    sdata: sample data obj
    conv: conversion dict, keys sample ids values new cluster assignments
    Stick this in your preprocess function (see common.py for subclassing help)

    """
    
    new_clusts = {}
    s_ids = [x.sample_id for x in sdata.samples]

    for s_id in s_ids:
        if s_id in conv:
            new_clusts.setdefault(conv[s_id], []).append(s_id)
        else:
            new_clusts.setdefault('Unknown', []).append(s_id)

    return new_clusts

def write_normal(sdata, filename):
    """

    Takes an sdata obj and writes out a tab-delimited datafile, suitable for ParseNormal
    useful to convert between data formats, writing PCA-selected data, etc

    """
    
    if not len(sdata.gene_names) > 0: 
        raise ValueError, "No gene names found! Unsuitable for this data format."

    sids = [x.sample_id for x in sdata.samples]

    f = open(filename, 'w')

    #Sample line, first row
    f.write("\t".join(['SAMPLE ID'] + sids))
    f.write("\n")

    #Data
    for i in xrange(len(sdata.gene_names)):
        f.write("\t".join([sdata.gene_names[i]] + [ str(sdata.M[j][i]) for j in xrange(len(sdata.samples)) ]))
        f.write("\n")

    f.close()

def make_def_clusters_from_log(logfile):
    """Takes a logfile and writes cluster definition files"""

    logdict = log_analyse.gen_cluster_dict(logfile)

    for clustname in logdict:
        name = clustname.split() #Most currently: # (colour)
        filen = 'cluster_' + str(name[0]) #cluster_0, etc

        f = open(filen, 'w')

        for sample in logdict[clustname]:
            f.write(sample)
            f.write("\n")

        f.close()

def remove_pc(sdata, num=1):
    """Remove the first num principle components from the data"""

    M = sdata.M
    
    avg = numpy.average(M, 0)

    M -= avg

    u, s, V = numpy.linalg.svd(M, 0)        #Decompose
    S = numpy.identity(s.shape[0]) * s

    for i in xrange(num):
        S[i][i] = 0.        #Sets the offending eigenvalue to 0

    sdata.M = numpy.dot(numpy.dot(u, S), V) + avg       #Recompose

    return sdata

def write_table(ndict, filename):
    """Write a tab delimited flat file, one key per line"""

    ls = ndict.keys()
    ls.sort()

    f = open(filename, 'w')
    
    for key in ls:
        f.write("\t".join([str(key), str(ndict[key])]))
        f.write("\n")

    f.close()

def write_ratio(s, clust1, clust2, filename, snr_threshold=0.5, pval_threshold=0.001, ttest=False):
    """Write SNR ratios given sdata obj, clust1 list or filename, clust2 list or filename, file to write to, threshold for SNR, threshold for pval, do ttest or not"""

    M = s.M

    f = open(filename, 'w')
    
    c1name, c1ind = get_indices(s, clust1)
    c2name, c2ind = get_indices(s, clust2)
    
    ratios = pca.snr(M, c1ind, c2ind, snr_threshold, ttest)
    
    f.write("%s vs %s:\n" % (c1name, c2name))
    f.write("--------------------\n")
    f.write("Gene ID\t\t%s Avg\t%s Avg\tSNR Ratio\tp-value\n" % (c1name, c2name))
    
    for ratio, gene_idx, mean1, mean2, pval in ratios:
        if not ttest or pval <= pval_threshold:
            f.write("\n%10s\t\t%f\t\t%f\t\t%f\t\t%s" % (s.gene_names[gene_idx], mean1, mean2, ratio, pval))

    f.close()
                        
def write_classifier(s, filename, clust1, clust2=None, threshold=None):
    """
    Writes a bayesian binary classifier
    
    See pca.binary_classifier for details
    
    s - SampleData obj
    filename - What to write the classification information to
    clust1 - a list or a file with one sample name on each line which composes the cluster you're trying to define
    clust2 - a list or an optional second cluster.  Otherwise it's every other sample.
    threshold - If you want to classify using only genes over a certain SNR threshold, use this.  None uses all.

    """

    #FIXME: This needs updating

    M = s.M

    f = open(filename, 'w')
    
    c1name, c1ind = get_indices(s, clust1)
    
    if clust2 is not None:
        c2name, c2ind = get_indices(s, clust2)
    else:
        c2ind = numpy.lib.arraysetops.setdiff1d(numpy.arange(len(s.samples)), numpy.array(c1ind))

    rlist, w0 = pca.binary_classifier(M, c1ind, c2ind, threshold)

    f.write("%s vs %s:\n" % (c1name, c2name))
    f.write("--------------------\n\n")

    #Returns (a, b), where a is w in (wi, i) pairs and b is w0
    f.write("w0 is %s\n" % w0)
    f.write("\nGene ID\t\tMultiplier\n\n")

    rlist.sort() #FIXME: abs?

    for result in rlist:
        f.write("%10s\t%f\n" % (s.gene_names[result[1]], result[0]))

    f.close()

def get_indices(s, filename):
    """
    Return the indices of the samples in filename in the sdata object
    
    """

    sams = list_or_files(filename)
    name = sams.keys()[0]
    
    return name, union([ x.sample_id for x in s.samples ], sams[name])[0]

def km(timeconv, eventconv, *clusts):
    """
    Draw the kaplan-meier curves for a number of clusters
    
    timeconv - file of tab-delim table of sample id -> survival time conversions
    eventconv - file of tab-delim table of sample id -> event conversions (ie, 1 for yes, 0 for no, anything else for NA)
    clusts - cluster filenames
    
    """
    tc = parsers.read_table(timeconv)
    ec = parsers.read_table(eventconv)

    labels = []
    times = []
    events = []

    for clust in clusts:

        cl = parsers.get_list_from_file(clust)
       
        ts = []
        ev = []

        for sam in cl:
            try:
                time = float(tc[sam])
                event = int(ec[sam])

                ts.append(time)
                ev.append(event)
            except:
                pass

        labels.append(clust)
        times.append(ts)
        events.append(ev)

    display.km(times, events, labels)

def compare_clust(*clusts):
    """

    Print some statistics about the similarities/differences between 'clusts',
    which are files with one sample_id on each line, as per usual.

    """

    cdict = list_or_files(*clusts)

    for fst, snd in comb(cdict.keys(), 2):
        fst_ind, snd_ind = union(cdict[fst], cdict[snd])
        num_common = len(fst_ind)

        if num_common:

            print('\n%s vs %s:' % (fst, snd))
            print('Common: %s (%f, %f) (%s/%s, %s/%s)' % (num_common, float(num_common) / len(cdict[fst]), float(num_common) / len(cdict[snd]), num_common, len(cdict[fst]), num_common, len(cdict[snd])))

def write_list(ls, filename):
    """

    Write a simple list to a file, each element on one line.

    Suitable for cluster definitions, etc

    """

    f = open(filename, 'w')

    for x in ls:
        f.write(str(x))
        f.write('\n')

    f.close()

def create_mvpa_dataset(s, *names):

    from mvpa.suite import Dataset
    
    dset = None

    def_clusters = list_or_files(*names)

    for name in def_clusters:
        n, ind = get_indices(s, {name: def_clusters[name]})
        
        if dset:
            dset += Dataset(samples=s.M.take(tuple(ind), 0), labels=n)
        else:
            dset = Dataset(samples=s.M.take(tuple(ind), 0), labels=n)

    return dset

def cv_workflow(s, fdict, clf):

    from mvpa.suite import CrossValidatedTransferError, TransferError, NFoldSplitter

    errors = []
    confusions = []

    for i in xrange(len(s.M[0]) - 1):
        pset = list(s.gene_names)[:-1]
        scale_probes(s, pset)
        dset = create_mvpa_dataset(s, fdict)
        cv = CrossValidatedTransferError(TransferError(clf), NFoldSplitter(), enable_states=['results', 'confusion'])
        er = cv(dset)
        errors.append((len(s.M[0]), er))
        confusions.append((len(s.M[0]), cv.confusion.asstring(description=1)))

    return errors, confusions
