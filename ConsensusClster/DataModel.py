import sys, pca, scripts
import copy
import pprint
from numpy import array, vstack, float32
import numpy as np

class SampleData(object):
    """

    BaseCluster

        Usage:
            
            BaseParserObj.samples.append(SampleData(sample_id, sample_num, sample_class, data, index))
        
            sample_id       - Label for the sample.  Need not be unique. (Optional, but highly recommended)
            sample_num      - Another label (Optional)
            sample_class    - Yet another label, usually for known subclasses (Optional)
            data            - Data for this particular sample.  Will be converted into a numpy array by
                              BaseParser, regardless of what format data takes when assigned.  Required.
            index           - Yet another label (Optional)

        Properties

            cluster_id      - Id of cluster to which this sample belongs.  Generally assigned by clustering algorithms.


    """

    def __init__(self, sample_id=None, sample_num=None, sample_class=None, data=None, index=None):

        if data is None:
            self.data = []
        else:
            self.data = data

        self.cluster_id = None
        self.sample_class = sample_class
        self.sample_id = sample_id
        self.sample_num = sample_num
        self.index = index

    def __getitem__(self, x):

        return self.data[x]


class BaseParser(object):
    """

    BaseParser
        Gracioulsy taken from the visualizations from Micheal sandiers 

        Common methods and actions for all parsers
        Designed to be subclassed.  All parsers must do the following:
        
        1: Have an __init__ method which takes data_file as an argument

        2: Have a _parse_data_file method which adds SampleData objects to self.samples, using
           the data from data_file.  Optionally, this can return a list of gene_ids which
           have incomplete information.  All data from this gene_id will be stripped from the
           final dataset.

        3: If the data vector entries have labels (e.g., gene probe names), they should be assigned
           to self.gene_names as a list, in the same length and order as the data vectors in each sample
    
        4: Be called "ParseSOMETHING" where SOMETHING is any name
           The gtk front end looks for ParseX names and puts them in a drop-down list for convenience

        Properties

            samples     - A list of SampleData objects
            gene_names  - A list of data vector entry labels

    """

    def __init__(self, data_file, console=None):

        self.samples = []  #List of SampleData instances

        self.gene_names = []

        self.console = console #Use this to write to the user; avoid print if possible
        
        self.datafile = data_file

        # data_handle = open(data_file, 'rU')

        incomplete = self._parse_data_file(self.datafile)

        # incomplete = data_file

        # data_handle.close()

        self.M = copy.deepcopy(data_file)

        for sam in self.samples:
            del sam.data

        self.gene_names = array(self.gene_names)

        # print "gene names"
        # pprint.pprint(self.gene_names)
        
        # print "self.M"
        # pprint.pprint(self.M)


        # if incomplete and len(self.gene_names):
        #     incomplete = dict.fromkeys(incomplete) #kill duplicates

        #     if len(incomplete) >= len(self.gene_names):
        #         raise ValueError, 'No complete genes or incorrectly reported incomplete data!'

        #     kept_indices = tuple([ x for x in xrange(len(self.gene_names)) if self.gene_names[x] not in incomplete ]) #FIXME: better alg plox

        #     self.gene_names = self.gene_names.take(kept_indices)
        #     self.M = self.M.take(kept_indices, 1)


    def _parse_data_file(self, data_handle):
        """Parse datafile into sample name<->number pairs and load data"""
        pass

    def __delitem__(self, x):

        inds = list(xrange(len(self.samples)))
        inds.pop(x)

        self.M = self.M.take(tuple(inds), 0)
        
        del self.samples[x]

    def __getitem__(self, x):

        return self.samples[x]

    def __add__(self, x):
        """Add two BaseParser objects together.  If the gene lists are not the same, they will be reduced to the common set."""

        #if not (self.gene_names == x.gene_names).all():
        #    raise ValueError, "Genes for each set are not identical!"

        if len(self.gene_names) and len(x.gene_names):
            fst, snd = scripts.union(self.gene_names, x.gene_names)

            if not len(fst):
                raise ValueError, "No matching gene names in either set! Cannot concatenate."

            gene_names = self.gene_names.take(tuple(fst))
            
            M = self.M.take(fst, 1) #Memory-intensive but we certainly don't want to
            N = x.M.take(snd, 1)    #change the objects themselves

        else:
            print('WARNING: Concatenating one or more sets without a gene list! Do so at your own risk!')

        c = NullParser()

        c.gene_names = gene_names
        c.samples = self.samples + x.samples
        c.M = vstack((M, N))

        return c


class ParseNormal(BaseParser):
    """

    ParseNormal

        This is the one you use when the data is just a table with no special characteristics
        Probes in rows, samples in columns.  Sample ids are in row 0, and gene ids
        are in column 0.
        
    """

    def __init__(self, data_file, RegionName, console=None):

        self.RegionName = RegionName
        BaseParser.__init__(self, data_file, console)
        # print data_file

    def _parse_data_file(self, data_handle):
        """Parse datafile into sample name<->number pairs and load probe data"""

        incomplete = []

        for sample in self.RegionName:
            self.samples.append(SampleData(sample_id=sample))

        for electrode_names in self.RegionName:
            self.gene_names.append(electrode_names)


        for i in range(len(self.samples)):
                self.samples[i].data.append(np.array(data_handle[i]))
                
        # for line in data_handle:
        #     a = line.split("\t")
        #     print a[1:]
        #     if a:

        #         if not self.samples:
        #             for sample in a[1:]:
        #                 sample = sample.strip()

        #                 self.samples.append(SampleData(sample_id=sample))

        #         else:
        #             self.gene_names.append(a[0])
        #             import pprint

        #             for i in range(len(self.samples)):
        #                 try:
        #                     self.samples[i].data.append(float(a[i+1]))
        #                 except:
        #                     print "creates exception", i
        #                     if self.console is not None:
        #                         self.console.write("Incomplete data for sample %s gene %s" % (self.samples[i].sample_id, a[0]))

        #                     self.samples[i].data.append(0.0)
        #                     incomplete.append(a[0])

        return incomplete
