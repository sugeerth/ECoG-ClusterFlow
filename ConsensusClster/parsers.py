"""

Text parsers for creating SampleData objects


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

import sys, pca, scripts
from numpy import array, vstack, float32

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
        
        data_handle = open(data_file, 'rU')

        incomplete = self._parse_data_file(data_handle)

        data_handle.close()

        self.M = array([ x.data for x in self.samples ]) #, dtype=float32) FIXME: Not until we update euclidean...

        for sam in self.samples:
            del sam.data

        self.gene_names = array(self.gene_names)

        if incomplete and len(self.gene_names):
            incomplete = dict.fromkeys(incomplete) #kill duplicates

            if len(incomplete) >= len(self.gene_names):
                raise ValueError, 'No complete genes or incorrectly reported incomplete data!'

            kept_indices = tuple([ x for x in xrange(len(self.gene_names)) if self.gene_names[x] not in incomplete ]) #FIXME: better alg plox

            self.gene_names = self.gene_names.take(kept_indices)
            self.M = self.M.take(kept_indices, 1)


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
    
    def normalise(self, log2=False, sub_medians=True, center=False, scale=False):
        """Perform matrix normalisation.  See pca.normalise for details."""

        self.M = pca.normalise(self.M, log2=log2, sub_medians=sub_medians, center=center, scale=scale)


class NullParser(BaseParser):
    """

    Null class.  Used to create new BaseParser objects.

    """

    def __init__(self):
        pass


class ParseNormal(BaseParser):
    """

    ParseNormal

        This is the one you use when the data is just a table with no special characteristics
        Probes in rows, samples in columns.  Sample ids are in row 0, and gene ids
        are in column 0.
        
    """

    def __init__(self, data_file, console=None):

        BaseParser.__init__(self, data_file, console)

    def _parse_data_file(self, data_handle):
        """Parse datafile into sample name<->number pairs and load probe data"""

        incomplete = []
        # line = data_handle.readline()
        # print line


        for line in data_handle:
            a = line.split("\t")
            print a[1:]
            if a:

                if not self.samples:
                    for sample in a[1:]:
                        sample = sample.strip()

                        self.samples.append(SampleData(sample_id=sample))

                else:
                    self.gene_names.append(a[0])
                    import pprint

                    for i in range(len(self.samples)):
                        try:
                            self.samples[i].data.append(float(a[i+1]))
                        except:
                            print "creates exception", i
                            if self.console is not None:
                                self.console.write("Incomplete data for sample %s gene %s" % (self.samples[i].sample_id, a[0]))

                            self.samples[i].data.append(0.0)
                            incomplete.append(a[0])

        return incomplete


class ParseNoSampleNames(BaseParser):
    """

    ParseNoSampleNames

        This parser is used when there's a table which has the probes in columns and the samples on rows
        The first row contains probe names, and there aren't any sample names, such as in a partek output file

    """

    def __init__(self, data_file, console=None):

        BaseParser.__init__(self, data_file, console)

    def _parse_data_file(self, data_handle):

        count = 0

        for line in data_handle:
            a = line.split("\t")
            if a:

                if not self.gene_names:
                    for name in a:
                        name = name.strip()

                        self.gene_names.append(name)

                else:
                    self.samples.append(SampleData(sample_id = 'Sample ' + str(count), data = [ float(x) for x in a ]))
                    count += 1


class ParseSTI(BaseParser):
    """

    ParseSTI
        
        Parse the table from Supplemental Table STI formatted as a text file using tab delimits

        Usage:
            
            sdata = parsers.ParseSTI(data_file)

            Where data_file is the filename containing the STI table.

    """

    def __init__(self, data_file, console=None):
        
        self.refs = []

        BaseParser.__init__(self, data_file, console)

    def _parse_data_file(self, data_handle):
        """Parse datafile into sample name<->number pairs and load probe data"""

        header_flag = False

        for line in data_handle:
            a = line.split("\t")

            if a:
                if not header_flag:
                    header_flag = True

                else:
                    seq = a[4].strip()
                    
                    if not a[0] or a[0] == '0':
                        self.refs.append(SampleData(sample_id=a[1], sample_class=a[3], data=[ str(x) for x in seq ]))
                    else:
                        self.samples.append(SampleData(sample_id=a[1], sample_class=a[3], data=[ str(x) for x in seq ]))


def read_table(filename):
    """

    Read a simple conversion table between one name and another, useful for converting between probe names and gene symbols.

    This really should parse the entire CSV in the future, but memory concerns have held me back for now.
    Maybe an SQLite database?

    Returns a dict of first-column: second-column associations

    """

    conv = dict()


    handle = open(filename, 'r')

    for line in handle:
        a = line.split("\t")

        if a:

            conv[a[0]] = " - ".join(a[1:]).strip()

    return conv

def get_list_from_file(filename):
    """

    get_list_from_file(filename)

        Simply returns a concatenated list of every line in filename, with the newlines stripped

    Returns: A list of strings

    """

    handle = open(filename, 'r')
    entries = [ x.strip() for x in handle ]
    handle.close()

    return entries
