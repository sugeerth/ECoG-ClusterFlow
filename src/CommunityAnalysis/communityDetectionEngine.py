import csv
import colorsys
import math
import colorsys
import time
from collections import defaultdict
from PySide import QtCore, QtGui
from PySide.QtCore import *
from ConsensusClster.cluster import ConsensusCluster
import warnings
warnings.filterwarnings("ignore")

import pyqtgraph as pg

import math, numpy, sys, random
import operator
import numpy as np
from itertools import combinations as comb

from scipy import cluster as cl

from sys import platform as _platform
import weakref
from random import randint
import pickle
import cProfile
import pprint
import copy 
import community as cm

# ... reading NIfTI 
import nibabel as nib
import numpy as np
# ... graph drawing
import networkx as nx
from GraphView.Edge import Edge 
from GraphView.Node import Node
from kmedoids import ClusterAlgorithms
import ConsensusClster.DataModel
import ConsensusClster.pca as pca

from random import shuffle


Number_of_Communities = 2
starttime = 4 
endtime = 9

class ConsensusCustomCluster(object):
    """docstring for ConsensusMediator"""
    def __init__(self, graphwidget, ClusterAlgorithms):
        super(ConsensusCustomCluster, self).__init__()
        self.graphWidget = graphwidget
        self.partition = dict()
        self.timestepPartition = dict()
        timestepDict = dict()
        length = len(self.graphWidget.g)

        syllable = self.graphWidget.Syllable
        self.syllable = syllable
        name = "ConsensusData/ConsensusCluster"+str(syllable)+str(4)+".json"

        self.timestepPartition = pickle.load(open(name))

    def prepareClsuterData(self, graph, timestep, syllable): 
        if not(self.syllable == syllable):
            self.syllable = syllable
            FileNames = '/Users/sugeerthmurugesan/Sites/Sankey/JSON_1.json'
            nameHeatmap = "ConsensusData/DeltaAreaChange"+str(syllable)+str(4)+"Heatmap.tsv"

            name = "ConsensusData/ConsensusCluster"+str(syllable)+str(4)+".json"
            print "The new Cluster Data has been loaded", name
            self.timestepPartition = pickle.load(open(name))

        distances = copy.deepcopy(nx.to_numpy_matrix(graph))
        try: 
            self.partition = copy.deepcopy(self.timestepPartition[timestep])
            if timestep == 0: 
                assert not(self.partition == None) 
            assert self.partition == self.timestepPartition[timestep]
        except KeyError:
            pass
        # print "No of communities",len(self.partition.keys()), len(set(self.partition.values()))
        return self.partition

    def changeSyllable(self, syllable):
        self.timestepPartition = dict()
        syllable = self.graphWidget.Syllable
        name = "ConsensusData/ConsensusCluster"+str(syllable)+str(4)+".json"

        self.timestepPartition = pickle.load(open(name))

class CustomCluster(object):
    """docstring for ConsensusMediator"""
    def __init__(self, graphwidget, ClusterAlgorithms):
        super(CustomCluster, self).__init__()
        self.graphWidget = graphwidget
        self.partition = dict()
        self.partition1 = dict()
        self.timestepPartition = dict()
        timestepDict = dict()
        length = len(self.graphWidget.g)

        name = "ConsensusData/TenRandomTimesteps.json"
        self.timestepPartition = pickle.load(open(name))

    def prepareClsuterData(self, graph, timestep, syllable): 
        distances = copy.deepcopy(nx.to_numpy_matrix(graph))
        
        x = [i for i in range(21)]
        y = [i for i in range(21,40)]
        z = [i for i in range(40,64)]

        q = [i for i in range(11,40)]
        r = [i for i in range(41,64)]

        ClusterStart = [[0,10,20,30], [2,9,11], [8,12,13], [1,12,14], [6,7,10], [1,15,20]] 
        #* MADE CHANGES 
        
        # if timestep > ClusterStart[syllable][0] and timestep < ClusterStart[syllable][1]: 
        #     for i in range(len(distances)):
        #         self.partition[i] = 0
                # if i in x: 
                #     self.partition[i] = 0
                # elif i in y: 
                #     self.partition[i] = 1
                # elif i in z: 
                #     self.partition[i] = 2

        # if timestep >= ClusterStart[syllable][0] and timestep <= ClusterStart[syllable][1]:
        #     for i in range(len(distances)):
        #         if i in x: 
        #             self.partition[i] = 0
        #         elif i in y: 
        #             self.partition[i] = 1
        #         elif i in z: 
        #             self.partition[i] = 2
        # elif timestep >= ClusterStart[syllable][1] and timestep <= ClusterStart[syllable][2]:
        #     self.partition = self.timestepPartition[timestep-10]
        #     try: 
        #         self.partition = copy.deepcopy(self.timestepPartition[timestep])
        #         if timestep == 0: 
        #             assert not(self.partition == None) 
        #         assert self.partition == self.timestepPartition[timestep]
        #     except KeyError:
        #         for j in range(len(distances)):
        #             if syllable == 0: 
        #                 self.partition[j] = randint(0, 1)
        #             else: 
        #                 if j in q:
        #                     self.partition[j] = 0
        #                 else:
        #                     self.partition[j] = randint(1, 2)


        # elif timestep >= ClusterStart[syllable][2] and timestep <= ClusterStart[syllable][3]: 
        #     for i in range(len(distances)):
        #         if i in q: 
        #             self.partition[i] = 0
        #         elif i in r: 
        #             self.partition[i] = 1

        print timestep



        if timestep > ClusterStart[syllable][0] and timestep < ClusterStart[syllable][1]:
            for i in range(len(distances)):
                if i in x: 
                    self.partition[i] = 0
                elif i in y: 
                    self.partition[i] = 1
                elif i in z: 
                    self.partition[i] = 2

        elif timestep > ClusterStart[syllable][2]-1 and timestep < ClusterStart[syllable][3]+1:
            for i in range(len(distances)):
                if i in x: 
                    self.partition[i] = 0
                elif i in y: 
                    self.partition[i] = 1
                elif i in z: 
                    self.partition[i] = 2
        else: 
            try: 
                self.partition = copy.deepcopy(self.timestepPartition[timestep])
                if timestep == 0: 
                    assert not(self.partition == None) 
                assert self.partition == self.timestepPartition[timestep]
            except KeyError:
                for j in range(len(distances)):
                    if syllable == 0: 
                        self.partition[j] = randint(0, 2)
                    else: 
                        if j in q:
                            self.partition[j] = 0
                        else:
                            self.partition[j] = randint(1, 2)
        if  timestep == 0:
            self.partition.clear()
            for i in range(len(distances)):
                if i in x: 
                    self.partition[i] = 0
                elif i in y: 
                    self.partition[i] = 1
                elif i in z: 
                    self.partition[i] = 2
                # self.partition[20] = 0
                # self.partition[39] = 1

            pprint.pprint(self.partition)
        return self.partition

class ConsensusMediator(object):
    """docstring for ConsensusMediator"""
    def __init__(self, graphwidget, ClusterAlgorithms):
        super(ConsensusMediator, self).__init__()
        self.graphWidget = graphwidget
        self.partition = dict()
        self.partition1 = dict()
        self.GlobalCDF = dict()
        self.area = dict()
        self.DeltaAreaTimestep = dict()
        self.deltaArea = np.zeros(10)
        plotKValues = pg.PlotWidget(pen = 'r')
        self.plotKValues =  plotKValues
        self.count = 0
        self.ClusterAlgorithms = ClusterAlgorithms 
        # Parameters
        self.pca_only = False
        self.kvalues = [2,3,4,5,6]
        self.subsamples = 30
        self.FinalPartiction =  dict() 
        self.subsample_fraction = 1
        self.ParsedData = None
        self.norm_var = False # variance == 1
        self.Timestep = None

    def getClusterData(self,data):
        return ConsensusCluster(data)

    def prepareClsuterData(self, graph, Timestep, syllable):
        print "Cluster Time ",Timestep, "Syllable:",syllable
        self.Timestep = Timestep
        distances = copy.deepcopy(nx.to_numpy_matrix(graph))

        samples = [x for x in xrange(len(distances))]
        self.ParsedData = ConsensusClster.DataModel.ParseNormal(distances, samples)
        idlist = [ x.sample_id for x in self.ParsedData.samples ]
        
        if len(dict.fromkeys(idlist)) != len(idlist):
            raise ValueError, 'One or more Sample IDs are not unique!\n\n\nHere is the data'+str(len(dict.fromkeys(idlist)))+'len'+str(len(idlist))

        if not self.pca_only:
            self._postprocess()
            kwds = []
            self.GlobalCDF = dict()
            self.partition1 = dict()
            self.deltaArea = dict()
            self.area = dict()

            self.newKValues = [2,3,4,5,6,7]

            for i in self.newKValues:
                self.partition1[i] = self.run_cluster(i, self.subsamples, self.subsample_fraction, self.norm_var, kwds)
            k = self.createDeltaArea(self.kvalues)

        if Timestep == 62:
            name = "ConsensusData/DeltaAreaChange"+str(self.graphWidget.Syllable)+str(4)+"Heatmap.tsv"
            name2  = "ConsensusData/AllClusterings"+str(self.graphWidget.Syllable)+str(4)+"Heatmap.tsv"

            with open(name, 'w') as outfile:
                kValues = 'day' #K-values
                timeSteps = 'hour' #hour
                value = 'value'
                outfile.write("{}\t{}\t{}\n".format(timeSteps,kValues,value))
                for i,j in self.DeltaAreaTimestep.iteritems():
                    KDict = j
                    print i, KDict
                    for k,value in KDict.iteritems():
                        outfile.write("{}\t{}\t{}\n".format(i,k,value))

            outfile.close()

            with open(name2, 'w') as outfile:
                pickle.dump(self.FinalPartiction, outfile)

            print name2, "is the file name for all data", "look at the K value data and choose Wisely:)"

        if k == 1: 
            partition = dict()
            for i in range(len(distances)):
                partition[i] = 0

            self.partition1[1] = partition
        self.FinalPartiction[self.Timestep] = self.partition1

        return self.partition1[k]

    def run_cluster(self, num_clusters, subsamples, subsample_fraction, norm_var, kwds):
        """
    
        Run the clustering routines, generate a heatmap of the consensus matrix, and fill the logs with cluster information.
    
        Each time this is run it will create a logfile with the number of clusters and subsamples in its name.  This contains
        information on which samples where clustered together for that particular K value.
    
        Usage: self.run_cluster(num_clusters, subsamples, subsample_fraction, norm_var, kwds)
    
            num_clusters        - K value, or the number of clusters for the clustering functions to find for each subsample.
            subsamples          - The number of subsampling iterations to run.  In each subsample, the genes, samples, or both may
                                  be randomly selected for clustering.  This helps to ensure robust clustering.  More subsamples, more
                                  robust clusters.
            subsample_fraction  - The fraction of SNPs, samples, or both to take each subsample.  0.8 is a good default.
            norm_var            - Boolen variable.  If True, , genes will be standardised to have variance 1 over all samples
                                  each clustering iteration.
            kwds                - Additional options to be sent to cluster.ConsensusCluster
    
        It's probably a very bad idea to subclass run_cluster.  The _report and _save_hmap functions are almost certainly what you want.
    
        """
        colour_map = None
        args = locals()

        del args['self']
        args.update(kwds)

        #Actual work
        clust_data = ConsensusCluster(self.ParsedData, **args)

        self.partition = self._report(clust_data,**args)

        self.histogram(clust_data.consensus_matrix)
        self.consensusCDF(num_clusters, clust_data.consensus_matrix)
        self.area1(num_clusters, clust_data.consensus_matrix)

        clust_data._reset_clusters() #Set cluster_ids to None
        return self.partition

    def _report(self, clust_data, **kwds):
        """

        _report is called by run_cluster after each clustering set at a particular k-value is complete.

        Its job is to inform the user which clusters went where.  This can be done to the screen and to the logfile using console.log()

        Subclassing:

            @only_once
            def _report(self, clust_data, console, **kwds):

                etc...

            clust_data.datapoints is a list of SampleData objects, each of which has a cluster_id attribute.  This attribute indicates
            cluster identity, and any SampleData objects that share it are considered to be in the same cluster.  This doesn't have to be
            1, 2, 3...etc.  In fact, it doesn't have to be a number.

            See display.ConsoleDisplay for logging/display usage.

            You may want to subclass _report if you want to report on additional information, or if you simply want to turn this logging feature off.

        """

        #SNR Threshold
        threshold = 0.5
        
        #Initialise the various dictionaries
        colour_map = kwds['colour_map']

        if hasattr(self, 'defined_clusters'):
            sample_id_to_cluster_def = {}

            for cluster_def in self.defined_clusters:
                for sample_id in self.defined_clusters[cluster_def]:
                    sample_id_to_cluster_def[sample_id] = cluster_def

        cluster_sample_ids = dict()
        cluster_sample_indices = dict()
    
        for clust_obj in [ (clust_data.datapoints[x].sample_id, clust_data.datapoints[x].cluster_id, x) for x in clust_data.reorder_indices ]:
            sample_id, cluster_id, sample_idx = clust_obj

            cluster_sample_ids.setdefault(cluster_id, []).append(sample_id)
            cluster_sample_indices.setdefault(cluster_id, []).append(sample_idx)
    
        #Start writing the log
        # print "\nClustering results"
        # print "---------------------"
    
        # print "\nNumber of clusters: %s\nNumber of subsamples clustered: %s\nFraction of samples/features used in subsample: %s" % (kwds['num_clusters'], kwds['subsamples'], kwds['subsample_fraction'])
        # print "\n---------------------"
        # print "\nClusters"

        cluster_list = list(enumerate(cluster_sample_ids)) #(num, cluster_id) pairs

        partition = dict()
        for cluster in cluster_list:
            cluster_num, cluster_id = cluster

            # print "\nCluster %s :\n" % (cluster_num)
    
            for sample_id in cluster_sample_ids[cluster_id]:
                if hasattr(self, 'defined_clusters'):
                    pass
                    # print "\t%s\t\t%s" % (sample_id, sample_id_to_cluster_def[sample_id])
                else:
                    partition[int(sample_id)] = cluster_num
                    # print "\t%s" % sample_id

        M = clust_data.M
        
        buffer = []
        clsbuffer = []
        
        if hasattr(self.ParsedData, 'gene_names'):
            for i, j in comb(xrange(len(cluster_list)), 2):
                clust1, clust2 = cluster_list[i], cluster_list[j] #Still num, id pairs
                
                ttest = True # v0.5: On by default
                if kwds.has_key('ttest'):
                    ttest = kwds['ttest']

                # ratios = pca.snr(M, cluster_sample_indices[clust1[1]], cluster_sample_indices[clust2[1]], threshold=threshold, significance=ttest)
                
                # if ratios:
                #     buffer.append("\nCluster %s vs %s:" % (clust1[0], clust2[0]))
                #     buffer.append("--------------------\n")
                #     buffer.append("Gene ID\t\tCluster %s Avg\tCluster %s Avg\tSNR Ratio\tp-value" % (clust1[0], clust2[0]))

                #     for ratio, gene_idx, mean1, mean2, pval in ratios:
                #         buffer.append("%10s\t\t%f\t\t%f\t\t%f\t\t%s" % (self.ParsedData.gene_names[gene_idx], mean1, mean2, ratio, pval))

                if kwds.has_key('classifier') and kwds['classifier'] and ratios:
                    clsbuffer.append("\nCluster %s vs %s:" % (clust1[0], clust2[0]))
                    clsbuffer.append("--------------------\n")

                    classif_list = pca.binary_classifier(M, cluster_sample_indices[clust1[1]], cluster_sample_indices[clust2[1]], threshold)
                    #Returns (a, b), where a is w in (wi, i) pairs and b is w0
                    clsbuffer.append("w0 is %s" % classif_list[1])
                    clsbuffer.append("\nGene ID\t\tMultiplier")

                    for result in classif_list[0]:
                        clsbuffer.append("%10s\t%f" % (self.ParsedData.gene_names[result[1]], result[0]))
        # def write_buffer(name, desc, buf):
        #     console.new_logfile(name)
        #     console.log(desc, display=False)

        #     for line in buf:
        #         console.log(line, display=False)

        # if buffer:
        #     write_buffer('SNR Results - %s clusters - %s subsamples' % (kwds['num_clusters'], kwds['subsamples']), "SNR-ranked features with ratio greater than %s" % threshold, buffer)

        # if clsbuffer:
        #     write_buffer('Binary Classifier - %s clusters - %s subsamples' % (kwds['num_clusters'], kwds['subsamples']), "Based on SNR-ranked features with ratio greater than %s" % threshold, clsbuffer)
        return partition

    def histogram(self,matrix):
        matrix= np.tril(matrix)
        
        matrix1 = matrix.flatten()
        nonZeroValues = np.flatnonzero(matrix1)
        matrix1 = matrix1[nonZeroValues]
        
        unique, counts = np.unique(matrix1, return_counts=True)

        self.HistogramValues = dict()
        for i,j in np.asarray((unique, counts)).T: 
            self.HistogramValues[float(format(i, '.1f'))] = j

    def consensusCDF(self,K, matrix):
        self.count = 0
        lEN = len(matrix)
        Sum = 0
        Denominator = (lEN*(lEN-1))/2
        CDF = dict()
        matrix = np.tril(matrix)
        for c in self.HistogramValues.keys():
            Sum = 0 
            for cumalativeC in self.HistogramValues.keys():
                if cumalativeC <= c: 
                    Sum += self.HistogramValues[cumalativeC]
            CDF[c] = Sum/Denominator

        self.GlobalCDF[K] = copy.deepcopy(CDF)
        del lEN, matrix

    def area1(self, K, matrix):
        ListValues = self.HistogramValues.keys()
        ListValues.sort()
        Sum = 0
        for i in range(1,len(ListValues)-1):
            Sum += (ListValues[i] - ListValues[i-1]) * self.GlobalCDF[K][ListValues[i]]

        self.area[K] = Sum

    def createDeltaArea(self, kvalues):
        for k in kvalues:
            if k == 2: 
                self.deltaArea[k] = self.area[k] 
            else: 
                try: 
                    if not(self.area[k] == 0): 
                        self.deltaArea[k] = (self.area[k+1] - self.area[k])/self.area[k]
                    else: 
                        self.deltaArea[k] = 0
                except: 
                    print "k+1", self.area[k+1], "k",self.area[k]

        k = max(self.deltaArea.iteritems(), key=operator.itemgetter(1))[0]
        l = max(self.deltaArea.iteritems(), key=operator.itemgetter(1))[1]
        if l == 0: 
            k = 1 
        self.DeltaAreaTimestep[self.Timestep] = self.deltaArea

        pprint.pprint(self.deltaArea)
        print "Max",k
        return k

    def _preprocess(self):
        """

        Any data preprocessing that needs to be done BEFORE PCA should be done by subclassing this method

        Subclassing:

            def _preprocess(self):

                etc

        _preprocess shouldn't return anything, so any preprocessing should be done by extracting self.sdata.sample[x].data objects and
        putting them back when you're done.

        example: Take sequence data found by parser and convert it into a binary agreement matrix by comparing it to some reference
                 sequence
        
        This method does nothing on its own.

        """

        pass

    def _postprocess(self):
        """

        Any data postprocessing that needs to be done AFTER PCA but BEFORE CLUSTERING should be done by subclassing this method

        Subclassing:

            def _postprocess(self):

                etc

        _postprocess shouldn't return anything, so any postprocessing should be done by extracting self.sdata.sample[x].data objects and
        putting them back when you're done.

        example: Choose a random subset of the data to cluster, rather than the entire set

        This method does nothing on its own.

        """

        pass

def ColorToInt(r,g,b, a=255):
    return a << 24 | r << 16 | g << 8 | b

class communityDetectionEngine(QtCore.QObject):
    CalculateColors = QtCore.Signal(int)
    CalculateFormulae = QtCore.Signal(bool)

    def __init__(self,Graphwidget,distinguishableColors,FontBgColor):
        super(communityDetectionEngine, self).__init__()
        self.Graphwidget= Graphwidget	
        self.distinguishableColors = distinguishableColors
        self.ClusterAlgorithms = ClusterAlgorithms(Graphwidget)
        self.ConsensusMediator = ConsensusMediator(Graphwidget, self.ClusterAlgorithms)
        self.CustomCluster = CustomCluster(Graphwidget, self.ClusterAlgorithms)
        self.ConsensusCustomCluster = ConsensusCustomCluster(Graphwidget, ClusterAlgorithms)

        self.plotKValues = self.ConsensusMediator.plotKValues

        self.communityMultiple = defaultdict(list)
        self.PreComputeState = False
        self.PreComputeData = None

        self.FontBgColor= FontBgColor
        self.dend = -3 

        # PLACE WHERE CHANGE NUMBER OF DEFAULT COMMUNITIES 
        self.Number_of_Communities = 4

        self.dendo = []
        self.NumberOfTimesCalled = -1
        self.level = -1

    @Slot(object)
    def initializePrecomputationObject(self, PreComputeObject):
        self.PreComputeState = True
        self.PreComputeData = PreComputeObject

    @Slot()
    def unsetPreComputationDate(self):
        self.PreComputeState = False
        if self.PreComputeData:
            self.PreComputeData.clear()

    def ChangeCommunityColor(self, level = -1):
        """
        Problem to tackle is when one needs to tell about the colors
        1) When one is intending to freeze the colors inside of there 
        """
        self.level = level
        print "The Enire System is currently in the",self.Graphwidget.AnimationMode,"Mode"\
        "This is the level", level
        
        if self.Graphwidget.AnimationMode:
            """For now the signal emits stuff that will 
            calculate the stability of the communities detected
            Future Work is to add few lines of code for incorporating 
            NMI instead of Jaccards index that better captures the 
            similarities between communities
            """
            self.calculateNewGraphPropertiesAndCommunitiesForAnimation(level)
            self.CalculateFormulae.emit(True)
        else: 
            if self.Graphwidget.TowChanged:
                self.calculateNewGraphPropertiesAndCommunities(level)
                self.CalculateColors.emit(self.Graphwidget.TowValue)
            else:  
                self.calculateNewGraphPropertiesAndCommunities(level)
                self.GenerateNewColors(len(set(self.Graphwidget.partition.values())))
                self.Graphwidget.ColorForVisit(self.Graphwidget.partition)

        self.Graphwidget.TowChanged = False

    def calculateNewGraphPropertiesAndCommunitiesForAnimation(self,level):
        """
        This function is only for computing formulae purposes
        """
        """Change change the underlying data in the correlation data """
        # For the purpose of including in the animation section 
        # gets the graph data!! 
        self.Graphwidget.g =  self.Graphwidget.Graph_data().DrawHighlightedGraph(self.Graphwidget.EdgeSliderValue)
        self.Graphwidget.ColorNodesBasedOnCorrelation = False 
        if not(self.PreComputeState):
            self.Graphwidget.partition=self.resolveCluster(self.Graphwidget.ClusteringAlgorithm,self.Graphwidget.g, self.Number_of_Communities)
        else: 
            self.Graphwidget.partition=copy.deepcopy(self.PreComputeData[self.Graphwidget.TimeStep])
        """ 
        Does this data change everytime something is initialized??
        """

        if self.PreComputeState:
            self.Graphwidget.PartitionOfInterest=copy.deepcopy(self.PreComputeData[self.Graphwidget.TimeStep]) 
        else: 
            self.Graphwidget.PartitionOfInterest=copy.deepcopy(self.Graphwidget.partition)
    """
    Bering referenced by many classes of methods
    """
    def resolveCluster(self, value, graph, Number_of_Communities = 4, precomputeTimestep=0):

        Number_of_Communities = self.Number_of_Communities
        if value == 0: 
            """Louvain"""
            print "Using Louvain for Community Analysis"
            partition=cm.best_partition(graph)
        elif value == 1: 
            """Hierarchical"""
            partition=self.ClusterAlgorithms.HierarchicalClustering(graph)
        elif value == 2:
            """Kmeans algorithm"""
            partition= self.ClusterAlgorithms.computeKmeans(Number_of_Communities,graph)
        elif value == 3: 
            partition= self.ClusterAlgorithms.computeKcliques(Number_of_Communities,graph)
        elif value == 4: 
            partition = self.ConsensusMediator.prepareClsuterData(graph,precomputeTimestep, self.Graphwidget.Syllable)
        elif value == 5: 
            partition = self.CustomCluster.prepareClsuterData(graph, self.Graphwidget.TimeStep, self.Graphwidget.Syllable)
        elif value == 6: 
            partition = self.ConsensusCustomCluster.prepareClsuterData(graph, self.Graphwidget.TimeStep, self.Graphwidget.Syllable)

        return partition

    def calculateNewGraphPropertiesAndCommunities(self,level):
        self.Graphwidget.g =  self.Graphwidget.Graph_data().DrawHighlightedGraph(self.Graphwidget.EdgeSliderValue)
        self.Graphwidget.ColorNodesBasedOnCorrelation = False 
        self.Graphwidget.partition=self.resolveCluster(self.Graphwidget.ClusteringAlgorithm,self.Graphwidget.g, self.Number_of_Communities)

        level = len(self.dendo)-1

        self.Graphwidget.MaxDepthLevel = level
        
        if not(level == -1): 
            if level > len(self.dendo)-1:
                level = len(self.dendo)-1 
                
    def updateCommunityColors(self,counter,TowPartition=None):
        """Finds out the length of one length ommunities in the algorithm spitted out by louvain"""
        self.communityMultiple.clear()
        for key,value in self.Graphwidget.partition.items():
            self.communityMultiple[value].append(key)
        k=0

        self.Graphwidget.communityMultiple.clear()
        self.Graphwidget.communityMultiple = self.communityMultiple
        self.oneLengthCommunities =[]
        
        for i in range(counter):
            if len(self.communityMultiple[i]) == 1: 
                self.oneLengthCommunities.append(i)
                for j in self.communityMultiple[i]:
                    pass
                k=k+1
        return k

    def timeStepColorGenerator(self, counter, TowPartition = None):
        self.Graphwidget.ColorVisit = []
        l = 1 
        self.Graphwidget.oneLengthCommunities = []

        for i in range(counter):
            if i in self.Graphwidget.oneLengthCommunities: 
                r, g, b = 255,0,0
                self.Graphwidget.clut[i] = (255 << 24 | r << 16 | g << 8 | b)
                self.Graphwidget.FontBgColor[i] = ColorBasedOnBackground(r,g,b)
                self.Graphwidget.ColorVisit.append((r,g,b,255))
            else:
                r, g, b = self.distinguishableColors[l]
                self.Graphwidget.FontBgColor[i] = ColorBasedOnBackground(r,g,b)
                self.Graphwidget.clut[i] = (255 << 24 | r << 16 | g << 8 | b)
                self.Graphwidget.ColorVisit.append((r,g,b,255))
                l = l + 1

    def timeStepAnimationGenerator(self,counter, Assignment, communityMultiple): 
        clut = copy.deepcopy(self.Graphwidget.clut)
        ColorVisit = copy.deepcopy(self.Graphwidget.ColorVisit)

        # have to pass the tests for these different criterias 
        assert communityMultiple == self.Graphwidget.partition
        assert communityMultiple == self.Graphwidget.PartitionOfInterest
        assert len(Assignment.keys()) == len(set(communityMultiple.values()))

        if len(clut) > 2 and len(Assignment)>2:
            self.Graphwidget.clut = np.zeros(len(self.Graphwidget.partition.keys()))
            self.Graphwidget.ColorVisit = defaultdict(list)
            l = 1 

            for key, Color in Assignment.items():
                if not(isinstance(Color, tuple)):
                    #Stuff where the previous color vist of graph partition should be included
                    self.Graphwidget.clut[key] =  clut[Color] 
                    self.Graphwidget.ColorVisit[key] = ColorVisit[Color]
                else:
                    self.Graphwidget.clut[key] = ColorToInt(Color[0], Color[1], Color[2])
                    self.Graphwidget.ColorVisit[key] = (Color[0], Color[1], Color[2])

    def AssignCommuntiesFromDerivedFromTow(self,TowPartition, TowInducedGraph,\
                                            TowMultipleValue, TowDataStructure,\
                                            timestep,syllable):
        self.Graphwidget.g =  copy.deepcopy(TowDataStructure)
        self.Graphwidget.ColorNodesBasedOnCorrelation = False 

        self.Graphwidget.partition.clear() 
        self.Graphwidget.partition= copy.deepcopy(TowPartition)

        if self.level == -1:
            self.level = len(self.dendo)-1
            self.Graphwidget.DendoGramDepth.emit(self.level)
        
        if not(self.level == -1): 
            if self.level > len(self.dendo)-1:
                self.level = len(self.dendo)-1 

    def changeClusterValue(self, Cluster):
        self.Number_of_Communities = Cluster

    def ChangeGraphDataStructure(self):
        self.Graphwidget.Graph_data().setdata(self.Graphwidget.correlationTable().data)

    def ChangeGraphWeights(self):
        for edge in self.Graphwidget.edges:
            edge().setWeight(self.Graphwidget.Graph_data().ThresholdData[edge().sourceNode().counter-1][edge().destNode().counter-1])
            edge().update()

    def GenerateNewColors(self,counter):
        k=self.updateCommunityColors(counter)
        self.Graphwidget.ColorVisit = []
        l = 1 
        for i in range(counter):
            if i in self.Graphwidget.oneLengthCommunities: 
                r, g, b = self.distinguishableColors[0]
                self.Graphwidget.clut[i] = (255 << 24 | r << 16 | g << 8 | b)
                self.Graphwidget.FontBgColor[i] = self.FontBgColor[0]
                self.Graphwidget.ColorVisit.append((r,g,b,255))
            else:
                r, g, b = self.distinguishableColors[l]
                self.Graphwidget.FontBgColor[i] = self.FontBgColor[l]
                self.Graphwidget.clut[i] = (255 << 24 | r << 16 | g << 8 | b)
                self.Graphwidget.ColorVisit.append((r,g,b,255))
                l = l + 1

    """universal algorithm for font based on backgroud color
    http://codepen.io/WebSeed/full/pvgqEq/"""
def ColorBasedOnBackground(r,g,b):
    """Perceptive lumincance"""
    a = 1 - (0.299*r + 0.587*g + 0.114*b)/255
    if  (a < 0.5):
        """dark bg"""
        return (255 << 24 | 0 << 16 | 0 << 8 | 0)
    else:
        """Light bg"""
        return (255 << 24 | 255 << 16 | 255 << 8 | 255)
