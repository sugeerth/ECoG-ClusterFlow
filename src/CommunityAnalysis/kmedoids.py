import numpy as np
import random
import csv
import sys
import math
from collections import defaultdict
from PySide import QtCore, QtGui
from sys import platform as _platform
import weakref
import cProfile
import os
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import KMeans
import networkx as nx

from ConsensusClster.cluster import ConsensusCluster

import pprint

class ClusterAlgorithms(QtCore.QObject):
    DataChange = QtCore.Signal(bool)
    def __init__(self,widget):
        super(ClusterAlgorithms,self).__init__()
        self.widget = widget
        print "Setting up K-means engine "

    def computeKcliques(self,Number_of_clusters,data):
        partition = dict()
        nb_clusters = Number_of_clusters # this is the number of cluster the dataset is supposed to be partitioned into
        distances = nx.to_numpy_matrix(data)
        clusterid = nx.k_clique_communities(data)

        # print "k_Cliques",clusterid
        # uniq_ids = list(set(clusterid))
        # new_ids = [ uniq_ids.index(val) for val in clusterid]

        # for i,value in enumerate(new_ids):
        #     partition[i] = value
        return partition


    def ConsensusCluster(self, data, subsamples, subsample_fraction, norm_var, kvalues): 
        """
        Performs consensus clustering algorithms here!!!
        """
        return
        partition = dict()
        stuff = []
        nb_clusters = 0 # this is the number of cluster the dataset is supposed to be partitioned into
        distances = nx.to_numpy_matrix(data)

        for i in kvalues:
            clusterid, error, nfound = KMeans(distances, nclusters= i, npass=300)
            uniq_ids = list(set(clusterid))
            new_ids = [ uniq_ids.index(val) for val in clusterid]

            for i,value in enumerate(new_ids):
                partition[i] = value
            stuff.append(partition)

    def computeKmeans(self,Number_of_clusters,data, iterations = 100):
        partition = dict()
        nb_clusters = Number_of_clusters # this is the number of cluster the dataset is supposed to be partitioned into
        distances = nx.to_numpy_matrix(data)

        clusterid, error, nfound = KMeans(distances, nclusters= nb_clusters, npass=300)

        uniq_ids = list(set(clusterid))
        new_ids = [ uniq_ids.index(val) for val in clusterid]

        for i,value in enumerate(new_ids):
            partition[i] = value
        return partition

    def HierarchicalClustering(self,data):
        distances = nx.to_numpy_matrix(data)
        hierarchy = linkage(distances)
        print hierarchy,"HIERRATCJY"
        Z = dendrogram(hierarchy)
        print Z
        return hierarchy

    def computeKmeansCustom(self,Number_of_clusters,data, iterations = 100):
        distances = nx.to_numpy_matrix(data)
        clusters,medoids = self.cluster(distances,Number_of_clusters)
        new_ids = [ uniq_ids.index(val) for val in clusterid]

    def cluster(self,distances, k=3):
        m = distances.shape[0] # number of points
        # Pick k random medoids.
        curr_medoids = np.array([-1]*k)
        while not len(np.unique(curr_medoids)) == k:
            curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
        old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
        new_medoids = np.array([-1]*k)
       
        # Until the medoids stop updating, do the following:
        while not ((old_medoids == curr_medoids).all()):
            # Assign each point to cluster with closest medoid.
            clusters = self.assign_points_to_clusters(curr_medoids, distances)

            # Update cluster medoids to be lowest cost point. 
            for curr_medoid in curr_medoids:
                cluster = np.where(clusters == curr_medoid)[0]
                new_medoids[curr_medoids == curr_medoid] = self.compute_new_medoid(cluster, distances)

            old_medoids[:] = curr_medoids[:]
            curr_medoids[:] = new_medoids[:]
        return clusters, curr_medoids

    def assign_points_to_clusters(self,medoids, distances):
        distances_to_medoids = distances[:,medoids]
        clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
        clusters[medoids] = medoids
        return clusters

    def compute_new_medoid(self,cluster, distances):
        mask = np.ones(distances.shape)
        mask[np.ix_(cluster,cluster)] = 0.
        cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
        costs = cluster_distances.sum(axis=1)
        return costs.argmin(axis=0, fill_value=10e9)