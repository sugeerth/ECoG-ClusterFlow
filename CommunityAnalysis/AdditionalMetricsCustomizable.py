import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import community as cm
import networkx as nx

class AdditionalMetricsCustomizable(object):
	    def __init__(self,Graphwidget):
	    	self.Graphwidget= Graphwidget	
	    def participation_coefficient(self,G, weighted_edges=False):
	        """"Compute participation coefficient for nodes.
	        
	        Parameters
	        ----------
	        G: graph
	          A networkx graph
	        weighted_edges : bool, optional
	          If True use edge weights
	        
	        Returns
	        -------
	        node : dictionary
	          Dictionary of nodes with participation coefficient as the value
	        
	        Notes
	        -----
	        The participation coefficient is calculated with respect to a community
	        affiliation vector. This function uses the community affiliations as determined
	        by the Louvain modularity algorithm (http://perso.crans.org/aynaud/communities/).
	        """
        	print G, "Why is this in int??" 
	        partition = cm.best_partition(G)
	        partition_list = []
	        for count in range(len(partition)):
	            partition_list.append(partition[count])
	        
	        n = G.number_of_nodes()
	        Ko = []
	        for node in range(n):
	            node_str = np.sum([G[node][x]['weight'] for x in G[node].keys()])
	            Ko.append(node_str)
	        Ko = np.array(Ko)
	        G_mat_weighted = np.array(nx.to_numpy_matrix(G))
	        G_mat = (G_mat_weighted != 0) * 1
	        D = np.diag(partition_list)
	        Gc = np.dot(G_mat, D)
	        Kc2 = np.zeros(n)
	        for i in range(np.max(partition_list) + 1):
	            Kc2 = Kc2 + (np.sum(G_mat_weighted * (Gc == i),1) ** 2)
        	P = np.ones(n) - (Kc2/(Ko **2))
        	import pprint
	        D = dict()
	        for i in range(len(P)):
	            D[i]=P[i]
	        return D