## Standard Python packages
#-*- coding: utf-8 -*-
from PySide import QtCore, QtGui
import pprint
try:
    # ... reading NIfTI 
    import numpy as np
    # ... graph drawing
    import networkx as nx

except:
    print "Couldn't import all required packages. See README.md for a list of required packages and installation instructions."
    raise

class GraphVisualization(QtGui.QWidget):
    """This class is responsible for GraphVisualization from the data given"""
    filename = ""
    regionSelected = QtCore.Signal(int)
    def __init__(self,data):
        super(GraphVisualization, self).__init__()
        self.data = data
        self.G = nx.from_numpy_matrix(self.data)  
        self.DrawHighlightedGraph()

    def setdata(self,data):
        self.data = data
        self.setG()

    def setG(self):
        self.G = nx.from_numpy_matrix(self.data)

    def Find_HighlightedEdges(self,weight = 0):
        self.ThresholdData = np.copy(self.data)
        low_values_indices = self.ThresholdData < weight  # Where values are low
        self.ThresholdData[low_values_indices] = 0
        # graterindices = [ (i,j) for i,j in np.ndenumerate(self.ThresholdData) if any(i > j) ] 
        # self.ThresholdData[graterindices[:1]] = 0
        # self.ThresholdData = np.tril(self.ThresholdData)
        # print self.ThresholdData, "is the data same??" 
        """
        test 2 highlighted edges there
        """
        # np.savetxt('test2.txt', self.ThresholdData, delimiter=',', fmt='%1.4e')
        self.g = nx.from_numpy_matrix(self.ThresholdData)  

    def DrawHighlightedGraph(self,weight=None):
        # print "Data in highlighted graph",self.data
        if not(weight):
            weight = self.data.min()
            # print "MinimumValue", weight
	self.Find_HighlightedEdges(0)
        return self.g
