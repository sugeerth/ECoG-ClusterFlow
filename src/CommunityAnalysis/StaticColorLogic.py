import numpy as np
import pprint
import weakref
import math
from collections import defaultdict
from PySide import QtCore, QtGui
from collections import OrderedDict
import copy
import csv
import math
from collections import deque
from PySide.QtCore import *
import simplejson
import traceback
from kmedoids import ClusterAlgorithms as cl
from graphviz import Digraph

from OpenGL.GLUT import *
from OpenGL.GL import *

from time import time
from math import *

from TrackingGraph.CommunitiesNode import CommunityGraphNode
from LegacyAPI import LegacyAPI

"""
Stuff to do include 
1) Colors like communties
2) interactive communities
http://arxiv.org/pdf/1407.5105.pdf
"""
import community as cm
try:
	# ... reading NIfTI 
	import nibabel as nib
	import numpy as np
	import pyqtgraph as pg
	# ... graph drawing
	import networkx as nx
	import operator
	from PySide.QtCore import *
	from PySide.QtGui import *
	from PySide.QtOpenGL import *
except:
	print "Couldn't import all required packages. See README.md for a list of required packages and installation instructions."
	raise

"""Class responsible for analysis of communities across timesteps"""

Capture = [0,15]
Interval = 13

FileNames = [('/Users/sugeerthmurugesan/Sites/Sankey/JSON_1.json',0,11),('/Users/sugeerthmurugesan/Sites/Sankey/JSON_2.json',12,23)\
,('/Users/sugeerthmurugesan/Sites/Sankey/JSON_3.json',23,34),('/Users/sugeerthmurugesan/Sites/Sankey/JSON_4.json',34,45)\
,('/Users/sugeerthmurugesan/Sites/Sankey/JSON_5.json',45,56)]

HeatmapFilename = "/Users/sugeerthmurugesan/Sites/Sankey/DeltaAreaChange4Heatmap.tsv"

ElectrodeSignalDataName = 'muDat'

timestep = 12
THRESHOLD_VALUE_TRACKING_GRAPH = 0.3
NumberOfSelectedEletrodes = 30
Number_of_Communities = 4
WIDTH = 1200
HEIGHT = 350
uniforms = [b"bending", b"folding", b"f_min", b"f_max", b"fold_map", b"circle"]
locations = {}
MT = -1 
bending = 200
folding_factors = [1] * 12
counter =  0

Width_value = 0 
Similarity = [[0.0]*16 for _ in xrange(16)] 
No_Of_Elements = [[0.0]*16 for _ in xrange(16)]


class SimilarityData(object): 
	"""
	Core method to compute the similarity between visualizations
	"""

	def computeSimilarityMatrices(self, CommunityObject):
		Kappa_matrix = np.zeros((len(CommunityObject.dataAccumalation), len(CommunityObject.communityMultiple)))
		
		for community1, nodes1 in CommunityObject.dataAccumalation.items(): 
			for community2, nodes2 in CommunityObject.communityMultiple.items():
				"""
				Complicated Simliarity metric for threshold
				"""
				elements = len(list(set(nodes1).intersection(nodes2)))
				Numerator = elements
				Denominator = len(set(nodes1)) + len(set(nodes2))
				val = float(float(Numerator)/float(Denominator))
				val2 = CommunityObject.SimiliarElements(nodes1,nodes2, community1, community2)
				No_Of_Elements[community1][community2] = val2
				Kappa_matrix[community1][community2] = val
		return Kappa_matrix,self.ThresholdAssignment(Kappa_matrix, CommunityObject) 

	def rook_is_ok_at(self,matrix, row, col):
		a= matrix.any(axis=1)
		b = matrix.any(axis=0) 

		if not(a[row]) and not(b[col]):
			return True 
		return False

	def NRooksProblem(self, height, width, iterations):
		global PermuteIndexes
		inverse = False

		def recurse(row, k):
			if (k == 0):
				self.solutions=+1 
				return
			for i in range(row,height):
				for j in range(0,width):
					if self.rook_is_ok_at(matrix, i,j):
						if inverse:
							Solutions[self.solutions].append((j,i))
						else:
							Solutions[self.solutions].append((i,j))
						matrix[i][j] = 1
						recurse(i+1, k-1)
						matrix[i][j] = 0
		k = height
		try: 
			if PermuteIndexes[(height,width)]:
				pass
		except KeyError:
			self.solutions = 0
			if height > width: 
				swap(height,width)
				inverse = True
			matrix=np.zeros((height,width))
			Solutions= [] * iterations
			recurse(0,height)
			PermuteIndexes[(height,width)] = copy.deepcopy(Solutions)
			assert self.solutions == iterations

	def permuteMatrices(self,height, width, matrix, colorAssignment):
		iterations = 0
		if height > width:
			iterations = math.factorial(height)/math.factorial(height - width)
			for i in range(height):
				for j in range(width):	
					pass
		elif width > height:
			iterations = math.factorial(width)/math.factorial(width - height)
		else: 
			iterations = math.factorial(width)/math.factorial(width - height)
		self.NRooksProblem(height,width, iterations)
		assert len(PermuteIndexes[(height,width)]) == iterations
		Sum = 0
		AllSum = []
		for i in PermuteIndexes[(height,width)]: 
			Sum = 0
			for j,k in i: 
				if matrix[j][k] == -1:
					matrix[j][k] = 0
				Sum = Sum + matrix[j][k]
			AllSum.append(Sum)
		Assignment = PermuteIndexes[(height,width)][np.argmax(AllSum)]
		return Assignment

	"""
	@/ Assign colors to different nodes are created, 
	@/ this is based on the colors that are previously calculated
	"""
	def AssignColors(self, CommunityObject, Kappa_matrix, Assignment, PreviousNodes):
		"""
		Color assignementt is happengin one by one, which should not be the case
		"""
		KappaMatrixForComputation = []
		PermuteDict = dict()
		PermuteAssignment = []
		x, y = Kappa_matrix.shape
		
		KappaMatrixForComputation = copy.deepcopy(Kappa_matrix)
		ColorAssignment = dict()
		if x >= 1 and y >= 1: 
			values= KappaMatrixForComputation < CommunityObject.thresholdValue  # Where values are low
			KappaMatrixForComputation[values] = 0
			height = len(KappaMatrixForComputation[:,0])
			width = len(KappaMatrixForComputation[0])

			"""
			Naive algorithm to assign stuff
			"""
			while True:
				if (len(ColorAssignment.keys()) == width) or (len(ColorAssignment.values()) == height) or KappaMatrixForComputation.any() == 0:
					break 
				m,n = np.unravel_index(KappaMatrixForComputation.argmax(), KappaMatrixForComputation.shape)
				ColorAssignment[n] = m
				KappaMatrixForComputation[m] = 0
				KappaMatrixForComputation[:,n] = 0 
				KappaMatrixForComputation[m,n] = 0

			"""
			Permutation based stuff
			"""
			if self.Permute:
				PermuteAssignment = self.permuteMatrices(height,width, Kappa_matrix, ColorAssignment)
			else:
				PermuteAssignment = dict()

			for l,q in PermuteAssignment:
				PermuteDict[q] = l

			"""adding new colors"""
			AssignedValues = set(ColorAssignment.values())
			AssignedKeys = set(ColorAssignment.keys())

			allElementsHeight = set(range(0,height))

			DifferenceElements = allElementsHeight - AssignedValues

			"""Case when some community colors are born again"""
			allElements = set(range(0,len(Kappa_matrix[0])))
			NewBornCommunities = allElements-AssignedKeys
			
			if len(allElementsHeight) > len(AssignedValues) and len(DifferenceElements) > 0:
				"""These things have a color with them so can store them into a queue"""
				for i in DifferenceElements:
					if PreviousNodes:
						try: 
							color = PreviousNodes[i].colorvalue.getRgb()
							CommunityObject.communitiesThatDie.append((color[0],color[1],color[2]))
						except AttributeError: 
							color = PreviousNodes[i].CommunityColor.getRgb()
							CommunityObject.communitiesThatDie.append((color[0],color[1],color[2]))

			for q in NewBornCommunities:
				if CommunityObject.NewCommunitiesToBeAssigned:
					ColorTobeAssigned = CommunityObject.NewCommunitiesToBeAssigned.popleft()
					ColorAssignment[q] = ColorTobeAssigned
					PermuteDict[q] = ColorTobeAssigned

				else: 
					if CommunityObject.communitiesThatDie:
						ColorAssignment[q] = CommunityObject.communitiesThatDie.popleft()
					else: 
						ColorAssignment[q] = (0,0,0)

			return ColorAssignment

	def ThresholdAssignment(self, Kappa_matrix, CommunityObject):
		from collections import defaultdict
		Assignment = defaultdict(list)

		for i in range(len(CommunityObject.dataAccumalation)):
			for j in range(len(CommunityObject.communityMultiple)): 
			# if i in Assignment.keys():
			# 	continue
				# if any(j in z for z in Assignment.values()):
				# 	continue
				print CommunityObject.thresholdValue, Kappa_matrix[i][j]  
				if CommunityObject.thresholdValue < Kappa_matrix[i][j]:
					Assignment[i].append(j)
				elif CommunityObject.thresholdValue > Kappa_matrix[i][j]:
					continue	

		return Assignment



class LogicForTimestep(object): 
	EdgeType = ['QtCore.Qt.SolidLine, QtCore.Qt.DashLine, QtCore.Qt.DotLine, QtCore.Qt.DashDotLine,QtCore.Qt.DashDotDotLine, QtCore.Qt.CustomDashLine'] 
	Data = SimilarityData()
	Data.Permute = False

	def changeEdgeProperties(self, CommunityObject, PreviousNodes):
		"""
		Previous data is self.NodeIds and current data self.NodeIds1
		"""
		Kappa_matrix, Assignment = self.Data.computeSimilarityMatrices(CommunityObject)
		# assert not(Kappa_matrix.any() == 0) 
		ColorAssignment = self.Data.AssignColors(CommunityObject, Kappa_matrix, Assignment, PreviousNodes)
		return Kappa_matrix, Assignment, ColorAssignment

	def changeColorsForNodesJustRendered(self, CommunityObject, ColorAssignment, Nodes, PreviousNodes, Kappa_matrix):
		"""
		Getting the data that needs to be visualized
		"""
		for toCommunity,fromCommunity in ColorAssignment.iteritems():
			try:
				if not(isinstance(fromCommunity,tuple)):
					pass
					# Nodes[toCommunity].PutColor(PreviousNodes[fromCommunity].CommunityColor)	
				else:
					"""Changing Color Important"""
					color = QtGui.QColor(fromCommunity[0],fromCommunity[1],fromCommunity[2])
					pass
					Nodes[toCommunity].PutColor(color)
					continue
			except IndexError as e:
				continue
