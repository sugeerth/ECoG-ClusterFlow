import numpy as np
from collections import defaultdict
from PySide import QtCore, QtGui
import copy
from PySide.QtCore import *

from math import *

try:
	from PySide.QtCore import *
	from PySide.QtGui import *
	from PySide.QtOpenGL import *
except:
	print "Couldn't import all required packages. See README.md for a list of required packages and installation instructions."
	raise

"""Class responsible for analysis of communities across timesteps"""

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

		return Kappa_matrix,No_Of_Elements,self.ThresholdAssignment(Kappa_matrix, CommunityObject) 

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
		print x,y
		
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

			if self.Permute:
				PermuteAssignment = None
			else:
				PermuteAssignment = dict()

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
		if self.Permute:
			return PermuteDict
		else: 
			return ColorAssignment

	def ThresholdAssignment(self, Kappa_matrix, CommunityObject):
		Assignment = defaultdict(list)
		
		for i in range(len(CommunityObject.dataAccumalation)):
			for j in range(len(CommunityObject.communityMultiple)):
				if Kappa_matrix[i][j] > CommunityObject.thresholdValue :
					Assignment[i].append(j)
				elif Kappa_matrix[i][j] < CommunityObject.thresholdValue :
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
		Kappa_matrix, No_Of_Elements, Assignment = self.Data.computeSimilarityMatrices(CommunityObject)
		ColorAssignment = self.Data.AssignColors(CommunityObject, Kappa_matrix, Assignment, PreviousNodes)
		return Kappa_matrix,No_Of_Elements,Assignment, ColorAssignment

	def changeColorsForNodesJustRendered(self, CommunityObject, ColorAssignment, Nodes, PreviousNodes, Kappa_matrix):
		"""
		Getting the data that needs to be visualized
		"""
		for toCommunity,fromCommunity in ColorAssignment.iteritems():
			try:
				if not(isinstance(fromCommunity,tuple)):
					Nodes[toCommunity].PutColor(PreviousNodes[fromCommunity].CommunityColor)	
				else:
					color = QtGui.QColor(fromCommunity[0],fromCommunity[1],fromCommunity[2])
					Nodes[toCommunity].PutColor(color)
					continue
			except IndexError as e:
				continue
