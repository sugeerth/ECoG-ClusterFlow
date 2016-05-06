import numpy as np
import pprint
import weakref
import math
from collections import defaultdict
from PySide import QtCore, QtGui
from collections import OrderedDict
from copy import copy
from collections import deque

"""
Stuff to do include 
1) Colors like communties
2) interactive communities

"""
import community as cm
try:
	# ... reading NIfTI 
	import nibabel as nib
	import numpy as np
	# ... graph drawing
	import networkx as nx
	import operator
except:
	print "Couldn't import all required packages. See README.md for a list of required packages and installation instructions."
	raise

"""Class responsible for analysis of communities across timesteps"""

timestep = 64
NumberOfSelectedEletrodes = 14


def SimiliarElements(nodes1, nodes2):
    intersectingElements = list(set(nodes1).intersection(nodes2))
    if (len(intersectingElements) == 0):
        return -1
    return len(intersectingElements)


class CommunitiesAcrossTimeStep(QtGui.QGraphicsView):
	def __init__(self,widget, electrode, electrodeUI):
		QtGui.QGraphicsView.__init__(self)
		self.Graph_interface = widget

		self.electrodeUI = electrodeUI
		self.electrode = electrode
		
		self.widget = widget
		
		self.distinguishableColors = self.widget.communityDetectionEngine.distinguishableColors

		self.setWindowTitle('Analysis Across Timesteps')
		# self.g = graphData
		self.Order =[]
		self.height = 250 
		self.counter = 0
		self.communityMultiple = defaultdict(list)
		self.width = 800
		self.timestepMatrix = np.zeros((timestep,NumberOfSelectedEletrodes,NumberOfSelectedEletrodes)) 
		self.dataAccumalation = defaultdict(list)
		self.AllNodes = None
		self._2NodeIds = []
		self.array_value = 0
		self.NodeIds = []
		self.stabilityValues = []
		self.height = 0 

		# Tow
		self.TowPartitionValue = dict()
		self.TowGraphdataStructure = []
		self.TowGraph = []
		self.TowMultiple = defaultdict(list)

		self.stabilityValues= []


		self.NewCommunitiesToBeAssigned = deque([j for i,j in enumerate(self.distinguishableColors) if i > 14])
		self.communitiesThatDie = deque()

		self.width = 0
		self.ReEnterColumnLoop = True
		self.setMinimumSize(500, 250)
		self.centerPos = (self.width/2 + 1, self.height -10)
		self.dendogram = None
		self.initUI()

	def initUI(self):
		scene = QtGui.QGraphicsScene(self)

		scene.setItemIndexMethod(QtGui.QGraphicsScene.NoIndex)
		scene.setSceneRect(-200, -200, 400, 400)
		self.setScene(scene)
		self.scene = scene
		self.setCacheMode(QtGui.QGraphicsView.CacheBackground)
		self.setRenderHint(QtGui.QPainter.Antialiasing)
		self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
		self.setResizeAnchor(QtGui.QGraphicsView.AnchorViewCenter)
		self.setInteractive(True)
		
		self.NodeIds = []
		self.centrality = []
		self.Scene_to_be_updated = scene
		self.setCacheMode(QtGui.QGraphicsView.CacheBackground)
		i = 0
		self.setSceneRect(self.Scene_to_be_updated.itemsBoundingRect())
		self.setScene(self.Scene_to_be_updated)
		self.fitInView(self.Scene_to_be_updated.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)
		self.scaleView(math.pow(2.0, -900/ 1040.0))
		self.initiateMatrix()

	def CalculateClustersForTow(self,value):
		"""The value is between 0 to 64 and need to find out a way where calculates all the clusters at this point and stores it here ina 
		a data structure
		Every time something is called this gets invoked  and the formulae is calculated
		Changes in color should be happening right here
		1) Retreive the data from the correlationTable"""

		print value, self.electrode.syllableUnit
		self.array_value
		self.TowGraphdataStructure = self.widget.correlationTable().FindAbsoluteValue(self.widget.correlationTable().dataProcess.ElectodeData['mat_ncv'][self.electrode.syllableUnit-1][value])
		# self.TowPartitionValue
		self.TowGraph = nx.from_numpy_matrix(self.TowGraphdataStructure)

		
		# self.TowPartitionValue = cm.best_partition(self.TowGraph)

		self.TowMultiple.clear()
		
		for key,value in self.TowPartitionValue.items():
			self.TowMultiple[value].append(key)
		
		pprint.pprint(self.TowMultiple)
		self.dataAccumalation = copy(self.TowMultiple)
		self.initiateMatrix()
		# self.widget.communityDetectionEngine.timeStepColorGenerator(Assignment,len(set(self.Graph_interface.partition.values())))

	def CalculateStabilityOfMatrices(self,state):
		""" """
		if state:


			self.stabilityValues.append(Value)

	"""Defined as the matrix that will be computed at every timestep
	Useful for analysis between timesteps"""
	def initiateMatrix(self):
		print "canged"
		self.counter = self.counter + 1
		self.communityMultiple.clear()

		try: 
			for key,value in self.Graph_interface.partition.items():
				self.communityMultiple[value].append(key)
			if self.electrode.timeStep > 1: 
				try: 
					# change array value for toggling between discrete and continuous
					if not(self.communityMultiple == self.dataAccumalation):
						for community1,nodes1 in self.dataAccumalation.items():
							for community2,nodes2 in self.communityMultiple.items():		
									self.timestepMatrix[self.array_value][community1][community2] = SimiliarElements(nodes1,nodes2)

					self.CalculateWidthAndHeight(self.array_value)
					if self.height > 2: 
						self.AffinityMapping(self.array_value)
						
					self.dataAccumalation.clear()	
					self.array_value = self.array_value + 1
					self.dataAccumalation = copy(self.communityMultiple)
				except IndexError as e:
					print "Index error({0}): {1}".format(e.errno, e.strerror)
		except AttributeError as e:
			pass

		"""The data is of two timestep in nature

		This computation is between counter-1 and counter timesteps. 
		The best way is run this function on a separate thread or use parallel programming. 

		"""

	def CalculateWidthAndHeight(self,array_value):
		# print "NOW HERE "
		for i in range(len(self.timestepMatrix[array_value])):
			if (self.timestepMatrix[array_value][i,0] == 0):
				# print self.timestepMatrix[array_value][i,0]
				self.height = i
				break
			for j in range(len(self.timestepMatrix[array_value])):
				if (self.timestepMatrix[array_value][0,j] == 0):
					self.width = j
					break
		print "Height and width are",self.height, self.width

		if self.width > self.height:
			print "New colors have to assigned"
		elif self.height > self.width:
			print "some old communities will be destroyed"
		else:
			print "Same number so no communities destroyed or born again" 
	"""Identifies the intersecting elements in the two lists"""

	def AffinityMapping(self,array_value):
		matrix = self.timestepMatrix[array_value][:self.height,:self.width]
		# print matrix
		MaxValuesAssignment = -3
		indexAssign = -1
		Index= 0
		ProportionScore = -1
		Assignment = dict()
		for i in range(self.width):
			if i in Assignment.keys():
				continue
			MaxValuesAssignment = -3
			CurrentColumn = matrix[:,i]
			# print "Column",CurrentColumn
			ColumnList = CurrentColumn.tolist()
			
			while (self.ReEnterColumnLoop):
				Max , Index= self.max1(ColumnList, Assignment)
				# Index = ColumnList.index(Max)
				self.AssignValues(ColumnList,Max,Index,Assignment,i,matrix)
			# print Assignment
			self.ReEnterColumnLoop = True

		AssignedValues = set(Assignment.values())
		AssignedKeys = set(Assignment.keys())

		if self.height > self.width:
			"""Case when some community colors are destroyed"""
			# print "performing stuff when some communities are destroyed"
			allElements = set(range(0,self.height))
			destroyedCommunities = allElements-AssignedValues
			# print "destroyed communities",destroyedCommunities

			for i in destroyedCommunities: 			
				self.communitiesThatDie.append(self.self.Graphwidget.ColorVisit[i][:3])

		elif self.width > self.height:
			"""Case when some community colors are born again"""
			# print "performing stuff when some communities are reintitialized"
			allElements = set(range(0,self.width))
			NewBornCommunities = allElements-AssignedKeys
			# print AssignedKeys
			# print "NewBorn communities",NewBornCommunities

			for i in NewBornCommunities:
				if self.communitiesThatDie:
					ColorTobeAssigned = self.communitiesThatDie.popleft()
					Assignment[i] = ColorTobeAssigned
				else: 
					ColorTobeAssigned = self.NewCommunitiesToBeAssigned.popleft()
					Assignment[i] = ColorTobeAssigned
		else: 
			print "no problem"


		# for i,j in Assignment.items():
		# 	print  i,"--->",j
		# 	pass

		""" Communities that die """

		# print "Communities that are dead"
		pprint.pprint(self.communitiesThatDie)
		# print "Communities that are reborn again"
		# pprint.pprint(self.NewCommunitiesToBeAssigned)
		# Index = [i for i, j in enumerate(CurrentColumn) if j == Max]

		""" Fix me for now just happen to coment this line out because at every timestep you just need the same colors"""
		self.widget.communityDetectionEngine.timeStepColorGenerator(Assignment,len(set(self.Graph_interface.partition.values())))

	def AssignValues(self,ColumnList,Max,Index,Assignment,columnIndex,matrix):
		"""My own score comparison with other scores"""

		if len(Assignment.values()) == self.height:
			self.ReEnterColumnLoop = False
			return  
		ProportionScore = self.CalculateProportionScore(ColumnList,Assignment,Max)
		# print "My Proportion Score", ProportionScore

		element = self.MaxRightHorizontalLines(columnIndex,Index,Assignment,matrix,Max,ProportionScore)
		# print "element",element, type(element)
		# print "Max values",element.values()

		try: 
			if ProportionScore > max(element.values()):
				# print "Index ",Max , Index, columnIndex
				Assignment[columnIndex] = Index 
				self.ReEnterColumnLoop = False
			else: 
				# print "Entering"
				# print "Values is ",max(element.values())
				column = max(element, key=element.get)
				# column = element.keys().index(max(element.values()))
				# print "Found better values so going for it", Index,column,"This is the element table", element, "This is proportional value",max(element.values())
				Assignment[column] = Index	
				self.ReEnterColumnLoop = True
		except ValueError:
			# print "Enter here"
			Assignment[columnIndex] = Index 
			self.ReEnterColumnLoop = False
	"""
	1 2 3
	|------------leftIndex,rightIndex , 3--MaxValue 
	3 4 5
	4 2 1
	"""
	def MaxRightHorizontalLines(self,leftIndex,rightIndex,AssignmentValues,matrix, MaxValue,ProportionScore):
		MaxV = -3
		Element = dict()
		index = 0

		for i in range(leftIndex+1,self.width): 
			if (i in AssignmentValues.keys()):
				continue
			MaxV = -3 

			if matrix[rightIndex][i] >= MaxValue: 
				Column = matrix[:,i]
				ColumnList = Column.tolist()
				# print "Column",Column

				MaxE, index= self.max1(ColumnList,AssignmentValues)
				# index = ColumnList.index(MaxE)

				Value = self.CalculateProportionScore(ColumnList,AssignmentValues,matrix[rightIndex][i])
				# print "index -->",index,"Max element-->", MaxE,"But ",matrix[rightIndex][i],"Proportional value",Value,"in [",rightIndex,"][",i,"]"
				Element[i] = Value
				# Element[index] = MaxE

		return Element
			# print index, i , MaxE

		# Index = self.keywithmaxval(Element)
		# if ((max(Element.values()) == MaxValue) and (Index == leftIndex)): 	
		# 	print "It is the same"

		# return Index

	def keywithmaxval(self,d):
	     """ a) create a list of the dict's keys and values; 
	         b) return the key with the max value"""  
	     v=list(d.values())
	     k=list(d.keys())
	     return k[v.index(max(v))]

	@staticmethod
	def indexOf(ColumnList, Assignment):
		k=0
		for k,i in enumerate(ColumnList):
			if (k in Assignment.values()):
				continue
			if i > max1:
				max1 = i
				index = k 
		return k

	@staticmethod
	def max1(ColumnList, Assignment):
		k=0
		max1 = -3 
		index = 0
		for k,i in enumerate(ColumnList):
			if (k in Assignment.values()):
				continue
			if i > max1:
				max1 =i 
				index = k
		return max1, index

	def CalculateProportionScore(self,ColumnList,Assignment,Max):
		sum1 = self.sum1(ColumnList,Assignment)
		if not(sum1 == 0): 
			ProportionScore = Max/sum1
		else: 
			ProportionScore = 0
		return ProportionScore

	@staticmethod
	def sum1(List,Assignment):
		sum1 = 0
		k = 0 
		for k,i in enumerate(List):
			if i == -1: 
				continue
			if (k in Assignment.values()):
				continue
			sum1 = sum1 + i
		return sum1

	def PrintAccumalatedData(self,state):
		"""Plot a graph that will mapp all the stability values"""
		print self.stabilityValues  

	def wheelEvent(self, event):
		self.scaleView(math.pow(2.0, -event.delta() / 1040.0))

	def scaleView(self, scaleFactor):
		factor = self.matrix().scale(scaleFactor, scaleFactor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()
		if factor < 0.07 or factor > 100:
			return
		self.scale(scaleFactor, scaleFactor)
		del factor
