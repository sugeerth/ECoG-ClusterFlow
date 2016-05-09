import numpy as np
import pprint
import weakref
import math
from collections import defaultdict
from PySide import QtCore, QtGui
from collections import OrderedDict
import copy
import csv
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
# from TrackingGraph.CommunitiesEdge import CommunitiesEdge
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

# FileNames = [('/Users/sugeerthmurugesan/Sites/Sankey/JSON_1.json',0,12),('/Users/sugeerthmurugesan/Sites/Sankey/JSON_2.json',12,24),\
# ('/Users/sugeerthmurugesan/Sites/Sankey/JSON_3.json',24,36),('/Users/sugeerthmurugesan/Sites/Sankey/JSON_4.json',36,48)\
# ,('/Users/sugeerthmurugesan/Sites/Sankey/JSON_5.json',48,60)]

FileNames = [('/Users/sugeerthmurugesan/Sites/Sankey/JSON_1.json',0,4),('/Users/sugeerthmurugesan/Sites/Sankey/JSON_2.json',4,8)\
,('/Users/sugeerthmurugesan/Sites/Sankey/JSON_3.json',8,12),('/Users/sugeerthmurugesan/Sites/Sankey/JSON_4.json',12,16)\
,('/Users/sugeerthmurugesan/Sites/Sankey/JSON_5.json',16,20)]

HeatmapFilename = "/Users/sugeerthmurugesan/Sites/Sankey/DeltaAreaChange4Heatmap.tsv"

# Changes for RealData
# ElectrodeSignalDataName = 'sigData'

ElectrodeSignalDataName = 'muDat'

timestep = 12
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
PermuteIndexes = dict()
PermuteIndexes[(1,1)] = [[(0,0)]]
PermuteIndexes[(1,2)] = [[(0,0)],[(0,1)]]
PermuteIndexes[(2,1)] = [[(0,0)],[(1,0)]]
PermuteIndexes[(2,2)] = [[(0,0),(1,1)],[(0,1),(1,0)]] 
PermuteIndexes[(3,2)] = [[(0,0),(1,1)],[(0,0),(2,1)], [(0,1),(1,0)], [(0,1),(2,0)], [(1,0),(2,1)], [(1,1),(2,0)]] 
PermuteIndexes[(2,3)] = [[(0,0),(1,1)],[(0,0),(1,2)], [(0,1),(1,0)], [(0,1),(1,2)], [(0,2),(0,0)], [(0,2),(1,0)]] 
PermuteIndexes[(1,3)] = [[(0,0)],[(0,1)],[(0,2)]]
PermuteIndexes[(3,1)] = [[(0,0)],[(1,0)],[(2,0)]]
PermuteIndexes[(3,3)] = [[(0,0),(1,1),(2,2)],[(0,0),(1,2),(2,1)], [(0,1),(1,0),(2,2)], [(0,1),(1,2),(2,0)], [(0,2),(1,0),(2,1)], [(0,2),(1,1),(2,0)]]  
PermuteIndexes[(3,4)] = [[(0,0),(1,1),(2,2)],[(0,0),(1,2),(2,1)], [(0,1),(1,0),(2,2)], [(0,1),(1,2),(2,0)], [(0,2),(1,0),(2,1)], [(0,2),(1,1),(2,0)]]  
PermuteIndexes[(4,4)] = [[(0,0),(1,1),(2,2),(3,3)],[(0,0),(1,1),(2,3),(3,2)],[(0,0),(1,2),(1,1),(3,3)],[(0,0),(1,2),(1,3),(3,1)],[(0,0),(1,3),(2,1),(3,2)]\
,[(0,0),(1,3),(2,2),(3,1)], [(0,1),(1,0),(2,2),(3,3)],[(0,1),(1,0),(2,3),(3,2)],[(0,1),(1,2),(2,0),(3,3)],[(0,1),(1,2),(2,3),(3,0)]\
,[(0,1),(1,3),(2,0),(3,2)],[(0,1),(1,3),(2,2),(3,0)],[(0,2),(1,0),(2,1),(3,3)],[(0,2),(1,0),(2,3),(3,1)],[(0,2),(1,1),(2,0),(3,2)]\
,[(0,2),(1,1),(2,3),(3,0)],[(0,2),(1,3),(2,0),(3,1)],[(0,2),(1,3),(2,1),(3,0)],[(0,3),(1,0),(2,1),(3,2)],[(0,3),(1,0),(2,2),(3,1)],\
[(0,3),(1,1),(2,0),(3,2)],[(0,3),(1,1),(2,2),(3,0)],[(0,3),(1,2),(2,0),(3,1)],[(0,3),(1,2),(2,1),(3,0)]]   
PermuteIndexes[(1,4)] = [[(0,0)],[(0,1)],[(0,2)],[(0,3)]]
PermuteIndexes[(2,4)] = [[(0,0),(1,1)],[(0,0),(1,2)], [(0,0),(1,3)], [(0,1),(1,0)], [(0,1),(1,2)], [(0,1),(1,3)], [(0,2),(1,0)]\
, [(0,1),(1,1)], [(0,2),(1,3)], [(0,3),(1,0)], [(0,3),(1,1)], [(0,3),(1,2)]]

Width_value = 0 
Similarity = [[0.0]*16 for _ in xrange(16)] 
No_Of_Elements = [[0.0]*16 for _ in xrange(16)]

class SimilarityData(object): 
	"""
	Core method to compute the similarity between visualizations
	"""

	def computeSimilarityMatrices(self, CommunityObject):
		# CommunityObject.communityMultiple.clear()

		Kappa_matrix = np.zeros((len(CommunityObject.dataAccumalation), len(CommunityObject.communityMultiple)))
		
		for community1, nodes1 in CommunityObject.dataAccumalation.items(): 
			for community2, nodes2 in CommunityObject.communityMultiple.items():
				"""
				Complicated Simliarity metric for threshold
				"""
				elements = len(list(set(nodes1).intersection(nodes2)))
				Numerator = elements
				# Numerator = CommunityObject.SetDeltaOperation(nodes1,nodes2)
				Denominator = len(set(nodes1)) + len(set(nodes2))
				val = float(float(Numerator)/float(Denominator))
				val2 = CommunityObject.SimiliarElements(nodes1,nodes2, community1, community2)
				No_Of_Elements[community1][community2] = val2
				Kappa_matrix[community1][community2] = val
		# assert len(Kappa_matrix) == 
		# pprint.pprint(Kappa_matrix)
		return Kappa_matrix,self.ThresholdAssignment(Kappa_matrix, CommunityObject) 
		"""
		 static boolean safeToAdd(int[] Q, int row, int col) {
		    // Unoccupied!
		    if (Q[col] != MT) {
		      return false;
		    }
		    // Do any columns have a rook in this row?
		    // Could probably stop at col here rather than Q.length
		    for (int c = 0; c < Q.length; c++) {
		      if (Q[c] == row) {
		        // Yes!
		        return false;
		      }
		    }
		    // All clear.
		    return true;
		  }

		solutions = 0;
		k = number_of_rooks;
		recurse(0,k);
		print solutions;
		...

		recurse(row, numberOfRooks) {
		   if (numberOfRooks == 0) {
		      ++solution;
		      return;
		      }
		   for(i=row; i<n; i++) {
		      for(j=0; j<n; j++) {
		         if (rook_is_ok_at(i, j)) {
		            place rook at i, j
		            recurse(i+1, numberOfRooks-1)
		            remove rook from i, j
		         }
		      }
		   }
		}

		"""
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

		import math
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
				# print j,k
				if matrix[j][k] == -1:
					matrix[j][k] = 0
				Sum = Sum + matrix[j][k]
			# print "----"
			AllSum.append(Sum)

		pprint.pprint(AllSum)
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
		# for i in PreviousNodes:
		# 	print i.CommunityColor
		KappaMatrixForComputation = []
		PermuteDict = dict()
		PermuteAssignment = []
		x, y = Kappa_matrix.shape
		
		# print x,y
		KappaMatrixForComputation = copy.deepcopy(Kappa_matrix)
		ColorAssignment = dict()
		if x >= 1 and y >= 1: 
			# pprint.pprint(Assignment)
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

			# print "ColorAssignment"
			for l,q in PermuteAssignment:
				PermuteDict[q] = l
			# ColorAssignment = Permt
			# print "printing"
			# pprint.pprint(PermuteDict)

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
			 	# QtGui.QMessageBox.information(self, "There are communities dying", "Python rocks!")
				for i in DifferenceElements:
					if PreviousNodes:
						try: 
							color = PreviousNodes[i].colorvalue.getRgb()
							CommunityObject.communitiesThatDie.append((color[0],color[1],color[2]))
						except AttributeError: 
							color = PreviousNodes[i].CommunityColor.getRgb()
							CommunityObject.communitiesThatDie.append((color[0],color[1],color[2]))

			for q in NewBornCommunities:
			 	# QtGui.QMessageBox.information(self, "There are new communities", "Python rocks!")
				if CommunityObject.NewCommunitiesToBeAssigned:
					ColorTobeAssigned = CommunityObject.NewCommunitiesToBeAssigned.popleft()
					ColorAssignment[q] = ColorTobeAssigned
					PermuteDict[q] = ColorTobeAssigned
					#FIXME DISABLED PERMUTE DICT
					# PermuteDict[q] = []

				else: 
					if CommunityObject.communitiesThatDie:
						ColorAssignment[q] = CommunityObject.communitiesThatDie.popleft()
						PermuteDict[q] = CommunityObject.communitiesThatDie.popleft()
						#FIXME DISABLED PERMUTE DICT
						# PermuteDict[q] = []
					else: 
						ColorAssignment[q] = (0,0,0)
						PermuteDict[q] = (0,0,0)


		print PermuteDict, ColorAssignment, PermuteDict == ColorAssignment
		if self.Permute:
			return PermuteDict
		else: 
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

		# print "Node"
		# for u in Nodes:
		# 	print u.Nodeidss 
		# 	print u.CommunityColor

		# print "Previous Node"
		# for k in PreviousNodes:
		# 	print k.Nodeidss 
		# 	print u.CommunityColor
		# for item in Nodes: 
		for toCommunity,fromCommunity in ColorAssignment.iteritems():
			try:
				if not(isinstance(fromCommunity,tuple)):
					# print "TO --> NOW",toCommunity,"FROM -->PREV",fromCommunity, PreviousNodes[fromCommunity].CommunityColor
					# print toCommunity, "IN NORMAL STUFF"
					# print "From--->",fromCommunity,PreviousNodes[fromCommunity].Nodeidss,"TO--->",toCommunity,item.Nodeidss
					
					#For now perform the color mixing algorithm
					# Kappa_matrix[:,] = 0 
					# self.blendColors(c1, c2, t)

					# print Nodes[toCommunity].Nodeidss,

					Nodes[toCommunity].PutColor(PreviousNodes[fromCommunity].CommunityColor)	
					# Nodes[toCommunity].PutColor(QtGui.QColor(0,0,255))
				else:
					"""Changing Color Important"""
					# print type(fromCommunity), len(fromCommunity)
					# print "NEW COLOR", toCommunity
					color = QtGui.QColor(fromCommunity[0],fromCommunity[1],fromCommunity[2])
					# color = QtGui.QColor(255,0,255)
					Nodes[toCommunity].PutColor(color)
					# color = QtGui.QColor(fromCommunity[0],fromCommunity[1],fromCommunity[2])
					# print "NewColor",color
					# item.PutColor(color)
					continue
			except IndexError as e:
				continue


		# for toCommunity,fromCommunity in ColorAssignment.iteritems():
		# 	if isinstance(fromCommunity,tuple): 
		# 		color = QtGui.QColor(fromCommunity[0],fromCommunity[1],fromCommunity[2])
		# 		Nodes[toCommunity].PutColor(color)

		# NewBorn = allElements - AssignedElements 

	# def blendColorValue(self,a, b, t):
 #    	return math.sqrt(((1-t)*(a*a))+ (t * (b*b)))

	# def blendAlphaValue(self,a, b, t):
 #    	return (1-t)*a + t*b;

	# def blendColors(self, c1, c2, t):
	#     for n in range(2): 
	#         ret[n] = self.blendColorValue(c1[n], c2[n], t)
	#     ret.a = self.blendAlphaValue(c1.a, c2.a, t)
	#     return ret
	
""" Work remaining to do is mainly deploying the tracking graph with different parameters
Working on consistent stuff """
class CommunitiesAcrossTimeStep(QtGui.QGraphicsView):
	sendLCDValues = QtCore.Signal(float)
	Logic = LogicForTimestep()

	def __init__(self,widget, electrode, electrodeUI, AcrossTimestep, Visualizer):
		QtGui.QGraphicsView.__init__(self)
		self.Graph_interface = widget
		self.AggregateList = []
		global timestep
		timestep = widget.OverallTimestep
		scene = QtGui.QGraphicsScene(self)
		scene.setItemIndexMethod(QtGui.QGraphicsScene.NoIndex)
		self.setScene(scene)
		self.toK = 0
		self.fromK = 0

		self.AnToK = 0 
		self.AnFrK = 0
		self.firstTIme = True
		self.LegacyAPI=LegacyAPI("AcrossTimestep")
		self.dot = Digraph(comment="The tracking graph")
		self.Scene_to_be_updated = scene
		self.setCacheMode(QtGui.QGraphicsView.CacheBackground)
		self.setRenderHint(QtGui.QPainter.Antialiasing)
		self.setInteractive(True)
		self.setViewportUpdateMode(QtGui.QGraphicsView.BoundingRectViewportUpdate)
		self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
		self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
		self.setResizeAnchor(QtGui.QGraphicsView.NoAnchor)
		self.scaleView(2.0)
		
		self.AcrossTimestep = AcrossTimestep
		self.electrodeUI = electrodeUI
		self.electrode = electrode
		self.Visualizer = Visualizer

		# plotWidget = pg.PlotWidget()
		# self.PlotWidget= plotWidget
		# self.PlotWidget = None
		self.widget = widget
		self.distinguishableColors = self.widget.communityDetectionEngine.distinguishableColors

		self.setWindowTitle('Analysis Across Timesteps')
		self.Order =[]
		self.previousTimestep = []

		# self.writer = csv.writer(open('data4.csv', 'wb'))
		# self.writer.writerow( ('source', 'target', 'value') )

		self.height = 340
		self.Offset = 0
		self.AggregateHashmap = dict()
		self.counter = 0
		self.variableWidth = 0
		self.communityMultiple = defaultdict(list)
		self.dataAccumalation = defaultdict(list)
		self.width = 1241
		self.AllNodes = None
		self.pointer = 0
		self.data5 = np.zeros((timestep+1))
		self.timestepMatrix = np.zeros((timestep+1,\
			NumberOfSelectedEletrodes,
			NumberOfSelectedEletrodes)) 

		self.firstTime = True
		self.PreComputeState = False
		self.ThresholdChange = True
		
		self.TrElectrodeFlag = False
		self.TrCommunityFlag = True

		self.first = True
		self.PreComputeData = None
		self.VisualizationTheme = "ThemeRiver"

		self._2NodeIds = []
		self.nodelist = [] 
		self.nodelist1 = [] 
		self.TrackOfAllData = []
		self.edgelist = []

		self.thresholdValue = 0.0
		self.array_value = 0
		self.NodeIds = []
		self.stabilityValues = []
		self.height = 0 
		self.AcrossTimestepUI = None

		self.TowPartitionValue = dict()
		self.TowGraphdataStructure = []
		self.TowGraph = []
		self.TowMultiple = defaultdict(list)
		self.stabilityValues= []
		self.communitiesThatDie = deque()

		self.NewCommunitiesToBeAssigned = deque([j for i,j in enumerate(self.distinguishableColors) if i > 9])

		self.width = 0
		self.ReEnterColumnLoop = True
		
		self.centerPos = (self.width/2 + 1, self.height -10)
		self.dendogram = None
		self.Kappa_matrix1 = None
		self.initUI()
		# self.InitiateLinespace()
		self.Assignment2 = dict()

		self.setSceneRect(self.Scene_to_be_updated.itemsBoundingRect())
		self.setScene(self.Scene_to_be_updated)
		self.fitInView(self.Scene_to_be_updated.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)

	@Slot(bool)
	def changePermute(self):
		global Logic
		self.Logic.Data.Permute = not(self.Logic.Data.Permute)

	def changeTrElectrode(self):
		self.TrElectrodeFlag = not(self.TrElectrodeFlag)
		self.update()

	def changeTrCommunity(self):
		self.TrCommunityFlag = not(self.TrCommunityFlag)
		self.update()

	def initUI(self):
		"""
		initialize all the components required
		"""
		self.NodeIds = []
		self.centrality = []

		scene = QtGui.QGraphicsScene(self)
		scene.setItemIndexMethod(QtGui.QGraphicsScene.NoIndex)
		scene.setSceneRect(-200, -200, 400, 400)
		self.setScene(scene)
		self.scene = scene
		self.setCacheMode(QtGui.QGraphicsView.CacheBackground)
		self.setRenderHint(QtGui.QPainter.Antialiasing)
		self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
		self.setResizeAnchor(QtGui.QGraphicsView.AnchorViewCenter)
		
		self.Scene_to_be_updated = scene
		self.setCacheMode(QtGui.QGraphicsView.CacheBackground)
		i = 0
		self.setSceneRect(self.Scene_to_be_updated.itemsBoundingRect())
		self.setScene(self.Scene_to_be_updated)
		self.fitInView(self.Scene_to_be_updated.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)
		self.scaleView(math.pow(2.0, -900/ 1040.0))		

	@Slot(object)
	def initializePrecomputationObject(self, PreComputeObject):
		self.PreComputeState = True
		self.PreComputeData = PreComputeObject
	
	@Slot()
	def LayoutChange(self,state):
		if state == "ThemeRiver": 
			self.VisualizationTheme = state
		elif state == "ObjectFlow":
			self.VisualizationTheme = state
		self.changeViewinGraph()
		self.scaleView(1.0001)

	@Slot()
	def stateChanges(self,state):
		self.ThresholdChange = not(self.ThresholdChange)
		# if self.ThresholdChange: 
		# 	print "Threshold Data Activated"
		# else: 
		# 	print "Not activated" 
	
	@Slot()
	def thresholdValueChanged(self, value): 
		if self.ThresholdChange:
			value = float(value)
			self.thresholdValue = value/10
			self.AcrossTimestepUI.thresholdlineedit.setText(str(self.thresholdValue))
			# for item in self.Scene_to_be_updated.items():
			# 	if isinstance(item, CommunitiesEdge):
			# 	 	item.setWeight(value)
			# self.changeViewinGraph()

	@Slot()
	def LineEditChanged(self):
		"""Handling Line Edit changes"""
		if self.ThresholdChange:
			text = (self.AcrossTimestepUI.thresholdlineedit.text().encode('ascii','ignore')).replace(' ','')
			self.AcrossTimestepUI.thresholdValue.setValue(int(text))

	@Slot()		
	def unsetPreComputationDate(self):
		self.PreComputeState = False
		if self.PreComputeData:
			self.PreComputeData.clear()

	@Slot(int)
	def CalculateClustersForTow(self,value):
		"""The value is between 0 to 64 and need to find out a way where calculates all the clusters at this point and stores it here ina 
		a data structure
		Every time something is called this gets invoked  and the formulae is calculated
		Changes in color should be happening right here
		1) Retreive the data from the correlationTable"""

		self.widget.correlationTable().changeTableContents(self.electrode.syllableUnit,value)
		self.widget.communityDetectionEngine.ChangeGraphDataStructure()
		self.widget.communityDetectionEngine.ChangeGraphWeights()

		self.TowGraphdataStructure = self.widget.Graph_data().DrawHighlightedGraph(self.widget.EdgeSliderValue)
		self.discreteTimesteps = value

		if self.PreComputeState:
			self.TowPartitionValue = self.PreComputeData[value]
		else:
			self.TowPartitionValue = self.widget.communityDetectionEngine.resolveCluster(self.widget.ClusteringAlgorithm,self.TowGraphdataStructure, self.widget.communityDetectionEngine.Number_of_Communities)  

		self.TowInducedGraph = cm.induced_graph(self.TowPartitionValue,self.TowGraphdataStructure)
		self.TowMultiple.clear()

		for key,value1 in self.TowPartitionValue.items():
			self.TowMultiple[value1].append(key)
		self.AssigNewValuesToGraphWidget(True)
		self.widget.Refresh()
		self.widget.scaleView(1.0001)
		# print "WARNING: Comunity coloring has been changed"
		self.changeViewinGraph()

	def noDuplicates(self, list1):
		print list1
		items = set([i for i in list1 if sum([1 for a in list1 if a == i]) > 1])
		# if items:
		# 	return True
		# else:
		# 	return False
		return True

	def truncate(self,no):
		if no < 20: 
			no = 45
		elif no > 230: 
			no = 200
		return no

	@Slot(bool)
	def CalculateStabilityOfMatrices(self,state):
		"""
		This function computes the correspondence between clsuters in the previous timestep vs the clusters in the next timestep
		Does the following things here in life 
		1) Calculates the similarity between clusters in the previous timesteps vs the next one  
		2) Sends out the values to all the other listerners 
 		5) Sends out the color information to the detection engine so that everyone can use the same coloring scheme 
		"""

		if self.firstTime:
			self.startTime = pg.ptime.time()
			self.firstTime = False

		# print self.electrode.dataProcess.ElectodeData[0][self.Graph_interface.TimeStep], self.Graph_interface.TimeStep
		self.variableWidth += 1
		self.communityMultiple.clear()

		""" 
		 = 1 - (all clusters in A (Delta) all clusters in B)/ (Sum of all elements in A and B) 
		""" 

		if self.communityMultiple:
			self.previousTimestep = copy.deepcopy(self.communityMultiple)

		for key,value in self.Graph_interface.PartitionOfInterest.items():
			self.communityMultiple[value].append(key)

		ModValue = 0
		counter = 0 
		Sum = 0.0 

		"""
		Change values for similiarity metric!! 
		"""
		Kappa_matrix = np.zeros((len(self.dataAccumalation), len(self.communityMultiple)))

		for community1, nodes1 in self.dataAccumalation.items():
			for community2, nodes2 in self.communityMultiple.items(): 
				Numerator = self.LegacyAPI.SetDeltaOperation(nodes1,nodes2)
				Denominator = len(set(nodes1)) + len(set(nodes2))
				val = float(1 - float(Numerator)/float(Denominator))
				Kappa_matrix[community1,community2] = val
				counter = counter + 1 

		"""Send the value to lcd viewer"""
		list1 = np.amax(Kappa_matrix, axis=1) 
		Value = np.mean(list1)

		self.sendLCDValues.emit(Value)
		self.updateSignals(self.electrode.timeStep,Value)
		
		self.NodeIds1 = [] 
		self.NodeIds1 = self.NodeIds #stuff in previous timestep 

		self.CreateTrackingNodes(self.communityMultiple) #stuff in current timestep 
		self.Kappa_matrix1 = []

		if not(self.ThresholdChange):
			self.Kappa_matrix1, AssignmentAcrossTime = self.initiateMatrix()
		else:
			self.Kappa_matrix1, AssignmentAcrossTime, colorAssignment = self.Logic.changeEdgeProperties(self, self.NodeIds1)
			if self.Graph_interface.TimeStep > -1:
				self.Logic.changeColorsForNodesJustRendered(self,colorAssignment,self.NodeIds,self.NodeIds1, self.Kappa_matrix1)

		#* Check for duplicates
		# self.CreateTrackingEdges(self.communityMultiple, self.dataAccumalation, AssignmentAcrossTime)

		Name = ""
		End = -1 
		Start = -1
		for name, start, end in FileNames:
			if self.Graph_interface.TimeStep > start and self.Graph_interface.TimeStep <= end:
				Name = name 
				Start = start 
				End = end
				if self.Graph_interface.TimeStep==start+1:
					self.toK = 0 
					self.fromK = 0 

		self.WriteTrackingData(AssignmentAcrossTime, Name, Start , End)
		self.ExchangeHeatmapData()
		# self.LegacyAPI.writeToaFile(self.AggregateList, self.dot)

		self.dataAccumalation = copy.deepcopy(self.communityMultiple)
		if self.Graph_interface.TimeStep == 61:
			self.SendValuesToElectrodeNodes(self.nodelist1)

		if AssignmentAcrossTime:
			self.AssigNewValuesToGraphWidget(False, colorAssignment)
		self.changeViewinGraph()


	def ExchangeHeatmapData(self):
		import csv
		name = 'ConsensusData/DeltaAreaChange'+str(self.electrode.syllableUnit)+str(4)+'Heatmap.tsv'
		f = open(name,'r')
		filedata = f.read()
		f.close()

		f = open(HeatmapFilename,'w')
		f.write(filedata)
		f.close()


		# with open('ConsensusData/DeltaAreaChange'+str(self.electrode.syllableUnit)+str(4)+'Heatmap.tsv','rb') as tsvin, open(HeatmapFilename,'wb') as tsvout:
		# 	tsvin = csv.reader(tsvin, delimiter='\t')
		# 	tsvout = csv.writer(tsvout, delimiter='\t')

		# 	for row in tsvin:
		# 		tsvout.writerows(row)

	def findElements1(self, a, b):
		return frozenset(a).intersection(b)

	def WriteTrackingData(self,AssignmentAcrossTime,Name ,Start ,End):
		sankeyJSON = dict()
		if (self.Graph_interface.TimeStep > Start) and (self.Graph_interface.TimeStep <= End) and self.Graph_interface.AnimationMode:
			self.toK += len(AssignmentAcrossTime.keys())
			self.AnToK+=len(AssignmentAcrossTime.keys())
			nodeDict = dict()
			EdgeDict = dict()

			# for i in self.NodeIds1:
				# i.
			for row, j in AssignmentAcrossTime.items():
				nodeDict = dict()
				if row == None or j == None: 
					continue
				valueRow = self.fromK+row
				valueRowName = self.AnFrK+row

				nodeDict["node"] = valueRow
				nodeDict["timestep"] = self.Graph_interface.TimeStep-1
				nodeDict["name"] = str(valueRowName)
				nodeDict["OriginalAssignmentValue"] = str(row)

				nodeDict["color"] = str("rgb"+str(self.NodeIds1[row].CommunityColor.getRgb()[:3])+"").replace("[", "").replace("]", "")

				NElements = self.dataAccumalation[row]
				# print row, self.Graph_interface.TimeStep-1, len(NElements)
				Elements = []
				
				for q in NElements:
					Elements.append(q)

				nodeDict["Elements"] = Elements
				opacity= self.ElectrodeOpacityCalc(self.electrode.syllableUnit, self.Graph_interface.TimeStep-1, Elements)
				
				if self.TrElectrodeFlag:
					if self.TrCommunityFlag:
						Color = tuple([self.truncate(int(round((opacity*x)))) for x in self.NodeIds1[row].CommunityColor.getRgb()[:3]])
					else: 
						Color = tuple([self.truncate(int(round(opacity*255))) for x in self.NodeIds1[row].CommunityColor.getRgb()[:3]])
				elif self.TrCommunityFlag:
					if not(self.TrElectrodeFlag):
						Color = tuple([self.truncate(int(round(x))) for x in self.NodeIds1[row].CommunityColor.getRgb()[:3]])

				# only for the second one 
				if Color == (186, 186, 186):
					Color = (229, 40, 44)

				if Color == (61, 200, 182):
					Color = (229, 122, 45)

				#darrk blue to black
				if Color == (22, 45, 166):
					Color = (0,0,0)

				# purple --- Brown
				if Color == (116, 45, 200):
					print "dont dont call me", self.Graph_interface.TimeStep
					Color = (200, 170, 122)

				# # light blue ---Green
				if Color == (200, 192, 203):
					print "call me", self.Graph_interface.TimeStep
					Color = (45, 153, 84)

				# Dark Blue -- blue
				if Color == (45, 45, 77):
					print "dont call me", self.Graph_interface.TimeStep
					Color = (45, 136, 201)


				# if Color == (123, 108, 217):
				# 	print "call me", self.Graph_interface.TimeStep
				# 	Color = (229, 122, 44)
				# if Color == (229, 122, 45):
				# 	Color = (123, 108, 216)
				# 	print "dont call me",self.Graph_interface.TimeStep

				nodeDict["color"] = str("rgb"+str(Color)+"").replace("[", "").replace("]", "")
				nodeDict["opacity"] = opacity
				# print Color, opacity , self.Graph_interface.TimeStep
 	  			self.nodelist.append(nodeDict)
	  			self.nodelist1.append(nodeDict)

				if self.firstTIme:
					self.firstTIme = False 
					self.dot.node(str(valueRow),str(valueRow))
				j = np.array(j)

				for column in j:  
					EdgeDict = dict()
					valueColumn = self.toK+column
					EdgeDict["source"] = valueRow
					EdgeDict["sourceColor"] = nodeDict["color"]
					EdgeDict["target"] = valueColumn
					EdgeDict["value"] = No_Of_Elements[row][column]

					Elements = Similarity[row][column]
					# assert len(Similarity[row][column])
					# EdgeDict["opacity"]= self.ElectrodeOpacityCalc(self.electrode.syllableUnit, self.Graph_interface.TimeStep, Elements)
					
					# self.writer.writerow((valueRow, valueColumn, int(self.Kappa_matrix1[row][column])))
					
					self.edgelist.append(EdgeDict)

					# print "[\'"+str(valueRow)+"','"+str(valueColumn)+"',"+str(float(self.Kappa_matrix1[row][column]))+"],"
					# self.dot.node(str(valueColumn),str(valueColumn))
					# print "edge", valueRow,valueColumn
					# self.dot.edge(str(valueRow),str(valueColumn))

			self.fromK = self.toK
			self.AnFrK = self.AnToK
			self.AggregateList.append(AssignmentAcrossTime)

		if (self.Graph_interface.TimeStep == End):
			import json
			with open(Name, 'w') as outfile:
				c = 0 
				for row in self.NodeIds:
					nodeDict = dict()
					R = self.fromK + c
					RName = self.AnFrK + c
					nodeDict["node"] = R
					nodeDict["timestep"] = -1
					nodeDict["name"] = str(RName)
					nodeDict["color"] = str("rgb"+str(self.NodeIds[c].CommunityColor.getRgb()[:3])+"").replace("[", "").replace("]", "")
					NElements  = Similarity[c]
					NElements = self.communityMultiple[c]
					# print NElements, c, self.Graph_interface.TimeStep-1, len(NElements), "Last"
					Elements = []
					for q in NElements:
						Elements.append(q)
					nodeDict["Elements"] = Elements
					opacity= self.ElectrodeOpacityCalc(self.electrode.syllableUnit, self.Graph_interface.TimeStep, Elements)
					
					if self.TrElectrodeFlag:
						if self.TrCommunityFlag:
							Color1 = tuple([self.truncate(int(round((opacity*x)))) for x in self.NodeIds[c].CommunityColor.getRgb()[:3]])
						else: 
							Color1 = tuple([self.truncate(int(round(opacity*255))) for x in self.NodeIds[c].CommunityColor.getRgb()[:3]])
					elif self.TrCommunityFlag:
						if not(self.TrElectrodeFlag):
							Color1 = tuple([int(round(x)) for x in self.NodeIds[c].CommunityColor.getRgb()[:3]])

					Color1 = (255,255,255)
					nodeDict["color"] = str("rgb"+str(Color1)+"").replace("[", "").replace("]", "")

					nodeDict["opacity"] = opacity
					nodeDict["OriginalAssignmentValue"] = str(c)

		  			self.nodelist.append(nodeDict)
		  			self.nodelist1.append(nodeDict)
		  			c =c +1
				sankeyJSON["nodes"] = self.nodelist
				sankeyJSON["links"] = self.edgelist

				# Give all the electrode nodes is all views colors and opacity

				self.nodelist = []
				self.edgelist = []

				# pprint.pprint(sankeyJSON["nodes"])
				# pprint.pprint(sankeyJSON["links"])
				json.dump(sankeyJSON, outfile,  indent=4)
			outfile.close()

	def ElectrodeOpacityCalc(self, syllable, TimeStep, elements):
		Sum = 0 
		k = 0
		# for i in range(64):
		# 	print self.electrode.dataProcess.ElectrodeSignals[ElectrodeSignalDataName][0,i,TimeStep]

		for element in elements: 
			Sum+= self.electrode.dataProcess.ElectrodeSignals[ElectrodeSignalDataName][syllable,element,TimeStep]
			k = k + 1
		try: 
			avg = float("{0:.2f}".format(Sum/float(k)))
		except ZeroDivisionError:
			avg = 1


		if str(avg).lower() == 'nan':
			avg = 1
		return avg

	@Slot()
	def flushData(self):
		self.data5 = np.zeros((timestep+1))
		self.updateSignals(0,0)
		self.AggregateList = []
		self.toK = 0
		self.fromK = 0
		self.firstTime = 0
		self.nodelist = []
		self.edgelist = []
		self.ColorAssignment = []
		self.PreviousNodes = []
		self.NodeIds1 = []
		self.NodeIds = []
		self.AnFrK =0 
		self.AnToK =0 
		self.Offset = 0

		sankeyJSON = dict()
		# import json
		# for i,j,k in FileNames:
		# 	with open(i, 'w') as outfile:
		# 		json.dump("", i,  indent=4)

		for item in self.Scene_to_be_updated.items():
			self.Scene_to_be_updated.removeItem(item)

		self.variableWidth = 0
		self.first = False

	@Slot()
	def changeVizOffset(self):
		self.Offset+=1
		self.SendValuesToElectrodeNodes(self.nodelist1, self.Offset)
		self.changeViewinGraph()

	def CreateTrackingNodes(self, partitionValues):
		i = 0
		self.counter = self.counter+1
		self.NodeIds = []
		sceneRect = self.sceneRect()

		# Create the nodes which are rated in the way that it is organized
		# Just create one layer of the nodes here!! for now
		
		for communities, sub_communities in partitionValues.items():
			i = i + 1
			node_value=CommunityGraphNode(self,communities, sub_communities)
			node_value.setPos(sceneRect.left() + self.variableWidth*100, i*40)
			self.NodeIds.append(node_value)
			self.scene.addItem(node_value)

		if self.first:
			value = len(partitionValues.values())
			self.first = False #*
			self.NewCommunitiesToBeAssigned = []
			self.NewCommunitiesToBeAssigned = deque([j for i,j in enumerate(self.distinguishableColors) if i > (10)])

	# def CreateTrackingEdges(self, partitionValues, BeforeTimestep, AssignmentAcrossTime):
	# 	i = 0
	# 	k = 0
	# 	for community1 in BeforeTimestep.keys():
	# 		for community in partitionValues.keys(): 
	# 			if self.ThresholdChange:
	# 				k = k + 1 
	# 				if community in AssignmentAcrossTime[community1]: 
	# 					self.scene.addItem(CommunitiesEdge(self,self.NodeIds1[community1],self.NodeIds[community],k,community1,community, self.Kappa_matrix1[community1][community]))
	# 			elif AssignmentAcrossTime[community1] == community: 
	# 				k = k + 1 
	# 				self.scene.addItem(CommunitiesEdge(self,self.NodeIds1[community1],self.NodeIds[community],k,community1,community, self.Kappa_matrix1[community1][community]))

	def CreateNodes(self, partitionValues):
		global Width_value
		i = 0
		sceneRect = self.sceneRect()
		self.NodeIds1 = []
		# Create the nodes which are rated in the way that it is organized
		# Just create one layer of the nodes here!! for now
		# print "Current partition Value",partitionValues

		for communities, sub_communities in partitionValues.items():
			i = i + 1
			node_value=CommunityGraphNode(self,communities, sub_communities)
			node_value.setPos(sceneRect.left() + self.variableWidth*100, i*40)
			self.NodeIds1.append(node_value)
			self.scene.addItem(node_value)

	def SendValuesToElectrodeNodes(self, nodelist, Offset = 0):
		# pprint.pprint(nodelist)
		# timestep at a range ONLY update that electrodeView, only then move onto the next one
		# now make a hashmap of this range to write it onto the place
		# make it really hard and fast ends at 8:00 pm
		hashmap = dict()
		# print
		c= 0 
		list1 = []
		hashmap[0] = 0
		ElectrodeViewObjectDummy = self.electrode.SmallMultipleElectrode[1]

		print "Slices",ElectrodeViewObjectDummy.slices
		for i in range(1,self.electrode.dataProcess.timestep):
			if not(i % ElectrodeViewObjectDummy.slices) == 0: 
				hashmap[i-1 - Offset] = c
			else: 
				hashmap[i-1 - Offset] = c
				c= c+1

		print "----"
		self.AggregateHashmap = copy.deepcopy(hashmap)
		self.UpdateColorsInElectrodeView(nodelist, Offset)
		# print "Calling another function to update the electrode View"

	def UpdateColorsInElectrodeView(self, nodelist, Offset = 0):
		ElectrodeViewObjectDummy = self.electrode.SmallMultipleElectrode[1]
		
		for i in nodelist:
			try: 
				ElectrodeViewNumberToUpdate = self.AggregateHashmap[i['timestep']]
			except KeyError: 
				continue
			# print i['timestep'], ElectrodeViewNumberToUpdate

			# getting the electrode view number 
			ElectrodeViewObject = self.electrode.SmallMultipleElectrode[ElectrodeViewNumberToUpdate]
			assert ElectrodeViewObject.ChunkNo == ElectrodeViewNumberToUpdate
			# get the elements in the community defined by that 
			CommunityNumber = i['OriginalAssignmentValue']
			Elements = i['Elements']

			Color = i['color'].replace("rgb(", "").replace(")", "").replace(" ", "").split(',')
			Color = map(int, Color)
			# print len(Elements)
			for element in Elements: 
				# print  element
				ActualElectrodeNumber = self.electrode.ElectrodeIds[element]
				norm = ElectrodeViewObject.ElectrodeOpacity[element].normalize(i['timestep'], ActualElectrodeNumber)
				ElectrodeViewObject.NodeIds[element].PutFinalColors(norm[0],norm[1], QtGui.QColor(Color[0],Color[1],Color[2]),i['timestep'],CommunityNumber,ElectrodeViewObjectDummy.slices)
				
				assert ElectrodeViewObject.NodeIds[element].counter == element
				ElectrodeViewObject.NodeIds[element].update()

				# print norm
				# Sum+= ElectrodeViewObject.normalize(self.electrode.dataProcess.ElectrodeSignals['muDat'][syllable,element,i['timestep']])

	def animate(self):
		self.elapsed = (self.eFFlapsed + self.sender().interval()) % 1000
		self.repaint()

	def paint(self, painter, option, widget):
		painter.setPen(QtCore.Qt.NoPen)
		painter.setBrush(QtCore.Qt.black)
		painter.setPen(QtGui.QPen(QtCore.Qt.black, 0))
		painter.drawEllipse(-4, -4, 10, 10)

	def drawBackground(self, painter, rect):
		sceneRect = self.sceneRect()
		textRect = QtCore.QRectF(sceneRect.left() + 4, sceneRect.top() + 4,
								 sceneRect.width() - 4, sceneRect.height() - 4)
		message = self.tr("Link Graph")

		font = painter.font()
		font.setBold(True)
		font.setPointSize(14)
		painter.setFont(font)
		painter.setPen(QtCore.Qt.lightGray)
		painter.drawText(textRect.translated(2, 2), message)
		painter.setPen(QtCore.Qt.black)
		painter.drawText(textRect, message)

	def changeViewinGraph(self):
		self.setSceneRect(self.Scene_to_be_updated.itemsBoundingRect())
		self.setScene(self.Scene_to_be_updated)
		x1,y1,x2,y2 = (self.Scene_to_be_updated.itemsBoundingRect()).getCoords()
		# x2-
		# self.fitInView(self.Scene_to_be_updated.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)
		self.fitInView(QtCore.QRectF(x2-50,y1,x2+100,y2), QtCore.Qt.KeepAspectRatio)
		# self.fitInView(QtCore.QRectF(x2+50+self.width/4+50, self.height/4+50, x2+250,3*self.height/6),QtCore.Qt.KeepAspectRatio)
		self.Scene_to_be_updated.update()
		self.update()

	def InitiateLinespace(self):
		pass
		x = np.linspace(0,100,64)
		y = x
		# self.PlotWidget.setYRange(0,1)
		# self.PlotWidget.setXRange(0,timestep+1,padding=0)
		# self.CurvePoint = self.PlotWidget.plot(pen={'color': 0.8, 'width': 1}, name= 'Community Stability')

	def updateScene(self):
		self.update()
		self.Scene_to_be_updated.update()

	def updateSignals(self, x,Value):
		pass
		# self.data5[x] = Value
		# self.pointer = self.pointer+1
		# self.CurvePoint.setData(self.data5)

	"""Defined as the matrix that will be computed at every timestep
	Useful for analysis between timesteps"""
	def initiateMatrix(self):
		try: 
			if self.electrode.timeStep > -19: 
				try: 
					# change array value for toggling between discrete and continuous timesteps 
					if self.dataAccumalation:
						if not(self.communityMultiple == self.dataAccumalation):
							for community1,nodes1 in self.dataAccumalation.items():		
								for community2,nodes2 in self.communityMultiple.items():
										self.timestepMatrix[self.Graph_interface.TimeStep][community1][community2] = self.SimiliarElements(nodes1,nodes2, community1, community2)
						self.CalculateWidthAndHeight(self.Graph_interface.TimeStep)
					if self.height > 0: 
						matrix,Assignment = self.AffinityMapping(self.Graph_interface.TimeStep)
				except IndexError as e:
					traceback.print_exc()
		except AttributeError as e:
			traceback.print_exc()

		return self.timestepMatrix[self.Graph_interface.TimeStep], Assignment
		"""
		The data is of two timestep in nature
		This computation is between counter-1 and counter timesteps. 
		The best way is run this function on a separate thread or use parallel programming. 
		"""

	def CalculateWidthAndHeight(self,array_value):
		for i in range(len(self.timestepMatrix[array_value])):
			if (self.timestepMatrix[array_value][i,0] == 0):
				self.height = i
				break
			for j in range(len(self.timestepMatrix[array_value])):
				if (self.timestepMatrix[array_value][0,j] == 0):
					self.width = j
					break

		if self.width > self.height:
			pass
		elif self.height > self.width:
			pass
		else:
			pass

	"""Identifies the intersecting elements in the two lists"""
	def SimiliarElements(self,nodes1, nodes2, community1, community2):
		intersectingElements = list(set(nodes1).intersection(nodes2))
		if (len(intersectingElements) == 0):
			return -1 
		Similarity[community1][community2] = intersectingElements
		return len(intersectingElements)

	def AffinityMapping(self,array_value): 
		matrix = self.timestepMatrix[array_value][:self.height,:self.width]
		MaxValuesAssignment = -3
		indexAssign = -1
		Index= 0
		ProportionScore = -1
		Assignment = dict()

		for i in range(self.height):
			if i in Assignment.keys():
				continue
			MaxValuesAssignment = -3
			CurrentColumn = matrix[:,i]
			ColumnList = CurrentColumn.tolist()
			
			while (self.ReEnterColumnLoop):
				Max , Index= self.max1(ColumnList, Assignment)
				self.AssignValues(ColumnList,Max,Index,Assignment,i,matrix)
			self.ReEnterColumnLoop = True

		AssignedValues = set(Assignment.values())
		AssignedKeys = set(Assignment.keys())

		if self.height > self.width:
			"""Case when some community colors are destroyed"""
			allElements = set(range(0,self.height))
			destroyedCommunities = allElements-AssignedValues
			try: 
				for k in destroyedCommunities: 		
					pass
			except AttributeError:
				traceback.print_exc()
				pass
				"""Sorry no communities to be included in the dead pool"""
		elif self.width > self.height:
			"""Case when some community colors are born again"""
			allElements = set(range(0,self.width))
			NewBornCommunities = allElements-AssignedKeys
			for q in NewBornCommunities:
				if self.NewCommunitiesToBeAssigned:
					ColorTobeAssigned = self.NewCommunitiesToBeAssigned.popleft()
					Assignment[q] = ColorTobeAssigned
				else: 
					try: 
						ColorTobeAssigned = self.communitiesThatDie.popleft()
					except IndexError:
						self.NewCommunitiesToBeAssigned = deque([j for i,j in enumerate(self.distinguishableColors) if i > 24])
						ColorTobeAssigned = self.NewCommunitiesToBeAssigned.popleft()
					Assignment[q] = ColorTobeAssigned
		else: 
			pass

		""" Communities that die """
		return matrix, Assignment

	def AssigNewValuesToGraphWidget(self,TowValues=False,Assignment=None):

		# print "SENDING/CHANGING COLORS EVERYWHERE"
		# print "Parameters", TowValues, Assignment
			
		if TowValues: 
			PartitionValueToBeVisit= copy.deepcopy(self.TowPartitionValue) 
			self.widget.communityDetectionEngine.\
			AssignCommuntiesFromDerivedFromTow(\
				self.TowPartitionValue,
				self.TowInducedGraph,
				self.TowMultiple,
				self.TowGraphdataStructure,
				self.discreteTimesteps,
				self.electrode.syllableUnit)
			self.widget.communityDetectionEngine.\
							timeStepColorGenerator(\
								len(set(PartitionValueToBeVisit.values())),\
			 					PartitionValueToBeVisit)
			self.widget.partition = copy.deepcopy(self.TowPartitionValue)
			self.widget.ColorForVisit(self.TowPartitionValue)
		else:
			PartitionValueToBeVisit= copy.deepcopy(self.communityMultiple)
			self.widget.communityDetectionEngine.\
							timeStepAnimationGenerator(\
								len(set(Assignment.keys())),
								Assignment, self.Graph_interface.PartitionOfInterest)
		
		""" Fix me for now just happen to coment this line out because at every timestep you just need the same colors"""

	def wheelEvent(self, event):
		self.scaleView(math.pow(2.0, -event.delta() / 1040.0))

	def scaleView(self, scaleFactor):
		factor = self.matrix().scale(scaleFactor, scaleFactor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()
		if factor < 0.07 or factor > 100:
			return
		self.scale(scaleFactor, scaleFactor)
		del factor
