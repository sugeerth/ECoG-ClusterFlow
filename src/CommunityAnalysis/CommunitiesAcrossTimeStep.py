import numpy as np
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
from TrackingGraph.CommunitiesEdge import CommunitiesEdge

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
from StaticColorLogic import LogicForTimestep, SimilarityData

"""Class responsible for analysis of communities across timesteps"""

Capture = [0,15]
Interval = 13

FileNames = [('/Users/sugeerthmurugesan/Sites/Sankey/JSON_1.json',0,12),('/Users/sugeerthmurugesan/Sites/Sankey/JSON_2.json',12,24)\
,('/Users/sugeerthmurugesan/Sites/Sankey/JSON_3.json',24,36),('/Users/sugeerthmurugesan/Sites/Sankey/JSON_4.json',36,48)\
,('/Users/sugeerthmurugesan/Sites/Sankey/JSON_5.json',48,6)]

HeatmapFilename = "/Users/sugeerthmurugesan/Sites/Sankey/DeltaAreaChange4Heatmap.tsv"


# ElectrodeSignalDataName = 'muDat'
ElectrodeSignalDataName = 'sigData'

timestep = 12
THRESHOLD_VALUE_TRACKING_GRAPH = 0.10
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

""" Work remaining to do is mainly deploying the tracking graph with different parameters
Working on consistent stuff """
class CommunitiesAcrossTimeStep(QtGui.QGraphicsView):
	sendLCDValues = QtCore.Signal(float)

	def __init__(self,widget, electrode, electrodeUI, AcrossTimestep, Visualizer, communityDetectionEngine):
		QtGui.QGraphicsView.__init__(self)
		self.Graph_interface = widget
		self.AggregateList = []

		scene = QtGui.QGraphicsScene(self)
		scene.setItemIndexMethod(QtGui.QGraphicsScene.NoIndex)
		self.setScene(scene)

		global timestep
		self.Logic = LogicForTimestep()
		timestep = widget.OverallTimestep
		self.toK = 0
		self.fromK = 0

		self.AnToK = 0 
		self.AnFrK = 0
		self.communityDetectionEngine = communityDetectionEngine
		self.firstTIme = True
		self.LegacyAPI=LegacyAPI("AcrossTimestep")
		self.dot = Digraph(comment="The tracking graph")
		
		self.AcrossTimestep = AcrossTimestep
		self.electrodeUI = electrodeUI
		self.electrode = electrode
		self.Visualizer = Visualizer

		self.widget = widget
		self.distinguishableColors = self.widget.communityDetectionEngine.distinguishableColors

		self.setWindowTitle('Analysis Across Timesteps')
		self.Order =[]
		self.previousTimestep = []

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
		self.No_Of_Elements = [[0.0]*16 for _ in xrange(16)]
		self.TrackOfAllData = []
		self.edgelist = []

		self.thresholdValue = THRESHOLD_VALUE_TRACKING_GRAPH
		self.array_value = 0
		self.presentNodeObjects = []
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
		self.Kappa_matrixForSimilarityComputation = None
		self.initUI()
		self.Assignment2 = dict()

		self.setSceneRect(self.Scene_to_be_updated.itemsBoundingRect())
		self.setScene(self.Scene_to_be_updated)
		self.fitInView(self.Scene_to_be_updated.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)

	def truncate(self,no):
		if no < 20: 
			no = 45
		elif no > 230: 
			no = 200
		return no

	def noDuplicates(self, list1):
		print list1
		items = set([i for i in list1 if sum([1 for a in list1 if a == i]) > 1])
		# if items:
		# 	return True
		# else:
		# 	return False
		return True

	@Slot(bool)
	def CalculateStabilityOfMatrices(self,state):
		"""
		This function computes the correspondence between clsuters in the previous timestep vs the clusters in the next timestep
		Does the following things here in life 
		1) Calculates the similarity between clusters in the previous timesteps vs the next one  
		2) Sends out the values to all the other listerners 
 		5) Sends out the color information to the detection engine so that everyone can use the same coloring scheme 
		"""
		self.communityMultiple.clear()

		if self.communityMultiple:
			self.previousTimestep = copy.deepcopy(self.communityMultiple)

		for key,value in self.communityDetectionEngine.ClusterPartitionOfInterest.items():
			self.communityMultiple[value].append(key)

		"""
		At this point we have previous timestep clustering results: 
		self.communityMultiple[value] this is being repnamed as self.previousTimestep
		Change values for similiarity metric!! 

		Making a similarity matrix where the no of rows are the number of communities in previous timeStep
		and the number of columns are the communities in the next timestep. 
		 -------
		|		| 	4 in previous timestep and 6 in the next one
		|		|
		|		|
		|		|
		 -------
		Similarity = 1 - (all clusters in A (Delta) all clusters in B)/ (Sum of all elements in A and B) 
		Can be replaced by NMI 
		"""
		self.previousNodeObjects = [] 
		self.previousNodeObjects = self.presentNodeObjects #stuff in previous timestep 

		self.CreateTrackingNodes(self.communityMultiple) #stuff in current timestep 
		self.Kappa_matrixForSimilarityComputation = []

		self.Kappa_matrixForSimilarityComputation, self.No_Of_Elements, AssignmentAcrossTime, colorAssignment = self.Logic.changeEdgeProperties(self, self.previousNodeObjects)
		if self.Graph_interface.TimeStep > -1:
			self.Logic.changeColorsForNodesJustRendered(self,colorAssignment,self.presentNodeObjects,self.previousNodeObjects, self.Kappa_matrixForSimilarityComputation)

		#* Check for duplicates
		# self.noDuplicates
		self.CreateTrackingEdges(self.communityMultiple, self.dataAccumalation, AssignmentAcrossTime)

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

		# if self.firstTime:
		# 	self.firstTime = False

		self.WriteTrackingData(AssignmentAcrossTime, Name, Start , End)
		self.ExchangeHeatmapData()

		self.dataAccumalation = copy.deepcopy(self.communityMultiple)

		if self.Graph_interface.TimeStep == 61:
			self.SendValuesToElectrodeNodes(self.nodelist1)

		if AssignmentAcrossTime:
			self.AssigNewValuesToGraphWidget(False, colorAssignment)

		self.changeViewinGraph()

	def WriteTrackingData(self,AssignmentAcrossTime,Name ,Start ,End):
		sankeyJSON = dict()
		if (self.Graph_interface.TimeStep > Start) and (self.Graph_interface.TimeStep <= End) and self.Graph_interface.communityDetectionEngine.AnimationMode:
			self.toK += len(AssignmentAcrossTime.keys())
			self.AnToK+=len(AssignmentAcrossTime.keys())
			nodeDict = dict()
			EdgeDict = dict()

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

				nodeDict["color"] = str("rgb"+str(self.previousNodeObjects[row].CommunityColor.getRgb()[:3])+"").replace("[", "").replace("]", "")

				NElements = self.dataAccumalation[row]
				Elements = []
				
				for q in NElements:
					Elements.append(q)

				nodeDict["Elements"] = Elements
				opacity= self.ElectrodeOpacityCalc(self.electrode.syllableUnit, self.Graph_interface.TimeStep-1, Elements)
				
				if self.TrElectrodeFlag:
					if self.TrCommunityFlag:
						Color = tuple([self.truncate(int(round((opacity*x)))) for x in self.previousNodeObjects[row].CommunityColor.getRgb()[:3]])
					else: 
						Color = tuple([self.truncate(int(round(opacity*255))) for x in self.previousNodeObjects[row].CommunityColor.getRgb()[:3]])
				elif self.TrCommunityFlag:
					if not(self.TrElectrodeFlag):
						Color = tuple([self.truncate(int(round(x))) for x in self.previousNodeObjects[row].CommunityColor.getRgb()[:3]])

				nodeDict["color"] = str("rgb"+str(Color)+"").replace("[", "").replace("]", "")
				nodeDict["opacity"] = opacity
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
					EdgeDict["value"] = self.No_Of_Elements[row][column]

					Elements = Similarity[row][column]

					self.edgelist.append(EdgeDict)

			self.fromK = self.toK
			self.AnFrK = self.AnToK
			self.AggregateList.append(AssignmentAcrossTime)

		if (self.Graph_interface.TimeStep == End):
			import json
			with open(Name, 'w') as outfile:
				c = 0 
				for row in self.presentNodeObjects:
					nodeDict = dict()
					R = self.fromK + c
					RName = self.AnFrK + c
					nodeDict["node"] = R
					nodeDict["timestep"] = -1
					nodeDict["name"] = str(RName)
					nodeDict["color"] = str("rgb"+str(self.presentNodeObjects[c].CommunityColor.getRgb()[:3])+"").replace("[", "").replace("]", "")

					NElements  = Similarity[c]
					NElements = self.communityMultiple[c]
					Elements = []
					for q in NElements:
						Elements.append(q)
					nodeDict["Elements"] = Elements
					opacity= self.ElectrodeOpacityCalc(self.electrode.syllableUnit, self.Graph_interface.TimeStep, Elements)
					
					if self.TrElectrodeFlag:
						if self.TrCommunityFlag:
							Color1 = tuple([self.truncate(int(round((opacity*x)))) for x in self.presentNodeObjects[c].CommunityColor.getRgb()[:3]])
						else: 
							Color1 = tuple([self.truncate(int(round(opacity*255))) for x in self.presentNodeObjects[c].CommunityColor.getRgb()[:3]])
					elif self.TrCommunityFlag:
						if not(self.TrElectrodeFlag):
							Color1 = tuple([int(round(x)) for x in self.presentNodeObjects[c].CommunityColor.getRgb()[:3]])

					Color1 = (255,255,255)
					nodeDict["color"] = str("rgb"+str(Color1)+"").replace("[", "").replace("]", "")
					
					nodeDict["opacity"] = opacity
					nodeDict["OriginalAssignmentValue"] = str(c)

		  			self.nodelist.append(nodeDict)
		  			self.nodelist1.append(nodeDict)
		  			c =c +1
				sankeyJSON["nodes"] = self.nodelist
				sankeyJSON["links"] = self.edgelist

				self.nodelist = []
				self.edgelist = []

				json.dump(sankeyJSON, outfile,  indent=4)
			outfile.close()


	def ExchangeHeatmapData(self):
		import csv
		name = 'ConsensusData/DeltaAreaChange'+str(self.electrode.syllableUnit)+str(4)+'Heatmap.tsv'
		f = open(name,'r')
		filedata = f.read()
		f.close()

		f = open(HeatmapFilename,'w')
		f.write(filedata)
		f.close()

	def ElectrodeOpacityCalc(self, syllable, TimeStep, elements):
		Sum = 0 
		k = 0

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
		self.AggregateList = []
		self.toK = 0
		self.fromK = 0
		self.firstTime = True
		self.nodelist = []
		self.edgelist = []
		self.ColorAssignment = []
		self.PreviousNodes = []
		self.previousNodeObjects = []
		self.presentNodeObjects = []
		self.AnFrK =0 
		self.AnToK =0 
		self.Offset = 0

		sankeyJSON = dict()

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
		self.presentNodeObjects = []
		sceneRect = self.sceneRect()

		# Create the nodes which are rated in the way that it is organized
		# Just create one layer of the nodes here!! for now
		
		for communities, sub_communities in partitionValues.items():
			i = i + 1
			node_value=CommunityGraphNode(self,communities, sub_communities)
			node_value.setPos(sceneRect.left() + self.variableWidth*100, i*40)
			self.presentNodeObjects.append(node_value)
			self.scene.addItem(node_value)

		if self.first:
			value = len(partitionValues.values())
			self.first = False #*
			self.NewCommunitiesToBeAssigned = []
			self.NewCommunitiesToBeAssigned = deque([j for i,j in enumerate(self.distinguishableColors) if i > (10)])
			for i in self.presentNodeObjects: 
				color = self.NewCommunitiesToBeAssigned.popleft()
				colorNew = QtGui.QColor(color[0],color[1],color[2])
				i.PutColor(colorNew)

	def CreateNodes(self, partitionValues):
		global Width_value
		i = 0
		self.previousNodeObjects = []
		# Create the nodes which are rated in the way that it is organized
		# Just create one layer of the nodes here!! for now

		for communities, sub_communities in partitionValues.items():
			i = i + 1
			node_value=CommunityGraphNode(self,communities, sub_communities)
			node_value.setPos(sceneRect.left() + self.variableWidth*100, i*40)
			self.previousNodeObjects.append(node_value)
			self.scene.addItem(node_value)

	def CreateTrackingEdges(self, partitionValues, BeforeTimestep, AssignmentAcrossTime):
		i = 0
		k = 0
		for community1 in BeforeTimestep.keys():
			for community in partitionValues.keys(): 
				if self.ThresholdChange:
					k = k + 1 
					if community in AssignmentAcrossTime[community1]: 
						self.scene.addItem(CommunitiesEdge(self,self.previousNodeObjects[community1],self.presentNodeObjects[community],k,community1,community, self.Kappa_matrixForSimilarityComputation[community1][community]))
				elif AssignmentAcrossTime[community1] == community: 
					k = k + 1 
					self.scene.addItem(CommunitiesEdge(self,self.previousNodeObjects[community1],self.presentNodeObjects[community],k,community1,community, self.Kappa_matrixForSimilarityComputation[community1][community]))

	def SendValuesToElectrodeNodes(self, nodelist, Offset = 0):
		# timestep at a range ONLY update that electrodeView, only then move onto the next one
		# now make a hashmap of this range to write it onto the place
		hashmap = dict()
		c= 0 
		list1 = []
		hashmap[0] = 0
		ElectrodeViewObjectDummy = self.electrode.SmallMultipleElectrode[1]

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

			# getting the electrode view number 
			ElectrodeViewObject = self.electrode.SmallMultipleElectrode[ElectrodeViewNumberToUpdate]
			assert ElectrodeViewObject.ChunkNo == ElectrodeViewNumberToUpdate

			# get the elements in the community defined by that 
			CommunityNumber = i['OriginalAssignmentValue']
			Elements = i['Elements']

			Color = i['color'].replace("rgb(", "").replace(")", "").replace(" ", "").split(',')
			Color = map(int, Color)

			for element in Elements: 
				# print  element
				ActualElectrodeNumber = self.electrode.ElectrodeIds[element]
				norm = ElectrodeViewObject.ElectrodeOpacity[element].normalize(i['timestep'], ActualElectrodeNumber)
				ElectrodeViewObject.NodeIds[element].PutFinalColors(norm[0],norm[1], QtGui.QColor(Color[0],Color[1],Color[2]),i['timestep'],CommunityNumber,ElectrodeViewObjectDummy.slices)
				assert ElectrodeViewObject.NodeIds[element].counter == element
				ElectrodeViewObject.NodeIds[element].update()

	def animate(self):
		self.elapsed = (self.eFFlapsed + self.sender().interval()) % 1000

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

	def updateScene(self):
		self.update()
		self.Scene_to_be_updated.update()

	def wheelEvent(self, event):
		self.scaleView(math.pow(2.0, -event.delta() / 1040.0))

	def scaleView(self, scaleFactor):
		factor = self.matrix().scale(scaleFactor, scaleFactor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()
		if factor < 0.07 or factor > 100:
			return
		self.scale(scaleFactor, scaleFactor)
		del factor


	"""Identifies the intersecting elements in the two lists"""
	def SimiliarElements(self,nodes1, nodes2, community1, community2):
		intersectingElements = list(set(nodes1).intersection(nodes2))
		if (len(intersectingElements) == 0):
			return -1 
		Similarity[community1][community2] = intersectingElements
		return len(intersectingElements)

	def AssigNewValuesToGraphWidget(self,TowValues=False,Assignment=None):
		if TowValues: 
			PartitionValueToBeVisit= copy.deepcopy(self.TowPartitionValue) 
			self.widget.communityDetectionEngine.AssignCommuntiesFromDerivedFromTow(self.TowPartitionValue,self.TowInducedGraph,self.TowMultiple,self.TowGraphdataStructure,self.discreteTimesteps,self.electrode.syllableUnit)
			self.widget.communityDetectionEngine.timeStepColorGenerator(len(set(PartitionValueToBeVisit.values())),PartitionValueToBeVisit)
			self.widget.partition = copy.deepcopy(self.TowPartitionValue)
			self.widget.ColorForVisit(self.TowPartitionValue)
		else:
			PartitionValueToBeVisit= copy.deepcopy(self.communityMultiple)
			self.widget.communityDetectionEngine.timeStepAnimationGenerator(len(set(Assignment.keys())),Assignment, self.communityDetectionEngine.ClusterPartitionOfInterest)
		
		""" Fix me for now just happen to coment this line out because at every timestep you just need the same colors"""

	@Slot(bool)
	def changePermute(self):
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
		self.presentNodeObjects = []
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
		
		self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)
		self.setInteractive(True)
		self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
		
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

	@Slot()
	def stateChanges(self,state):
		self.ThresholdChange = not(self.ThresholdChange)
	
	@Slot()
	def thresholdValueChanged(self, value): 
		if self.ThresholdChange:
			value = float(value)
			self.thresholdValue = value/10
			self.AcrossTimestepUI.thresholdlineedit.setText(str(self.thresholdValue))

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
			self.TowPartitionValue = self.communityDetectionEngine.resolveCluster(self.communityDetectionEngine.ClusteringAlgorithm,self.TowGraphdataStructure, self.communityDetectionEngine.Number_of_Communities)  

		self.TowInducedGraph = cm.induced_graph(self.TowPartitionValue,self.TowGraphdataStructure)
		self.TowMultiple.clear()

		for key,value1 in self.TowPartitionValue.items():
			self.TowMultiple[value1].append(key)
		self.AssigNewValuesToGraphWidget(True)
		self.widget.Refresh()
		self.widget.scaleView(1.0001)
		self.changeViewinGraph()
		print "WARNING: Comunity coloring has been changed"


