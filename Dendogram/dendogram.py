import csv
import numpy as np
import pprint
import weakref
import tempfile
import time
import math
from PySide import QtCore, QtGui
from collections import OrderedDict


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
except:
	print "Couldn't import all required packages. See README.md for a list of required packages and installation instructions."
	raise

"""Class responsible for generating the Dendogram"""

class Edge(QtGui.QGraphicsItem):
	Pi = math.pi
	TwoPi = 2.0 * Pi

	Type = QtGui.QGraphicsItem.UserType + 2

	def __init__(self, sourceNode, destNode):
		QtGui.QGraphicsItem.__init__(self)

		self.arrowSize = 10.0
		self.ColorEdgesFlag = True
		self.colorvalue = []
		self.sourcePoint = QtCore.QPointF()
		self.destPoint = QtCore.QPointF()
		self.setAcceptedMouseButtons(QtCore.Qt.NoButton)
		self.source = weakref.ref(sourceNode)
		self.dest = weakref.ref(destNode)
		self.source().addEdge(self)
		self.dest().addEdge(self)
		self.adjust()

	def type(self):
		return Edge.Type

	def sourceNode(self):
		return self.source()

	def PutColor(self,colorvalue):
		self.colorvalue = colorvalue
		self.CommunityColor = QtGui.QColor(colorvalue)
		self.NodeCommunityColor = True

	def setSourceNode(self, node):
		self.source = weakref.ref(node)
		self.adjust()

	def destNode(self):
		return self.dest()

	def setDestNode(self, node):
		self.dest = weakref.ref(node)
		self.adjust()

	def adjust(self):
		if not self.source() or not self.dest():
			return

		line = QtCore.QLineF(self.mapFromItem(self.source(), 0, 0), self.mapFromItem(self.dest(), 0, 0))
		length = line.length()

		if length == 0.0:
			return

		edgeOffset = QtCore.QPointF((line.dx() * 10) / length, (line.dy() * 10) / length)

		self.prepareGeometryChange()
		self.sourcePoint = line.p1() + edgeOffset
		self.destPoint = line.p2() - edgeOffset

	def boundingRect(self):
		"""
		Computes the bounding recatangle for every edge 
		"""
		if not self.source() or not self.dest():
			return QtCore.QRectF()

		return QtCore.QRectF(self.sourcePoint,
							 QtCore.QSizeF(self.destPoint.x() - self.sourcePoint.x(),
										   self.destPoint.y() - self.sourcePoint.y()))

	def paint(self, painter, option, widget):
		if not self.source() or not self.dest():
			return

		# Draw the line itself.
		line = QtCore.QLineF(self.sourcePoint, self.destPoint)

		if not(self.ColorEdgesFlag):
			return
		if line.length() == 0.0:
			return


		painter.setPen(QtGui.QPen(QtCore.Qt.black, 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
		painter.drawLine(line)

		painter.setBrush(QtCore.Qt.black)

class DendoNode(QtGui.QGraphicsItem):
	Type = QtGui.QGraphicsItem.UserType + 1

	def __init__(self, graphWidget, Nodeid = -1):
		QtGui.QGraphicsItem.__init__(self)
		self.setAcceptHoverEvents(True)
		self.opacityValue = 255
		self.NodeCommunityColor = True
		self.graph = weakref.ref(graphWidget)
		self.edgeList = []
		self.SubNodesOfCommunityNodes = None
		self.node = []
		self.subNodes = []
		self.colorvalue = []
		self.communityNode = None

		self.Nodeidss = Nodeid
		self.CommunityColor = []
		self.Abbr = self.graph().widget.correlationTable().AbbrName
		self.Brain_Regions = self.graph().widget.correlationTable().RegionName[0]
		if not(Nodeid == -1): 
			self.setToolTip(str("Left Supra Marginal gyrus" + "\n" + "LSMG"))
		else: 
			self.setToolTip(str(Nodeid))
		self.newPos = QtCore.QPointF()
		self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
		self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
		self.setCacheMode(self.DeviceCoordinateCache)
		self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
		self.setFlag(QtGui.QGraphicsItem.ItemUsesExtendedStyleOption)
		self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)

		self.setZValue(-1)

	def type(self):
		return DendoNode.Type

	def addEdge(self, edge):
		self.edgeList.append(weakref.ref(edge))
		edge.adjust()

	def edges(self):
		return self.edgeList

	def setSubNodes(self,subNodes):
		self.subNodes = subNodes
		stri = "" 
		for i in self.subNodes: 
			stri = stri + self.Brain_Regions[i]+"\n"
		self.setToolTip(str(stri))

	def setCommunity(self,node):
		self.node = node
		stri = "" 
		for i in node: 
			for j in i: 
				stri = stri + self.Brain_Regions[j]+"\n"
		self.Nodeidss = node[0][0]
		self.setToolTip(str(stri))

	def PutColor(self,colorvalue,alphaValue = -1):
		self.colorvalue = colorvalue
		self.CommunityColor = QtGui.QColor(colorvalue)
		# if not(alphaValue==-1): 
		# 	self.CommunityColor.setAlpha(alphaValue)
		# 	print alphaValue
		self.NodeCommunityColor = True
		self.update()

	def setTransparent(self):
		for i in self.graph().AllNodes:
			if i is self:
				i.CommunityColor.setAlpha = 255
			else:
				i.CommunityColor.setAlpha = 45
			i.update()

		for i in self.graph().AllEdges:
			if (i.source() is self) or (i.dest() is self):
				i.ColorEdgesFlag = True
			else:
				i.ColorEdgesFlag = False
			i.update()

	def boundingRect(self):
		adjust = 2.0
		return QtCore.QRectF(-10 - adjust, -10 - adjust,
							 23 + adjust, 23 + adjust)

	def shape(self):
		path = QtGui.QPainterPath()
		path.addEllipse(-10, -10, 20, 20)
		return path

	def paint(self, painter, option, widget):
		painter.setPen(QtCore.Qt.NoPen)
		painter.setBrush(self.CommunityColor)
		painter.setPen(QtGui.QPen(QtCore.Qt.black, 0))
		painter.drawEllipse(-4, -4, 10, 10)

	def selectCommunities(self):
		nodeidss = self.Nodeidss
		self.graph().widget.NodeIds[nodeidss].communitySelected(nodeidss,False,1)

	def selectSubCommunities(self,subNodes):
		self.graph().widget.NodeIds[self.Nodeidss].alledgesupdate()
		for i in self.subNodes: 
			for edge in self.graph().widget.edges:
				sliderBool = self.graph().widget.EdgeSliderValue < edge().weight
				sourceBool = edge().sourceNode().counter-1 in self.subNodes
				destBool = edge().destNode().counter-1 in self.subNodes
				if (sourceBool and destBool) and sliderBool:
					edge().ColorEdges() 
					edge().update()
				edge().update()

	def setSubNodesAttached(self,SubNodesOfCommunityNodes):
		self.SubNodesOfCommunityNodes = SubNodesOfCommunityNodes

	def makeOpaqueCommunityNodes(self,communityNode):
		for node1 in communityNode:
			node1.CommunityColor.setAlpha(255)
			node1.setSelected(False)
			node1.update()
		
		edges1 = self.graph().widget.communityObject.edges

		for edge in edges1:
			edge.communityAlpha(True)
			edge.update()

	def makeTransparentCommunityNodes(self,communityNode):
		for node1 in communityNode:
			node1.CommunityColor.setAlpha(50)
			node1.setSelected(False)
			node1.update()

		edges1 = self.graph().widget.communityObject.edges

		for edge in edges1:
			edge.communityAlpha(False)
			edge.update()

	def selectCommunitiess(self,node):
		self.graph().widget.NodeIds[self.Nodeidss].alledgesupdate()
		node = self.node
		for i in self.node: 
			node = node + i

		# Hack for the paper  
		communityNode = self.graph().widget.communityObject.nodes
		NodesToBeHighlightedInCommunityGraph = self.SubNodesOfCommunityNodes 

		self.makeTransparentCommunityNodes(communityNode)

		for node1 in communityNode:
			for i in NodesToBeHighlightedInCommunityGraph:
				if (i.Nodeidss==node1.counter-1): 
					self.graph().widget.communityObject.NodeIds[node1.counter-1].setSelected(True)

		edges1 = self.graph().widget.communityObject.edges

		for edge in edges1:
			for i in NodesToBeHighlightedInCommunityGraph:
				for j in NodesToBeHighlightedInCommunityGraph:
					bool1 = (self.graph().widget.partition[i.subNodes[0]]) == edge.sourceId and self.graph().widget.partition[j.subNodes[0]] == edge.destId 
					bool2 = (self.graph().widget.partition[i.subNodes[0]]) == edge.destId and self.graph().widget.partition[j.subNodes[0]] == edge.sourceId
					if bool1 or bool2: 
						edge.communityAlpha(True)
					
		self.graph().widget.communityGraphUpdate()

		for i in node:
			for edge in self.graph().widget.edges:
				sliderBool = self.graph().widget.EdgeSliderValue < edge().weight
				sourceBool = edge().sourceNode().counter-1 in node
				destBool = edge().destNode().counter-1 in node
				if (sourceBool and destBool) and sliderBool:
					edge().ColorEdges() 
					edge().update()
				edge().update()

		self.graph().widget.Refresh()

	def hoverEnterEvent(self, event):
		self.graph().widget.NodeIds[self.Nodeidss].allnodesupdate()
		self.makeOpaqueCommunityNodes(self.graph().widget.communityObject.nodes)
		# self.setTransparent()

		if self.subNodes: 
			# logic for selecting only those nodes which are in self.subNodes
			for i in self.subNodes:
				self.graph().widget.NodeIds[i].setSelected(True)
			self.selectSubCommunities(self.subNodes)
		elif self.node: 
			for i in self.node:
				for j in i: 
					self.graph().widget.NodeIds[j].setSelected(True)
			self.selectCommunitiess(self.node) # slecet the communities
		else: 
			self.selectCommunities()
			self.graph().widget.NodeIds[self.Nodeidss].setSelected(True)
		self.update()
		self.graph().widget.communityGraphUpdate()
		QtGui.QGraphicsItem.hoverEnterEvent(self, event)
		return
	def mousePressEvent(self, event):
		self.makeOpaqueCommunityNodes(self.graph().widget.communityObject.nodes)
		self.graph().widget.NodeIds[self.Nodeidss].unsetOpaqueNodes()
		self.graph().widget.NodeIds[self.Nodeidss].SelectedNode(self.Nodeidss,False, 1)
		self.graph().widget.NodeIds[self.Nodeidss].setSelected(True)
		QtGui.QGraphicsItem.mousePressEvent(self, event)

class dendogram(QtGui.QGraphicsView):
	def __init__(self,widget, graphData):
		QtGui.QGraphicsView.__init__(self)

		self.g = graphData
		self.Order =[]
		self.height = 250 
		self.widget = widget
		self.width = 550
		self.AllNodes = None
		self._2NodeIds = []
		self.NodeIds = []
		self._3NodeIds = []
		self.setMinimumSize(400, 150)
		self.centerPos = (self.width/2 + 1, self.height -5)
		self.dendogram = None
		self.generateDendogram()
		self.initUI()
	
	def dendogramPos(self):

		for level in range(len(self.dendogram)):
			self.Order.append(OrderedDict(sorted(self.dendogram[level].items(), key=lambda t: t[1])))

		try: 
			if self.dendogram[1]:
					d =dict()
					for i,j in self.dendogram[0].items():
						d[i] = self.dendogram[1][j]
					d = OrderedDict(sorted(d.items(), key=lambda (k,v): v))
					for i,j in d.items():
						self.Order[0][i] = j
					self.Order[0] = OrderedDict(sorted(self.Order[0].items(), key=lambda (k,v): v))
					for i,j in self.Order[0].items():
						self.Order[0][i] = self.dendogram[0][i]
		except IndexError:
			pass

		"""
							width/2, height  
								  *
							------|-----
							*			*
							|			|
						----|----   ----|-----
						|	|	|	|	|	 |
						|	|	|	|	|	 |

		(HEIGHT-100)/len(sel.dendogram) 
		width-50 / len(self.self.dendogram[0])

		1) center node
		2) iterate then next level 
		3) then second level 
		4) then edges between these 
		"""
		self.centerPos = (self.width/2 + 1, self.height -10)
		heightBlock = self.height/len(self.dendogram)
		widthBlock = np.zeros(len(self.dendogram))

		# HAck to set level to be one 
		if self.widget.level == -1: 
			self.widget.level = 1
			self.widget.propagateLevelValue.emit(self.widget.level)

		for i in range(len(self.dendogram)): 
			widthBlock[i] = self.width/len(self.dendogram[i])

		height = 0
		width = 0

		self.LeafNodeIds = []
		for node,Community in self.Order[0].items():
			node_value = DendoNode(self,node)
			node_value.setPos(width,height)
			try: 
				if self.dendogram[1]:
					_1temp = self.Order[1][Community]
				if self.dendogram[2]:
					_2temp = self.Order[2][_1temp]
			except IndexError:
					temp = Community

			if self.widget.level == 1: 
				node_value.PutColor(self.widget.clut[_1temp])
			elif self.widget.level == 2:
				node_value.PutColor(self.widget.clut[_2temp])
			else:
				node_value.PutColor(self.widget.clut[Community])
			# node_value.PutColor(self.widget.clut[temp])
			self.LeafNodeIds.append(node_value)
			self.scene.addItem(node_value)			
			width = width + widthBlock[0] 
		j = 0 
		counter = 0
		height = -heightBlock

		for i in self.Order[0].values(): 
			if j > 0:
				if i == self.Order[0].values()[j-1] :
					counter = counter + 1
				else:
					counter = 0
			try:
				if not(i == self.Order[0].values()[j+1]):
					a =  self.LeafNodeIds[j-counter].x() + (self.LeafNodeIds[j].x() - self.LeafNodeIds[j-counter].x())/2 
					node_value =DendoNode(self,i)
					node_value.setPos(a,height)
					self.NodeIds.append(node_value)
					try: 
						if self.dendogram[1]:
							_1temp = self.Order[1][i]
						if self.dendogram[2]:
							_2temp = self.Order[2][_1temp]
					except IndexError:
							temp = i

					if self.widget.level == 1: 
						node_value.PutColor(self.widget.clut[_1temp])
					elif self.widget.level == 2:
						node_value.PutColor(self.widget.clut[_2temp])
					else:
						node_value.PutColor(self.widget.clut[i])

					self.scene.addItem(node_value)
					CorressIds = []
					for p in range(counter+1):
						self.scene.addItem(Edge(node_value,self.LeafNodeIds[j-p]))
						CorressIds.append(self.LeafNodeIds[j-p].Nodeidss)
					node_value.setSubNodes(CorressIds)
			except IndexError:
				a =  self.LeafNodeIds[j-counter].x() + (self.LeafNodeIds[j].x() - self.LeafNodeIds[j-counter].x())/2 
				node_value = DendoNode(self,i)
				node_value.setPos(a,height)

				try: 
					if self.dendogram[1]:
						_1temp = self.Order[1][i]
					if self.dendogram[2]:
						_2temp = self.Order[2][_1temp]
						print self.widget.clut[_2temp]
				except IndexError:
						temp = i

				# print self.widget.level 
				
				if self.widget.level == 1: 
					node_value.PutColor(self.widget.clut[_1temp])
				elif self.widget.level == 2:
					node_value.PutColor(self.widget.clut[_2temp])
				else:
					node_value.PutColor(self.widget.clut[i])


				self.NodeIds.append(node_value)
				self.scene.addItem(node_value)
				CorressIds = []
				for p in range(counter+1):
						self.scene.addItem(Edge(node_value,self.LeafNodeIds[j-p]))
						CorressIds.append(self.LeafNodeIds[j-p].Nodeidss)
				node_value.setSubNodes(CorressIds)
			j = j + 1

		j=0
		height = height -heightBlock
		counter = 0
		try:
			if self.dendogram[1]: 
				for i in self.Order[1].values(): 
					if j > 0:
					 	if i == self.Order[1].values()[j-1] :
							counter = counter + 1  
						else:
							counter = 0
					try:
						if not(i == self.Order[1].values()[j+1]):
							a =  self.NodeIds[j-counter].x() + (self.NodeIds[j].x() - self.NodeIds[j-counter].x())/2 
							node_value =DendoNode(self)
							node_value.setPos(a,height)

							if self.widget.level == 1: 
								node_value.PutColor(self.widget.clut[i])
							elif self.widget.level == 2:
								print self.dendogram[2][i], i 
								node_value.PutColor(self.widget.clut[self.dendogram[2][i]])
							else:	
								node_value.PutColor((128 << 24 | int(128) << 16 | int(128) << 8 | int(128)))

							# node_value.PutColor(self.widget.clut[i])
							self._2NodeIds.append(node_value)
							self.scene.addItem(node_value)
							CommunityNodes = []
							SubNodesOfRootNode = []
							for q in range(counter+1):
								self.scene.addItem(Edge(node_value,self.NodeIds[j-q]))
								CommunityNodes.append(self.NodeIds[j-q].subNodes)
								SubNodesOfRootNode.append(self.NodeIds[j-i])
							node_value.setSubNodesAttached(SubNodesOfRootNode)
							node_value.setCommunity(CommunityNodes)

					except IndexError:
						a =  self.NodeIds[j-counter].x() + (self.NodeIds[j].x() - self.NodeIds[j-counter].x())/2 
						node_value =DendoNode(self)
						node_value.setPos(a,height)

						if self.widget.level == 1: 
							node_value.PutColor(self.widget.clut[i])
						elif self.widget.level == 2:
							node_value.PutColor(self.widget.clut[self.dendogram[2][i]])
						else:
							node_value.PutColor((128 << 24 | int(128) << 16 | int(128) << 8 | int(128)))

						# node_value.PutColor(self.widget.clut[i])
						self._2NodeIds.append(node_value)
						self.scene.addItem(node_value)
						CommunityNodes = []
						SubNodesOfRootNode = []
						for q in range(counter+1):
							self.scene.addItem(Edge(node_value,self.NodeIds[j-q]))
							CommunityNodes.append(self.NodeIds[j-q].subNodes)
							SubNodesOfRootNode.append(self.NodeIds[j-q])
						node_value.setSubNodesAttached(SubNodesOfRootNode)
						node_value.setCommunity(CommunityNodes)
					j = j + 1
		except IndexError:
			pass
		# LEVEL 2 of the DENDOGRAM
		j=0
		height = height -heightBlock
		counter = 0
		try: 
			if self.dendogram[2]:
				for i in self.Order[2].values(): 
					if j > 0:
					 	if i == self.Order[2].values()[j-1] :
							counter = counter + 1  
						else:
							counter = 0
					try:
						if not(i == self.Order[2].values()[j+1]):
							a =  self._2NodeIds[j-counter].x() + (self._2NodeIds[j].x() - self._2NodeIds[j-counter].x())/2 
							node_value =DendoNode(self)
							node_value.setPos(a,height)

							if self.widget.level == 2:
								node_value.PutColor(self.widget.clut[i])
							else: 
								node_value.PutColor((128 << 24 | int(128) << 16 | int(128) << 8 | int(128)))

							self._3NodeIds.append(node_value)
							self.scene.addItem(node_value)

							CommunityNodes = []
							SubNodesOfRootNode = []

							for f in range(counter+1):
								self.scene.addItem(Edge(node_value,self._2NodeIds[j-f]))

					except IndexError:
						a =  self._2NodeIds[j-counter].x() + (self._2NodeIds[j].x() - self._2NodeIds[j-counter].x())/2 
						node_value =DendoNode(self)
						node_value.setPos(a,height)

						if self.widget.level == 2:
							node_value.PutColor(self.widget.clut[i])
						else: 
							node_value.PutColor((128 << 24 | int(128) << 16 | int(128) << 8 | int(128)))
						self._3NodeIds.append(node_value)
						self.scene.addItem(node_value)
						CommunityNodes = []
						SubNodesOfRootNode = []
						for z in range(counter+1):
							self.scene.addItem(Edge(node_value,self._2NodeIds[j-z]))
					j = j + 1	
		except IndexError:
			pass
		try:
			if self.dendogram[3]:
				print "IN constrcution will see Level 3 of the dendogram view" 
				pass
		except IndexError:
				print "Total Levels of hierarchy!!",len(self.dendogram)


		self.AllNodes = [item for item in self.scene.items() if isinstance(item, DendoNode)]
		self.AllEdges = [item for item in self.scene.items() if isinstance(item, Edge)]

		del self.NodeIds,self.LeafNodeIds,self._2NodeIds,self._3NodeIds

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
	 	self.dendogramPos()
		self.setSceneRect(self.Scene_to_be_updated.itemsBoundingRect())
		self.setScene(self.Scene_to_be_updated)
		self.fitInView(self.Scene_to_be_updated.itemsBoundingRect(),QtCore.Qt.KeepAspectRatio)
		self.scaleView(math.pow(2.0, -600/ 1040.0))

	def wheelEvent(self, event):
		self.scaleView(math.pow(2.0, -event.delta() / 1040.0))

	def scaleView(self, scaleFactor):
		factor = self.matrix().scale(scaleFactor, scaleFactor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()
		if factor < 0.07 or factor > 100:
			return
		self.scale(scaleFactor, scaleFactor)
		del factor

	def generateDendogram(self):
		self.Order = []
		self.dendogram = cm.generate_dendogram(self.g)
		for level in range(len(self.dendogram)):
			self.Order.append(OrderedDict(sorted(self.dendogram[level].items(), key=lambda t: t[1])))
