from PySide import QtCore, QtGui
import weakref
import math
distinguishableColors = [(123,123,123), (0,0,0) , (251,0,0), (178,223,138), (22,26,228), (253,191,111), (106,61,154), (255,255,153),(177,89,40), (0,92,49), (43,206,72), (255,204,153),(128,128,128), (148,255,181), (143, 124, 0), (157, 204, 0),(216,191,216),(194, 0, 136), (0, 51, 128), (255, 168, 187), (66, 102, 0),(255, 0, 16), (94, 241, 242), (0, 153, 143), (224, 255, 102),(116, 10, 255), (153, 0, 0), (255, 192, 203)]
FontBgColor = [4294967295, 4294967295, 4278190080, 4278190080, 4294967295, 4278190080, 4294967295, 4278190080, 4294967295, 4294967295, 4278190080, 4278190080, 4278190080, 4278190080, 4294967295, 4278190080, 4278190080, 4294967295, 4294967295, 4278190080, 4294967295, 4294967295, 4278190080, 4294967295, 4278190080, 4294967295, 4294967295, 4278190080]


class CommunityGraphNode(QtGui.QGraphicsItem):
	Type = QtGui.QGraphicsItem.UserType + 1

	def __init__(self, graphWidget, communities, correspondingNodes, Nodeid = -1):
		QtGui.QGraphicsItem.__init__(self)
		self.setAcceptHoverEvents(False)
		self.opacityValue = 255
		self.NodeCommunityColor = True
		self.graph = weakref.ref(graphWidget)
		self.edgeList = []
		self.SubNodesOfCommunityNodes = None
		self.node = []
		self.graphWidget = graphWidget
		self.subNodes = []
		self.colorvalue = []
		self.communityNode = None

		self.radius = 15
		self.CommunityColor = None
		self.colorvalue = None

		self.Nodeidss = communities
		self.CommunityColor = []

		Tooltip = ""
		for i in correspondingNodes:
			Tooltip+= str(i) +"\n"

		self.setToolTip(str(Tooltip))

		self.correspondingNodes = correspondingNodes
		# self.Abbr = self.graph().widget.correlationTable().AbbrName
		# self.Brain_Regions = self.graph().widget.correlationTable().RegionName[0]
		# else: 
		# 	self.setToolTip(str(Nodeid))
		self.newPos = QtCore.QPointF()
		# self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
		self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
		self.setCacheMode(self.DeviceCoordinateCache)
		# self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
		self.setFlag(QtGui.QGraphicsItem.ItemUsesExtendedStyleOption)
		self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)

		self.PutColor(self.graphWidget.widget.clut[self.Nodeidss])

		self.setZValue(-1)

	def type(self):
		return CommunityGraphNode.Type

	def addEdge(self, edge):
		self.edgeList.append(weakref.ref(edge))
		edge.adjust()

	def edges(self):
		return self.edgeList

	# def setSubNodes(self,subNodes):
	# 	self.subNodes = subNodes
	# 	stri = "" 
	# 	for i in self.subNodes: 
	# 		stri = stri + self.Brain_Regions[i]+"\n"
	# 	self.setToolTip(str(stri))

	# def setCommunity(self,node):
	# 	self.node = node
	# 	stri = "" 
	# 	for i in node: 
	# 		for j in i: 
	# 			stri = stri + self.Brain_Regions[j]+"\n"
	# 	self.Nodeidss = node[0][0]
	# 	self.setToolTip(str(stri))

	# def PutColor(self,colorvalue,alphaValue = -1):
	# 	self.colorvalue = colorvalue
	# 	self.CommunityColor = QtGui.QColor(colorvalue)
	# 	# if not(alphaValue==-1): 
	# 	# 	self.CommunityColor.setAlpha(alphaValue)
	# 	# 	print alphaValue
	# 	self.NodeCommunityColor = True
	# 	self.update()

	# def setTransparent(self):
	# 	for i in self.graph().AllNodes:
	# 		if i is self:
	# 			i.CommunityColor.setAlpha = 255
	# 		else:
	# 			i.CommunityColor.setAlpha = 45
	# 		i.update()

	# 	for i in self.graph().AllEdges:
	# 		if (i.source() is self) or (i.dest() is self):
	# 			i.ColorEdgesFlag = True
	# 		else:
	# 			i.ColorEdgesFlag = False
	# 		i.update()
		
	def boundingRect(self):
		adjust = 2.0
		return QtCore.QRectF(-40 - adjust, -40 - adjust,
							 70 + adjust, 70 + adjust)

	def shape(self):
		path = QtGui.QPainterPath()
		path.addEllipse(-40, -40, 70, 70)
		return path

	def PutColor(self,colorvalue):
		# print "Value", self.Nodeidss
		self.colorvalue = colorvalue
		self.CommunityColor = QtGui.QColor(colorvalue)
		self.update()

	def paint(self, painter, option, widget):
		painter.setPen(QtCore.Qt.NoPen)

		if self.graphWidget.VisualizationTheme == "ObjectFlow": 
			self.drawSubClusters(painter, self.radius)
			painter.setBrush(self.CommunityColor)
			painter.setPen(QtGui.QPen(QtCore.Qt.black, 0))
			if (option.state & QtGui.QStyle.State_Selected):
				circle_path = QtGui.QPainterPath()
				painter.setPen(QtGui.QPen(QtCore.Qt.blue, 3))        
				circle_path.addEllipse(QtCore.QPointF(0,0),self.radius+2,self.radius+2);
				painter.drawPath(circle_path)
			else:
				painter.drawEllipse(-4, -4, self.radius, self.radius)

		elif self.graphWidget.VisualizationTheme == "ThemeRiver":
			self.drawSubClustersTheme(painter,option, self.radius)

		# Drawing the CirclePath Should denote a value 
	def drawSubClustersTheme(self, painter,option, Radius):
		Clusters = len(self.correspondingNodes)	
		step = float(math.pi/Clusters) 
		Radius = self.radius
		angle = 0.0 
		c = 0

		for i in self.correspondingNodes:
			painter.setBrush(self.CommunityColor)
			painter.setPen(QtGui.QPen(QtCore.Qt.black, 0))	

			radius = float(self.graphWidget.widget.Centrality[i]*100)
			Thickness = float(radius) * 2

			if (option.state & QtGui.QStyle.State_Selected):
				painter.drawRect(-12, -12, 25, Thickness)
			else: 
				painter.drawRect(-12, -12, 20, Thickness)

			# x = Radius * math.sin((2*c*math.pi)/Clusters)
			# y = Radius * math.cos((2*c*math.pi)/Clusters)
			# c = c + 1
			# # print radiuss
			# painter.drawEllipse(x,y,radius, radius)
			# angle = angle + step
		# Drawing the CirclePath Should denote a value 
	def drawSubClusters(self, painter, radius):

		# Extrapolate the graph metrics and rank
		Clusters = len(self.correspondingNodes)	
		step = float(math.pi/Clusters) 
		Radius = self.radius
		angle = 0.0 
		c = 0

		for i in self.correspondingNodes:
			painter.setBrush(QtCore.Qt.blue)
			painter.setPen(QtGui.QPen(QtCore.Qt.black, 0))	

			x = Radius * math.sin((2*c*math.pi)/Clusters)
			y = Radius * math.cos((2*c*math.pi)/Clusters)

			c = c + 1
			radius = float(self.graphWidget.widget.Centrality[i]*100)
			# print radiuss
			painter.drawEllipse(x,y,radius, radius)
			angle = angle + step

	# def selectCommunities(self):
	# 	nodeidss = self.Nodeidss
	# 	self.graph().widget.NodeIds[nodeidss].communitySelected(nodeidss,False,1)

	# def selectSubCommunities(self,subNodes):
	# 	self.graph().widget.NodeIds[self.Nodeidss].alledgesupdate()
	# 	for i in self.subNodes: 
	# 		for edge in self.graph().widget.edges:
	# 			sliderBool = self.graph().widget.EdgeSliderValue < edge().weight
	# 			sourceBool = edge().sourceNode().counter-1 in self.subNodes
	# 			destBool = edge().destNode().counter-1 in self.subNodes
	# 			if (sourceBool and destBool) and sliderBool:
	# 				edge().ColorEdges() 
	# 				edge().update()
	# 			edge().update()

	# def setSubNodesAttached(self,SubNodesOfCommunityNodes):
	# 	self.SubNodesOfCommunityNodes = SubNodesOfCommunityNodes

	# def makeOpaqueCommunityNodes(self,communityNode):
	# 	for node1 in communityNode:
	# 		node1.CommunityColor.setAlpha(255)
	# 		node1.setSelected(False)
	# 		node1.update()
		
	# 	edges1 = self.graph().widget.communityObject.edges

	# 	for edge in edges1:
	# 		edge.communityAlpha(True)
	# 		edge.update()

	# def makeTransparentCommunityNodes(self,communityNode):
	# 	for node1 in communityNode:
	# 		node1.CommunityColor.setAlpha(50)
	# 		node1.setSelected(False)
	# 		node1.update()

	# 	edges1 = self.graph().widget.communityObject.edges

	# 	for edge in edges1:
	# 		edge.communityAlpha(False)
	# 		edge.update()

	# def selectCommunitiess(self,node):
	# 	self.graph().widget.NodeIds[self.Nodeidss].alledgesupdate()
	# 	node = self.node
	# 	for i in self.node: 
	# 		node = node + i

	# 	# Hack for the paper  
	# 	communityNode = self.graph().widget.communityObject.nodes
	# 	NodesToBeHighlightedInCommunityGraph = self.SubNodesOfCommunityNodes 

	# 	self.makeTransparentCommunityNodes(communityNode)

	# 	for node1 in communityNode:
	# 		for i in NodesToBeHighlightedInCommunityGraph:
	# 			if (i.Nodeidss==node1.counter-1): 
	# 				self.graph().widget.communityObject.NodeIds[node1.counter-1].setSelected(True)

	# 	edges1 = self.graph().widget.communityObject.edges

	# 	for edge in edges1:
	# 		for i in NodesToBeHighlightedInCommunityGraph:
	# 			for j in NodesToBeHighlightedInCommunityGraph:
	# 				bool1 = (self.graph().widget.partition[i.subNodes[0]]) == edge.sourceId and self.graph().widget.partition[j.subNodes[0]] == edge.destId 
	# 				bool2 = (self.graph().widget.partition[i.subNodes[0]]) == edge.destId and self.graph().widget.partition[j.subNodes[0]] == edge.sourceId
	# 				if bool1 or bool2: 
	# 					edge.communityAlpha(True)
					
	# 	self.graph().widget.communityGraphUpdate()

	# 	for i in node:
	# 		for edge in self.graph().widget.edges:
	# 			sliderBool = self.graph().widget.EdgeSliderValue < edge().weight
	# 			sourceBool = edge().sourceNode().counter-1 in node
	# 			destBool = edge().destNode().counter-1 in node
	# 			if (sourceBool and destBool) and sliderBool:
	# 				edge().ColorEdges() 
	# 				edge().update()
	# 			edge().update()

	# 	self.graph().widget.Refresh()

	def makeOpaqueCommunityNodes(self,communityNode):
		for node1 in communityNode:
			node1.CommunityColor.setAlpha(55)
			node1.setSelected(False)
			node1.update()
		
		edges1 = self.graphWidget.widget.communityObject.edges

		for edge in edges1:
			edge.communityAlpha(False)
			edge.update()

	def hoverEnterEvent(self, event):
		self.graphWidget.widget.NodeIds[self.Nodeidss].allnodesupdate()
		self.makeOpaqueCommunityNodes(self.graphWidget.widget.communityObject.nodes)

		self.setSelected(True)
		self.update()

		for i in self.graphWidget.widget.communityMultiple[self.Nodeidss]:
			self.graphWidget.widget.NodeIds[i].setSelected(True)
			self.graphWidget.widget.NodeIds[i].CommunityColor.setAlpha(255) 
		
		self.update()

		self.graphWidget.widget.communityGraphUpdate()
		QtGui.QGraphicsItem.hoverEnterEvent(self, event)
		return

	def mousePressEvent(self, event):
		# print "Cluster that is responsible",self.Nodeidss 
		# self.makeOpaqueCommunityNodes(self.graph().widget.communityObject.nodes)
		# self.graph().widget.NodeIds[self.Nodeidss].unsetOpaqueNodes()
		# self.graph().widget.NodeIds[self.Nodeidss].SelectedNode(self.Nodeidss,False, 1)
		# self.graph().widget.NodeIds[self.Nodeidss].setSelected(True)
		QtGui.QGraphicsItem.mousePressEvent(self, event)
