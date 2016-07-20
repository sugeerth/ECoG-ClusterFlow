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
		self.newPos = QtCore.QPointF()
		self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
		self.setCacheMode(self.DeviceCoordinateCache)
		self.setFlag(QtGui.QGraphicsItem.ItemUsesExtendedStyleOption)
		self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)

		self.PutColor(self.graphWidget.widget.communityDetectionEngine.clut[self.Nodeidss])

		self.setZValue(-1)

	def type(self):
		return CommunityGraphNode.Type

	def addEdge(self, edge):
		self.edgeList.append(weakref.ref(edge))
		edge.adjust()

	def edges(self):
		return self.edgeList

	def boundingRect(self):
		adjust = 2.0
		print -40 - adjust, -40 - adjust,70 + adjust, 70 + adjust
		return QtCore.QRectF(-40 - adjust, -40 - adjust,
							 70 + adjust, 70 + adjust)

	def shape(self):
		path = QtGui.QPainterPath()
		path.addEllipse(-40, -40, 70, 70)
		return path

	def PutColor(self,colorvalue):
		self.colorvalue = colorvalue
		self.CommunityColor = QtGui.QColor(colorvalue)
		self.update()





