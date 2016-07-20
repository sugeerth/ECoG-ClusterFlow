
from PySide import QtCore, QtGui

import math
import weakref

try:
    import numpy as np


except:
    print "Couldn't import all required packages. See README.md for a list of required packages and installation instructions."
    raise

def ColorToInt(color):
    r, g, b, a = map(np.uint32, color)
    return a << 24 | r << 16 | g << 8 | b

class Edge(QtGui.QGraphicsItem):

    Pi = math.pi
    TwoPi = 2.0 * Pi

    Type = QtGui.QGraphicsItem.UserType + 2

    def __init__(self, graphWidget, sourceNode, destNode, counter, sourceId, destId, MaxValue,weight=1,ForCommunities=False,):
        QtGui.QGraphicsItem.__init__(self)
        self.setAcceptHoverEvents(False)
        self.EdgeThreshold = MaxValue - 0.01
        self.ColorEdgesFlag = False
        self.index = counter
        self.Alpha = 0.2
        self.sourceId = sourceId 
        self.destId = destId
        self.ColorMap = True
        self.ForCommunities= ForCommunities
        self.HighlightedColorMap = False
        self.communityWeight = weight
        self.edgeThickness = 1
        self.thickHighlightedEdges = 3 
        self.ColorOnlySelectedNodesFlag =False
        if math.isnan(weight):
            weight = 0
        
        self.weight = weight

        if ForCommunities:
            self.communtiyColor = ColorToInt((0,0,0,255))
            self.communtiyColor1 = QtGui.QColor(self.communtiyColor)
            self.setToolTip(str("InterModular Strength:  "+"{0:.2f}".format(weight)))
        self.sourcePoint = QtCore.QPointF()
        self.destPoint = QtCore.QPointF()
        self.graph = weakref.ref(graphWidget)
        self.source = weakref.ref(sourceNode)
        self.dest = weakref.ref(destNode)
        self.EdgeColor = QtGui.QColor(self.graph().EdgeColor[self.index])
        self.source().addEdge(self)

    def type(self):
        return Edge.Type

    def destNode(self):
        return self.dest()

    def sourceNode(self):
        return self.source()
    def sourceNode(self):
        return self.source()

    def setWeight(self,weight):
        self.weight = float(weight)

    def getNodes(self,community):
        return self.graph().communityMultiple[community]

    def hoverEnterEvent(self, event):
        if self.ForCommunities:
            self.selectEdges()
            self.update()
            return
        QtGui.QGraphicsItem.hoverEnterEvent(self, event)

    def allnodesupdate(self):
        Nodes = self.graph().nodes
        for node in Nodes:
            node().setSelected(False)
            node().unsetOpaqueNodes()
            node().WhitePaint = False
            node().update()

    def selectInterModularEdges(self,communtiy1,community2):
        pass


    def selectEdges(self):
        communtiy1 = self.getNodes(self.sourceId)
        community2 = self.getNodes(self.destId)
        self.selectInterModularEdges(communtiy1,community2)

    def alledgesupdate(self):
        edges = self.graph().edges
        for edge in edges:
            edge().ColorEdgesFlag = False

    def setSourceNode(self, node):
        self.source = weakref.ref(node)
        self.adjust()

    def destNode(self):
        return self.dest()

    def setHighlightedColorMap(self,state):
        self.HighlightedColorMap = state

    def ColorOnlySelectedNode(self,state):
        self.ColorOnlySelectedNodesFlag = state

    def setDestNode(self, node):
        self.dest = weakref.ref(node)
        self.adjust()
    
    def setColorMap(self,state):
        self.ColorMap = state

    def ColorEdges(self):
        self.ColorEdgesFlag = True
        self.update()
 
    def setEdgeThickness(self,value):
        self.edgeThickness = float(value)
        self.update()

    def Threshold(self,value):
        # print "Threshold in Edge"
        self.EdgeThreshold = float(value)
        self.update()

    def adjust(self):
        if not self.source() or not self.dest():
            return
        line = QtCore.QLineF(self.mapFromItem(self.source(), 0, 0), self.mapFromItem(self.dest(), 0, 0))
        self.sourcePoint = line.p1() #+ edgeOffset
        self.destPoint = line.p2() #- edgeOffset

    def boundingRect(self):
        """
        Computes the bounding recatangle for every edge 
        """
        if not self.source() or not self.dest():
            return QtCore.QRectF()

        return QtCore.QRectF(self.sourcePoint,
                             QtCore.QSizeF(self.destPoint.x() - self.sourcePoint.x(),
                                           self.destPoint.y() - self.sourcePoint.y()))

    def communityAlpha(self,boolValue,value=-1):
        if boolValue:
            self.communtiyColor1.setAlpha(255)
        else:
            self.communtiyColor1.setAlpha(55)
        self.update()

