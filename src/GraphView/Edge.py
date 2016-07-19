
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

    def sourceNode(self):
        return self.source()

    def setWeight(self,weight):
        self.weight = float(weight)
