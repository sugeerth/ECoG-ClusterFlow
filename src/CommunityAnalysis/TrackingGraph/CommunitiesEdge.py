
from PySide import QtCore, QtGui

import math
import pprint
import weakref

try:
    import numpy as np


except:
    print "Couldn't import all required packages. See README.md for a list of required packages and installation instructions."
    raise

def ColorToInt(color):
    r, g, b, a = map(np.uint32, color)
    return a << 24 | r << 16 | g << 8 | b

class CommunitiesEdge(QtGui.QGraphicsItem):

    Pi = math.pi
    TwoPi = 2.0 * Pi

    Type = QtGui.QGraphicsItem.UserType + 2

    def __init__(self, graphWidget, sourceNode, destNode, counter, sourceId, destId, weight):
        QtGui.QGraphicsItem.__init__(self)
        self.setAcceptHoverEvents(True)

        self.ForCommunities = False
        self.EdgeThreshold = 0 
        self.ColorEdgesFlag = False
        self.index = counter
        self.Alpha = 0.2
        self.sourceId = sourceId 
        self.destId = destId
        self.ColorMap = True
        self.HighlightedColorMap = False
        self.communityWeight = weight
        self.edgeThickness = 1
        self.thickHighlightedEdges = 3 
        self.ColorOnlySelectedNodesFlag =False
        if math.isnan(weight):
            weight = 0

        intersectingElements = list(set(sourceNode.correspondingNodes).intersection(destNode.correspondingNodes))

        # self.setToolTip(str(intersectingElements))
        # if (len(intersectingElements) == 0):
        #     return -1 
        self.sourcePoint = QtCore.QPointF()
        self.destPoint = QtCore.QPointF()
        self.graph = weakref.ref(graphWidget)
        self.source = weakref.ref(sourceNode)
        self.dest = weakref.ref(destNode)
        self.EdgeColor = QtGui.QColor(QtCore.Qt.black)
        self.source().addEdge(self)

        self.Color = sourceNode.CommunityColor
        self.weight = weight*2 + 2 

        # intersect = list(set(sourceNode.Nodeidss).intersection(destNode.Nodeidss))
        # pprint.pprint(intersectingElements)
        Tooltip = ""
        for i in intersectingElements:
            Tooltip+= str(i) +"\n"

        self.setToolTip(Tooltip)
        # print sourceNode.CommunityColor, sourceNode.colorvalue, sourceNode.Nodeidss

    def type(self):
        return Edge.Type

    def sourceNode(self):
        return self.source()

    def setWeight(self,weight):
        self.weight = float(weight)
        self.weight = weight*2 + 2 

    def getNodes(self,community):
        return self.graph().communityMultiple[community]

    def hoverEnterEvent(self, event):
        if self.ForCommunities:
            self.selectEdges()
            self.update()
            return
        QtGui.QGraphicsItem.hoverEnterEvent(self, event)

    def allnodesupdate(self):
        # Nodes = [item for item in self.scene().items() if isinstance(item, Node)]
        Nodes = self.graph().nodes
        for node in Nodes:
            node().setSelected(False)
            node().unsetOpaqueNodes()
            node().WhitePaint = False
            node().update()

    def selectInterModularEdges(self,communtiy1,community2):
        edges = self.graph().edges
        
        self.allnodesupdate()
        self.alledgesupdate()


        for i in communtiy1:
                self.graph().NodeIds[i].setOpaqueNodes()

        for j in community2: 
                self.graph().NodeIds[j].setOpaqueNodes()

        for edge in edges: 
            sourceBool = edge().sourceNode().counter-1 in communtiy1
            destBool = edge().destNode().counter-1 in community2
            if (sourceBool and destBool): 
                edge().ColorEdges()

        for edge in edges: 
            sourceBool = edge().sourceNode().counter-1 in community2
            destBool = edge().destNode().counter-1 in communtiy1

            if (sourceBool and destBool): 
                edge().ColorEdges()

        self.graph().Refresh()


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
        # if self.weight == 0: 
            # return
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

    def paint(self, painter, option, widget):
        if not self.source() or not self.dest():
            return

        if self.weight == 0: 
                return

        # Draw the line itself.
        if self.ColorOnlySelectedNodesFlag: 
            if not(self.ColorEdgesFlag):
                return

        line = QtCore.QLineF(self.sourcePoint, self.destPoint)
        # Should FIX the thickness values!!! fix me
        painter.save()
        """
        Painting the edge colors based on various factors
        Not painting the edges if they are below certain threshold 
        Painting the edges to be black or just based on their colors 
        Painting highlighted colormaps 
        edge Thickness is a function of  the weight of the edges 
        drawing z values so that they do not overalpp with others
        """

        if self.ColorMap:
            if self.EdgeThreshold < self.weight:
                if not(self.ColorEdgesFlag):
                    self.setZValue(1)
                    self.EdgeColor.setAlpha(255)
                    # painter.setPen(QtGui.QPen(self.EdgeColor ,self.edgeThickness , QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

                    pen = QtGui.QPen(QtGui.QPen(QtGui.QColor(self.Color),2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    pen.setWidthF(self.weight)
                    painter.setPen(pen)

                    painter.drawLine(line)
                else: 
                    self.setZValue(2)
                    if not(self.HighlightedColorMap):
                        # pointer to green
                        # painter.setPen(QtGui.QPen(self.EdgeColor ,self.communityWeight , QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                        # painter.setPen(QtGui.QPen(QtCore.Qt.darkGreen, self.weight , QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))


                        pen = QtGui.QPen(QtCore.Qt.darkGreen,2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                        pen.setWidthF(self.weight)
                        painter.setPen(pen)
                    else:
                        self.EdgeColor.setAlpha(255)
                        # painter.setPen(QtGui.QPen(QtCore.Qt.darkGray,self.weight , QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

                        pen = QtGui.QPen(QtCore.Qt.darkGreen,2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                        pen.setWidthF(self.weight)
                        painter.setPen(pen)

                        # painter.setPen(QtGui.QPen(self.EdgeColor, self.thickHighlightedEdges , QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
                    painter.drawLine(line)
        else: 
            if self.EdgeThreshold < self.weight:
                if not(self.ColorEdgesFlag):
                    self.setZValue(1)
                    # painter.setPen(QtGui.QPen(QtCore.Qt.black ,self.weight , QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))

                    pen = QtGui.QPen(QtCore.Qt.black ,2 , QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                    pen.setWidthF(self.weight)
                    painter.setPen(pen)
                    painter.drawLine(line)
                else: 
                    self.setZValue(2)
                    if not(self.HighlightedColorMap):
                       pen = QtGui.QPen(QtCore.Qt.darkGreen, self.weight , QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                       pen.setWidthF(self.weight)
                    else:
                       pen = QtGui.QPen(QtGui.QColor(self.Color), self.weight , QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)
                       pen.setWidthF(self.weight)
                    painter.setPen(pen) 
                    painter.drawLine(line)

        painter.restore()
