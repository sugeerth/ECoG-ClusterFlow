import time
from PySide import QtCore, QtGui
import weakref
import pprint
import numpy as np
from Edge import Edge 

class Translate(QtCore.QObject):
    def __init__(self):
        QtCore.QObject.__init__(self)
    def set(self,string):
        return  str(self.tr(string))

class Node(QtGui.QGraphicsItem):
    Type = QtGui.QGraphicsItem.UserType + 1

    def __init__(self, graphWidget, counter,correlationTable,ForCommunities=False):
        QtGui.QGraphicsItem.__init__(self)

        """Accepting hover events """
        self.setAcceptHoverEvents(False)

        self.opacityValue = 255

        self.ScalarSize = False
        self.graph = weakref.ref(graphWidget)
        self.edgeList = []
        self.Red = False
        self.ForCommunities = True
        self.ForCommunities = ForCommunities
        self.CommunityColor = None
        self.MousePressede = True
        self.setTransp = True
        self.NodeCommunityColor = False
        self.counter = counter
        self.aplha = 0.2
        self.colorTransparency = True 
        self.First = True
        self.degreeCentrality = 1.0
        
        self.Brain_Regions = correlationTable.RegionName[0]
        
        try: 
            for i in range (self.counter-1):
                self.Brain_Regions[i] = self.Brain_Regions[i].replace(' ','')
        except IndexError:
            print "Exception", i

        self.colorvalue = []

        self.CommunityTextColor = 4278190080
        self.Selected = False 
        self.Translate = Translate()
        self.WhitePaint= False
        self.newPos = QtCore.QPointF()
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        self.setFlag(QtGui.QGraphicsItem.ItemUsesExtendedStyleOption)
        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)

        if not(self.ForCommunities):
            self.nodeColor = QtGui.QColor(self.graph().DataColor[self.counter])

        self.setCacheMode(self.DeviceCoordinateCache)
        if (len(correlationTable.data) > 150): 
            self.nodesize = 22 - (2000*len(correlationTable.data)*0.00001)% 3
        self.nodesize = 23
        self.i = 0
        self.setZValue(3)
        self.AlphaValue = 254


    def type(self):
        return Node.Type

    def addEdge(self, edge):
        self.edgeList.append(weakref.ref(edge))
        edge.adjust()

    def hoverLeaveEvent(self, event):
        pass
    
    def edges(self):
        return self.edgeList


    def SetCommunityColor(self):
        self.NodeCommunityColor = True

    def ResetCommunityColor(self):
        self.NodeCommunityColor = False

    def advance(self):
        if self.newPos == self.pos():
            return False

        self.setPos(self.newPos)
        return True

    def boundingRect(self):
        adjust = 2.0
        return QtCore.QRectF(-45 - adjust, -45 - adjust,
                             75 + adjust, 75 + adjust)

    def NodeColor(self):
        self.NodeCommunityColor = False
        self.update()

    def PutTextColor(self,colorvalue):
        self.TextColor = colorvalue
        self.CommunityTextColor = QtGui.QColor(colorvalue)
        self.update()

    def PutColor(self,colorvalue):
        self.colorvalue = colorvalue
        self.CommunityColor = QtGui.QColor(colorvalue)
        self.NodeCommunityColor = True
        self.update()

    def PutColorFromOtherClass(self,colorvalue, alphaValue = -1, AlphaDraw = False):
        pass

    def ChangeNodeSize(self,value):
        self.nodesize = 10 + value

    def setNodeSize(self,value,nodeSizeFactor,Rank,Zscore):
        pass

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(-10, -10, 20, 20)
        return path

    def itemChange(self, change, value):
        return QtGui.QGraphicsItem.itemChange(self, change, value)

    def Flush(self):
        pass

    """ Bitwise operation to change the alpha values"""
    @staticmethod
    def changeAlpha(Alpha,color):
        return ((int(Alpha)) | (16777215 & int(color)))

    def setOpaqueNodes(self): 
        pass

    def unsetOpaqueNodes(self): 
        pass

    def SelectedNode(self,value,FromWidget, count = 1):
        pass

    def getCommunity(self,value):
        return self.graph().partition[value]

    def getNodes(self,community):
        return self.graph().communityMultiple[community]

    def communitySelected(self,value,FromWidget, count = 1):
        pass

    def alledgesupdate(self):
        edges = self.graph().edges
        for edge in edges:
            edge().ColorEdgesFlag = False

    def selectCommunities(self):
        community = self.getNodes(self.counter-1)
        self.communitySelected(community[0],False,1)

