
import time
from PySide import QtCore, QtGui
import weakref
import pprint
import numpy as np
from Edge import Edge 

Scad = [87,102,105,107,121,123,134,154,185]
Bolo = [88,90,93,95,102,104,106,107,110,138,150]

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

    def itemChange(self, change, value):
        return QtGui.QGraphicsItem.itemChange(self, change, value)

