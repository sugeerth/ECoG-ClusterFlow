
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
        
        # self.Abbr = correlationTable.AbbrName
        self.Brain_Regions = correlationTable.RegionName[0]
        
        try: 
            for i in range (self.counter-1):
                self.Brain_Regions[i] = self.Brain_Regions[i].replace(' ','')
        except IndexError:
            print "Exception", i

        if not(self.ForCommunities): 
            pass
            # self.setToolTip(str(self.Abbr[self.counter -1][0]+ " , " + str(self.counter)))
        else: 
            self.setToolTip(str(counter))
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
        # self.colorvalue = colorvalue
        # self.CommunityColor = QtGui.QColor(colorvalue)
        # self.AlphaValue = alphaValue
        # # set the graph nodes to the transparent
        # if not(alphaValue==-1):
        #     if AlphaDraw: 
        #         self.CommunityColor.setAlpha(alphaValue)
        #     else:
        #         self.CommunityColor.setAlpha(255)
        # self.NodeCommunityColor = True
        # self.update()

    def ChangeNodeSize(self,value):
        self.nodesize = 10 + value
        # self.update()

    def setNodeSize(self,value,nodeSizeFactor,Rank,Zscore):
        pass
        # self.degreeCentrality = float(value)
        # self.setToolTip(str("\n"+nodeSizeFactor+":" + "{0:.2f}".format(self.degreeCentrality) + "\nRank:" + str(Rank) + "\nZ-score:" + "{0:.2f}".format(Zscore)))
        # self.nodesize = 7 + 15 * value
        # self.nodesize = 17

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(-10, -10, 20, 20)
        return path

    def paint(self, painter, option, widget):
        pass
 #        painter.setPen(QtCore.Qt.NoPen)
 #        self.font =  painter.font()
 #        self.font.setBold(False)
 #        fm = QtGui.QFontMetrics(painter.font())
 #        wi=fm.maxWidth()
 #        ht=fm.height()

 #        if not(self.ForCommunities):
 #            mess = self.Translate.set(str(self.Brain_Regions[self.counter-1]))
	#     self.font.setPointSize(11)
 #            rect= fm.boundingRect(mess+str(22))
 #            painter.setFont(self.font)
 #            dx = -rect.width()/2 + rect.width()/5 + 3
 #            dy = rect.height()/2- 0.5
 #            # self.nodesize = 16
 #            rect.translate(dx,dy)
	# else: 
 #            mess = self.Translate.set(str(self.counter))
 #            self.font.setPointSize(11)
 #            rect= fm.boundingRect(mess+str(25))
 #            painter.setFont(self.font)
 #            dx = -rect.width()/2 + rect.width()/5 +1
 #            dy = rect.height()/2 - 2.5 -1
 #            self.nodesize = 17
 #            rect.translate(dx,dy)

 #        # from BrainViewer import DataColor
 #        if self.NodeCommunityColor:
 #            if option.state & QtGui.QStyle.State_Selected:
 #                painter.save()
 #                self.Selected = True 
 #                self.WhitePaint = True
 #                self.CommunityColor.setAlpha(255)
 #                painter.setBrush(self.CommunityColor)
 #            else:
 #                painter.save()
 #                self.Selected = False
 #                self.WhitePaint = False
 #                painter.setBrush(self.CommunityColor)

 #        else: 
 #            if (option.state & QtGui.QStyle.State_Selected):
 #                painter.save()
 #                self.Selected = True
 #                self.WhitePaint = True
 #                #FIXME Green 
 #                painter.setBrush(QtCore.Qt.darkGreen)
 #                # painter.setBrush(self.NodeColor)
 #            else:
 #                painter.save()
 #                self.Selected = False
 #                self.WhitePaint = False
 #                painter.setBrush(self.nodeColor)

 #        painter.setPen(QtGui.QPen(QtCore.Qt.black, 0))        
 #        painter.drawEllipse(QtCore.QPointF(0,0),self.nodesize,self.nodesize-2)
        
 #        if self.NodeCommunityColor:
 #            painter.setPen(self.CommunityTextColor)
 #        else:
 #            painter.setPen(QtGui.QPen(QtCore.Qt.black, 0))        

 #        if self.WhitePaint:
 #                circle_path = QtGui.QPainterPath()
 #                painter.setPen(QtGui.QPen(QtCore.Qt.blue, 2))        
 #                circle_path.addEllipse(QtCore.QPointF(0,0),self.nodesize,self.nodesize-2);
 #                painter.drawPath(circle_path)
 #                painter.drawText(rect,QtCore.Qt.TextSingleLine,mess)
 #        else:
 #                painter.drawText(rect,QtCore.Qt.TextSingleLine,mess)
 #        painter.restore()

        # if self.NodeCommunityColor:
        #     painter.setPen(self.CommunityTextColor)
        # else:
        #     painter.setPen(QtGui.QPen(QtCore.Qt.black, 0))        

        # if self.Red:    
        #     # Color the stuff as Red
        #     painter.save()
        #     red = QtGui.QColor(255,0,0,self.AlphaValue)
        #     painter.setBrush(red)
        #     self.WhitePaint = True
        #     painter.setPen(QtGui.QPen(QtCore.Qt.red, 0))        
        #     painter.drawEllipse(QtCore.QPointF(0,0),self.nodesize,self.nodesize-2)
        # else: 
        #     painter.save()
        #     red = QtGui.QColor(0,0,0,self.AlphaValue)
        #     painter.setBrush(red)
        #     self.WhitePaint = True
        #     painter.setPen(QtGui.QPen(QtCore.Qt.black, 0))        
        #     painter.drawEllipse(QtCore.QPointF(0,0),self.nodesize,self.nodesize-2)

    def itemChange(self, change, value):
        return QtGui.QGraphicsItem.itemChange(self, change, value)

    def Flush(self):
        pass
        # edges = [item for item in self.scene().items() if isinstance(item, Edge)]
        # for edge in edges:
        #     edge.ColorEdgesFlag = False

    """ Bitwise operation to change the alpha values"""
    @staticmethod
    def changeAlpha(Alpha,color):
        return ((int(Alpha)) | (16777215 & int(color)))

    def setOpaqueNodes(self): 
        pass
        # self.colorTransparency = False
        # self.nodeColor = QtGui.QColor(self.graph().DataColor[self.counter])
        # if self.NodeCommunityColor:
        #     self.CommunityColor.setAlpha(255)
        # else: 
        #     self.nodeColor.setAlpha(255)
        # self.update()

    def unsetOpaqueNodes(self): 
        pass
        # self.colorTransparency = True
        # self.nodeColor = QtGui.QColor(self.graph().DataColor[self.counter])
        # if not(self.setTransp):
        #     return

        # if self.NodeCommunityColor:
        #     self.CommunityColor.setAlpha(25)
        # else:

        #     self.nodeColor.setAlpha(25)
        # self.update()

    def SelectedNode(self,value,FromWidget, count = 1):
        pass
        # self.Flush()
        # self.graph().Graph_Color(value+1)
        # edges = [item for item in self.scene().items() if isinstance(item, Edge)]
        # """
        # Idea is to color the nodes that are allowed by the endges 
        # which will be 
        # Toggle between the ideas 
        # """
        # self.allnodesupdate()

        # for edge in edges:
        #     sliderBool = self.graph().EdgeSliderValue < edge.weight
        #     sourceBool = edge.sourceNode() is self
        #     destBool = edge.destNode() is self

        #     if (sourceBool or destBool) and sliderBool:
        #         edge.ColorEdges() 
        #         if sourceBool: 
        #             edge.destNode().setOpaqueNodes()
        #         else: 
        #             edge.sourceNode().setOpaqueNodes() 
        # if not(FromWidget):
        #     self.graph().regionSelected.emit(value)
        #     self.graph().Refresh()

    def getCommunity(self,value):
        return self.graph().partition[value]

    def getNodes(self,community):
        return self.graph().communityMultiple[community]

    def communitySelected(self,value,FromWidget, count = 1):
        pass
        # self.Flush()
        # # self.graph().Graph_Color(value+1)
        # selectedCommunity = self.getCommunity(value)
        # nodesInCommunity = self.getNodes(selectedCommunity)

        # # print "nods in community:",nodesInCommunity,"\n selected community:", selectedCommunity
        # # edges = [item for item in self.scene().items() if isinstance(item, Edge)]

        # edges = self.graph().edges
        # """
        # Idea is to color the nodes that are allowed by the endges 
        # which will be 
        # Toggle between the ideas 
        # """
        
        # self.allnodesupdate()
        # if self.ForCommunities:
        #     self.alledgesupdate()
        # # Updating the communities 
        # #Inefficient
        # for k,v in self.graph().partition.iteritems():
        #         if v == selectedCommunity:
        #             self.graph().NodeIds[k].setOpaqueNodes()

        # # Coloring the edges
        # for edge in edges:
        #     sliderBool = self.graph().EdgeSliderValue < edge().weight
        #     sourceBool = edge().sourceNode().counter-1 in nodesInCommunity
        #     destBool = edge().destNode().counter-1 in nodesInCommunity
        #     if (sourceBool and destBool) and sliderBool:
        #         edge().ColorEdges() 


        # self.graph().Refresh()

        # if not(FromWidget):
        #     self.graph().regionSelected.emit(value)
        #     self.graph().Refresh()

    def allnodesupdate(self):
        pass
        # # Nodes = [item for item in self.scene().items() if isinstance(item, Node)]
        # Nodes = self.graph().nodes
        # for node in Nodes:
        #     node().setSelected(False)
        #     node().unsetOpaqueNodes()
        #     node().WhitePaint = False
        #     node().update()

    def alledgesupdate(self):
        edges = self.graph().edges
        for edge in edges:
            edge().ColorEdgesFlag = False

    def selectCommunities(self):
        community = self.getNodes(self.counter-1)
        self.communitySelected(community[0],False,1)

    def associatedNodes(self):
        pass

    # def hoverEnterEvent(self, event):

    #     if self.ForCommunities:
    #         self.selectCommunities()
    #         self.update()
    #         return
        
    #     if not(self.NodeCommunityColor):            
    #         self.SelectedNode(self.counter-1,False, 1)
    #         self.setSelected(True)
    #     else: 
    #         self.associatedNodes()
    #         self.communitySelected(self.counter-1, False, 1)

    #     # If the brain regions are in community mode then quit 
    #     QtGui.QGraphicsItem.hoverEnterEvent(self, event)

    # def mousePressEvent(self, event):
    #     if self.ForCommunities:
    #         self.unsetOpaqueNodes()
    #         self.SelectedNode(self.counter-1,False, 1)
    #         # self.selectCommunities()
    #         self.update()
    #         return

    #     # Hack to  update the edges of the nodes 
    #     # if not(self.NodeCommunityColor):
    #     self.SelectedNode(self.counter-1,False, 1)
    #     # else: 
    #     #     self.communitySelected(self.counter-1,False,1)

    #     # If the brain regions are in community mode then quit 
    #     QtGui.QGraphicsItem.mousePressEvent(self, event)
