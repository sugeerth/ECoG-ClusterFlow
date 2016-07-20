
import csv
import numpy as np
import pprint
import weakref
import tempfile
import time
import math
from PySide import QtCore, QtGui
from collections import OrderedDict

import sys
from PySide.QtCore import *
import copy
from PySide.QtGui import *

import community as cm
try:
    # ... reading NIfTI
    import numpy as np
    # ... graph drawing
    import networkx as nx
except:
    print "Couldn't import all required packages. See README.md for a list of required packages and installation instructions."
    raise

Scad = [87,102,105,107,121,123,134,154,185]
Bolo = [88,90,93,95,102,104,106,107,110,138,150]

class ElectrodeNode(QtGui.QGraphicsItem):
    Type = QtGui.QGraphicsItem.UserType + 1


    def __init__(self, ImageLabel, counter, tooltip, contextFlag):
        QtGui.QGraphicsItem.__init__(self)
        self.minNodeVal = 7
        self.setAcceptHoverEvents(False)
        self.opacityValue = 255
        self.opacity = 255
        self.slices = 1
        self.ElectrodeData = ImageLabel
        self.ImageLabel = ImageLabel.ElectrodeData

        self.communityMemebership = []
        self.TimeStepRange = []
        self.chart = [True,False,False] 
        self.AlphaValue = []
        self.xy = None
        self.NodeCommunityColor = True
        self.Glyph = False
        self.graph = weakref.ref(self.ImageLabel.graphWidget)
        self.edgeList = []
        self.numberCalled= 0
        self.ColorQ = []
        self.SubNodesOfCommunityNodes = None
        self.node = []
        self.Red = False
        self.subNodes = []
        self.colorvalue = []
        self.communityNode = None
        self.nodesize = 12
        self.EA = None
        self.Highlight = False
        self.AcrossCommunityMode = False
        self.actualValue = []

        self.counter = counter
        self.Nodeidss = tooltip
        self.CommunityColor = []
        self.Abbr = self.ImageLabel.graphWidget.correlationTable().AbbrName
        self.Brain_Regions = self.graph().correlationTable().RegionName[0]

        self.newPos = QtCore.QPointF()
        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
        self.setCacheMode(self.DeviceCoordinateCache)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        self.setFlag(QtGui.QGraphicsItem.ItemUsesExtendedStyleOption)
        self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(1)

    def type(self):
        return DendoNode.Type

    def addEdge(self, edge):
        self.edgeList.append(weakref.ref(edge))
        edge.adjust()

    def edges(self):
        return self.edgeList

    """
    Setting the colors and the opacity from the guru ElectrodeView 
    Note this is done for all nodes of the available Electrode Views 
    """
    def PutColor(self,colorvalue, alphaValue = 255, AlphaDraw = False):
        self.colorvalue = colorvalue

        if self.Glyph:
            self.ColorQ.append(self.CommunityColor)
        self.CommunityColor = QtGui.QColor(colorvalue)

        # set the graph nodes to the transparent
        if alphaValue == None:
            self.CommunityColor.setAlpha(0) 
            self.opacity = 0
        elif not(alphaValue==-1):
            if AlphaDraw: 
                self.CommunityColor.setAlpha(alphaValue)
            else:
                self.CommunityColor.setAlpha(255)

        self.opacity = alphaValue

        self.NodeCommunityColor = True
        self.update()

    def PutFinalColors(self,opacity,actualValue, colorvalue,timesterange,\
        communityMemebership,slices= 4, alphaValue=255,\
         AllFourColors = None, AlphaDraw = False):
        self.slices = slices
        self.numberCalled += 1
        self.CommunityColor = QtGui.QColor(colorvalue)

        if opacity == None:
            opacity = 0

        self.colorvalue = colorvalue
        if self.Glyph:
            self.communityMemebership.append(communityMemebership)
            self.TimeStepRange.append(timesterange) 
            self.ColorQ.append(self.CommunityColor)
            self.actualValue.append(actualValue)
            self.opacity = opacity
            self.AlphaValue.append(opacity)
            counter= 0 
            new = []
            for value in self.TimeStepRange:
                new.append(value)

            self.setToolTip(str(self.AlphaValue)+"\n"+str(self.actualValue) + str(new)+ "\n"+communityMemebership)

            if len(self.ColorQ) > self.slices:
                self.AlphaValue = self.AlphaValue[-self.slices:]  
                self.ColorQ = self.ColorQ[-self.slices:]
                self.actualValue = self.actualValue[-self.slices:] 
                self.TimeStepRange = self.TimeStepRange[-self.slices:]
                self.communityMemebership = self.communityMemebership[-self.slices:]
                self.setToolTip(str(self.AlphaValue)+"\n:" + str(self.TimeStepRange)+ "\n")

        self.NodeCommunityColor = True
        self.update()

    def boundingRect(self):
        adjust = 2.0
        return QtCore.QRectF(-45 - adjust, -45 - adjust,
                             75 + adjust, 75 + adjust)

    def shape(self):
        path = QtGui.QPainterPath()
        path.addEllipse(-10, -10, 20, 20)
        return path

    def setGlyph(self, state):
        if state:
            self.Glyph = True
        else: 
            pass

    def ScalarNodeSize(self,value):
        self.nodesize *= value

    def setNodeSize(self,value,nodeSizeFactor):
        self.degreeCentrality = float(value)
        self.nodesize = self.minNodeVal + 1 * value

    def PutEA(self,EA):
        self.EA= EA
        self.update()

    def contextMenuEvent(self, event):
        menu = QtGui.QMenu()
        testAction = QtGui.QAction('Select This Community Over Time', None)
        CommunityAction= QtGui.QAction('Select This Community', None)
        testAction.triggered.connect(self.SelectCommunitiesOverTime)
        CommunityAction.triggered.connect(self.SelectingCommunityInThisTimestep)
        menu.addAction(testAction)
        menu.addAction(CommunityAction)
        menu.exec_(event.screenPos())

    def SelectCommunitiesOverTime(self):
        """
        Highlight everywhere the color that is selected here???
        """
        # color of the communitycolor, communitynumber, timestep range,source to destination
        timesteprange = [0,1,2,3] # data from electrodeview
        self.ElectrodeData.CommunitySelectAcrossTime.emit(self.ColorQ[0], self.ColorQ,self.ElectrodeData.ChunkNo)

    def SelectingCommunityInThisTimestep(self):
        # color of the communitycolor, communitynumber, timestep range,source to destination
        timesteprange = [0,1,2,3] # data from electrodeview
        self.ElectrodeData.CommunitySelectPerTime.emit(self.ColorQ[0], self.ColorQ,self.ElectrodeData.ChunkNo)

    def paint(self, painter, option, widget):
        painter.setPen(QtCore.Qt.NoPen)
        self.radius = self.nodesize

        if self.AlphaValue < self.ImageLabel.opacityThreshold: 
            return

        if self.Glyph:
            if self.ImageLabel.GlyphUnit == 0: #normal Glyphs
                painter.setBrush(self.CommunityColor)
                rectangle = QtCore.QRectF(0, 0, (self.radius+12), (self.radius+12))
                startAngle = 0
                angle = float(360/self.slices)
                spanAngle = angle
                j = -1
                i = 0
                startAngle = 270

                if not(self.AcrossCommunityMode): 

                    if self.slices==1: 
                        painter.setPen(QtGui.QPen(self.CommunityColor, 0.1))
                        radius = (float(self.opacity/255)*18)
                        self.drawOnePie(painter, 255, radius)
                    else:
                        for row in range(self.slices):
                            angle = 360*1/self.slices
                            radius = (float(self.AlphaValue[j]/255)*18)
                           
                            rectangle = QtCore.QRectF( -radius, -radius, radius*2, radius*2);
                            color = copy.deepcopy(self.ColorQ[j])
                            painter.setBrush(color)

                            painter.setPen(QtGui.QPen(color, 0))

                            self.drawRoundGlyphs(row,startAngle,angle,painter,rectangle)

                            startAngle += angle
                            startAngle = startAngle % 360
                            j = j+1
                else:
                    if not(self.Highlight):
                        if self.slices==1: 
                            painter.setPen(QtGui.QPen(self.CommunityColor, 0.1))
                            self.drawOnePie(painter, 170)
                        else:
                            for row in range(self.slices):
                                angle = 360*1/self.slices

                                radius = (float(self.AlphaValue[j]/255)*18)
                                rectangle = QtCore.QRectF( -radius, -radius, radius*2, radius*2);
                                color = copy.deepcopy(self.ColorQ[j])

                                NewColor = copy.deepcopy(self.ColorQ[j])
                                painter.setBrush(NewColor)

                                painter.setPen(QtGui.QPen(NewColor, 0))

                                self.drawRoundGlyphs(row,startAngle,angle,painter,rectangle)

                                startAngle += angle
                                startAngle = startAngle % 360
                                j = j+1
                    else: 
                        if self.slices==1: 
                            painter.setPen(QtGui.QPen(QtCore.Qt.blue, 2))
                            radius = (float(self.opacity/255)*15)
                            self.drawOnePie(painter, 255, radius)
                        else:
                            for row in range(self.slices):

                                angle = 360*1/self.slices

                                radius = (float(self.AlphaValue[j]/255)*15)
                                rectangle = QtCore.QRectF( -radius, -radius, radius*2, radius*2);
                                color = copy.deepcopy(self.ColorQ[j])
                                painter.setBrush(color)
                                painter.setPen(QtGui.QPen(color, 0))


                                self.drawRoundGlyphs(row,startAngle,angle,painter,rectangle)

                                startAngle += angle
                                startAngle = startAngle % 360
                                j = j+1
            elif self.ImageLabel.GlyphUnit == 1: #opacity Glyphs 
                painter.setBrush(self.CommunityColor)
                startAngle = 0
                angle = float(360/self.slices)
                spanAngle = angle
                j = 0
                i = 0
                startAngle = 270

                if not(self.AcrossCommunityMode): 
                    if self.slices==1: 
                        painter.setPen(QtGui.QPen(self.CommunityColor, 0))
                        self.drawOnePie(painter,self.opacity)
                    else:
                        for row in range(self.slices):
                            angle = 360*1/self.slices

                            radius = 11
                            rectangle = QtCore.QRectF( -radius, -radius, radius*2, radius*2);

                            color = copy.deepcopy(self.ColorQ[j])
                            color.setAlpha(self.AlphaValue[j]) 

                            painter.setPen(QtGui.QPen(color, 0.1))
                            
                            painter.setBrush(color)

                            self.drawRoundGlyphs(row,startAngle,angle,painter,rectangle)

                            startAngle += angle
                            startAngle = startAngle % 360
                            j = j+1
                else:
                    if not(self.Highlight):
                        if self.slices==1: 
                            painter.setPen(QtGui.QPen(QtCore.Qt.white, 0.1))
                            self.drawOnePie(painter, 170)
                        else:
                            for row in range(self.slices):
                                angle = 360*1/self.slices

                                radius = 11
                                rectangle = QtCore.QRectF( -radius, -radius, radius*2, radius*2);
                                color = copy.deepcopy(self.ColorQ[j])
                                color.setAlpha(self.normalize(self.AlphaValue[j])) 

                                NewColor = copy.deepcopy(self.ColorQ[j])
                                NewColor.setAlpha(170)
                                painter.setBrush(NewColor)

                                painter.setPen(QtGui.QPen(NewColor, 0.1))

                                self.drawRoundGlyphs(row,startAngle,angle,painter,rectangle)

                                startAngle += angle
                                startAngle = startAngle % 360
                            j = j+1
                    else: 

                        if self.slices==1: 
                            painter.setPen(QtGui.QPen(QtCore.Qt.blue, 2))
                            self.drawOnePie(painter, (float(self.opacity/255)*20))
                        else:
                            for row in range(self.slices):
                                angle = 360*1/self.slices
                                radius = 11
                                rectangle = QtCore.QRectF( -radius, -radius, radius*2, radius*2);

                                color = copy.deepcopy(self.ColorQ[j])
                                color.setAlpha(self.normalize(self.AlphaValue[j])) 

                                painter.setBrush(color)
                                painter.setPen(QtGui.QPen(color, 0.1))

                                self.drawRoundGlyphs(row,startAngle,angle,painter,rectangle)

                                startAngle += angle
                                startAngle = startAngle % 360
                                j = j+1
            elif self.ImageLabel.GlyphUnit == 2:
                    self.drawBarChart(painter,self.Highlight)
        else:
            self.CommunityColor.setAlpha(self.opacity)
            if not(self.ImageLabel.graphWidget.ColorNodesBasedOnCorrelation):
                if self.CommunityColor: 
                    painter.setBrush(self.CommunityColor)
                    painter.drawEllipse(QtCore.QPointF(0,0),self.nodesize,self.nodesize)
                else: 
                    painter.setBrush(self.CommunityColor)
                    painter.setPen(QtGui.QPen(QtCore.Qt.transparent, 0))
                    painter.drawEllipse(0,0,14,14)
            else: 
                    nodeColor = QtGui.QColor(self.ImageLabel.graphWidget.DataColor[self.counter+1])
                    painter.setBrush(nodeColor)
                    painter.drawEllipse(QtCore.QPointF(0,0),self.nodesize,self.nodesize)
            if option.state & QtGui.QStyle.State_Selected:
                circle_path = QtGui.QPainterPath()
                painter.setPen(QtGui.QPen(QtCore.Qt.blue, 2))        
                circle_path.addEllipse(QtCore.QPointF(0,0),self.nodesize+1,self.nodesize+1);
                painter.drawPath(circle_path)

    def drawRoundGlyphs(self, row,startAngle, angle, painter, rectangle):
        if row == self.slices-1:
            if not(startAngle + angle == 270):
                delta = 270 - (startAngle + angle) 
                painter.drawPie(rectangle,int(startAngle*16) * -1, int((angle+delta)*16) * -1)
            else:
                painter.drawPie(rectangle,(int(startAngle*16) * -1), (int(angle*16) * -1))
        else:
            painter.drawPie(rectangle,(int(startAngle*16) * -1), (int(angle*16)*-1))


    def drawBarChart(self, painter, Highlight): 
        radiusOld = 10

        width = float(radiusOld*2/self.slices)
        assert self.slices == len(self.ColorQ)
        setWidth = 0 
        j = -1 
        for i in range(self.slices): 
            if not(self.Highlight):
                painter.setPen(QtGui.QPen(QtCore.Qt.black, 0.1)) 
            else: 
                painter.setPen(QtGui.QPen(QtCore.Qt.blue, 1))  

            painter.setBrush(self.ColorQ[j])
            radius = (float(self.AlphaValue[j]/255)*20)
            rectangle = QtCore.QRectF(10-setWidth, 10, width, -radius);
            painter.drawRect(rectangle)
            setWidth += width 
            j = j-1

    def normalize(self, value):
        if value > 255:
            return 255
        elif value < 0: 
            return 0
        else: 
            return value

    def drawOnePie(self,painter, Opacity=255, radius = 11):
        radius = int(radius)
        self.CommunityColor.setAlpha(Opacity)
        painter.setPen(QtGui.QPen(self.CommunityColor, 0.1))

        painter.setBrush(self.CommunityColor)
        painter.drawEllipse(QtCore.QPointF(0,0),radius, radius)

    def makeOpaqueCommunityNodes(self,communityNode):
        for node1 in communityNode:
            node1.CommunityColor.setAlpha(255)
            node1.setSelected(False)
            node1.update()
        
        edges1 = self.ImageLabel.graphWidget.communityObject.edges

        for edge in edges1:
            edge.communityAlpha(True)
            edge.update()

    def reset():
        self.opacity = 255

    def electrodeDistance(self, length, height, xy, radius):
        mylength, myheight = xy 
        a= abs(length - mylength)
        b=abs(myheight - height)
        ratio = math.sqrt(a*a+b*b)
        return ratio

    def computeZoomLevelCluster(self, slice, xy, radius):

        def opacityCare(value):
            if value > 255: 
                value = 255
            elif value < 0: 
                value = 0
            return value

        # Assertion error with community multiple

        membershipNo = self.ImageLabel.graphWidget.communityDetectionEngine.FinalClusterPartition[self.counter]
        clusterIds = self.ImageLabel.CommunitiesAcrossTimeStep.communityMultiple[membershipNo]
        ratio = np.zeros((len(clusterIds),int(len(self.ImageLabel.graphWidget.communityDetectionEngine.FinalClusterPartition))))
        # see if the community across timestep is consistant
        assert len(self.ImageLabel.CommunitiesAcrossTimeStep.communityMultiple.keys()) == len(set(self.ImageLabel.graphWidget.communityDetectionEngine.FinalClusterPartition.values()))

        i= 0
        for node in self.ElectrodeData.NodeIds:
            if not(node.counter in clusterIds):
                x,y = node.xy
                node.setSelected(False)
                node.nodesize = 2
                node.radius = 2
                node.opacity = 255
                try: 
                    k=0
                    for j in clusterIds:
                        ratio[k][node.counter] = self.electrodeDistance(x,y,self.ElectrodeData.NodeIds[j].xy,radius)
                        k=k+1
                except ZeroDivisionError:
                    continue
                i=i+1
            else: 
                node.nodesize = 1.4
                node.radius = 1.4
                node.opacity = 40
                assert (node.counter in clusterIds)
                for j in range(len(clusterIds)):
                    ratio[j][node.counter] = 0

        for k in range(len(clusterIds)):
            minV = min(ratio[k])
            maxV = max(ratio[k])
            rangeV = maxV-minV

            for i,j in enumerate(ratio[k]):
                    ratio1 = (j-minV)/rangeV
                    if ratio1 < 0.3 and ratio1 > 0:
                        try:
                            value = opacityCare((1/ratio1)* 20)
                            self.ElectrodeData.NodeIds[i].opacity = 255
                        except ZeroDivisionError:
                            self.ElectrodeData.NodeIds[i].nodesize=16.3 * 1.4
                            self.ElectrodeData.NodeIds[i].radius=16.3 * 1.4
                            self.ElectrodeData.NodeIds[i].opacity = 255
                        node.update()
                    if ratio1 == 0:
                        assert i in clusterIds
                        self.ElectrodeData.NodeIds[i].nodesize= 16.3 * 1.4
                        self.ElectrodeData.NodeIds[i].radius=16.3 * 1.4
                        self.ElectrodeData.NodeIds[i].opacity = 255
                        node.update()

    def clustervalue(self, i, clusterIds):
        if i in clusterIds:
            i = i + 1
            self.clustervalue(i, clusterIds)
        else: 
            return i

    def computeZoomLevel(self, slice, xy, radius):

        def opacityCare(value):
            if value > 255: 
                value = 255
            elif value < 0: 
                value = 0
            return value

        ratio = dict()
        i= 0
        for node in self.ElectrodeData.NodeIds: 
            x,y = node.xy
            node.nodesize = 0.2
            node.radius = 0.2
            node.opacity = 255
            try: 
                ratio[i] = self.electrodeDistance(x,y,xy,radius)
            except ZeroDivisionError:
                continue
            i=i+1

        minV = min(ratio.values())
        maxV = max(ratio.values())

        rangeV = maxV-minV

        for i,j in ratio.items():
            ratio1 = (j-minV)/rangeV
            if ratio1 < 0.7:
                try:
                    self.ElectrodeData.NodeIds[i].nodesize= (1/ratio1)* 2 
                    self.ElectrodeData.NodeIds[i].radius = (1/ratio1)* 2
                    value = opacityCare((1/ratio1)* 30)
                    self.ElectrodeData.NodeIds[i].opacity = 255
                except ZeroDivisionError:
                    self.ElectrodeData.NodeIds[i].nodesize= 16.3 * 1.4
                    self.ElectrodeData.NodeIds[i].radius= 16.3 * 1.4
                    self.ElectrodeData.NodeIds[i].opacity = 255
                node.update()
   
    def mousePressEvent(self, event):
        self.ElectrodeData.Refresh()
        radius = 1

        self.ImageLabel.NodeSelected.emit(self.counter)
        QtGui.QGraphicsItem.mousePressEvent(self, event)