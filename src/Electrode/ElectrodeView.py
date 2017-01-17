### Standard Python packages
#-*- coding: utf-8 -*-
import csv
import colorsys
import math
import colorsys
import time
from collections import defaultdict
from PySide import QtCore, QtGui
from PySide.QtCore import *
from PySide.QtGui import *

import warnings
warnings.filterwarnings("ignore")

from sys import platform as _platform
import weakref
import cProfile
import pprint
# import time
import community as cm
# ... reading NIfTI 
import numpy as np
# ... graph drawing
import networkx as nx

# Changes for RealData

ElectrodeSignalDataName = 'sigData'
# ElectrodeSignalDataName = 'muDat'

timesteps = 12
NumberofTimesteps = 85

#Max and Min are either computed per channel 
from ElectrodeNode import ElectrodeNode

MaxVal = -99999
MinVal = 999999
minOpac = 40
maxOpac = 255
MaxToggling = True

class GraphicsBalloonTextItem(QtGui.QGraphicsTextItem):
    """
    This class draws a nice text balloon. The balloon can be
    pointing above or below depending upon the orientation attribute.
    """
    def __init__(self, parent, orientation='above'):
        super(GraphicsBalloonTextItem, self).__init__(parent)
        self.orientation = orientation

    def boundingRect(self):
        """
        Returns the new bounding rect which makes room for the
        balloon label and arrow.
        """
        rect = super(GraphicsBalloonTextItem, self).boundingRect()
        if self.orientation == 'above':
            # balloon above the point
            return rect.adjusted(-1, 0, 1, 12)
        else:
            # balloon below the point
            return rect.adjusted(-1, -12, 1, 0)

    def paint(self, painter, option, widget):
        """
        This method does the drawing.
        """
        painter.setPen(QtCore.Qt.darkGray)
        painter.setBrush(QtGui.QColor(250, 245, 209))
        
        adjustedRect = self.boundingRect() # the rectangle around the text

        if self.orientation == 'above':
            # should draw the label balloon above the point
            adjustedRect.adjust(0, 0, 0, -12)
            vertices = [QtCore.QPointF(adjustedRect.width()/2 - 6,
                                       adjustedRect.height() - 1),
                        QtCore.QPointF(adjustedRect.width()/2,
                                       adjustedRect.height() + 12),
                        QtCore.QPointF(adjustedRect.width()/2 + 6,
                                       adjustedRect.height() - 1)]
        else:
            # should draw the label balloon below the point
            adjustedRect.adjust(0, 12, 0, 0)
            vertices = [QtCore.QPointF(adjustedRect.width()/2 - 6, 1),
                        QtCore.QPointF(adjustedRect.width()/2, -12),
                        QtCore.QPointF(adjustedRect.width()/2 + 6, 1)]

        # paint the balloon rectangle
        painter.drawRoundedRect(adjustedRect, 8, 8)

        # paint the balloon arrow triangle fill
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawPolygon(vertices)

        # paint the balloon arrow triangle stroke
        painter.setPen(QtCore.Qt.darkGray)

        painter.drawLine(vertices[0], vertices[1])
        painter.drawLine(vertices[2], vertices[1])

        # Finally call the parent paint method to draw the actual text
        super(GraphicsBalloonTextItem, self).paint(painter, option, widget)

class Translate(QtCore.QObject):
    def __init__(self):
        QtCore.QObject.__init__(self)
    def set(self,string):
        return  str(self.tr(string))

class ElectrodeOpacity(object):
        def __init__(self, ElectrodView, ElectrodeNumber, counter): 
            self.ElectrodeSignalData = ElectrodView.ElectrodeData.ElectrodeSignal[ElectrodeSignalDataName]
            self.ElectrodView = ElectrodView
            self.ElectrodeNumber = ElectrodeNumber
            self.counter = counter
            self.value = 0
            self.TimesVisited =0 
            self.DefMinValMaxVal(ElectrodeNumber)

        """Defines max and min values across all timesteps"""
        def DefMinValMaxVal(self, ElectrodeNo):
            global MaxVal, MinVal
            MaxVal = np.nanmax(self.ElectrodeSignalData)
            MinVal = np.nanmin(self.ElectrodeSignalData)
            # MinVal*=100
            # MaxVal*=100

        def RegenerateMaxMin(self,timeStep):
            global MaxVal, MinVal
            CurrentSyllable = self.ElectrodView.ElectrodeData.syllableUnit
            if not(self.ElectrodView.MaxButtonCheck):
                list1 = [] 
                for i in range(256):
                    list1.append(self.ElectrodeSignalData[CurrentSyllable][i][timeStep])
                MaxVal = max(list1)
                MinVal = min(list1)
            

        """
        1) Given the values of a electrode finds out the normalized min max values for every 
        along every timestep
        2) Normalization is performed for at every timestep, can be precomputed, later 
        """
        def normalize(self, timestep, ElementNumber):
            """
            I HAVE NO IDEA WHAT THIS FUNCTION DOES 
            """
            global MaxVal, MinVal
            global minOpac, maxOpac 

            ElementNumber = int(ElementNumber)

            # Synthetic Dataset
            # Number= np.nonzero(self.ElectrodView.ElectrodeData.ElectrodeIds == ElementNumber)[0][0]

            # Real Dataset
            Number = ElementNumber
            
            assert self.ElectrodeNumber == ElementNumber
            self.TimesVisited = self.TimesVisited + 1
            "Given the timestep the function computes opacity per channel based on all of the timesteps"

            CurrentSyllable = self.ElectrodView.ElectrodeData.syllableUnit
            actualValue = self.ElectrodeSignalData[CurrentSyllable][Number][timestep]
            
            # print Number, timestep, actualValue

            x = actualValue

            if x >= MaxVal: 
                x = MaxVal
            if x < MinVal-0.01:
                x = None

            if x or x == 0: 
                # Multiplying by 10 to identify structure
                PercentDistrbution = float((x-MinVal)/(MaxVal-MinVal)) * 10
                OpacityValue = float(PercentDistrbution)*(maxOpac-minOpac) + minOpac
            else: 
                assert (x == None and OpacityValue == None)

            if OpacityValue > 255:
                    OpacityValue = 255
            if OpacityValue < 0: 
                    OpacityValue = 0

            self.value = x
            return OpacityValue, actualValue


class ElectrodeView(QtGui.QGraphicsView):
    DataLink = QtCore.Signal(int)
    # color of the communitycolor, communitynumber, timestep range,source to destination÷
    CommunitySelectPerTime = QtCore.Signal(list, list ,  int)
    EmitSelectedElectrodeView = QtCore.Signal(int, int, int)

    # color of the communitycolor, communitynumber, timestep range,source to destination÷
    CommunitySelectAcrossTime = QtCore.Signal(list, list ,int)

    def __init__(self,ElectrodeData, i = None, width = None, height = None):
        QtGui.QGraphicsView.__init__(self)
        global MaxVal, MinVal, timesteps
        self.smallMultiple = False
        if i: 
            self.smallMultiple = True
        if width and height:
            self.width  =1301
            self.height =861
            self.setMinimumSize(QtCore.QSize(275,275))
        else: 
            self.width  =1301
            self.height =861
            self.setMinimumSize(QtCore.QSize(275,275))

        self.Translate = Translate()
        timesteps = ElectrodeData.dataProcess.timestep
        # self.setAcceptHoverEvents(True)

        self.ChunkNo = i
        self.ElectrodeData = ElectrodeData
        self.NodeIds = []
        self.slices = self.ElectrodeData.slices
        scene = QtGui.QGraphicsScene(self)
        scene.setItemIndexMethod(QtGui.QGraphicsScene.NoIndex)

        self.setRubberBandSelectionMode(QtCore.Qt.IntersectsItemShape) 

        # scene.setSceneRect(-200, -200, 400, 400)

        self.transparent= False
        self.setScene(scene)
        self.scene = scene
        self.setCacheMode(QtGui.QGraphicsView.CacheBackground)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtGui.QGraphicsView.AnchorViewCenter)
        self.setInteractive(True)
        self.setTransformationAnchor(QtGui.QGraphicsView.AnchorUnderMouse)
        self.scaleView(0.0)
        self.ElectrodeOpacity = []
        self.setScene(scene)
        self.MaxButtonCheck = True
        self.Cluster1 = False
        
        self.Scene_to_be_updated = scene
        self.setCacheMode(QtGui.QGraphicsView.CacheBackground)
        self.setViewportUpdateMode(QtGui.QGraphicsView.BoundingRectViewportUpdate)
        self.setDragMode(QtGui.QGraphicsView.ScrollHandDrag)

        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setInteractive(True)
        
        self.PaintElectrodes()

        pixItem = QtGui.QGraphicsPixmapItem(self.ElectrodeData.PixMap)
        self.scene.addItem(pixItem)    
        if self.ChunkNo is None:
            self.ChunkNo = 1
        if self.slices is None:
            self.slices = 1
        self.ballon = GraphicsBalloonTextItem(str(str(self.ChunkNo*self.slices)+"-"+str((self.ChunkNo+1)*self.slices)))
        self.ballon.setPos(550,300)
        self.scene.addItem(self.ballon)

        self.setSceneRect(self.Scene_to_be_updated.itemsBoundingRect())
        self.setScene(self.Scene_to_be_updated)

        rect =QtCore.QRectF(self.width/2+115, self.height/4+100, 3*self.width/6-60,3*self.height/5+30)
        rect.translate(-470,-180)

        self.fitInView(rect,QtCore.Qt.KeepAspectRatio)

        self.MaxVal = MaxVal
        self.MinVal = MinVal

        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        delete = QAction(self)
        delete.setText("Select Community")
        delete.triggered.connect(self.removeButton)
        self.addAction(delete)


    def changeSliceNumber(self,value = -1):
        number = int((self.ElectrodeData.Visualizer.Slices.text().encode('ascii','ignore')).replace(' ',''))
        self.slices = number
        self.scene.removeItem(self.ballon)
        if self.ChunkNo is None:
            self.ChunkNo = 1
        if self.slices is None:
            self.slices = 1
        if self.slices == 1:
            self.ballon = GraphicsBalloonTextItem(str(self.ChunkNo))
        else:
            self.ballon = GraphicsBalloonTextItem(str(str(self.ChunkNo*self.slices)+"-"+str((self.ChunkNo+1)*self.slices)))
        self.ballon.setPos(550,300)
        self.scene.addItem(self.ballon)
        self.UpdateColors()
        self.RefreshInteractivityData()

    def drawBackground(self,painter,rect):
        pass

    def changeMaxOpacity(self,value):
        global MaxVal,MinVal
        MaxVal = value*self.MaxVal*0.01

        self.ElectrodeData.Visualizer.Max.setText("{0:.2f}".format(MaxVal))
        self.ElectrodeData.Visualizer.Min.setText("{0:.2f}".format(MinVal))

        self.UpdateColors()
        self.RefreshInteractivityData()

    def HighlightThisColor(self, CommunityColor,ColorQ):
        for node in self.NodeIds:
            node.AcrossCommunityMode =True
            node.Highlight = False

            if node.ColorQ == ColorQ: 
                node.Highlight = True
            node.update()
        # print "* Done"

    def HighlightAcrossTime(self, CommunityColor, ColorQ):
        for node in self.NodeIds:
            node.AcrossCommunityMode =True
            node.Highlight = False
            if node.ColorQ == ColorQ: 
                node.Highlight = True
            node.update()
        # print "* Done"

    def RefreshInteractivityData(self): 
        for node in self.NodeIds:
            node.AcrossCommunityMode=False
            node.Highlight=False
            node.update()
        # print "* Done"

    def PaintElectrodes(self):
        counter = 0
        k = 0

        for x,y in zip(self.ElectrodeData.mat['xy'][0],self.ElectrodeData.mat['xy'][1]):
            if self.ElectrodeData.graphWidget.CommunityMode:
                try: 
                    temp = self.ElectrodeData.graphWidget.partition[counter]
                except IndexError:
                    temp = 0            

                if counter == len(self.ElectrodeData.ElectrodeIds): 
                        break
                if k == self.ElectrodeData.ElectrodeIds[counter]:  
                    node_value=ElectrodeNode(self,counter,k,self.ElectrodeData.contextFlag)
                    
                    # initialize electrode opacity
                    opacity=ElectrodeOpacity(self, k, counter)
                    self.ElectrodeOpacity.append(opacity)

                    node_value.PutColor(self.ElectrodeData.graphWidget.communityDetectionEngine.clut[counter])
                    node_value.xy = (x,y)
                    node_value.translate(0,25)
                    node_value.setPos(QtCore.QPointF(x,y))
                    self.NodeIds.append(node_value)
                    self.scene.addItem(node_value)
                    counter = counter+1 
                k = k + 1


    def setMaxVal(self,value):
        global MaxVal
        MaxVal = value
        self.UpdateColors()

    def setMinVal(self,value):
        global MinVal
        MinVal = value
        self.UpdateColors()

    @Slot(bool)
    def changeMaxState(self,state):
        self.MaxButtonCheck = not(self.MaxButtonCheck)

    def Refresh(self):
        for node in self.NodeIds:
            node.update()
        self.Scene_to_be_updated.update()

    def regenerateElectrodes(self,timestep):
        for node in self.ElectrodeOpacity:
            node.RegenerateMaxMin(timestep)

    @staticmethod
    def unselectNodes(node):
        node.setSelected(False)

    def SelectNode(self,regionId):
        self.NodeIds[regionId].setSelected(True)

    def UpdateColors(self):
        self.regenerateElectrodes(self.ElectrodeData.timeStep)
        for node in self.NodeIds:
            self.unselectNodes(node)
            try:
                temp = self.ElectrodeData.graphWidget.partition[node.counter]
            except IndexError:
                temp = 0 

            if self.ElectrodeData.ScalarSize: 
                Size = eval('self.ElectrodeData.graphWidget.'+self.ElectrodeData.electrodeSizeFactor+'[node.counter-1]')
                node.setNodeSize(Size,self.ElectrodeData.electrodeSizeFactor)
            else: 
                Size = 0.4 
                node.setNodeSize(Size,"nothing to Display")

            if not(self.ElectrodeData.nodeSizeFactor == 1):
                node.ScalarNodeSize(self.ElectrodeData.nodeSizeFactor)

        if self.ElectrodeData.Glyph:
            for node in self.NodeIds:
                node.setGlyph(True)

        if self.ElectrodeData.ElectrodeScreenshot:
            pixmap = QtGui.QImage(self.scene.sceneRect().size().toSize())
            pAin = QtGui.QPainter(pixmap)
            self.scene.render(pAin,QtCore.QRectF(self.width/4+50, self.height/4+50, 3*self.width/6,3*self.height/6))
            fileName = str("Time_"+str(self.ElectrodeData.timeStep)+"_Syllable_"+str(self.ElectrodeData.syllableUnit)+"_Alg_"+str(self.ElectrodeData.clusterActivated)+".png")
            pixmap1 = QtGui.QPixmap.fromImage(pixmap)

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_T:
            print "transparent"
            self.transparent = not(self.transparent)
        if key == QtCore.Qt.Key_D:
            print "THIS IS THE TIMESTEP",(self.ChunkNo)*self.slices + self.ElectrodeData.CommunitiesAcrossTimeStep.Offset, (self.ChunkNo+1) * self.slices + self.ElectrodeData.CommunitiesAcrossTimeStep.Offset
        if key == QtCore.Qt.Key_R:
            print "Refresh the Entire Screen"
            self.RefreshInteractivityData()
        if key == QtCore.Qt.Key_C:
            print "changed"
            self.Cluster1= not(self.Cluster1) 
        if key == QtCore.Qt.Key_A:
            print "A"
            self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            self.showFullScreen()
            self.update()
        if key == QtCore.Qt.Key_Z:
            print "Changing offset"
            self.ElectrodeData.CommunitiesAcrossTimeStep.changeVizOffset()

    def wheelEvent(self, event):
        self.scaleView(math.pow(2.0, -event.delta() / 1040.0))

    def scaleView(self, scaleFactor):
        factor = self.matrix().scale(scaleFactor, scaleFactor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()
        if factor < 0.07 or factor > 100:
            return
        self.scale(scaleFactor, scaleFactor)
        del factor

    # def contextMenuEvent(self, event):
    #     menu = QtGfui.QMenu()
    #     testAction = QtGui.QAction('Refresh Screen', None)
    #     testAction.triggered.connect(self.RefreshInteractivityData)
    #     menu.addAction(testAction)
    #     # menu.exec_(event.globalPos())