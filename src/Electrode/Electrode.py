from PySide import QtCore, QtGui
import tempfile
import pprint
import colorsys
from PySide.QtCore import *
import time
import numpy as np
import math
from PySide.QtCore import QTimer, SIGNAL
from PIL import Image, ImageDraw

import community as cm

import json
import yaml
import pickle
try:
    # ... reading NIfTI 
    import numpy as nm
# ... graph drawing655g
    import networkx as nx

except:
    print "Couldn't import all required packages. See README.md for a list of required packages and installation instructions."
    raise

from ElectrodeView import ElectrodeView
from collections import defaultdict
# from CommunityAnalysis.CommunitiesAcrossTimeStep import CommunitiesAcrossTimeStep

def ColorToInt(color):
    r, g, b, a = map(np.uint32, color)
    return r << 24 | r << 16 | g << 8 | b

"""
Send data when the timestep is changed, communities are changed, to all the other views of the prototype. 
Signals for 
1) Timings 
2) Communities 
3) Colors in (Planning to use pallettes for this purpose) 

Make code as generalizable as possible 
"""
timestep = 19
DefaultNumberOfClusters = 4
LayoutWidth =1241 
LayoutHeight = 461

# Change the ratio based on the number of timesteps provided 
Row = 4
Column = 16
TotalNo = 64

RowXColumn =4

from PySide import QtGui


class LayoutForSmallMultiples(QtGui.QWidget):
    def __init__(self, Electrode, smallMultiples, x_interval, y_interval):
        super(LayoutForSmallMultiples, self).__init__()
        self.smallMultiples = smallMultiples
        # self.TrackingGraphView = TrackingGraphView
        self.Electrode = Electrode
        self.x_interval = x_interval
        self.y_interval = y_interval
        self.setMinimumSize(QtCore.QSize(701,561))
        self.initUI()
        
    def initUI(self):
        pass

def returnThresholdedDataValues(data, weight = 0):
    ThresholdData = np.copy(data)
    low_values_indices = ThresholdData < weight  # Where values are low
    ThresholdData[low_values_indices] = 0
    return nx.from_numpy_matrix(ThresholdData) 

class PreComputeShared:
    __shared_state = {}

    def __init__(self, graphWidget, ElectrodeData):
        self.__dict__ = self.__shared_state
        self.ElectrodeData = ElectrodeData
        self.state = self.ElectrodeData
        self.graphWidget = graphWidget
        # self.ClusterAlgorithms = ClusterAlgorithms
        self.PreComputeCommunities = defaultdict(list)

    def ComputeCommunities(self,syllable, ClusteringAlgo, Number_of_clusters): 
        self.DataSyllable = self.ElectrodeData[syllable]

        # assert np.shape(self.DataSyllable) == (256,64)
        self.PreComputeCommunities.clear()

        # change this value for clustering of differenct issues
        for i in range(timestep):
            #get the raw data
            """Need to be tested"""
            graphData = returnThresholdedDataValues(self.DataSyllable[i])
            # compute the clusters 
            self.graphWidget.Timestep = int(i)
            self.PreComputeCommunities[int(i)] = self.graphWidget.communityDetectionEngine.resolveCluster(ClusteringAlgo,graphData, Number_of_clusters, i)
        return self.PreComputeCommunities

    def WriteOnaFile(self, name):
        assert not(self.PreComputeCommunities == None)
        with open(name, 'w') as outfile:
            pickle.dump(self.PreComputeCommunities, outfile)

    def ReadFromFile(self, name):
        assert not(self.PreComputeCommunities == None)
        output_json = pickle.load(open(name))
        assert output_json == self.PreComputeCommunities

    def setters(self):
        return self.state

class ImageLabel(QtGui.QGraphicsView):

    CommunityColor = QtCore.Signal(list)
    CommunityColorAndDict = QtCore.Signal(list,dict)
    CommunityMode = QtCore.Signal(bool)
    NodeSelected = QtCore.Signal(int)
    AnimationSignal1 = QtCore.Signal(int,int)
    StopAnimationSignal = QtCore.Signal(bool)
    changeCommunityColors = QtCore.Signal(int)
    TowValuesChanged = QtCore.Signal(int)
    syllabeSignal = QtCore.Signal(int)
    GlyphSignal = QtCore.Signal(int)
    ClusteringAlgorithmChange = QtCore.Signal(int)
    NumberOfClusterChange = QtCore.Signal(int)
    clusterObject = QtCore.Signal(object)

    selectSeedNode = QtCore.Signal(int)


    def __init__(self,dataProcess,correlationTable,colorTable,selectedcolor,counter,graphWidget ,electrodeUI,  Visualizer,Brain_image_filename):
        QtGui.QGraphicsView.__init__(self)
        # self.setMouseTracking(True)

        # Object tracking
        global timestep
        timestep = dataProcess.timestep
        self.Chunks = dataProcess.timestep
        Visualizer.Max1.setText(str(DefaultNumberOfClusters))
        self.colorTable = colorTable
        self.Visualizer = Visualizer
        self.electrodeUI = electrodeUI
    
        # Default values of the variables        
        self.GlyphUnit = 1
        self.CommunitiesAcrossTimeStep= None
        self.ElectrodeInterface = None
        self.slices = 4 

        self.correlationTable = correlationTable  
        self.dataProcess = dataProcess
        self.graphWidget = graphWidget
        self.mat = self.dataProcess.mat
        self.syllableUnit = self.dataProcess.syllableUnit
        self.ElectrodeIds = self.dataProcess.ElectrodeIds
        self.ElectrodeSignal = self.dataProcess.ElectrodeSignals
        self.ElectrodeData= self.dataProcess.ElectodeData
        self.timeStep = self.dataProcess.timestep
        self.data = correlationTable.data
        self.Brain_image_filename = Brain_image_filename

        self.PreComputeDataObject = PreComputeShared(self.graphWidget, self.ElectrodeData)
        # self.ClusterAlgorithms = self.ClusterAlgorithms()

        self.communityPos = dict()
        self.number_of_Connected_Components = []
        self.modularity =[]
        self.NodeIds = []
        self.SmallMultipleElectrode = []

        self.timeInterval = 2000000
        # Size tracking 
        self.ScalarSize = False
        self.contextFlag = False
        self.FreezeColors =False
        self.Glyph = False
        self.OpacityOn = True
        self.ElectrodeScreenshot = False
        self.PreComputed = False
        self.saveHistoryPLotsState = False
        
        self.Network = []
        self.electrodeSizeFactor ="Centrality"
        self.clusterActivated = 0

        self.ColorVisit =list()
        self.ColorToBeSentToVisit = list()

        # integer values
        self.fromAnimate = 0 
        self.TowValue = -1
        self.opacityThreshold = -1 
        self.toAnimate1 = timestep - 2
        self.No_Clusters = DefaultNumberOfClusters
        self.nodeSizeFactor = 1 

        # Scene specific values
        scene = QtGui.QGraphicsScene(self)
        scene.setItemIndexMethod(QtGui.QGraphicsScene.NoIndex)
        self.setScene(scene)
        self.setMinimumSize(400, 370)

        # add timer options
        self.timer = QtCore.QTimer(self)
        self.timer = QTimer()  # set up your QTimer
        self.timer.setInterval(20000000)
        self.timer.timeout.connect(self.ChangeAnimation)  # connect it to your update function

        # FIX ME 85 number of electrodes 
        self.counter = len(correlationTable.header)

        self.selectedColor = selectedcolor
        self.AddWidgets()

    def AddWidgets(self):
        im = self.dataProcess.im
        draw = ImageDraw.Draw(im)
        self.setSizePolicy(QtGui.QSizePolicy.Policy.Expanding, QtGui.QSizePolicy.Policy.Expanding)
        
        # Saving the image file as an output file
        self.label = QtGui.QLabel()
        self.NodeSlider()
        im.save(self.Brain_image_filename)

        # Loading the pixmap for better analysis
        loadedImage = QtGui.QImage()
        loadedImage.load(self.Brain_image_filename)
        self.PixMap = QtGui.QPixmap.fromImage(loadedImage)
        
        self.ElectrodeView = ElectrodeView(self)
        x_interval = LayoutWidth/6 
        y_interval = LayoutHeight/6

        for i in range(self.Chunks+1):
            self.SmallMultipleElectrode.append(ElectrodeView(self, i, x_interval,y_interval))
            CommunitySelectPerTime = QtCore.Signal(list, int ,list, list)
            self.SmallMultipleElectrode[i].CommunitySelectPerTime.connect(self.SelectingCommunitiesInaTimestep)
            self.SmallMultipleElectrode[i].DataLink.connect(self.GettingDataFromSmallMultiples)
            self.SmallMultipleElectrode[i].CommunitySelectAcrossTime.connect(self.SelectingCommunitiesAcrossaTimestep)

        self.LayoutForSmallMultiples = LayoutForSmallMultiples(self,self.SmallMultipleElectrode,x_interval,y_interval)

        # # Use pycharm to have a better way to do it
        self.ImageView = QtGui.QHBoxLayout()
        self.ImageView2 = QtGui.QHBoxLayout()
        self.ImageView2.setContentsMargins(0, 0, 0, 0)
        self.ImageView2.addWidget(self.ElectrodeView)
        self.ImageView2.setContentsMargins(0, 0, 0, 0)

        # Image additions 
        vbox = QtGui.QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.addLayout(self.ImageView2)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.scene = QtGui.QGraphicsScene(0, 0,500 ,600)
        self.setContentsMargins(0, 0, 0, 0)

        self.setLayout(vbox)

    @Slot(bool)
    def CheckScreenshotImages(self):
        self.ElectrodeScreenshot = True

    @Slot(int)
    def displayLCDNumber(self,value):
        self.Visualizer.lcdNumber1.display(value)
        # self.electrodeUI.lcdNumber.display(value)

    @Slot(bool)
    def ElectrodeNodeSize(self,state):
        if state: 
            self.ScalarSize = True
        else:
            self.ScalarSize = False
        self.ElectrodeView.UpdateColors()
        # self.UpdateColorsSmallMultiples()

    @Slot(int)
    def timeIntervalAdjust(self,value):
        self.timeInterval = 2000000
        # self.timer.setInterval(self.timeInterval)

    @Slot(int)
    def SyllableOneClicked(self):
        self.syllabeSignal.emit(0)

    @Slot(int)
    def SyllableTwoClicked(self):
        self.syllabeSignal.emit(1)

    @Slot(int)
    def SyllableThreeClicked(self):
        self.syllabeSignal.emit(2)

    @Slot(int)
    def SyllableFourClicked(self):
        self.syllabeSignal.emit(3)

    @Slot(int)
    def SyllableFiveClicked(self):
        self.syllabeSignal.emit(4)

    @Slot(int)
    def SyllableSixClicked(self):
        self.syllabeSignal.emit(5)
    
    """
    Selection of various glyphs 
    """
    @Slot(int)
    def AreaGlyphClicked(self):
        self.GlyphSignal.emit(0)

    @Slot(int)
    def BarGlyphClicked(self):
        self.GlyphSignal.emit(2)

    @Slot(int)
    def OpacityGlyphClicked(self):
        self.GlyphSignal.emit(1)

    @Slot(bool)
    def FreezeColors(self,state):
        if state:
            self.FreezeColors = True
            """Initiate color change immediately"""
        else:
            self.FreezeColors = False

    @Slot()
    def MinPressed(self):
        MinVal = float((self.Visualizer.Min.text().encode('ascii','ignore')).replace(' ',''))

        self.ElectrodeView.setMinVal(MinVal)
        for i in range(self.Chunks+1):
            self.SmallMultipleElectrode[i].setMinVal(MinVal)

        self.ElectrodeView.UpdateColors()   
        # self.UpdateColorsSmallMultiples()

    """
    Saving the current state of the visualization tool 
    This allows us to gather the results at a later time  
    """
    @Slot()
    def SaveState(self):
        print "Will save the dataset as filename",str(self.timeStep)+str(self.syllableUnit)+str(self.graphWidget.ClusteringAlgorithm)+".json"

    @Slot()
    def MaxPressed(self):
        MaxVal = int((self.Visualizer.Max.text().encode('ascii','ignore')).replace(' ',''))
        # self.ElectrodeView.MaxVal = MaxVal
        self.ElectrodeView.setMaxVal(MaxVal)
        self.ElectrodeView.UpdateColors()
        # self.UpdateColorsSmallMultiples()

    @Slot(bool)
    def checkboxActivated(self,state):
        self.electrodeSizeFactor = state
        self.ElectrodeView.UpdateColors()
        # self.UpdateColorsSmallMultiples()

    @Slot(str)
    def ClusterActivated(self,state):
        """0-louvain, 1-k-means , 2-hierarchical"""
        # Reset the precomputation state to False
        self.PreComputeState = False
        self.graphWidget.PreComputeState = False
        self.RefreshPreComputeData()
        if state == "Louvain": 
            self.clusterActivated = 0
            self.ClusteringAlgorithmChange.emit(0)
            print "Toggle the software state to Louvain"
        elif state == "Hierarchical":
            self.clusterActivated = 1
            self.ClusteringAlgorithmChange.emit(1)
            print "Toggle state to Hierarrchical clustering"
        elif state == "K-medoids":
            self.clusterActivated = 2
            self.ClusteringAlgorithmChange.emit(2)
            print "Toggle state to K-medoids"
        elif state == "ConsensusClustering":
            self.clusterActivated = 4
            self.ClusteringAlgorithmChange.emit(4)
            print "Toggle state to ConsensusClster"
            print "NICE"
        elif state == "CustomCluster":
            self.clusterActivated = 5
            self.ClusteringAlgorithmChange.emit(5)
            print "Toggle state to CustomCluster"
        elif state == "ConsensusPreComputed":
            self.clusterActivated = 6
            self.ClusteringAlgorithmChange.emit(6)
            print "Toggle state to Precomputed Consensus Cluster"
        elif state == "LocalGraphClustering":
            self.clusterActivated = 7
            self.ClusteringAlgorithmChange.emit(7)
            print "Toggle state to Local Graph Clustering"
        # self.PreComputeClusters()

    @Slot()
    def ResetButton(self):
        """ Make the funcitonality to re adjust all data structures etc"""  
        self.timer.stop()
        self.timeStep = self.fromAnimate 
        self.timeStepSlider.setValue(self.fromAnimate)
        self.graphWidget.ToggleAnimationMode(True)
        self.CommunitiesAcrossTimeStep.flushData()
        self.CommunitiesAcrossTimeStep.variableWidth = 0

    @Slot()
    def LineEditChanged(self):
        """Handling Line Edit changes"""
        self.No_Clusters = int((self.Visualizer.Max1.text().encode('ascii','ignore')).replace(' ',''))
        Number_of_clusters = self.No_Clusters
        self.graphWidget.communityDetectionEngine.changeClusterValue(Number_of_clusters)
        self.ResetButton()
        self.RefreshPreComputeData()
        self.changeClusterConfiguration(self.No_Clusters)

    @Slot(int)
    def GettingDataFromSmallMultiples(self, data):
        print data

    def RefreshInteractiveData(self):
        for i in self.SmallMultipleElectrode:
            i.RefreshInteractiveData()
            i.update()

    def SelectingCommunitiesInaTimestep(self, CommunityColor ,CommunityIndex, SmallMultiplesCounter):
        # retrieve those electrodes and highlight the electrode now.
        self.SmallMultipleElectrode[SmallMultiplesCounter].HighlightThisColor(CommunityColor,CommunityIndex)

    def SelectingCommunitiesAcrossaTimestep(self, CommunityColor ,CommunityIndex, SmallMultiplesCounter):
        # self.SmallMultipleElectrode[SmallMultiplesCounter].HighlightAcrossTime(CommunityColor,CommunityIndex)
        for ElectroV in self.SmallMultipleElectrode:
            ElectroV.HighlightAcrossTime(CommunityColor, CommunityIndex)
            ElectroV.update()


    def RefreshPreComputeData(self):
        self.PreComputed = False 
        self.ElectrodeInterface.Visualizer.PreCompute1.setChecked(False)

        self.CommunitiesAcrossTimeStep.unsetPreComputationDate()
        self.graphWidget.communityDetectionEngine.unsetPreComputationDate()

    def changeClusterConfiguration(self,no):
        # print "WARNING: Going to precompute the data with",no,"Clusters"
        # self.graphWidget.communityDetectionEngine.Number_of_clusters = no
        self.ElectrodeInterface.Visualizer.PreCompute1.setChecked(True)
        self.PreComputeClusters(True)

    @Slot(bool)
    def MultipleTimeGlyph(self, state):
        self.Glyph = state
        self.ElectrodeView.UpdateColors()
        self.UpdateColorsSmallMultiples()

    @Slot(bool)
    def PreComputeClusters(self, state):
        self.PreComputed = state
        if self.PreComputed:
            print "Will now precompute stuff for Clustering Algorithm",self.graphWidget.ClusteringAlgorithm
            print "If it is not louvain Will precompute clusters with ",self.No_Clusters,"CLUSTERS"
            print "THIS SHOULD BE INVOKED when syllables are changed"
            self.PreComputeStuff(self.clusterActivated, self.syllableUnit, self.No_Clusters)
        else: 
            self.RefreshPreComputeData()

    def UpdateColorsSmallMultiples(self):
        for i in range(self.Chunks+1):
            self.SmallMultipleElectrode[i].UpdateColors()

    def PreComputeStuff(self, Algo,syllable, Number_of_clusters):
        PreComputeCommunities = self.PreComputeDataObject.ComputeCommunities(syllable, Algo ,Number_of_clusters)
        print "Deriving clusters for ",Algo, "length", len(PreComputeCommunities)
        
        print "#* Writing on a File"

        name = "ConsensusData/ConsensusCluster"+str(syllable)+str(Algo)+".json"
        print name, "is the file name"
        
        self.PreComputeDataObject.WriteOnaFile(name)
        self.PreComputeDataObject.ReadFromFile(name)

        self.clusterObject.emit(PreComputeCommunities)

    @Slot(int)
    def changeMaxOpacity(self,value):
        # self.ElectrodeView.sendOpacityValue(value)
        # print "Hello", value
        for i in range(self.Chunks+1):
            self.SmallMultipleElectrode[i].changeMaxOpacity(value)

        # Sending Values to AcrossTimeStep for slider Change
        self.CommunitiesAcrossTimeStep.UpdateColorsInElectrodeView(self.CommunitiesAcrossTimeStep.nodelist1,self.CommunitiesAcrossTimeStep.Offset)
        self.UpdateColorsSmallMultiples()

    @Slot(int)
    def TowValueChanged(self,value):
        value = int(value)
        self.TowValue = value
        self.TowValuesChanged.emit(value)

    @Slot()
    def InitiateTrackingGraph(self):
        print self.fromAnimate, "to ", self.toAnimate1

    @Slot(bool)
    def OpacityToggling(self,state):
        if state: 
            self.OpacityOn = True
        else:
            self.OpacityOn = False
        self.ElectrodeView.UpdateColors()
        self.UpdateColorsSmallMultiples()

    @Slot(int)
    def nodeSizeChanged(self,value):
        self.nodeSizeFactor = 0.3 + value * 0.01 
        self.ElectrodeView.UpdateColors()
        self.UpdateColorsSmallMultiples()

    @Slot(int)
    def OpacityThreshold(self,value):
        self.opacityThreshold = int(value*0.01*255)
        self.Visualizer.Mid.setText("{0:.0f}".format(self.opacityThreshold))
        self.ElectrodeView.UpdateColors()
        self.UpdateColorsSmallMultiples()

    @Slot(int)
    def FromAnimate(self, value):
        self.fromAnimate = value

    @Slot(int)
    def ToAnimate(self, value):
        self.toAnimate1 = value

    @Slot()
    def ChangeAnimation(self):
        # print "TIMER ANIMATION THAT IS SENT ALL OVER THE WORLD"
        # time.sleep(self.timeInterval)
        if self.timeStep == 0: 
            self.ChangeValue(0)

        #* the ROOT CAUSE OF EVIL
        self.timeStepSlider.setValue(self.timeStep)
        self.timeStep = self.timeStep + 1

        if (self.timeStep > self.toAnimate1): 
            self.timer.stop()
            self.StopAnimationSignal.emit(True)
            self.graphWidget.AnimationMode = False
            # self.Refresh()
    @Slot()
    def playButtonFunc(self):
        self.timer.start(self.fromAnimate)

    # @Slot()
    # def ResetButton(self):
    #     print "pressing the reset button", "Enter\
    #     the time window if you would want to change it??","default is", self.fromAnimate,"to",self.toAnimate1
    #     self.timer.stop()
    #     # self.

    @Slot()
    def stopButtonFunc(self):
        self.timer.stop()
        self.StopAnimationSignal.emit(True)
    
    @Slot(int)
    def SliceInterval(self, value):
        self.Visualizer.Slices.setText(str(value))
        self.SmallMultipleElectrode[0].EmitSelectedElectrodeView.emit(0,50,value)
        
        for i in range(self.Chunks+1):
            self.SmallMultipleElectrode[i].changeSliceNumber(value) 

        self.CommunitiesAcrossTimeStep.SendValuesToElectrodeNodes(self.CommunitiesAcrossTimeStep.nodelist1)
        self.slices = value

    @Slot(int)
    def changeSliceNumber(self):
        value = int((self.Visualizer.Slices.text().encode('ascii','ignore')).replace(' ',''))
        self.Visualizer.Slices.setText(str(value))
        self.Visualizer.SliceInterval.setValue(value)
        self.SmallMultipleElectrode[0].EmitSelectedElectrodeView.emit(0,50,value)
        
        for i in range(self.Chunks+1):
            self.SmallMultipleElectrode[i].changeSliceNumber(value) 

        self.CommunitiesAcrossTimeStep.SendValuesToElectrodeNodes(self.CommunitiesAcrossTimeStep.nodelist1)
        self.slices = value

    def SaveHistoryPlots(self): 
        self.saveHistoryPLotsState = not(self.saveHistoryPLotsState)

    def resizeEvent(self, event):
        super(ImageLabel, self).resizeEvent(event)
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def mouseReleaseEvent(self, event):  
        posMouse =  event.pos()
        return super(ImageLabel, self).mouseReleaseEvent(event)

    def updateElectrodeView(self,state):
        self.ElectrodeView.UpdateColors()
        self.UpdateColorsSmallMultiples()


    def calculateColors(self):
        self.ElectrodeView.UpdateColors()
        self.UpdateColorsSmallMultiples()


    def colorRelativeToRegion(self,regionId):
        self.ElectrodeView.UpdateColors()
        self.UpdateColorsSmallMultiples()
        self.ElectrodeView.SelectNode(regionId)
        #* in questions

    def NodeSlider(self):
        self.timeStepSlider = QtGui.QSlider(QtCore.Qt.Horizontal,self)
        self.timeStepSlider.setRange(0, timestep)
        self.timeStepSlider.setValue(0)
        self.timeStepSlider.setToolTip("Time Step: %0.2f" % (self.timeStep))
        self.timeStepSlider.setTracking(False)
        self.timeStepSlider.valueChanged[int].connect(self.ChangeValue)
        self.timeStepSlider.sliderReleased.connect(self.unsetVariables)
        self.timeStepSlider.hide()
        # self.timeStepSlider.show()

    """Changes that needs to be done to unset the sliders in question"""
    def unsetVariables(self):
        print "unsetting slider"
        self.graphWidget.AnimationMode = False 

    """There has been a change in the timestep sliders"""
    def ChangeValue(self,value):
        # print "WARNING TIME HAS CHANGED TO",value, self.timeStep
        assert self.timeStep == value
        # self.timeStep = value
        self.timeStepSlider.setToolTip("Time Step: %0.2f" % (self.timeStep))
        # print "changes in timesteps", self.timeStep
        self.changesInTimestepSyllable()

    """Select a Glyph of interest"""
    def selectGlyph(self,GlyphUnit):
        self.GlyphUnit = GlyphUnit
        print "Selecting Glyphs"
        print "WARNING THE Glyphs are changed please have a look" 
        self.CommunitiesAcrossTimeStep.UpdateColorsInElectrodeView(self.CommunitiesAcrossTimeStep.nodelist1,self.CommunitiesAcrossTimeStep.Offset)
        self.ElectrodeView.UpdateColors()
        self.UpdateColorsSmallMultiples()

    """Select a syllable that will be appropritation"""
    def selectSyllable(self,syllableUnit):
        self.syllableUnit = syllableUnit
        print "WARNING THE syllables are changed please have a look" 
        # Reset the precomputation state to False
        self.PreComputeState = False
        self.graphWidget.PreComputeState = False
        
        # print "SYLLABLE HAS CHANGED, CHEK IF TIMESTEP IS EQUAL TO SYLLABLE",syllableUnit,self.timeStep
        self.graphWidget.correlationTable().changeTableContents(syllableUnit, self.timeStep)
        self.data = self.correlationTable.data
        self.ResetButton()
        self.CommunitiesAcrossTimeStep.flushData()
        self.playButtonFunc()

        # communitiesAcrossTimeStep.flushData()
        # Electrode.playButtonFunc() 

        # self.PreComputed = False
        # self.ElectrodeInterface.electrodeUI.PreCompute.setChecked(False)
        self.RefreshPreComputeData()
        # print "Syllable changes but with the same tow value", self.graphWidget.TowValue 
        self.graphWidget.changeStuffDuetoTowChange(self.graphWidget.communityDetectionEngine.TowValue)
        self.ElectrodeView.UpdateColors()
        self.UpdateColorsSmallMultiples()

    def changesInTimestepSyllable(self):
        self.graphWidget.correlationTable().changeTableContents(self.syllableUnit, self.timeStep)
        self.data = self.correlationTable.data
        # self.graphWidget.changeStuffDuetoTowChange(self.graphWidget.TowValue)
        # self.ElectrodeView.regenerateElectrodes(self.timeStep)
        self.ElectrodeView.UpdateColors()
        self.UpdateColorsSmallMultiples()

        # self.graphWidget.changeTimeStepSyllable(self.syllableUnit, self.timeStep)
        self.AnimationSignal1.emit(self.syllableUnit, self.timeStep)

    def RefreshElectrodes():
        for ElectroV in self.SmallMultipleElectrode:
            ElectroV.RefreshElectrodes()
            ElectroV.update()

    def ColorForVisit(self,partiton):
        self.ColorToBeSentToVisit = []
        for key,value in partition.items():
            self.ColorToBeSentToVisit.append(self.ColorVisit[value])

    def informationDisplay(self):
        print "TimeStep: ", self.timeStep, "\nSyllable: ",self.syllableUnit,"\nCommunities formed: ",len(set(self.graphWidget.partition.values())),"\nModularity: ",self.modularity,"\n"

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_A:
            print "Please print something"
        if key == QtCore.Qt.Key_C:
            self.contextFlag = not(self.contextFlag)
        if key == QtCore.Qt.Key_Q:
            self.RefreshElectrodes()

    def wheelEvent(self, event):
        self.scaleView(math.pow(2.0, -event.delta() / 1040.0))

    def scaleView(self, scaleFactor):
        factor = self.matrix().scale(scaleFactor, scaleFactor).mapRect(QtCore.QRectF(0, 0, 5, 5)).width()
        if factor < 0.07 or factor > 100:
            return
        self.scale(scaleFactor, scaleFactor)
        del factor

# 