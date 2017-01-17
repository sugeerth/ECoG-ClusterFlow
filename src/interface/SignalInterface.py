from PySide import QtCore, QtGui
import weakref

TotalNo = 61

class Interface(QtCore.QObject):
	DataChange = QtCore.Signal(bool)
	def __init__(self, widget, Ui, Electrode, electrodeUI, communitiesAcrossTimeStep,\
	 Tab_1_CorrelationTable, Tab_2_CorrelationTable, Visualizer, quantData, quantTableObject, Graph_Layout):
		self.widget = widget
		self.Ui = Ui
		# self.CustomWebView = None
		self.Graph_Layout = Graph_Layout
		self.quantData = quantData
		self.quantTableObject = quantTableObject 
		self.Electrode = Electrode
		self.electrodeUI = electrodeUI
		self.communitiesAcrossTimeStep = communitiesAcrossTimeStep
		self.Tab_1_CorrelationTable = Tab_1_CorrelationTable
		self.Tab_2_CorrelationTable = Tab_2_CorrelationTable

		#Signal Connections
		self.PreComputeSignals()
		self.connectAnimationSignals()
		self.communityModeSignals()
		self.NodeSelectionSignals()
		self.ReferenceSignals()
		self.ColorSignals()
		self.quantSignals()
		self.Graph_LayoutSignals()
		self.GlyphSignals()
		self.TimeStepInterval()
		self.TableSignals()
		# self.WebSignals()
		self.LinkingWithinTables()
		self.SyllableSignals()
		# self.CommentedSignals()


	def TimeStepInterval(self):
		pass


	def LinkingWithinTables(self):
		# self.Tab_1_CorrelationTable.selectedRegionChanged.connect(self.widget.NodeSelected)
		self.Tab_1_CorrelationTable.selectedRegionChanged.connect(self.Tab_2_CorrelationTable.selectRegion)
		self.Tab_2_CorrelationTable.selectedRegionChanged.connect(self.Tab_1_CorrelationTable.selectedRegionChanged)
		self.widget.CommunityColorAndDict.connect(self.Tab_1_CorrelationTable.setRegionColors)
		self.widget.CommunityColorAndDict.connect(self.Tab_2_CorrelationTable.setRegionColors)
	def WebSignals(self):
		self.Electrode.StopAnimationSignal.connect(self.CustomWebView.reload)

	def connectAnimationSignals(self):
		self.Electrode.AnimationSignal1.connect(self.widget.changeTimeStepSyllable)
		self.Electrode.StopAnimationSignal.connect(self.widget.ToggleAnimationMode)

	def TableSignals(self):
		self.widget.regionSelected.connect(self.Tab_1_CorrelationTable.selectRegion)
		self.widget.regionSelected.connect(self.Tab_2_CorrelationTable.selectRegion)

	def PreComputeSignals(self):
		self.Electrode.clusterObject.connect(self.communitiesAcrossTimeStep.initializePrecomputationObject)
		self.Electrode.clusterObject.connect(self.widget.communityDetectionEngine.initializePrecomputationObject)
		print "Setting signals for precomputtation"

	def communityModeSignals(self):
		self.widget.CommunityMode.connect(self.Electrode.updateElectrodeView)
		self.Electrode.NumberOfClusterChange.connect(self.widget.ComputeUpdatedClusters)

	def NodeSelectionSignals(self):
		self.widget.regionSelected.connect(self.Electrode.colorRelativeToRegion)
		self.Tab_1_CorrelationTable.selectedRegionChanged.connect(self.Electrode.colorRelativeToRegion)
		# self.Electrode.NodeSelected.connect(self.widget.NodeSelected)
		self.Tab_1_CorrelationTable.selectedRegionChanged.connect(self.Electrode.colorRelativeToRegion)

	def ReferenceSignals(self):
		self.communitiesAcrossTimeStep.sendLCDValues.connect(self.Electrode.displayLCDNumber)
		self.Electrode.TowValuesChanged.connect(self.widget.changeStuffDuetoTowChange)
		self.Electrode.ClusteringAlgorithmChange.connect(self.widget.ClusterChangeHappening)
		self.Electrode.NumberOfClusterChange.connect(self.widget.ComputeUpdatedClusters)
		self.Electrode.selectSeedNode.connect(self.widget.communityDetectionEngine.setCounterValues)

	def ColorSignals(self):
		self.widget.CalculateColors1.connect(self.communitiesAcrossTimeStep.CalculateClustersForTow)
		self.widget.CalculateFormulae.connect(self.communitiesAcrossTimeStep.CalculateStabilityOfMatrices)
	def GlyphSignals(self): 
		self.Electrode.GlyphSignal.connect(self.Electrode.selectGlyph)

	def SyllableSignals(self):
		self.Electrode.syllabeSignal.connect(self.Electrode.selectSyllable)
		# self.syllable.selectedSyllableChanged.connect(self.Electrode.selectSyllable) 
	def quantSignals(self):
		self.widget.ThresholdChange.connect(self.quantData.ThresholdChange)
		self.quantData.DataChange.connect(self.quantTableObject.setTableModel)

	def Graph_LayoutSignals(self):
		self.widget.DendoGramDepth.connect(self.Graph_Layout.SliderValueChanged)
		self.widget.propagateLevelValue.connect(self.Graph_Layout.LevelValueInSlider)

	def CommentedSignals(self):
		# self.widget.CommunityColorAndDict.connect(self.Electrode.setRegionColors)
		# self.Electrode.AnimationSignal.connect(self.correlationTable.changeTableContents)
		# self.widget.WidgetUpdate.connect(self.Electrode.updateElectrodeView)
		# self.Electrode.AnimationSignal.connect(self.communitiesAcrossTimeStep.initiateMatrix)
		# self.Electrode.AnimationSignal1.connect(self.communitiesAcrossTimeStep.updateStabilitySignal)
		# self.Electrode.syllabeSignal.connect(self.widget.changeTimeStepSyllable)
		pass
