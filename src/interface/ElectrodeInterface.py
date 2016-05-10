from PySide import QtCore, QtGui
import weakref

timestep = 20
DefaultNumberOfClusters = 4

class ElectrodeInterface(QtCore.QObject):
	def __init__(self, widget, Ui, Electrode, electrodeUI, communitiesAcrossTimeStep, Tab_1_CorrelationTable, Tab_2_CorrelationTable, Visualizer):
		global timestep
		timestep = widget.OverallTimestep
		self.widget = widget
		self.view = None
		self.MainSmallMultipleInterface = None
		self.Ui = Ui
		self.Electrode = Electrode
		self.Visualizer = Visualizer
		self.electrodeUI = electrodeUI
		self.communitiesAcrossTimeStep = communitiesAcrossTimeStep
		self.Tab_1_CorrelationTable = Tab_1_CorrelationTable
		self.InterfaceLayout = None

		#Signal Connections
		self.ConnectWidgets()
		self.ConnectPlayStop()
		self.connectSyllables()
		self.connectGlyphs()
		self.ConnectDataValues()
		self.MaintainLayout()

	def ConnectWidgets(self):
		# self.electrodeUI.OpacityThreshold.valueChanged[int].connect(self.Electrode.OpacityThreshold)
		self.electrodeUI.ValueSlider1.hide()
		self.Visualizer.ValueSlider1.valueChanged[int].connect(self.Electrode.changeMaxOpacity)

		self.Visualizer.opacitySignals1.stateChanged.connect(self.Electrode.OpacityToggling)
		self.Visualizer.ElecNodes1.stateChanged.connect(self.Electrode.ElectrodeNodeSize)
		self.Visualizer.ElectrodeSize.activated[str].connect(self.Electrode.checkboxActivated)
		self.Visualizer.NodeSize1.activated[str].connect(self.Electrode.ClusterActivated)
		self.Visualizer.nodeSize1.valueChanged[int].connect(self.Electrode.nodeSizeChanged)
		self.Visualizer.timeInterval1.valueChanged[int].connect(self.Electrode.timeIntervalAdjust)
		self.Visualizer.lcdNumber1.setPalette(QtCore.Qt.red)
		self.Visualizer.lcdNumber1.display(0.45)
		self.Visualizer.Glyphs.stateChanged.connect(self.Electrode.MultipleTimeGlyph)
		self.Visualizer.ElectrodeImages.stateChanged.connect(self.Electrode.CheckScreenshotImages)
		self.Visualizer.PreCompute1.stateChanged.connect(self.Electrode.PreComputeClusters)
		self.Visualizer.Permute.stateChanged.connect(self.communitiesAcrossTimeStep.changePermute)
		self.Visualizer.TrElectrode.stateChanged.connect(self.communitiesAcrossTimeStep.changeTrElectrode)
		self.Visualizer.TrCommunity.stateChanged.connect(self.communitiesAcrossTimeStep.changeTrCommunity)
		self.Visualizer.SliceInterval.valueChanged[int].connect(self.Electrode.SliceInterval)
		
		# self.electrodeUI.GraphPlots.stateChanged.connect(self.Electrode.SaveHistoryPlots)
		self.Visualizer.MaxButtonCheck.stateChanged.connect(self.Electrode.ElectrodeView.changeMaxState)
		self.Visualizer.Max1.returnPressed.connect(self.Electrode.LineEditChanged)
		self.Visualizer.Slices.returnPressed.connect(self.Electrode.changeSliceNumber)

		self.Visualizer.Min.returnPressed.connect(self.Electrode.MinPressed)
		self.Visualizer.Max.returnPressed.connect(self.Electrode.MaxPressed)
		# self.electrodeUI.Max_2.returnPressed.connect(self.widget.communityDetectionEngine.changeCommunities)

		self.Visualizer.Tow1.valueChanged.connect(self.Electrode.TowValueChanged)
		self.Visualizer.Tow1.setMaximum(timestep)

	def ConnectPlayStop(self):
		self.Visualizer.pushButton2.setToolTip('This is a <b>PlayButton</b> widget')
		self.Visualizer.pushButton2.clicked.connect(self.Electrode.playButtonFunc)
		self.Visualizer.pushButton3.setToolTip('This is a <b>StopButton</b> widget')
		self.Visualizer.pushButton3.clicked.connect(self.Electrode.stopButtonFunc)
		self.Visualizer.Reset1.clicked.connect(self.Electrode.ResetButton)

	# self.electrodeUI.ClusterNodes.stateChanged.connect(self.FreezeColors)

	def MaintainLayout(self):
		self.InterfaceWidget =QtGui.QWidget()
		self.InterfaceWidget.setContentsMargins(0, 0, 0, 0)
		self.InterfaceLayout = QtGui.QVBoxLayout()
		ConsensusPlot = QtGui.QHBoxLayout()
# 
		self.InterfaceLayout.setContentsMargins(0, 0, 0, 0)
		# self.communitiesAcrossTimeStep.PlotWidget.setMinimumSize(200,100)
		# self.communitiesAcrossTimeStep.PlotWidget.setMaximumSize(200,100)

		# ConsensusPlot.addWidget(self.communitiesAcrossTimeStep.PlotWidget)
		self.InterfaceLayout.addWidget(self.Visualizer.MainTab)
		self.InterfaceLayout.setContentsMargins(0, 0, 0, 0)

		self.InterfaceLayout.addLayout(ConsensusPlot)
		self.InterfaceLayout.setContentsMargins(0, 0, 0, 0)
		
		self.InterfaceWidget.setContentsMargins(0, 0, 0, 0)
		self.InterfaceWidget.setLayout(self.InterfaceLayout)
		self.InterfaceWidget.setContentsMargins(0, 0, 0, 0)
		self.InterfaceWidget.show()

	# def connectCustomWebView(self, view):
	# 	self.view = view
	# 	self.Visualizer.SliceInterval.valueChanged[int].connect(self.view.slicesChanged)

	def connectCustomWebView(self, view, MainSmallMultipleInterface):
		self.view = view
		self.MainSmallMultipleInterface = MainSmallMultipleInterface
		self.Visualizer.SliceInterval.valueChanged[int].connect(self.view.slicesChanged)
		self.Visualizer.SliceInterval.valueChanged[int].connect(self.MainSmallMultipleInterface.sliceLayoutChanged)

	def connectGlyphs(self):
		self.Visualizer.AreaGlyph.clicked.connect(self.Electrode.AreaGlyphClicked)
		self.Visualizer.OpacityGlyph.clicked.connect(self.Electrode.OpacityGlyphClicked)
		self.Visualizer.BarGlyph.clicked.connect(self.Electrode.BarGlyphClicked)

	def connectSyllables(self):
		self.Visualizer.one1.clicked.connect(self.Electrode.SyllableOneClicked)
		self.Visualizer.two1.clicked.connect(self.Electrode.SyllableTwoClicked)
		self.Visualizer.three1.clicked.connect(self.Electrode.SyllableThreeClicked)
		self.Visualizer.four1.clicked.connect(self.Electrode.SyllableFourClicked)
		self.Visualizer.five1.clicked.connect(self.Electrode.SyllableFiveClicked)
		self.Visualizer.six1.clicked.connect(self.Electrode.SyllableSixClicked)
		# self.Visualizer.five1.hide()
		# self.Visualizer.six1.hide()

	def ConnectDataValues(self):
		self.Visualizer.Max.setText("{0:.2f}".format(self.Electrode.ElectrodeView.MaxVal))
		self.Visualizer.Min.setText("{0:.2f}".format(self.Electrode.ElectrodeView.MinVal))
		# self.electrodeUI.Mid.setText("{0:.0f}".format(self.Electrode.opacityThreshold))




