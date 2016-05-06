from PySide import QtCore, QtGui
import weakref

class CommunitiesAcrossTimeStepInterface(QtGui.QWidget):
	DataChange = QtCore.Signal(bool)
	def __init__(self, AcrossTimestepUI, AcrossTimestepObject):
		QtGui.QWidget.__init__(self)

		self.UIWidget = QtGui.QVBoxLayout()  
		self.Widget1 = QtGui.QWidget()

		self.AcrossTimestepUI = AcrossTimestepUI
		self.AcrossTimestepObject = AcrossTimestepObject
		self.defaultValue = "0.3"
		self.ConnectControls()
		self.doControls()
		self.UIWidget.setContentsMargins(0,0,0,0)
		self.UIWidget.addWidget(self.AcrossTimestepUI)
		self.UIWidget.setContentsMargins(0,0,0,0)
		# self.AcrossTimestepUI.thresholdvalue.hide()
		self.setLayout(self.UIWidget)

	def ConnectControls(self):
		self.AcrossTimestepUI.VizTheme.activated[str].connect(self.AcrossTimestepObject.LayoutChange)
		self.AcrossTimestepUI.threshold.stateChanged.connect(self.AcrossTimestepObject.stateChanges)
		self.AcrossTimestepUI.thresholdvalue.valueChanged[int].connect(self.AcrossTimestepObject.thresholdValueChanged)

		self.AcrossTimestepUI.thresholdlineedit.setText(str(self.defaultValue))
		self.AcrossTimestepUI.thresholdlineedit.returnPressed.connect(self.AcrossTimestepObject.LineEditChanged)

	def doControls(self):
		self.AcrossTimestepUI.Layout.setContentsMargins(0,0,0,0)
		# self.AcrossTimestepUI.Layout.addWidget(self.AcrossTimestepObject)
		self.AcrossTimestepUI.Layout.setContentsMargins(0,0,0,0)
		# self.Widget1.setLayout(self.AcrossTimestepUI.Layout)
		# self.UIWidget.addWidget(self.AcrossTimestepUI) 	
		
		# self.UIWidget.addLayout(self.AcrossTimestepUI.Layout)

		# self.Widget1.setLayout(self.UIWidget)
		# # self.UIWidget.show()