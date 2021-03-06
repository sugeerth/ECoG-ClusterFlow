from PySide import QtCore, QtGui , QtUiTools
import os
import sys

#Loading UI Files
loader = QtUiTools.QUiLoader()

CURR = sys.path[0]
CURR+='/UIFiles'

ui = loader.load(os.path.join(CURR, "interface.ui"))
dataSetLoader = loader.load(os.path.join(CURR, "datasetviewer.ui"))
screenshot = loader.load(os.path.join(CURR, "screeshot.ui"))
electrodeUI = loader.load(os.path.join(CURR, "electrodeui.ui"))
AcrossTimestep = loader.load(os.path.join(CURR, "hgkjh.ui"))
Visualizer = loader.load(os.path.join(CURR, "visualizer.ui"))
Visualizer.setWindowTitle('CheckBox')

def setTitle():
    Visualizer.setWindowTitle('Brain Visualizer')

setTitle()