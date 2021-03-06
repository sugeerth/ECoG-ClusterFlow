from PySide import QtWebKit
import numpy as np
import pprint
import weakref
import math
from collections import defaultdict
from PySide import QtCore, QtGui
from collections import OrderedDict
import copy
import csv
from collections import deque
from PySide.QtCore import *
import traceback

from PySide.QtWebKit import QWebFrame

from time import time
from math import *

Default = 1420.36

class StupidClass(QtCore.QObject): 
    IntervalSignal = QtCore.Signal(int, int, int, int)
    SliceSignal = QtCore.Signal(int, int)

    sendTimeStepFromSankey = QtCore.Signal(int, int, int)

    def __init__(self,WebView,startval=42):
        QtCore.QObject.__init__(self)
        self.Start= startval 
        self.WebView = WebView
        self.sendTimeStepFromSankey.connect(self.WebView.sendTimeStepFromSankey)

    @QtCore.Slot(str,str,str)  
    def showMessage(self, msg, msg2, msg3):  
        """Open a message box and display the specified message."""  
        self.sendTimeStepFromSankey.emit(int(msg),int(msg2),int(msg3))

  	@QtCore.Slot(str,str,str,str)
	def brushTimesteps(self, Start, End, StartSliceNoGlobal, EndSliceNoGlobal):
		print Start, End, StartSliceNoGlobal, EndSliceNoGlobal

    def _pyVersion(self):  
        """Return the Python version."""  
        return sys.version  
  
    def DummyCallFromJavascript(self, message):
    	print message

    def setIntervals(self, Start):
    	print "setting Stuff"
    	self.Start = Start 
	
    @QtCore.Slot(int)
    def interval1(self):
    	print "Read the value", self.Start
    	return self.Start

    def _dummy(self):
    	return "Writing it there"

    """Python interpreter version property."""  
    pyVersion = Property(str, fget=_dummy) 
    Start1= Property(int, interval1, setIntervals)
  
class CustomWebView(QtWebKit.QWebView):
	sendLCDValues = QtCore.Signal(float)
	sendTimeStepFromSankey = QtCore.Signal(int, int, int)

	def __init__(self, url):
		QtWebKit.QWebView.__init__(self)

		self.setMaximumSize(1930,210) 
		self.setMinimumSize(1930,210) 

		self.SmallMultipleLayout = None
		self.js = None

		self.setContentsMargins(0, 0, 0, 0)
		self.load(QtCore.QUrl(url))
		self.setContentsMargins(0, 0, 0, 0)

		self.signalConnection()
		self.url = url

		frame = self.page().mainFrame()
		frame.setScrollBarPolicy(QtCore.Qt.Vertical , QtCore.Qt.ScrollBarAlwaysOn)
		frame.setScrollBarPolicy(QtCore.Qt.Horizontal , QtCore.Qt.ScrollBarAlwaysOff)

	def signalConnection(self):
		self.connect(self.page().mainFrame(), QtCore.SIGNAL("javaScriptWindowObjectCleared ()"), self.javaScriptWindowObjectCleared)
		self.settings().setAttribute(QtWebKit.QWebSettings.WebAttribute.DeveloperExtrasEnabled, True)
		QtCore.QObject.connect(self.page(), QtCore.SIGNAL('selectionChanged()'), self.selectionChanged)
	
	def javaScriptConsoleMessage(self, msg, line, source):
		print '%s line %d: %s' % (source, line, msg)

	def linkHoverd(url):
		pass

	def urlChanged(url):
		pass
 
	def loadStarted():
		pass
 
	def selectionChanged():
		print "selection hovered!"

	def titleChanged(title):
		pass
 
	def statusBarMessage (text):
		pass
 
	def loadFinished(result):
		pass

	def loadProgress(prog):
		pass

	def javaScriptWindowObjectCleared(self):
		self.js = StupidClass(self)
		self.page().mainFrame().addToJavaScriptWindowObject("pyObj", self.js)
		self.page().mainFrame().evaluateJavaScript (" var MyParameters=1;")

	def BrushTimesteps(self, Start, End, StartSliceNoGlobal, EndSliceNoGlobal):
		# Display only these Timesteps in the software 
		print Start, End, StartSliceNoGlobal, EndSliceNoGlobal

	def slicesChanged(self,slices):
		print "Slices have been changed",self.SmallMultipleLayout.lengthPixels
		# self.setMaximumSize(self.SmallMultipleLayout.lengthPixels,100) 
		# self.setMinimumSize(self.SmallMultipleLayout.lengthPixels,100) 
		# self.js.SliceSignal.emit(slices, self.SmallMultipleLayout.lengthPixels)

	def EmitTimestepRanges(self, Start, End, slices):
		self.js.Start1 = Start
		self.js.IntervalSignal.emit(Start, End, slices, 0)

	def reload(self):
		self.load(QtCore.QUrl(self.url))
