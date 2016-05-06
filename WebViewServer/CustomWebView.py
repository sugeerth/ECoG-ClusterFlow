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
import simplejson
import pyqtgraph as pg
import traceback
from graphviz import Digraph

from OpenGL.GLUT import *
from OpenGL.GL import *

#include <QWebFrame>
from PySide.QtWebKit import QWebFrame

from time import time
from math import *

Default = 1420.36

url = 'http://localhost/~sugeerthmurugesan/Sankey/index.html'
url2 = 'http://localhost/~sugeerthmurugesan/Sankey/Working.html'
# url = "http://localhost/~sugeerthmurugesan/"

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
		# print "Came TO BrushTImesteps"
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

	def __init__(self):
		QtWebKit.QWebView.__init__(self)

		self.setMaximumSize(1430,210) 
		self.setMinimumSize(1430,210) 

		self.SmallMultipleLayout = None
		self.js = None

		self.setContentsMargins(0, 0, 0, 0)
		self.load(QtCore.QUrl(url))
		self.setContentsMargins(0, 0, 0, 0)

		self.signalConnection()

		frame = self.page().mainFrame()
		frame.setScrollBarPolicy(QtCore.Qt.Vertical , QtCore.Qt.ScrollBarAlwaysOn)
		frame.setScrollBarPolicy(QtCore.Qt.Horizontal , QtCore.Qt.ScrollBarAlwaysOff)

	def signalConnection(self):
		self.connect(self.page().mainFrame(), QtCore.SIGNAL("javaScriptWindowObjectCleared ()"), self.javaScriptWindowObjectCleared)
		self.settings().setAttribute(QtWebKit.QWebSettings.WebAttribute.DeveloperExtrasEnabled, True)

		# QtCore.QObject.connect(self, QtCore.SIGNAL('loadStarted ()'), self.loadStarted)
		# QtCore.QObject.connect(self, QtCore.SIGNAL('loadFinished(bool)'), self.loadFinished)
		# QtCore.QObject.connect(self, QtCore.SIGNAL('loadProgress(int)'), self.loadProgress)
		# QtCore.QObject.connect(self.page(), QtCore.SIGNAL('linkHovered(const QString&,const QString&,const QString&)'), self.linkHovered)

		# QtCore.QObject.connect(self, QtCore.SIGNAL('urlChanged(const QUrl &)'), self.urlChanged)
		QtCore.QObject.connect(self.page(), QtCore.SIGNAL('selectionChanged()'), self.selectionChanged)
		# QtCore.QObject.connect(self, QtCore.SIGNAL('titleChanged(const QString & )'), self.titleChanged)
		# QtCore.QObject.connect(self.page(), QtCore.SIGNAL('statusBarMessage(const QString&)'), self.statusBarMessage)
	
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
		 
	# def linkHovered(a, b, c):
	# 	print "link hovered!"
	# 	print a
	# 	print b
	# 	print c

	def javaScriptWindowObjectCleared(self):
		self.js = StupidClass(self)
		self.page().mainFrame().addToJavaScriptWindowObject("pyObj", self.js)
		self.page().mainFrame().evaluateJavaScript (" var MyParameters=1;")

	def BrushTimesteps(self, Start, End, StartSliceNoGlobal, EndSliceNoGlobal):
		# Display only these Timesteps in the software 
		print Start, End, StartSliceNoGlobal, EndSliceNoGlobal

	def slicesChanged(self,slices):
		print "slices are being changed please take a look"
		


				
		# print "Slices have been changed",self.SmallMultipleLayout.lengthPixels
		# self.setMaximumSize(self.SmallMultipleLayout.lengthPixels,100) 
		# self.setMinimumSize(self.SmallMultipleLayout.lengthPixels,100) 
		# self.js.SliceSignal.emit(slices, self.SmallMultipleLayout.lengthPixels)

	def EmitTimestepRanges(self, Start, End, slices):
		self.js.Start1 = Start
		self.js.IntervalSignal.emit(Start, End, slices, 0)

	def reload(self):
		self.load(QtCore.QUrl(url))
