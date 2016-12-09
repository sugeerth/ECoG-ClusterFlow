import csv
import sys
import math
from collections import defaultdict
from PySide import QtCore, QtGui
from sys import platform as _platform
import weakref
import cProfile
import os

class QuantData(QtCore.QObject):
	DataChange = QtCore.Signal(bool)
	def __init__(self,widget):
		super(QuantData,self).__init__()
		data = widget.correlationTableObject
		self.widget = widget
		self.BrainRegions = data.RegionName[0]
		self.data_list = []
		self.header = ['Regions', 'Centrality','Participation','Betweenness']

	def ThresholdChange(self,State):
		self.data_list = []

		self.DataChange.emit(True)

	def getHeader(self):
		return self.header

	def getData_list(self):
		return self.data_list