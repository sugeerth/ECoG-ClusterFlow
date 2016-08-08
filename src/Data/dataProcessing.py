from PySide import QtCore, QtGui
import scipy.io
import math
import weakref
import scipy
from PIL import Image
import json
from pprint import pprint
import numpy as np
import pprint

# ElectrodeSignalDataName = 'muDat'
ElectrodeSignalDataName = 'sigData'

"""
This is the data that needs to be changed based on the format of the data
"""
Network_File='/Users/sugeerthmurugesan/ProjectRepos/OrthoProject/GraphData/karate.json'

class dataProcessing(object):
    	def __init__(self, Brain_image_filename,Electrode_ElectrodeData_filename,Electrode_mat_filename,ElectrodeSignals):
			self.im = Image.open(Brain_image_filename)
			self.syllableUnit = 0 
			self.Timestep =0

			self.ElectrodeSignals = scipy.io.loadmat(ElectrodeSignals)
			self.mat = scipy.io.loadmat(Electrode_mat_filename)
			self.connectivityData=scipy.io.loadmat(Electrode_ElectrodeData_filename)

			# Changes for artificial data 
			# Data=scipy.io.loadmat(Electrode_ElectrodeData_filename)
			# temp = Data['electrode']
			# self.ElectrodeIds = temp[0]
			# self.ElectodeData = Data['C']

			# Changes for RealData
			# print len(self.ElectodeData)
			self.ElectrodeIds = [i for i in range(len(self.ElectrodeSignals[ElectrodeSignalDataName][0]))]
			self.ElectodeData = self.connectivityData['conData']


			self.syllable, self.timestep, self.N , self.N  = np.shape(self.ElectodeData)
			self.timestep = self.timestep - 1

			""" The variables names for the new connecivity matrices, 
			C == correlation matrix 
			syllable == 6 syllables
			time = mapping between electrodes
			electrode == 58 electrodes  
			"""

