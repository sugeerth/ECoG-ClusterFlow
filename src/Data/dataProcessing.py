
from PySide import QtCore, QtGui
import scipy.io
import math
import weakref
import scipy
from PIL import Image
import numpy as np
import pprint

ElectrodeSignalDataName = 'sigData'
ElectrodeConnectivityDataName = 'conDat'

# Changes for RealData
Connectvity_filename ='/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerEpilepsyData/enhancedConData.mat'

class dataProcessing(object):
	    def __init__(self, Brain_image_filename,Electrode_ElectrodeData_filename,Electrode_mat_filename,ElectrodeSignals,ElectodeData):
			self.im = Image.open(Brain_image_filename)
			self.syllableUnit = 0 
			self.Timestep =0

			self.ElectrodeSignals = scipy.io.loadmat(ElectrodeSignals)
			Data=scipy.io.loadmat(Electrode_ElectrodeData_filename)

			self.mat = scipy.io.loadmat(Electrode_mat_filename)

			temp = Data['electrode']
			self.ElectrodeIds = temp[0]
			self.ElectodeData = Data['C']

			self.syllable, self.timestep, self.N , self.N  = np.shape(self.ElectodeData)
			self.timestep = self.timestep - 1

			""" The variables names for the new connecivity matrices, 
			C == correlation matrix 
			syllable == 6 syllables

			time = mapping between electrodes
			electrode == 58 electrodes  
			"""