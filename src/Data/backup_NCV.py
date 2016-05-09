
from PySide import QtCore, QtGui
import scipy.io
import math
import weakref
import scipy
from PIL import Image
import numpy as np

class dataProcessing(object):
	    def __init__(self, Brain_image_filename,SelectedElectrodes_filename,Electrode_ElectrodeData_filename,Electrode_mat_filename,ElectrodeSignals, Electrode_Ids_filename , Electrode_data_filename):
			self.im = Image.open(Brain_image_filename)
			self.syllableUnit = 0 
			self.timestep =0

			Data=scipy.io.loadmat(Electrode_ElectrodeData_filename)
			self.mat = scipy.io.loadmat(Electrode_mat_filename)

			self.ElectrodeIds2 = scipy.io.loadmat(Electrode_Ids_filename)
			self.ElectodeData = scipy.io.loadmat(Electrode_data_filename)

			self.ElectodeData= scipy.io.loadmat(Electrode_data_filename)
			self.ElectrodeIds = Data['electrode']

			print self.ElectrodeIds2['electids'] , self.ElectrodeIds , 
			Index = []
			k = 0 
			for index,data in enumerate(self.ElectrodeIds2['electids'][0]):
				for index2, data2 in enumerate(self.ElectrodeIds[0]):
						if data == data2:
							Index.append(index)

			print "INdex data",Index
			self.ElectodeData = self.ElectodeData['mat_ncv']


			newData = np.zeros((6,64,54,54))
			for i in range(6):
				for j in range(64):
					for k in range(54):
						for l in range(54):
							newData[i][j][k][l] = self.ElectodeData[i][j][Index[k]][Index[l]]  

			# print self.ElectodeDa/ta[1][4]

			# print "old data", self.ElectodeData[1][2][Index[0]][Index[1]]
			# print "new data", newData[1][2][0][1]
			# print np.matrix.sum(np./subtract(self.ElectodeData, newData))

			self.ElectodeData = np.copy(newData)
			# print 

			# self.ElectrodeIds = Data['electrode']
			# self.ElectodeData = Data['C']

			self.ElectrodeSignals = scipy.io.loadmat(ElectrodeSignals)
			
			assert len(self.ElectrodeIds[0]) == 54,  "The length is s(%d)" % len(self.ElectrodeIds[0])

			""" The variables names for the new connecivity matrices, 
			C == correlation matrix 
			syllable == 6 syllables

			time = mapping between electrodes
			electrode == 58 electrodes  
			"""





