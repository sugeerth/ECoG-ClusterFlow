"""
The script generates artificaial data required for visualization in the python initiated tool 

The funtions are very simple, that mentions the number of syllables to be generated, and the 
number of functionals to be generated
"""


import csv
import sys
import math
import datetime
from collections import defaultdict
from collections import OrderedDict
from sys import platform as _platform
# from PIL import Image
import weakref
import cProfile
import pprint
import os
import scipy.io
import random
import numpy as np

#3Clusters 
BAA = 0 
BOO = 1 
DAA = 2 
DOO = 3 
PAA = 4
POO = 5

x = [i for i in range(20)]
y = [i for i in range(21,39)]
z = [i for i in range(40,64)]

ClusterStart = [[0,10,20,30], [2,9,11], [8,12,13], [1,12,14], [6,7,10], [1,15,20]] 

Number_of_Electrode = 16
timeStep= 64
X = 10
Y = 20
Z = 35

XRes = 1301 
YRes = 861

LOW = float(.3)
MED = float(.5)
HIGH = float(.9)

syllable = [0] 

Electrode_mat_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerProject/KrisDataset/EC2.reg_grd.mat'
Electrode_data_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerProject/KrisDataset/NCV_mat_bic.mat'
Brain_image_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerProject/KrisDataset/EC2.brain_redo1.jpg'
Electrode_Ids_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerProject/KrisDataset/EC2.grphElectIDs.mat' 
SelectedElectrodes_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerProject/KrisDataset/SelectedElectrodes.txt' 
ElectrodeSignals = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerProject/KrisDataset/EC2.babodabopapo.muDat.mat'
Electrode_ElectrodeData_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerProject/KrisDataset/Bolasso_output_algnd.mat' 

class DataModel(object):
    def __init__(self,projectname):
        super(DataModel,self).__init__()
        self.projectname = projectname
        # Data=scipy.io.loadmat(i)
        self.N = Number_of_Electrode
        sq = self.N * self.N
        self.adjacencyMatrix = np.zeros((timeStep,sq,sq))
        self.Electrode = np.zeros((6,sq,timeStep))

        Data=scipy.io.loadmat(Electrode_ElectrodeData_filename)
        temp = Data['electrode']
        self.ElectrodeIds = temp[0]

        # self.x = []
        # self.y = []
        # self.z = []
        # self.x = [int(self.ElectrodeIds[i]) for i in range(20)]
        # self.y = [int(self.ElectrodeIds[i]) for i in range(21,39)]
        # self.z = [int(self.ElectrodeIds[i]) for i in range(40,54)]


        self.x = [i for i in range(20)]
        self.y = [i for i in range(21,39)]
        self.z = [i for i in range(40,64)]

        self.start1 = [i for i in range(6)]


        self.sq = sq
        # self.SeeElectrode_ElectrodeData_filename()

        # self.fillAdjacencyMatrixBaa()
        self.ElectrodeSignalsBaa(sq)

        # self.fillAdjacencyMatrixBoo()
        # self.ElectrodeSignalsBoo(sq)

        # self.fillAdjacencyMatrixDaa()
        # self.ElectrodeSignalsDaa(sq)

        # self.fillAdjacencyMatrixDoo()
        # self.ElectrodeSignalsDoo(sq)

        # self.fillAdjacencyMatrixPaa()
        # self.ElectrodeSignalsPaa(sq)

        # self.fillAdjacencyMatrixPoo()
        # self.ElectrodeSignalsPoo(sq)

        # self.writeData(sq)
        self.writeSignalData()

        # self.ElectrodeSignals(sq)
        # self.GenerateDataSets()
        # self.ElectrodePositions(self.N)
        # name = "SynthentheticElectric.mat"

        # name = "SynthentheticElectricPos.mat"
        # self.readMatFile(name)

    def ElectrodePositions(self,sq):
        name = "SyntheticGeneratedData/SynthentheticElectricPos.mat"
        # Data=scipy.io.loadmat(Electrode_mat_filename)
        Data = np.zeros((2,64))
        
        Mid = (XRes/2 , YRes/2)
        XStart= XRes/4
        XEnd = Mid[0] + XRes/4  
        XRange = (XEnd - XStart)/Number_of_Electrode

        YStart = YRes/4
        YEnd = Mid[1] + YRes/4 

        YRange = (YEnd - YStart)/Number_of_Electrode

        k=0
        for i in range(sq):
            XStart = XRes/4 
            for j in range(sq):
                Data[0][k] = XStart
                XStart = XStart+YRange
                Data[1][k] = YStart
                k=k+1
            YStart = YStart + YRange
        print Data

        xy = [Data]
        scipy.io.savemat(name, mdict={'xy': Data})

    def ElectrodeSignalsBaa(self, sq):
        Data=scipy.io.loadmat(ElectrodeSignals)
        i = BAA
        for j in range(sq):
            for k in range(timeStep):
                if k >= ClusterStart[0][0] and k <= ClusterStart[0][1]:
                    if j in self.x: 
                        self.Electrode[i][j][k] = LOW
                    elif j in self.y: 
                        self.Electrode[i][j][k] = LOW
                    elif j in self.z: 
                        self.Electrode[i][j][k] = LOW
                elif k >= ClusterStart[0][2] and k <= ClusterStart[0][3]:
                    if j in self.x: 
                        self.Electrode[i][j][k] = HIGH
                    elif j in self.y: 
                        self.Electrode[i][j][k] = HIGH
                    elif j in self.z: 
                        self.Electrode[i][j][k] = HIGH
                else:
                    if j in self.x: 
                        if j in self.start1:
                            self.Electrode[i][j][k] = HIGH
                        else:
                            self.Electrode[i][j][k] = LOW
                    else: 
                        self.Electrode[i][j][k] = LOW
                    # self.start.append(self.start[-1]+1)
                    # print self.start[-1]
    def ElectrodeSignalsBoo(self, sq):
        Data=scipy.io.loadmat(ElectrodeSignals)
        i = BOO
        for j in range(sq):
            for k in range(timeStep):
                if k > ClusterStart[1][0] and k < ClusterStart[1][1]:
                    if j in self.x: 
                        self.Electrode[i][j][k] = MED
                    elif j in self.y: 
                        self.Electrode[i][j][k] = LOW
                    elif j in self.z: 
                        self.Electrode[i][j][k] = HIGH
                else: 
                    self.Electrode[i][j][k] = LOW

    def ElectrodeSignalsDaa(self, sq):
        Data=scipy.io.loadmat(ElectrodeSignals)
        i = DAA
        for j in range(sq):
            for k in range(timeStep):
                if k > ClusterStart[2][0] and k < ClusterStart[2][1]:
                    if j in self.x: 
                        self.Electrode[i][j][k] = MED
                    elif j in self.y: 
                        self.Electrode[i][j][k] = HIGH
                    elif j in self.z: 
                        self.Electrode[i][j][k] = LOW
                else: 
                    self.Electrode[i][j][k] = LOW

    def ElectrodeSignalsDoo(self, sq):
        Data=scipy.io.loadmat(ElectrodeSignals)
        i = DOO
        for j in range(sq):
            for k in range(timeStep):
                if k > ClusterStart[3][0] and k < ClusterStart[3][1]:
                    if j in self.x: 
                        self.Electrode[i][j][k] = MED
                    elif j in self.y: 
                        self.Electrode[i][j][k] = HIGH
                    elif j in self.z: 
                        self.Electrode[i][j][k] = LOW
                else: 
                    self.Electrode[i][j][k] = LOW

    def ElectrodeSignalsPaa(self, sq):
        Data=scipy.io.loadmat(ElectrodeSignals)
        i = PAA
        for j in range(sq):
            for k in range(timeStep):
                if k > ClusterStart[4][0] and k < ClusterStart[4][1]:
                    if j in self.x: 
                        self.Electrode[i][j][k] = MED
                    elif j in self.y: 
                        self.Electrode[i][j][k] = HIGH
                    elif j in self.z: 
                        self.Electrode[i][j][k] = LOW
                else: 
                    self.Electrode[i][j][k] = LOW

    def ElectrodeSignalsPoo(self, sq):
        Data=scipy.io.loadmat(ElectrodeSignals)
        i = POO
        for j in range(sq):
            for k in range(timeStep):
                if k > ClusterStart[5][0] and k < ClusterStart[5][1]:
                    if j in self.x: 
                        self.Electrode[i][j][k] = MED
                    elif j in self.y: 
                        self.Electrode[i][j][k] = HIGH
                    elif j in self.z: 
                        self.Electrode[i][j][k] = LOW
                else: 
                    self.Electrode[i][j][k] = LOW

    def writeSignalData(self): 
        name = "SyntheticGeneratedData/SyntheticElectric.mat"
        muDat = [self.Electrode]
        pprint.pprint(self.Electrode[0][109][0])
        scipy.io.savemat(name, mdict={'muDat': self.Electrode})

    def GenerateDataSets(self):
        self.GenerateBrainFileFormat()
        self.SeeElectrode_ElectrodeData_filename()

    def SeeElectrode_ElectrodeData_filename(self):
        Data=scipy.io.loadmat(Electrode_ElectrodeData_filename)

    def fillAdjacencyMatrixBaa(self):
        """
        matrix ordering
        """
        first = True
        sq = self.N * self.N
        i = BAA
        for j in range(timeStep): 
            k = l = m = n = o = p = 0
            if j > ClusterStart[0][0] and j < ClusterStart[0][1]:
                q=0 
                for k in range(X):
                    if first: 
                        self.x.append(q)
                        q=q+1
                    for l in range(X): 
                        self.adjacencyMatrix[i][j][k][l] = 0.5
                for m in range(1,Y+1):
                    if first:
                        self.y.append(q)
                        q=q+1
                    for n in range(1,Y+1):
                        self.adjacencyMatrix[i][j][k+m][l+n] = 0.5 
                for o in range(1,Z+1):
                    if first:
                        self.z.append(q)
                        q=q+1
                    for p in range(1,Z+1): 
                        self.adjacencyMatrix[i][j][k+m+o][l+n+p] = 0.5
                if first:
                    first =False

            else:  
                for q in range(sq):
                    for r in range(sq):
                        self.adjacencyMatrix[i][j][q][r] = random.uniform(0,1)

        pprint.pprint(self.adjacencyMatrix[i][5])
        # print "x=",self.x
        # print "y=",self.y
        # print "z=",self.z

    def fillAdjacencyMatrixBoo(self):
        """
        matrix ordering
        """
        first = True
        sq = self.N * self.N
        i = BOO
        for j in range(timeStep): 
            k = l = m = n = o = p = 0
            if j > ClusterStart[1][0] and j < ClusterStart[1][1]:
                q=0 
                for k in range(X):
                    if first: 
                        self.x.append(q)
                        q=q+1
                    for l in range(X): 
                        self.adjacencyMatrix[i][j][k][l] = 0.7
                for m in range(1,Y+1):
                    if first:
                        self.y.append(q)
                        q=q+1
                    for n in range(1,Y+1):
                        self.adjacencyMatrix[i][j][k+m][l+n] = 0.7
                for o in range(1,Z+1):
                    if first:
                        self.z.append(q)
                        q=q+1
                    for p in range(1,Z+1): 
                        self.adjacencyMatrix[i][j][k+m+o][l+n+p] = 0.7
                if first:
                    first =False
            else:  
                for q in range(sq):
                    for r in range(sq):
                        self.adjacencyMatrix[i][j][q][r] = random.uniform(0,1)

    def fillAdjacencyMatrixDaa(self):
        """
        matrix ordering
        """
        first = True
        sq = self.N * self.N
        i = DAA
        for j in range(timeStep): 
            k = l = m = n = o = p = 0
            if j > ClusterStart[2][0] and j < ClusterStart[0][1]:
                q=0 
                for k in range(X):
                    if first: 
                        self.x.append(q)
                        q=q+1
                    for l in range(X): 
                        self.adjacencyMatrix[i][j][k][l] = 0.1
                for m in range(1,Y+1):
                    if first:
                        self.y.append(q)
                        q=q+1
                    for n in range(1,Y+1):
                        self.adjacencyMatrix[i][j][k+m][l+n] = 0.1
                for o in range(1,Z+1):
                    if first:
                        self.z.append(q)
                        q=q+1
                    for p in range(1,Z+1): 
                        self.adjacencyMatrix[i][j][k+m+o][l+n+p] = 0.1
                if first:
                    first =False
            else:  
                for q in range(sq):
                    for r in range(sq):
                        self.adjacencyMatrix[i][j][q][r] = random.uniform(0,1)

    def fillAdjacencyMatrixDoo(self):
        """
        matrix ordering
        """
        first = True
        sq = self.N * self.N
        i = DOO
        for j in range(timeStep): 
            k = l = m = n = o = p = 0
            if j > ClusterStart[3][0] and j < ClusterStart[3][1]:
                q=0 
                for k in range(X):
                    if first: 
                        self.x.append(q)
                        q=q+1
                    for l in range(X): 
                        self.adjacencyMatrix[i][j][k][l] = 0.7
                for m in range(1,Y+1):
                    if first:
                        self.y.append(q)
                        q=q+1
                    for n in range(1,Y+1):
                        self.adjacencyMatrix[i][j][k+m][l+n] = 0.7
                for o in range(1,Z+1):
                    if first:
                        self.z.append(q)
                        q=q+1
                    for p in range(1,Z+1): 
                        self.adjacencyMatrix[i][j][k+m+o][l+n+p] = 0.7
                if first:
                    first =False
            else:  
                for q in range(sq):
                    for r in range(sq):
                        self.adjacencyMatrix[i][j][q][r] = random.uniform(0,1)

    def fillAdjacencyMatrixPaa(self):
        """
        matrix ordering
        """
        first = True
        sq = self.N * self.N
        i = PAA
        for j in range(timeStep): 
            k = l = m = n = o = p = 0
            if j > ClusterStart[4][0] and j < ClusterStart[4][1]:
                q=0 
                for k in range(X):
                    if first: 
                        self.x.append(q)
                        q=q+1
                    for l in range(X): 
                        self.adjacencyMatrix[i][j][k][l] = 0.7
                for m in range(1,Y+1):
                    if first:
                        self.y.append(q)
                        q=q+1
                    for n in range(1,Y+1):
                        self.adjacencyMatrix[i][j][k+m][l+n] = 0.7
                for o in range(1,Z+1):
                    if first:
                        self.z.append(q)
                        q=q+1
                    for p in range(1,Z+1): 
                        self.adjacencyMatrix[i][j][k+m+o][l+n+p] = 0.7
                if first:
                    first =False
            else:  
                for q in range(sq):
                    for r in range(sq):
                        self.adjacencyMatrix[i][j][q][r] = random.uniform(0,1)

    def fillAdjacencyMatrixPoo(self):
        """
        matrix ordering
        """
        first = True
        sq = self.N * self.N
        i = POO
        for j in range(timeStep): 
            k = l = m = n = o = p = 0
            if j > ClusterStart[5][0] and j < ClusterStart[5][1]:
                q=0 
                for k in range(X):
                    if first: 
                        self.x.append(q)
                        q=q+1
                    for l in range(X): 
                        self.adjacencyMatrix[i][j][k][l] = 0.7
                for m in range(1,Y+1):
                    if first:
                        self.y.append(q)
                        q=q+1
                    for n in range(1,Y+1):
                        self.adjacencyMatrix[i][j][k+m][l+n] = 0.7
                for o in range(1,Z+1):
                    if first:
                        self.z.append(q)
                        q=q+1
                    for p in range(1,Z+1): 
                        self.adjacencyMatrix[i][j][k+m+o][l+n+p] = 0.7
                if first:
                    first =False
            else:  
                for q in range(sq):
                    for r in range(sq):
                        self.adjacencyMatrix[i][j][q][r] = random.uniform(0,1)

    def writeData(self,sq):
        name = "SyntheticGeneratedData/SyntheticCorrelationData.mat"
        electrod = [ i for i in range(sq)]
        electrode = [electrod]
        obj_arr = np.zeros((2,), dtype=np.object)
        obj_arr[0] = electrode[0]
        scipy.io.savemat(name, mdict={'electrode': obj_arr[0], 'C': self.adjacencyMatrix})

    def GenerateBrainFileFormat(self):
        Data=scipy.io.loadmat(Electrode_mat_filename)
        self.im = Image.open(Brain_image_filename)
        self.TotalX, self.TotalY = np.shape(self.im)
        foo = np.array(self.im)
        
        # Compute the median of the non-zero elements
        m = np.median(foo[foo > -13])

        # Assign the median to the zero elements 
        foo[foo >= 0] = m
        scipy.misc.imsave('SyntheticGeneratedData/BrainBackground.jpg', foo)

    def readMatFile(self, name):
        Data=scipy.io.loadmat(name)
        File = Data
        print np.shape(File['xy'])
        print File['xy']

dm = DataModel('qpid')






