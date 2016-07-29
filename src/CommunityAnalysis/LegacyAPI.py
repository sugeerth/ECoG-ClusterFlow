from collections import defaultdict
from collections import OrderedDict
import copy
from collections import deque
import simplejson
import traceback
from graphviz import Digraph

from time import time
from math import *

"""
Stuff to do include 
1) Colors like communties
2) interactive communities

"""

class LegacyAPI(object):
	"""docstring for LegacyAPI"""
	def __init__(self, arg):
		super(LegacyAPI, self).__init__()
		self.arg = arg

	def indexOf(self,ColumnList, Assignment):
		k=0
		for k,i in enumerate(ColumnList):
			if (k in Assignment.values()):
				continue
			if i > max1:
				max1 = i
				index = k 
		return k

	def keywithmaxval(self,d):
		 """ a) create a list of the dict's keys and values; 
			 b) return the key with the max value"""  
		 v=list(d.values())
		 k=list(d.keys())
		 return k[v.index(max(v))]

	def writeToaFile(self, AggregateList, dot1):
		f =open( "PartitionData.txt", "wb" )
		dot1.render('test-output/round-table.gv', view=True)
		import simplejson
		simplejson.dump(AggregateList ,f) 
		f.close()

	def SetDeltaOperation(self,node1, node2):
		"""
		node1 is from A communityMultiple and node2 is from Setb tow partition
		"""
		node1Set = set(node1) 
		node2Set = set(node2)
		differenceA = node1Set - node2Set 
		differenceB = node2Set - node1Set
		elements = len(differenceA | differenceB)
		return elements

	def max1(self, ColumnList, Assignment):
		k=0
		max1 = -3 
		index = 0
		for k,i in enumerate(ColumnList):
			if (k in Assignment.values()):
				continue
			if i > max1:
				max1 =i 
				index = k
		return max1, index

	def CalculateProportionScore(self,ColumnList,Assignment,Max):
		sum1 = self.sum1(ColumnList,Assignment)
		if not(sum1 == 0): 
			ProportionScore = Max/sum1
		else: 
			ProportionScore = 0
		return ProportionScore

	def sum1(self,List,Assignment):
		sum1 = 0
		k = 0 
		for k,i in enumerate(List):
			if i == -1: 
				continue
			if (k in Assignment.values()):
				continue
			sum1 = sum1 + i
		return sum1

	def AssignValues(self,ColumnList,Max,Index,Assignment,columnIndex,matrix):
		"""My own score comparison with other scores"""
 		if len(Assignment.values()) == self.height:
 			self.ReEnterColumnLoop = False
 			return  
 		ProportionScore = self.CalculateProportionScore(ColumnList,Assignment,Max)
 		# print "My Proportion Score", ProportionScore
 
 		element = self.MaxRightHorizontalLines(columnIndex,Index,Assignment,matrix,Max,ProportionScore)
 		# print "element",element, type(element)
 		# print "Max values",element.values()
 
 		try: 
 			if ProportionScore > max(element.values()):
 				# print "Index ",Max , Index, columnIndex
 				Assignment[columnIndex] = Index 
 				self.ReEnterColumnLoop = False
 			else: 
 				# print "Entering"
 				# print "Values is ",max(element.values())
 				column = max(element, key=element.get)
 				# column = element.keys().index(max(element.values()))
 				# print "Found better values so going for it", Index,column,"This is the element table", element, "This is proportional value",max(element.values())
 				Assignment[column] = Index	
 				self.ReEnterColumnLoop = True
		except ValueError:
			# print "Enter here"
			Assignment[columnIndex] = Index 
			self.ReEnterColumnLoop = False
	"""
	1 2 3
	|------------leftIndex,rightIndex , 3--MaxValue 
	3 4 5
	4 2 1
	"""
	def MaxRightHorizontalLines(self,leftIndex,rightIndex,AssignmentValues,matrix, MaxValue,ProportionScore):
		MaxV = -3
		Element = dict()
		index = 0

		for i in range(leftIndex+1,self.width): 
			if (i in AssignmentValues.keys()):
				continue
			MaxV = -3 

			if matrix[rightIndex][i] >= MaxValue: 
				Column = matrix[:,i]
				ColumnList = Column.tolist()
				# print "Column",Column

				MaxE, index= self.max1(ColumnList,AssignmentValues)
				# index = ColumnList.index(MaxE)

				Value = self.CalculateProportionScore(ColumnList,AssignmentValues,matrix[rightIndex][i])
				# print "index -->",index,"Max element-->", MaxE,"But ",matrix[rightIndex][i],"Proportional value",Value,"in [",rightIndex,"][",i,"]"
				Element[i] = Value
				# Element[index] = MaxE

		return Element

	def PrintAccumalatedData(self,state):
		"""Plot a graph that will mapp all the stability values"""
		print self.stabilityValues  
