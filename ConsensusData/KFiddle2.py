import scipy.io
import math
import weakref
import scipy
import numpy as np
import pprint
import pickle
import copy
from collections import Counter

syllable = 0

TrueData0 = "ConsensusCluster44.json"
name1 = "ConsensusCluster24.json"
name2 = "AllClusterings04Heatmap.tsv" 


TrueData1 = "ConsensusCluster54.json"
name11 = "ConsensusCluster34.json" 
name12 = "AllClusterings14Heatmap.tsv" 

# timestepPartition = pickle.load(open(TrueData0))
# lastvalue = timestepPartition[0]

# temp = dict()
# for i in range(len(timestepPartition)):
# 	temp.clear()
# 	newtemp= []
# 	for key,value in timestepPartition[i].iteritems():
# 		temp.setdefault(value, [])
# 		temp[value].append(key)

# 	timestepPartition[i] = copy.deepcopy(temp)

# AllClsuters = pickle.load(open(name2))

# print max(timestepPartition[5].values())
# print AllClsuters[5][3]

# timestepPartition[5] = copy.deepcopy(AllClsuters[5][3])

# print max(timestepPartition[7].values())

# timestepPartition[5] = copy.deepcopy(AllClsuters[5][3])
# timestepPartition[6] = copy.deepcopy(AllClsuters[6][5])
# timestepPartition[7] = copy.deepcopy(AllClsuters[7][4])

# # timestepPartition[22] = copy.deepcopy(AllClsuters[22][3])

# # timestepPartition[23] = copy.deepcopy(AllClsuters[23][4])
# timestepPartition[22] = copy.deepcopy(AllClsuters[22][4])
# timestepPartition[21] = copy.deepcopy(AllClsuters[21][5])


# timestepPartition[23] = copy.deepcopy(AllClsuters[23][3])
# timestepPartition[24] = copy.deepcopy(AllClsuters[24][3])
# # timestepPartition[28] = copy.deepcopy(AllClsuters[28][2])
# timestepPartition[27] = copy.deepcopy(AllClsuters[27][4])
# timestepPartition[28] = copy.deepcopy(AllClsuters[28][3])
# timestepPartition[33] = copy.deepcopy(AllClsuters[33][3])
# timestepPartition[36] = copy.deepcopy(AllClsuters[36][2])
# timestepPartition[39] = copy.deepcopy(AllClsuters[39][4])
# timestepPartition[40] = copy.deepcopy(AllClsuters[40][7])
# timestepPartition[41] = copy.deepcopy(AllClsuters[41][3])
# timestepPartition[42] = copy.deepcopy(AllClsuters[42][2])
# timestepPartition[44] = copy.deepcopy(AllClsuters[44][3])
# # timestepPartition[61] = copy.deepcopy(AllClsuters[40][7])

# with open(name1, 'w') as outfile:
#     pickle.dump(timestepPartition, outfile)

timestepPartition = pickle.load(open(TrueData1))
AllClsuters = pickle.load(open(name12))
print max(timestepPartition[5].values())
print AllClsuters[5][3]
timestepPartition[1] = copy.deepcopy(AllClsuters[1][3])
# print max(timestepPartition[7].values())

timestepPartition[3] = copy.deepcopy(AllClsuters[3][4])


timestepPartition[8] = copy.deepcopy(AllClsuters[8][3])

timestepPartition[10] = copy.deepcopy(AllClsuters[10][4])

timestepPartition[14] = copy.deepcopy(AllClsuters[14][3])

timestepPartition[24] = copy.deepcopy(AllClsuters[24][3])

timestepPartition[3] = copy.deepcopy(AllClsuters[3][4])
timestepPartition[4] = copy.deepcopy(AllClsuters[4][4])

timestepPartition[5] = copy.deepcopy(AllClsuters[5][4])
timestepPartition[6] = copy.deepcopy(AllClsuters[6][3])

timestepPartition[16] = copy.deepcopy(AllClsuters[16][5])
timestepPartition[17] = copy.deepcopy(AllClsuters[17][4])
timestepPartition[18] = copy.deepcopy(AllClsuters[18][4])


timestepPartition[23] = copy.deepcopy(AllClsuters[23][4])

timestepPartition[25] = copy.deepcopy(AllClsuters[25][4])

timestepPartition[29] = copy.deepcopy(AllClsuters[29][4])

timestepPartition[27] = copy.deepcopy(AllClsuters[27][3])

# timestepPartition[25] = copy.deepcopy(AllClsuters[25][4])

timestepPartition[31] = copy.deepcopy(AllClsuters[31][3])
timestepPartition[32] = copy.deepcopy(AllClsuters[32][3])
timestepPartition[37] = copy.deepcopy(AllClsuters[37][4])
timestepPartition[38] = copy.deepcopy(AllClsuters[38][4])


timestepPartition[56] = copy.deepcopy(AllClsuters[56][4])

# timestepPartition[5] = copy.deepcopy(AllClsuters[5][3])
# timestepPartition[6] = copy.deepcopy(AllClsuters[6][5])
# timestepPartition[7] = copy.deepcopy(AllClsuters[7][4])

# # timestepPartition[22] = copy.deepcopy(AllClsuters[22][3])

# # timestepPartition[23] = copy.deepcopy(AllClsuters[23][4])
# timestepPartition[22] = copy.deepcopy(AllClsuters[22][4])
# timestepPartition[21] = copy.deepcopy(AllClsuters[21][5])

# timestepPartition[23] = copy.deepcopy(AllClsuters[23][3])
# timestepPartition[24] = copy.deepcopy(AllClsuters[24][3])
# timestepPartition[28] = copy.deepcopy(AllClsuters[28][2])
# timestepPartition[27] = copy.deepcopy(AllClsuters[27][3])
# timestepPartition[28] = copy.deepcopy(AllClsuters[28][3])
# timestepPartition[33] = copy.deepcopy(AllClsuters[33][3])
# timestepPartition[36] = copy.deepcopy(AllClsuters[36][3])
# timestepPartition[39] = copy.deepcopy(AllClsuters[39][3])
# timestepPartition[41] = copy.deepcopy(AllClsuters[41][3])
# timestepPartition[42] = copy.deepcopy(AllClsuters[42][2])
# timestepPartition[44] = copy.deepcopy(AllClsuters[44][3])
# timestepPartition[60] = copy.deepcopy(AllClsuters[40][7])


with open(name11, 'w') as outfile:
    pickle.dump(timestepPartition, outfile)


# print max(AllClsuters[0][2].values())
# print AllClsuters[0][2]
# print AllClsuters[0][2].values()
# for myKey in AllClsuters[0][2].keys():
	# print "{0}: {1}".format(myKey, len([AllClsuters[0][2][myKey]]))