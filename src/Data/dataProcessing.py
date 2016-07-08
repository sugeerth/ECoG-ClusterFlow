
import json
from pprint import pprint
import numpy as np
import pprint

"""
This is the data that needs to be changed based on the format of the data
"""
Network_File='/Users/sugeerthmurugesan/ProjectRepos/OrthoProject/GraphData/karate.json'

class dataProcessing(object):
    def __init__(self):
		with open(Network_File) as data_file:    
	    	self.data = json.load(data_file)

