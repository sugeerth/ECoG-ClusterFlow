# Artificial Data 
import os
import os.path as path
import sys

CURR =  path.abspath(path.join(__file__ ,"..")) # going one directory up 
CURR = os.path.join(CURR, "SyntheticGeneratedData")

Electrode_mat_filename = os.path.join(CURR, "EC2.reg_grd.mat")
Brain_image_filename = os.path.join(CURR,"EC2.brain_redo1.jpg")
ElectrodeSignals =os.path.join(CURR,"SyntheticElectric.mat")
Electrode_ElectrodeData_filename = os.path.join(CURR,"Bolasso_output_algnd.mat") 

url =  'http://localhost/Sankey/artificialWorking.html'

FileNames = [('/Users/sugeerthmurugesan/Sites/Sankey/JSON_1.json',0,12),('/Users/sugeerthmurugesan/Sites/Sankey/JSON_2.json',12,24)\
,('/Users/sugeerthmurugesan/Sites/Sankey/JSON_3.json',24,36),('/Users/sugeerthmurugesan/Sites/Sankey/JSON_4.json',36,48)\
,('/Users/sugeerthmurugesan/Sites/Sankey/JSON_5.json',48,60)]

HeatmapFilename = "/Users/sugeerthmurugesan/Sites/Sankey/DeltaAreaChange4Heatmap.tsv"

#set the flag data around here for convenience
GraphWindowShowFlag = False
MainWindowShowFlag = True
ElectrodeWindowShowFlag = False
CorrelationTableShowFlag = False
debugTrackingView = False