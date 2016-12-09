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

#Real dataset to be visualzed 
# Electrode_mat_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerEpilepsyData/regData.mat'
# Brain_image_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerEpilepsyData/ec58_blank_clean_enhanced.png'
# ElectrodeSignals = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerEpilepsyData/enhancedSigData.mat'
# Electrode_ElectrodeData_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerEpilepsyData/enhancedConData.mat' 


# Electrode_mat_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerProject/KrisDataset/EC2.reg_grd.mat'
# Brain_image_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerProject/KrisDataset/EC2.brain_redo1.jpg'
# ElectrodeSignals ='/Users/sugeerthmurugesan/LBLProjects/workingCopy/brain-visWorkingRealData/SyntheticGeneratedData/SyntheticElectric.mat'
# # Electrode_ElectrodeData_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerProject/KrisDataset/Bolasso_output_algnd.mat' 
# Electrode_ElectrodeData_filename = '/Users/sugeerthmurugesan/LBLProjects/ELectrode/SummerProject/KrisDataset/Bolasso_output_algnd.mat' 

#set the flag data around here for convenience
GraphWindowShowFlag = False
MainWindowShowFlag = True
ElectrodeWindowShowFlag = False
CorrelationTableShowFlag = False
debugTrackingView = False