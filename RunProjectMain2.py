"""
Setting the PYTHONPATH and then running the python 

os.environ["PYTHONPATH"] = "<PATH_TO_YOUR_GITHUB_REPO>/ECoG-ClusterFlow/src"
os.system("python <PATH_TO_YOUR_GITHUB_REPO>/ECoG-ClusterFlow/src/BrainViewer.py")
"""
# os.environ["PYTHONPATH"] = "/Users/sugeerthmurugesan/LBLProjects/workingCopy/gitUptoDateRepo/ECoG-ClusterFlow/src"
import os
import os.path as path
import sys

CURR =  path.abspath(path.join(__file__ ,"..")) # going one directory up 
CURR = os.path.join(CURR, "src/BrainViewer.py")

os.system("python "+str(CURR))
