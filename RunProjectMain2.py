"""
Setting the PYTHONPATH and then running the python 

os.environ["PYTHONPATH"] = "<PATH_TO_YOUR_GITHUB_REPO>/ECoG-ClusterFlow/src"
os.system("python <PATH_TO_YOUR_GITHUB_REPO>/ECoG-ClusterFlow/src/BrainViewer.py")
"""
import os,sys

os.environ["PYTHONPATH"] = "/Users/sugeerthmurugesan/LBLProjects/workingCopy/gitUptoDateRepo/ECoG-ClusterFlow/src"
os.system("python /Users/sugeerthmurugesan/LBLProjects/workingCopy/gitUptoDateRepo/ECoG-ClusterFlow/src/BrainViewer.py")