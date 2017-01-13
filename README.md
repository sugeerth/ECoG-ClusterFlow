# Hierarchical Spatio-Temporal Visual Analysis of ECoG Data #
Please click to watch the overview video.
 [![ScreenShot](https://github.com/sugeerth/ECoG-ClusterFlow/blob/FinalWorkingTool/src/Images/Synthetic.png)](https://vimeo.com/175328739)
 
<!--  [![ScreenShot](https://github.com/sugeerth/ECoG-ClusterFlow/blob/FinalWorkingTool/src/Images/Uload.png)](https://vimeo.com/175328739)
======= -->

# ECoG ClusterFlow #
We present ECoG ClusterFlow, an interactive visual analysis system for the exploration of high-resolution Electrocorticography (ECoG) data. Our system detects and visualizes dynamic high-level structures, such as communities, using the time-varying spatial connectivity network derived from the high-resolution ECoG data. ECoG ClusterFlow provides a multi-scale visualization of the spatio-temporal patterns underlying the time-varying communities using two views: 1) an overview summarizing the evolution of clusters over time and 2) a hierarchical glyph-based technique that uses data aggregation and small multiples techniques to visualize the propagation of patterns in their spatial domain. ECoG ClusterFlow makes it possible 1) to compare the spatio-temporal evolution patterns for continuous and discontinuous time-frames, 2) to aggregate data to compare and contrast temporal information at varying levels of granularity, 3) to investigate the evolution of spatial patterns without occluding the spatial context information. We present the results of our case studies on both simulated data and real epileptic seizure data aimed at evaluating the effectiveness of our approach in discovering meaningful patterns and insights to the domain experts in our team.

### Required dependencies ###
  numpy
  networkx
  nibabel
  pydot
  community
  pygraphviz
  PySide
  decorator
################################

### Getting Started  ###
Note: Tested on OS X 10.11.6 and Ubuntu 14.04

	# Conda installation, if you dont have conda, then install via (http://conda.pydata.org/docs/install/quick.html)
	conda install python=2.7
	conda install anaconda-client
	conda config --add channels anaconda
	conda config --add channels pdrops  
	conda config --add channels allank
	conda config --add channels asmeurer 
	conda config --add channels menpo
	conda config --add channels conda-forge
	
	conda install -c sugeerth ecogCluster
	
	#Download ECoG Cluster repository and then goto src folder 
	
	ecogClusterdir/src> RunProjectMain.py 
		Happy Analysis! 

####Major Files
	**BrainViewerDataPaths.py** -- path for the dataset

	**RunMainProject.py** -- path for running the application

##Running the Tool 
        modulyzerdir/src> RunMainProject.py
        
Contributing
------------

See [Contributing](CONTRIBUTING.md)

### Citation Information###
Please cite ECoG ClusterFlow in your publications if it helps your research:

	@inproceedings{Murugesan:2016:HSV:2975167.2985688,
	 author = {Murugesan, Sugeerth and Bouchard, Kristofer and Chang, Edward and Dougherty, Max and Hamann, Bernd and Weber, Gunther H.},
	 title = {Hierarchical Spatio-temporal Visual Analysis of Cluster Evolution in Electrocorticography Data},
	 booktitle = {Proceedings of the 7th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics},
	 series = {BCB '16},
	 year = {2016},
	 isbn = {978-1-4503-4225-4},
	 location = {Seattle, WA, USA},
	 pages = {630--639},
	 numpages = {10},
	 url = {http://doi.acm.org/10.1145/2975167.2985688},
	 doi = {10.1145/2975167.2985688},
	 acmid = {2985688},
	 publisher = {ACM},
	 address = {New York, NY, USA},
	 keywords = {Brain Imaging, Electrocorticography, Graph Visualization, Linked Views, Neuroinformatics},
	} 
The ECoG ClusterFlow Project makes use of the following libraries
* [Numpy](https://pypi.python.org/pypi/numpy/1.11.0), Copyright (C) 2004-2016, NetworkX is distributed with the BSD license
* [NIBABEL](http://nipy.org/nibabel/), Copyright (c) 2012-2016, The MIT License
* [PYDOT](https://pypi.python.org/pypi/pydot), Copyright (c) 2005-2011 Ero Carrera, Distributed under MIT license
* [PYTHON LOUVAIN](https://pypi.python.org/pypi/python-louvain), Copyright (c) 2009, Thomas Aynaud <thomas.aynaud@lip6.fr>
* [PYGRAPHVIZ](https://pypi.python.org/pypi/python-louvain), Copyright (C) 2006-2015 by Aric Hagberg, PyGraphviz is distributed with a BSD license
* [VTK](http://www.vtk.org/), Copyright (c) 1993-2008 Ken Martin, Will Schroeder, Bill Lorensen,  BSD license.
* [PYSIDE](https://pypi.python.org/pypi/PySide/1.2.4), Copyright (C) 1991, 1999 Free Software Foundation, Inc
, GNU LESSER GENERAL PUBLIC LICENSE


For full license information regarding included and used software please refer to the file LICENSE.

### License Information ###
ECoG ClusterFlow is released under the [BSD license](https://github.com/sugeerth/ECoG ClusterFlow/blob/master/LICENSE).

ECoG ClusterFlow Copyright (c) 2016, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy).  All rights reserved.
 
If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Innovation and Partnerships department at IPO@lbl.gov referring to " Brain Modulyzer (2016-149),."
 
NOTICE.  This software was developed under funding from the U.S. Department of Energy.  As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, prepare derivative works, and perform publicly and display publicly.  Beginning five (5) years after the date permission to assert copyright is obtained from the U.S. Department of Energy, and subject to any subsequent five (5) year renewals, the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
        

