# Hierarchical Spatio-Temporal Visual Analysis of ECoG Data #

 ![ScreenShot](https://github.com/sugeerth/ECoG-ClusterFlow/blob/FinalWorkingTool/src/Images/Synthetic.png)
 
 ![ScreenShot](https://raw.githubusercontent.com/sugeerth/ECoG-ClusterFlow/master/src/Images/Uload.png?token=AD8LYFkvUliWTEDgaXDsxE1EtePPAER4ks5XOlSnwA%3D%3D)

# ECoG ClusterFlow #
We present ECoG ClusterFlow, an interactive visual analysis system for the exploration of high-resolution Electrocorticography (ECoG) data. Our system detects and visualizes dynamic high-level structures, such as communities, using the time-varying spatial connectivity network derived from the high-resolution ECoG data. ECoG ClusterFlow provides a multi-scale visualization of the spatio-temporal patterns underlying the time-varying communities using two views: 1) an overview summarizing the evolution of clusters over time and 2) a hierarchical glyph-based technique that uses data aggregation and small multiples techniques to visualize the propagation of patterns in their spatial domain. ECoG ClusterFlow makes it possible 1) to compare the spatio-temporal evolution patterns for continuous and discontinuous time-frames, 2) to aggregate data to compare and contrast temporal information at varying levels of granularity, 3) to investigate the evolution of spatial patterns without occluding the spatial context information. We present the results of our case studies on both simulated data and real epileptic seizure data aimed at evaluating the effectiveness of our approach in discovering meaningful patterns and insights to the domain experts in our team.

### Required dependencies ###
    PySide
    networkx 
    numpy (BrainViewer)
    pygraphviz (BrainViewer)
    communtiy (BrainViewer)
################################



***********************************************************************
matrix_filename = '/<edit your path to these files>/27nodeMatrix.csv'
template_filename = '/<edit your path to these files>/ch2better.nii.gz'
parcelation_filename = '/<edit your path to these files>/allROIs.nii.gz'
************************************************************************
