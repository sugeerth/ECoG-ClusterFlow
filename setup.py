import os
import setuptools

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
	name="ECoGClusterFlow",
	version="1.0.0",
	maintainer="Sugeerth Murugesan",
	maintainer_email="smuru@ucdavis.edu",
	description=("A visualizer for ECoG brain Networks"),
	license="BSD",
	package_data={'ECoGClusterFlow':['src/*']},
	scripts=['RunProjectMain2.py'],
	url="https://github.com/sugeerth/ECoG-ClusterFlow",
	long_description=read('README.md'),
	classifiers=[
		"Development Status :: 4 - Beta",
		"Environment :: X11 Applications",
		"Intended Audience :: Science/Research",
		"Natural Language :: English",
		"Programming Language :: Python :: 2.7",
		"Topic :: Scientific/Engineering :: Visualization",
	],
	platforms=['any'],
)