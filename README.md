# stochasticSVD
a distributed implementation of singular value decomposition on Hadoop MapReduce


## HDFS
cluster is stored on _hdfs://cs4480-101.cs.cityu.edu.hk:9000_  

## Usage
This repo is to implement a distributed version of singluar value decomposition, using 
stochastic gradient decent.   

## Version
Hadoop 2.6.0

## Dictionary Usage
We build a index-word dictionary(tsv file) for word-to-number conversion.  
The col index in Our Word-Document / TF-IDF Matrix are based on this index-word 
dictionary. 