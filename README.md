# flybase_miner
A text mining app for Drosophila genes

The goal of this script is to provide insight into network connectivity and function of a list of genes, 
using the flybase data repository. 

This python script will take a list of flybase gene IDs, and determine which papers reference each gene.
For all papers common to a gene pair, it will extract the abstracts, cluster using kmeans, and report the papers
based on the cluster into which they fall. 

It will also draw the network between all genes, according to how many papers co-reference them.

