import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import CameronsDBgenerator as gen
'exec(%matplotlib inline)'


def main():
	gen.save_as_csv(gen.generateData(False, False), 'output.csv')
	data = pd.read_csv('output.csv')
	data.head()

	#create the dendrogram 
	plt.figure(figsize=(10, 7))  
	plt.title("Dendrograms")  
	dend = shc.dendrogram(shc.linkage(data, method='ward'))
	plt.axhline(y=30, color='r', linestyle='--')


	cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
	print(cluster.fit_predict(data)) #show the list of clusters the data is in. ie for two clusters: [1, 1, 0, 1, 0]

	plt.show() #show the dendrogram


main()