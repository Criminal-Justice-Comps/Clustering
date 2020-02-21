'''
Code written by a Carleton College COMPS group for our study on algorithms in the criminal justice system.
All code is original.
Project members are Kellen Dorchen, Emilee Fulton, Carlos Garcia, Cameron Kline-Sharpe, Dillon Lanier and Javin White
Written January 28-2020.
The project advisor was Layla Oesper in the Computer Science department.
'''

import unittest
import math
import csv

MINIMUM = []
MAXIMUM = []
RANGE = []
SIZE = int(150)
KEYS = []
FIRST_ITERATION = True

import numpy as np
distanceMatrix = []

"""https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage"""
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import seaborn as sns

Z = []
'''



*********************************************************************
THIS CODE IS NOT IN USE, ALLOWS THE CALCULATION OF EUCLIDEAN DISTANCE
*********************************************************************




"""input(s) a list of numerical data to be recognized as a vector and outputs
	the magnitude for the given vector1"""
def magnitude(vector):
	sum_of_squares = 0
	for i in range(len(vector)):
		sum_of_squares += pow(vector[i], 2)
	magnitude = math.sqrt(sum_of_squares)
	return magnitude

"""input(s) two vectors and returns a vector that is the sum of the two"""
def addVectors(vector1, vector2):
	assert(len(vector1) == len(vector2) and "Length doesn't match")
	sum_vector = []
	for i in range(len(vector1)):
		sum_vector.append(vector1[i] + vector2[i])
	return sum_vector

"""input(s) a vector and a scalar. Returns a scaled vector using the inputs"""
def scale(vector, scalar):
	scaled_vector = []
	for i in range(len(vector)):
		scaled_vector.append(vector[i] * scalar)
	return scaled_vector

"""input(s) a list/vector. returns a unit vector based off the input"""
def unitVector(vector):
	magn = magnitude(vector)
	scalar = 1/magn
	unit = scale(vector, scalar)
	return unit

"""input(s) two numerical vectors/lists. Returns the euclidean distance between
	the inputs"""
def euclideanDistance(vector1, vector2):
	assert(len(vector1) == len(vector2)), "Different size vectors"
	distance = 0
	for i in range(len(vector1)):
		 distance += pow(abs(vector1[i]) - abs(vector2[i]), 2)
	distance = math.sqrt(distance)
	return distance



*********************************************************************
THIS CODE IS NOT IN USE, ALLOWS THE CALCULATION OF EUCLIDEAN DISTANCE
*********************************************************************




'''

"""input(s) two numeric vectors/lists. Returns a distance between the two"""
def gowerNumeric(vector1, vector2):
	assert(len(vector1) == len(vector2)), "Different size vectors"
	distance = 0
	for i in range(len(vector1)):
		 distance += abs(vector1[i] - vector2[i])/RANGE[i]
	return distance

"""input(s) two categorical vectors/lists. Returns the hamming distance"""
def hammingDistance(vector1, vector2):
	assert(len(vector1) == len(vector2)), "Different size vectors"
	distance = 0
	for i in range(len(vector1)):
		if vector1[i] != vector2[i]:
			distance += 1
	return distance

"""input(s) two clusters with a numerical and categorical vector. Returns the
	Gower Distance between the two."""
def gowerDistance(cluster1, cluster2):
	numeric_distance = gowerNumeric(cluster1.num_vector,
									cluster2.num_vector)
	categorical_distance = hammingDistance(cluster1.cat_vector,
									cluster2.cat_vector)

	distance = numeric_distance + categorical_distance
	distance = distance / (len(cluster1.num_vector) + len(cluster1.cat_vector))
	return distance

"""input(s) a cluster and returns whether or not it has any children"""
def hasChildren(cluster):
	if cluster.children != []:
		return True
	return False

"""input(s) two clusters. Return the min_dist between them"""
def compareChildrenMin(children1, children2):
	min_dist = math.inf
	dist = None
	#basecase
	if not hasChildren(children1):
		if not hasChildren(children2):
			dist = gowerDistance(children1, children2)
			return dist

		else:
			for child in children2.children:
				dist = compareChildrenMin(children1, child)
				if dist < min_dist:
					min_dist = dist

	else:
		if not hasChildren(children2):
			for child in children1.children:
				dist = compareChildrenMin(child, children2)
				if dist < min_dist:
					min_dist = dist
		else:
			for child in children1.children:
				for other_child in children2.children:
					dist = compareChildrenMin(child, other_child)
					if dist < min_dist:
						min_dist = dist
	return min_dist

"""INPUT: A list of clusters
	OUTPUT: Indexes of the two closest clusters according to single-link/min-link"""
def minLink(list_of_clusters):
	min_dist = math.inf
	closest_clusters = None
	for i in range(len(list_of_clusters)):
		for j in range(i+1, len(list_of_clusters)):
			cluster_i = list_of_clusters[i]
			cluster_j = list_of_clusters[j]
			dist = compareChildrenMin(cluster_i, cluster_j)
			if(dist < min_dist):
				min_dist = dist
				closest_clusters = (i, j)
	return closest_clusters, min_dist

"""input(s) two clusters. Return the max_dist between them"""
def compareChildrenMax(children1, children2):
	max_dist = -math.inf
	dist = None
	#basecase
	if not hasChildren(children1):
		if not hasChildren(children2):
			dist = gowerDistance(children1, children2)
			return dist

		else:
			for child in children2.children:
				dist = compareChildrenMax(children1, child)
				if dist > max_dist:
					max_dist = dist

	else:
		if not hasChildren(children2):
			for child in children1.children:
				dist = compareChildrenMax(child, children2)
				if dist > max_dist:
					max_dist = dist
		else:
			for child in children1.children:
				for other_child in children2.children:
					dist = compareChildrenMax(child, other_child)
				if dist > max_dist:
					max_dist = dist
	return max_dist

'''Also known as complete-link.
	INPUT: list of clusters
	OUTPUT: Indexes of two clusters we want to combine based on the max-link rules'''
def maxLink(list_of_clusters):
	max_dist = -math.inf
	closest_clusters = None
	for i in range(len(list_of_clusters)):
		for j in range(i+1, len(list_of_clusters)):
			cluster_i = list_of_clusters[i]
			cluster_j = list_of_clusters[j]
			dist = compareChildrenMax(cluster_i, cluster_j)
			if(dist > max_dist):
				max_dist = dist
				closest_clusters = (i, j)
	return closest_clusters, max_dist

"""input(s) two clusters.
	OUTPUT: the average_dist between them"""
def compareChildrenAverage(children1, children2):
	Av_dist = 0
	dist = None
	size_cluster_1 = 1 #keep track of how many data points in the clusters
	size_cluster_2 = 1
	#basecase
	if not hasChildren(children1):
		if not hasChildren(children2):
			dist = gowerDistance(children1, children2)
			return dist
		else:
			for child in children2.children:
				size_cluster_2 += 1
				dist = compareChildrenAverage(children1, child)
				Av_dist += dist
	else:
		if not hasChildren(children2):
			size_cluster_1 += 1
			for child in children1.children:
				dist = compareChildrenMax(child, children2)
				Av_dist += dist
		else:
			size_cluster_2 += 1
			size_cluster_1 += 1
			for child in children1.children:
				for other_child in children2.children:
					dist = compareChildrenMax(child, other_child)
					Av_dist += dist
	return Av_dist * (1/ (size_cluster_1 * size_cluster_2))

'''	INPUT: list of clusters
	OUTPUT: Indexes of two clusters we want to combine based on the Average-link rules'''
def averageLink(list_of_clusters):
	Mindistance = math.inf
	closest_clusters = None
	for i in range(len(list_of_clusters)):
		for j in range(i+1, len(list_of_clusters)):
			cluster_i = list_of_clusters[i]
			cluster_j = list_of_clusters[j]
			dist = compareChildrenAverage(cluster_i, cluster_j)
			if cluster_i.id < SIZE and cluster_j.id < SIZE and FIRST_ITERATION:
				if distanceMatrix[cluster_i.id, cluster_j.id] == np.inf:
					distanceMatrix[cluster_i.id, cluster_j.id] = dist
					distanceMatrix[cluster_j.id, cluster_i.id] = dist
			if(dist < Mindistance):
				Mindistance = dist
				closest_clusters = (i, j)
	return closest_clusters, Mindistance

class Cluster:
  def __init__(self, numerical_vector, ID, other_info = [],categorical_vector = [], children = []):
	  self.num_vector = numerical_vector
	  self.id = ID
	  self.cat_vector = categorical_vector
	  self.other_info = other_info
	  self.children = children
	  self.parent = None
	  self.observations = 1

def printCluster(cluster):
	print("Memory address for following cluster is: ", cluster)
	print("ID number is: ", cluster.id)
	print("Numerical Vector: ", cluster.num_vector)
	print("Categorical Vector: ", cluster.cat_vector)
	print("Other(decile_score, person_id): ", cluster.other_info)
	print("Children: ", cluster.children)
	print("Parent: ", cluster.parent)
	print()

def printTextDendrogram(root_cluster):
	printCluster(root_cluster)
	if root_cluster.children is not None:
		for child in root_cluster.children:
			printTextDendrogram(child)

"""input(s) two clusters that will combine to form a new parent cluster"""
def createNewCluster(cluster1, cluster2):

	global SIZE
	list_of_chldren = [cluster1, cluster2]
	newCluster = Cluster(None, SIZE, None, None, list_of_chldren)
	SIZE += 1
	newCluster.observations = cluster1.observations + cluster2.observations
	cluster1.parent = newCluster
	cluster2.parent = newCluster
	return newCluster

def hierarchicalClustering(data, linkageMethod = "complete"):
	if(len(data) == 0):
		return None
	if(len(data) == 1):
		root_cluster = data[0]

		return root_cluster
	global FIRST_ITERATION
	clustersToCombine, corr_distance = averageLink(data)
	FIRST_ITERATION = False
	cluster1 = data[clustersToCombine[0]]
	cluster2 = data[clustersToCombine[1]]
	data.remove(cluster1)
	data.remove(cluster2)
	newCluster = createNewCluster(cluster1, cluster2)
	Z.append([int(cluster1.id), int(cluster2.id), float(corr_distance), int(newCluster.observations)])
	data.append(newCluster)
	return hierarchicalClustering(data, linkageMethod)

'''INPUT: Data from our .csv files
	OUTPUT: a list of clusters, each cluster represents a person in our .csv files'''
def loadData():
	'''
	*********************************
	NOW LOADING THE NUMERIC DATA
	*********************************
	'''
	is_first = 1 #bool to know if its the headers (ie age, # of offenses)
	list_of_clusters = []
	keys_numeric = []
	new_id = 0
	with open ('../datasets/TrainValidateTest/ValidateFeaturesNumericClustering.csv', mode='r') as csvfile:
		for line in csvfile:
			#If it's the first line, get the column headers into "keys"
			if is_first:
				is_first = 0
				keys_numeric = line.split(",")
				continue
			num_data = line.split(",") #split the line we are on into individual features
			other_info = []
			''' REMOVE THE FEATURES WE DON'T WANT'''
			other_info.append(num_data.pop(keys_numeric.index("decile_score")))
			ID = num_data.pop(keys_numeric.index("person_id"))
			''' REMOVE THE FEATURES WE DON'T WANT'''

			#Set global variables MINIMUM and MAXIMUM-- these are used for Gowers Distance
			if MINIMUM == []:
				for i in range(len(num_data)):
					MINIMUM.append(math.inf)
			if MAXIMUM == []:
				for i in range(len(num_data)):
					MAXIMUM.append(-math.inf)
			for i in range (len(num_data)): #for each feature in a row
				num_data[i] = float(num_data[i]) #set the string to a float value (assumes numeric)
				if num_data[i] > MAXIMUM[i]: #update max and min for each feature global (for gowers distance)
					MAXIMUM[i] = num_data[i]
				if num_data[i] < MINIMUM[i]:
					MINIMUM[i] = num_data[i]
			list_of_clusters.append(Cluster(num_data, new_id, other_info)) #create the cluster and add it to our list of clusters
			new_id += 1
	for q in range(len(MAXIMUM)): #calculation to get the range of each numeric feature (used for gowers distance)
		RANGE.append(MAXIMUM[q] - MINIMUM[q])
	csvfile.close()
	'''
	*********************************
	NOW LOADING THE CATEGORICAL DATA
	*********************************
	'''
	keys_categorical = []
	cat_data = []
	is_first = 1
	i = 0
	with open ('../datasets/TrainValidateTest/ANA Merged/ANAMergedValidateFeaturesCategorical.csv', mode='r') as csvfile:
		for line in csvfile:
			#If it's the first line, get the column headers into "keys"
			if is_first:
				is_first = 0
				keys_categorical = line.split(",")
				#Remove columns we don't want here
				continue
			cat_data = line.split(",") #split the line we are on into individual features
			cat_data.pop(keys_categorical.index("person_id"))
			cat_data[-1] = cat_data[-1][:-2]
			list_of_clusters[i].cat_vector = cat_data
			i += 1 #This allows us to update the categorical vector, necesary since we already made the cluster above
	keys_numeric.remove("decile_score")
	keys_numeric[-1] = keys_numeric[-1][:-2]
	for key in keys_numeric:
		KEYS.append(str(key))
	keys_categorical.remove("person_id")
	keys_categorical[-1] = keys_categorical[-1][:-2]
	for key in keys_categorical:
		KEYS.append(str(key))
	return list_of_clusters

def saveData(Z):
	with open("AverageLinkTrainDataClusterMatrix.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(Z)

def getLeaves(cluster):
	list_of_datapoints = []
	#run BFS to get leaves/datapoints
	queue = cluster.children
	while len(queue) != 0:
		cur_cluster = queue.pop()
		if cur_cluster.children != []:
			queue.append(cur_cluster.children[0])
			queue.append(cur_cluster.children[1])
		else:
			data = []
			data.append(cur_cluster.id)
			for i in cur_cluster.num_vector:
				data.append(i)
			for j in cur_cluster.cat_vector:
				data.append(j)
			list_of_datapoints.append(data)
	return list_of_datapoints

def saveCluster(cluster, saveAs):
	datapoints = [KEYS]
	for data in getLeaves(cluster):
		datapoints.append(data)
	with open(saveAs+".csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(datapoints)

def splitClusters(root, k):
	k_clusters = [root]

	while len(k_clusters) != k:
		recent_cluster_id = -1
		index = None
		for i in range(len(k_clusters)):
			if k_clusters[i].id > recent_cluster_id:
				recent_cluster_id = k_clusters[i].id
				index = i
		removed = k_clusters.pop(index)
		k_clusters.append(removed.children[0])
		k_clusters.append(removed.children[1])

	i = 1
	for cluster in k_clusters:
		saveCluster(cluster, "AverageLinkTrainDataCluster"+str(k)+"_"+str(i))
		i += 1

def main():
	#args = parse_args()
	global Z
	global SIZE
	global distanceMatrix
	cluster_list = loadData()
	SIZE = len(cluster_list)
	distanceMatrix = np.ones((SIZE, SIZE)) * np.inf
	np.fill_diagonal(distanceMatrix, 0)
	root_cluster = hierarchicalClustering(cluster_list)
	saveData(Z)
	splitClusters(root_cluster, 3)
	sns.heatmap(distanceMatrix, cmap="BuPu")
	#print(shc.maxdists(Z))
	plt.figure(figsize = (16,9))
	dend = shc.dendrogram(Z)
	plt.axhline(y=2.0, color='r', linestyle='--')
	plt.show() #show the dendrogram


if __name__ == '__main__':
	main()
