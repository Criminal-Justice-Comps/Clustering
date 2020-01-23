import unittest
import math

MINIMUM = []
MAXIMUM = []
RANGE = []
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
"""input(s) two numeric vectors/lists. Returns a distance between the two"""
def gowerNumeric(vector1, vector2):
    assert(len(vector1) == len(vector2)), "Different size vectors"
    distance = 0
    for i in range(len(vector1)):
         distance += abs(abs(vector1[i]) - abs(vector2[i]))/RANGE[i]
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
    euclid_distance = gowerNumeric(cluster1.num_vector,
                                    cluster2.num_vector)
    categorical_distance = hammingDistance(cluster1.cat_vector,
                                    cluster2.cat_vector)/len(cluster1.cat_vector)
    distance = euclid_distance + categorical_distance
    return distance

"""input(s) a cluster and returns whether or not it has any children"""
def hasChildren(cluster):
    if cluster.children is not None:
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

"""goes through a list of clusters and gets the indexes of the two closest
    clusters according to single-link/min-link"""
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
    return closest_clusters

"""input(s) two clusters. Return the max_dist between them"""
def compareChildrenMax(children1, children2):
    max_dist = -math.inf
    dist = None
    #basecase
    if not hasChildren(children1):
        if not hasChildren(children2):
            dist = euclideanDistance(children1.num_vector,
                                            children2.num_vector)
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

def maxLink(list_of_clusters):
        max_dist = -math.inf
        closest_clusters = None
        for i in range(len(list_of_clusters)):
            for j in range(i+1, len(list_of_clusters)):
                cluster_i = list_of_clusters[i]
                cluster_j = list_of_clusters[j]
                dist = compareChildrenMin(cluster_i, cluster_j)
                if(dist > max_dist):
                    max_dist = dist
                    closest_clusters = (i, j)
        return closest_clusters
class Cluster:
  def __init__(self, numerical_vector, other_info, categorical_vector = None,
                children = None):
    self.num_vector = numerical_vector
    self.cat_vector = categorical_vector
    self.other_info = other_info
    self.children = children
    self.parent = None

def printCluster(cluster):
    print("Memory address for following cluster is: ", cluster)
    print("Numerical Vector: ", cluster.num_vector)
    print("Categorical Vector: ", cluster.cat_vector)
    print("Children: ", cluster.children)
    print("Parent: ", cluster.parent)
    print()

def printDendrogram(dendrogram):
    printCluster(dendrogram)
    if dendrogram.children is not None:
        for child in dendrogram.children:
            printDendrogram(child)

"""input(s) two clusters that will combine to form a new parent cluster"""
def createNewCluster(cluster1, cluster2):

    list_of_chldren = [cluster1, cluster2]
    newCluster = Cluster(None, None, list_of_chldren)
    cluster1.parent = newCluster
    cluster2.parent = newCluster
    return newCluster

def hierarchicalClustering(data, radius = None):
    if(len(data) == 0):
        return None
    if(len(data) == 1):
        only = data[0]
        return only

    #using min-link
    clustersToCombine = minLink(data)
    cluster1 = data[clustersToCombine[0]]
    cluster2 = data[clustersToCombine[1]]
    data.remove(cluster1)
    data.remove(cluster2)
    newCluster = createNewCluster(cluster1, cluster2)
    data.append(newCluster)
    return hierarchicalClustering(data)

def main():
    #open some CSV file, read through it, and create a matrix of vectors
    #assume numerical data only for now
    is_first = 1
    list_of_clusters = []
    keys = []
    with open ('../datasets/TrainValidateTest/TrainNumericFeatures.csv', mode='r') as csvfile:
        for line in csvfile:
            #If it's the first line, get the column headers into "keys"
            if is_first:
                is_first = 0
                keys = line.split(",")
                print(keys)
                continue
            num_data = line.split(",") #split the line we are on into individual features
            num_data.pop(keys.index("decile_score")) # remove the features we don't want
            num_data.pop(keys.index("person_id")) # remove the features we don't want
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
            list_of_clusters.append(Cluster(num_data)) #create the cluster and add it to our list of clusters
    for q in range(len(MAXIMUM)): #calculation to get the range of each numeric feature (used for gowers distance)
        RANGE.append(MAXIMUM[q] - MINIMUM[q])
    csvfile.close()
    keys_categorical = []
    cat_data = []
    is_first = 1
    i = 0
    with open ('../datasets/TrainValidateTest/TrainCategoricalFeatures.csv', mode='r') as csvfile:
        for line in csvfile:
            #If it's the first line, get the column headers into "keys"
            if is_first:
                is_first = 0
                keys_categorical = line.split(",")
                #Remove columns we don't want here
                print(keys_categorical)
                continue
            cat_data = line.split(",") #split the line we are on into individual features
            list_of_clusters[i].cat_vector = cat_data
            i += 1
    dendrogram = hierarchicalClustering(list_of_clusters[:10])
    printDendrogram(dendrogram)


if __name__ == '__main__':
    main()
