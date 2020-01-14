import unittest
import math

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
"""input(s) two vectors/lists. Returns the euclidean distance between the inputs"""
def euclideanDistance(vector1, vector2):
    assert(len(vector1) == len(vector2)), "Different size vectors"
    distance = 0
    for i in range(len(vector1)):
         distance += pow(abs(vector1[i]) - abs(vector2[i]), 2)
    distance = math.sqrt(distance)
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
            dist = euclideanDistance(children1.num_vector,
                                            children2.num_vector)
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
  def __init__(self, numerical_vector, categorical_vector = None,
                children = None):
    self.num_vector = numerical_vector
    self.cat_vector = categorical_vector
    self.children = children
    self.parent = None

def printCluster(cluster):
    print("Numerical Vector: ", cluster.num_vector)
    print("Categorical Vector: ", cluster.cat_vector)
    print("Children: ", cluster.children)
    print("Parent: ", cluster.parent)

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
    with open ('../datasets/NumericFeaturesDataSmall.csv', mode='r') as csvfile:
        for line in csvfile:
            if is_first:
                is_first = 0
                keys = line.split(",")
                print(keys)
                continue
            num_data = line.split(",")
            for i in range (len(num_data)):
                if i != keys.index("person_id") and i != keys.index("decile_score"):
                    num_data[i] = float(num_data[i])
            num_data.pop(keys.index("decile_score"))
            num_data.pop(keys.index("person_id"))
            list_of_clusters.append(Cluster(num_data))
    dendrogram = hierarchicalClustering(list_of_clusters)
    printDendrogram(dendrogram)


if __name__ == '__main__':
    main()
