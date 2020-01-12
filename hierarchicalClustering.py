import unittest
import math

def magnitude(vector):
    sum_of_squares = 0
    for i in range(len(vector)):
        sum_of_squares += pow(vector[i], 2)
    magnitude = math.sqrt(sum_of_squares)
    return magnitude

def addVectors(vector1, vector2):
    assert(len(vector1) == len(vector2) and "Length doesn't match")
    sum_vector = []
    for i in range(len(vector1)):
        sum_vector.append(vector1[i] + vector2[i])
    return sum_vector

def scale(vector, scalar):
    scaled_vector = []
    for i in range(len(vector)):
        scaled_vector.append(vector[i] * scalar)
    return scaled_vector

def unitVector(vector):
    magn = magnitude(vector)
    scalar = 1/magn
    unit = scale(vector, scalar)
    return unit

def euclideanDistance(vector1, vector2):
    assert(len(vector1) == len(vector2)), "Different size vectors"
    distance = 0
    for i in range(len(vector1)):
         distance += pow(abs(vector1[i]) - abs(vector2[i]), 2)
    distance = math.sqrt(distance)
    return distance

def hasChildren(cluster):
    if cluster.children is not None:
        return True
    return False

def compareChildren(children1, children2):
    min_dist = math.inf
    dist = None
    #basecase
    if not hasChildren(children1):
        if not hasChildren(children2):
            dist = euclideanDistance(children1.num_vector,
                                            children2.num_vector)
            return dist

        else:
            for child in children2:
                dist = compareChildren(children1, child)
                if dist < min_dist:
                    min_dist = dist

    else:
        for child in children1:
            dist = compareChildren(child, children2)
            if dist < min_dist:
                min_dist = dist
        # if not hasChildren(children2):
        #     dist = compareChildren(children1.children[0], children2)
        #     dist = compareChildren(children1.children[1], children2)
        # else:
        #     dist = compareChildren(children1.children[0], children2.children[1])

    return min_dist
#goes through a list of clusters and gets the index of the two closest ones
def minLink(list_of_clusters):
    min_dist = math.inf
    closest_clusters = None
    for i in range(len(list_of_clusters)):
        for j in range(i+1, len(list_of_clusters)):
            cluster_i = list_of_clusters[i]
            cluster_j = list_of_clusters[j]
            if hasChildren(cluster_i):
                children_i = cluster_i.children
                children_j = cluster_j.children
                if hasChildren(cluster_j):
                    dist = compareChildren(children_i, children_j)
                else:
                    dist = compareChildren(children_i, cluster_j)
            else:
                if hasChildren(cluster_j):
                    children_j = cluster_j.children
                    dist = compareChildren(cluster_i, children_j)
                else:
                    dist = compareChildren(cluster_i, cluster_j)
            if(dist < min_dist):
                min_dist = dist
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

def createNewCluster(cluster1, cluster2):

    list_of_chldren = [cluster1, cluster2]
    newCluster = Cluster(None, None, list_of_chldren)
    cluster1.parent = newCluster
    cluster2.parent = newCluster
    return newCluster

def getCentroid(vector1, vector2):
    centroid = scale(addVectors(vector1, vector2), 0.5)
    return centroid

def hierarchicalClustering(data, radius = None):
    if(len(data) == 0):
        return None
    if(len(data) == 1):
        only = data[0]
        return only


    #using min-link
    # cluster1tocombine = None
    # cluster2tocombine = None
    # minDist = math.inf
    # for i in range(len(data)):
    #     cluster = data[i]
    #     for nextCluster in data[i:]:
    #         distance = euclideanDistance(cluster.num_data, nextCluster.num_data)
    #         if distance < minDist:
    #             minDist = distance
    #             cluster1tocombine = cluster
    #             cluster2tocombine = nextCluster
    #     cur_magn = cluster.num_data
    clustersToCombine = minLink(data)
    newCluster = createNewCluster(data.pop(clustersToCombine[0]), data.pop(clustersToCombine[1]))
    data.append(newCluster)
    printCluster(newCluster)
    return hierarchicalClustering(data)

    # list_of_centroids = []
    # smallest_distance, closest_clusters = minDistance(data)
    # cluster1 = data[closest_clusters[0]]
    # cluster2 = data[closest_clusters[1]]
    # cluster1_num_vec = cluster1.num_vector
    # cluster2_num_vec = cluster2.num_vector
    # cluster3_num_vec = getCentroid(cluster1_num_vec, cluster2_num_vec)
    # print("############################################")
    # print("Cluster1 is: ", cluster1_num_vec)
    # print("Cluster2 is: ", cluster2_num_vec)
    # print("The new cluster is: ", cluster3_num_vec)
    # print("############################################")

def main():
    #open some CSV file, read through it, and create a matrix of vectors
    #assume numerical data only for now
    is_first = 1
    list_of_clusters = []
    with open ('../datasets/NumericFeaturesDataSmall.csv', mode='r') as csvfile:
        for line in csvfile:
            if is_first:
                is_first = 0
                continue
            num_data = line.split(",")
            for i in range (len(num_data)):
                num_data[i] = float(num_data[i])
            list_of_clusters.append(Cluster(unitVector(num_data)))
    hierarchicalClustering(list_of_clusters[:5])
    # print("No items in data", hierarchicalClustering([]))
    # print("One item in data", hierarchicalClustering([x]))
    # print("||||||||||||||||||||||||||||||||||||||||||||")
    # print("Multiple items, data", hierarchicalClustering(list_of_clusters))

if __name__ == '__main__':
    main()
