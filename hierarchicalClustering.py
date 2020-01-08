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

#goes through a list of clusters and gets the two closest ones
def minDistance(list_of_clusters):
    min_dist = math.inf
    closest_clusters = None
    for i in range(len(list_of_clusters)):
        for j in range(i+1, len(list_of_clusters)):
            dist = euclideanDistance(list_of_clusters[i].num_vector,
                list_of_clusters[j].num_vector)
            if(dist < min_dist):
                min_dist = dist
                closest_clusters = (i, j)
    return min_dist, closest_clusters

class Clustroid:
  def __init__(self, numerical_vector, categorical_vector = None,
                    parent = None, radius = None):
    self.num_vector = numerical_vector
    self.cat_vector = categorical_vector
    self.parent = parent
    self.radius = radius # Limit the size of a cluster, probably won't need

def getCentroid(vector1, vector2):
    centroid = scale(addVectors(vector1, vector2), 0.5)
    return centroid

def hierarchicalClustering(data, radius = None):
    if(len(data) == 0):
        return None
    if(len(data) == 1):
        only = Clustroid(data[0])
        return only
    list_of_centroids = []
    smallest_distance, closest_clusters = minDistance(data)
    cluster1 = data[closest_clusters[0]]
    cluster2 = data[closest_clusters[1]]
    cluster1_num_vec = cluster1.num_vector
    cluster2_num_vec = cluster2.num_vector
    cluster3_num_vec = getCentroid(cluster1_num_vec, cluster2_num_vec)
    print("############################################")
    print("Cluster1 is: ", cluster1_num_vec)
    print("Cluster2 is: ", cluster2_num_vec)
    print("The new cluster is: ", cluster3_num_vec)
    print("############################################")
    return closest_clusters
    '''if both the vectors in closest_vectors are centroids, then merge clusters
    and optimize a new centroid. Else if includes a single centroid, then
    optimize that centroid and add the non centroid to the cluster. Otherwise,
    both of the closest_vectors are not part of any cluster, form a new centroid.

    Create a Centroid, it should contain the parent centroid, a list of vectors
    that are a part of its cluster, its range(?), a vector reoresented by its
    location
    '''

def main():
    #open some CSV file, read through it, and create a matrix of vectors

    #get numerical data

    #assume numerical data only for now

    x = [1, 10]
    y =  [1, 1]
    z = [-4, 1]
    num_data = [x, y, z]

    list_of_clustroids = []
    for i in range(len(num_data)):
        new_clustroid = Clustroid(unitVector(num_data[i]))
        list_of_clustroids.append(new_clustroid)

    print(x)
    print(y)
    print(z)
    print("--------------------------------------------------")
    unitX = unitVector(x)
    unitY = unitVector(y)
    unitZ = unitVector(z)
    print(unitX)
    print(unitY)
    print(unitZ)
    print("--------------------------------------------------")
    print("||||||||||||||||||||||||||||||||||||||||||||")
    print("No items in data", hierarchicalClustering([]))
    print("One item in data", hierarchicalClustering([x]))
    print("||||||||||||||||||||||||||||||||||||||||||||")
    print("Multiple items, data", hierarchicalClustering(list_of_clustroids))

if __name__ == '__main__':
    main()
