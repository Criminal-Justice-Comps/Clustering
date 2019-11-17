import unittest
import math

x = [1, 10]
y =  [1, 1]
z = [-4, 1]
num_data = [x, y, z]

def magnitude(vector):
    sum_of_squares = 0
    for i in range(len(vector)):
        sum_of_squares += pow(vector[i], 2)
    magnitude = math.sqrt(sum_of_squares)
    return magnitude

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

# change to take in a list of clusters
def minDistance(list_of_vectors):
    min_dist = math.inf
    closest_vectors = None
    for i in range(len(list_of_vectors)):
        for j in range(i+1, len(list_of_vectors)):
            dist = euclideanDistance(list_of_vectors[i], list_of_vectors[j])
            if(dist < min_dist):
                min_dist = dist
                closest_vectors = (i, j)
    return min_dist, closest_vectors

class Clustroid:
  def __init__(self, numerical_vector, categorical_vector = None,
                    parent = None, radius = None):
    self.num_vector = numerical_vector
    self.cat_vector = categorical_vector
    self.parent = parent
    self.radius = radius # Limit the size of a cluster, probably won't need

def hierarchicalClustering(data, radius = None):

    list_of_centroids = []
    smallest_distance, closest_vectors = minDistance(data)
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
    list_of_clustroids = []
    for i in range(len(num_data)):
        new_clustroid = Clustroid(unitVector(num_data[i]))
        list_of_clustroids.append(new_clustroid)

    #turn numerical data into unit vectors
    # list_of_uv = []
    # for i in range(len(num_data)):
    #     unit_vec = unitVector(num_data[i])
    #     list_of_uv.append(unit_vec)


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
    print(hierarchicalClustering(list_of_uv))

if __name__ == '__main__':
    main()
