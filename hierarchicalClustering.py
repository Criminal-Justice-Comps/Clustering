import unittest
import math

x = [1, 10]
y =  [1, 1]
z = [-4, 1]
data = [x, y, z]
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
    print("####################")
    for i in range(len(vector1)):
         distance += pow(vector1[i] - vector2[i], 2)
         print(distance)
    distance = math.sqrt(distance)
    print("####################")
    return distance

def minDistance(list_of_vectors):
    min_dist = 10
    closest_vectors = ()
    print("*********************")
    for i in range(len(list_of_vectors)):
        print("MIN = ", min_dist)
        for j in range(i+1, len(list_of_vectors)):
            dist = euclideanDistance(list_of_vectors[i], list_of_vectors[j])

            print(dist)

            if(dist < min_dist):
                min_dist = dist
                closest_vectors = (i, j)
    print("*********************")
    return min_dist, closest_vectors

def hierarchicalClustering(data):

    list_of_uv = []
    for i in range(len(data)):
        unit_vec = unitVector(data[i])
        list_of_uv.append(unit_vec)

    list_of_centroids = []
    for i in range(len(list_of_uv)):
        for j in range(i+1, len(list_of_uv)):
            continue
    smallest_distance = 10

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
# print(euclideanDistance(unitX, unitY))
# print(euclideanDistance(unitX, unitZ))
# print(euclideanDistance(unitY, unitZ))
print(minDistance([unitX,unitY,unitZ]))
