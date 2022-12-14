'''
Derived from https://github.com/dionysus/2D_3D/blob/main/projection.py
'''
import numpy as np

def normalization(vector):
    square = np.square(vector)
    sum_vect = np.sum(square)
    return np.sqrt(sum_vect)

def cross_product(distance1, distance2):
    c1 = distance1[1]*distance2[2]-(distance2[1]*distance1[2])
    c2 = distance1[0]*distance2[2]-(distance2[0]*distance1[2])
    c3 = distance1[0]*distance2[1]-(distance2[0]*distance1[1])
    normal = np.array([c1, -c2, c3])
    value = normalization(normal)
    return normal / value

def projected_point(distance1, point1, distance2, point2):
    normal = cross_product(distance1, distance2)
    matrix = np.zeros((3, 3))
    matrix[0] = distance1
    matrix[1] = -distance2
    matrix[2] = normal
    matrix = matrix.T
    parameters = np.linalg.solve(matrix, point2-point1)
    first_projected = point1 + parameters[0] * distance1
    second_projected = point2 + parameters[1] * distance2
    mid_x = (first_projected[0] + second_projected[0]) / 2
    mid_y = (first_projected[1] + second_projected[1]) / 2
    mid_z = (first_projected[2] + second_projected[2]) / 2
    projected = np.array([mid_x, mid_y, mid_z])
    return projected