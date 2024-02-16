import numpy as np
import os
#import sys
#from scipy.io import savemat
import math
import matplotlib.pyplot as plt
import pandas as pd

file_path = '/content/drive/MyDrive/Colab_Notebooks/Pointnet_Pointnet2_pytorch/data/ROISkullShapeFeatureData/'
file_list = sorted(os.listdir('/content/drive/MyDrive/Colab_Notebooks/Pointnet_Pointnet2_pytorch/data/ROISkullShapeFeatureData'))

def read_csv_file(file_name):
    df = pd.read_csv(file_name, header=None, names=['x', 'y', 'z', 'label'])
    
    vertex = np.array([df['x'], df['y'], df['z']], dtype=float)
    vertex = vertex.T
    label = np.array([df['label']])
    
    return vertex, label


# read all data
point_all_subjects_all_parts = []
label_all_subjects_all_parts = []
points_all_parts = []
labels_all_parts = []
# Read for the first subject
vertex_0, label_0 = read_csv_file(file_path + file_list[0])
vertex_0 = vertex_0.reshape(1, -1, 3)

vertex_all_subjects = vertex_0
label_all_subjects = label_0

for i_subject in range(1, len(file_list)):
    vertex_i, label_i = read_csv_file(file_path + file_list[i_subject])
    vertex_i = vertex_i.reshape(1, -1, 3)

    vertex_all_subjects =  np.concatenate((vertex_all_subjects, vertex_i), axis=0)
    label_all_subjects = np.concatenate((label_all_subjects, label_i), axis=0)
    print(file_list[i_subject])


# Normalize the database
def pc_normalize(pc):
    num_points = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc_norm = pc[0] - centroid
    pc_norm = np.reshape(pc_norm, (1,1,3))
    
    for idx in range(1,len(pc)):
        pci = pc[idx] - centroid
        pci = np.reshape(pci, (1,1,3))
        pc_norm = np.concatenate((pc_norm, pci), axis=1)
    
    pc_norm = pc_norm / 0.12
    
    return pc_norm

vertex_all_subject_1_norm = pc_normalize(vertex_all_subjects[0,:,:].reshape((-1,3)))

def Rx_matrix(alpha):
    return np.array([[1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)]])

def Ry_matrix(beta):
    return np.array([[np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]])

def Rz_matrix(gamma):
    return np.array([[np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1]])


def rotation_x_axis(A_Matrix, alpha):
    R_x = Rx_matrix(alpha)
    A_rotate = np.zeros(A_Matrix.shape)
    for irow in range(A_Matrix.shape[0]):
        A_rotate[irow, :] = np.matmul(R_x, np.transpose(A_Matrix[irow,:]))

    return A_rotate

def rotation_y_axis(A_Matrix, beta):
    R_y = Ry_matrix(beta)
    A_rotate = np.zeros(A_Matrix.shape)
    for irow in range(A_Matrix.shape[0]):
        A_rotate[irow, :] = np.matmul(R_y, np.transpose(A_Matrix[irow,:]))

    return A_rotate

def rotation_z_axis(A_Matrix, gamma):
    R_z = Rz_matrix(gamma)
    A_rotate = np.zeros(A_Matrix.shape)
    for irow in range(A_Matrix.shape[0]):
        A_rotate[irow, :] = np.matmul(R_z, np.transpose(A_Matrix[irow,:]))

    return A_rotate

def rotation_xyz_axis(A_Matrix, alpha, beta, gamma):
    # rotate around x-axis
    A_rotate_x = rotation_x_axis(A_Matrix, alpha)
    # rotate around y-axis
    A_rotate_xy = rotation_y_axis(A_rotate_x, beta)
    # rotate around z-axis
    A_rotate_xyz = rotation_z_axis(A_rotate_xy, gamma)

    return A_rotate_xyz

def augmentedbyrotation(A_matrix):
    #
    A_matrix_augmented = A_matrix
    A_matrix_augmented = A_matrix_augmented.reshape(1,-1,3)

    A_matrix_30x = rotation_x_axis(A_matrix[:,:], math.pi/6)
    A_matrix_45x = rotation_x_axis(A_matrix[:,:], math.pi/4)

    A_matrix_30y = rotation_y_axis(A_matrix[:,:], math.pi/6)
    A_matrix_45y = rotation_y_axis(A_matrix[:,:], math.pi/4)

    A_matrix_30z = rotation_z_axis(A_matrix[:,:], math.pi/6)
    A_matrix_45z = rotation_z_axis(A_matrix[:,:], math.pi/4)

    A_matrix_45xyz = rotation_xyz_axis(A_matrix[:,:], math.pi/4, math.pi/4, math.pi/4)

    A_matrix_augmented = np.concatenate((A_matrix_augmented, A_matrix_30x.reshape(1,-1,3), A_matrix_45x.reshape(1,-1,3),
                                       A_matrix_30y.reshape(1,-1,3), A_matrix_45y.reshape(1,-1,3), A_matrix_30z.reshape(1,-1,3),
                                       A_matrix_45z.reshape(1,-1,3), A_matrix_45xyz.reshape(1,-1,3)), axis=0)
    return A_matrix_augmented

vertex_rotation_augmented = augmentedbyrotation(vertex_all_subjects[0,:,:])

for i_subject in range(1,vertex_all_subjects.shape[0]):
    A_matrix_augmented = augmentedbyrotation(vertex_all_subjects[i_subject,:,:])
    vertex_rotation_augmented = np.concatenate((vertex_rotation_augmented, A_matrix_augmented), axis=0)
    print(i_subject)

vertex_rotation_augmented.shape