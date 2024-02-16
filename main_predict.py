import argparse
import os
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import timeit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--npoint', type=int, default=16384, help='point Number')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    return parser.parse_args()

args = parse_args()


point_all_parts = np.load('data/vertex_rotation_augmented_sample.npy')
label_all_parts = np.load('data/label_rotation_augmented_sample.npy')

points_data = point_all_parts.transpose(0,2,1)
label_data = np.zeros([2632,1],dtype=int)
targets_data = label_all_parts

points_test = points_data[2112:2624,:,0:args.npoint]
label_test = label_data[2112:2624,:]
targets_test = targets_data[2112:2624,0:args.npoint]


num_classes = 1
num_part = 11

'''MODEL LOADING'''
MODEL = importlib.import_module('pointnet2_part_seg_ssg')
#MODEL = importlib.import_module(args.model)
classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()

checkpoint = torch.load('log/part_seg/fo5_n2048_lr0.001_OpAdam_batch16/checkpoints/best_model_40.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def calculateIoU(predicts, targets, partID):
    index_partID_predict = (predicts == partID).nonzero(as_tuple=True)[0]
    index_partID_target = (targets == partID).nonzero(as_tuple=True)[0]
    combine_set = torch.unique(torch.cat((index_partID_predict,index_partID_target)))
    IoU = (len(index_partID_predict) + len(index_partID_target) - len(combine_set))/ len(combine_set)*100
    return IoU

def calculateMeanIoU(predicts, targets):
    npoint_predict = []
    npoint_target = []
    npoint_combine = []
    for partID in range(11):
        index_partID_predict = np.where(predicts == partID)[0]
        index_partID_target = np.where(targets == partID)[0]
        combine_set = np.unique(np.concatenate((index_partID_predict,index_partID_target)))
        npoint_predict.append(len(index_partID_predict))
        npoint_target.append(len(index_partID_target))
        npoint_combine.append(len(combine_set))
    IoU = (np.sum(npoint_predict) + np.sum(npoint_target) - np.sum(npoint_combine)) / np.sum(npoint_combine)*100
    return IoU


start = timeit.default_timer()
with torch.no_grad():
    test_metrics = {}
    total_seen = 0
    total_seen_class = [0 for _ in range(num_part)]
    # define classifier
    classifier = classifier.eval()
    targets_test_all = torch.empty(0)
    predicts_test_all = torch.empty(0)
    mean_correct = []
    for batch_id in range(int(targets_test.shape[0]/args.batch_size)):
        label_test_batch = np.zeros([args.batch_size,1], dtype=int)
        label_test_batch = torch.Tensor(label_test_batch)
        targets_test_batch = targets_test[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,
                                            0:args.npoint]
        targets_test_batch = torch.Tensor(targets_test_batch)
        points_test_batch = points_test[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,:,
                                            0:args.npoint]#.data.numpy()
        points_test_batch = torch.Tensor(points_test_batch)
        #predict
        points_test_batch, label_test_batch, targets_test_batch = points_test_batch.float().cuda(), label_test_batch.long().cuda(), targets_test_batch.long().cuda()
        # just adjust
        #points_test = points_test.transpose(2, 1)
        seg_pred, _ = classifier(points_test_batch, to_categorical(label_test_batch, num_classes))
        #
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        targets_test_batch = targets_test_batch.reshape(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(targets_test_batch.data).cpu().sum()
        mean_correct.append(correct.item() / (args.batch_size * args.npoint))
        #
        targets_test_all = torch.cat((targets_test_all,targets_test_batch.cpu()))
        predicts_test_all = torch.cat((predicts_test_all, pred_choice.cpu()))
    #
    test_metrics['accuracy'] = np.mean(mean_correct)
    print(f'{round(np.mean(mean_correct)*100,2)}') # accuracy: 
    print(f'{round(calculateMeanIoU(predicts_test_all, targets_test_all),2)}') #mean IoU: 

stop = timeit.default_timer()
print(round(stop - start,2))


isubj = 0

for idx in range(11):
  #print('IoU for part ' + str(idx) + ' ' + str(calculateIoU(targets_test_all[isubj*32768:isubj*32768+32768], predicts_test_all[isubj*32768:isubj*32768+32768], idx)))
  print(round(calculateIoU(targets_test_all, predicts_test_all, idx),2))

isubj = 8
# Example data (replace with your actual data)
points = points_test[isubj,:,:]  # Example point cloud data
label = targets_test_all[isubj*16384: isubj*16384 +16384]  # Example labels for each point
label_predict = predicts_test_all[isubj*16384: isubj*16384 +16384]
# Define colors for each label
label_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']

# Plot target
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot each point with its corresponding color based on the label
for i in range(11):
    points_subset = points[:, label == i]
    color = label_colors[i]
    ax.scatter(points_subset[0], points_subset[1], points_subset[2], c=color, label=f'Label {i}')

ax.set_facecolor('none')
ax.set_axis_off()
ax.grid(False)
plt.show()


# Plot predict
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot each point with its corresponding color based on the label
for i in range(11):
    points_subset = points[:, label_predict == i]
    color = label_colors[i]
    ax.scatter(points_subset[0], points_subset[1], points_subset[2], c=color, label=f'Label {i}')

ax.set_facecolor('none')
ax.set_axis_off()
ax.grid(False)
plt.show()


# Plot error
# Create label_compare: 1 for correct prediction, 0 for incorrect prediction
label_compare = (label == label_predict).int()
label_compare_np = label_compare.numpy()

# Define colors for correct and incorrect predictions
colors = ['g' if val == 1 else 'r' for val in label_compare_np]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the point cloud with colors based on label_compare
ax.scatter(points[0], points[1], points[2], c=colors)
ax.set_facecolor('none')
ax.set_axis_off()
ax.grid(False)
plt.show()