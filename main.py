import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_ssg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default='fo1_n16384_lr0.001_OpAdam_batch16', help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=16384, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    return parser.parse_args()

def log_string(str):
    logger.info(str)
    print(str)

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    num_classes = 1
    num_part = 11

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model_30.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    start_epoch = 0
    global_epoch = start_epoch
    try:
        loss_train = np.load(str(exp_dir)+'/train_loss.npy')[0:start_epoch]
        loss_valid = np.load(str(exp_dir)+'/valid_loss.npy')[0:start_epoch]
        acc_train = np.load(str(exp_dir)+'/train_acc.npy')[0:start_epoch]
        acc_valid = np.load(str(exp_dir)+'/valid_acc.npy')[0:start_epoch]
    except:
        loss_train = []
        loss_valid = []
        acc_train = []
        acc_valid = []

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        #print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        loss_train_batch = []
        for batch_id in range(int(targets_train.shape[0]/args.batch_size)):
            optimizer.zero_grad()

            label_train_batch = np.zeros([args.batch_size,1], dtype=int)
            label_train_batch = torch.Tensor(label_train_batch)
            targets_train_batch = targets_train[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,:]
            
            points_train_batch = points_train[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,:,:]#.data.numpy()
            points_train_batch[:, :, 0:3] = provider.random_scale_point_cloud(points_train_batch[:, :, 0:3])
            points_train_batch[:, :, 0:3] = provider.shift_point_cloud(points_train_batch[:, :, 0:3])
            points_train_batch = torch.Tensor(points_train_batch)

            targets_train_batch = targets_train_batch.reshape(-1, 1)[:, 0]
            targets_train_batch = torch.Tensor(targets_train_batch)
            points_train_batch, label_train_batch, targets_train_batch = points_train_batch.float().cuda(), label_train_batch.long().cuda(), targets_train_batch.long().cuda()

            seg_pred, trans_feat = classifier(points_train_batch, to_categorical(label_train_batch, num_classes))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(targets_train_batch.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))
            loss = criterion(seg_pred, targets_train_batch, trans_feat)
            loss.backward()
            optimizer.step()

            loss_train_batch.append(loss.cpu().item())

        loss_train = np.append(loss_train, np.mean(loss_train_batch))
        train_instance_acc = np.mean(mean_correct)
        acc_train = np.append(acc_train, np.mean(mean_correct))
        
        #if (epoch%30 == 0):
        np.save(str(exp_dir)+'/train_acc.npy',acc_train)
        np.save(str(exp_dir)+'/train_loss.npy',loss_train)
        
        log_string('Train accuracy is: %.5f' % train_instance_acc)

        with torch.no_grad():
            test_metrics = {}
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            
            classifier = classifier.eval()
            mean_correct_test = []
            loss_valid_batch = []

            for batch_id in range(int(targets_valid.shape[0]/args.batch_size)):
                label_test_batch = np.zeros([args.batch_size,1], dtype=int)
                label_test_batch = torch.Tensor(label_test_batch)
                targets_test_batch = targets_valid[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,:]
                targets_test_batch = torch.Tensor(targets_test_batch)
                points_test_batch = points_valid[0+batch_id*args.batch_size:args.batch_size+batch_id*args.batch_size,:,:]#.data.numpy()
                points_test_batch = torch.Tensor(points_test_batch)

                points_test_batch, label_test_batch, targets_test_batch = points_test_batch.float().cuda(), label_test_batch.long().cuda(), targets_test_batch.long().cuda()
                seg_valid, trans_feat_valid = classifier(points_test_batch, to_categorical(label_test_batch, num_classes))

                seg_valid = seg_valid.contiguous().view(-1, num_part)
                targets_test_batch = targets_test_batch.reshape(-1, 1)[:, 0]
                loss_val = criterion(seg_valid, targets_test_batch, trans_feat_valid)
                loss_valid_batch.append(loss_val.cpu().item())

                pred_choice = seg_valid.data.max(1)[1]
                correct = pred_choice.eq(targets_test_batch.data).cpu().sum()
                mean_correct_test.append(correct.item() / (args.batch_size * args.npoint))
            
            loss_valid = np.append(loss_valid, np.mean(loss_valid_batch))
            acc_valid = np.append(acc_valid, np.mean(mean_correct_test))
            test_metrics['accuracy'] = np.mean(mean_correct_test)

            #if (epoch%30 == 0):
            np.save(str(exp_dir)+'/valid_acc.npy',acc_valid)
            np.save(str(exp_dir)+'/valid_loss.npy',loss_valid)

        log_string('Test Accuracy: %f ' % (test_metrics['accuracy']))

        if (epoch%10 == 0):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model_' + str(epoch) + '.pth'
            log_string('Saving at %s' % savepath)
            state = {
                    'epoch': epoch,
                    'train_acc': train_instance_acc,
                    'test_acc': test_metrics['accuracy'],
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            torch.save(state, savepath)
            log_string('Saving model....')
            
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            
        global_epoch += 1



if __name__ == '__main__':
    args = parse_args()

    point_all_parts = np.load('data/vertex_rotation_augmented_sample.npy')
    label_all_parts = np.load('data/label_rotation_augmented_sample.npy')

    points_data = point_all_parts.transpose(0,2,1)
    #label_data = np.zeros([2632,1],dtype=int)
    targets_data = label_all_parts
    
    #points_train = points_data[528:2368,:,0:args.npoint]
    #label_train = label_data[528:2368,:]
    #targets_train = targets_data[528:2368,0:args.npoint]
    # test (528), val (264), train (1840)
    # Test (0:528)      (528:1056)          (1056:1584) (1584:2112) (2112:2632)
    # train (528:2368)  (0:264, 1056:2632)  (0:792, 1584:2632) (0:1320, 2112:2632)

    #points_train = np.concatenate([points_data[0:1320,:,0:args.npoint], points_data[2112:2632,:,0:args.npoint]], axis=0)
    #targets_train = np.concatenate([targets_data[0:1320,0:args.npoint], targets_data[2112:2632,0:args.npoint]], axis=0)

    points_train = points_data[528:2368,:,0:args.npoint]
    targets_train = targets_data[528:2368,0:args.npoint]

    points_valid = points_data[2368:2632,:,0:args.npoint]
    #label_valid = label_data[2368:2632,:]
    targets_valid = targets_data[2368:2632,0:args.npoint]
    

    main(args)
