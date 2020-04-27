"""Modified from: http://slazebni.cs.illinois.edu/fall18/assignment3_part2.html """

import os
import random
import cv2
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from resnet_yolo import resnet50
from yolo_loss import YoloLoss
from dataset import VocDetectorDataset
from eval_voc import evaluate
from predict import predict_image
from config import VOC_CLASSES, COLORS
from generate_csv import output_submission_csv
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_root_train = 'VOCdevkit_2007/VOC2007/JPEGImages/'
file_root_test = 'VOCdevkit_2007/VOC2007test/JPEGImages/'
annotation_file_train = 'voc2007.txt'
annotation_file_test = 'voc2007test.txt'

parser = argparse.ArgumentParser(description='YOLO Object detection')
parser.add_argument('--B', type=int, default=2,
                    help='number of bounding box predictions per cell')
parser.add_argument('--S', type=int, default=14,
                    help='width/height of network output grid')
parser.add_argument('--learning-rate', type=float, default=0.001,
                    help='Learning Rate')
parser.add_argument('--num-epochs', type=int, default=50,
                    help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=24,
                    help='Batch Size')
parser.add_argument('--name', type=str, required=False, default="",
                    help='Name of the experiment')
parser.add_argument('--lambda-coord', type=float, default=5,
                    help='Yolo loss component coefficient: λ in order to focus more on detection')
parser.add_argument('--lambda-noobj', type=float, default=0.5,
                    help='Yolo loss component coefficient: Down-weight loss from Class probability boxes that don’t contain objects')
parser.add_argument('--model-path', type=str, required=False, default=None,
                    help='Path to saved model')
parser.add_argument('--eval',  dest='eval', action='store_true', help='Evaluation mode')

def train(args, net, train_loader, test_loader):
    B = args.B
    S = args.S
    
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    
    lambda_coord = args.lambda_coord
    lambda_noobj = args.lambda_coord

    criterion = YoloLoss(S, B, lambda_coord, lambda_noobj)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    ## Train detector
    best_test_loss = np.inf

    for epoch in range(num_epochs):
        net.train()
        
        # Update learning rate late in training
        if epoch == 30 or epoch == 40:
            learning_rate /= 10.0

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))
        
        total_loss = 0.
        
        for i, (images, target) in enumerate(train_loader):
            images, target = images.to(device), target.to(device)
            
            pred = net(images)
            loss = criterion(pred,target)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                      % (epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
        args.summary_writer.add_scalar("loss/train",total_loss / (i+1), epoch)
        
        # evaluate the network on the test data
        with torch.no_grad():
            test_loss = 0.0
            net.eval()
            for i, (images, target) in enumerate(test_loader):
                images, target = images.to(device), target.to(device)

                pred = net(images)
                loss = criterion(pred,target)
                test_loss += loss.item()
            test_loss /= len(test_loader)
            args.summary_writer.add_scalar("loss/val",test_loss, epoch)
        
        if best_test_loss > test_loss:
            best_test_loss = test_loss
            print('Updating best test loss: %.5f' % best_test_loss)
            torch.save(net.state_dict(),'{}_best_detector.pth'.format(args.name) if len(args.name) else 'best_detector.pth' )
        torch.save(net.state_dict(),'{}_detector.pth'.format(args.name) if len(args.name) else 'detector.pth')


def main():
    global args
    args = parser.parse_args()
    setattr(args, "summary_writer", SummaryWriter(log_dir="tb_log/{}".format(args.name if len(args.name) else None )))

    load_network_path = args.model_path
    batch_size = args.batch_size
    S = args.S

    ''' To implement Yolo we will rely on a pretrained classifier as the backbone for our detection network. 
    PyTorch offers a variety of models which are pretrained on ImageNet in the [`torchvision.models`]
    (https://pytorch.org/docs/stable/torchvision/models.html) package. In particular, we will use the ResNet50
    architecture as a base for our detector. This is different from the base architecture in the Yolo paper
    and also results in a different output grid size (14x14 instead of 7x7). Models are typically pretrained on 
    ImageNet since the dataset is very large. The pretrained model provides a very useful weight initialization 
    for our detector, so that the network is able to learn quickly and effictively.
    '''

    if args.eval:
        if load_network_path is None:
            print("Model path not specified!!")
            exit(0)
        else:
            print('Loading saved network from {}'.format(load_network_path))
            net = resnet50().to(device)
            net.load_state_dict(torch.load(load_network_path))
        # To evaluate detection results we use mAP (mean of average precision over each class)
        net.eval()
        test_aps = evaluate(net, test_dataset_file=annotation_file_test)
        output_submission_csv('my_solution.csv', test_aps)
    else:
        pretrained = True
        # use to load a previously trained network
        if load_network_path is not None:
            print('Loading saved network from {}'.format(load_network_path))
            net = resnet50().to(device)
            net.load_state_dict(torch.load(load_network_path))
        else:
            print('Load pre-trained model')
            net = resnet50(pretrained=pretrained).to(device)

        ''' Since Pascal is a small dataset (5000 in train+val) we have combined the train and val splits
        to train our detector. The train dataset loader also using a variety of data augmentation techniques
        including random shift, scaling, crop, and flips. Data augmentation is slightly more complicated for 
        detection dataset since the bounding box annotations must be kept consistent through the transformations.
        Since the output of the dector network we train is an SxSx(B*5+C), we use an encoder to convert the 
        original bounding box coordinates into relative grid bounding box coordinates corresponding to the the
        expected output. We also use a decoder which allows us to convert the opposite direction into image 
        coordinate bounding boxes.
        '''
        train_dataset = VocDetectorDataset(root_img_dir=file_root_train,dataset_file=annotation_file_train,train=True, S=S)
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
        print('Loaded %d train images' % len(train_dataset))
        
        test_dataset = VocDetectorDataset(root_img_dir=file_root_test,dataset_file=annotation_file_test,train=False, S=S)
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
        print('Loaded %d test images' % len(test_dataset))

        train(args, net, train_loader, test_loader)

if __name__ == '__main__':
    main()