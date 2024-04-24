import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import os
from matplotlib import pyplot as plt
from custom_dataset import PorosityDataset,PorosityDataset_mat,find_classes, PorosityDataset_mat_sigs
import random 
from tqdm import tqdm
import glob
from random import seed, shuffle
import argparse
import numpy as np
from torch.utils.data import DataLoader
import logging
### comprehensive file for camera calibration and perspective transformation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = str, default='/mnt/d/16_fatigue_bar_registration/mapped_dataset/slide_window/K_section_scalogram_update/')
    parser.add_argument("--net", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "resnet101", 'resnet152'])
    parser.add_argument("--loss_fn", type = str, default = "BCE", choices= ["BCE", "Poission", "Both"])
    parser.add_argument("--output_log_file", type=str, default="loss_cur.npy")
    parser.add_argument(
        "--train_list",
        "--list",
        nargs="+",
        help="<Required> corner points of MP image",
        required=True,
    )
    ## train_list = [batch_szie, epoch, learning rate]

    args = parser.parse_args()
    return args

def parse_input_points(args):
    points = args.train_list[0].split()
    points_int = np.array([eval(i) for i in points])  # transform the list of str to int
    points_int = np.reshape(points_int, (3, 1))
    assert points_int.shape == (
        3,
        1,
    )
    return points_int

def create_dataset(args):
    data_dir = args.data
    seed(42)
    images_t = glob.glob(data_dir + 'k*.mat')
    shuffle(images_t)
    [train_len, eval_len, test_len] = [int(len(images_t)*0.7), int(len(images_t)*0.1), int(len(images_t)*0.2)]
    train_indice = images_t[:train_len]
    eval_indice = images_t[train_len:train_len+eval_len]
    test_indice = images_t[train_len+eval_len:]

    transform_dataset = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    trainset = PorosityDataset_mat_sigs(train_indice, transform_dataset)
    evalset = PorosityDataset_mat_sigs(eval_indice,transform_dataset)
    testset = PorosityDataset_mat_sigs(test_indice,transform_dataset)
    return trainset, evalset, testset

def model_resnet(args):
    if args.net == "resnet18":
        model = models.resnet18()
    elif args.net == "resnet34":
        model = models.resnet34()
    elif args.net == "resnet50":
        model = models.resnet50()
    elif args.net == "resnet101":
        model = models.resnet101()
    elif args.net == "resnet152":
        model = models.resnet152()
    
    layer = model.conv1
    # Creating new Conv2d layer
    new_layer = nn.Conv2d(in_channels=7, 
                    out_channels=layer.out_channels, 
                    kernel_size=layer.kernel_size, 
                    stride=layer.stride, 
                    padding=layer.padding,
                    bias=layer.bias)
    model.conv1 = new_layer
    if args.loss_fn == "BCE":
        if args.net == "resnet50" or args.net == "resnet101":
            model.fc = nn.Linear(2048, 2)
        else:
            model.fc = nn.Linear(512,2)
    elif args.loss_fn == "Poission":
        model.fc = nn.Linear(512, 4)
    elif args.loss_fn == "Both":
        model.fc = nn.Linear(512,6)
    model = nn.DataParallel(model)
    model.to(device)
    return model


def trainer_resnet(args, trainset, evalset, testset, device, logger):
    args = parse_args()
    points_train = np.float32(parse_input_points(args))
    BATCH_SIZE = int(points_train[0][0])
    train_dataloader = DataLoader(dataset=trainset, 
                              batch_size= BATCH_SIZE, # how many samples per batch?
                              shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=testset, 
                                batch_size=15, 
                                shuffle=False) # don't usually need to shuffle testing data
    eval_dataloader = DataLoader(dataset=evalset, 
                                batch_size=50, 
                                shuffle=False) # don't usually need to shuffle testing data
    train_dataloader, test_dataloader, eval_dataloader

    trainSteps = len(trainset) // BATCH_SIZE
    evalSteps = len(evalset) // BATCH_SIZE

    model = model_resnet(args)
    model
    optimizer = optim.Adam(model.parameters(), lr= points_train[2][0])
    if args.loss_fn == "BCE":
        loss_fn = nn.CrossEntropyLoss()
        H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
        }
        cur_best = 0
        print('Start Training')
        for i in range(0, int(points_train[1][0])):
            model.train()
            Train_loss_cur,Val_loss_cur, train_correct, val_correct = 0,0,0,0
            counter = 0
            for (x,y) in train_dataloader:
                x=x.type(torch.float)
                (x, y) = (x.to(device), y.squeeze().type(torch.LongTensor).to(device))
                pred = model(x)
                loss = loss_fn(pred, y[:,0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                Train_loss_cur += loss.item()
                #print(pred, y)
                cur_correct = (pred.argmax(1) == y[:,0]).sum().item()
                train_correct += (pred.argmax(1) == y[:,0]).sum().item()
                if counter % 50 == 0:
                    print('Step {}/{}'.format(counter, trainSteps))
                    print('Loss cross_entropy: {}. Accuracy: {}.'.format(loss.item(), cur_correct/x.size()[0]))
                    logger.info('Step {}/{}'.format(counter, trainSteps))
                    logger.info('Loss cross_entropy: {}. Accuracy: {}.'.format(loss.item(), cur_correct/x.size()[0]))
                counter += 1  
            
            with torch.no_grad():
                model.eval()
                for (x,y) in eval_dataloader:
                    x=x.type(torch.float)
                    (x,y) = (x.to(device), y.squeeze().type(torch.LongTensor).to(device))
                    pred = model(x)
                    Val_loss_cur += loss_fn(pred,y[:,0]).item()
                    val_correct += (pred.argmax(1) == y[:,0]).sum().item()
            
            avg_trainloss = Train_loss_cur/trainSteps
            avg_evalloss = Val_loss_cur/evalSteps
            trainCorrect = train_correct / len(trainset)
            evalCorrect = val_correct / len(evalset)
            H["train_loss"].append(avg_trainloss)
            H["train_acc"].append(trainCorrect)
            H["val_loss"].append(avg_evalloss)
            H["val_acc"].append(evalCorrect)

            print("[INFO] EPOCH: {}/{}".format(i + 1, points_train[1]))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                avg_trainloss, trainCorrect))
            print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
                avg_evalloss, evalCorrect))
            logger.info("[INFO] EPOCH: {}/{}".format(i + 1, points_train[1]))
            logger.info("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                avg_trainloss, trainCorrect))
            logger.info("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
                avg_evalloss, evalCorrect))
            if evalCorrect  > cur_best:
                torch.save(model.state_dict(), args.net+'_Scalogram_10epoch_1e-3.pth')
                cur_best = evalCorrect
                print('Save current best model')

    elif args.loss_fn == "Poission":
        loss_fn = nn.PoissonNLLLoss()
        H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
        }
    elif args.loss_fn == "Both":
        loss_fn1 = nn.CrossEntropyLoss()
        loss_fn2 = nn. PoissonNLLLoss()
        H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
        }

        




if __name__ == "__main__":


    args = parse_args()

    logging.basicConfig(filename=args.net + "std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG) 
    points_train = np.float32(parse_input_points(args))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    [train_set, eval_set, test_set] = create_dataset(args)

    print('Dataset: \n Trainset: {}, Validation: {}, Test: {}'.format(len(train_set), 
                                                                      len(eval_set), len(test_set)))
    print(points_train)
    print(args.net + '.pth')
    trainer_resnet(args, train_set, eval_set, test_set, device, logger)



    








