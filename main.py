#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time 
import random
from torch.optim import Adam
from facial_data_process import *
from models import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from utils import *
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import xgboost as xgb
import scipy.stats as st
import pickle
import sys
from ecg_data_process import create_ecg_data
from scipy.optimize import differential_evolution
from numpy import tensordot
from numpy.linalg import norm


    
def predict_img(args, model, test_loader, epoch, method, path, fold, n, test_ids): 
    results_path = path + '{}/'.format(method)
    if not os.path.exists(results_path):
            os.makedirs(results_path)  
    if args.test:
        checkpoint = torch.load(results_path + 'repeat{}_fold{}_epoch{}.pt'.format(n, fold, epoch))
        model.load_state_dict(checkpoint["model_state_dict"]) 
            
    results_f = results_path + '{}_restults.txt'.format(method) 
    
    if args.cuda:    
        model.cuda()
    model.eval()
    test_acc = 0
    total_num = 0
    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0
    correct_id_res = []
    incorrect_id_res = []
    correct_id_foc = []
    incorrect_id_foc = []
    
    y_pred = []
    y_true = []
    y_prob = []
    
    i = 0
    for x_test, test_y in test_loader:
        total_num += len(test_y)
        with torch.no_grad():
            yhats = model(to_gpu(args.cuda, x_test))
        
        yhats = yhats.cpu().detach().numpy()       
        ### the output probability        
        prob = [[1-y.item(),y.item()] for y in yhats]
        y_prob.extend(prob)
                
        ### if preds > 0.5, preds = 1, otherwise, preds = 0
        pred = [y.item() > 0.5 for y in yhats]
        y_pred.extend(list(map(int, pred)))
        
        test_y = test_y.detach().numpy()    
        y_true.extend(test_y.tolist())
        
        test_acc += sum(test_y == np.array(pred))
        
       
        ## calculate how many samples are predicted correctly.
        
     
        for t, p in zip(test_y, pred):
            if t == p and t.item() == 0:
                rest_true += 1
                correct_id_res.append(test_ids[i])
            elif t != p and t.item() == 0:
                focus_false += 1
                incorrect_id_res.append(test_ids[i])
            elif t == p and t.item() == 1:
                focus_true += 1
                correct_id_foc.append(test_ids[i])
            else:
                rest_false += 1
                incorrect_id_foc.append(test_ids[i])
            i+=1

    rest_precision = rest_true/(rest_true+rest_false)
    focus_precision = focus_true/(focus_true+focus_false)
    rest_recall = rest_true/(rest_true+focus_false)
    focus_recall = focus_true/(focus_true+rest_false)
    
    print("This is the {}th repeat {}th fold".format(n, fold))            
    print("rest_true is ",rest_true)  
    print("focus_false is ",focus_false)  
    print("focus_true is ",focus_true)    
    print("rest_false is ",rest_false)

    print('rest precision is',rest_precision)
    print('focus precision is',focus_precision)
    print('rest recall is',rest_recall)
    print('focus recall is',focus_recall)
    
    print("total number of samples is: ",total_num)
    acc = test_acc.item()/total_num
    print("test accuracy is {}".format(acc))
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("repeat {}, fold {} \n".format(n, fold))
        f.writelines("the number of rest samples that are correctly classified is {} \n".format(rest_true))
        f.writelines("the number of rest samples that are incorrectly classified is {} \n".format(focus_false))
        f.writelines("the number of focus samples that are correctly classified is {} \n".format(focus_true))
        f.writelines("the number of focus samples that are incorrectly classified is {} \n".format(rest_false))

        f.writelines("the rest precision is {} \n".format(rest_precision))
        f.writelines("the focus precision is {} \n".format(focus_precision))
        f.writelines("the rest recall is {} \n".format(rest_recall))
        f.writelines("the focus recall is {} \n".format(focus_recall))
        
        f.writelines("The test accracy of {} is {} \n".format(method, acc))
    
    ### save correctly classified sample idex into files
    pred_pkl = results_path + '/pred_pkl/'
    if not os.path.exists(pred_pkl):
        os.makedirs(pred_pkl) 
        
    pred_idx1 = pred_pkl + '{}_repeat{}_fold{}_pred_index1.pkl'.format(method, n, fold)
    pred_idx2 = pred_pkl + '{}_repeat{}_fold{}_pred_index2.pkl'.format(method, n, fold)
    true_ind = pred_pkl + '{}_repeat{}_fold{}_true_index.pkl'.format(method, n, fold)
    
    pred_arry = np.array([correct_id_res, incorrect_id_res, correct_id_foc, incorrect_id_foc], dtype=object)
    pickle.dump(pred_arry, open(pred_idx1, "wb"))

    pred_res = []
    pred_foc = []
    pred_res.extend(correct_id_res)
    pred_res.extend(incorrect_id_foc)
    pred_foc.extend(correct_id_foc)
    pred_foc.extend(incorrect_id_res)
    pred_res_foc = np.array([pred_res, pred_foc])
    pickle.dump(pred_res_foc, open(pred_idx2, "wb"))  

    true_res = []
    true_foc = []
    true_res.extend(correct_id_res)
    true_res.extend(incorrect_id_res)
    true_foc.extend(correct_id_foc)
    true_foc.extend(incorrect_id_foc)    
    
    true_res_foc = np.array([true_res, true_foc])
    pickle.dump(true_res_foc, open(true_ind, "wb")) 
    
    ### save the predicted probility and class label into files
    results_prob = pred_pkl + '{}_repeat{}_fold{}_probs.pkl'.format(method, n, fold)
    results_pred = pred_pkl + '{}_repeat{}_fold{}_preds.pkl'.format(method, n, fold)
    results_true =  pred_pkl + '{}_repeat{}_fold{}_true.pkl'.format(method, n, fold)  
    
    pickle.dump(y_prob, open(results_prob, "wb"))
    pickle.dump(y_pred, open(results_pred, "wb"))
    pickle.dump(y_true, open(results_true, "wb"))
           
    fig_path = results_path + "/{}_figures/".format(method)  
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)            
                
    plot_confusion2(y_true, y_pred, method, fig_path, fold, n, labels = [0,1])
    
    return  y_true, y_pred, y_prob, acc, rest_precision, focus_precision, rest_recall, focus_recall, correct_id_res, \
            incorrect_id_res, correct_id_foc, incorrect_id_foc

def predict_svm_xgb(args, model, x_test, y_test, path, method, fold, n, test_ids):
    rest_true = 0
    rest_false = 0
    focus_true = 0
    focus_false = 0  
    focus_pre_ls = []
    rest_pre_ls = []
    acc_ls = []
    
    correct_id_res = []
    incorrect_id_res = []
    correct_id_foc = []
    incorrect_id_foc = []
       
    results_path = path + '{}/'.format(method)   
    if not os.path.exists(results_path):
            os.makedirs(results_path) 
     
    pk_f = results_path +'repeat{}_fold{}.pkl'.format(n, fold)
    if args.test:
        with open(pk_f, 'rb') as f:
            model = pickle.load(f)
        
    results_f = results_path + '{}_restults.txt'.format(method)
    
    yhats = model.predict_proba(x_test)
    # y_prob = yhats[:,1]
    y_pred = np.argmax(yhats, axis=1).tolist()
    print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
  
    ## calculate how many samples are predicted correctly.
    
    for t, p, i in zip(y_test, y_pred, test_ids):
        if t == p and t.item() == 0:
            rest_true += 1
            correct_id_res.append(i)
        elif t != p and t.item() == 0:
            focus_false += 1
            incorrect_id_res.append(i)
        elif t == p and t.item() == 1:
            focus_true += 1
            correct_id_foc.append(i)
        else:
            rest_false += 1
            incorrect_id_foc.append(i)
        
            
    rest_precision = rest_true/(rest_true+rest_false)
    focus_precision = focus_true/(focus_true+focus_false)
    rest_recall = rest_true/(rest_true+focus_false)
    focus_recall = focus_true/(focus_true+rest_false)

    
    acc = accuracy_score(y_test, y_pred)
    
    rest_pre_ls.append(rest_precision)
    focus_pre_ls.append(focus_precision)
    acc_ls.append(acc)
    
    print("accuracy is %0.4f \n" % (acc))
    print("focus has %0.4f precision\n" % (focus_precision))
    print("rest has %0.4f precision \n" % (rest_precision))  
    print("focus has %0.4f recall\n" % (focus_recall))
    print("rest has %0.4f recall \n" % (rest_recall))     

    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.writelines("the number of rest samples that are correctly classified is {} \n".format(rest_true))
        f.writelines("the number of rest samples that are incorrectly classified is {} \n".format(focus_false))
        f.writelines("the number of focus samples that are correctly classified is {} \n".format(focus_true))
        f.writelines("the number of focus samples that are incorrectly classified is {} \n".format(rest_false))
        f.writelines("the rest precision is {} \n".format(rest_precision))
        f.writelines("the focus precision is {} \n".format(focus_precision))
        f.writelines("the rest recall is {} \n".format(rest_recall))
        f.writelines("the focus recall is {} \n".format(focus_recall))       
        f.writelines("The test accracy of {} is {} \n".format(method, acc))
        
    class_names = ['rest', 'focus']  
    plot_confusion(model, x_test, y_test, class_names, results_path, method, fold, n)
    
    ### save correctly classified sample idex into files
    pred_pkl = results_path + '/pred_pkl/'
    if not os.path.exists(pred_pkl):
        os.makedirs(pred_pkl) 
    pred_idx1 = pred_pkl + '{}_repeat{}_fold{}_pred_index1.pkl'.format(method, n, fold)
    pred_idx2 = pred_pkl + '{}_repeat{}_fold{}_pred_index2.pkl'.format(method, n, fold)
    true_ind = pred_pkl + '{}_repeat{}_fold{}_true_index.pkl'.format(method, n, fold)
    
    pred_arry = np.array([correct_id_res, incorrect_id_res, correct_id_foc, incorrect_id_foc],dtype=object)
    pickle.dump(pred_arry, open(pred_idx1, "wb"))

    pred_res = []
    pred_foc = []
    pred_res.extend(correct_id_res)
    pred_res.extend(incorrect_id_foc)
    pred_foc.extend(correct_id_foc)
    pred_foc.extend(incorrect_id_res)
    pred_res_foc = np.array([pred_res, pred_foc], dtype=object)
    pickle.dump(pred_res_foc, open(pred_idx2, "wb"))  

    true_res = []
    true_foc = []
    true_res.extend(correct_id_res)
    true_res.extend(incorrect_id_res)
    true_foc.extend(correct_id_foc)
    true_foc.extend(incorrect_id_foc)    
    
    true_res_foc = np.array([true_res, true_foc],dtype=object)
    pickle.dump(true_res_foc, open(true_ind, "wb")) 
    
    ### save the predicted probility and class label into files
    results_prob = pred_pkl + '{}_repeat{}_fold{}_probs.pkl'.format(method, n, fold)
    results_pred = pred_pkl + '{}_repeat{}_fold{}_preds.pkl'.format(method, n, fold)
    results_true =  pred_pkl + '{}_repeat{}_fold{}_true.pkl'.format(method, n, fold)  
    
    pickle.dump(yhats, open(results_prob, "wb"))
    pickle.dump(y_pred, open(results_pred, "wb"))
    pickle.dump(y_test, open(results_true, "wb"))
               
    
    return y_pred, yhats, acc, rest_precision, focus_precision, rest_recall, focus_recall, correct_id_res, \
           incorrect_id_res, correct_id_foc, incorrect_id_foc



def get_img_probs_for_ensemble(args, model, data_loader):
    """
    Get class probabilities from a PyTorch binary classifier.
    This is used only for learning ensemble weights on the training fold.
    It does not save files and does not use any test labels.
    """
    if args.cuda:
        model.cuda()

    model.eval()
    y_true = []
    y_prob = []

    with torch.no_grad():
        for x, y in data_loader:
            yhats = model(to_gpu(args.cuda, x))
            yhats = yhats.cpu().detach().numpy().reshape(-1)
            prob = np.stack([1.0 - yhats, yhats], axis=1)
            y_prob.append(prob)
            y_true.append(y.numpy().reshape(-1))

    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    return y_true, y_prob


def apply_fixed_weight_ensemble(preds, y_true, weights):
    """
    Apply fixed ensemble weights to the current test fold.
    Important: this function does NOT optimize weights using y_true.
    It only evaluates fixed weights.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)

    preds = [np.asarray(p, dtype=np.float64) for p in preds]
    y_prob = np.zeros_like(preds[0], dtype=np.float64)

    for w, p in zip(weights, preds):
        y_prob += w * p

    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.asarray(y_true).reshape(-1)

    acc = accuracy_score(y_true, y_pred)

    rest_true = int(np.sum((y_true == 0) & (y_pred == 0)))
    focus_false = int(np.sum((y_true == 0) & (y_pred == 1)))
    focus_true = int(np.sum((y_true == 1) & (y_pred == 1)))
    rest_false = int(np.sum((y_true == 1) & (y_pred == 0)))

    return acc, y_pred.tolist(), rest_true, focus_false, focus_true, rest_false


def optimal_wight_ensemble_no_test_leak(preds_train, y_train, preds_test, y_test):
    """
    Learn ensemble weights from the outer training fold only, then apply the
    fixed weights to the outer test fold.

    This replaces calls like:
        optimal_wight_ensemble(preds_test, y_test)

    because optimizing weights directly on y_test leaks test-label information.
    """
    y_train = np.asarray(y_train).reshape(-1)
    n_models = len(preds_train)

    def objective(weights):
        acc, _, _, _, _, _ = apply_fixed_weight_ensemble(preds_train, y_train, weights)
        return -acc

    bounds = [(0.0, 1.0)] * n_models
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=0,
        polish=True
    )

    weights = np.asarray(result.x, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)

    acc, y_pred, rest_true, focus_false, focus_true, rest_false = apply_fixed_weight_ensemble(
        preds_test,
        y_test,
        weights
    )

    return acc, y_pred, rest_true, focus_false, focus_true, rest_false, weights

def train_model(model, train_loader, test_loader, num_epochs, path, method, fold, n):
    results_path = path + '{}/'.format(method) 
    if not os.path.exists(results_path):
            os.makedirs(results_path)     

    
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = nn.BCELoss()
    if args.cuda:
        criterion = criterion.cuda()

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs): 
        losses = 0
        avg_loss = 0.
        val_loss = 0.
        model.train()
        # print(train_loader)
        for i, sample_batch in enumerate(train_loader):
        
            x_train, y_train = sample_batch
            preds = model(to_gpu(args.cuda, x_train))
            optimizer.zero_grad()

            loss = criterion(preds.squeeze(), to_gpu(args.cuda, y_train).float())
            
            loss.backward()
            optimizer.step()
            
            losses += loss.item()
            
        avg_loss = losses / len(train_loader)            
        train_losses.append(loss.item()) 

        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val = to_gpu(args.cuda, x_val)
                y_val = to_gpu(args.cuda, y_val)
                
                model.eval()
    
                yhat = model(x_val)
                val_l = criterion(yhat.squeeze(), y_val.float())
                val_loss += val_l.item()
                
            val_losses.append(val_loss)
            
        print('Repeat {} Fold {} Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(n, fold,
            epoch, num_epochs, i, len(train_loader), loss.item()))
        if epoch == num_epochs-1:
            checkpoint = results_path +'repeat{}_fold{}_epoch{}.pt'.format(n, fold, epoch+1)
            save_objs(model, epoch+1, avg_loss, optimizer, checkpoint)
  
    plot_loss(method, train_losses, val_losses, results_path)
    
    
def main(args,results_face, results_ecg):
    
    ###########################################################################
    ### create dataset for facial image
    image_path = './data/images_224x224/'
    
    image_dir = image_path + 'images/'
    img_csv = 'image.csv'
    
    # check_path_cnn = args.results + 'checkpt_cnn'
    # check_path_vgg = args.results + 'checkpt_vgg'
    
    k_folds = 5
    repeat = 3
    seed = [0, 42, 10]
       
    transform = transforms.Compose([
        transforms.ToTensor()
#        transforms.CenterCrop(224)
#        transforms.RandomHorizontalFlip(p =0.5),
#        transforms.RandomRotation(15),
#        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    _, _, dataset_224 = create_datasets(args.batch_size,transform, image_path, image_dir, img_csv)
     
    acc_ensem_fac0 = []
    acc_ensem_fac = []
    acc_vgg_fac = []
    acc_svm_fac = []
    acc_xgb_fac = []
    
    rest_pre_ensem_fac0 = []
    focus_pre_ensem_fac0 = []
    rest_pre_ensem_fac = []
    focus_pre_ensem_fac = []
    rest_pre_vgg_fac = []
    focus_pre_vgg_fac = []
    rest_pre_svm_fac = []
    focus_pre_svm_fac = []
    rest_pre_xgb_fac = []
    focus_pre_xgb_fac = []

    rest_rec_ensem_fac0 = []
    focus_rec_ensem_fac0 = []
    rest_rec_ensem_fac = []
    focus_rec_ensem_fac = []
    rest_rec_vgg_fac = []
    focus_rec_vgg_fac = []
    rest_rec_svm_fac = []
    focus_rec_svm_fac = []
    rest_rec_xgb_fac = []
    focus_rec_xgb_fac = []    
    
    
    
    x_224 = []
    y_224 = []    
    for i in range(len(dataset_224)):    
        x_224.append(dataset_224[i][0].numpy().flatten())
        y_224.append(dataset_224[i][1].numpy())
    x_224 = np.asarray(x_224)
    y_224 = np.asarray(y_224)
    
    ###########################################################################
    ### create dataset for ecg signals  
    ### ecg image dataset
    image_path_ecg = './data/ecg_img/'
    image_dir_ecg = image_path_ecg + 'images/'  
    batch_size_ecg = 256
    _, _, dataset_ecg_img = create_datasets(batch_size_ecg, transform, image_path_ecg, image_dir_ecg, img_csv)

    
    ### ecg signals
    x_ecg,y_ecg = create_ecg_data(time_s=480, window_s=3)
    tensor_x_ecg = torch.Tensor(x_ecg) 
    tensor_y_ecg = torch.Tensor(y_ecg)
    dataset_ecg = TensorDataset(tensor_x_ecg,tensor_y_ecg)

    acc_ensem_ecg0 = []
    acc_ensem_ecg = []
    acc_vgg_ecg = []
    acc_cnn_ecg = []    
    acc_svm_ecg = []
    acc_xgb_ecg = []
    
    rest_pre_ensem_ecg0 = []
    focus_pre_ensem_ecg0 = []
    rest_pre_ensem_ecg = []
    focus_pre_ensem_ecg = []
    rest_pre_vgg_ecg = []
    focus_pre_vgg_ecg = []
    rest_pre_cnn_ecg = []
    focus_pre_cnn_ecg = []
    rest_pre_svm_ecg = []
    focus_pre_svm_ecg = []
    rest_pre_xgb_ecg = []
    focus_pre_xgb_ecg = []

    rest_rec_ensem_ecg0 = []
    focus_rec_ensem_ecg0 = []
    rest_rec_ensem_ecg = []
    focus_rec_ensem_ecg = []
    rest_rec_vgg_ecg = []
    focus_rec_vgg_ecg = []
    rest_rec_cnn_ecg = []
    focus_rec_cnn_ecg = []
    rest_rec_svm_ecg = []
    focus_rec_svm_ecg = []
    rest_rec_xgb_ecg = []
    focus_rec_xgb_ecg = []    


    rest_pre_ensem_svm0 = []
    focus_pre_ensem_svm0 = []
    rest_rec_ensem_svm0= []
    focus_rec_ensem_svm0 = []
    acc_ensem_svm0 = []
    rest_pre_ensem_svm = []
    focus_pre_ensem_svm = []
    rest_rec_ensem_svm = []
    focus_rec_ensem_svm = []
    acc_ensem_svm = []
     
    
    rest_pre_ensem_xgb0 = []
    focus_pre_ensem_xgb0 = []
    rest_rec_ensem_xgb0 = []
    focus_rec_ensem_xgb0 = []
    acc_ensem_xgb0= []
    rest_pre_ensem_xgb = []
    focus_pre_ensem_xgb = []
    rest_rec_ensem_xgb = []
    focus_rec_ensem_xgb = []
    acc_ensem_xgb = []    

    rest_pre_ensem_vgg0 = []
    focus_pre_ensem_vgg0 = []
    rest_rec_ensem_vgg0 = []
    focus_rec_ensem_vgg0 = []
    acc_ensem_vgg0 = []
    rest_pre_ensem_vgg = []
    focus_pre_ensem_vgg = []
    rest_rec_ensem_vgg = []
    focus_rec_ensem_vgg = []
    acc_ensem_vgg = []
    
    rest_pre_ensem_vgg_cnn0 = []
    focus_pre_ensem_vgg_cnn0 = []
    rest_rec_ensem_vgg_cnn0 = []
    focus_rec_ensem_vgg_cnn0 = []
    acc_ensem_vgg_cnn0= []    
    rest_pre_ensem_vgg_cnn = []
    focus_pre_ensem_vgg_cnn = []
    rest_rec_ensem_vgg_cnn = []
    focus_rec_ensem_vgg_cnn = []
    acc_ensem_vgg_cnn = [] 

    rest_pre_ensem_ensem0 = []
    focus_pre_ensem_ensem0 = []
    rest_rec_ensem_ensem0 = []
    focus_rec_ensem_ensem0 = []
    acc_ensem_ensem0 = []    
    rest_pre_ensem_ensem1 = []
    focus_pre_ensem_ensem1 = []
    rest_rec_ensem_ensem1 = []
    focus_rec_ensem_ensem1 = []
    acc_ensem_ensem1 = []
    
    
    for n in range(repeat):
        kfold = KFold(n_splits=k_folds, random_state= seed[n], shuffle=True)
        
        vgg_correct_res = []
        vgg_incorrect_res = []
        vgg_correct_foc = []
        vgg_incorrect_foc = []
        
        svm_correct_res = []
        svm_incorrect_res = []
        svm_correct_foc = []
        svm_incorrect_foc = []

        xgb_correct_res = []
        xgb_incorrect_res = []
        xgb_correct_foc = []
        xgb_incorrect_foc = []
        
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset_224)):
            print(f'FOLD {fold}')
            start = datetime.now() 
            ###################################################################
            ### use facial images to train vgg, svm and xgboost
            ###################################################################
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            ### create trainset and testset for pretained vgg
            train_loader_224 = torch.utils.data.DataLoader(
                              dataset_224, batch_size=args.batch_size, sampler=train_subsampler)
            
            testset = torch.utils.data.Subset(dataset_224, test_ids.tolist())
            test_loader_224 = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

            # Ordered training loader used only to learn ensemble weights.
            # It is separate from train_loader_224, whose sampler is random.
            ensemble_trainset_224 = torch.utils.data.Subset(dataset_224, train_ids.tolist())
            ensemble_train_loader_224 = torch.utils.data.DataLoader(ensemble_trainset_224, batch_size=args.batch_size, shuffle=False)
            y_train_ensemble_224 = y_224[train_ids]
            
            ### create train set and test set for svm
            x_train_224, y_train_224 = x_224[train_ids], y_224[train_ids]
            x_test_224, y_test_224 = x_224[test_ids], y_224[test_ids]           
            x_train_224, y_train_224 = shuffle(x_train_224, y_train_224) ## only shuffle train dataset

            method1 = 'Pre-trained_VGG_video'
            vgg_face = alexnet()
            if args.cuda:
                vgg_face = vgg_face.cuda()
            reset_weights_vgg(vgg_face)
            num_epochs = 150
            
            start_vgg = datetime.now() 
            if not args.test:
                train_model(vgg_face, train_loader_224, test_loader_224, num_epochs, results_face, method1, fold, n)  

            
            ### calculate the training time of the model
            train_time = datetime.now()
            train_time_vgg = train_time - start_vgg
            print('the training time of method {} is {}'.format(method1, train_time_vgg))
            ### get the size of the model
            p = pickle.dumps(vgg_face)
            vgg_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method1,vgg_size ))
                
            y_vgg_f, pred_vgg_f, prob_vgg_f, acc_vgg_f, rest_pre_vgg_f, focus_pre_vgg_f, rest_rec_vgg_f, focus_rec_vgg_f,\
            correct_res_vgg, incorrect_res_vgg, correct_foc_vgg, incorrect_foc_vgg = \
            predict_img(args, vgg_face, test_loader_224, num_epochs, method1, results_face, fold, n, test_ids)        
            
            test_time_vgg = datetime.now() - train_time
            print('the testing time of method {} is {}'.format(method1, test_time_vgg))
            
            method2 = 'SVM_face'
            svm_face= svm.SVC(kernel='poly', probability=True)
            path_svm_face = results_face + '{}/'.format(method2) 
            if not os.path.exists(path_svm_face):
                os.makedirs(path_svm_face) 
            pk_svm_face = path_svm_face + 'repeat{}_fold{}.pkl'.format(n, fold)
            # svm_face= svm.SVC(kernel='rbf', C=5.0, probability=True)
            start_svm = datetime.now() 
            if not args.test:
                svm_face.fit(x_train_224,y_train_224)
            ### save trained svm_face
                with open(pk_svm_face, 'wb') as f1:
                    pickle.dump(svm_face, f1)
          
            ### calculate the training time of the model
            train_time = datetime.now()
            train_time_svm= train_time - start_svm
            print('the training time of method {} is {}'.format(method2, train_time_svm))   
            ### get the size of the model
            p = pickle.dumps(svm_face)
            svm_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method2,svm_size))
                   
            pred_svm_f, prob_svm_f, acc_svm_f, rest_pre_svm_f, focus_pre_svm_f, rest_rec_svm_f, focus_rec_svm_f, \
            correct_res_svm, incorrect_res_svm, correct_foc_svm, incorrect_foc_svm = \
                predict_svm_xgb(args, svm_face, x_test_224, y_test_224, results_face, method2, fold, n, test_ids)   
            
            test_time_svm= datetime.now() - train_time
            print('the testing time of method {} is {}'.format(method2, test_time_svm))
            
            method3 = 'XGBoost_face'
            xgb_face= xgb.XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=5, min_child_weight=1, gamma=0,  subsample=0.8,\
                                  colsample_bytree=0.8, objective= 'binary:logistic', scale_pos_weight=1, seed=27)  
                
            path_xgb_face = results_face + '{}/'.format(method3) 
            if not os.path.exists(path_xgb_face):
                os.makedirs(path_xgb_face) 
            pk_xgb_face = path_xgb_face + 'repeat{}_fold{}.pkl'.format(n, fold)
            
            start_xgb = datetime.now() 
            if not args.test:
                xgb_face.fit(x_train_224,y_train_224)
            
                ### save trained xgb_face
                with open(pk_xgb_face, 'wb') as f2:
                    pickle.dump(xgb_face, f2)

            ### calculate the training time of the model
            train_time = datetime.now()
            train_time_xgb= train_time - start_xgb
            print('the training time of method {} is {}'.format(method3, train_time_xgb))   
            ### get the size of the model
            p = pickle.dumps(xgb_face)
            xgb_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method3,xgb_size))            
                   
            pred_xgb_f, prob_xgb_f, acc_xgb_f, rest_pre_xgb_f, focus_pre_xgb_f, rest_rec_xgb_f, focus_rec_xgb_f, \
            correct_res_xgb, incorrect_res_xgb, correct_foc_xgb, incorrect_foc_xgb = \
                predict_svm_xgb(args, xgb_face, x_test_224, y_test_224, results_face, method3, fold, n, test_ids)
            
            # print('y_test_224 == y_vgg ?', y_test_224==y_vgg_f)
            test_time_xgb= datetime.now() - train_time
            print('the testing time of method {} is {}'.format(method3, test_time_xgb))

            # Learn optimized ensemble weights from the outer training fold only.
            # The current test fold labels are used only for final evaluation.
            _, prob_train_vgg_f = get_img_probs_for_ensemble(args, vgg_face, ensemble_train_loader_224)
            if args.test:
                with open(pk_svm_face, 'rb') as f1:
                    svm_face = pickle.load(f1)
                with open(pk_xgb_face, 'rb') as f2:
                    xgb_face = pickle.load(f2)
            prob_train_svm_f = svm_face.predict_proba(x_224[train_ids])
            prob_train_xgb_f = xgb_face.predict_proba(x_224[train_ids])
            

            preds = [prob_vgg_f, prob_svm_f, prob_xgb_f]
            preds_train = [prob_train_vgg_f, prob_train_svm_f, prob_train_xgb_f]
            acc_ensem_f0,y_ensem_f0, rest_true_f0,focus_false_f0, focus_true_f0, rest_false_f0 = equal_wight_ensemble(preds, y_test_224)
            acc_ensem_f,y_ensem_f, rest_true_f,focus_false_f, focus_true_f, rest_false_f, weights_f = optimal_wight_ensemble_no_test_leak(preds_train, y_train_ensemble_224, preds, y_test_224)
            print('face optimized ensemble weights learned from training fold:', weights_f)


            rest_pre_ensem_f0 = rest_true_f0/(rest_true_f0+rest_false_f0)
            focus_pre_ensem_f0 = focus_true_f0/(focus_true_f0+focus_false_f0)           
            rest_rec_ensem_f0 = rest_true_f0/(rest_true_f0+focus_false_f0)
            focus_rec_ensem_f0 = focus_true_f0/(focus_true_f0+rest_false_f0)
            rest_pre_ensem_f = rest_true_f/(rest_true_f+rest_false_f)
            focus_pre_ensem_f = focus_true_f/(focus_true_f+focus_false_f)           
            rest_rec_ensem_f = rest_true_f/(rest_true_f+focus_false_f)
            focus_rec_ensem_f = focus_true_f/(focus_true_f+rest_false_f)
                        
        
            rest_pre_ensem_fac0.append(rest_pre_ensem_f0)
            focus_pre_ensem_fac0.append(focus_pre_ensem_f0)
            rest_pre_ensem_fac.append(rest_pre_ensem_f)
            focus_pre_ensem_fac.append(focus_pre_ensem_f)            
            
            rest_pre_vgg_fac.append(rest_pre_vgg_f)
            focus_pre_vgg_fac.append(focus_pre_vgg_f)
            rest_pre_svm_fac.append(rest_pre_svm_f)
            focus_pre_svm_fac.append(focus_pre_svm_f)            
            rest_pre_xgb_fac.append(rest_pre_xgb_f)
            focus_pre_xgb_fac.append(focus_pre_xgb_f)  
            
            rest_rec_ensem_fac0.append(rest_rec_ensem_f0)
            focus_rec_ensem_fac0.append(focus_rec_ensem_f0)
            rest_rec_ensem_fac.append(rest_rec_ensem_f)
            focus_rec_ensem_fac.append(focus_rec_ensem_f)
            
            rest_rec_vgg_fac.append(rest_rec_vgg_f)
            focus_rec_vgg_fac.append(focus_rec_vgg_f)
            rest_rec_svm_fac.append(rest_rec_svm_f)
            focus_rec_svm_fac.append(focus_rec_svm_f)            
            rest_rec_xgb_fac.append(rest_rec_xgb_f)
            focus_rec_xgb_fac.append(focus_rec_xgb_f)  
            
            acc_ensem_fac0.append(acc_ensem_f0)
            acc_ensem_fac.append(acc_ensem_f)
            acc_vgg_fac.append(acc_vgg_f)
            acc_svm_fac.append(acc_svm_f)
            acc_xgb_fac.append(acc_xgb_f)
                                           
            fig_path = results_face + "/emsemble_figures/"
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plot_confusion2(y_vgg_f, y_ensem_f, args.method, fig_path, fold, n, labels = [0,1])    
          
                    
                
            ###################################################################
            ### use ecg signals to train 1d_cnn, vgg, svm and xgboost
            ###################################################################
            
            ### create train set and test set for pretained vgg
            train_loader_ecg_img = torch.utils.data.DataLoader(
                              dataset_ecg_img, batch_size=batch_size_ecg, sampler=train_subsampler)
            
            testset_ecg = torch.utils.data.Subset(dataset_ecg_img, test_ids.tolist())
            test_loader_ecg_img = torch.utils.data.DataLoader(testset_ecg, batch_size=batch_size_ecg, shuffle=False)

            # Ordered ECG-image training loader used only to learn ensemble weights.
            ensemble_trainset_ecg_img = torch.utils.data.Subset(dataset_ecg_img, train_ids.tolist())
            ensemble_train_loader_ecg_img = torch.utils.data.DataLoader(ensemble_trainset_ecg_img, batch_size=batch_size_ecg, shuffle=False)
            
                     
            method4 = 'Pretrained_VGG_ECG'
            vgg_ecg = alexnet()
            if args.cuda:
                vgg_ecg = vgg_ecg.cuda()
            reset_weights_vgg(vgg_ecg)
    
            num_epochs = 350
            start_vgg_ecg = datetime.now() 
            if not args.test:
                train_model(vgg_ecg, train_loader_ecg_img, test_loader_ecg_img, num_epochs, results_ecg, method4, fold, n)
                
            ### calculate the training time of the model
            train_time = datetime.now()
            train_time_vgg_ecg =  train_time - start_vgg_ecg
            print('the training time of method {} is {}'.format(method4, train_time_vgg_ecg))
            ### get the size of the model
            p = pickle.dumps(vgg_ecg)
            vgg_ecg_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method4, vgg_ecg_size))
            y_vgg_e, pred_vgg_e, prob_vgg_e, acc_vgg_e, rest_pre_vgg_e, focus_pre_vgg_e, rest_rec_vgg_e, focus_rec_vgg_e, y_vgg_e_p_r,\
            y_vgg_e_p_f, true_rest, true_focus = predict_img(args, vgg_ecg, test_loader_ecg_img, num_epochs, method4, results_ecg, fold, n, test_ids)  
            
            test_time_vgg_ecg =  datetime.now() - train_time
            print('the test time of method {} is {}'.format(method4, test_time_vgg_ecg))
            
            ### create train set and test set for 1d cnn
            train_loader_ecg = torch.utils.data.DataLoader(
                              dataset_ecg, batch_size=batch_size_ecg, sampler=train_subsampler)
            
            testset_ecg2 = torch.utils.data.Subset(dataset_ecg, test_ids.tolist())
            test_loader_ecg = torch.utils.data.DataLoader(testset_ecg2, batch_size=batch_size_ecg, shuffle=False)

            # Ordered ECG-signal training loader used only to learn ensemble weights.
            ensemble_trainset_ecg = torch.utils.data.Subset(dataset_ecg, train_ids.tolist())
            ensemble_train_loader_ecg = torch.utils.data.DataLoader(ensemble_trainset_ecg, batch_size=batch_size_ecg, shuffle=False)
            y_train_ensemble_ecg = y_ecg[train_ids]
            
            
            method5 = '1D_CNN_ECG'
            cnn_ecg=CNN_1d()
            if args.cuda:
                cnn_ecg = cnn_ecg.cuda()
                
            cnn_ecg.apply(reset_weights)
    
            num_epochs = 1500
            start_cnn_ecg = datetime.now() 
            if not args.test:
                train_model(cnn_ecg, train_loader_ecg, test_loader_ecg, num_epochs, results_ecg, method5, fold, n)
            
            ### calculate the training time of the model
            train_time = datetime.now()
            train_time_cnn_ecg = train_time - start_cnn_ecg
            print('the training time of method {} is {}'.format(method5, train_time_cnn_ecg))
            ### get the size of the model
            p = pickle.dumps(cnn_ecg)
            cnn_ecg_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method5, cnn_ecg_size))
            y_cnn_e, pred_cnn_e, prob_cnn_e, acc_cnn_e, rest_pre_cnn_e, focus_pre_cnn_e, rest_rec_cnn_e, focus_rec_cnn_e, \
            y_cnn_f_p_r, y_cnn_f_p_f, true_rest, true_focus =\
                predict_img(args, cnn_ecg, test_loader_ecg, num_epochs, method5, results_ecg, fold, n, test_ids)        
            
            test_time_cnn_ecg = datetime.now() - train_time
            print('the testing time of method {} is {}'.format(method5, test_time_cnn_ecg))
            
                
            ### create train set and test set for svm and xgboost
            x_train_ecg, y_train_ecg = x_ecg[train_ids], y_ecg[train_ids]
            x_test_ecg, y_test_ecg = x_ecg[test_ids], y_ecg[test_ids]           
            x_train_ecg, y_train_ecg = shuffle(x_train_ecg, y_train_ecg) ## only shuffle train dataset 
            
            method6 = 'SVM_ECG'
            svm_ecg = svm.SVC(kernel='rbf', probability=True)
            path_svm_ecg = results_ecg + '{}/'.format(method6) 
            if not os.path.exists(path_svm_ecg):
                os.makedirs(path_svm_ecg) 
            pk_svm_ecg = path_svm_ecg + 'repeat{}_fold{}.pkl'.format(n, fold)
            
            start_svm_ecg = datetime.now() 
            if not args.test:
                svm_ecg.fit(x_train_ecg, y_train_ecg)
            ### save trained svm_ecg
                with open(pk_svm_ecg, 'wb') as f3:
                    pickle.dump(svm_ecg, f3)
                                      
            ### calculate the training time of the model
            train_time = datetime.now()
            train_time_svm_ecg= train_time - start_svm_ecg
            print('the training time of method {} is {}'.format(method6, train_time_svm_ecg))   
            ### get the size of the model
            p = pickle.dumps(svm_ecg)
            svm_ecg_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method6,svm_ecg_size))
            pred_svm_e, prob_svm_e, acc_svm_e, rest_pre_svm_e, focus_pre_svm_e, rest_rec_svm_e, focus_rec_svm_e, y_svm_f_p_r,\
            y_svm_f_p_f, true_rest, true_focus = predict_svm_xgb(args, svm_ecg, x_test_ecg, y_test_ecg, results_ecg, method6, fold, n, test_ids)             
            test_time_svm_ecg= datetime.now() - train_time
            print('the testing time of method {} is {}'.format(method6, test_time_svm_ecg))
            
            
            method7 = 'XGBoost_ECG'
            xgb_ecg= xgb.XGBClassifier(learning_rate =0.1, n_estimators=50, max_depth=5, min_child_weight=1, gamma=0,  subsample=0.8,\
                                  colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)  
            path_xgb_ecg = results_ecg + '{}/'.format(method7) 
            if not os.path.exists(path_xgb_ecg):
                os.makedirs(path_xgb_ecg) 
            pk_xgb_ecg = path_xgb_ecg + 'repeat{}_fold{}.pkl'.format(n, fold)
                       
            start_xgb_ecg = datetime.now() 
            if not args.test:
                xgb_ecg.fit(x_train_ecg,y_train_ecg)
                ### save trained svm_ecg
                with open(pk_xgb_ecg, 'wb') as f4:
                    pickle.dump(xgb_ecg, f4)            
            
            ### calculate the traiing time of the model
            train_time = datetime.now()
            train_time_xgb_ecg= train_time - start_xgb_ecg
            print('the training time of method {} is {}'.format(method7, train_time_xgb_ecg))   
            ### get the size of the model
            p = pickle.dumps(xgb_ecg)
            xgb_ecg_size = sys.getsizeof(p)
            print("the size of method {} is {}".format(method7,xgb_ecg_size))            
            pred_xgb_e, prob_xgb_e, acc_xgb_e, rest_pre_xgb_e, focus_pre_xgb_e, rest_rec_xgb_e, focus_rec_xgb_e, y_xgb_f_p_r, y_xgb_f_p_f, \
            true_rest, true_focus = predict_svm_xgb(args, xgb_ecg, x_test_ecg, y_test_ecg, results_ecg, method7, fold, n, test_ids) 
            
            test_time_xgb_ecg= datetime.now() - train_time 
            print('the testing time of method {} is {}'.format(method7, test_time_xgb_ecg)) 
            # print('y_test_ecg == y_vgg_e ?', y_test_ecg==y_vgg_e)  
            # print('y_test_ecg == y_cnn_e',y_test_ecg == y_cnn_e)
            # print('y_test_224 == y_test_ecg',y_test_224== y_test_ecg)
            
            # Learn ECG ensemble weights from the outer training fold only.
            _, prob_train_vgg_e = get_img_probs_for_ensemble(args, vgg_ecg, ensemble_train_loader_ecg_img)
            _, prob_train_cnn_e = get_img_probs_for_ensemble(args, cnn_ecg, ensemble_train_loader_ecg)
            if args.test:
                with open(pk_svm_ecg, 'rb') as f3:
                    svm_ecg = pickle.load(f3)
                with open(pk_xgb_ecg, 'rb') as f4:
                    xgb_ecg = pickle.load(f4)
            prob_train_svm_e = svm_ecg.predict_proba(x_ecg[train_ids])
            prob_train_xgb_e = xgb_ecg.predict_proba(x_ecg[train_ids])

            # preds = [prob_vgg_e, prob_cnn_e, prob_svm_e, prob_xgb_e]
            preds = [prob_vgg_e, prob_svm_e, prob_xgb_e]
            preds_train = [prob_train_vgg_e, prob_train_svm_e, prob_train_xgb_e]
            acc_ensem_e0, y_ensem_e0, rest_true_e0, focus_false_e0, focus_true_e0, rest_false_e0 = equal_wight_ensemble(preds, y_test_224)           
            rest_pre_ensem_e0 = rest_true_e0/(rest_true_e0+rest_false_e0)
            focus_pre_ensem_e0 = focus_true_e0/(focus_true_e0+focus_false_e0)           
            rest_rec_ensem_e0 = rest_true_e0/(rest_true_e0+focus_false_e0)
            focus_rec_ensem_e0 = focus_true_e0/(focus_true_e0+rest_false_e0)
           
            acc_ensem_e, y_ensem_e, rest_true_e, focus_false_e, focus_true_e, rest_false_e, weights_e = optimal_wight_ensemble_no_test_leak(preds_train, y_train_ensemble_ecg, preds, y_test_224)
            print('ECG optimized ensemble weights learned from training fold:', weights_e)            
            rest_pre_ensem_e = rest_true_e/(rest_true_e+rest_false_e)
            focus_pre_ensem_e = focus_true_e/(focus_true_e+focus_false_e)           
            rest_rec_ensem_e = rest_true_e/(rest_true_e+focus_false_e)
            focus_rec_ensem_e = focus_true_e/(focus_true_e+rest_false_e)
            
            rest_pre_ensem_ecg0.append(rest_pre_ensem_e0)
            focus_pre_ensem_ecg0.append(focus_pre_ensem_e0)
            rest_pre_ensem_ecg.append(rest_pre_ensem_e)
            focus_pre_ensem_ecg.append(focus_pre_ensem_e)
            
            
            rest_pre_vgg_ecg.append(rest_pre_vgg_e)
            focus_pre_vgg_ecg.append(focus_pre_vgg_e)
            rest_pre_cnn_ecg.append(rest_pre_cnn_e)
            focus_pre_cnn_ecg.append(focus_pre_cnn_e)
            rest_pre_svm_ecg.append(rest_pre_svm_e)
            focus_pre_svm_ecg.append(focus_pre_svm_e)            
            rest_pre_xgb_ecg.append(rest_pre_xgb_e)
            focus_pre_xgb_ecg.append(focus_pre_xgb_e)  
            
            rest_rec_ensem_ecg0.append(rest_rec_ensem_e0)
            focus_rec_ensem_ecg0.append(focus_rec_ensem_e0)
            rest_rec_ensem_ecg.append(rest_rec_ensem_e)
            focus_rec_ensem_ecg.append(focus_rec_ensem_e)            
            
            rest_rec_vgg_ecg.append(rest_rec_vgg_e)
            focus_rec_vgg_ecg.append(focus_rec_vgg_e)
            rest_rec_cnn_ecg.append(rest_rec_cnn_e)
            focus_rec_cnn_ecg.append(focus_rec_cnn_e)
            rest_rec_svm_ecg.append(rest_rec_svm_e)
            focus_rec_svm_ecg.append(focus_rec_svm_e)            
            rest_rec_xgb_ecg.append(rest_rec_xgb_e)
            focus_rec_xgb_ecg.append(focus_rec_xgb_e)  
            
            acc_ensem_ecg0.append(acc_ensem_e0)
            acc_ensem_ecg.append(acc_ensem_e)
            
            acc_vgg_ecg.append(acc_vgg_e)
            acc_cnn_ecg.append(acc_cnn_e)
            acc_svm_ecg.append(acc_svm_e)
            acc_xgb_ecg.append(acc_xgb_e)
                                           
            fig_path = results_ecg + "/emsemble_figures/"
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plot_confusion2(y_vgg_e, y_ensem_e, args.method, fig_path, fold, n, labels = [0,1])  
            
            ###################################################################
            ## ensemble on each model.
            ### ensemble on svm
            preds = [prob_svm_f, prob_svm_e]
            acc_ensem_s0, y_ensem_svm0, rest_true_svm0, focus_false_svm0, focus_true_svm0, rest_false_svm0 = equal_wight_ensemble(preds, y_test_224)
            preds_train = [prob_train_svm_f, prob_train_svm_e]
            acc_ensem_s, y_ensem_svm, rest_true_svm, focus_false_svm, focus_true_svm, rest_false_svm, weights_svm = optimal_wight_ensemble_no_test_leak(preds_train, y_train_ensemble_224, preds, y_test_224)
            print('SVM optimized ensemble weights learned from training fold:', weights_svm)  

            rest_pre_ensem_s0 = rest_true_svm0/(rest_true_svm0+rest_false_svm0)
            focus_pre_ensem_s0 = focus_true_svm0/(focus_true_svm0+focus_false_svm0)           
            rest_rec_ensem_s0 = rest_true_svm0/(rest_true_svm0+focus_false_svm0)
            focus_rec_ensem_s0 = focus_true_svm0/(focus_true_svm0+rest_false_svm0)
            
            rest_pre_ensem_s = rest_true_svm/(rest_true_svm+rest_false_svm)
            focus_pre_ensem_s = focus_true_svm/(focus_true_svm+focus_false_svm)            
            rest_rec_ensem_s = rest_true_svm/(rest_true_svm+focus_false_svm)
            focus_rec_ensem_s = focus_true_svm/(focus_true_svm+rest_false_svm)            
            
            rest_pre_ensem_svm0.append(rest_pre_ensem_s0)
            focus_pre_ensem_svm0.append(focus_pre_ensem_s0)
            rest_rec_ensem_svm0.append(rest_rec_ensem_s0)
            focus_rec_ensem_svm0.append(focus_rec_ensem_s0)
            acc_ensem_svm0.append(acc_ensem_s0)

            rest_pre_ensem_svm.append(rest_pre_ensem_s)
            focus_pre_ensem_svm.append(focus_pre_ensem_s)
            rest_rec_ensem_svm.append(rest_rec_ensem_s)
            focus_rec_ensem_svm.append(focus_rec_ensem_s)
            acc_ensem_svm.append(acc_ensem_s)
            
            svm_en_fig_path = args.results + "/emsemble_svm_figures/"
            if not os.path.exists(svm_en_fig_path):
                os.makedirs(svm_en_fig_path)
            plot_confusion2(y_test_224, y_ensem_svm, args.method, svm_en_fig_path, fold, n, labels = [0,1])  

            ### ensemble on xgboost
            preds = [prob_xgb_f, prob_xgb_e]
            acc_ensem_x0, y_ensem_xgb0, rest_true_xgb0, focus_false_xgb0, focus_true_xgb0, rest_false_xgb0 = equal_wight_ensemble(preds, y_test_224)
            preds_train = [prob_train_xgb_f, prob_train_xgb_e]
            acc_ensem_x, y_ensem_xgb, rest_true_xgb, focus_false_xgb, focus_true_xgb, rest_false_xgb, weights_xgb = optimal_wight_ensemble_no_test_leak(preds_train, y_train_ensemble_224, preds, y_test_224)
            print('XGBoost optimized ensemble weights learned from training fold:', weights_xgb) 
            
        
            rest_pre_ensem_x0 = rest_true_xgb0/(rest_true_xgb0 +rest_false_xgb0)
            focus_pre_ensem_x0 = focus_true_xgb0/(focus_true_xgb0 + focus_false_xgb0)
            rest_rec_ensem_x0 = rest_true_xgb0/(rest_true_xgb0 + focus_false_xgb0)
            focus_rec_ensem_x0 = focus_true_xgb0/(focus_true_xgb0 + rest_false_xgb0)
            
            rest_pre_ensem_x = rest_true_xgb/(rest_true_xgb +rest_false_xgb)
            focus_pre_ensem_x = focus_true_xgb/(focus_true_xgb + focus_false_xgb)
            rest_rec_ensem_x = rest_true_xgb/(rest_true_xgb + focus_false_xgb)
            focus_rec_ensem_x = focus_true_xgb/(focus_true_xgb + rest_false_xgb)            
            
            rest_pre_ensem_xgb0.append(rest_pre_ensem_x0)
            focus_pre_ensem_xgb0.append(focus_pre_ensem_x0)
            rest_rec_ensem_xgb0.append(rest_rec_ensem_x0)
            focus_rec_ensem_xgb0.append(focus_rec_ensem_x0)
            acc_ensem_xgb0.append(acc_ensem_x0)

            rest_pre_ensem_xgb.append(rest_pre_ensem_x)
            focus_pre_ensem_xgb.append(focus_pre_ensem_x)
            rest_rec_ensem_xgb.append(rest_rec_ensem_x)
            focus_rec_ensem_xgb.append(focus_rec_ensem_x)
            acc_ensem_xgb.append(acc_ensem_x)
            
            xgb_en_fig_path = args.results + "/emsemble_xgb_figures/"
            if not os.path.exists(xgb_en_fig_path):
                os.makedirs(xgb_en_fig_path)
            plot_confusion2(y_test_224, y_ensem_xgb, args.method, xgb_en_fig_path, fold, n, labels = [0,1])  
            
            ### ensemble on vgg
            preds = [prob_vgg_f, prob_vgg_e]
            acc_ensem_v0, y_ensem_vgg0, rest_true_vgg0, focus_false_vgg0, focus_true_vgg0, rest_false_vgg0 = equal_wight_ensemble(preds, y_test_224)
            preds_train = [prob_train_vgg_f, prob_train_vgg_e]
            acc_ensem_v, y_ensem_vgg, rest_true_vgg, focus_false_vgg, focus_true_vgg, rest_false_vgg, weights_vgg = optimal_wight_ensemble_no_test_leak(preds_train, y_train_ensemble_224, preds, y_test_224)
            print('VGG optimized ensemble weights learned from training fold:', weights_vgg)                        
            
        
            rest_pre_ensem_v0 = rest_true_vgg0/(rest_true_vgg0 +rest_false_vgg0)
            focus_pre_ensem_v0 = focus_true_vgg0/(focus_true_vgg0 + focus_false_vgg0)
            rest_rec_ensem_v0 = rest_true_vgg0/(rest_true_vgg0 + focus_false_vgg0)
            focus_rec_ensem_v0 = focus_true_vgg0/(focus_true_vgg0 + rest_false_vgg0)
            
            rest_pre_ensem_v = rest_true_vgg/(rest_true_vgg +rest_false_vgg)
            focus_pre_ensem_v = focus_true_vgg/(focus_true_vgg + focus_false_vgg)
            rest_rec_ensem_v = rest_true_vgg/(rest_true_vgg + focus_false_vgg)
            focus_rec_ensem_v = focus_true_vgg/(focus_true_vgg + rest_false_vgg)
            
            rest_pre_ensem_vgg0.append(rest_pre_ensem_v0)
            focus_pre_ensem_vgg0.append(focus_pre_ensem_v0)
            rest_rec_ensem_vgg0.append(rest_rec_ensem_v0)
            focus_rec_ensem_vgg0.append(focus_rec_ensem_v0)
            acc_ensem_vgg0.append(acc_ensem_v0)

            rest_pre_ensem_vgg.append(rest_pre_ensem_v)
            focus_pre_ensem_vgg.append(focus_pre_ensem_v)
            rest_rec_ensem_vgg.append(rest_rec_ensem_v)
            focus_rec_ensem_vgg.append(focus_rec_ensem_v)
            acc_ensem_vgg.append(acc_ensem_v)
            
            vgg_en_fig_path = args.results + "/emsemble_vgg_figures/"
            if not os.path.exists(vgg_en_fig_path):
                os.makedirs(vgg_en_fig_path)
            plot_confusion2(y_test_224, y_ensem_vgg, args.method, vgg_en_fig_path, fold, n, labels = [0,1])  
            
            ### ensemble of vgg and 1d_cnn
            preds = [prob_vgg_f, prob_cnn_e]
            acc_ensem_v_c0, y_ensem_vgg_cnn0, rest_true_vgg_cnn0, focus_false_vgg_cnn0, focus_true_vgg_cnn0, rest_false_vgg_cnn0 = equal_wight_ensemble(preds, y_test_224)
            preds_train = [prob_train_vgg_f, prob_train_cnn_e]
            acc_ensem_v_c, y_ensem_vgg_cnn, rest_true_vgg_cnn, focus_false_vgg_cnn, focus_true_vgg_cnn, rest_false_vgg_cnn, weights_vgg_cnn = optimal_wight_ensemble_no_test_leak(preds_train, y_train_ensemble_224, preds, y_test_224)
            print('VGG + CNN optimized ensemble weights learned from training fold:', weights_vgg_cnn)                        
            
        
            rest_pre_ensem_v_c0 = rest_true_vgg_cnn0/(rest_true_vgg_cnn0 +rest_false_vgg_cnn0)
            focus_pre_ensem_v_c0 = focus_true_vgg_cnn0/(focus_true_vgg_cnn0 + focus_false_vgg_cnn0)
            rest_rec_ensem_v_c0 = rest_true_vgg_cnn0/(rest_true_vgg_cnn0 + focus_false_vgg_cnn0)
            focus_rec_ensem_v_c0 = focus_true_vgg_cnn0/(focus_true_vgg_cnn0 + rest_false_vgg_cnn0)
            
            rest_pre_ensem_v_c = rest_true_vgg_cnn/(rest_true_vgg_cnn +rest_false_vgg_cnn)
            focus_pre_ensem_v_c = focus_true_vgg_cnn/(focus_true_vgg_cnn + focus_false_vgg_cnn)
            rest_rec_ensem_v_c = rest_true_vgg_cnn/(rest_true_vgg_cnn + focus_false_vgg_cnn)
            focus_rec_ensem_v_c = focus_true_vgg_cnn/(focus_true_vgg_cnn + rest_false_vgg_cnn)
            
            rest_pre_ensem_vgg_cnn0.append(rest_pre_ensem_v_c0)
            focus_pre_ensem_vgg_cnn0.append(focus_pre_ensem_v_c0)
            rest_rec_ensem_vgg_cnn0.append(rest_rec_ensem_v_c0)
            focus_rec_ensem_vgg_cnn0.append(focus_rec_ensem_v_c0)
            acc_ensem_vgg_cnn0.append(acc_ensem_v_c0)

            rest_pre_ensem_vgg_cnn.append(rest_pre_ensem_v_c)
            focus_pre_ensem_vgg_cnn.append(focus_pre_ensem_v_c)
            rest_rec_ensem_vgg_cnn.append(rest_rec_ensem_v_c)
            focus_rec_ensem_vgg_cnn.append(focus_rec_ensem_v_c)
            acc_ensem_vgg_cnn.append(acc_ensem_v_c) 
                       
            vgg_1dcnn_fig_path = args.results + "/emsemble_vgg_1dcnn_figures/"
            if not os.path.exists(vgg_1dcnn_fig_path):
                os.makedirs(vgg_1dcnn_fig_path)
            plot_confusion2(y_test_224, y_ensem_vgg_cnn, args.method, vgg_1dcnn_fig_path, fold, n, labels = [0,1])              
            
            ### ensemble of all methods
            # preds = [prob_vgg_f, prob_svm_f, prob_xgb_f, prob_vgg_e, prob_cnn_e, prob_svm_e, prob_xgb_e]
            preds = [prob_vgg_f, prob_svm_f, prob_xgb_f, prob_vgg_e, prob_svm_e, prob_xgb_e]
            acc_ensem_en0, y_ensem_en0, rest_true_en0, focus_false_en0, focus_true_en0, rest_false_en0 = equal_wight_ensemble(preds, y_test_224)
            preds_train = [prob_train_vgg_f, prob_train_svm_f, prob_train_xgb_f, prob_train_vgg_e, prob_train_svm_e, prob_train_xgb_e]
            acc_ensem_en, y_ensem_en, rest_true_en, focus_false_en, focus_true_en, rest_false_en, weights_all = optimal_wight_ensemble_no_test_leak(preds_train, y_train_ensemble_224, preds, y_test_224)
            print('All-method optimized ensemble weights learned from training fold:', weights_all) 
            
        
            rest_pre_ensem_en0 = rest_true_en0/(rest_true_en0 +rest_false_en0)
            focus_pre_ensem_en0 = focus_true_en0/(focus_true_en0 + focus_false_en0)
            rest_rec_ensem_en0 = rest_true_en0/(rest_true_en0 + focus_false_en0)
            focus_rec_ensem_en0 = focus_true_en0/(focus_true_en0 + rest_false_en0)
            
            rest_pre_ensem_en = rest_true_en/(rest_true_en +rest_false_en)
            focus_pre_ensem_en = focus_true_en/(focus_true_en + focus_false_en)
            rest_rec_ensem_en = rest_true_en/(rest_true_en + focus_false_en)
            focus_rec_ensem_en = focus_true_en/(focus_true_en + rest_false_en)            
            
            rest_pre_ensem_ensem0.append(rest_pre_ensem_en0)
            focus_pre_ensem_ensem0.append(focus_pre_ensem_en0)
            rest_rec_ensem_ensem0.append(rest_rec_ensem_en0)
            focus_rec_ensem_ensem0.append(focus_rec_ensem_en0)
            acc_ensem_ensem0.append(acc_ensem_en0)

            rest_pre_ensem_ensem1.append(rest_pre_ensem_en)
            focus_pre_ensem_ensem1.append(focus_pre_ensem_en)
            rest_rec_ensem_ensem1.append(rest_rec_ensem_en)
            focus_rec_ensem_ensem1.append(focus_rec_ensem_en)
            acc_ensem_ensem1.append(acc_ensem_en)
            
            all_fig_path = args.results + "/emsemble_figures/"
            if not os.path.exists(all_fig_path):
                os.makedirs(all_fig_path)
            plot_confusion2(y_test_224, y_ensem_en, args.method, all_fig_path, fold, n, labels = [0,1])
            
            end = datetime.now() - start   
            
            
            vgg_correct_res.extend(correct_res_vgg)
            vgg_incorrect_res.extend(incorrect_res_vgg)
            vgg_correct_foc.extend(correct_foc_vgg)
            vgg_incorrect_foc.extend(incorrect_foc_vgg)
            
            svm_correct_res.extend(correct_res_svm)
            svm_incorrect_res.extend(incorrect_res_svm)
            svm_correct_foc.extend(correct_foc_svm)
            svm_incorrect_foc.extend(incorrect_foc_svm)

            xgb_correct_res.extend(correct_res_xgb)
            xgb_incorrect_res.extend(incorrect_res_xgb)
            xgb_correct_foc.extend(correct_foc_xgb)
            xgb_incorrect_foc.extend(incorrect_foc_xgb)
            
            
        ### print incorrectly classified rest images by svm, xgboost and vgg.
        imgs_path = results_face + "/classified_images/"
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path) 
            
        prediction_images(vgg_correct_res, vgg_incorrect_res, vgg_correct_foc, vgg_incorrect_foc, dataset_224, imgs_path, method1, n) 
        prediction_images(svm_correct_res, svm_incorrect_res, svm_correct_foc, svm_incorrect_foc, dataset_224, imgs_path, method2, n)
        prediction_images(xgb_correct_res, xgb_incorrect_res, xgb_correct_foc, xgb_incorrect_foc, dataset_224, imgs_path, method3, n)
    
    ###########################################################################
    #### results of facial images based methods
    ### results of accuracy
    acc_ensem_mean_f0 = np.array(acc_ensem_fac0).mean()
    acc_ensem_std_f0 = np.array(acc_ensem_fac0).std()
    acc_ensem_conf_f0 = st.t.interval(0.95, len(acc_ensem_fac0)-1, loc=np.mean(acc_ensem_fac0), scale=st.sem(acc_ensem_fac0))
    acc_ensem_mean_f = np.array(acc_ensem_fac).mean()
    acc_ensem_std_f = np.array(acc_ensem_fac).std()
    acc_ensem_conf_f = st.t.interval(0.95, len(acc_ensem_fac)-1, loc=np.mean(acc_ensem_fac), scale=st.sem(acc_ensem_fac))    
    
    acc_vgg_mean_f = np.array(acc_vgg_fac).mean()
    acc_vgg_std_f = np.array(acc_vgg_fac).std()
    acc_vgg_conf_f = st.t.interval(0.95, len(acc_vgg_fac)-1, loc=np.mean(acc_vgg_fac), scale=st.sem(acc_vgg_fac))
    acc_svm_mean_f = np.array(acc_svm_fac).mean()
    acc_svm_std_f = np.array(acc_svm_fac).std()
    acc_svm_conf_f = st.t.interval(0.95, len(acc_svm_fac)-1, loc=np.mean(acc_svm_fac), scale=st.sem(acc_svm_fac))
    acc_xgb_mean_f = np.array(acc_xgb_fac).mean()
    acc_xgb_std_f = np.array(acc_xgb_fac).std()
    acc_xgb_conf_f = st.t.interval(0.95, len(acc_xgb_fac)-1, loc=np.mean(acc_xgb_fac), scale=st.sem(acc_xgb_fac))

    ### results of precision
    rest_pre_ensem_mean_f0 = np.array(rest_pre_ensem_fac0).mean()
    rest_pre_ensem_std_f0 = np.array(rest_pre_ensem_fac0).std() 
    rest_pre_ensem_conf_f0 = st.t.interval(0.95, len(rest_pre_ensem_fac0)-1, loc=np.mean(rest_pre_ensem_fac0), scale=st.sem(rest_pre_ensem_fac0)) 
    focus_pre_ensem_mean_f0 = np.array(focus_pre_ensem_fac0).mean()
    focus_pre_ensem_std_f0 = np.array(focus_pre_ensem_fac0).std()
    focus_pre_ensem_conf_f0 = st.t.interval(0.95, len(focus_pre_ensem_fac0)-1, loc=np.mean(focus_pre_ensem_fac0), scale=st.sem(focus_pre_ensem_fac0))
    
    rest_pre_ensem_mean_f = np.array(rest_pre_ensem_fac).mean()
    rest_pre_ensem_std_f = np.array(rest_pre_ensem_fac).std() 
    rest_pre_ensem_conf_f = st.t.interval(0.95, len(rest_pre_ensem_fac)-1, loc=np.mean(rest_pre_ensem_fac), scale=st.sem(rest_pre_ensem_fac))
    focus_pre_ensem_mean_f = np.array(focus_pre_ensem_fac).mean()
    focus_pre_ensem_std_f = np.array(focus_pre_ensem_fac).std()
    focus_pre_ensem_conf_f = st.t.interval(0.95, len(focus_pre_ensem_fac)-1, loc=np.mean(focus_pre_ensem_fac), scale=st.sem(focus_pre_ensem_fac))
    
    rest_pre_vgg_mean_f = np.array(rest_pre_vgg_fac).mean()
    rest_pre_vgg_std_f = np.array(rest_pre_vgg_fac).std() 
    rest_pre_vgg_conf_f = st.t.interval(0.95, len(rest_pre_vgg_fac)-1, loc=np.mean(rest_pre_vgg_fac), scale=st.sem(rest_pre_vgg_fac))
    focus_pre_vgg_mean_f = np.array(focus_pre_vgg_fac).mean()
    focus_pre_vgg_std_f = np.array(focus_pre_vgg_fac).std()
    focus_pre_vgg_conf_f = st.t.interval(0.95, len(focus_pre_vgg_fac)-1, loc=np.mean(focus_pre_vgg_fac), scale=st.sem(focus_pre_vgg_fac))

    rest_pre_svm_mean_f = np.array(rest_pre_svm_fac).mean()
    rest_pre_svm_std_f = np.array(rest_pre_svm_fac).std()  
    rest_pre_svm_conf_f = st.t.interval(0.95, len(rest_pre_svm_fac)-1, loc=np.mean(rest_pre_svm_fac), scale=st.sem(rest_pre_svm_fac))
    focus_pre_svm_mean_f = np.array(focus_pre_svm_fac).mean()
    focus_pre_svm_std_f = np.array(focus_pre_svm_fac).std() 
    focus_pre_svm_conf_f = st.t.interval(0.95, len(focus_pre_svm_fac)-1, loc=np.mean(focus_pre_svm_fac), scale=st.sem(focus_pre_svm_fac))
    
    rest_pre_xgb_mean_f = np.array(rest_pre_xgb_fac).mean()
    rest_pre_xgb_std_f = np.array(rest_pre_xgb_fac).std() 
    rest_pre_xgb_conf_f = st.t.interval(0.95, len(rest_pre_xgb_fac)-1, loc=np.mean(rest_pre_xgb_fac), scale=st.sem(rest_pre_xgb_fac))
    focus_pre_xgb_mean_f = np.array(focus_pre_xgb_fac).mean()
    focus_pre_xgb_std_f = np.array(focus_pre_xgb_fac).std()
    focus_pre_xgb_conf_f = st.t.interval(0.95, len(focus_pre_xgb_fac)-1, loc=np.mean(focus_pre_xgb_fac), scale=st.sem(focus_pre_xgb_fac))
    
    ### results of recall
    rest_rec_ensem_mean_f0 = np.array(rest_rec_ensem_fac0).mean()
    rest_rec_ensem_std_f0 = np.array(rest_rec_ensem_fac0).std() 
    rest_rec_ensem_conf_f0 = st.t.interval(0.95, len(rest_rec_ensem_fac0)-1, loc=np.mean(rest_rec_ensem_fac0), scale=st.sem(rest_rec_ensem_fac0))
    focus_rec_ensem_mean_f0 = np.array(focus_rec_ensem_fac0).mean()
    focus_rec_ensem_std_f0 = np.array(focus_rec_ensem_fac0).std()
    focus_rec_ensem_conf_f0 = st.t.interval(0.95, len(focus_rec_ensem_fac0)-1, loc=np.mean(focus_rec_ensem_fac0), scale=st.sem(focus_rec_ensem_fac0))
    
    rest_rec_ensem_mean_f = np.array(rest_rec_ensem_fac).mean()
    rest_rec_ensem_std_f = np.array(rest_rec_ensem_fac).std() 
    rest_rec_ensem_conf_f = st.t.interval(0.95, len(rest_rec_ensem_fac)-1, loc=np.mean(rest_rec_ensem_fac), scale=st.sem(rest_rec_ensem_fac))
    focus_rec_ensem_mean_f = np.array(focus_rec_ensem_fac).mean()
    focus_rec_ensem_std_f = np.array(focus_rec_ensem_fac).std()
    focus_rec_ensem_conf_f = st.t.interval(0.95, len(focus_rec_ensem_fac)-1, loc=np.mean(focus_rec_ensem_fac), scale=st.sem(focus_rec_ensem_fac))
    
    rest_rec_vgg_mean_f = np.array(rest_rec_vgg_fac).mean()
    rest_rec_vgg_std_f = np.array(rest_rec_vgg_fac).std() 
    rest_rec_vgg_conf_f = st.t.interval(0.95, len(rest_rec_vgg_fac)-1, loc=np.mean(rest_rec_vgg_fac), scale=st.sem(rest_rec_vgg_fac))
    focus_rec_vgg_mean_f = np.array(focus_rec_vgg_fac).mean()
    focus_rec_vgg_std_f = np.array(focus_rec_vgg_fac).std()
    focus_rec_vgg_conf_f = st.t.interval(0.95, len(focus_rec_vgg_fac)-1, loc=np.mean(focus_rec_vgg_fac), scale=st.sem(focus_rec_vgg_fac))

    rest_rec_svm_mean_f = np.array(rest_rec_svm_fac).mean()
    rest_rec_svm_std_f = np.array(rest_rec_svm_fac).std()  
    rest_rec_svm_conf_f = st.t.interval(0.95, len(rest_rec_svm_fac)-1, loc=np.mean(rest_rec_svm_fac), scale=st.sem(rest_rec_svm_fac))
    focus_rec_svm_mean_f = np.array(focus_rec_svm_fac).mean()
    focus_rec_svm_std_f = np.array(focus_rec_svm_fac).std() 
    focus_rec_svm_conf_f = st.t.interval(0.95, len(focus_rec_svm_fac)-1, loc=np.mean(focus_rec_svm_fac), scale=st.sem(focus_rec_svm_fac))
    
    rest_rec_xgb_mean_f = np.array(rest_rec_xgb_fac).mean()
    rest_rec_xgb_std_f = np.array(rest_rec_xgb_fac).std() 
    rest_rec_xgb_conf_f = st.t.interval(0.95, len(rest_rec_xgb_fac)-1, loc=np.mean(rest_rec_xgb_fac), scale=st.sem(rest_rec_xgb_fac))
    focus_rec_xgb_mean_f = np.array(focus_rec_xgb_fac).mean()
    focus_rec_xgb_std_f = np.array(focus_rec_xgb_fac).std()
    focus_rec_xgb_conf_f = st.t.interval(0.95, len(focus_rec_xgb_fac)-1, loc=np.mean(focus_rec_xgb_fac), scale=st.sem(focus_rec_xgb_fac))    
    
   
    
    ### print accuracy
    print("facial image equal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_f0, acc_ensem_std_f0,acc_ensem_conf_f0[0],acc_ensem_conf_f0[1]))
    print("facial image optimal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_f, acc_ensem_std_f,acc_ensem_conf_f[0],acc_ensem_conf_f[1]))
    print("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method1, acc_vgg_mean_f, acc_vgg_std_f, acc_vgg_conf_f[0],acc_vgg_conf_f[1]))
    print("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method2, acc_svm_mean_f, acc_svm_std_f, acc_svm_conf_f[0],acc_svm_conf_f[1]))
    print("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method3, acc_xgb_mean_f, acc_xgb_std_f, acc_xgb_conf_f[0],acc_xgb_conf_f[1]))
    
    ### print precision
    print("facial image equal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_f0, rest_pre_ensem_std_f0, rest_pre_ensem_conf_f0[0], rest_pre_ensem_conf_f0[1]))
    print("facial image equal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_f0, focus_pre_ensem_std_f0, focus_pre_ensem_conf_f0[0],focus_pre_ensem_conf_f0[1]))
    print("facial image optimal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_f, rest_pre_ensem_std_f, rest_pre_ensem_conf_f[0], rest_pre_ensem_conf_f[1]))
    print("facial image optimal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_f, focus_pre_ensem_std_f, focus_pre_ensem_conf_f[0],focus_pre_ensem_conf_f[1]))
    print("facial image method vgg: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_vgg_mean_f, rest_pre_vgg_std_f, rest_pre_vgg_conf_f[0],rest_pre_vgg_conf_f[1]))
    print("facial image method vgg: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_vgg_mean_f, focus_pre_vgg_std_f, focus_pre_vgg_conf_f[0], focus_pre_vgg_conf_f[1]))
    print("facial image method svm: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_svm_mean_f, rest_pre_svm_std_f, rest_pre_svm_conf_f[0], rest_pre_svm_conf_f[1]))
    print("facial image method svm: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_svm_mean_f, focus_pre_svm_std_f, focus_pre_svm_conf_f[0], focus_pre_svm_conf_f[1]))   
    print("facial image method xgb: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_xgb_mean_f, rest_pre_xgb_std_f, rest_pre_xgb_conf_f[0], rest_pre_xgb_conf_f[1]))
    print("facial image method xgb: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_xgb_mean_f, focus_pre_xgb_std_f, focus_pre_xgb_conf_f[0], focus_pre_xgb_conf_f[1]))

    ### print recall
    print("facial image equal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_f0, rest_rec_ensem_std_f0, rest_rec_ensem_conf_f0[0], rest_rec_ensem_conf_f0[1]))
    print("facial image equal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_f0, focus_rec_ensem_std_f0, focus_rec_ensem_conf_f0[0],focus_rec_ensem_conf_f0[1]))
    print("facial image optimal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_f, rest_rec_ensem_std_f, rest_rec_ensem_conf_f[0], rest_rec_ensem_conf_f[1]))
    print("facial image optimal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_f, focus_rec_ensem_std_f, focus_rec_ensem_conf_f[0],focus_rec_ensem_conf_f[1]))
    print("facial image method vgg: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_vgg_mean_f, rest_rec_vgg_std_f, rest_rec_vgg_conf_f[0],rest_rec_vgg_conf_f[1]))
    print("facial image method vgg: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_vgg_mean_f, focus_rec_vgg_std_f, focus_rec_vgg_conf_f[0], focus_rec_vgg_conf_f[1]))
    print("facial image method svm: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_svm_mean_f, rest_rec_svm_std_f, rest_rec_svm_conf_f[0], rest_rec_svm_conf_f[1]))
    print("facial image method svm: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_svm_mean_f, focus_rec_svm_std_f, focus_rec_svm_conf_f[0], focus_rec_svm_conf_f[1]))   
    print("facial image method xgb: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_xgb_mean_f, rest_rec_xgb_std_f, rest_rec_xgb_conf_f[0], rest_rec_xgb_conf_f[1]))
    print("facial image method xgb: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_xgb_mean_f, focus_rec_xgb_std_f, focus_rec_xgb_conf_f[0], focus_rec_xgb_conf_f[1]))    

    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f1 = results_face + '{}_restults.txt'.format(args.method)
    with open(results_f1, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        ### write accuracy
        f.write("facial image equal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_f0, acc_ensem_std_f0,acc_ensem_conf_f0[0],acc_ensem_conf_f0[1]))
        f.write("facial image optimal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_f, acc_ensem_std_f,acc_ensem_conf_f[0],acc_ensem_conf_f[1])) 
        f.write("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method1, acc_vgg_mean_f, acc_vgg_std_f, acc_vgg_conf_f[0],acc_vgg_conf_f[1]))
        f.write("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method2, acc_svm_mean_f, acc_svm_std_f, acc_svm_conf_f[0],acc_svm_conf_f[1]))
        f.write("facial image method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method3, acc_xgb_mean_f, acc_xgb_std_f, acc_xgb_conf_f[0],acc_xgb_conf_f[1]))
        f.write('--'*40 +'\n')
        ### write precision
        f.write("facial image equal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_f0, rest_pre_ensem_std_f0, rest_pre_ensem_conf_f0[0], rest_pre_ensem_conf_f0[1]))
        f.write("facial image equal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_f0, focus_pre_ensem_std_f0, focus_pre_ensem_conf_f0[0],focus_pre_ensem_conf_f0[1]))
        f.write("facial image optimal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_f, rest_pre_ensem_std_f, rest_pre_ensem_conf_f[0], rest_pre_ensem_conf_f[1]))
        f.write("facial image optimal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_f, focus_pre_ensem_std_f, focus_pre_ensem_conf_f[0],focus_pre_ensem_conf_f[1]))
        f.write("facial image method vgg: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_vgg_mean_f, rest_pre_vgg_std_f, rest_pre_vgg_conf_f[0],rest_pre_vgg_conf_f[1]))
        f.write("facial image method vgg: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_vgg_mean_f, focus_pre_vgg_std_f, focus_pre_vgg_conf_f[0], focus_pre_vgg_conf_f[1]))
        f.write("facial image method svm: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_svm_mean_f, rest_pre_svm_std_f, rest_pre_svm_conf_f[0], rest_pre_svm_conf_f[1]))
        f.write("facial image method svm: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_svm_mean_f, focus_pre_svm_std_f, focus_pre_svm_conf_f[0], focus_pre_svm_conf_f[1]))   
        f.write("facial image method xgb: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_xgb_mean_f, rest_pre_xgb_std_f, rest_pre_xgb_conf_f[0], rest_pre_xgb_conf_f[1]))
        f.write("facial image method xgb: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_xgb_mean_f, focus_pre_xgb_std_f, focus_pre_xgb_conf_f[0], focus_pre_xgb_conf_f[1]))

        f.write('--'*40 +'\n')
        ### write recall
        f.write("facial image equal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_f0, rest_rec_ensem_std_f0, rest_rec_ensem_conf_f0[0], rest_rec_ensem_conf_f0[1]))
        f.write("facial image equal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_f0, focus_rec_ensem_std_f0, focus_rec_ensem_conf_f0[0],focus_rec_ensem_conf_f0[1]))
        f.write("facial image optimal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_f, rest_rec_ensem_std_f, rest_rec_ensem_conf_f[0], rest_rec_ensem_conf_f[1]))
        f.write("facial image optimal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_f, focus_rec_ensem_std_f, focus_rec_ensem_conf_f[0],focus_rec_ensem_conf_f[1]))
        f.write("facial image method vgg: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_vgg_mean_f, rest_rec_vgg_std_f, rest_rec_vgg_conf_f[0],rest_rec_vgg_conf_f[1]))
        f.write("facial image method vgg: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_vgg_mean_f, focus_rec_vgg_std_f, focus_rec_vgg_conf_f[0], focus_rec_vgg_conf_f[1]))
        f.write("facial image method svm: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_svm_mean_f, rest_rec_svm_std_f, rest_rec_svm_conf_f[0], rest_rec_svm_conf_f[1]))
        f.write("facial image method svm: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_svm_mean_f, focus_rec_svm_std_f, focus_rec_svm_conf_f[0], focus_rec_svm_conf_f[1]))   
        f.write("facial image method xgb: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_xgb_mean_f, rest_rec_xgb_std_f, rest_rec_xgb_conf_f[0], rest_rec_xgb_conf_f[1]))
        f.write("facial image method xgb: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_xgb_mean_f, focus_rec_xgb_std_f, focus_rec_xgb_conf_f[0], focus_rec_xgb_conf_f[1]))    
        f.write('--'*40 +'\n')
        f.write('facial image based: the size of method vgg is {} bytes, and the training time is {} \n'.format(vgg_size, train_time_vgg))
        f.write('facial image based: the size of method vgg is {} bytes, and the testing time is {} \n'.format(vgg_size, test_time_vgg))
        f.write('facial image based: the size of method svm is {} bytes, and the training time is {} \n'.format(svm_size, train_time_svm))
        f.write('facial image based: the size of method svm is {} bytes, and the testing time is {} \n'.format(svm_size, test_time_svm))
        f.write('facial image based: the size of method xgb is {} bytes, and the training time is {} \n'.format(xgb_size, train_time_xgb))
        f.write('facial image based: the size of method xgb is {} bytes, and the testing time is {} \n'.format(xgb_size, test_time_xgb))
    ###########################################################################
    ### results of ecg signal based methods
    ### results of accuracy
    acc_ensem_mean_ecg0 = np.array(acc_ensem_ecg0).mean()
    acc_ensem_std_ecg0 = np.array(acc_ensem_ecg0).std()
    acc_ensem_conf_ecg0 = st.t.interval(0.95, len(acc_ensem_ecg0)-1, loc=np.mean(acc_ensem_ecg0), scale=st.sem(acc_ensem_ecg0))
    acc_ensem_mean_ecg = np.array(acc_ensem_ecg).mean()
    acc_ensem_std_ecg = np.array(acc_ensem_ecg).std()
    acc_ensem_conf_ecg = st.t.interval(0.95, len(acc_ensem_ecg)-1, loc=np.mean(acc_ensem_ecg), scale=st.sem(acc_ensem_ecg))
    
    acc_vgg_mean_ecg = np.array(acc_vgg_ecg).mean()
    acc_vgg_std_ecg = np.array(acc_vgg_ecg).std()
    acc_vgg_conf_ecg = st.t.interval(0.95, len(acc_vgg_ecg)-1, loc=np.mean(acc_vgg_ecg), scale=st.sem(acc_vgg_ecg))
    acc_cnn_mean_ecg = np.array(acc_cnn_ecg).mean()
    acc_cnn_std_ecg = np.array(acc_cnn_ecg).std()
    acc_cnn_conf_ecg = st.t.interval(0.95, len(acc_cnn_ecg)-1, loc=np.mean(acc_cnn_ecg), scale=st.sem(acc_cnn_ecg))
    acc_svm_mean_ecg = np.array(acc_svm_ecg).mean()
    acc_svm_std_ecg = np.array(acc_svm_ecg).std()
    acc_svm_conf_ecg = st.t.interval(0.95, len(acc_svm_ecg)-1, loc=np.mean(acc_svm_ecg), scale=st.sem(acc_svm_ecg))
    acc_xgb_mean_ecg = np.array(acc_xgb_ecg).mean()
    acc_xgb_std_ecg = np.array(acc_xgb_ecg).std()
    acc_xgb_conf_ecg = st.t.interval(0.95, len(acc_xgb_ecg)-1, loc=np.mean(acc_xgb_ecg), scale=st.sem(acc_xgb_ecg))

    ### results of precision
    rest_pre_ensem_mean_ecg0 = np.array(rest_pre_ensem_ecg0).mean()
    rest_pre_ensem_std_ecg0 = np.array(rest_pre_ensem_ecg0).std() 
    rest_pre_ensem_conf_ecg0 = st.t.interval(0.95, len(rest_pre_ensem_ecg0)-1, loc=np.mean(rest_pre_ensem_ecg0), scale=st.sem(rest_pre_ensem_ecg0))
    focus_pre_ensem_mean_ecg0 = np.array(focus_pre_ensem_ecg0).mean()
    focus_pre_ensem_std_ecg0 = np.array(focus_pre_ensem_ecg0).std()
    focus_pre_ensem_conf_ecg0= st.t.interval(0.95, len(focus_pre_ensem_ecg0)-1, loc=np.mean(focus_pre_ensem_ecg0), scale=st.sem(focus_pre_ensem_ecg0))
    
    rest_pre_ensem_mean_ecg = np.array(rest_pre_ensem_ecg).mean()
    rest_pre_ensem_std_ecg = np.array(rest_pre_ensem_ecg).std() 
    rest_pre_ensem_conf_ecg = st.t.interval(0.95, len(rest_pre_ensem_ecg)-1, loc=np.mean(rest_pre_ensem_ecg), scale=st.sem(rest_pre_ensem_ecg))
    focus_pre_ensem_mean_ecg = np.array(focus_pre_ensem_ecg).mean()
    focus_pre_ensem_std_ecg = np.array(focus_pre_ensem_ecg).std()
    focus_pre_ensem_conf_ecg = st.t.interval(0.95, len(focus_pre_ensem_ecg)-1, loc=np.mean(focus_pre_ensem_ecg), scale=st.sem(focus_pre_ensem_ecg))
    
    rest_pre_vgg_mean_ecg = np.array(rest_pre_vgg_ecg).mean()
    rest_pre_vgg_std_ecg = np.array(rest_pre_vgg_ecg).std() 
    rest_pre_vgg_conf_ecg = st.t.interval(0.95, len(rest_pre_vgg_ecg)-1, loc=np.mean(rest_pre_vgg_ecg), scale=st.sem(rest_pre_vgg_ecg))
    focus_pre_vgg_mean_ecg = np.array(focus_pre_vgg_ecg).mean()
    focus_pre_vgg_std_ecg = np.array(focus_pre_vgg_ecg).std()
    focus_pre_vgg_conf_ecg = st.t.interval(0.95, len(focus_pre_vgg_ecg)-1, loc=np.mean(focus_pre_vgg_ecg), scale=st.sem(focus_pre_vgg_ecg))

    rest_pre_cnn_mean_ecg = np.array(rest_pre_cnn_ecg).mean()
    rest_pre_cnn_std_ecg = np.array(rest_pre_cnn_ecg).std() 
    rest_pre_cnn_conf_ecg = st.t.interval(0.95, len(rest_pre_cnn_ecg)-1, loc=np.mean(rest_pre_cnn_ecg), scale=st.sem(rest_pre_cnn_ecg))
    focus_pre_cnn_mean_ecg = np.array(focus_pre_cnn_ecg).mean()
    focus_pre_cnn_std_ecg = np.array(focus_pre_cnn_ecg).std()
    focus_pre_cnn_conf_ecg = st.t.interval(0.95, len(focus_pre_cnn_ecg)-1, loc=np.mean(focus_pre_cnn_ecg), scale=st.sem(focus_pre_cnn_ecg))
    
    rest_pre_svm_mean_ecg = np.array(rest_pre_svm_ecg).mean()
    rest_pre_svm_std_ecg = np.array(rest_pre_svm_ecg).std()  
    rest_pre_svm_conf_ecg = st.t.interval(0.95, len(rest_pre_svm_ecg)-1, loc=np.mean(rest_pre_svm_ecg), scale=st.sem(rest_pre_svm_ecg))
    focus_pre_svm_mean_ecg = np.array(focus_pre_svm_ecg).mean()
    focus_pre_svm_std_ecg = np.array(focus_pre_svm_ecg).std() 
    focus_pre_svm_conf_ecg = st.t.interval(0.95, len(focus_pre_svm_ecg)-1, loc=np.mean(focus_pre_svm_ecg), scale=st.sem(focus_pre_svm_ecg))
    
    rest_pre_xgb_mean_ecg = np.array(rest_pre_xgb_ecg).mean()
    rest_pre_xgb_std_ecg = np.array(rest_pre_xgb_ecg).std() 
    rest_pre_xgb_conf_ecg = st.t.interval(0.95, len(rest_pre_xgb_ecg)-1, loc=np.mean(rest_pre_xgb_ecg), scale=st.sem(rest_pre_xgb_ecg))
    focus_pre_xgb_mean_ecg = np.array(focus_pre_xgb_ecg).mean()
    focus_pre_xgb_std_ecg = np.array(focus_pre_xgb_ecg).std()
    focus_pre_xgb_conf_ecg = st.t.interval(0.95, len(focus_pre_xgb_ecg)-1, loc=np.mean(focus_pre_xgb_ecg), scale=st.sem(focus_pre_xgb_ecg))
    
    ### results of recall
    rest_rec_ensem_mean_ecg0 = np.array(rest_rec_ensem_ecg0).mean()
    rest_rec_ensem_std_ecg0 = np.array(rest_rec_ensem_ecg0).std() 
    rest_rec_ensem_conf_ecg0 = st.t.interval(0.95, len(rest_rec_ensem_ecg0)-1, loc=np.mean(rest_rec_ensem_ecg0), scale=st.sem(rest_rec_ensem_ecg0))
    focus_rec_ensem_mean_ecg0 = np.array(focus_rec_ensem_ecg0).mean()
    focus_rec_ensem_std_ecg0 = np.array(focus_rec_ensem_ecg0).std()
    focus_rec_ensem_conf_ecg0 = st.t.interval(0.95, len(focus_rec_ensem_ecg0)-1, loc=np.mean(focus_rec_ensem_ecg0), scale=st.sem(focus_rec_ensem_ecg0))
    
    rest_rec_ensem_mean_ecg = np.array(rest_rec_ensem_ecg).mean()
    rest_rec_ensem_std_ecg = np.array(rest_rec_ensem_ecg).std() 
    rest_rec_ensem_conf_ecg = st.t.interval(0.95, len(rest_rec_ensem_ecg)-1, loc=np.mean(rest_rec_ensem_ecg), scale=st.sem(rest_rec_ensem_ecg))
    focus_rec_ensem_mean_ecg = np.array(focus_rec_ensem_ecg).mean()
    focus_rec_ensem_std_ecg = np.array(focus_rec_ensem_ecg).std()
    focus_rec_ensem_conf_ecg = st.t.interval(0.95, len(focus_rec_ensem_ecg)-1, loc=np.mean(focus_rec_ensem_ecg), scale=st.sem(focus_rec_ensem_ecg))
    
    rest_rec_vgg_mean_ecg = np.array(rest_rec_vgg_ecg).mean()
    rest_rec_vgg_std_ecg = np.array(rest_rec_vgg_ecg).std() 
    rest_rec_vgg_conf_ecg = st.t.interval(0.95, len(rest_rec_vgg_ecg)-1, loc=np.mean(rest_rec_vgg_ecg), scale=st.sem(rest_rec_vgg_ecg))
    focus_rec_vgg_mean_ecg = np.array(focus_rec_vgg_ecg).mean()
    focus_rec_vgg_std_ecg = np.array(focus_rec_vgg_ecg).std()
    focus_rec_vgg_conf_ecg = st.t.interval(0.95, len(focus_rec_vgg_ecg)-1, loc=np.mean(focus_rec_vgg_ecg), scale=st.sem(focus_rec_vgg_ecg))

    rest_rec_cnn_mean_ecg = np.array(rest_rec_cnn_ecg).mean()
    rest_rec_cnn_std_ecg = np.array(rest_rec_cnn_ecg).std() 
    rest_rec_cnn_conf_ecg = st.t.interval(0.95, len(rest_rec_cnn_ecg)-1, loc=np.mean(rest_rec_cnn_ecg), scale=st.sem(rest_rec_cnn_ecg))
    focus_rec_cnn_mean_ecg = np.array(focus_rec_cnn_ecg).mean()
    focus_rec_cnn_std_ecg = np.array(focus_rec_cnn_ecg).std()
    focus_rec_cnn_conf_ecg = st.t.interval(0.95, len(focus_rec_cnn_ecg)-1, loc=np.mean(focus_rec_cnn_ecg), scale=st.sem(focus_rec_cnn_ecg))
    
    rest_rec_svm_mean_ecg = np.array(rest_rec_svm_ecg).mean()
    rest_rec_svm_std_ecg = np.array(rest_rec_svm_ecg).std()  
    rest_rec_svm_conf_ecg = st.t.interval(0.95, len(rest_rec_svm_ecg)-1, loc=np.mean(rest_rec_svm_ecg), scale=st.sem(rest_rec_svm_ecg))
    focus_rec_svm_mean_ecg = np.array(focus_rec_svm_ecg).mean()
    focus_rec_svm_std_ecg = np.array(focus_rec_svm_ecg).std() 
    focus_rec_svm_conf_ecg = st.t.interval(0.95, len(focus_rec_svm_ecg)-1, loc=np.mean(focus_rec_svm_ecg), scale=st.sem(focus_rec_svm_ecg))
    
    rest_rec_xgb_mean_ecg = np.array(rest_rec_xgb_ecg).mean()
    rest_rec_xgb_std_ecg = np.array(rest_rec_xgb_ecg).std() 
    rest_rec_xgb_conf_ecg = st.t.interval(0.95, len(rest_rec_xgb_ecg)-1, loc=np.mean(rest_rec_xgb_ecg), scale=st.sem(rest_rec_xgb_ecg))
    focus_rec_xgb_mean_ecg = np.array(focus_rec_xgb_ecg).mean()
    focus_rec_xgb_std_ecg = np.array(focus_rec_xgb_ecg).std()
    focus_rec_xgb_conf_ecg = st.t.interval(0.95, len(focus_rec_xgb_ecg)-1, loc=np.mean(focus_rec_xgb_ecg), scale=st.sem(focus_rec_xgb_ecg))    
    
    
    ### print accuracy
    print("ecg signal equal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_ecg0, acc_ensem_std_ecg0,acc_ensem_conf_ecg0[0],acc_ensem_conf_ecg0[1]))
    print("ecg signal optimal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_ecg, acc_ensem_std_ecg,acc_ensem_conf_ecg[0],acc_ensem_conf_ecg[1]))
    print("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method4, acc_vgg_mean_ecg, acc_vgg_std_ecg, acc_vgg_conf_ecg[0],acc_vgg_conf_ecg[1]))
    print("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method5, acc_cnn_mean_ecg, acc_cnn_std_ecg, acc_cnn_conf_ecg[0],acc_cnn_conf_ecg[1]))
    print("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method6, acc_svm_mean_ecg, acc_svm_std_ecg, acc_svm_conf_ecg[0],acc_svm_conf_ecg[1]))
    print("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method7, acc_xgb_mean_ecg, acc_xgb_std_ecg, acc_xgb_conf_ecg[0],acc_xgb_conf_ecg[1]))
    
    ### print precision
    print("ecg signal equal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_ecg0, rest_pre_ensem_std_ecg0, rest_pre_ensem_conf_ecg0[0], rest_pre_ensem_conf_ecg0[1]))
    print("ecg signal equal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_ecg0, focus_pre_ensem_std_ecg0, focus_pre_ensem_conf_ecg0[0],focus_pre_ensem_conf_ecg0[1]))
    print("ecg signal optimal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_ecg, rest_pre_ensem_std_ecg, rest_pre_ensem_conf_ecg[0], rest_pre_ensem_conf_ecg[1]))
    print("ecg signal optimal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_ecg, focus_pre_ensem_std_ecg, focus_pre_ensem_conf_ecg[0],focus_pre_ensem_conf_ecg[1]))
    print("ecg signal method vgg: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_vgg_mean_ecg, rest_pre_vgg_std_ecg, rest_pre_vgg_conf_ecg[0],rest_pre_vgg_conf_ecg[1]))
    print("ecg signal method vgg: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_vgg_mean_ecg, focus_pre_vgg_std_ecg, focus_pre_vgg_conf_ecg[0], focus_pre_vgg_conf_ecg[1]))
    print("ecg signal method cnn: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_cnn_mean_ecg, rest_pre_cnn_std_ecg, rest_pre_cnn_conf_ecg[0],rest_pre_cnn_conf_ecg[1]))
    print("ecg signal method cnn: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_cnn_mean_ecg, focus_pre_cnn_std_ecg, focus_pre_cnn_conf_ecg[0], focus_pre_cnn_conf_ecg[1]))
    print("ecg signal method svm: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_svm_mean_ecg, rest_pre_svm_std_ecg, rest_pre_svm_conf_ecg[0], rest_pre_svm_conf_ecg[1]))
    print("ecg signal method svm: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_svm_mean_ecg, focus_pre_svm_std_ecg, focus_pre_svm_conf_ecg[0], focus_pre_svm_conf_ecg[1]))   
    print("ecg signal method xgb: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_xgb_mean_ecg, rest_pre_xgb_std_ecg, rest_pre_xgb_conf_ecg[0], rest_pre_xgb_conf_ecg[1]))
    print("ecg signal method xgb: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_xgb_mean_ecg, focus_pre_xgb_std_ecg, focus_pre_xgb_conf_ecg[0], focus_pre_xgb_conf_ecg[1]))

    ### print recall
    print("ecg signal equal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_ecg0, rest_rec_ensem_std_ecg0, rest_rec_ensem_conf_ecg0[0], rest_rec_ensem_conf_ecg0[1]))
    print("ecg signal equal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_ecg0, focus_rec_ensem_std_ecg0, focus_rec_ensem_conf_ecg0[0],focus_rec_ensem_conf_ecg0[1]))
    print("ecg signal optimal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_ecg, rest_rec_ensem_std_ecg, rest_rec_ensem_conf_ecg[0], rest_rec_ensem_conf_ecg[1]))
    print("ecg signal optimal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_ecg, focus_rec_ensem_std_ecg, focus_rec_ensem_conf_ecg[0],focus_rec_ensem_conf_ecg[1]))
    print("ecg signal method vgg: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_vgg_mean_ecg, rest_rec_vgg_std_ecg, rest_rec_vgg_conf_ecg[0],rest_rec_vgg_conf_ecg[1]))
    print("ecg signal method vgg: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_vgg_mean_ecg, focus_rec_vgg_std_ecg, focus_rec_vgg_conf_ecg[0], focus_rec_vgg_conf_ecg[1]))
    print("ecg signal method cnn: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_cnn_mean_ecg, rest_rec_cnn_std_ecg, rest_rec_cnn_conf_ecg[0],rest_rec_cnn_conf_ecg[1]))
    print("ecg signal method cnn: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_cnn_mean_ecg, focus_rec_cnn_std_ecg, focus_rec_cnn_conf_ecg[0], focus_rec_cnn_conf_ecg[1]))
    print("ecg signal method svm: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_svm_mean_ecg, rest_rec_svm_std_ecg, rest_rec_svm_conf_ecg[0], rest_rec_svm_conf_ecg[1]))
    print("ecg signal method svm: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_svm_mean_ecg, focus_rec_svm_std_ecg, focus_rec_svm_conf_ecg[0], focus_rec_svm_conf_ecg[1]))   
    print("ecg signal method xgb: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_xgb_mean_ecg, rest_rec_xgb_std_ecg, rest_rec_xgb_conf_ecg[0], rest_rec_xgb_conf_ecg[1]))
    print("ecg signal method xgb: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_xgb_mean_ecg, focus_rec_xgb_std_ecg, focus_rec_xgb_conf_ecg[0], focus_rec_xgb_conf_ecg[1]))    

    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f = results_ecg + '{}_restults.txt'.format(args.method)
    with open(results_f, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        ### write accuracy
        f.write("ecg signal equal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_ecg0, acc_ensem_std_ecg0,acc_ensem_conf_ecg0[0],acc_ensem_conf_ecg0[1]))
        f.write("ecg signal optimal weight method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, acc_ensem_mean_ecg, acc_ensem_std_ecg,acc_ensem_conf_ecg[0],acc_ensem_conf_ecg[1]))
        f.write("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method4, acc_vgg_mean_ecg, acc_vgg_std_ecg, acc_vgg_conf_ecg[0],acc_vgg_conf_ecg[1]))
        f.write("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method5, acc_cnn_mean_ecg, acc_cnn_std_ecg, acc_cnn_conf_ecg[0],acc_cnn_conf_ecg[1]))
        f.write("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method6, acc_svm_mean_ecg, acc_svm_std_ecg, acc_svm_conf_ecg[0],acc_svm_conf_ecg[1]))
        f.write("ecg signal method %s: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (method7, acc_xgb_mean_ecg, acc_xgb_std_ecg, acc_xgb_conf_ecg[0],acc_xgb_conf_ecg[1]))
    
        f.write('--'*40 +'\n')
        ### write precision
        f.write("ecg signal equal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_ecg0, rest_pre_ensem_std_ecg0, rest_pre_ensem_conf_ecg0[0], rest_pre_ensem_conf_ecg0[1]))
        f.write("ecg signal equal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_ecg0, focus_pre_ensem_std_ecg0, focus_pre_ensem_conf_ecg0[0],focus_pre_ensem_conf_ecg0[1]))
        f.write("ecg signal optimal weight method %s: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_pre_ensem_mean_ecg, rest_pre_ensem_std_ecg, rest_pre_ensem_conf_ecg[0], rest_pre_ensem_conf_ecg[1]))
        f.write("ecg signal optimal weight method %s: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_pre_ensem_mean_ecg, focus_pre_ensem_std_ecg, focus_pre_ensem_conf_ecg[0],focus_pre_ensem_conf_ecg[1]))
        f.write("ecg signal method vgg: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_vgg_mean_ecg, rest_pre_vgg_std_ecg, rest_pre_vgg_conf_ecg[0],rest_pre_vgg_conf_ecg[1]))
        f.write("ecg signal method vgg: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_vgg_mean_ecg, focus_pre_vgg_std_ecg, focus_pre_vgg_conf_ecg[0], focus_pre_vgg_conf_ecg[1]))
        f.write("ecg signal method cnn: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_cnn_mean_ecg, rest_pre_cnn_std_ecg, rest_pre_cnn_conf_ecg[0],rest_pre_cnn_conf_ecg[1]))
        f.write("ecg signal method cnn: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_cnn_mean_ecg, focus_pre_cnn_std_ecg, focus_pre_cnn_conf_ecg[0], focus_pre_cnn_conf_ecg[1]))
        f.write("ecg signal method svm: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_svm_mean_ecg, rest_pre_svm_std_ecg, rest_pre_svm_conf_ecg[0], rest_pre_svm_conf_ecg[1]))
        f.write("ecg signal method svm: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_svm_mean_ecg, focus_pre_svm_std_ecg, focus_pre_svm_conf_ecg[0], focus_pre_svm_conf_ecg[1]))   
        f.write("ecg signal method xgb: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_xgb_mean_ecg, rest_pre_xgb_std_ecg, rest_pre_xgb_conf_ecg[0], rest_pre_xgb_conf_ecg[1]))
        f.write("ecg signal method xgb: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_xgb_mean_ecg, focus_pre_xgb_std_ecg, focus_pre_xgb_conf_ecg[0], focus_pre_xgb_conf_ecg[1]))
        f.write('--'*40 +'\n')
        ### write recall
        f.write("ecg signal equal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_ecg0, rest_rec_ensem_std_ecg0, rest_rec_ensem_conf_ecg0[0], rest_rec_ensem_conf_ecg0[1]))
        f.write("ecg signal equal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_ecg0, focus_rec_ensem_std_ecg0, focus_rec_ensem_conf_ecg0[0],focus_rec_ensem_conf_ecg0[1]))
        f.write("ecg signal optimal weight method %s: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, rest_rec_ensem_mean_ecg, rest_rec_ensem_std_ecg, rest_rec_ensem_conf_ecg[0], rest_rec_ensem_conf_ecg[1]))
        f.write("ecg signal optimal weight method %s: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (args.method, focus_rec_ensem_mean_ecg, focus_rec_ensem_std_ecg, focus_rec_ensem_conf_ecg[0],focus_rec_ensem_conf_ecg[1]))
        f.write("ecg signal method vgg: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_vgg_mean_ecg, rest_rec_vgg_std_ecg, rest_rec_vgg_conf_ecg[0],rest_rec_vgg_conf_ecg[1]))
        f.write("ecg signal method vgg: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_vgg_mean_ecg, focus_rec_vgg_std_ecg, focus_rec_vgg_conf_ecg[0], focus_rec_vgg_conf_ecg[1]))
        f.write("ecg signal method cnn: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_cnn_mean_ecg, rest_rec_cnn_std_ecg, rest_rec_cnn_conf_ecg[0],rest_rec_cnn_conf_ecg[1]))
        f.write("ecg signal method cnn: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_cnn_mean_ecg, focus_rec_cnn_std_ecg, focus_rec_cnn_conf_ecg[0], focus_rec_cnn_conf_ecg[1]))
        f.write("ecg signal method svm: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_svm_mean_ecg, rest_rec_svm_std_ecg, rest_rec_svm_conf_ecg[0], rest_rec_svm_conf_ecg[1]))
        f.write("ecg signal method svm: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_svm_mean_ecg, focus_rec_svm_std_ecg, focus_rec_svm_conf_ecg[0], focus_rec_svm_conf_ecg[1]))   
        f.write("ecg signal method xgb: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_xgb_mean_ecg, rest_rec_xgb_std_ecg, rest_rec_xgb_conf_ecg[0], rest_rec_xgb_conf_ecg[1]))
        f.write("ecg signal method xgb: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_xgb_mean_ecg, focus_rec_xgb_std_ecg, focus_rec_xgb_conf_ecg[0], focus_rec_xgb_conf_ecg[1]))    

        f.write('--'*40 +'\n')
        f.write('ecg signal based: the size of method vgg is {} bytes, and the training time is {} \n'.format(vgg_ecg_size, train_time_vgg_ecg))
        f.write('ecg signal based: the size of method vgg is {} bytes, and the testing time is {} \n'.format(vgg_ecg_size, test_time_vgg_ecg))
        f.write('ecg signal based: the size of method 1d cnn is {} bytes, and the training time is {} \n'.format(cnn_ecg_size, train_time_cnn_ecg))
        f.write('ecg signal based: the size of method 1d cnn is {} bytes, and the testing time is {} \n'.format(cnn_ecg_size, test_time_cnn_ecg))
        f.write('ecg signal based: the size of method svm is {} bytes, and the training time is {} \n'.format(svm_ecg_size, train_time_svm_ecg))
        f.write('ecg signal based: the size of method svm is {} bytes, and the testing time is {} \n'.format(svm_ecg_size, test_time_svm_ecg))
        f.write('ecg signal based: the size of method xgb is {} bytes, and the training time is {} \n'.format(xgb_ecg_size, train_time_xgb_ecg))
        f.write('ecg signal based: the size of method xgb is {} bytes, and the testing time is {} \n'.format(xgb_ecg_size, test_time_xgb_ecg))        
    ###########################################################################
    ### resutls of ensemble on each model
    ### equal weight ensemble of svm
    acc_ensem_mean_svm0 = np.array(acc_ensem_svm0).mean()
    acc_ensem_std_svm0 = np.array(acc_ensem_svm0).std()
    acc_ensem_conf_svm0 = st.t.interval(0.95, len(acc_ensem_svm0)-1, loc=np.mean(acc_ensem_svm0), scale=st.sem(acc_ensem_svm0))   
    ### results of precision
    rest_pre_ensem_mean_svm0 = np.array(rest_pre_ensem_svm0).mean()
    rest_pre_ensem_std_svm0 = np.array(rest_pre_ensem_svm0).std() 
    rest_pre_ensem_conf_svm0 = st.t.interval(0.95, len(rest_pre_ensem_svm0)-1, loc=np.mean(rest_pre_ensem_svm0), scale=st.sem(rest_pre_ensem_svm0))
    focus_pre_ensem_mean_svm0 = np.array(focus_pre_ensem_svm0).mean()
    focus_pre_ensem_std_svm0 = np.array(focus_pre_ensem_svm0).std()
    focus_pre_ensem_conf_svm0 = st.t.interval(0.95, len(focus_pre_ensem_svm0)-1, loc=np.mean(focus_pre_ensem_svm0), scale=st.sem(focus_pre_ensem_svm0))
    ### results of recall
    rest_rec_ensem_mean_svm0 = np.array(rest_rec_ensem_svm0).mean()
    rest_rec_ensem_std_svm0 = np.array(rest_rec_ensem_svm0).std() 
    rest_rec_ensem_conf_svm0= st.t.interval(0.95, len(rest_rec_ensem_svm0)-1, loc=np.mean(rest_rec_ensem_svm0), scale=st.sem(rest_rec_ensem_svm0))
    focus_rec_ensem_mean_svm0 = np.array(focus_rec_ensem_svm0).mean()
    focus_rec_ensem_std_svm0 = np.array(focus_rec_ensem_svm0).std()
    focus_rec_ensem_conf_svm0 = st.t.interval(0.95, len(focus_rec_ensem_svm0)-1, loc=np.mean(focus_rec_ensem_svm0), scale=st.sem(focus_rec_ensem_svm0))    
    
    ### optimal weight ensemble of svm
    acc_ensem_mean_svm = np.array(acc_ensem_svm).mean()
    acc_ensem_std_svm = np.array(acc_ensem_svm).std()
    acc_ensem_conf_svm = st.t.interval(0.95, len(acc_ensem_svm)-1, loc=np.mean(acc_ensem_svm), scale=st.sem(acc_ensem_svm))   
    ### results of precision
    rest_pre_ensem_mean_svm = np.array(rest_pre_ensem_svm).mean()
    rest_pre_ensem_std_svm = np.array(rest_pre_ensem_svm).std() 
    rest_pre_ensem_conf_svm = st.t.interval(0.95, len(rest_pre_ensem_svm)-1, loc=np.mean(rest_pre_ensem_svm), scale=st.sem(rest_pre_ensem_svm))
    focus_pre_ensem_mean_svm = np.array(focus_pre_ensem_svm).mean()
    focus_pre_ensem_std_svm = np.array(focus_pre_ensem_svm).std()
    focus_pre_ensem_conf_svm = st.t.interval(0.95, len(focus_pre_ensem_svm)-1, loc=np.mean(focus_pre_ensem_svm), scale=st.sem(focus_pre_ensem_svm))
    ### results of recall
    rest_rec_ensem_mean_svm = np.array(rest_rec_ensem_svm).mean()
    rest_rec_ensem_std_svm = np.array(rest_rec_ensem_svm).std() 
    rest_rec_ensem_conf_svm = st.t.interval(0.95, len(rest_rec_ensem_svm)-1, loc=np.mean(rest_rec_ensem_svm), scale=st.sem(rest_rec_ensem_svm))
    focus_rec_ensem_mean_svm = np.array(focus_rec_ensem_svm).mean()
    focus_rec_ensem_std_svm = np.array(focus_rec_ensem_svm).std()
    focus_rec_ensem_conf_svm = st.t.interval(0.95, len(focus_rec_ensem_svm)-1, loc=np.mean(focus_rec_ensem_svm), scale=st.sem(focus_rec_ensem_svm))  
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f3 = args.results + '{}_restults.txt'.format(args.method)
    with open(results_f3, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.write("svm equal weight ensemble method: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (acc_ensem_mean_svm0, acc_ensem_std_svm0, acc_ensem_conf_svm0[0],acc_ensem_conf_svm0[1]))
        f.write("svm equal weight ensemble method: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_ensem_mean_svm0, rest_pre_ensem_std_svm0, rest_pre_ensem_conf_svm0[0], rest_pre_ensem_conf_svm0[1]))
        f.write("svm equal weight ensemble method: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_ensem_mean_svm0, focus_pre_ensem_std_svm0, focus_pre_ensem_conf_svm0[0], focus_pre_ensem_conf_svm0[1]))
        f.write("svm equal weight ensemble method: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_ensem_mean_svm0, rest_rec_ensem_std_svm0, rest_rec_ensem_conf_svm0[0], rest_rec_ensem_conf_svm0[1]))
        f.write("svm equal weight ensemble method: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_ensem_mean_svm0, focus_rec_ensem_std_svm0, focus_rec_ensem_conf_svm0[0], focus_rec_ensem_conf_svm0[1]))
        f.write('--'*40 +'\n')
        f.write("svm optimal weight ensemble method: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (acc_ensem_mean_svm, acc_ensem_std_svm, acc_ensem_conf_svm[0],acc_ensem_conf_svm[1]))
        f.write("svm optimal weight ensemble method: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_ensem_mean_svm, rest_pre_ensem_std_svm, rest_pre_ensem_conf_svm[0], rest_pre_ensem_conf_svm[1]))
        f.write("svm optimal weight ensemble method: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_ensem_mean_svm, focus_pre_ensem_std_svm, focus_pre_ensem_conf_svm[0], focus_pre_ensem_conf_svm[1]))
        f.write("svm optimal weight ensemble method: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_ensem_mean_svm, rest_rec_ensem_std_svm, rest_rec_ensem_conf_svm[0], rest_rec_ensem_conf_svm[1]))
        f.write("svm optimal weight ensemble method: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_ensem_mean_svm, focus_rec_ensem_std_svm, focus_rec_ensem_conf_svm[0], focus_rec_ensem_conf_svm[1]))

    ###########################################################################    
    ### equal weight ensemble of xgb
    acc_ensem_mean_xgb0 = np.array(acc_ensem_xgb0).mean()
    acc_ensem_std_xgb0 = np.array(acc_ensem_xgb0).std()
    acc_ensem_conf_xgb0 = st.t.interval(0.95, len(acc_ensem_xgb0)-1, loc=np.mean(acc_ensem_xgb0), scale=st.sem(acc_ensem_xgb0))   
    ### results of precision
    rest_pre_ensem_mean_xgb0 = np.array(rest_pre_ensem_xgb0).mean()
    rest_pre_ensem_std_xgb0 = np.array(rest_pre_ensem_xgb0).std() 
    rest_pre_ensem_conf_xgb0 = st.t.interval(0.95, len(rest_pre_ensem_xgb0)-1, loc=np.mean(rest_pre_ensem_xgb0), scale=st.sem(rest_pre_ensem_xgb0))
    focus_pre_ensem_mean_xgb0 = np.array(focus_pre_ensem_xgb0).mean()
    focus_pre_ensem_std_xgb0= np.array(focus_pre_ensem_xgb0).std()
    focus_pre_ensem_conf_xgb0 = st.t.interval(0.95, len(focus_pre_ensem_xgb0)-1, loc=np.mean(focus_pre_ensem_xgb0), scale=st.sem(focus_pre_ensem_xgb0))
    ### results of recall
    rest_rec_ensem_mean_xgb0 = np.array(rest_rec_ensem_xgb0).mean()
    rest_rec_ensem_std_xgb0 = np.array(rest_rec_ensem_xgb0).std() 
    rest_rec_ensem_conf_xgb0 = st.t.interval(0.95, len(rest_rec_ensem_xgb0)-1, loc=np.mean(rest_rec_ensem_xgb0), scale=st.sem(rest_rec_ensem_xgb0))
    focus_rec_ensem_mean_xgb0 = np.array(focus_rec_ensem_xgb0).mean()
    focus_rec_ensem_std_xgb0 = np.array(focus_rec_ensem_xgb0).std()
    focus_rec_ensem_conf_xgb0 = st.t.interval(0.95, len(focus_rec_ensem_xgb0)-1, loc=np.mean(focus_rec_ensem_xgb0), scale=st.sem(focus_rec_ensem_xgb0))    
    
    ### optimal weight ensemble of xgb
    acc_ensem_mean_xgb = np.array(acc_ensem_xgb).mean()
    acc_ensem_std_xgb = np.array(acc_ensem_xgb).std()
    acc_ensem_conf_xgb = st.t.interval(0.95, len(acc_ensem_xgb)-1, loc=np.mean(acc_ensem_xgb), scale=st.sem(acc_ensem_xgb))   
    ### results of precision
    rest_pre_ensem_mean_xgb = np.array(rest_pre_ensem_xgb).mean()
    rest_pre_ensem_std_xgb = np.array(rest_pre_ensem_xgb).std() 
    rest_pre_ensem_conf_xgb = st.t.interval(0.95, len(rest_pre_ensem_xgb)-1, loc=np.mean(rest_pre_ensem_xgb), scale=st.sem(rest_pre_ensem_xgb))
    focus_pre_ensem_mean_xgb = np.array(focus_pre_ensem_xgb).mean()
    focus_pre_ensem_std_xgb = np.array(focus_pre_ensem_xgb).std()
    focus_pre_ensem_conf_xgb = st.t.interval(0.95, len(focus_pre_ensem_xgb)-1, loc=np.mean(focus_pre_ensem_xgb), scale=st.sem(focus_pre_ensem_xgb))
    ### results of recall
    rest_rec_ensem_mean_xgb = np.array(rest_rec_ensem_xgb).mean()
    rest_rec_ensem_std_xgb = np.array(rest_rec_ensem_xgb).std() 
    rest_rec_ensem_conf_xgb = st.t.interval(0.95, len(rest_rec_ensem_xgb)-1, loc=np.mean(rest_rec_ensem_xgb), scale=st.sem(rest_rec_ensem_xgb))
    focus_rec_ensem_mean_xgb = np.array(focus_rec_ensem_xgb).mean()
    focus_rec_ensem_std_xgb = np.array(focus_rec_ensem_xgb).std()
    focus_rec_ensem_conf_xgb = st.t.interval(0.95, len(focus_rec_ensem_xgb)-1, loc=np.mean(focus_rec_ensem_xgb), scale=st.sem(focus_rec_ensem_xgb)) 
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f3 = args.results + '{}_restults.txt'.format(args.method)
    with open(results_f3, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.write("xgb equal weight ensemble method: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (acc_ensem_mean_xgb0, acc_ensem_std_xgb0, acc_ensem_conf_xgb0[0],acc_ensem_conf_xgb0[1]))
        f.write("xgb equal weight ensemble method: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_ensem_mean_xgb0, rest_pre_ensem_std_xgb0, rest_pre_ensem_conf_xgb0[0], rest_pre_ensem_conf_xgb0[1]))
        f.write("xgb equal weight ensemble method: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_ensem_mean_xgb0, focus_pre_ensem_std_xgb0, focus_pre_ensem_conf_xgb0[0], focus_pre_ensem_conf_xgb0[1]))
        f.write("xgb equal weight ensemble method: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_ensem_mean_xgb0, rest_rec_ensem_std_xgb0, rest_rec_ensem_conf_xgb0[0], rest_rec_ensem_conf_xgb0[1]))
        f.write("xgb equal weight ensemble method: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_ensem_mean_xgb0, focus_rec_ensem_std_xgb0, focus_rec_ensem_conf_xgb0[0], focus_rec_ensem_conf_xgb0[1]))
        f.write('--'*40 +'\n')
        f.write("xgb optimal weight ensemble method: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (acc_ensem_mean_xgb, acc_ensem_std_xgb, acc_ensem_conf_xgb[0],acc_ensem_conf_xgb[1]))
        f.write("xgb optimal weight ensemble method: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_ensem_mean_xgb, rest_pre_ensem_std_xgb, rest_pre_ensem_conf_xgb[0], rest_pre_ensem_conf_xgb[1]))
        f.write("xgb optimal weight ensemble method: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_ensem_mean_xgb, focus_pre_ensem_std_xgb, focus_pre_ensem_conf_xgb[0], focus_pre_ensem_conf_xgb[1]))
        f.write("xgb optimal weight ensemble method: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_ensem_mean_xgb, rest_rec_ensem_std_xgb, rest_rec_ensem_conf_xgb[0], rest_rec_ensem_conf_xgb[1]))
        f.write("xgb optimal weight ensemble method: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_ensem_mean_xgb, focus_rec_ensem_std_xgb, focus_rec_ensem_conf_xgb[0], focus_rec_ensem_conf_xgb[1]))
                
    ###########################################################################    
    ### equal weight ensemble of vgg
    acc_ensem_mean_vgg0 = np.array(acc_ensem_vgg0).mean()
    acc_ensem_std_vgg0 = np.array(acc_ensem_vgg0).std()
    acc_ensem_conf_vgg0 = st.t.interval(0.95, len(acc_ensem_vgg0)-1, loc=np.mean(acc_ensem_vgg0), scale=st.sem(acc_ensem_vgg0))   
    ### results of precision
    rest_pre_ensem_mean_vgg0 = np.array(rest_pre_ensem_vgg0).mean()
    rest_pre_ensem_std_vgg0 = np.array(rest_pre_ensem_vgg0).std() 
    rest_pre_ensem_conf_vgg0 = st.t.interval(0.95, len(rest_pre_ensem_vgg0)-1, loc=np.mean(rest_pre_ensem_vgg0), scale=st.sem(rest_pre_ensem_vgg0))
    focus_pre_ensem_mean_vgg0 = np.array(focus_pre_ensem_vgg0).mean()
    focus_pre_ensem_std_vgg0 = np.array(focus_pre_ensem_vgg0).std()
    focus_pre_ensem_conf_vgg0 = st.t.interval(0.95, len(focus_pre_ensem_vgg0)-1, loc=np.mean(focus_pre_ensem_vgg0), scale=st.sem(focus_pre_ensem_vgg0))
    ### results of recall
    rest_rec_ensem_mean_vgg0 = np.array(rest_rec_ensem_vgg0).mean()
    rest_rec_ensem_std_vgg0 = np.array(rest_rec_ensem_vgg0).std() 
    rest_rec_ensem_conf_vgg0 = st.t.interval(0.95, len(rest_rec_ensem_vgg0)-1, loc=np.mean(rest_rec_ensem_vgg0), scale=st.sem(rest_rec_ensem_vgg0))
    focus_rec_ensem_mean_vgg0 = np.array(focus_rec_ensem_vgg0).mean()
    focus_rec_ensem_std_vgg0 = np.array(focus_rec_ensem_vgg0).std()
    focus_rec_ensem_conf_vgg0 = st.t.interval(0.95, len(focus_rec_ensem_vgg0)-1, loc=np.mean(focus_rec_ensem_vgg0), scale=st.sem(focus_rec_ensem_vgg0))    

    ### optimal weight ensemble of vgg
    acc_ensem_mean_vgg = np.array(acc_ensem_vgg).mean()
    acc_ensem_std_vgg = np.array(acc_ensem_vgg).std()
    acc_ensem_conf_vgg = st.t.interval(0.95, len(acc_ensem_vgg)-1, loc=np.mean(acc_ensem_vgg), scale=st.sem(acc_ensem_vgg))   
    ### results of precision
    rest_pre_ensem_mean_vgg = np.array(rest_pre_ensem_vgg).mean()
    rest_pre_ensem_std_vgg = np.array(rest_pre_ensem_vgg).std() 
    rest_pre_ensem_conf_vgg = st.t.interval(0.95, len(rest_pre_ensem_vgg)-1, loc=np.mean(rest_pre_ensem_vgg), scale=st.sem(rest_pre_ensem_vgg))
    focus_pre_ensem_mean_vgg = np.array(focus_pre_ensem_vgg).mean()
    focus_pre_ensem_std_vgg = np.array(focus_pre_ensem_vgg).std()
    focus_pre_ensem_conf_vgg = st.t.interval(0.95, len(focus_pre_ensem_vgg)-1, loc=np.mean(focus_pre_ensem_vgg), scale=st.sem(focus_pre_ensem_vgg))
    ### results of recall
    rest_rec_ensem_mean_vgg = np.array(rest_rec_ensem_vgg).mean()
    rest_rec_ensem_std_vgg = np.array(rest_rec_ensem_vgg).std() 
    rest_rec_ensem_conf_vgg = st.t.interval(0.95, len(rest_rec_ensem_vgg)-1, loc=np.mean(rest_rec_ensem_vgg), scale=st.sem(rest_rec_ensem_vgg))
    focus_rec_ensem_mean_vgg = np.array(focus_rec_ensem_vgg).mean()
    focus_rec_ensem_std_vgg = np.array(focus_rec_ensem_vgg).std()
    focus_rec_ensem_conf_vgg = st.t.interval(0.95, len(focus_rec_ensem_vgg)-1, loc=np.mean(focus_rec_ensem_vgg), scale=st.sem(focus_rec_ensem_vgg))    
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f3 = args.results + '{}_restults.txt'.format(args.method)
    with open(results_f3, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.write("vgg equal weight ensemble method: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (acc_ensem_mean_vgg0, acc_ensem_std_vgg0, acc_ensem_conf_vgg0[0],acc_ensem_conf_vgg0[1]))
        f.write("vgg equal weight ensemble method: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_ensem_mean_vgg0, rest_pre_ensem_std_vgg0, rest_pre_ensem_conf_vgg0[0], rest_pre_ensem_conf_vgg0[1]))
        f.write("vgg equal weight ensemble method: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_ensem_mean_vgg0, focus_pre_ensem_std_vgg0, focus_pre_ensem_conf_vgg0[0], focus_pre_ensem_conf_vgg0[1]))
        f.write("vgg equal weight ensemble method: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_ensem_mean_vgg0, rest_rec_ensem_std_vgg0, rest_rec_ensem_conf_vgg0[0], rest_rec_ensem_conf_vgg0[1]))
        f.write("vgg equal weight ensemble method: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_ensem_mean_vgg0, focus_rec_ensem_std_vgg0, focus_rec_ensem_conf_vgg0[0], focus_rec_ensem_conf_vgg0[1]))
        f.write('--'*40 +'\n')
        f.write("vgg optimal weight ensemble method: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (acc_ensem_mean_vgg, acc_ensem_std_vgg, acc_ensem_conf_vgg[0],acc_ensem_conf_vgg[1]))
        f.write("vgg optimal weight ensemble method: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_ensem_mean_vgg, rest_pre_ensem_std_vgg, rest_pre_ensem_conf_vgg[0], rest_pre_ensem_conf_vgg[1]))
        f.write("vgg optimal weight ensemble method: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_ensem_mean_vgg, focus_pre_ensem_std_vgg, focus_pre_ensem_conf_vgg[0], focus_pre_ensem_conf_vgg[1]))
        f.write("vgg optimal weight ensemble method: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_ensem_mean_vgg, rest_rec_ensem_std_vgg, rest_rec_ensem_conf_vgg[0], rest_rec_ensem_conf_vgg[1]))
        f.write("vgg optimal weight ensemble method: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_ensem_mean_vgg, focus_rec_ensem_std_vgg, focus_rec_ensem_conf_vgg[0], focus_rec_ensem_conf_vgg[1]))
    ###########################################################################    
    ### equal weight ensemble of vgg and 1d_cnn
    acc_ensem_mean_vgg_cnn0 = np.array(acc_ensem_vgg_cnn0).mean()
    acc_ensem_std_vgg_cnn0 = np.array(acc_ensem_vgg_cnn0).std()
    acc_ensem_conf_vgg_cnn0 = st.t.interval(0.95, len(acc_ensem_vgg_cnn0)-1, loc=np.mean(acc_ensem_vgg_cnn0), scale=st.sem(acc_ensem_vgg_cnn0))   
    ### results of precision
    rest_pre_ensem_mean_vgg_cnn0 = np.array(rest_pre_ensem_vgg_cnn0).mean()
    rest_pre_ensem_std_vgg_cnn0 = np.array(rest_pre_ensem_vgg_cnn0).std() 
    rest_pre_ensem_conf_vgg_cnn0 = st.t.interval(0.95, len(rest_pre_ensem_vgg_cnn0)-1, loc=np.mean(rest_pre_ensem_vgg_cnn0), scale=st.sem(rest_pre_ensem_vgg_cnn0))
    focus_pre_ensem_mean_vgg_cnn0 = np.array(focus_pre_ensem_vgg_cnn0).mean()
    focus_pre_ensem_std_vgg_cnn0 = np.array(focus_pre_ensem_vgg_cnn0).std()
    focus_pre_ensem_conf_vgg_cnn0 = st.t.interval(0.95, len(focus_pre_ensem_vgg_cnn0)-1, loc=np.mean(focus_pre_ensem_vgg_cnn0), scale=st.sem(focus_pre_ensem_vgg_cnn0))
    ### results of recall
    rest_rec_ensem_mean_vgg_cnn0 = np.array(rest_rec_ensem_vgg_cnn0).mean()
    rest_rec_ensem_std_vgg_cnn0 = np.array(rest_rec_ensem_vgg_cnn0).std() 
    rest_rec_ensem_conf_vgg_cnn0 = st.t.interval(0.95, len(rest_rec_ensem_vgg_cnn0)-1, loc=np.mean(rest_rec_ensem_vgg_cnn0), scale=st.sem(rest_rec_ensem_vgg_cnn0))
    focus_rec_ensem_mean_vgg_cnn0 = np.array(focus_rec_ensem_vgg_cnn0).mean()
    focus_rec_ensem_std_vgg_cnn0 = np.array(focus_rec_ensem_vgg_cnn0).std()
    focus_rec_ensem_conf_vgg_cnn0 = st.t.interval(0.95, len(focus_rec_ensem_vgg_cnn0)-1, loc=np.mean(focus_rec_ensem_vgg_cnn0), scale=st.sem(focus_rec_ensem_vgg_cnn0))    

    ### optimal weight ensemble of vgg and 1d_cnn
    acc_ensem_mean_vgg_cnn = np.array(acc_ensem_vgg_cnn).mean()
    acc_ensem_std_vgg_cnn = np.array(acc_ensem_vgg_cnn).std()
    acc_ensem_conf_vgg_cnn = st.t.interval(0.95, len(acc_ensem_vgg_cnn)-1, loc=np.mean(acc_ensem_vgg_cnn), scale=st.sem(acc_ensem_vgg_cnn))   
    ### results of precision
    rest_pre_ensem_mean_vgg_cnn = np.array(rest_pre_ensem_vgg_cnn).mean()
    rest_pre_ensem_std_vgg_cnn = np.array(rest_pre_ensem_vgg_cnn).std() 
    rest_pre_ensem_conf_vgg_cnn = st.t.interval(0.95, len(rest_pre_ensem_vgg_cnn)-1, loc=np.mean(rest_pre_ensem_vgg_cnn), scale=st.sem(rest_pre_ensem_vgg_cnn))
    focus_pre_ensem_mean_vgg_cnn = np.array(focus_pre_ensem_vgg_cnn).mean()
    focus_pre_ensem_std_vgg_cnn = np.array(focus_pre_ensem_vgg_cnn).std()
    focus_pre_ensem_conf_vgg_cnn = st.t.interval(0.95, len(focus_pre_ensem_vgg_cnn)-1, loc=np.mean(focus_pre_ensem_vgg_cnn), scale=st.sem(focus_pre_ensem_vgg_cnn))
    ### results of recall
    rest_rec_ensem_mean_vgg_cnn = np.array(rest_rec_ensem_vgg_cnn).mean()
    rest_rec_ensem_std_vgg_cnn = np.array(rest_rec_ensem_vgg_cnn).std() 
    rest_rec_ensem_conf_vgg_cnn = st.t.interval(0.95, len(rest_rec_ensem_vgg_cnn)-1, loc=np.mean(rest_rec_ensem_vgg_cnn), scale=st.sem(rest_rec_ensem_vgg_cnn))
    focus_rec_ensem_mean_vgg_cnn = np.array(focus_rec_ensem_vgg_cnn).mean()
    focus_rec_ensem_std_vgg_cnn = np.array(focus_rec_ensem_vgg_cnn).std()
    focus_rec_ensem_conf_vgg_cnn = st.t.interval(0.95, len(focus_rec_ensem_vgg_cnn)-1, loc=np.mean(focus_rec_ensem_vgg_cnn), scale=st.sem(focus_rec_ensem_vgg_cnn))    
        
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f3 = args.results + '{}_restults.txt'.format(args.method)
    with open(results_f3, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.write("vgg and 1d_cnn equal weight ensemble method: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (acc_ensem_mean_vgg_cnn0, acc_ensem_std_vgg_cnn0, acc_ensem_conf_vgg_cnn0[0],acc_ensem_conf_vgg_cnn0[1]))
        f.write("vgg and 1d_cnn equal weight ensemble method: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_ensem_mean_vgg_cnn0, rest_pre_ensem_std_vgg_cnn0, rest_pre_ensem_conf_vgg_cnn0[0], rest_pre_ensem_conf_vgg_cnn0[1]))
        f.write("vgg and 1d_cnn equal weight ensemble method: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_ensem_mean_vgg_cnn0, focus_pre_ensem_std_vgg_cnn0, focus_pre_ensem_conf_vgg_cnn0[0], focus_pre_ensem_conf_vgg_cnn0[1]))
        f.write("vgg 1d_cnn equal weight ensemble method: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_ensem_mean_vgg_cnn0, rest_rec_ensem_std_vgg_cnn0, rest_rec_ensem_conf_vgg_cnn0[0], rest_rec_ensem_conf_vgg_cnn0[1]))
        f.write("vgg 1d_cnn equal weight ensemble method: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_ensem_mean_vgg_cnn0, focus_rec_ensem_std_vgg_cnn0, focus_rec_ensem_conf_vgg_cnn0[0], focus_rec_ensem_conf_vgg_cnn0[1]))
        f.write('--'*40 +'\n')
        f.write("vgg 1d_cnn optimal weight ensemble method: acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (acc_ensem_mean_vgg_cnn, acc_ensem_std_vgg_cnn, acc_ensem_conf_vgg_cnn[0],acc_ensem_conf_vgg_cnn[1]))
        f.write("vgg 1d_cnn optimal weight ensemble method: rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_ensem_mean_vgg_cnn, rest_pre_ensem_std_vgg_cnn, rest_pre_ensem_conf_vgg_cnn[0], rest_pre_ensem_conf_vgg_cnn[1]))
        f.write("vgg 1d_cnn optimal weight ensemble method: focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_ensem_mean_vgg_cnn, focus_pre_ensem_std_vgg_cnn, focus_pre_ensem_conf_vgg_cnn[0], focus_pre_ensem_conf_vgg_cnn[1]))
        f.write("vgg 1d_cnn optimal weight ensemble method: rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_ensem_mean_vgg_cnn, rest_rec_ensem_std_vgg_cnn, rest_rec_ensem_conf_vgg_cnn[0], rest_rec_ensem_conf_vgg_cnn[1]))
        f.write("vgg 1d_cnn optimal weight ensemble method: focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_ensem_mean_vgg_cnn, focus_rec_ensem_std_vgg_cnn, focus_rec_ensem_conf_vgg_cnn[0], focus_rec_ensem_conf_vgg_cnn[1]))
 
    ###########################################################################    
    ### equal weight method: ensemble on all methods
    acc_ensem_mean_en0 = np.array(acc_ensem_ensem0).mean()
    acc_ensem_std_en0 = np.array(acc_ensem_ensem0).std()
    acc_ensem_conf_en0 = st.t.interval(0.95, len(acc_ensem_ensem0)-1, loc=np.mean(acc_ensem_ensem0), scale=st.sem(acc_ensem_ensem0))   
    ### results of precision
    rest_pre_ensem_mean_en0 = np.array(rest_pre_ensem_ensem0).mean()
    rest_pre_ensem_std_en0 = np.array(rest_pre_ensem_ensem0).std() 
    rest_pre_ensem_conf_en0 = st.t.interval(0.95, len(rest_pre_ensem_ensem0)-1, loc=np.mean(rest_pre_ensem_ensem0), scale=st.sem(rest_pre_ensem_ensem0))
    focus_pre_ensem_mean_en0 = np.array(focus_pre_ensem_ensem0).mean()
    focus_pre_ensem_std_en0 = np.array(focus_pre_ensem_ensem0).std()
    focus_pre_ensem_conf_en0 = st.t.interval(0.95, len(focus_pre_ensem_ensem0)-1, loc=np.mean(focus_pre_ensem_ensem0), scale=st.sem(focus_pre_ensem_ensem0))
    ### results of recall
    rest_rec_ensem_mean_en0 = np.array(rest_rec_ensem_ensem0).mean()
    rest_rec_ensem_std_en0 = np.array(rest_rec_ensem_ensem0).std() 
    rest_rec_ensem_conf_en0 = st.t.interval(0.95, len(rest_rec_ensem_ensem0)-1, loc=np.mean(rest_rec_ensem_ensem0), scale=st.sem(rest_rec_ensem_ensem0))
    focus_rec_ensem_mean_en0 = np.array(focus_rec_ensem_ensem0).mean()
    focus_rec_ensem_std_en0 = np.array(focus_rec_ensem_ensem0).std()
    focus_rec_ensem_conf_en0 = st.t.interval(0.95, len(focus_rec_ensem_ensem0)-1, loc=np.mean(focus_rec_ensem_ensem0), scale=st.sem(focus_rec_ensem_ensem0))  
    
    ### optimal weight method: ensemble on all methods
    acc_ensem_mean_en1 = np.array(acc_ensem_ensem1).mean()
    acc_ensem_std_en1 = np.array(acc_ensem_ensem1).std()
    acc_ensem_conf_en1 = st.t.interval(0.95, len(acc_ensem_ensem1)-1, loc=np.mean(acc_ensem_ensem1), scale=st.sem(acc_ensem_ensem1))   
    ### results of precision
    rest_pre_ensem_mean_en1 = np.array(rest_pre_ensem_ensem1).mean()
    rest_pre_ensem_std_en1 = np.array(rest_pre_ensem_ensem1).std() 
    rest_pre_ensem_conf_en1 = st.t.interval(0.95, len(rest_pre_ensem_ensem1)-1, loc=np.mean(rest_pre_ensem_ensem1), scale=st.sem(rest_pre_ensem_ensem1))
    focus_pre_ensem_mean_en1 = np.array(focus_pre_ensem_ensem1).mean()
    focus_pre_ensem_std_en1 = np.array(focus_pre_ensem_ensem1).std()
    focus_pre_ensem_conf_en1 = st.t.interval(0.95, len(focus_pre_ensem_ensem1)-1, loc=np.mean(focus_pre_ensem_ensem1), scale=st.sem(focus_pre_ensem_ensem1))
    ### results of recall
    rest_rec_ensem_mean_en1 = np.array(rest_rec_ensem_ensem1).mean()
    rest_rec_ensem_std_en1 = np.array(rest_rec_ensem_ensem1).std() 
    rest_rec_ensem_conf_en1 = st.t.interval(0.95, len(rest_rec_ensem_ensem1)-1, loc=np.mean(rest_rec_ensem_ensem1), scale=st.sem(rest_rec_ensem_ensem1))
    focus_rec_ensem_mean_en1 = np.array(focus_rec_ensem_ensem1).mean()
    focus_rec_ensem_std_en1 = np.array(focus_rec_ensem_ensem1).std()
    focus_rec_ensem_conf_en1 = st.t.interval(0.95, len(focus_rec_ensem_ensem1)-1, loc=np.mean(focus_rec_ensem_ensem1), scale=st.sem(focus_rec_ensem_ensem1)) 
    
    now = datetime.now() 
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    results_f3 = args.results + '{}_restults.txt'.format(args.method)
    with open(results_f3, "a") as f:
        f.write('*'*40 + date_time + '*'*40 +'\n')
        f.write("equal weight ensemble all methods (ave all): acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (acc_ensem_mean_en0, acc_ensem_std_en0, acc_ensem_conf_en0[0],acc_ensem_conf_en0[1]))
        f.write("equal weight ensemble all methods (ave all): rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_ensem_mean_en0, rest_pre_ensem_std_en0, rest_pre_ensem_conf_en0[0], rest_pre_ensem_conf_en0[1]))
        f.write("equal weight ensemble all methods (ave all): focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_ensem_mean_en0, focus_pre_ensem_std_en0, focus_pre_ensem_conf_en0[0], focus_pre_ensem_conf_en0[1]))
        f.write("equal weight ensemble all methods (ave all): rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_ensem_mean_en0, rest_rec_ensem_std_en0, rest_rec_ensem_conf_en0[0], rest_rec_ensem_conf_en0[1]))
        f.write("equal weight ensemble all methods (ave all): focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_ensem_mean_en0, focus_rec_ensem_std_en0, focus_rec_ensem_conf_en0[0], focus_rec_ensem_conf_en0[1]))
        f.write('--'*40 +'\n')
        f.write("optimal weight ensemble all methods (ave all): acc with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (acc_ensem_mean_en1, acc_ensem_std_en1, acc_ensem_conf_en1[0],acc_ensem_conf_en1[1]))
        f.write("optimal weight ensemble all methods (ave all): rest prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_pre_ensem_mean_en1, rest_pre_ensem_std_en1, rest_pre_ensem_conf_en1[0], rest_pre_ensem_conf_en1[1]))
        f.write("optimal weight ensemble all methods (ave all): focus prec with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_pre_ensem_mean_en1, focus_pre_ensem_std_en1, focus_pre_ensem_conf_en1[0], focus_pre_ensem_conf_en1[1]))
        f.write("optimal weight ensemble all methods (ave all): rest recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (rest_rec_ensem_mean_en1, rest_rec_ensem_std_en1, rest_rec_ensem_conf_en1[0], rest_rec_ensem_conf_en1[1]))
        f.write("optimal weight ensemble all methods (ave all): focus recall with mean of %0.4f, a std of %0.4f, and conf_int of (%0.4f, %0.4f)\n" % (focus_rec_ensem_mean_en1, focus_rec_ensem_std_en1, focus_rec_ensem_conf_en1[0], focus_rec_ensem_conf_en1[1]))
        f.write('--'*40 +'\n')
        f.write('the totall runing time is {} \n'.format(end))
    
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='')
    parser.add_argument('--method', type=str, default='ensemble',
                        help='') 
    parser.add_argument('--test', type=bool, default=False,
                        help='')      
    parser.add_argument('--results', type=str, default='./weighted_ensemble_results_6mins/',
                        help='')  
    parser.add_argument('--cuda', type=bool,  default=True,
                        help='use CUDA')  
    parser.add_argument('--device_id', type=str, default='0')
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id  
 
    # seed_everything(args.seed)
    
    torch.cuda.is_available()
    
    results_face = args.results+'facial_image/'
    results_ecg = args.results+'ecg/'
    if not os.path.exists(results_face):
        os.makedirs(results_face)
    if not os.path.exists(results_ecg):
        os.makedirs(results_ecg)    

    main(args, results_face, results_ecg)

    
