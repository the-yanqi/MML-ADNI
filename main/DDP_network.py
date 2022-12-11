import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path
import os
import pandas as pd
import torch.nn.init as init
import torch.optim as optim
from time import time
from copy import copy, deepcopy
import numpy as np

import torchmetrics
from model import *

def weights_init(m):
    """Initialize the weights of convolutional and fully connected layers"""
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.xavier_normal_(m.weight.data)


def test(model, dataloader, metric, classifier, device, verbose=True):
    """
    Computes the balanced accuracy of the model

    :param model: the network (subclass of nn.Module)
    :param dataloader: a dataloader wrapping a dataset
    :param gpu: if True a gpu is used
    :return: balanced accuracy of the model (float)
    """
    model.eval()
    all_prediction_scores = []
    
    # The test must not interact with the learning
    with torch.no_grad():
        for step, sample in enumerate(dataloader):
            if step == 3:
                break
            images, diagnoses = sample['image'].cuda(non_blocking=True), sample['diagnosis_after_12_months'].cuda(non_blocking=True)
            if 'joint' == classifier:
                tab_inputs = sample['tab_data'].cuda(non_blocking=True)
                outputs = model(images, tab_inputs)
            else:
                outputs = model(images)
            # save for compute train acc
            pred_scores = F.softmax(outputs,1)
            acc_val_batch = metric(pred_scores, diagnoses)

            all_prediction_scores.append(pred_scores)
            print('Step {} / total {} processed'.format(step, len(dataloader)))

    acc = metric.compute().cpu().data.numpy()
    if verbose:
        print('Validation ACC: ' + str(acc))
    metric.reset()

    all_prediction_scores = torch.cat(all_prediction_scores,0).cpu().data.numpy()
    return acc, all_prediction_scores

def compute_balanced_accuracy(predicted_list, truth_list):
    # Computation of the balanced accuracy
    component = len(np.unique(truth_list))

    cluster_diagnosis_prop = np.zeros(shape=(component, component))
    for i, predicted in enumerate(predicted_list):
        truth = truth_list[i]
        cluster_diagnosis_prop[predicted, truth] += 1

    acc = 0
    diags_represented = []
    for i in range(component):
        diag_represented = np.argmax(cluster_diagnosis_prop[i])
        acc += cluster_diagnosis_prop[i, diag_represented] / np.sum(cluster_diagnosis_prop.T[diag_represented])
        diags_represented.append(diag_represented)

    acc = acc * 100 / component

    return acc

    
    
####################################################################################

def run_DDP(gpu,nr,gpus, world_size, model, optimizer, device, batch_size=4,  epochs=100, results_path='scratch/di2078/results', model_name='model', classifier='vgg'):
    from torch.utils.data import DataLoader
    from data import BidsMriBrainDataset, ToTensor, GaussianSmoothing
    from training_functions import run
    import torchvision
    import torch.multiprocessing as mp
    import torch.distributed as dist
    import time

    
    ############################################################
    rank = nr * gpus + gpu
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
    backend='nccl',
   init_method='env://',
    world_size=world_size,
    rank=rank
    )
    ############################################################
    
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    composed = torchvision.transforms.Compose([GaussianSmoothing(0), ToTensor(True)])
    train_path='/scratch/yx2105/shared/MLH/data/train.csv'
    test_path='/scratch/yx2105/shared/MLH/data/test.csv'
    valid_path='/scratch/yx2105/shared/MLH/data/val.csv'
    sigma = 0

    trainset = BidsMriBrainDataset(train_path, transform=composed)
    testset = BidsMriBrainDataset(test_path, transform=composed)
    validset = BidsMriBrainDataset(valid_path, transform=composed)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,num_replicas=world_size,rank=rank)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        testset,num_replicas=world_size,rank=rank)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,sampler=test_sampler)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        validset,num_replicas=world_size,rank=rank)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,sampler=valid_sampler)
    ############################################################
    #results_path = train_args['results_path']
    t0 = time.time()

    print('Length training set', len(trainset))
    print('Length validation set', len(validset))
    print('Length test set', len(testset))
    
    print('Loading complete.')
    
    """
    Training a model using a validation set to find the best parameters

    :param model: The neural network to train
    :param trainloader: A Dataloader wrapping the training set
    :param validloader: A Dataloader wrapping the validation set
    :param epochs: Maximal number of epochs
    :param save_interval: The number of epochs before a new tests and save
    :param results_path: the folder where the results are saved
    :param model_name: the name given to the results files
    :param tol: the tolerance allowing the model to stop learning, when the training accuracy is (100 - tol)%
    :param gpu: boolean defining the use of the gpu (if not cpu are used)
    :param lr: the learning rate used by the optimizer
    :return:
        total training time (float)
        epoch of best accuracy for the validation set (int)
        parameters of the model corresponding to the epoch of best accuracy
    """

    # prepare metrics
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=3, average='macro')
    model.metric = metric
    model.to(gpu)
    model = nn.parallel.DistributedDataParallel(model,device_ids=[gpu])

    # define loss function
    class_weights = torch.tensor([0.15,0.8,0.05],dtype=torch.float).cuda(gpu) #CN,AD,MCI
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(gpu)

    # prepare result file
    results_df = pd.DataFrame(columns=['epoch', 'training_time', 'loss_train', 'acc_train', 'acc_validation'])
    filename = path.join(results_path, 'performance.csv')
    with open(filename, 'w') as f:
        results_df.to_csv(f, index=False, sep='\t')

    model.train()
    acc_valid_max = 0
    best_epoch = 0
    best_model = copy(model)
    print('Before training epoch')
    # The program stops when the network learnt the training data
    epoch = 0
    acc_valid = 0
    while epoch < epochs: #and acc_train < 100 - tol:
        
        loss_train = []
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            if i == 10:
                break
            trainloader.sampler.set_epoch(epoch)
            #inputs, labels = data['image'].to(device), data['diagnosis_after_12_months'].to(device)
            inputs, labels = data['image'].cuda(non_blocking=True), data['diagnosis_after_12_months'].cuda(non_blocking=True)
            optimizer.zero_grad()
            if 'joint' in classifier:
                tab_inputs = data['tab_data'].cuda(non_blocking=True)
                outputs = model(inputs, tab_inputs)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())

            # save for compute train acc
            pred_scores = F.softmax(outputs, 1)
            acc_train_batch = metric(pred_scores, labels)

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every 10 mini-batches
                print('Training epoch %d : step %d / %d loss: %f' %
                      (epoch + 1,  i + 1, len(trainloader),running_loss))
                print('ACC train at this batch: {}'.format(acc_train_batch))
                running_loss = 0.0
        
        print('Training finished Epoch: %d' % (epoch + 1))
        save_interval = 1
        # save performance
        if results_path is not None and epoch % save_interval == save_interval - 1:
            # compute train acc
            acc_train = metric.compute().cpu().data.numpy()
            print('Training ACC: {}'.format(acc_train))
            metric.reset()
            training_time = time.time() - t0

            # evaluate and compute val metric
            acc_valid, all_prediction_scores = test(model, validloader, metric, classifier, device=device)

            row = np.array([epoch + 1, training_time, np.mean(loss_train) , acc_train ,acc_valid
                           ]).reshape(1, -1)
            row_df = pd.DataFrame(row, columns=['epoch', 'training_time', 'loss_train', 'acc_train', 'acc_validation'])                                  
            with open(filename, 'a') as f:
                row_df.to_csv(f, header=False, index=False, sep='\t')

            # save model
            if acc_valid > acc_valid_max:
                acc_valid_max = copy(acc_valid)
                best_epoch = copy(epoch)
                best_model = deepcopy(model)
                if rank == 0:
                    torch.save(model.state_dict(), os.path.join(results_path,"checkpoint_best_val.ckpt"))
                np.save(os.path.join(results_path,"predictions_best_val"), all_prediction_scores)
        print('Validation finished Epoch: {}'.format(epoch + 1))

        epoch += 1

    parameters_found = {'training_time': time() - t0,
            'best_epoch': best_epoch,
            'best_model': best_model,
            'acc_valid_max': acc_valid_max
           }

    #parameters_found = train_DDP(gpu, model, trainloader, validloader, optimizer, device, epochs=1000, save_interval=1, results_path=results_path, model_name='model', tol=0.0)
    
####################################################################################
if __name__ == '__main__':
    from data import BidsMriBrainDataset, ToTensor, GaussianSmoothing
    from training_functions import run
    import torchvision
    import argparse

    parser = argparse.ArgumentParser()

    # DDP
    parser.add_argument("--local_rank", dest = 'local_rank',type=int, default=0)
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    #####################################################
    parser.add_argument("results_path", type=str,
                        help="where the outputs are stored")
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training')
    # Network structure
    parser.add_argument("--classifier", type=str, default='basic',
                        help='classifier selected')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of images in batch')
    parser.add_argument("--name", type=str, default='network',
                        help="name given to the outputs and checkpoints of the parameters")
    parser.add_argument("-save", "--save_interval", type=int, default=1,
                        help="the number of epochs done between the tests and saving")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1.0,
                        help='the learning rate of the optimizer (*0.00005)')

    # Managing device
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Uses gpu instead of cpu if cuda is available')
    parser.add_argument('--on_cluster', action='store_true', default=False,
                        help='to work on the cluster of the ICM')

    args = parser.parse_args()


    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    lr = args.learning_rate * 0.00005
    results_path = path.join(args.results_path, args.name)
    if not path.exists(results_path):
        os.makedirs(results_path)
        
    #composed = torchvision.transforms.Compose([GaussianSmoothing(sigma=args.sigma), ToTensor(gpu=args.gpu)])
    #train_path='/scratch/yx2105/shared/MLH/data/train.csv'
    #test_path='/scratch/yx2105/shared/MLH/data/test.csv'
    #valid_path='/scratch/yx2105/shared/MLH/data/val.csv'
    #sigma = 0

    #trainset = BidsMriBrainDataset(train_path, transform=composed)
    #testset = BidsMriBrainDataset(test_path, transform=composed)
    #validset = BidsMriBrainDataset(valid_path, transform=composed)


    if args.classifier == 'vgg':
        classifier = VGG(n_classes=args.n_classes)
    elif args.classifier == 'cnn':
        classifier = CNNModel(n_classes=args.n_classes)
    elif 'joint' in args.classifier:
        img_classifier = args.classifier.split('_')[1]
        classifier = joint_model(tab_in_shape = 49, enc_shape = 8, n_classes = 3, classifier=img_classifier)
    else:
        raise ValueError('Unknown classifier')

    # Initialization
    classifier.apply(weights_init)
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, classifier.parameters()), lr=lr,weight_decay=1e-4)

    # Training
    #best_params = run(classifier, trainset, validset, testset, optimizer, device=device, batch_size=args.batch_size, folds=args.cross_validation, epochs=args.epochs, results_path=results_path, model_name=args.name,
                                   #save_interval=args.save_interval)
     
    
    #best_params = run_DDP(classifier, optimizer, device=device, batch_size=args.batch_size, folds=10, epochs=100, results_path=results_path, model_name='model')
    world_size = args.gpus * args.nodes
    torch.multiprocessing.spawn(run_DDP, args=(args.nr,args.gpus,world_size, classifier, optimizer, device, args.batch_size, args.epochs, results_path, 'model', args.classifier), nprocs=args.gpus)
    



