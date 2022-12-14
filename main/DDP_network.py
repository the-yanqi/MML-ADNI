import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import torch.nn.init as init
import torch.optim as optim
from time import time
from copy import copy, deepcopy
import numpy as np
from torch.utils.data import DataLoader
from data import BidsMriBrainDataset, ToTensor, GaussianSmoothing, MriDataset, standardizer
from training_functions import run
import torchvision
import torch.multiprocessing as mp
import torch.distributed as dist
import time
import nibabel as nib
from nilearn import plotting
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from nilearn.image import mean_img
from torchmetrics import MetricCollection
import torchmetrics
from model import *
import torchio as tio

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
            # if step == 3:
            #     break
            images, diagnoses = sample['image'].cuda(non_blocking=True), sample['diagnosis_after_12_months'].cuda(non_blocking=True)
            if 'joint' in classifier:
                tab_inputs = sample['tab_data'].cuda(non_blocking=True)
                outputs = model(images, tab_inputs)
            else:
                outputs = model(images)
            # save for compute train acc
            pred_scores = F.softmax(outputs,1)
            acc_val_batch = metric(pred_scores, diagnoses)

            all_prediction_scores.append(pred_scores)
            print('Step {} / total {} processed'.format(step, len(dataloader)))

    valid_metric = metric.compute()
    if verbose:
        print('Validation ACC: ' + str(valid_metric['Accuracy'].cpu().data.numpy()))
    metric.reset()

    all_prediction_scores = torch.cat(all_prediction_scores,0).cpu().data.numpy()
    return valid_metric, all_prediction_scores

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

def run_DDP(gpu,train_list,test_list,valid_list, nr, gpus, world_size, model, optimizer, device, batch_size=4,  
            epochs=100, results_path='scratch/di2078/results', model_name='model', classifier='vgg',
            phase = 'training', model_path = None):

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
    
    #composed = torchvision.transforms.Compose([GaussianSmoothing(0), ToTensor(True)])

    train_path='/scratch/yx2105/shared/MLH/data/train.csv'
    test_path='/scratch/yx2105/shared/MLH/data/test.csv'
    valid_path='/scratch/yx2105/shared/MLH/data/val.csv'
    sigma = 0
    
    
   
    #composed = tio.transforms.Compose([tio.RandomFlip(axes=(0,1,2)),tio.RandomBlur(std=0.3)])
    #trainset = MriDataset(train_list, composed)
    testset = MriDataset(test_list, None)
    validset = MriDataset(valid_list, None)

    
    #train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True, num_replicas=world_size, rank=rank)
    #trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        testset, shuffle=False, num_replicas=world_size, rank=rank)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,sampler=test_sampler)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        validset, shuffle=False, num_replicas=world_size, rank=rank)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,sampler=valid_sampler)
    ############################################################

    t0 = time.time()

    
    print('Length validation set', len(validset))
    print('Length test set', len(testset))    
    print('Loading complete.')
   

    # prepare metrics
    acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3, average='macro')
    auroc_metric = torchmetrics.AUROC(task="multiclass", num_classes=3)
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=3)
    metric = MetricCollection([acc_metric, auroc_metric, f1_metric ])

    model.metric = metric
    model.to(gpu)
    model = nn.parallel.DistributedDataParallel(model,device_ids=[gpu])

    # define loss function
    #class_weights = torch.tensor([0.15,0.8,0.05],dtype=torch.float).cuda(gpu) #CN,AD,MCI
    #criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    # prepare result file
    
    results_df = pd.DataFrame(columns=['epoch', 'training_time', 'loss_train', 'acc_train', 'acc_validation', 'auroc_validation', 'f1-score_validation'])
    filename = os.path.join(results_path, 'performance.csv')
    
    with open(filename, 'w') as f:
            results_df.to_csv(f, index=False, sep='\t')

    model.train()
    acc_valid_max = 0
    best_epoch = 0
    best_model = copy(model)
    print('Before training epoch')
    # The program stops when the network learnt the training data
    if phase == 'training':
        epoch = 0
        acc_valid = 0
        while epoch < epochs: #and acc_train < 100 - tol:
            
            
            #################################
            #new trainset for every epoch
            cn=0
            mci=0
            ad=0
            train_list_epoch=[]
            import random
            random.shuffle(train_list)
            for x in range(len(train_list)):
                if mci == 200 and ad == 100 and cn == 100:break
                mean = torch.mean(train_list[x]['image'])
                std = torch.std(train_list[x]['image'])
                norm = torchvision.transforms.Normalize(mean,std) 
                if train_list[x]['diagnosis_after_12_months'] == 0 and cn < 200:
                    cn = cn+1
                    #train_list[x]['image'] =  norm(train_list[x]['image'])
                    train_list_epoch.append(train_list[x])
                if train_list[x]['diagnosis_after_12_months'] == 1 and ad < 100:
                    ad = ad+1
                    #train_list[x]['image'] =  norm(train_list[x]['image'])
                    train_list_epoch.append(train_list[x])
                if train_list[x]['diagnosis_after_12_months'] == 2 and mci < 200:
                    mci = mci+1
                    #train_list[x]['image'] =  norm(train_list[x]['image'])
                    train_list_epoch.append(train_list[x])
            
            trainset = MriDataset(train_list_epoch, None)
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True, num_replicas=world_size, rank=rank)
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,sampler=train_sampler)
            print('Length training set', len(trainset))
            ################################
       ############################################################################

            loss_train = []
            running_loss = 0
            for i, data in enumerate(trainloader, 0):
                trainloader.sampler.set_epoch(epoch)

                inputs, labels = data['image'].cuda(non_blocking=True), data['diagnosis_after_12_months'].cuda(non_blocking=True)
                optimizer.zero_grad()
                if 'joint' in classifier:
                    tab_inputs = data['tab_data'].to(device)
                    outputs = model(inputs, tab_inputs)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()

                loss_train.append(loss.item())

                # save for compute train acc
                preds_score = F.softmax(outputs, 1)
                #print(preds_score,labels,"pred and label")
                acc_train_collection = metric(preds_score, labels)
                #print(acc_train_collection,"acc train")
                acc_train_batch = acc_train_collection['Accuracy']
                # print statistics
                running_loss += loss.item()
                if i % 5 == 4:  # print every 10 mini-batches
                    print('Training epoch %d : step %d / %d loss: %f' %
                        (epoch + 1,  i + 1, len(trainloader),running_loss))
                    print('ACC train at this batch: {}'.format(acc_train_batch))
                    running_loss = 0.0
            
            print('Finished Epoch: %d' % (epoch + 1))
            save_interval = 1
            # save performance
            if results_path is not None and epoch % save_interval == save_interval - 1:
                # compute train acc
                metric_train = metric.compute()
                acc_train = metric_train['Accuracy'].cpu().data.numpy()
                print('Training ACC: {}'.format(acc_train))
                metric.reset()
                training_time = time.time() - t0

                # evaluate and compute val metric
                metric_valid, all_prediction_scores = test(model, validloader, metric, classifier, device=device)

                acc_valid = metric_valid['Accuracy'].cpu().data.numpy()
                auroc_valid = metric_valid['AUROC'].cpu().data.numpy()
                f1_valid = metric_valid['F1Score'].cpu().data.numpy()
                if rank == 0:
                    row = np.array([epoch + 1, training_time, np.mean(loss_train) , acc_train ,acc_valid, auroc_valid, f1_valid
                                ]).reshape(1, -1)
                    row_df = pd.DataFrame(row, columns=['epoch', 'training_time', 'loss_train', 'acc_train', 'acc_validation', 'auroc_validation', 'f1-score_validation'])                                  
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')

                # save model
                if acc_valid > acc_valid_max:
                    acc_valid_max = copy(acc_valid)
                    auroc_valid_max = copy(auroc_valid)
                    f1_valid_max = copy(f1_valid)
                    best_epoch = copy(epoch)
                    best_model = deepcopy(model)
                    if rank == 0:
                        torch.save(model.state_dict(), os.path.join(results_path,"checkpoint_best_val.ckpt"))
                    np.save(os.path.join(results_path,"predictions_best_val"), all_prediction_scores)
            print('Validation finished Epoch: {}'.format(epoch + 1))
            epoch += 1

        if rank == 0:
            row = np.array([best_epoch, time.time() - t0, 000 , 000 , acc_valid_max, auroc_valid_max,f1_valid_max
                        ]).reshape(1, -1)
            row_df = pd.DataFrame(row, columns=['epoch', 'training_time', 'loss_train', 'acc_train', 'acc_validation', 'auroc_validation', 'f1-score_validation'])                                  
            with open(filename, 'a') as f:
                row_df.to_csv(f, header=False, index=False, sep='\t')


    elif phase == 'inference':
        weights = torch.load(model_path)
        model.load_state_dict(weights)
        metric_test, all_prediction_scores = test(model, testloader, device=device, classifier = train_args['classifier'])

        np.save(os.path.join(results_path,"predictions_best_test"), all_prediction_scores)
        print(metric_test)
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
    parser.add_argument("--phase", type=str, default='training',
                        help='[training, inference]')
    parser.add_argument("--model_path", type=str, default='/path/to/model/weights',
                        help='classifier selected')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of classes in the dataset')
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
    results_path = os.path.join(args.results_path, args.name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)


    if args.classifier == 'vgg':
        classifier = VGG(n_classes=args.n_classes)
    elif args.classifier == 'cnn':
        classifier = CNNModel(n_classes=args.n_classes)
    elif 'joint' in args.classifier:
        img_classifier = args.classifier.split('_')[-1]
        classifier = joint_model(tab_in_shape = 49, enc_shape = 8, n_classes = 3, classifier=img_classifier)
    else:
        raise ValueError('Unknown classifier')

    # Initialization
    classifier.apply(weights_init)
    optimizer = optim.Adam(filter(lambda param: param.requires_grad, classifier.parameters()), lr=lr,weight_decay=1e-5)
    ########################################
    # Training
    import pickle
    
    with open("/scratch/di2078/shared/MLH/data/MML-ADNI/main/train_pickle", "rb") as f:
         train_list = pickle.load(f)
    print("loaded train")
    with open("/scratch/di2078/shared/MLH/data/MML-ADNI/main/test_pickle", "rb") as f1:
         test_list = pickle.load(f1)
    print("loaded test")

    with open("/scratch/di2078/shared/MLH/data/MML-ADNI/main/valid_pickle", "rb") as f2:
         valid_list = pickle.load(f2)
    print("loaded valid")
    ######################################

    world_size = args.gpus * args.nodes
    torch.multiprocessing.spawn(run_DDP, args=(train_list,test_list,valid_list,args.nr,args.gpus,world_size, classifier, 
                        optimizer, device, args.batch_size, args.epochs, results_path, 'model', args.classifier, args.phase, args.model_path), nprocs=args.gpus)
    

