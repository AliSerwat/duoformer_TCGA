# python step_4_ScaleFormer_NorCanNec_20240505_3_classes_20X_NorCanNecExp13_0.py
# cd /uu/sci.utah.edu/tolga_scratch/bodong/projects/Xiaoya_Transformer/programs_and_results/models/
# exp 12: compare to 2024 04 13, scales = 4, total_epoch = 75, initial_lr=0.00001, val_test_batchsize is larger, added time write on training, added val_test_batchsize, added "if batch_index%bs==0 and program_mode !='only_test':""
# compare with previous nested vs unnested,
# we changed the following:
# Random crop + random rotation
# More accurate color normalization on H and E channel.(get mean, std of training set)
# We handled unbalanced train, val and test set. fixed number of each label in training, balanced val and test accuracy. 

#Things to change:
#date, exp, exp(20220xxx_expxx)
#for co-training, use "model_co_training_result", not "model_result" nor "model_backup"(search them)
#"params_model"
#image augmentation: size 256 or 224, random crop or random rotation
#GPU

#TO DO: check number of each label in train labeled. 
# length of training set, val set, test set
# 287975
# 110488
# 104787
# Train labeled each class sample number:
# [84578, 180471, 7932]
# Validation each class sample number:
# [19638, 79382, 1301]
# Test each class sample number:
# [15323, 62565, 6168]

import torchvision.transforms as transforms
import os
import torchvision.models as models
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import copy
from datasets_my_lib.sampler import RandomSampler, BatchSampler
from pytorch_balanced_sampler.sampler import SamplerFactory
from collections import Counter
import warnings

from __init__ import build_model

#cd /uu/sci.utah.edu/tolga_scratch/bodong/projects/Xiaoya_Transformer/programs_and_results/models/
#python step_4_ScaleFormer_NorCanNec_20240505_3_classes_20X_NorCanNecExp13_0.py
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']='1'#A1 may need to put it before import torch
WORK_ON_LAB_SERVER=True
program_mode='normal_training'#'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
initial_lr=0.001
lr_decay_factor=1.0#0.9
lr_patience=10000#15
total_epoch = 75
MANUAL_SET_num_iter_in_one_epoch=1000#1000#-1 means we don't manual set
train_batchsize=6
val_test_batchsize=64#max for GPU 24 GB
#label_batchsize=40#How many labeled samples in one batch!!!!!!!!!!!!!!!!!!!!Should be multiple of NUM_CLASSES
#loss_contrastive_margin=40.0
#loss_contrastive_weight=0.2*float(label_batchsize)/float(batchsize)#The value should be 0.2x(ratio of labeled data) in batch. 0.2x30/64=0.09375
#unlabel_batchsize=batchsize-label_batchsize
NUM_CLASSES=3
CLASSES_TYPES=['Normal_Type', 'Cancer', 'Necrosis']
SAME_CROP_ROTATION_AUGMENTATION_FOR_H_E=False#!!!!!!!!!!!!!!!!!!!!!!!!!!!
torch.cuda.empty_cache()
num_workers=4
original_tile_size_divided_by_sqrt2=282#400/sqrt(2)=282
training_tile_size=224#256
my_modelname = 'ScaleFormer'
bs=25#how frequent to show on terminal, show once each bs batches
save_latest_epoch_frequency=10
params_model={
        "image1_channels":1,
        "image2_channels":1,
        "classification_number":NUM_CLASSES,
    }
save_models_folder='model_ScaleFormer_backup'
save_results_folder='model_ScaleFormer_result'


#Xiaoya
classes = 3 # 100 for cifar100, 10 for cifar 10,svhn
scales = 4
depth= 12  # number of blocks
proj_dim = 768 # proj_dim: 384
heads = 12
mlp_ratio = 4.
patch_size=32 # 224 // 7
num_patches = 49
attn_drop_out = 0.
proj_drop_out = 0.
freeze_backbone = False
backbone = 'r50' # 'r18
init_values = None # 1e-5 for layer scale(from CaiT), no layer scale if None




#TCGA only, 20X mean std:
#Summary: mean_H: 0.17384087240926754  var_H: 0.035183881951123125 std_H: 0.18757367072999112
#Summary: mean_E: 0.2507321271710507  var_E: 0.02390038938049867 std_E: 0.15459750767880662
train_transformer = transforms.Compose([
    #######################transforms.ToPILImage(mode=None)#Converts a torch.*Tensor of shape C(1,3,4) x H x W or a numpy ndarray of shape H x W x C to a PIL Image while preserving the value range.
    ##transforms.Resize(256),
    ##transforms.RandomResizedCrop((256),scale=(0.64,0.64)),
    transforms.RandomRotation(360),###Random rotation to any angle!!!!!!Could be 30 degree, fill 0 for outside of image
    transforms.CenterCrop((original_tile_size_divided_by_sqrt2,original_tile_size_divided_by_sqrt2)),#400/sqrt(2)
    transforms.RandomCrop((training_tile_size,training_tile_size)),
    transforms.RandomHorizontalFlip(),###
    transforms.RandomVerticalFlip(),
    
    ## random brightness and random contrast
    #transforms.ToTensor(),#Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ##normalize
])

val_transformer = transforms.Compose([
    transforms.CenterCrop((training_tile_size,training_tile_size)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #normalize
])

test_transformer = transforms.Compose([
    transforms.CenterCrop((training_tile_size,training_tile_size)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #normalize
])

train_color_transformer_3channels=transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1),#changable
    #transforms.ColorJitter(brightness=0.1),
])

#####################################################################################################################################

def read_txt_category(txt_category):
    with open(txt_category) as file:
        lines = file.readlines()
    line_data = [line.strip() for line in lines]
    return line_data

class GPDataset(Dataset):
    def __init__(self, root_dir_RGB, list_of_each_GP_txt_path, dataset_type):
        # root_dir_RGB: folder that saves all RGB raw tiles
        # txt_benign: txt file that saves name of labeled tiles
        # txt_unlabeled: txt file that saves name of unlabeled tiles
        #https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/4
        #https://github.com/khornlund/pytorch-balanced-sampler
        self.root_dir_RGB = root_dir_RGB
        self.txt_path = list_of_each_GP_txt_path #[txt_GP_favorable, txt_GP_unfavorable]
        self.classes = CLASSES_TYPES
        self.num_cls = len(self.classes)
        self.img_list_RGB=[]
        self.dataset_type=dataset_type
        self.class_idxs=[]#for balanced sampler
        self.current_start_idx=0
        self.each_class_number=[]
        if self.dataset_type=="train" or self.dataset_type=="train_with_unlabeled" or \
            self.dataset_type=="train_only_labeled" or self.dataset_type=="validation" or self.dataset_type=="test":
            for c in range(self.num_cls):#c is different classes
                cls_list_RGB = [[os.path.join(self.root_dir_RGB,item), c] for item in read_txt_category(self.txt_path[c])]
                self.img_list_RGB += cls_list_RGB

                self.class_idxs.append(list(range(self.current_start_idx,self.current_start_idx+len(cls_list_RGB))))
                self.current_start_idx+=len(cls_list_RGB)

                self.each_class_number.append(len(cls_list_RGB))
            self.len_labeled=len(self.img_list_RGB)#empty!!!!, use _H
        
    def __len__(self):
        return len(self.img_list_RGB)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_RGB=np.asarray(Image.open(self.img_list_RGB[idx][0]))
        image_RGB=image_RGB.astype(float)
        image_RGB=image_RGB/255.0
        #image_RGB = np.expand_dims(image_RGB, axis=0)
        tensor_of_image_RGB=torch.from_numpy(image_RGB.copy()).float()#change numpy array to tensor
        tensor_of_image_RGB = tensor_of_image_RGB.permute(2, 0, 1)
        if idx==0:
            print('tensor_of_image_RGB.shape')
            print(tensor_of_image_RGB.shape)
        if self.dataset_type=="train" or self.dataset_type=="train_with_unlabeled" or \
            self.dataset_type=="train_only_labeled" or self.dataset_type=="train_only_unlabeled":
            tensor_of_image_RGB=train_color_transformer_3channels(tensor_of_image_RGB)#color transform: color jittering can only apply channel 3(1)
            tensor_of_image_RGB=train_transformer(tensor_of_image_RGB)
        else:
            tensor_of_image_RGB=test_transformer(tensor_of_image_RGB)
        
        sample = {'img': tensor_of_image_RGB,
                    'label': int(self.img_list_RGB[idx][1]),
                    'labeled': 1.0}
        return sample

device = 'cuda'

def train(optimizer, epoch, train_only_labeled_loader, num_iter_in_one_epoch):
    my_model.train()#will let program update each layer by backpropagation
    train_loss = 0
    train_correct = 0
    count_labeled = 0
    train_only_labeled_iter = iter(train_only_labeled_loader)
    batch_index=0
    for iter_index in range(num_iter_in_one_epoch):
        batch_samples_only_labeled = train_only_labeled_iter.next()
        #dict type has no attribute "to": batch_samples_only_labeled, batch_samples_only_unlabeled = batch_samples_only_labeled.to(device), batch_samples_only_unlabeled.to(device)
        if batch_index<1:
            print('Count occurence of each class in labeled portion of batch:')
            print(torch.bincount(batch_samples_only_labeled['label']))
        # data1=torch.cat((batch_samples_only_labeled['img'], batch_samples_only_unlabeled['img1']), dim=0)
        # target=torch.cat((batch_samples_only_labeled['label'], batch_samples_only_unlabeled['label']), dim=0)
        # labeled=torch.cat((batch_samples_only_labeled['labeled'], batch_samples_only_unlabeled['labeled']), dim=0)

        data=batch_samples_only_labeled['img']
        target=batch_samples_only_labeled['label']
        labeled=batch_samples_only_labeled['labeled']
        shuffle_indexes = torch.randperm(data.shape[0])
        #Shuffle again!!!
        data = data[shuffle_indexes]
        target = target[shuffle_indexes]
        labeled = labeled[shuffle_indexes]
        
        # move data to device
        data, target, labeled = data.to(device), target.to(device), labeled.to(device)

        output = my_model(data)
        
        loss_func = nn.CrossEntropyLoss(reduction="none")
        loss_cross_entropy_batch=loss_func(output, target.long())

        #XY:
        # lambda_reg = 0.2
        # l2_regularization = 0
        # for param in my_model.parameters():
        #     l2_regularization += torch.norm(param)**2  XY
        #loss_cross_entropy=torch.sum(loss_cross_entropy_batch + lambda_reg * l2_regularization)

        loss_cross_entropy=torch.sum(loss_cross_entropy_batch)

        #loss=loss_cross_entropy#This is total sum loss in whole batch, but will print avg loss in each case 
        if batch_index==0:
            print('loss_cross_entropy in first batch=')
            print(loss_cross_entropy)
        train_loss += loss_cross_entropy
        
        optimizer.zero_grad()
        loss_cross_entropy.backward()#backpropogate
        optimizer.step()#upgrade weight

        output_labeled=output#[labeled_boolean_list]
        target_labeled=target#[labeled_boolean_list]
        count_labeled+=torch.sum(labeled)
        #if torch.numel(output_labeled)>0:#check number of elements
        pred = output_labeled.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target_labeled.long().view_as(pred)).sum().item()
    
        # Display progress and write to tensorboard
        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.4f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index+1, num_iter_in_one_epoch,
                100.0 * (batch_index+1) / num_iter_in_one_epoch, loss_cross_entropy.item()/ bs))
        batch_index+=1
    print('Train set: Average loss: {:.4f}, Accuracy in labeled data: {}/{} ({:.4f}%)\n'.format(
        train_loss/count_labeled, train_correct, count_labeled,
        100.0 * train_correct / count_labeled))
    f = open(save_results_folder+'/20240505_NorCanNecExp13_0_train01_{}.txt'.format(my_modelname), 'a+')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    f.write('Train set: Epoch: {} Average loss: {:.4f}, Accuracy in labeled data: {}/{} ({:.4f}%), current time= {}\n'.format(epoch, 
        train_loss/count_labeled, train_correct, count_labeled,
        100.0 * float(train_correct) / float(count_labeled), current_time))
    f.close()#!!!!!!!!!!!!Should modify train_loss/count_labeled!!!!!!!!!!!!!!

#val process is defined here

def val(my_val_model):
    
    my_val_model.eval()
    val_loss = 0.0
    correct = 0
    results = []
    
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    if program_mode =='only_test':
        confusion_matrix = [ [0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    # Don't update model
    with torch.no_grad():
        pred_list=[]
        score_list=[]
        target_list=[]
        # Predict
        print('Start validation...')
        for batch_index, batch_samples in enumerate(val_loader):
            data, target, labeled = batch_samples['img'].to(device), \
                batch_samples['label'].to(device), batch_samples['labeled'].to(device)
            output = my_val_model(data)
            val_loss += loss_func(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)#Returns the indices of the maximum values along an axis.
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            target_np=target.long().cpu().numpy()
            pred_list= np.concatenate((pred_list, pred.cpu().numpy()), axis=None)
            score_list= np.concatenate((score_list, score.cpu().numpy()[:,1]), axis=None)
            target_list= np.concatenate((target_list, target_np), axis=None)
            raw_unbalanced_val_acc=100.0 * correct / len(val_loader.dataset)
            if batch_index%bs==0 and program_mode !='only_test':
                print(f'working on val batch index {batch_index}')
        
#######################This block is for visualizing results#########################################
            if program_mode =='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
                path_val_scores=save_results_folder+'/20240505_NorCanNecExp13_0_val_scores.txt'
                f = open(path_val_scores, 'a+')
                for row in range(score.size(dim=0)):
                    # #for i in range(params_model["classification_number"]):
                    # #    f.write('{:.4f} '.format(score[row][i]))
                    # f.write('{} '.format(pred[row][0]))
                    # f.write('{} '.format(target[row]))
                    # f.write('\n')
                    confusion_matrix[pred[row][0]][target[row]]+=1
                # if batch_index%bs==0:
                #     print(f'working on {batch_index}/ {row}')
                f.close()
                if batch_index==0:
                    print("Scores, prediction, target of each sample are saved in "+path_val_scores)

        print('correct number in val is {}/{}. (we need to calculate balanced accuracy)'.format(correct, len(val_loader.dataset)))
        val_loss/=float(len(val_loader.dataset))

        if program_mode =='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
            path_val_scores=save_results_folder+'/20240505_NorCanNecExp13_0_val_scores.txt'
            f = open(path_val_scores, 'a+')
            f.write('confusion matrix: each row counts prediction, each column counts ground truth\n')
            for row in range(NUM_CLASSES):
                for col in range(NUM_CLASSES):
                    f.write(f'{confusion_matrix[row][col]}   ')
                f.write('\n')
            f.close()
           
    return target_list, score_list, pred_list, raw_unbalanced_val_acc, val_loss
    
#test process is defined here 
def test(my_test_model, epoch):
    
    my_test_model.eval()
    test_loss = 0.0
    correct = 0
    
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    #if program_mode =='only_test':
    confusion_matrix = [ [0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    # Don't update model
    with torch.no_grad():
        pred_list=[]
        score_list=[]
        target_list=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target, labeled = batch_samples['img'].to(device), \
                batch_samples['label'].to(device), batch_samples['labeled'].to(device)
            output = my_test_model(data)
            
            test_loss += loss_func(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            target_np=target.long().cpu().numpy()
            pred_list= np.concatenate((pred_list, pred.cpu().numpy()), axis=None)
            score_list= np.concatenate((score_list, score.cpu().numpy()[:,1]), axis=None)
            target_list= np.concatenate((target_list, target_np), axis=None)
            raw_unbalanced_test_acc=100.0 * correct / len(test_loader.dataset)
            if batch_index%bs==0 and program_mode !='only_test':
                print(f'working on test batch index {batch_index}')

#######################This block is for visualizing results#########################################
            #if program_mode =='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
            path_test_scores=save_results_folder+'/20240505_NorCanNecExp13_0_test_scores.txt'
            f = open(path_test_scores, 'a+')
            for row in range(score.size(dim=0)):
                # #for i in range(params_model["classification_number"]):
                # #    f.write('{:.4f} '.format(score[row][i]))
                # f.write('{} '.format(pred[row][0]))
                # f.write('{} '.format(target[row]))
                # f.write('\n')
                confusion_matrix[pred[row][0]][target[row]]+=1
            if batch_index%10==0:
                print(f'working on {batch_index}/ {row}\n')
            f.close()
            if batch_index==0:
                print("Scores, prediction, target of each sample are saved in "+path_test_scores)

        print('correct number in test is {}/{}. (we need to calculate balanced accuracy)'.format(correct, len(test_loader.dataset)))
        test_loss/=float(len(test_loader.dataset))

        #if program_mode =='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
        path_val_scores=save_results_folder+'/20240505_NorCanNecExp13_0_test_scores.txt'
        f = open(path_val_scores, 'a+')
        f.write(f'Epoch: {epoch}: confusion matrix: each row counts prediction, each column counts ground truth\n')
        for row in range(NUM_CLASSES):
            for col in range(NUM_CLASSES):
                f.write(f'{confusion_matrix[row][col]}   ')
            f.write('\n')
        f.close()

    return target_list, score_list, pred_list, raw_unbalanced_test_acc, test_loss

if __name__ == '__main__':
    if WORK_ON_LAB_SERVER:
        txt_root_dir='/usr/sci/tolga_lab/bodong/dataset_TCGA/TCGA_kidney_cancer/tiles_list_3_classes/dataset_txt_files_20230708_normal_vs_cancer_vs_necrosis_20X/'
        tile_loc='/usr/sci/tolga_lab/bodong/dataset_TCGA/TCGA_kidney_cancer/tiles/tiles_20230301_1_20X/'
    else:
        txt_root_dir='/uufs/chpc.utah.edu/common/HIPAA/proj_knudsen/TCGA-H&E-Images/kirc_Bodong/dataset/XXXXXXXXXXXXXXXX/'
        tile_loc='/uufs/chpc.utah.edu/common/HIPAA/proj_knudsen/TCGA-H&E-Images/kirc_Bodong/dataset/tiles_20230301_1_20X/'
    #class_idxs=[]#https://github.com/khornlund/pytorch-balanced-sampler/blob/master/pytorch_balanced_sampler/sampler.py
    #GPDataset definition changed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    trainset_only_labeled = GPDataset(
                            root_dir_RGB=tile_loc,
                            #['Normal_Type', 'Low_Risk', 'Intermediate_Risk','High_Risk','Necrosis']
                            list_of_each_GP_txt_path=[
                                txt_root_dir+'20230708_train_labeled_0_Normal_Type_tiles_list.txt',
                                txt_root_dir+'20230708_train_labeled_1_Cancer_tiles_list.txt',
                                txt_root_dir+'20230708_train_labeled_2_Necrosis_tiles_list.txt'],
                            dataset_type='train_only_labeled'#"train"(only use labeled data)  'train_with_unlabeled'(use both labeled, unlabeled)
                            )#train_transformerHE is not used if SAME_CROP_ROTATION_AUGMENTATION_FOR_H_E is False
    
    valset = GPDataset(
                            root_dir_RGB=tile_loc,
                            list_of_each_GP_txt_path=[
                                txt_root_dir+'20230708_val_0_Normal_Type_tiles_list.txt',
                                txt_root_dir+'20230708_val_1_Cancer_tiles_list.txt',
                                txt_root_dir+'20230708_val_2_Necrosis_tiles_list.txt'],
                            dataset_type='validation')
    testset = GPDataset(
                            root_dir_RGB=tile_loc,
                            list_of_each_GP_txt_path=[
                                txt_root_dir+'20230708_test_0_Normal_Type_tiles_list.txt',
                                txt_root_dir+'20230708_test_1_Cancer_tiles_list.txt',
                                txt_root_dir+'20230708_test_2_Necrosis_tiles_list.txt'],
                            dataset_type='test')
    print('length of training set labeled, val set, test set')
    print(trainset_only_labeled.__len__())
    print(valset.__len__())
    print(testset.__len__())

    print('Train labeled each class sample number:')
    print(trainset_only_labeled.each_class_number)

    print('Validation each class sample number:')
    print(valset.each_class_number)

    print('Test each class sample number:')
    print(testset.each_class_number)

    # Check whether the specified path exists or not
    save_models_path_Exist = os.path.exists('./'+save_models_folder)
    if not save_models_path_Exist:
        os.makedirs('./'+save_models_folder)
        print("The new directory: "+save_models_folder+ " for saving models is created!")
    save_results_path_Exist = os.path.exists('./'+save_results_folder)
    if not save_results_path_Exist:
        os.makedirs('./'+save_results_folder)
        print("The new directory: "+save_models_folder+ " for saving results is created!")

    num_iter_in_one_epoch=int(trainset_only_labeled.__len__()/train_batchsize)
    if MANUAL_SET_num_iter_in_one_epoch>0:
        num_iter_in_one_epoch=MANUAL_SET_num_iter_in_one_epoch
    #Drawing with replacement means that after the item is drawn, it is placed back into the group and might be drawn again in the next draw(batch).
    # sampler_labeled = RandomSampler(trainset_only_labeled, replacement=True, num_samples=num_iter_in_one_epoch * label_batchsize)
    # batch_sampler_labeled = BatchSampler(sampler_labeled, label_batchsize, drop_last=True)  # yield a batch of samples one time
    
    #https://github.com/khornlund/pytorch-balanced-sampler
    batch_sampler_labeled = SamplerFactory().get(
        class_idxs=trainset_only_labeled.class_idxs,#index start with 0: [[0,1,2,3],[4,5,6],[7],[8],[9]], 5 classes
        batch_size=train_batchsize,
        n_batches=num_iter_in_one_epoch,#how many batches in one epoch
        alpha=1.0,#totally balanced
        kind='fixed'#fixed number of each label in one batch
        )

    train_only_labeled_loader = DataLoader(trainset_only_labeled, batch_sampler=batch_sampler_labeled, num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(valset, batch_size=val_test_batchsize, num_workers=num_workers, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=val_test_batchsize, num_workers=num_workers, drop_last=False, shuffle=False)

    #Xiaoya
    my_model = build_model(depth = depth, 
           embed_dim = proj_dim,  
           num_heads = heads, 
           num_classes = classes, 
           num_layers = scales, 
           num_patches = num_patches,
           proj_dim = proj_dim,  
           mlp_ratio = mlp_ratio, 
           attn_drop_rate = attn_drop_out, proj_drop_rate = proj_drop_out,
           freeze_backbone = freeze_backbone, backbone = backbone)

    #my_model = models.resnet18(pretrained=True)#.cuda()#changable origin resnet18
    #my_modelname = 'ResNet18'#changable#####################
    #num_classes=NUM_CLASSES
    #num_ftrs = my_model.fc.in_features
    #my_model.fc = nn.Linear(num_ftrs, num_classes)
    
    get_best_model=0
    best_epoch_balanced_test_acc=0.0
    ################################################### train######################################################
    if program_mode !='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
        start_epoch=1
        best_acc=0.0
        best_epoch=1
        if program_mode=='resume_best_training':# 'normal_training', 'resume_best_training', 'resume_latest_training', 'only_test'
            #choose current_best for path2weights and current_result_path
            path2weights=save_models_folder+'/20240505_NorCanNecExp13_0_current_best.pt'
            my_model.load_state_dict(torch.load(path2weights))
            current_result_path=save_results_folder+'/20240505_NorCanNecExp13_0_current_best_for_resuming.txt'
        elif program_mode=='resume_latest_training':# 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
            #choose current_latest for path2weights and current_result_path
            path2weights=save_models_folder+'/20240505_NorCanNecExp13_0_current_latest.pt'
            my_model.load_state_dict(torch.load(path2weights))
            current_result_path=save_results_folder+'/20240505_NorCanNecExp13_0_current_latest_for_resuming.txt'
        if program_mode=='resume_best_training' or program_mode=='resume_latest_training':
            current_result_file = open(current_result_path, 'r')
            current_epoch=current_result_file.readline()
            start_epoch=int(current_epoch)+1
            balanced_val_acc=float(current_result_file.readline())
            balanced_test_acc=float(current_result_file.readline())
            initial_lr=float(current_result_file.readline())
            best_epoch=int(current_result_file.readline())
            best_acc=float(current_result_file.readline())
            best_epoch_balanced_test_acc=float(current_result_file.readline())
            current_result_file.close()
        
        acc_list = []
        vote_pred = np.zeros(valset.__len__())
        vote_score = np.zeros(valset.__len__())


        #optimizer = optim.RMSprop(my_model.parameters(), lr=initial_lr, weight_decay=1e-8, momentum=0.9)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_factor, patience=lr_patience)  # goal: minimize loss

        #Xiaoya
        optimizer = optim.Adam(my_model.parameters(), lr=initial_lr,weight_decay=1e-4) 
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=initial_lr, steps_per_epoch=MANUAL_SET_num_iter_in_one_epoch, epochs=total_epoch)


        device = torch.device("cuda:0")
        my_model.to(device) 
        
        for epoch in range(start_epoch, total_epoch+1):
            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
            current_lr=optimizer.param_groups[0]['lr']
            train(optimizer, epoch, train_only_labeled_loader, num_iter_in_one_epoch)
            
            val_target_list, val_score_list, val_pred_list, raw_unbalanced_val_acc, val_loss = val(my_model)
            scheduler.step(val_loss)
            TP=[0]*NUM_CLASSES
            TN=[0]*NUM_CLASSES
            FN=[0]*NUM_CLASSES
            FP=[0]*NUM_CLASSES
            p=[0.0]*NUM_CLASSES
            r=[0.0]*NUM_CLASSES
            F1=[0.0]*NUM_CLASSES
            TPR=[0.0]*NUM_CLASSES
            TNR=[0.0]*NUM_CLASSES
            balanced_val_acc=0.0
            for class_index in range(NUM_CLASSES):
                TP[class_index]=((val_pred_list == class_index) & (val_target_list == class_index)).sum()
                TN[class_index] = ((val_pred_list != class_index) & (val_target_list != class_index)).sum()
                FN[class_index] = ((val_pred_list != class_index) & (val_target_list == class_index)).sum()
                FP[class_index] = ((val_pred_list == class_index) & (val_target_list != class_index)).sum()
                if TP[class_index]+FP[class_index]==0:
                    p[class_index]=1.0
                else:
                    p[class_index]=float(TP[class_index])/float(TP[class_index]+FP[class_index])
                if TP[class_index]+FN[class_index]==0:
                    r[class_index]=1.0
                else:
                    r[class_index]=float(TP[class_index]) / float(TP[class_index] + FN[class_index])
                if r[class_index]+p[class_index]==0.0:
                    F1[class_index]=0.0
                else:
                    F1[class_index] = 2 * r[class_index] * p[class_index] / (r[class_index] + p[class_index])
                TPR[class_index]=r[class_index]
                TNR[class_index]=float(TN[class_index])/float(TN[class_index]+FP[class_index])
                balanced_val_acc+=float(TP[class_index])/float(TP[class_index]+FN[class_index])#It is avg r
            balanced_val_acc/=NUM_CLASSES
            #AUC = roc_auc_score(val_target_list, val_score_list)
            
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('The epoch is {}, balanced val accuracy: {:.4f}, average loss: {}, previous best balanced val {} with balanced test{} at epoch{}\n'.format(epoch, balanced_val_acc, val_loss, best_acc, best_epoch_balanced_test_acc, best_epoch))
            f = open(f'{save_results_folder}/20240505_NorCanNecExp13_0_val01_{my_modelname}.txt', 'a+')
            f.write('Val: The epoch is {}, current_lr= {}, balanced accuracy: {:.4f}, average loss: {},TP= [{}], TN= [{}], FN= [{}], FP= [{}], F1= [{}], TPR= [{}], TNR= [{}], Current Time ={}\n\n'.format(epoch, current_lr, balanced_val_acc,
                val_loss,", ".join(str(item) for item in TP),", ".join(str(item) for item in TN),", ".join(str(item) for item in FN),", ".join(str(item) for item in FP),
                ", ".join(str(item) for item in F1),", ".join(str(item) for item in TPR),", ".join(str(item) for item in TNR),current_time))
            f.close()

            if best_acc<balanced_val_acc:
                best_acc=balanced_val_acc
                best_epoch=epoch
                my_best_model=copy.deepcopy(my_model)
                get_best_model=1
                torch.save(my_model.state_dict(), save_models_folder+"/20240505_NorCanNecExp13_0_current_best.pt".format(my_modelname))
                test_target_list, test_score_list, test_pred_list, raw_unbalanced_test_acc, test_loss = test(my_model,epoch)
                
                TP=[0]*NUM_CLASSES
                TN=[0]*NUM_CLASSES
                FN=[0]*NUM_CLASSES
                FP=[0]*NUM_CLASSES
                p=[0.0]*NUM_CLASSES
                r=[0.0]*NUM_CLASSES
                F1=[0.0]*NUM_CLASSES
                TPR=[0.0]*NUM_CLASSES
                TNR=[0.0]*NUM_CLASSES
                balanced_test_acc=0.0
                for class_index in range(NUM_CLASSES):
                    TP[class_index]=((test_pred_list == class_index) & (test_target_list == class_index)).sum()
                    TN[class_index] = ((test_pred_list != class_index) & (test_target_list != class_index)).sum()
                    FN[class_index] = ((test_pred_list != class_index) & (test_target_list == class_index)).sum()
                    FP[class_index] = ((test_pred_list == class_index) & (test_target_list != class_index)).sum()
                    if TP[class_index]+FP[class_index]==0:
                        p[class_index]=1.0
                    else:
                        p[class_index]=float(TP[class_index])/float(TP[class_index]+FP[class_index])
                    if TP[class_index]+FN[class_index]==0:
                        r[class_index]=1.0
                    else:
                        r[class_index]=float(TP[class_index]) / float(TP[class_index] + FN[class_index])
                    if r[class_index]+p[class_index]==0.0:
                        F1[class_index]=0.0
                    else:
                        F1[class_index] = 2 * r[class_index] * p[class_index] / (r[class_index] + p[class_index])
                    TPR[class_index]=r[class_index]
                    TNR[class_index]=float(TN[class_index])/float(TN[class_index]+FP[class_index])
                    balanced_test_acc+=float(TP[class_index])/float(TP[class_index]+FN[class_index])#It is avg r
                balanced_test_acc/=NUM_CLASSES
                #AUC = roc_auc_score(val_target_list, val_score_list)

                print('balanced test acc',balanced_test_acc)

                best_epoch_balanced_test_acc=balanced_test_acc
                current_result_path=save_results_folder+'/20240505_NorCanNecExp13_0_current_best_for_resuming.txt'
                current_result_file = open(current_result_path, 'w')#overwrite
                current_result_file.write(f'{epoch}\n')
                current_result_file.write(f'{balanced_val_acc}\n')
                current_result_file.write(f'{balanced_test_acc}\n')
                current_result_file.write(f'{current_lr}\n')
                current_result_file.write(f'{best_epoch}\n')
                current_result_file.write(f'{best_acc}\n')
                current_result_file.write(f'{best_epoch_balanced_test_acc}\n')
                current_result_file.close()

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                f = open(f'{save_results_folder}/20240505_NorCanNecExp13_0_test01_{my_modelname}.txt', 'a+')
                f.write('Test: The epoch is {}, balanced val accuracy: {:.4f}, balanced test accuracy: {:.4f}, average loss: {},TP= [{}], TN= [{}], FN= [{}], FP= [{}], F1= [{}], TPR= [{}], TNR= [{}], Current Time ={}\n\n'.format(epoch, balanced_val_acc, balanced_test_acc,
                    test_loss,", ".join(str(item) for item in TP),", ".join(str(item) for item in TN),", ".join(str(item) for item in FN),", ".join(str(item) for item in FP),
                    ", ".join(str(item) for item in F1),", ".join(str(item) for item in TPR),", ".join(str(item) for item in TNR),current_time))
                f.close()
                
            if epoch%save_latest_epoch_frequency==0:
                torch.save(my_model.state_dict(), save_models_folder+"/20240505_NorCanNecExp13_0_current_latest.pt".format(my_modelname))
                current_result_path=save_results_folder+'/20240505_NorCanNecExp13_0_current_latest_for_resuming.txt'
                current_result_file = open(current_result_path, 'w')#overwrite
                current_result_file.write(f'{epoch}\n')
                current_result_file.write(f'{balanced_val_acc}\n')
                current_result_file.write(f'{balanced_test_acc}\n')#This test_acc may not be this epoch's test_acc
                current_result_file.write(f'{current_lr}\n')
                current_result_file.write(f'{best_epoch}\n')
                current_result_file.write(f'{best_acc}\n')
                current_result_file.write(f'{best_epoch_balanced_test_acc}\n')
                current_result_file.close()

        f = open(f'{save_results_folder}/20240505_NorCanNecExp13_0_val01_{my_modelname}.txt', 'a+')
        f.write('best epoch: {} best balanced validation accuracy:{} balanced test acc at best val: {}'.format(best_epoch,best_acc,best_epoch_balanced_test_acc))
        f.close()
            
    #############this block is only for reloading model and testing, comment training above and testing at bottom#########################################
    if program_mode =='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
        path2test_weights=save_models_folder+"/20240505_NorCanNecExp13_0_current_best.pt"
        my_model.load_state_dict(torch.load(path2test_weights))
        device = torch.device("cuda:0")
        my_model.to(device) 
        val_target_list, val_score_list, val_pred_list, raw_unbalanced_val_acc, val_loss = val(my_model)
        TP=[0]*NUM_CLASSES
        TN=[0]*NUM_CLASSES
        FN=[0]*NUM_CLASSES
        FP=[0]*NUM_CLASSES
        p=[0.0]*NUM_CLASSES
        r=[0.0]*NUM_CLASSES
        F1=[0.0]*NUM_CLASSES
        TPR=[0.0]*NUM_CLASSES
        TNR=[0.0]*NUM_CLASSES
        balanced_val_acc=0.0
        for class_index in range(NUM_CLASSES):
            TP[class_index]=((val_pred_list == class_index) & (val_target_list == class_index)).sum()
            TN[class_index] = ((val_pred_list != class_index) & (val_target_list != class_index)).sum()
            FN[class_index] = ((val_pred_list != class_index) & (val_target_list == class_index)).sum()
            FP[class_index] = ((val_pred_list == class_index) & (val_target_list != class_index)).sum()
            if TP[class_index]+FP[class_index]==0:
                p[class_index]=1.0
            else:
                p[class_index]=float(TP[class_index])/float(TP[class_index]+FP[class_index])
            if TP[class_index]+FN[class_index]==0:
                r[class_index]=1.0
            else:
                r[class_index]=float(TP[class_index]) / float(TP[class_index] + FN[class_index])
            if r[class_index]+p[class_index]==0.0:
                F1[class_index]=0.0
            else:
                F1[class_index] = 2 * r[class_index] * p[class_index] / (r[class_index] + p[class_index])
            TPR[class_index]=r[class_index]
            TNR[class_index]=float(TN[class_index])/float(TN[class_index]+FP[class_index])
            balanced_val_acc+=float(TP[class_index])/float(TP[class_index]+FN[class_index])#It is avg r
        balanced_val_acc/=NUM_CLASSES
        print('On validation: ')
        print('TP=')
        print(TP)
        print('TN=')
        print(TN)
        print('FN=')
        print(FN)
        print('FP=')
        print(FP)
        print('precision')
        print(p)
        print('recall')
        print(r)
        print('F1')
        print(F1)
        print('balanced_val_acc')
        print(balanced_val_acc)
        print('\n')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f = open(f'{save_results_folder}/20240505_NorCanNecExp13_0_val01_load_model_{my_modelname}.txt', 'a+')
        f.write('Val: balanced accuracy: {:.4f}, average loss: {},TP= [{}], TN= [{}], FN= [{}], FP= [{}], p= [{}], r= [{}], F1= [{}], TPR= [{}], TNR= [{}], \
            Current Time ={}\n\n'.format(balanced_val_acc,
            val_loss,", ".join(str(item) for item in TP),", ".join(str(item) for item in TN),", ".join(str(item) for item in FN),", ".join(str(item) for item in FP),
            ", ".join(str(item) for item in p),", ".join(str(item) for item in r),", ".join(str(item) for item in F1),", ".join(str(item) for item in TPR),", ".join(str(item) for item in TNR),current_time))
        f.close()

        test_target_list, test_score_list, test_pred_list, raw_unbalanced_test_acc, test_loss = test(my_model,0)

        TP=[0]*NUM_CLASSES
        TN=[0]*NUM_CLASSES
        FN=[0]*NUM_CLASSES
        FP=[0]*NUM_CLASSES
        p=[0.0]*NUM_CLASSES
        r=[0.0]*NUM_CLASSES
        F1=[0.0]*NUM_CLASSES
        TPR=[0.0]*NUM_CLASSES
        TNR=[0.0]*NUM_CLASSES
        balanced_test_acc=0.0
        for class_index in range(NUM_CLASSES):
            TP[class_index]=((test_pred_list == class_index) & (test_target_list == class_index)).sum()
            TN[class_index] = ((test_pred_list != class_index) & (test_target_list != class_index)).sum()
            FN[class_index] = ((test_pred_list != class_index) & (test_target_list == class_index)).sum()
            FP[class_index] = ((test_pred_list == class_index) & (test_target_list != class_index)).sum()
            if TP[class_index]+FP[class_index]==0:
                p[class_index]=1.0
            else:
                p[class_index]=float(TP[class_index])/float(TP[class_index]+FP[class_index])
            if TP[class_index]+FN[class_index]==0:
                r[class_index]=1.0
            else:
                r[class_index]=float(TP[class_index]) / float(TP[class_index] + FN[class_index])
            if r[class_index]+p[class_index]==0.0:
                F1[class_index]=0.0
            else:
                F1[class_index] = 2 * r[class_index] * p[class_index] / (r[class_index] + p[class_index])
            TPR[class_index]=r[class_index]
            TNR[class_index]=float(TN[class_index])/float(TN[class_index]+FP[class_index])
            balanced_test_acc+=float(TP[class_index])/float(TP[class_index]+FN[class_index])#It is avg r
        balanced_test_acc/=NUM_CLASSES
        print('On testing: ')
        print('TP=')
        print(TP)
        print('TN=')
        print(TN)
        print('FN=')
        print(FN)
        print('FP=')
        print(FP)
        print('precision')
        print(p)
        print('recall')
        print(r)
        print('F1')
        print(F1)
        print('balanced_test_acc')
        print(balanced_test_acc)
        print('\n')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f = open(f'{save_results_folder}/20240505_NorCanNecExp13_0_test01_load_model_{my_modelname}.txt', 'a+')
        f.write('Test: balanced accuracy: {:.4f}, average loss: {},TP= [{}], TN= [{}], FN= [{}], FP= [{}], precision=[{}], recall=[{}], F1= [{}], TPR= [{}], TNR= [{}], \
            Current Time ={}\n\n'.format(balanced_test_acc,
            test_loss,", ".join(str(item) for item in TP),", ".join(str(item) for item in TN),", ".join(str(item) for item in FN),", ".join(str(item) for item in FP),
            ", ".join(str(item) for item in p),", ".join(str(item) for item in r),", ".join(str(item) for item in F1),", ".join(str(item) for item in TPR),", ".join(str(item) for item in TNR),current_time))
        f.close()
    if get_best_model==1:
        torch.save(my_best_model.state_dict(), save_models_folder+"/20240505_NorCanNecExp13_0_val01_{}_{}_balanced_val_acc{}_balanced_test_acc{}.pt".format(my_modelname,best_epoch,best_acc,best_epoch_balanced_test_acc))
