# dimensione del batch
BATCH_SIZE = 32
# n° di epoche del training
NUM_OF_EPOCHS = 100

# definisce se usare augmentation nel training set
USE_AUG = True
# definisce se loggare i dati in Tensorboard
log_to_tb = False
# definisce se usare la ResNet18 (true) o LeNet5 (false)
using_res = True
# nome del file .pth che conterrà la rete allenata e l'eventuale cartella di tensorboard
NET_NAME = 'resnet50_aug'

SAVE_PATH = 'trained/' + NET_NAME + '.pth'
SAVE_PATH2 = 'trained/' + NET_NAME + '_best.pth'
RUNS_PATH = 'runs/' + NET_NAME

train_path = 'data/car_brand_logos/Train/'
test_path = 'data/car_brand_logos/Test/'

import torch, torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Resize, ToPILImage, CenterCrop, Normalize, Compose
from torchvision.transforms.functional import to_grayscale, to_tensor, rotate, hflip, affine, adjust_brightness
import matplotlib.pyplot as plt

import os
import random
import pandas as pd
from torchvision.io import read_image

from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from sklearn.metrics import classification_report, confusion_matrix

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import seaborn as sn
import numpy as np

import io
from PIL import Image
from PIL.features import pilinfo

import traceback
import warnings
warnings.filterwarnings("error")

import imgaug.augmenters as iaa

def getLabelList(path):
    only_dirs = [ name for name in os.listdir(path) if 
                 os.path.isdir(os.path.join(path, name)) ]
    only_dirs.sort()
    ret = {}
    index = 0
    
    for d in only_dirs:
        new_path = path + d
        label = only_dirs.index(d)
        for img in [ name for name in os.listdir(new_path) ]:
            ret[index] = [img, label]
            index += 1
    
    return pd.Series(ret)

def getImgPool(path):
    only_dirs = [ name for name in os.listdir(path) if 
                 os.path.isdir(os.path.join(path, name)) ]
    only_dirs.sort()
    ret = {}
    index = 0
    
    for d in only_dirs:
        new_path = path + d
        label = only_dirs.index(d)
        for img in [ name for name in os.listdir(new_path) ]:
            abs_path = new_path + '/' + img
            ret[index] = [abs_path, label, d]
            index += 1
    
    return pd.Series(ret)

# convert a given format image hidden in JPG format, converting it to a JPG and overwriting the original with its format (WEBP or PNG)

def convertPNGImage(path, start_format):
    img = Image.open(path, formats=[start_format])
    new_path = path[:-3] + 'jpg'
    img.convert('RGB').save(new_path)
    image = Image.open(new_path, formats=['JPEG'])
    return image    

if using_res:
    class CustomImageDataset(Dataset):
        def __init__(self, path_labels, transform=None, target_transform=None, use_aug=True):
            self.img_labels = getLabelList(path_labels)
            self.images = getImgPool(path_labels)
            self.transform = transform
            self.target_transform = target_transform
            self.use_aug = use_aug

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            try: 
                image = Image.open(self.images.loc[idx][0])
                if Image.MIME[image.format] == 'image/png': #formato errato
                        image = convertPNGImage(self.images.loc[idx][0], 'PNG')
            except Exception:
                print('Found error at {} in position {}'.format(self.images.loc[idx][0], idx))
                
            image = ToTensor()(image)
            
            if image.shape[0] != 3: #converto le immagini in bianco e nero e RGBA in RGB
                image = ToPILImage()(image).convert('RGB')  
            else: #immagine già RGB
                image = ToPILImage()(image)
                
            preprocess = Compose([
                Resize(300),
                CenterCrop(300),
                ToTensor(),
            ])
            image = preprocess(image)
            
            #parte di data augmentation
            if self.use_aug:
                im_np = np.array((image*255).permute(1,2,0), dtype='uint8')

                # augmentations
                bright = iaa.AddToBrightness((-50,50))
                rotate = iaa.Rotate((-20, 20))
                hflip = iaa.Fliplr(0.5)
                tr_x = iaa.TranslateX(px=(-100, 100))
                tr_y = iaa.TranslateY(px=(-100, 100))
                gauss = iaa.imgcorruptlike.GaussianNoise(severity=1)
                blend = iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToHue((-100, 100)))
                drop_channel = iaa.Dropout2d(p=0.5)
                mult = iaa.Multiply((0.5, 1.5), per_channel=0.5)

                seq = iaa.Sequential([
                    iaa.Sometimes(0.5, bright),
                    iaa.Sometimes(0.8, rotate),
                    iaa.Sometimes(0.5, hflip),
                    iaa.Sometimes(0.5, tr_x),
                    iaa.Sometimes(0.5, tr_y),
                    iaa.Sometimes(0.5, gauss),
                    iaa.Sometimes(0.5, blend),
                    iaa.Sometimes(0.5, drop_channel),
                    iaa.Sometimes(0.5, mult)                    
                ])

                img = seq(image=im_np)
                image = to_tensor(img)
                
                
            
            preprocess2 = Compose([ # secondo preprocess dell'immagine per evitare problemi con il cambio luminosità
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = preprocess2(image)
            
            label = self.img_labels.loc[idx][1]

            return image, label
        
train_data = CustomImageDataset(path_labels=train_path, transform=ToTensor(), target_transform=ToTensor(), use_aug=USE_AUG)
test_data = CustomImageDataset(path_labels=test_path, transform=ToTensor(), target_transform=ToTensor(), use_aug=False)

print (f'N° immagini nel train dataset: {train_data.__len__()}')
print (f'N° immagini nel test dataset: {test_data.__len__()}')

train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_dl = DataLoader(test_data, batch_size=8, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

if using_res:
    net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    net.to(device)
    
in_features = net.fc.in_features
net.fc.out_features = 8
net.to(device)
    
#test struttura rete
params = list(net.parameters())
print(len(params))
print(params[0].size())  # verifico che il primo layer convolutivo sia stato costruito/caricato

# loss function
criterion = nn.CrossEntropyLoss()
# uso stochastic gradient descent
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_OF_EPOCHS)

print(next(net.parameters()).device)

best_accuracy = 0
for epoch in range(NUM_OF_EPOCHS):  # ciclo sul dataset
    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_dl):
        # prelevo immagini e label dal dataloader, poi le adatto
        inputs, labels = data
        batch_len = len(labels)
        inputs = inputs.to(device)
        inputs = inputs.to(torch.float32)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs) # processo immagini, ottenendo predizioni
        loss = criterion(outputs, labels) # calcolo loss
        loss.backward() # backpropagation
        optimizer.step()

        running_loss += loss.item()
        skip = 50
        if i > 0  and i % skip == 0:    # stampa training loss ogni 50 * dimensione batch (50*8 qui)
            print(f'[{epoch + 1}, {i}/{len(train_dl)}] loss: {(running_loss/skip):.3f}')
            # salvataggio della train loss in tensorboard
            if log_to_tb:
                writer.add_scalar('training loss',
                               running_loss / skip,
                               (epoch * len(train_data)) + (i * batch_len))
            running_loss = 0.0
    scheduler.step()

    # fine di un'epoca, calcolo test loss e accuratezza (siccome qui il test dataset è piccolo)
    with torch.no_grad():
        net.eval()
        test_loss = 0
        test_acc = 0
        test_total = 0
        for _, data in enumerate(test_dl):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0) 
            test_acc += (predicted == labels).sum().item() 
            
        actual_test_loss = test_loss/len(test_dl)
        accuracy = (100 * test_acc / test_total)
        
        # print(f"[{epoch + 1}] test loss: {(actual_test_loss):.3f} , test acc: {accuracy}")
        summary = f"[{epoch + 1}] test loss: {(actual_test_loss):.3f} , test acc: {accuracy}\n"
        with open('metrics.txt', 'a') as f:
            f.write(summary)
        print(summary)
        if log_to_tb: # salvataggio test loss e accuratezza in tensorboard
            writer.add_scalar('test loss',
                           actual_test_loss,
                           epoch+1)
            writer.add_scalar('accuracy',
                           accuracy,
                           epoch+1)

        # se l'accuratezza migliora, salvo il modello (migliore rispetto a prima)
        if accuracy > best_accuracy:
           best_accuracy = accuracy
           torch.save(net.state_dict(), SAVE_PATH2)
        

print('Fine Addestramento')
print(f"Miglior acc: {best_accuracy:.5f}")

#salvo il modello prodotto alla 25a epoca
torch.save(net.state_dict(), SAVE_PATH)

# nohup /bin/python3 train.py > log.txt 2>&1 &
# tail -f log.txt