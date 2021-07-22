""" This code is written by Abduallah Damash 2281772
   at 21/06/2021 for CNG483 course to present project 3
  "Identity recognition based on iris biometric data"
Middle East Technical University, All Right Saved"""
import os
import time

from matplotlib import pyplot

timestr = time.strftime("%Y_%m_%d_%H_%M")
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from numpy import mean
############## TENSORBOARD ########################
import torch.nn.functional as F
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs"
from tqdm import tqdm
writer = SummaryWriter('runs/mnist1')
###################################################
import matplotlib.pyplot as plt
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 7)         #numer of kernal is 7, stride is 1 , padding is 0
        self.pool = nn.MaxPool2d(2, 2)          #number of stride is 2
        self.conv2 = nn.Conv2d(8, 4, 3)         # OUTPUT Channles are 4 with kernal of 3, stride is 1 , padding is 0
        self.fc1 = nn.Linear((30-1)**2*4, 2048) #Applying the formula with Batch size of 4 and ALL phptos ARE 128*128
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 801) # OUTPUT is 400 subjects
        ''' The Formula for calculating the  the Output Tensor (Image) of a Conv Layer
        (((W - K + 2P) / S) + 1)
        WHERE, 
        W = Input size
        K = Size (width) of kernels used in the Conv Layer (Filter size)
        S = Stride
        P = Padding
        '''
        ''' The Formula for calculating the Output Tensor (Image) of a MaxPool Layer
             (((W - Ps) / S) + 1)
             WHERE, 
             W = Input size
             S = Stride
             Ps = Pool size
        '''
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 8, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 4, 2, 4
        #print(x.shape)
        x = x.view(-1, (30-1)**2*4)          # -> n, 3364
        x = F.relu(self.fc1(x))               # -> n, 2048
        x = F.relu(self.fc2(x))               # -> n, 1024
        x = self.fc3(x)                       # -> n, 800
        output = F.log_softmax(x.float(), dim=1)  # optimizing the loss using log_softmax
        return output

class Dataloder(Dataset):
  def __init__(self,features_,labels_):
    self.features_ = torch.from_numpy(features_)
    self.labels_ = torch.from_numpy(labels_)
    self.n_samples = features_.shape[0]
    print('LOADED:',self.n_samples)

  def __len__(self):
    return self.n_samples

  def __getitem__(self, index):
    return self.features_[index], self.labels_[index]

def trainModeal (TrainArray, TrainLabal, epochs,batch):
  print('Start Training & Validation....')
  # Train and valid the model
  model = ConvNet()
  # Model Parameter for Training the Nural Network #
  opti = optim.RMSprop(model.parameters(), lr=0.0001)
  criteria = nn.CrossEntropyLoss()
  # Hyper Parameter #
  num_epochs = epochs
  batch_size = batch
  # Setting the Dataset and loaded to be ready for Training
  train_Tex_loader = Dataloder(TrainArray, TrainLabal)
  # Resizing the Dataset
  train_loader = DataLoader(dataset=train_Tex_loader,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
  # Setting The parameter for training each Epoch
  total_samples = len(train_loader)
  n_iterations = math.ceil(total_samples / batch_size)
  print(f'Total Samples Data {total_samples},Epcoh {num_epochs}, Batch Size {batch_size}, Number of Iteration {n_iterations}')
  LOSS_HISTORY = []
  ACC_HISTORY = []
  for epoch in range(num_epochs):
    COUNTER = 0
    correct = 0
    total_loss = 0.0
    for i, (features, labels) in enumerate(train_loader):
      train_input, train_label = Variable(features), Variable(labels)
      #print(train_input.shape)
      train_output = model(train_input.float())
      loss = criteria(train_output.float(), train_label.long())

      opti.zero_grad()
      loss.backward()
      opti.step()
      total_loss += loss.item()
      _, predicted = torch.max(train_output.data, 1)

      correct += (predicted == train_label).sum().item()
      COUNTER += batch_size

    total_loss = total_loss / COUNTER
    correct = correct / COUNTER
    print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {total_loss}, Acc: {correct}')

    LOSS_HISTORY.append(total_loss)
    ACC_HISTORY.append(correct)
  # Plot the accercy and loss and save them
  plt.plot(range(epochs), LOSS_HISTORY)
  plt.savefig(f'loss_Train_{timestr}.png')
  plt.figure()
  plt.plot(range(epochs), ACC_HISTORY)
  plt.savefig(f'acc_Train_{timestr}.png')
  print('Finished Training & Validation....')
  PATH = './cnnTrain.pth'
  torch.save(model.state_dict(), PATH)
  return 1

def testModeal (TestArray,TestLabal, epochs,batch):
  print('Start Testing....')
  # Test the model
  model = ConvNet()
  # Model Parameter for Testing the Nural Network #
  opti = optim.RMSprop(model.parameters(), lr=0.0001)
  criteria = nn.CrossEntropyLoss()
  # Hyper Parameter #
  num_epochs = epochs
  batch_size = batch
  # Setting the Dataset and loaded to be ready for Testing
  test_loader = Dataloder(TestArray, TestLabal)
  # Resizing the Dataset
  Test_loader = DataLoader(dataset=test_loader,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=0)
  # Setting The parameter for Testing each Epoch
  total_samples = len(Test_loader)
  n_iterations = math.ceil(total_samples / batch_size)
  print(f'Total Samples Data {total_samples},Epcoh {num_epochs}, '
        f'Batch Size {batch_size}, Number of Iteration {n_iterations}')
  LOSS_HISTORY = []
  ACC_HISTORY = []
  for epoch in range(num_epochs):
      COUNTER = 0
      correct = 0
      total_loss = 0.0
      for i, (features, labels) in enumerate(Test_loader):
          test_input, test_label = Variable(features), Variable(labels)
          # print(train_input.shape)
          train_output = model(test_input.float())
          loss = criteria(train_output.float(), test_label.long())

          opti.zero_grad()
          loss.backward()
          opti.step()
          total_loss += loss.item()
          _, predicted = torch.max(train_output.data, 1)

          correct += (predicted == test_label).sum().item()
          COUNTER += batch_size

      total_loss = total_loss / COUNTER
      correct = correct / COUNTER
      #print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {total_loss}, Acc: {correct}')
      LOSS_HISTORY.append(total_loss)
      ACC_HISTORY.append(correct)
  # Plot the accercy and loss and save them
  acc = 100* mean(ACC_HISTORY)
  los = 100* mean(LOSS_HISTORY)
  print(f'Accuracy of the network with 400 subjects: {acc} %'
        f'Loss of the network with 400 subjects: {los} %')
  plt.plot(range(epochs), LOSS_HISTORY)
  plt.savefig(f'loss_Test_{timestr}.png')
  plt.figure()
  plt.plot(range(epochs), ACC_HISTORY)
  plt.savefig(f'acc_Test_{timestr}.png')
  print('Finished Testing....')
  PATH = './cnnTest.pth'
  torch.save(model.state_dict(), PATH)
  return 1

def PreProcessData (Folder, ParmaterFile):
    print("Pre Processing Image starting now.... ")
    Data = os.listdir(Folder)
    paramter = ParmaterFile
    # Read The Parameter.tex to process the photos
    PhotoNames = pd.read_csv(paramter, sep=',',
                             usecols=['Name of the iris sample', ' iris x-coordinate', ' iris y-coordinate',
                                      ' iris radius'],
                             index_col=0, skiprows=1)
    # Sort The list according to how windows sort images in Database
    PhotoNames = PhotoNames.sort_values(by=["Name of the iris sample"], ascending=True)
    # Assign Each Coordinate
    xcoor = PhotoNames[' iris x-coordinate']
    ycoor = PhotoNames[' iris y-coordinate']
    radius = PhotoNames[' iris radius']
    # Paramter to obtine the train, valid and test sets
    # since we have tow subjects for each eighth photos, the {counter} will count for that
    # {Person} Counter to obtain the label for each person
    # {numindex} for going through the photos to crop it
    counter = int (0)
    Person = int(1)
    numindex= int(0)
    TrainSet=[]
    ValidSet=[]
    TestSet = []
    TrainLabel=[]
    ValidLabel=[]
    TestLabel = []
    #there are 3 problems - i solved one where is the next one u said
    for img in tqdm(Data):
        if (img != 'Parameters.txt' and img == PhotoNames.index[numindex]):
            # Rest the Counter After 8 rounds, and move to the next two subjects
            if (counter == 8):
                counter = int(0)
                Person = Person +2
            newimage = cv2.imread(f'{Folder}/{img}')
            ####
            # Show the original image before modifying
            # print(img,PhotoNames.index[numindex])
             #print(newimage.dtype)
            pyplot.imshow(newimage)
            pyplot.show()
            ####

            # Deduct the IRIS place to crop it
            StartHight = ycoor[numindex] - radius[numindex]
            EndHight = ycoor[numindex] + radius[numindex]
            StartWidth = xcoor[numindex] - radius[numindex]
            Endwidth = xcoor[numindex] + radius[numindex]
            # Crop the image, and resize to have 128*128 pixels
            Cropimage = newimage[StartWidth:Endwidth, StartHight:EndHight]
            Cropimage1 = cv2.resize(Cropimage, (128, 128))
            Cropimage1 = Cropimage1.transpose((2,0,1))
            Writeimage = cv2.resize(Cropimage, (640, 480))
            # vis = np.concatenate((newimage, Writeimage), axis=1)
            # cv2.imwrite(f'combined{timestr}.png', vis)
            ####
            # Show the Cropped Image
            pyplot.imshow(Cropimage)
            pyplot.show()
            # print(Cropimage1.shape, Cropimage1)
            #####"
            ########################
            #  BEGIN of Classify the photos for each set
            Cropimage1Array = np.array(Cropimage1)
            if (counter == 0 or counter==1):
                TrainSet.append(Cropimage1Array)
                TrainLabel.append(Person) # f"person{Person}"
            elif(counter == 2 or counter==3):
                TrainSet.append(Cropimage1Array)
                TrainLabel.append(Person+1)
            elif (counter == 4 ):
                ValidSet.append(Cropimage1Array)
                ValidLabel.append(Person)
            elif (counter == 5 ):
                TestSet.append(Cropimage1Array)
                TestLabel.append(Person)
            elif (counter == 6 ):
                ValidSet.append(Cropimage1Array)
                ValidLabel.append(Person+1)
            elif (counter == 7 ):
                TestSet.append(Cropimage1Array)
                TestLabel.append(Person+1)
            #  END of  Classification
            ########################
            counter = counter+1
            numindex = numindex + 1
        else:
            print(img, PhotoNames.index[counter]) # To see Which images are not matching
    # Transfer All list inro numpay arrays
    TrainSetA = np.array(TrainSet)
    ValidSetA = np.array(ValidSet)
    TestSetA = np.array(TestSet)
    TrainLabelA = np.array(TrainLabel)
    ValidLabelA = np.array(ValidLabel)
    TestLabelA = np.array(TestLabel)
    # Print the shape of prossing date which expexted to be 1600*128*128 subjects
    print(f"Train Set Shape{TrainSetA.shape} Valid Set Shape{ValidSetA.shape} Test Set Shape{TestSetA.shape}")
    print(f"Train Label Shape{TrainLabelA.shape} Valid Label Shape{ValidLabelA.shape} Test Label Shape{TestLabelA.shape}")
    print("END of Pre Processing Image starting now.... ")
    #print(TrainLabelA, ValidLabelA, TestLabelA)
    return TrainSetA,ValidSetA,TestSetA, TrainLabelA,ValidLabelA,TestLabelA

print('Wlecome Dear to "Identity recognition based on iris biometric data" Program :)')
# Obtine the Dataset and Pre procsses it by cropping the IRIS and resize it to be 128*128
TrainSet,ValidSet,TestSet, TrainLabel,ValidLabel,TestLabel = PreProcessData(
    Folder= r'Database',
    ParmaterFile= r'Database/Parameters.txt')
#Train and then valid the data set with the convelation network
trainModeal(TrainSet,TrainLabel, epochs=10, batch=4)
trainModeal(ValidSet,ValidLabel, epochs=10, batch=4)
#Test the data set with the convelation network
testModeal(TestSet,TestLabel,epochs=10,batch=4)

print('Hope you enjoy ir, Goodbye :( ')
