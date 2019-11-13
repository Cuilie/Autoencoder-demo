import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary
%matplotlib inline

# Set Hyperparameters

epoch = 10
batch_size =100
learning_rate = 0.0005

# Download Data

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = dset.MNIST("./", train=False,  transform=transforms.ToTensor(), target_transform=None, download=True)


# Set Data Loader(input pipeline)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=batch_size,shuffle=True)
train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,32,3,padding=1),   # batch x 16 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,32,3,padding=1),   # batch x 16 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,64,3,padding=1),  # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64,64,3,padding=1),  # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2)   # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),  # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.Conv2d(128,128,3,padding=1),  # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128,4,3,padding=1),  # batch x 4 x 7 x 7
                        nn.ReLU(),
                        nn.BatchNorm2d(4),
        )
          
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        return out
    


# Decoder 
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
#                          stride=1, padding=0, output_padding=0,
#                          groups=1, bias=True)
# output_height = (height-1)*stride + kernel_size - 2*padding + output_padding
# batch x 512 -> batch x 1 x 28 x 28

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(4,16,3,2,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(16),
                        nn.ConvTranspose2d(16,32,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,64,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.ConvTranspose2d(64,128,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128)
        )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(128,32,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,4,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(4),
                        nn.ConvTranspose2d(4,2,3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(2),
                        nn.ConvTranspose2d(2,1,3,2,1,1),
                        nn.ReLU()
        )
        
    def forward(self,x):
        out = x.view(batch_size,4,7,7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


# train encoder and decoder
# save and load model
load = False
learning_rate = 0.0003
best_loss = torch.Tensor([float('inf')]).cuda()
if load:
  try:
      best_loss, encoder, decoder = torch.load('./deno_autoencoder1.pkl')
      parameters = list(encoder.parameters())+ list(decoder.parameters())
      optimizer = torch.optim.Adam(parameters, lr=learning_rate)
      print("\n--------model restored--------\n")
  except Exception as e:
      print(e)
      print("\n--------model not restored--------\n")
      pass
else:
  print('init new model')
  encoder = Encoder().cuda()
  decoder = Decoder().cuda()
  parameters = list(encoder.parameters())+ list(decoder.parameters())
  loss_func = nn.MSELoss()
  optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)

for i in range(epoch):
    for image,label in train_loader:
        # #guess noise
        # image_n = torch.mul(image+0.25, 0.1 * noise)
        # ## mask noise
        image_n = image.clone()
        #image_n = noising(image_n)
        ###
        image = Variable(image).cuda()
        image_n = Variable(image_n).cuda()
        optimizer.zero_grad()
        output = encoder(image_n)
        output = decoder(output)
        loss = loss_func(output,image)
        loss.backward()
        optimizer.step()
    if loss.data < best_loss.data:
      print('save model')
      torch.save([best_loss,encoder,decoder],'./deno_autoencoder2.pkl')
      best_loss = loss
    print(loss)