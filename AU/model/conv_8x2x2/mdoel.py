#covlutional autoencoder
import threading
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    # x = 0.5 * (x + 1)
    # x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 16, 3, stride=2, padding=1),  # b, 16, 3, 3
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




def train(load,learning_rate):
  best_loss = torch.Tensor([float('inf')]).cuda()
  if load:
    try:
        model = torch.load('./conv_autoencoder.pkl').cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)
        print("\n--------model restored--------\n")
    except Exception as e:
        print(e)
        print("\n--------model not restored--------\n")
        model = autoencoder().cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)

  else:
      model = autoencoder().cuda()
      criterion = nn.MSELoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                weight_decay=1e-5)


  print('training...')
  for epoch in range(num_epochs):
    global flag
    if flag:
      for data in dataloader:
          img, _ = data
          img = Variable(img).cuda()
          # ===================forward=====================
          output = model(img)
          loss = criterion(output, img)
          # ===================backward====================
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      # ===================log========================
      print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, loss.item()))
      if epoch % 10 == 0:
          output[output<0.5]=0.0
          output[output>=0.5]=1.0
          pic = to_img(output.cpu().data)
          plt.imshow(output[0].cpu().data[0],cmap='gray')
          plt.show()
          save_image(pic, './dc_img/image_{}.png'.format(epoch))
      if loss.data < best_loss.data:
        print('save model')
        torch.save(model, './conv_autoencoder.pkl')
        best_loss = loss  
    else: 
      print('break')
      break
  
