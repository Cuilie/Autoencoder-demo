import os
import torch
import torchvision
from torch import nn
import torch.utils.data as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

# if not os.path.exists('./mlp_img'):
#     os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

pretrain = False
num_epochs = 10
batch_size = 512
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


tensor_x = torch.from_numpy(data_X) # transform to torch tensors
tensor_y =torch.from_numpy(data_Y)

my_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
train_dataloader = utils.DataLoader(my_dataset,batch_size = batch_size,shuffle = True) # create your dataloader
test_dataloader = utils.DataLoader(my_dataset,batch_size = 20000,shuffle = True) # create your dataloader


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 8),
            nn.ReLU(True),
            nn.Linear(8, 1))
        self.decoder = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(True),
            nn.Linear(8, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 2))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



model = autoencoder()

if pretrain == True:
    model = torch.load('model/model1.pt')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    # ===================test&plot========================
    if epoch % 1 == 0:
        for data in test_dataloader:

            testimg, _ = data
            testimg = testimg.view(testimg.size(0), -1)
            testimg = Variable(testimg).float()
            testoutput = model(testimg.float())
            loss = criterion(testoutput, testimg)
            print('Error in all data:epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data.item()))

            approximation = testoutput.detach().numpy()

            fig =plt.figure()
            plt.scatter(approximation[:,0],approximation[:,1],c = mat['Y_data'][0])
            fig.savefig('images/' + str(epoch)+'.png')

            break

    for data in train_dataloader:

        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).float()
        # ===================forward=====================
        output = model(img.float())
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # ===================log========================
    # print('epoch [{}/{}], loss:{:.4f}'
    #      .format(epoch + 1, num_epochs, loss.data.item()))

    # ===================save========================
    if epoch % 20 == 0:
        torch.save(model, 'model/model1.pt')

torch.save(model.state_dict(), './sim_autoencoder.pth')


os.chdir('images')
images = []
filenames=sorted((fn for fn in os.listdir('.') if fn.endswith('.png')),key=lambda x: int(re.sub('[a-z.]', '', x)))
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('gif.gif', images,duration=0.05)

os.chdir(os.path.dirname(os.getcwd()))
