import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader , TensorDataset
from paths import *
from helpers import *

class ResBlock(nn.Module):
    def __init__(self,in_chan,out_chan,proj = False,pool = 0, bottleneck = False, input_height = 0, input_width = 0):

        super(ResBlock,self).__init__()

        self.proj = None
        self.pool = None

        if proj:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chan,out_chan,(1,1)),
                nn.BatchNorm2d(out_chan)
            )

        if pool == 1:
            self.pool = nn.Sequential(
                nn.Conv2d(in_chan,in_chan,(input_height - (input_height // 2) + 1, input_width - (input_width // 2) + 1)),
                nn.BatchNorm2d(in_chan)
            )
        elif pool == 2:
            self.pool = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride= 2)
            )

        if not bottleneck:
            self.block =  nn.Sequential(
                nn.Conv2d(in_chan,out_chan,(3,3),padding= 1),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chan,out_chan,(3,3),padding= 1),
                nn.BatchNorm2d(out_chan)
            )
        else:
            self.block =  nn.Sequential(
                nn.Conv2d(in_chan,out_chan,(1,1)),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chan,out_chan,(3,3),padding= 1),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_chan,out_chan,(1,1)),
                nn.BatchNorm2d(out_chan),
            )
        
    def forward(self,x):
        id = x
        if self.pool is not None:
            id = self.pool(x)
        out = self.block(id)
        
        if self.proj is not None:
            id = self.proj(id)

        out += id
        out = nn.functional.relu(out)
        return out


faces_mom = np.array([cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in np.load(FACES_PATH + "mom.npy")])
w, h = faces_mom[0].shape

faces_dexter = np.array([cv.cvtColor(cv.resize(img,(h, w)), cv.COLOR_BGR2GRAY) for img in np.load(FACES_PATH + "dexter.npy")])
faces_deedee = np.array([cv.cvtColor(cv.resize(img,(h, w)), cv.COLOR_BGR2GRAY) for img in np.load(FACES_PATH + "deedee.npy")])
faces_dad = np.array([cv.cvtColor(cv.resize(img,(h, w)), cv.COLOR_BGR2GRAY) for img in np.load(FACES_PATH + "dad.npy")])
faces_unknown = np.array([cv.cvtColor(cv.resize(img,(h, w)), cv.COLOR_BGR2GRAY) for img in np.load(FACES_PATH + "unknown.npy")])
background = np.array([cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in np.load(BACKGROUND_PATH + "mom_background.npy")])
faces_flip = np.flip(faces_mom, axis= 2)

X = np.concatenate((faces_dexter, faces_flip, faces_dad, faces_mom, faces_deedee, faces_unknown, background))
y = np.concatenate((np.array([0 for _ in range(len(faces_dexter))]), np.array([1 for _ in range(len(faces_flip))]), np.array([0 for _ in range(len(faces_dad))]), np.array([1 for _ in range(len(faces_mom))]), np.array([0 for _ in range(len(faces_deedee))]), np.array([0 for _ in range(len(faces_unknown))]),np.array([0 for _ in range(len(background))])))

x_train = torch.stack([transforms.ToTensor()(x) for x in X])

train_dataset = TensorDataset(torch.tensor(x_train, dtype= torch.float32),torch.tensor(y, dtype= torch.float32))
train_data_loader = DataLoader(train_dataset,batch_size= 64 , shuffle= True)

lr = 0.001
"""
model = nn.Sequential(ResBlock(3, 32,True, bottleneck= False),
                        ResBlock(32,32,bottleneck= False),
                        #ResBlock(32,32,bottleneck= False),
                        #ResBlock(32,32,bottleneck= False),
                        ResBlock(32,64,True, 2, False),
                        ResBlock(64,64, bottleneck= False),
                        #ResBlock(64,64, bottleneck= False),
                        #ResBlock(64,64, bottleneck= False),
                        #ResBlock(64,128,True, 2, False, 55, 69),
                        #ResBlock(128,128, bottleneck= False),
                        #ResBlock(128,128, bottleneck= False),
                        #ResBlock(128,128, bottleneck= False),
                        #ResBlock(128,256,True, 2, False, 27, 34),
                        #ResBlock(256,256, bottleneck= False),
                        #ResBlock(256,256, bottleneck= False),
                        #ResBlock(256,256, bottleneck= False),
                        nn.AdaptiveAvgPool2d((1, 1)), 
                        nn.Flatten(),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )
"""

model = nn.Sequential(
        nn.Conv2d(1, 32, (3, 3)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride= 2),
        nn.Conv2d(32, 64, (3, 3)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride= 2),
        nn.Conv2d(64, 128, (3, 3)),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride= 2),
        nn.Conv2d(128, 256, (3, 3)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride= 2),
        nn.Conv2d(256, 512, (3, 3)),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)), 
        nn.Flatten(),
        nn.Linear(512, 1),
        nn.Sigmoid()
)

loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
device = torch.device("cuda")
model.to(device)

for epoch in tqdm(range(35)):
    model.train()
    for x_batch, y_batch in train_data_loader:
        x_batch, y_batch = x_batch.cuda() , y_batch.cuda()
        y = y_batch.unsqueeze(1)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        x_batch, y_batch = x_batch.cpu() , y_batch.cpu()

    torch.save(model, f"CNNMom5straturiGray{epoch+1}E42kbackground.pth")