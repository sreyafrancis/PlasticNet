import torch
import torch.nn as nn
import torch.nn.functional as F

from PlasticBlock import PlasticBlock
from shootCloud import shootCloud


#  https://github.com/tensorflow/models/tree/master/research/transformer
class SpatialTransformer(nn.Module):
    def __init__(self, k=64):
        super(SpatialTransformer, self).__init__()
        self._cuda = 'cuda' # torch.device('cuda')
        self.k = k

        self.Conv1 = torch.nn.Conv1d(self.k, self.k, 1)
        self.BN1 = nn.BatchNorm1d(self.k)

        self.Conv2 = torch.nn.Conv1d(self.k, self.k * 2, 1)
        self.BN2 = nn.BatchNorm1d(self.k * 2)

        self.Conv16 = torch.nn.Conv1d(self.k * 2, self.k * 16, 1)
        self.BN16 = nn.BatchNorm1d(self.k * 16)

        self.Mlp16 = nn.Linear(self.k * 16, self.k * 8)
        self.BN8 = nn.BatchNorm1d(self.k * 8)

        self.Mlp8 = nn.Linear(self.k * 8, self.k * 4)
        self.BN4 = nn.BatchNorm1d(self.k * 4)

        self.Mlp4 = nn.Linear(self.k * 4, self.k * self.k)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.BN1(self.Conv1(x)))
        x = F.relu(self.BN2(self.Conv2(x)))
        x = F.relu(self.BN16(self.Conv16(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.k*16)

        x = F.relu(self.BN8(self.Mlp16(x)))
        x = F.relu(self.BN4(self.Mlp8(x)))
        x = self.Mlp4(x)

        eye = torch.eye(self.k, requires_grad=True, device=self._cuda).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + eye
        x = x.view(-1, self.k, self.k)
        return x


class Encoder(nn.Module):

    def __init__(self, m=52, k=64):
        super(Encoder, self).__init__()

        # number of output parameters
        # base depth nb of feature maps
        self.k = k

        """
        conv 1d : 64
        conv 1d : 128
        conv 1d : 1024
        maxpooling
        Mlp 1024
        batch norm (instead of dropout)
        Mlp 256
        Mlp 52
               output x : size 52
        """

        self.Conv1 = torch.nn.Conv1d((16+1)*self.k, self.k*8, 1)
        self.BN8 = nn.BatchNorm1d(self.k*8)

        self.Conv2 = torch.nn.Conv1d(self.k*8, self.k*4, 1)
        self.BN4 = nn.BatchNorm1d(self.k*4)

        self.Conv3 = torch.nn.Conv1d(self.k*4, self.k*2, 1)
        self.BN2 = nn.BatchNorm1d(self.k*2)

        self.Mlp_out = nn.Linear(self.k*2, m)

    def forward(self, features, x):
        x = torch.cat([features, x], 1)
        x = F.relu(self.BN8(self.Conv1(x)))
        x = F.relu(self.BN4(self.Conv2(x)))
        x = F.relu(self.BN2(self.Conv3(x)))
        x = self.Mlp_out(x.transpose(2, 1).contiguous())
        x = torch.max(x, 1, keepdim=True)[0]
        params = torch.split(torch.squeeze(x), [40, 9, 3], dim=1)
        shape = params[0]
        affine = params[1]
        rotation = params[2]
        shape = torch.sigmoid(0.1*shape)
        affine = F.relu(affine)
        rotation = torch.sigmoid(0.05 * rotation)

        x = torch.cat([shape, affine, rotation], 1)
           # x.view(batch_size, self.m, 1)

        return x

def e_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Norm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Implementation of PointNet
# https://arxiv.org/abs/1612.00593
class PlasticNet(nn.Module):

    def __init__(self, n=256, k=64):
        super(PlasticNet, self).__init__()

        self.n = n
        self.k = k
        self.Encoder = Encoder()
        self.Transformer = SpatialTransformer(self.k)

        self.Encoder.apply(e_init)

        self.batch_size = 1

        """
        conv 1d : 64
        batch norm (instead of dropout)
        transformation
        matrix multiplication
        conv 1d : 128
        conv 1d : 1024
        maxpooling
        Encoder
            output x : size 52
        """

        self.Conv1 = torch.nn.Conv1d(3, self.k, 1)
        self.Conv2 = torch.nn.Conv1d(self.k, self.k * 2, 1)
        self.Conv16 = torch.nn.Conv1d(self.k * 2, self.k * 16, 1)
        self.BN1 = nn.BatchNorm1d(self.k)
        self.BN2 = nn.BatchNorm1d(self.k * 2)
        self.BN16 = nn.BatchNorm1d(self.k * 16)

    def forward(self, x):
        # Mlp 1
        x = F.relu(self.BN1(self.Conv1(x)))
        features = x
        res = self.Transformer(x)
        # T @ F matmul
        x = torch.bmm(x.transpose(2, 1), res).transpose(2, 1)

        # Mlp 2
        x = F.relu(self.BN2(self.Conv2(x)))

        # Pooling
        x = self.BN16(self.Conv16(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.k*16, 1).repeat(1, 1, self.n)

        # Encoder
        x = self.Encoder(features, x)

        return x

    def loss(self, seed, params):
        return ((seed - params)**2).mean()


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.DoubleTensor')
    test_batch = 32
    block = PlasticBlock()
    Vb, _, Fb = block.get_obj()
    cloud = torch.from_numpy(shootCloud(Vb, Fb).reshape((1, 3, 256)).repeat(test_batch, axis=0)).cuda().double()

    net = PlasticNet().to('cuda:0')
    predict = net(cloud)
    print('PlasticNet : ', predict.size())
