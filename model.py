import math
import torch
import torch.nn as nn


# NOTE: simple 3D encoding. All method use 3 directions.
class encoding_func_3D:
    def __init__(self, name, param=None, device='cpu', dim_x=3):
        self.name = name

        if name == 'none': self.dim=2
        elif name == 'basic': self.dim=4
        else:
            self.dim = param[1]
            if name == 'RFF':
                self.b = param[0]*torch.randn(1,dim_x,int(param[1]/2), device=device)   # make it to have batch_size=1
            else:
                print('Undifined encoding!')
                
    def __call__(self, x):
        if self.name == 'none':
            return x
        elif self.name == 'basic':
            emb = torch.cat((torch.sin((2.*math.pi*x)),torch.cos((2.*math.pi*x))),-1)
            emb = emb/(emb.norm(dim=1).max())
        elif (self.name == 'RFF')|(self.name == 'rffb'):
            emb = torch.cat((torch.sin((2.*math.pi*x).bmm(self.b)),torch.cos((2.*math.pi*x).bmm(self.b))),-1)   # batch_size=1
        return emb

    
class Neural_Prior(nn.Module):
    def __init__(self, input_size=1000, dim_x=3, filter_size=128, act_fn='relu', layer_size=8, output_feat=False):
        super().__init__()
        self.input_size = input_size
        self.layer_size = layer_size
        self.output_feat = output_feat
        
        self.nn_layers = nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(nn.Sequential(nn.Linear(dim_x, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(nn.Sigmoid())
            for _ in range(layer_size-1):
                self.nn_layers.append(nn.Sequential(nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(nn.Sigmoid())
            self.nn_layers.append(nn.Linear(filter_size, dim_x))
        else:
            self.nn_layers.append(nn.Sequential(nn.Linear(dim_x, dim_x)))

    def forward(self, x):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        if self.output_feat:
            feat = []
        for layer in self.nn_layers:
            x = layer(x)
            if self.output_feat and layer == nn.Linear:
                feat.append(x)

        if self.output_feat:
            return x, feat
        else:
            return x
      