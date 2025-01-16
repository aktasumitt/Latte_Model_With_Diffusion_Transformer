import torch
import torch.nn as nn
from layers import Transformer,Condition,Patch_Embedding


class Diffusion_Transformer(nn.Module):
    """
    d_model: size of hidden layers
    dk_model: part of hidden size on multi head attention
    embed_dim: Embedding dimention
    label_size: label size for target of image dataset
    device: device "cuda" or "cpu"
    channel_size: channel size of image
    N_time: number of stacked temporal and spatial transformer
    """
    def __init__(self,embed_dim,label_size,d_model,dk_model,INPUT_DIM,N_times,device):
        super(Diffusion_Transformer,self).__init__()
        
        self.d_model=d_model
        self.n_times=N_times
        self.INPUT_DIM=INPUT_DIM
        
        # Condition
        self.condition_layer=Condition(embed_dim,label_size,device)
        
        # Patch_Embedding
        self.patch_embedding=Patch_Embedding(INPUT_DIM=INPUT_DIM,embed_dim=embed_dim,devices=device)
        
        # Transformer stacks
        self.temporal_stack=nn.ModuleList([Transformer(embed_dim,d_model,dk_model,"temporal") for _ in range(N_times)])
        self.spatial_stack=nn.ModuleList([Transformer(embed_dim,d_model,dk_model,"spatial") for _ in range(N_times)])

        # output_layer
        self.out_mlp=nn.Sequential(nn.LayerNorm(d_model),
                                   nn.Linear(d_model,INPUT_DIM))
    
    def forward(self,x,t,label):
        
        # condition embedding for t and label
        condition=self.condition_layer(t,label)
        
        # input_layer
        x=self.patch_embedding(x)
        
        # Transformers layers (spatial and temporal)
        for i in range(self.n_times):
            x=self.spatial_stack[i](x,condition)
            x=self.temporal_stack[i](x,condition)
        
        # Reshape and output layer and reshape
        B,F,P,D=x.shape
        x=x.reshape(B*F,P,D)
        output=self.out_mlp(x)
        

        return output.reshape(B,F,P,-1)
    
        