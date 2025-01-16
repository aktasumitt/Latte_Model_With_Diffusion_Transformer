import torch.nn as nn

# Encoder
class Encoder(nn.Module):
    def __init__(self, channel_size,hidden_dim=256):
        super(Encoder,self).__init__()
        
        self.hidden_dim=hidden_dim
        
        self.conv_block=nn.Sequential(nn.Conv2d(channel_size,hidden_dim,kernel_size=4,stride=2,padding=1,padding_mode="reflect"),
                                      nn.ReLU(),
                                      nn.Conv2d(hidden_dim,hidden_dim,kernel_size=4,stride=2,padding=1,padding_mode="reflect")
                                      )
        
        self.residual_block1=self.residual()
        self.residual_block2=self.residual()
    
    def residual(self):
        residual=nn.Sequential(nn.ReLU(),
                               nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=1,padding_mode="reflect"),
                               nn.ReLU(),
                               nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=1))
        
        return residual

    def forward(self,x):
        B,T,C,H,W=x.shape
        x=x.reshape(B*T,C,H,W)
        
        x=self.conv_block(x)
        r1=self.residual_block1(x)+x
        out=self.residual_block2(r1)+r1
        
        (_,hid,h1,w1)=out.shape

        return out.reshape(B,T,hid,h1,w1)
    
 
 

 
# DECODER
class Decoder(nn.Module):
    def __init__(self, channel_size,hidden_dim=256):
        super(Decoder,self).__init__()
        self.hidden_dim=hidden_dim
        
        self.residual_block1=self.residual()
        self.residual_block2=self.residual()
        
        self.convtranspose_block=nn.Sequential(nn.ConvTranspose2d(hidden_dim,hidden_dim,kernel_size=4,stride=2,padding=1),
                                               nn.ReLU(),
                                               nn.ConvTranspose2d(hidden_dim,channel_size,kernel_size=4,stride=2,padding=1)
                                               )
    
    def residual(self):
        residual=nn.Sequential(nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=1,padding_mode="reflect"),
                               nn.ReLU(),
                               nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=1))
        
        return residual

    def forward(self,x):
        B,T,C,H,W=x.shape
        x=x.reshape(B*T,C,H,W)
        
        r1=nn.functional.relu(self.residual_block1(x)+x)
        r2=nn.functional.relu(self.residual_block2(r1)+r1)
        out=self.convtranspose_block(r2)
        
        (_,hid,h1,w1)=out.shape
        return out.reshape(B,T,hid,h1,w1)


    