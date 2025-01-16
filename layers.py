import torch
import torch.nn as nn

# Multihead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,dk_model,MASK:bool):
        super(MultiHeadAttention,self).__init__()
        
        self.MASK=MASK # MASK variable for masked MHA
        self.dk=dk_model # dk value for MHA
        self.d_model=d_model # d_model is hidden layer size for model
        
        self.q_layer=nn.Linear(d_model,d_model,bias=False) # Query layers
        self.k_layer=nn.Linear(d_model,d_model,bias=False) # key layers 
        self.v_layer=nn.Linear(d_model,d_model,bias=False) # value layers 
        
        # Projection layer for output of MHA
        self.projection_layer=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(0.1)    
    
    def forward(self,query_data,key_data,value_data):
        
        # Query, key and value
        query=self.q_layer(query_data).reshape(query_data.shape[0],query_data.shape[1],-1,self.dk).permute(0,2,1,3)
        key=self.k_layer(key_data).reshape(key_data.shape[0],key_data.shape[1],-1,self.dk).permute(0,2,1,3)
        value=self.v_layer(value_data).reshape(value_data.shape[0],value_data.shape[1],-1,self.dk).permute(0,2,1,3)
        
        # dot product      
        scaled_dot_product=torch.matmul(query,torch.transpose(key,dim0=-2,dim1=-1)) / (self.dk**(1/2)) # Q x (K).T / root(dk)
        
        # masking
        if self.MASK==True:
            mask_tensor=torch.triu(torch.ones_like(scaled_dot_product),diagonal=1)*(-(1e13)) # -1e13 is mask value that too low value like a -(infitive)
            scaled_dot_product=scaled_dot_product+mask_tensor
        
        # Attention_weights
        attention_weights=nn.functional.softmax(scaled_dot_product,-1)
        
        output_att=torch.matmul(attention_weights,value)
        
        # output     
        out_concat=output_att.permute(0,2,1,3).reshape_as(query_data) # Permute >> (B,8,max_len,64) ---> Concatinate >>>(B,max_len,8,64)--->(B,max_len,512)

        return self.dropout(self.projection_layer(out_concat))
    
    
# Feed forward (4x) 
class FeedForward(nn.Module):
    
    def __init__(self,d_model):
        super(FeedForward,self).__init__()
        
        self.FF1=nn.Linear(d_model,d_model*4)
        self.FF2=nn.Linear(d_model*4,d_model)
        self.dropout=nn.Dropout(0.1)
        
    def forward(self,data):
        
        ff1_out=nn.functional.gelu((self.FF1(data)))
        
        return self.dropout(self.FF2(ff1_out))



# AdaLN (Adaptive Layer Norm) for shifting and scaling
class AdaLN(nn.Module):
    def __init__(self,embed_size,d_model):
        super(AdaLN,self).__init__()
        
        self.beta_layer=nn.Linear(embed_size,d_model,bias=False)
        self.gama_layer=nn.Linear(embed_size,d_model,bias=False)
        self.alpha_layer=nn.Linear(embed_size,d_model,bias=False)
    
    def forward(self,x,condition):
        alpha=self.alpha_layer(condition)[:,:,None].permute(0,2,1).repeat(1*int(x.shape[0]/condition.shape[0]),x.shape[1],1)  # for scale subblocks
        
        gama=self.gama_layer(condition)[:,:,None].permute(0,2,1).repeat(1*int(x.shape[0]/condition.shape[0]),x.shape[1],1)  # gama for scaling
        beta=self.beta_layer(condition)[:,:,None].permute(0,2,1).repeat(1*int(x.shape[0]/condition.shape[0]),x.shape[1],1)  # beta for shifting

        scale=gama*x # scaling
        shifting=scale+beta  # shifting
       
        return shifting,alpha

# we add embedding layer to positional encoded time 
class Condition(nn.Module):
    """
    Condition is silu(positional_encoding(t) + embed(label))
    """
    
    def __init__(self,embed_dim,label_size,devices):
        super(Condition,self).__init__()
        self.devices=devices
        self.embed_dim=embed_dim 
        self.embedding=nn.Embedding(label_size,embed_dim) # Embedding for labels

    def Positional_encoding(self, t):
        
        inverse = 1 / 10000 ** (torch.arange(1, self.embed_dim, 2, dtype=torch.float) / self.embed_dim).to(self.devices)        
        
        # Tekrarlama işlemi
        repeated_t = t.unsqueeze(-1).repeat(1, (self.embed_dim // 2)).to(self.devices)  
        
        # pos_A ve pos_B'nin oluşturulması
        pos_A = torch.sin(repeated_t * inverse)
        pos_B = torch.cos(repeated_t * inverse)
        
        return torch.cat([pos_A, pos_B], dim=-1)       
    
    def forward(self,t,label):
        
        if label== None:
            return nn.functional.silu(self.Positional_encoding(t))
            
        else:
            return nn.functional.silu(self.embedding(label)+self.Positional_encoding(t))
    

class Transformer(nn.Module):
    """ 
    - Temporal Transformer is like that become input shape (B*P,F,D) of transformer encoder
    - Spatial Transformer is like that become input shape (Batchsize*Frame,Patch,d_model) of transformer encoder
    
    Args:
        type_of_transformer (str): "temporal" or "spatial"
    """
    def __init__(self,embed_dim,d_model,dk_model,type_of_transformer:str):
        super(Transformer,self).__init__()
        self.type_of_t=type_of_transformer
        self.mlp=FeedForward(d_model=d_model)
        self.mha=MultiHeadAttention(d_model=d_model,dk_model=dk_model,MASK=False)
        self.layer_norm=nn.LayerNorm(d_model)
        self.AdaLN=AdaLN(embed_size=embed_dim,d_model=d_model)
        
        
    def forward(self,x,condition):
        if self.type_of_t=="spatial":
            # x shape is (Batch*Frame,patch,d_model)
            B,F,P,D=x.shape
            x=x.reshape(B*F,P,D)
        
        else: 
            # x shape is (Batch*patch,frame,d_model), ("temporal")
            B,F,P,D=x.shape
            x=x.permute(0,2,1,3).reshape(B*P,F,D)
        
        ln=self.layer_norm(x)
        adaln,alpha=self.AdaLN(ln,condition) # scale and shift
        mha=self.mha(adaln,adaln,adaln)*alpha + x # scale and residual connection
        
        ln=self.layer_norm(mha)
        adaln,alpha=self.AdaLN(ln,condition)
        mlp=self.mlp(adaln)*alpha + mha # scale and residual connection
        
        if self.type_of_t=="spatial":
            return mlp.reshape(B,F,P,D)
        else:
            return mlp.reshape(B,P,F,D).permute(0,2,1,3)
        

# Patch Embedding
class Patch_Embedding(nn.Module):
    def __init__(self,INPUT_DIM,embed_dim,devices):
        super(Patch_Embedding,self).__init__()
        
        self.embed_dim=embed_dim
        self.devices=devices
        
        # embedding for patches
        self.patch_Embedding=nn.Linear(INPUT_DIM,embed_dim)

    def Absolute_Positional_Embedding(self,x,frame_size):
        
        # Positions of text sequences
        position=torch.arange(0,frame_size,1).reshape(frame_size,1).to(self.devices)

        # Even and odd tensors as embedding size
        even_i=torch.arange(0,self.embed_dim,2).to(self.devices)
        odd_i=torch.arange(0,self.embed_dim,2).to(self.devices)

        # Calculate power to use in sinus and cos fuction 
        even_pow=torch.pow(10000,(2*even_i)/self.embed_dim)
        odd_pow=torch.pow(10000,(2*odd_i)/self.embed_dim)
        
        # Sin and cos function to calculate position even and odd row
        PE_sin=torch.sin(position/even_pow)
        PE_cos=torch.cos(position/odd_pow)
        
        # Concat odd and even positions for reached positional encoding tensor
        positional_enc=torch.stack([PE_sin,PE_cos],dim=-1).flatten(start_dim=-2,end_dim=-1)
        positional_enc=positional_enc.unsqueeze(-2)
        print(positional_enc.shape)
        
        return x + positional_enc
        
    def forward(self,x):
        
        x=self.patch_Embedding(x)
        frame_size=x.shape[1]
        out = self.Absolute_Positional_Embedding(x,frame_size=frame_size)
        
        return out
        