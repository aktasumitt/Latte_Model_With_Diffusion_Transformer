import torch
import tqdm
from einops import rearrange

class Diffusion():
    def __init__(self,beta_start,beta_end,n_timesteps,devices:None):
        
        self.n_timesteps=n_timesteps
        self.devices=devices
        
        # For the formula to aplly noising and denoising process
        self.beta=torch.linspace(beta_start,beta_end,n_timesteps).to(self.devices)
        self.alpha=1-self.beta
        self.alpha_hat=torch.cumprod(self.alpha,dim=0)
        
    def Noising_to_Image(self,x,t):
        
        sqrt_alpha_hat=torch.sqrt(self.alpha_hat[t])[:,None,None,None,None] # Boyutlandırma
        sqrt_one_minus_alpha_hat=torch.sqrt(1-self.alpha_hat[t])[:,None,None,None,None]
        
        noise=torch.randn_like(x)
        
        noisy_img=(sqrt_alpha_hat*x)+(sqrt_one_minus_alpha_hat*noise)
        
        return noisy_img,noise
    
    def Random_Timesteps(self,batch_size):
        return torch.randint(1,self.n_timesteps,(batch_size,)).to(self.devices)
    
    
    def Denoising(self,model,Vae_Decoder,video,labels,PATCH_SIZE): # Test the model with random noisy img
        """
        We always need to patchify before diffusion transformer , and need to depatchify after difussion transfoermer
        """
        
        
        prog_bar=tqdm.tqdm(range(self.n_timesteps),"Prediction Image Step")
        model.eval()
        
        x=torch.randn_like(video).to(self.devices) # x WİTH RANDOM NOİSE
        
        # Patchify
        patched_noisy_latent=rearrange(x,"B F C (H P1) (W P2) -> B F (H W) (C P1 P2)",P1=PATCH_SIZE,P2=PATCH_SIZE)
        
        for i in reversed(range(1,self.n_timesteps)):
            T=(torch.ones(video.shape[0])*i).long().to(self.devices)
    
            predicted_noise=model(patched_noisy_latent,T,labels)
            
            # CFG predicted_noise. This process about, if we train conditional, after we need to predict uncoditional.
            # We use torch lerp to aproach conditional prediction from unconditional smoothly with 3 scale factor
            if labels!=None:
                predicted_noise_unc=model(patched_noisy_latent,T,None)
                predicted_noise=torch.lerp(predicted_noise_unc,predicted_noise,3) 
            
            # De Patchify
            depatched_noise=rearrange(predicted_noise,"B F (H W) (C P1 P2) ->B F C (H P1) (W P2)",P1=PATCH_SIZE,P2=PATCH_SIZE,H=x.shape[-1]//PATCH_SIZE)
            
            beta=self.beta[T][:,None,None,None,None]
            alpha=self.alpha[T][:,None,None,None,None]
            alpha_hat=self.alpha_hat[T][:,None,None,None,None]
            
            noise=(torch.randn_like(x) if i>1 else torch.zeros_like(x)).to(self.devices)
            
            x = (1/alpha_hat) * (x-((1-alpha)/torch.sqrt(1-alpha_hat))*depatched_noise) +(torch.sqrt(beta)*noise)
            prog_bar.update(1)

        prog_bar.close()
        

        model.train()
        
        # Decoder input
        x=Vae_Decoder(x)
        
        x=(x.clamp(-1,1) + 1) / 2
        x=(x*255).type(torch.uint8)
        return x