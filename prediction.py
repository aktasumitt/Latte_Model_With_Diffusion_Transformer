import torch
import matplotlib.pyplot as plt
import imageio
import numpy as np

def Prediction(Prediction:bool,label:str,class_to_idx,Diffussion_Model,Vae_Encoder,Vae_Decoder,video,devices,PATCH_SIZE):
    if Prediction == False:
        print("Prediction is not going to do") 
    
    else: 
        video_random=torch.rand_like(video).to(devices).to(float)
        latent_video=Vae_Encoder(video_random)
        label_idx=class_to_idx[label]
        labels=torch.tensor([label_idx]).repeat(video.shape[0],).to(devices)
        
        video_pred=Diffussion_Model.Denoising(model=Diffussion_Model,Vae_Decoder=Vae_Decoder,video=latent_video,labels=labels,PATCH_SIZE=PATCH_SIZE)

        # Video tensorünü numpy formatına dönüştür
        video_numpy = video_pred.permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)

        # Videoyu kaydet
        imageio.mimwrite('Generated_video.mp4', video_numpy, fps=30)
