import torch
import tqdm
from einops import rearrange


# Training Func
def Training(EPOCH,Train_Dataloader,Vae_Encoder,Vae_Decoder,Difussion,optimizer,loss_fn,Model,Save_Checkpoint_fn,Checkpoints_dir,STARTING_EPOCH=1,PATCH_SIZE=16,Tensorboard=None,devices="cpu"):
        
    for epoch in range(STARTING_EPOCH,EPOCH+1):
        
        train_loss_value=0
        
        progress_bar=tqdm.tqdm(range(len(Train_Dataloader)),"Training Progress")
            
        for batch_train,(video,label) in enumerate(Train_Dataloader):
                                    
            video_train=video.to(devices)
            label_train=label.to(devices)
            
            ## ALL RESHAPİNG MADE IN MODELS CLASS ##
            
            # Encoder for latent from video
            latent_video=Vae_Encoder(video_train)
            
            # Difussion the latent
            t=Difussion.Random_Timesteps(video_train.shape[0])
            noisy_latent,noise=Difussion.Noising_to_Image(latent_video,t)
            
            # Patchify
            patched_noisy_latent=rearrange(noisy_latent,"B F C (H P1) (W P2) -> B F (H W) (C P1 P2)",P1=PATCH_SIZE,P2=PATCH_SIZE)
            
            # We use Classifier free guidance
            CFG_SCALE=torch.randint(1,101,(1,)).item()
            if CFG_SCALE<10:
                labels_train=None
            
            # Train Difussion Transformer with noisy latent and output is noise prediction
            optimizer.zero_grad()
            noise_pred=Model(patched_noisy_latent,t,label_train)
            
            # Depatch Predictied Noise
            depatched_noise=rearrange(noise_pred,"B F (H W) (C P1 P2) ->B F C (H P1) (W P2)",P1=PATCH_SIZE,P2=PATCH_SIZE,H=latent_video.shape[-1]//PATCH_SIZE)
            
            loss_train=loss_fn(depatched_noise,noise)       
            loss_train.backward()
            optimizer.step()
            
            # Calculating loss
            train_loss_value+=loss_train.item()   
            progress_bar.update(1)
            break

        progress_bar.set_postfix({"EPOCHS":epoch,
                                  "BATCH":batch_train+1,
                                  "Loss_Train":train_loss_value/(batch_train+1)})            
            
        Tensorboard.add_scalar("Loss_Train",train_loss_value/(batch_train+1),global_step=epoch)
        
        progress_bar.close()
        # Save Checkpoint
        Save_Checkpoint_fn(epoch=epoch,optimizer=optimizer,model=Model,save_dir=Checkpoints_dir)
        
        # Denoise process from random noise and video upsampling to normal shape with vae decoder
        video_prediction=Difussion.Denoising(Model,Vae_Decoder,latent_video,label_train,PATCH_SIZE) # Denoise with
                
        # Tensorboard
        Tensorboard.add_video("pred video",video_prediction,epoch,fps=300)
        break
        