# VİDEO datasetini olustur (b,f,c,h,w)
# eğitilmiş vae encodere ver oncesinde (b*f,c,h,w) olustur
# sonrasında tekrar (b,f,hidden_dim,latent_dim,latent_dim) olustur ve difussion procesi gerçekleştir
# sonrasında patchify işlemini gerçekleştir (b,f,p,hidden_dim),"[hidden dim, vae'de ogrenilen channel size aslında]""
# Sonrasında difussion transformera ver ve noise tahmin et
# sonrasında reshapele (b,f,p,hidden_dim) "bunu transformerda yapıyoruz" ve denoisingi uygula
# sonrasında egitilmiş vae decodera ver ve videoyu olustur tekrar

import dataset,diffusion,model,Vae_Encoder_Decoder,config,Checkpoints,train,prediction
import torch,warnings
from torch.utils.tensorboard import SummaryWriter

# Filter Warnings
warnings.filterwarnings("ignore")

# Check Devices:
devices=("cuda" if torch.cuda.is_available() else "cpu")

# Create Tensorboard:
Tensorboard=SummaryWriter("Tensorboard_Writer")

# Loading videos
video_list=dataset.Loading_video(config.VIDEO_PATH)
        
# Transform video shape
transformed_video=dataset.video_transform(video_list,config.RESHAPED_VIDEO_SIZE)

if config.PADDING==True:
    # Padding video
    padded_video=dataset.Pad_video(transformed_video,MAX_FRAME_SIZE=config.PADDING_FRAME_SIZE)

if config.CROP==True:
    # Or Cropping video
    cropped_video=dataset.Crop_video(transformed_video,crop_frame_size=config.CROP_FRAME_SIZE)

# Create Dataset
class_data=torch.randint(0,config.LABEL_SIZE,size=(len(transformed_video),),dtype=torch.int)
Train_Dataset=dataset.Dataset(padded_video,class_data)

# Random Split
train_Dataset,Valid_Dataset,Test_Dataset=dataset.Random_split(dataset=Train_Dataset,test_split=config.TEST_SIZE,valid_split=config.VALID_SIZE)

# Create Dataloader
Train_Dataloader,Valid_Dataloader,Test_Dataloader=dataset.Dataloader(train=train_Dataset,test=Test_Dataset,valid=Valid_Dataset,batch_size=config.BATCH_SIZE)

# Model Encoder for reach latents
Model_Encoder=Vae_Encoder_Decoder.Encoder(channel_size=config.CHANNEL_SIZE,hidden_dim=config.HIDDEN_DIM_VAE).to(devices)
Model_Encoder.eval() # Dont train

#Diffusion process model
Diffusion_fn=diffusion.Diffusion(config.BETA_START,config.BETA_END,config.N_TIMESTEPS,config.RESHAPED_VIDEO_SIZE,devices)

# Difussion Transformer to predict noise
Diffusion_Transformer=model.Diffusion_Transformer(embed_dim=config.EMBED_DIM_DIT,label_size=10,d_model=config.HIDDEN_DIM_DIT,
                                                  dk_model=config.DK_MODEL,INPUT_DIM=config.DIT_INPUT_DIM,N_times=config.N_TIMES,device=devices).to(devices)

# Model_Decoder to generate video from latent
Model_Decoder=Vae_Encoder_Decoder.Decoder(channel_size=config.CHANNEL_SIZE,hidden_dim=config.HIDDEN_DIM_VAE).to(devices)
Model_Decoder.eval() # Dont train 

# Optimizer and Loss
optimizer=torch.optim.Adam(params=Diffusion_Transformer.parameters(),lr=config.LR)
loss_fn=torch.nn.MSELoss()

# Load checkpoint if you have
STARTING_EPOCH=Checkpoints.Load_Checkpoint(checkpoint_dir=config.CHECKPOINT_DIR,model=Diffusion_Transformer,optimizer=optimizer,LOAD=config.LOAD_CHECKPOINT)

# Training
train.Training(config.EPOCH,Train_Dataloader,Model_Encoder,Model_Decoder,Diffusion_fn,optimizer,loss_fn,Diffusion_Transformer,Checkpoints.Save_Checkpoint
               ,config.CHECKPOINT_DIR,STARTING_EPOCH,PATCH_SIZE=config.PATCH_SIZE,Tensorboard=Tensorboard,devices=devices)

# Prediction
prediction.Prediction(config.PREDICTION,label="cat",class_to_idx=Train_Dataset.class_to_idx,
                      Diffussion_Model=Diffusion_Transformer,devices=devices,Vae_Decoder=Model_Decoder,Vae_Encoder=Model_Encoder,
                      PATCH_SIZE=config.PATCH_SIZE,video=train_Dataset[0][0])























