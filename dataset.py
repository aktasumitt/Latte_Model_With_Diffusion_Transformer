import torch
import torchvision.io as io 
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
import glob
import tqdm

# Loading Video , "crop or pad",and resize 
def Loading_video(path):
    video_list=[]
    PB=tqdm.tqdm((range(len(glob.glob(path+"\*")))),"Loading Videos")
    for files_path in glob.glob(path+"\*"):
        # Video ve ses yüklenir, ancak sadece video verisini kullanıyoruz
        video_frames, _, _i = io.read_video(files_path, pts_unit='sec',end_pts=240,output_format="TCHW")  # pts_unit='sec' zaman damgasını saniye biriminde döndürür
        video_list.append(video_frames)  
        PB.update(1)
    PB.close()
    return video_list

# Video Resize
def video_transform(video_list,resize_shape=512):
    transformed_list=[]
    PB=tqdm.tqdm(range(len(video_list)),"Resize Videos")
    for video in video_list:
        resized_video = transforms.Resize((resize_shape, resize_shape))(video) # Resize each video : (F,C,512,512)
        transformed_list.append(resized_video)
        PB.update(1)
    PB.close()
    return transformed_list
    
# Cropping video
def Crop_video(video_list,crop_frame_size):
    # "crop_frame=(VİDEO_FRAME_SIZE-240)" yani (crop_frame/fps) kadar da saniye kırpmıs olduk [örneğin video 30 fpsse 60 frame kırptıysak 2 sn kırptık ]
    cropped_list=[]
    
    PB=tqdm.tqdm(range(len(video_list)),"Cropping Videos")
    for video in video_list:
        video_cropped=video[:crop_frame_size,:,:,:] # Crop transformed video frame size : (CROP_FRAME_SIZE,C,512,512)
        cropped_list.append(video_cropped)
        PB.update(1)
    PB.close()
    return torch.stack(cropped_list)

# Padding Video
def Pad_video(video_list,MAX_FRAME_SIZE):
    # "pad_frame=(MAX_FRAME_SIZE-VİDEO_FRAME_SIZE)" yani pad_frame/fps kadar da siyah saniye eklemiş olduk [örneğin video 30 fps ise, 60 frame padlediysek 2 sn ekledik sona ]
    padded_list=[]
    
    PB=tqdm.tqdm(range(len(video_list)),"Padding Videos")
    for video in video_list:
        T,C,H,W=video.shape
        pad=torch.zeros((int(MAX_FRAME_SIZE-T),C,H,W)) # pad transformed video frame size : (max_frame_size,C,512,512)
        video_frames_resized=torch.cat([video,pad])
        padded_list.append(video_frames_resized)
        PB.update(1)
    PB.close()
    return torch.stack(padded_list)    


# Create Dataset 
class Dataset(Dataset):
    def __init__(self,video_data,class_data):
        
        self.transformer=transforms.Compose([transforms.Normalize((0.5,),(0.5,))])
        self.video_data=video_data
        self.class_data=class_data
        
    def __len__(self):
        return len(self.video_data)
    def class_to_idx(self):
        return 
    def __getitem__(self, index):
        video_data=self.transformer(self.video_data)
                
        return (video_data[index],self.class_data[index])
    
# Random Split    
def Random_split(dataset,test_split,valid_split):
    
    valid_size=int(len(dataset)*valid_split)
    test_size=int(len(dataset)*test_split)
    train_Size=int(len(dataset)-(test_size+valid_size))
    
    train,test,valid=random_split(dataset,[train_Size,test_size,valid_size])
    
    return train,valid,test


# Create Dataloader
def Dataloader(train=None,valid=None,test=None,batch_size=None):
    
    train_load=DataLoader(dataset=train,batch_size=batch_size,shuffle=True)
    test_load=DataLoader(dataset=test,batch_size=batch_size,shuffle=False)
    valid_load=DataLoader(dataset=valid,batch_size=batch_size,shuffle=False)
    
    print("Dataloaders were created...\n  ")
    
    return train_load,valid_load,test_load
