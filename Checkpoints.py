import torch


# Save Checkpoints
def Save_Checkpoint(epoch,optimizer,model,save_dir):
    
    checkpoint={"Epoch":epoch,
                "Optimizer_State":optimizer.state_dict(),
                "Model_State":model.state_dict()}
    
    torch.save(checkpoint,f=save_dir)
    print("\n...Checkpoint is saved...\n")


# Load Checkpoints
def Load_Checkpoint(LOAD:bool,checkpoint_dir:str,model,optimizer):
    if LOAD==True:
        checkpoint=torch.load(checkpoint_dir)
        
        start_epoch=checkpoint["Epoch"]
        optimizer.load_state_dict(checkpoint["Optimizer_State"])
        model.load_state_dict(checkpoint["Model_State"])
        
        print(f"\nLoading is complated. Training will start {start_epoch+1}.epoch\n")
    
    else: 
        start_epoch=0
        print(f"\nCheckpoint was not loaded. Training will start {start_epoch+1}.epoch\n")
    
    
    return start_epoch+1


