import os
import torch

class Config:
       
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    save_dir = "checkpoints_UCSD"
    os.makedirs(save_dir, exist_ok=True)
    
    best_model_path = os.path.join(save_dir, "best_model.pth")
    last_model_path = os.path.join(save_dir, "last_model.pth")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    
    resume_training = False
    

    
       anomaly_ranges_dict = {
           "Test001": [(66, 180)],
           "Test002": [(100, 180)],
           "Test003": [(5, 146)],
           "Test004": [(36, 180)]
       }


