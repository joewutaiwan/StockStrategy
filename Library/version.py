import os
from datetime import datetime
from keras.models import Model

def save(result_model, 
    save_encoder_path,
    result_name,
    time_str):
    
    dir_path = save_encoder_path
    dir_version_path = save_encoder_path + "/" + time_str
    if not os.path.exists(dir_version_path):
        os.makedirs(dir_version_path)
    result_model.save(dir_path + "/" + result_name)
    result_model.save(dir_version_path + "/" + result_name)

