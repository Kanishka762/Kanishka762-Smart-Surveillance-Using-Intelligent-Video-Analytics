import os
from os.path import join, dirname

cwd = os.getcwd()

data_path = join(cwd, 'data')
dotenv_path = join(data_path, '.env')

static_path = join(cwd, 'static')
lmdb_path = join(static_path,'lmdb')
image_path = join(static_path,'image')
frame_path = join(static_path,'frames')
crop_path = join(static_path,'crops')
ipfs_tempdata_path = join(static_path,'ipfs_data')
gif_path = join(static_path, 'Gif_output')
hls_path = join(static_path, 'Hls_output')  
logs_path = join(static_path, 'logs')  

device_path = join(data_path, 'device_details.txt')