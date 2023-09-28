import os
from os.path import join, dirname
import shutil
from modules.components.load_paths import *
from modules.components.clean_dir import remove_cnts
import logging
import logging.config

if os.path.exists(static_path):
    shutil.rmtree(static_path)
if os.path.exists(static_path) is False:
    os.mkdir(static_path)

if os.path.exists(gif_path):
    shutil.rmtree(gif_path)
if os.path.exists(gif_path) is False:
    os.mkdir(gif_path)

if os.path.exists(hls_path):
    shutil.rmtree(hls_path)
if os.path.exists(hls_path) is False:
    os.mkdir(hls_path)

if os.path.exists(lmdb_path):
    remove_cnts(lmdb_path)
if not os.path.exists(lmdb_path):
    os.makedirs(lmdb_path)

if os.path.exists(image_path):
    remove_cnts(image_path)
if not os.path.exists(image_path):
    os.makedirs(image_path)
    
if os.path.exists(frame_path):
    remove_cnts(frame_path)
if not os.path.exists(frame_path):
    os.makedirs(frame_path)
    
if os.path.exists(crop_path):
    remove_cnts(crop_path)
if not os.path.exists(crop_path):
    os.makedirs(crop_path)

if os.path.exists(logs_path):
    remove_cnts(logs_path)
if not os.path.exists(logs_path):
    os.makedirs(logs_path)


def loadLogger():
    logger_path = join(static_path, 'logs')
    # print( join(data_path, 'log.conf'))
    log_conf = join(data_path, 'log.conf')

    logging.config.fileConfig(fname=log_conf, disable_existing_loggers=False)
    logger = logging.getLogger('BackendLogger')
    return logger
