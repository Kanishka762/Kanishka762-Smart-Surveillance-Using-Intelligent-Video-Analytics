import os
from os.path import join, dirname
import shutil
from modules.components.load_paths import *
from modules.components.clean_dir import remove_cnts

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


