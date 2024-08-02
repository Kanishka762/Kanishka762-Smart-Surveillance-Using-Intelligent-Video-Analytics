from modules.components.load_paths import *
from init import loadLogger


import os 
import ast
from dotenv import load_dotenv
import queue


mem_data_queue = queue.Queue()
logger = loadLogger()

load_dotenv(dotenv_path)

rtsp_links = ast.literal_eval(os.getenv("rtsp_links"))

def create_device_dict():  # sourcery skip: remove-unused-enumerate
    try:
        with open(device_path, "r") as file:
            # Read all lines from the file
            devices = file.readlines()
            devices = [line.strip() for line in devices]
            return devices
    except Exception as e:
        logger.error("An error occurred while structuring device details", exc_info=e)

