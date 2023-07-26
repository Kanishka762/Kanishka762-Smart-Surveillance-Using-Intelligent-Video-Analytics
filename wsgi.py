from init import cleanup
from modules.components.load_paths import *
from modules.deepstream.rtsp2frames import start_deepstream
from modules.components.structure_dev_dict import create_device_dict

if __name__ == '__main__':
    cleanup()
    dev_details = create_device_dict()
    start_deepstream(dev_details)

