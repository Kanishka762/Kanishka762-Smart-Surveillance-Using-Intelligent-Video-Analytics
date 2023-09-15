import threading
import multiprocessing

from init import *
from modules.components.load_paths import *
from modules.deepstream.rtsp2frames import main
from modules.components.structure_dev_dict import create_device_dict
from modules.db.db_push import gif_push
from modules.data_process.frame_data_process import frame_2_dict



if __name__ == '__main__':
    
    threading.Thread(target = gif_push).start()
    threading.Thread(target = frame_2_dict).start()
    # threading.Thread(target = merge_global).start()


    # p.daemon=True
    # p.start()
    dev_details = create_device_dict()
    print(dev_details)
    main(dev_details)

