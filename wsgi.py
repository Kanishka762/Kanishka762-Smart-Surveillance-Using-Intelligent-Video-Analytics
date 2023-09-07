import threading
import multiprocessing

from init import *
from modules.components.load_paths import *
from modules.deepstream.rtsp2frames import main
from modules.components.structure_dev_dict import create_device_dict
from modules.db.db_push import gif_push
from modules.data_process.frame_data_process import frame_2_dict
from modules.face_recognition_pack.membersSubscriber import startMemberService
from modules.face_recognition_pack.recog_objcrop_face import load_lmdb_fst

import queue
# global_lock = threading.Lock()
#/home/srihari/deepstreambackend/modules/face_recognition_pack/membersSubscriber.py

mem_data_queue = queue.Queue()

if __name__ == '__main__':
    
    threading.Thread(target = gif_push).start()
    threading.Thread(target = frame_2_dict).start()
    # threading.Thread(target = merge_global).start()

    threading.Thread(target = startMemberService, args=(mem_data_queue,)).start()
    threading.Thread(target=load_lmdb_fst, args=(mem_data_queue,)).start()
    # p.daemon=True
    # p.start()
    dev_details = create_device_dict(mem_data_queue)
    print(dev_details)
    main(dev_details)

