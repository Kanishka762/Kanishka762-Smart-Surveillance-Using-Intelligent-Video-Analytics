from init import *
from modules.components.load_paths import *
from modules.deepstream.person_model import start_person_model
from modules.deepstream.fire_model import start_fire_model
from modules.components.structure_dev_dict import create_device_dict
import threading
firesmoke = []
basee = []
if __name__ == '__main__':
    dev_details = create_device_dict()
    for each in dev_details:
        if "Fire/Smoke" in each["subscriptions"]:
            firesmoke.append(each)
        else:
            basee.append(each)

    threading.Thread(target=start_person_model,args=(basee,)).start()
    start_fire_model(firesmoke)

