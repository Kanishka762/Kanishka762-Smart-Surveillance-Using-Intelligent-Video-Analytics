from os.path import join, dirname
from dotenv import load_dotenv
import os
from db_fetch import fetch_db

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

def filter_devices():
    tenant_ids = os.getenv("TENANT_IDS")
    # print(tenant_ids)
    dev_det = fetch_db()
    # print(device_det[0])
    updated_device_det =[]
    for i in range(len(dev_det)):
        if dev_det[i][1] in tenant_ids:
            updated_device_det.append(dev_det[i])
    return updated_device_det

