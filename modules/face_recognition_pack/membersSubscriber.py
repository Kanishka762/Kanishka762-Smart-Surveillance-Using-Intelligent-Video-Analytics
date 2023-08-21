#multi treading 
import asyncio
import lmdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
from pathlib import Path
import glob
import os
import PIL
import subprocess as sp
import nats

from nats.aio.client import Client as NATS

from modules.face_recognition_pack.facedatainsert_lmdb import add_member_to_lmdb
from modules.face_recognition_pack.recog_objcrop_face import load_lmdb_list


# from modules.face_recognition_pack.recog_objcrop_face import cb

nc = NATS()

async def cb(msg):
    try :
        data = (msg.data)
        #print(data)
        data  = data.decode()
        data = json.loads(data)
        print(data)
        status = add_member_to_lmdb(data)
        if status:
            load_lmdb_list()
            subject = msg.subject
            reply = msg.reply
            data = msg.data.decode()
            await nc.publish(msg.reply,b'ok')
            print("Received a message on '{subject} {reply}': {data}".format(
                subject=subject, reply=reply, data=data))
        
    except TypeError as e:
        print(TypeError," nats add member error >> ", e)
        
    finally:
        print("done with work ")
        # sem.release()

async def face_update_main():
    #await member_video_ipfs(member_did, member_name, member_cid)
    await nc.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
    # sub = await nc.subscribe("member.update.*", cb=cb)
    sub = await nc.subscribe("service.member_update", cb=cb)

def startMemberService():
    print("starting member service")
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(face_update_main())
    # loop.run_forever()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(face_update_main())
    loop.run_forever()

    # loop.close()

# if __name__ == "__main__":
#     startMemberService()

    # while True:

    #     asyncio.run(face_update_main())


    
    # except RuntimeError as e:
    #     print("error ", e)
    #     print(torch.cuda.memory_summary(device=None, abbreviated=False), "cuda")

  #  b'{"memberId":"did:ckdr:Ee1roJXJH4z2WkORnANm5gBpYUoIz7+6q9/1Gkr6y0KnFA==","faceid":["Qmf8ahSfoVeAVjkzamdsCZgnkw6FuuphSoAciGgH7zSvUP"],"type":"ADD_FACE","createdAt":"2023-02-02T16:42:49.233+05:30"}'
#{'id': 'tuxSv3G6ljsjjM2gTr-qN0sOL7HkB37j', 'type': 'FACEID', 'member': [{'memberId': 'did:ckdr:Ee1roJXJH4z2WkORnANm5gBpYUoIz7+6q9/1Gkr6y0KnFA==', 'faceCID': ['Qmf8ahSfoVeAVjkzamdsCZgnkw6FuuphSoAciGgH7zSvUP'], 'role': 'admin'}]}