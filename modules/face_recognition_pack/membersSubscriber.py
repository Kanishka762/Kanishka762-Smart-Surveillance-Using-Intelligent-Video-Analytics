from init import loadLogger
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

logger = loadLogger()

nc = NATS()

async def cb(msg):
    try :
        global mem_data_queue
        data = (msg.data)
        data  = data.decode()
        data = json.loads(data)
        logger.info("received member data")
        logger.info(data)
        mem_data_queue.put([data])
        subject = msg.subject
        reply = msg.reply
        data = msg.data.decode()
        await nc.publish(msg.reply,b'ok')
        data("Received a message on '{subject} {reply}': {data}".format(
            subject=subject, reply=reply, data=data))
        
    except TypeError as e:
        # print(TypeError," nats add member error >> ", e)
        logger.error("nats add member error >> ", exc_info=e)

        
    finally:
        logger.info("done with work ")
        # sem.release()

async def face_update_main():
    try:
        logger.info("establishing connection with nats")
        await nc.connect(servers=["nats://216.48.181.154:5222"] , reconnect_time_wait= 50 ,allow_reconnect=True, connect_timeout=20, max_reconnect_attempts=60)
        sub = await nc.subscribe("member.update.faceid", cb=cb)
    except Exception as e:
        logger.error("An error occurred while connecting to nats", exc_info=e)
        
def startMemberService(q):
    try:
        global mem_data_queue
        mem_data_queue = q
        logger.info("starting member service")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(face_update_main())
        loop.run_forever()
    except Exception as e:
        logger.error("An error occurred while starting member service", exc_info=e)
        

  #  b'{"memberId":"did:ckdr:Ee1roJXJH4z2WkORnANm5gBpYUoIz7+6q9/1Gkr6y0KnFA==","faceid":["Qmf8ahSfoVeAVjkzamdsCZgnkw6FuuphSoAciGgH7zSvUP"],"type":"ADD_FACE","createdAt":"2023-02-02T16:42:49.233+05:30"}'
#{'id': 'tuxSv3G6ljsjjM2gTr-qN0sOL7HkB37j', 'type': 'FACEID', 'member': [{'memberId': 'did:ckdr:Ee1roJXJH4z2WkORnANm5gBpYUoIz7+6q9/1Gkr6y0KnFA==', 'faceCID': ['Qmf8ahSfoVeAVjkzamdsCZgnkw6FuuphSoAciGgH7zSvUP'], 'role': 'admin'}]}