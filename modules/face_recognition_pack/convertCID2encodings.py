
import face_recognition 
import cv2
import subprocess as sp
import os

from modules.components.load_paths import *
# from modules.face_recognition_pack.facedatainsert_lmdb import add_member_to_lmdb
from dotenv import load_dotenv

load_dotenv(dotenv_path)
ipfs_url = os.getenv("ipfs")



def cid_to_imagenumpy(cid):
    global image_path
    #'ipfs --api={ipfs_url} add {file_path} -Q'.format(ipfs_url=ipfs_url, file_path=src_file)
    command = 'ipfs --api={ipfs_url} get {cid}'.format(ipfs_url=ipfs_url,cid=cid)
    output = sp.getoutput(command)
    im_path = image_path+'/'+str(cid)+".jpg"
    os.rename(cid, im_path)
    faceNumpy = cv2.imread(im_path)
    return faceNumpy

def faceNumpy2encodings(faceNumpy):
    try:
        encoding = face_recognition.face_encodings(faceNumpy)[0]
        return encoding
    except:
        # cv2.imwrite("er.jpg",faceNumpy)
        pass


def convertMemData2encoding(MemberPublish):
    list_of_members =  MemberPublish["member"]
    for each_member in list_of_members:
        faceCID = each_member["faceCID"]
        memberId = each_member["memberId"]
        class_type = each_member["type"]
        faceNumpy = cid_to_imagenumpy(faceCID[0])
        encodings = faceNumpy2encodings(faceNumpy)
        return encodings, class_type, memberId



