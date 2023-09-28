from modules.components.load_paths import *
from init import loadLogger
from dotenv import load_dotenv
import time
import os
import psycopg2
from datetime import datetime
from pytz import timezone
import cv2
from psycopg2 import sql
import asyncio 
import datetime
import pytz
from modules.gif.gif_creation import result_queue

logger = loadLogger()

load_dotenv(dotenv_path)

ipfs_url = os.getenv("ipfs")

pg_url = os.getenv("pghost")
pgdb = os.getenv("pgdb")
pgport = os.getenv("pgport")
pguser = os.getenv("pguser")
pgpassword = os.getenv("pgpassword")
ddns_env = os.getenv("DDNS_NAME")
place = os.getenv("place")
timezone = pytz.timezone(f'{place}')  #assign timezone
num = 0

dt = datetime.datetime.now(timezone)

def create_connection():
    try:
        return psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
    except Exception as e:
        logger.error("Error while establishing connection to DB", exc_info=e)
        return None

def create_hls_url(device_id, ddns_name):
    try:
        hostname = ddns_env if ddns_name is None else ddns_name
        return f'https://{hostname}/live/{device_id}/{device_id}.m3u8'
    except Exception as e:
        logger.error("An error occurred in create_hls_url", exc_info=e)
        
def update_device_uri(device_id, hls_url):
    query = 'UPDATE "DeviceMetaData" SET uri=%s WHERE "deviceId"=%s;'
    try:
        return execute_HLS_update(query, hls_url, device_id)
    except psycopg2.Error as e:
        logger.error("PostgreSQL error occurred while HLS PUSH: ", exc_info=e)
        return False


# TODO Rename this here and in `update_device_uri`
def execute_HLS_update(query, hls_url, device_id):
    try:
        connection = create_connection()
        if connection is None:
            logger.error("Could not connect to DB")
            return False
        cursor = connection.cursor()
        cursor.execute(query, (hls_url, device_id))
        connection.commit()
        cursor.close()
        connection.close()
        logger.info("Updated the uri column in device table")
        return True
    except Exception as e:
        logger.error("An error occurred while updating uri column in device table", exc_info=e)
        return False
    
def retry_transaction(device_id, hls_url, max_retries=5, delay=0.2):
    try:
        retry_count = 0
        while retry_count < max_retries:
            logger.info(f"Retrying transaction after {delay} seconds...")
            time.sleep(delay)
            if update_device_uri(device_id, hls_url):
                logger.info("Transaction succeeded on retry")
                return
            delay *= 2
            retry_count += 1
        logger.info("Transaction failed after maximum retries")
    except Exception as e:
        logger.error("An error occurred while retrying transaction", exc_info=e)

def gst_hls_push(deviceInfo):
    try:
        if len(deviceInfo) > 0:
            for item in deviceInfo:
                if item == '':
                    continue

                device_id = item['deviceId']
                ddns_name = item['ddns']
                hls_url = create_hls_url(device_id, ddns_name)

                if not update_device_uri(device_id, hls_url):
                    retry_transaction(device_id, hls_url)
        else:
            logger.info("HLS deviceInfo is empty")
    except Exception as e:
        logger.error("Could not push HLS", exc_info=e)
        
def thumbnail_db_insertion(cursor, thumbnail_id, device_id, img_name, img_timestamp, gif_cid, tenant_id):
    if thumbnail_id is not None:
        logger.info(f"THUMBNAIL ID: {thumbnail_id}")
        cursor.execute(sql.SQL("UPDATE {} SET uri = %s, {} = %s, {} = %s, {} = %s WHERE {} = %s").format(
            sql.Identifier("Images"), sql.Identifier("timeStamp"), sql.Identifier("createdAt"),
            sql.Identifier("updatedAt"), sql.Identifier("thumbnailId")),
                        [str(gif_cid), img_timestamp, img_timestamp, img_timestamp, thumbnail_id])
        logger.info("Updated the Image table for Thumbnail")
    else:
        insert_thumbnail_and_image(cursor, device_id, img_name, img_timestamp, gif_cid, tenant_id)

def get_thumbnail_id(device_id, cursor):
    cursor.execute(sql.SQL("SELECT id FROM {} WHERE {} = %s;").format(sql.Identifier("Thumbnails"), sql.Identifier("deviceId")), [device_id])
    return cursor.fetchone()

def insert_thumbnail_and_image(cursor, device_id, img_name, img_timestamp, gif_cid, tenant_id):
    thumbnail_id = insert_thumbnail(cursor, img_name, img_timestamp, device_id)
    if thumbnail_id is not None:
        insert_image(cursor, img_name, img_timestamp, gif_cid, tenant_id, thumbnail_id)

def insert_thumbnail(cursor, img_name, img_timestamp, device_id):
    try:
        cursor.execute("""
            INSERT INTO "Thumbnails" (id, name, "timeStamp", "deviceId", "createdAt", "updatedAt")
            SELECT uuid_generate_v4(), %(name)s, %(timeStamp)s, %(deviceId)s, NOW(), NOW()
            RETURNING id;
        """, {'name': img_name, 'timeStamp': img_timestamp, 'deviceId': device_id})
        return cursor.fetchone()[0]
    except Exception as e:
        logger.error("Error while inserting thumbnail", exc_info=e)
        return None

def insert_image(cursor, img_name, img_timestamp, gif_cid, tenant_id, thumbnail_id):
    try:
        cursor.execute("""
            INSERT INTO "Images" (id, name, "timeStamp", uri, "tenantId", "activityId", "thumbnailId", "logId", "createdAt", "updatedAt")
            SELECT uuid_generate_v4(), %(name)s, %(timeStamp)s, %(uri)s, %(tenantId)s, %(activityId)s, %(thumbnailId)s, %(logId)s, NOW(), NOW();
        """, {'name': img_name, 'timeStamp': img_timestamp, 'uri': str(gif_cid), 'tenantId': tenant_id, 'activityId': None, 'thumbnailId': thumbnail_id, 'logId': None})
        logger.info("Inserted the Thumbnail column")
    except Exception as e:
        logger.error("Error while inserting image", exc_info=e)
        
def gif_variables(device_info):
    try:
        device_id = device_info['deviceId']
        tenant_id = device_info['tenantId']
        img_timestamp = str(datetime.datetime.now(timezone))
        img_name = f"THUMBNAIL_{img_timestamp}"
        logger.info("Created variables for gif push")
        return device_id, tenant_id, img_timestamp, img_name
    except Exception as e:
        logger.error("Error while creating variables for gif push", exc_info=e)
        return None
    
def gif_push():
    while True:
        try:
            logger.info("Started listening to thumbnail")
            global result_queue
            file_path, device_info, gifBatch, gif_cid = result_queue.get()
            device_id, tenant_id, img_timestamp, img_name = gif_variables(device_info)
        except Exception as e:
            logger.error("Error while calling gif variables", exc_info=e)

        try:
            connection = create_connection()
            if connection is None:
                logger.error("Could not connect to DB")
                continue
            cursor = connection.cursor()
            thumbnail_id = get_thumbnail_id(device_id, cursor)
            
            thumbnail_db_insertion(cursor, thumbnail_id, device_id, img_name, img_timestamp, gif_cid, tenant_id)

            connection.commit()
            cursor.close()
            connection.close()
        except psycopg2.errors.SerializationFailure as e:
            logger.error("Transaction serialization failure", exc_info=e)
            connection.rollback()
            cursor.close()
            connection.close()
            max_retries = 5
            delay = 0.2
            retry_count = 0

            while retry_count < max_retries:
                logger.info(f"Retrying transaction after {delay} seconds...")
                time.sleep(delay)
                connection = create_connection()
                if connection is None:
                    continue

                try:
                    cursor = connection.cursor()
                    thumbnail_id = get_thumbnail_id(device_id, cursor)

                    if thumbnail_id is not None:
                        cursor.execute(sql.SQL("UPDATE {} SET uri = %s, {} = %s, {} = %s, {} = %s WHERE {} = %s").format(
                            sql.Identifier("Images"), sql.Identifier("timeStamp"), sql.Identifier("createdAt"),
                            sql.Identifier("updatedAt"), sql.Identifier("thumbnailId")),
                                       [str(gif_cid), img_timestamp, img_timestamp, img_timestamp, thumbnail_id])
                        logger.info("Updated the Thumbnail column")
                    else:
                        insert_thumbnail_and_image(cursor, device_id, img_name, img_timestamp, gif_cid, tenant_id)

                    connection.commit()
                    cursor.close()
                    connection.close()
                    logger.info("Transaction succeeded on retry")
                    break
                except psycopg2.errors.SerializationFailure as e:
                    logger.error("Transaction serialization failure", exc_info=e)
                    connection.rollback()
                    cursor.close()
                    connection.close()
                    delay *= 2
                    retry_count += 1
                except Exception as e:
                    logger.error("PostgreSQL error occurred: ", exc_info=e)
                    connection.rollback()
                    cursor.close()
                    connection.close()
                    return
            else:
                logger.info("Transaction failed after maximum retries")
        except Exception as e:
            logger.error("PostgreSQL error occurred: ", exc_info=e)
            connection.rollback()
            cursor.close()
            connection.close()
            continue
            

