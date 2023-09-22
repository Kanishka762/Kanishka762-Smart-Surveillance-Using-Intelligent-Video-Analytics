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
num = 0

def gst_hls_push(deviceInfo):
    if len(deviceInfo) > 0:
        for item in deviceInfo:
            try:
                if item == '':
                    continue

                device_id = item['deviceId']
                ddns_name = item['ddns']

                hostname = ddns_env if ddns_name is None else ddns_name
                #TODO: 
                #move to .env 
                hls_url = f'https://{hostname}/live/{device_id}/{device_id}.m3u8'   

                query='''UPDATE "DeviceMetaData" SET uri=%s WHERE "deviceId"=%s;'''
                logger.info("created variables for gst_hls_push")

            except Exception as e:
                logger.error("error in creating variables for gst_hls_push", exc_info=e)

            try:

                connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)

                cursor=connection.cursor()     

                cursor.execute(query, (hls_url, device_id))           

                connection.commit()
                cursor.close()
                connection.close()    
                logger.info("Updated the uri column in device table")
            except psycopg2.errors.SerializationFailure as e:
                _extracted_from_gst_hls_push_64(
                    "Transaction serialization failure: ",
                    e,
                    connection,
                    cursor,
                )
                #TODO: 
                #move to .env 
                max_retries = 5
                delay = 0.2
                retry_count = 0

                while retry_count < max_retries:
                    logger.info(f"Retrying transaction after {delay} seconds...")
                    time.sleep(delay)
                    try:
                        # Establish a connection to the PostgreSQL database
                        connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
                        # Create a cursor object
                        cursor=connection.cursor()
                        # Execute the update statement with the specified values
                        cursor.execute(query, (hls_url, device_id))
                        connection.commit()
                        cursor.close()
                        connection.close() 
                        logger.info("Transaction succeeded on retry")
                    except psycopg2.errors.SerializationFailure as e:
                        _extracted_from_gst_hls_push_64(
                            "Transaction serialization failure:",
                            e,
                            connection,
                            cursor,
                        )
                        delay *= 2
                        retry_count += 1
                    except Exception as e:
                        _extracted_from_gst_hls_push_64(
                            "Postges error occured: ", e, connection, cursor
                        )
                        return
                logger.info("Transaction failed after maximum retries")

            except Exception as e:
                _extracted_from_gst_hls_push_64(
                    "Postges error occured: ", e, connection, cursor
                )
                return
                            #time.sleep(15)
    else:
        try:
            raise Exception("there are no devices in deviceInfo")
        except Exception as e:
            logger.error("", exc_info=e)


# TODO Rename this here and in `gst_hls_push`
def _extracted_from_gst_hls_push_64(arg0, e, connection, cursor):
    logger.error(arg0, exc_info=e)
    connection.rollback()
    cursor.close()
    connection.close()

def gif_push():

    while True:
        try:
            logger.info("started listning to thumbnail")
            # logger.info("creating variables for gif push")
            global result_queue
            file_path, device_info, gifBatch, gif_cid = result_queue.get()

            deviceId = device_info['deviceId']
            tenantId = device_info['tenantId']
        
            img_timestamp = str(datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            img_name = f"THUMBNAIL_{img_timestamp}"
            
            # print(f"IMAGE TIMESTAMP : {img_timestamp}")
            logger.info("created variables for gif push")
        except Exception as e:
            logger.error("error in creating variables for gif push", exc_info=e)


        
        try:
            # logger.info("created variables for gif push")
            connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
            cursor=connection.cursor()  
        except Exception as e:
            logger.error("error while establishing connection to DB", exc_info=e)
        
        try:
            cursor.execute(sql.SQL("SELECT COUNT(*) FROM {} WHERE {} = %s;").format(sql.Identifier("Thumbnails"), sql.Identifier("deviceId")), [deviceId])
            row_count = cursor.fetchone()[0]

            if row_count > 0:
                
                cursor.execute(sql.SQL("SELECT id FROM {} WHERE {} = %s;").format(sql.Identifier("Thumbnails"), sql.Identifier("deviceId")), [deviceId])
                thumbnail_id = cursor.fetchone()[0]
                
                logger.info(f"THUMBNAIL ID: {thumbnail_id}")
                # Update the uri column in Images table where thumbnailId matches the fetched id
                cursor.execute(sql.SQL("UPDATE {} SET uri = %s, {} = %s, {} = %s, {} = %s WHERE {} = %s").format(sql.Identifier("Images"),sql.Identifier("timeStamp"),sql.Identifier("createdAt"),sql.Identifier("updatedAt"), sql.Identifier("thumbnailId")), [str(gif_cid), img_timestamp, img_timestamp, img_timestamp, thumbnail_id])
                logger.info("Updated the Thumbnail column")
            else:
                query = """
                            WITH inserted_thumbnail AS (
                            INSERT INTO "Thumbnails" (id, name, "timeStamp", "deviceId", "createdAt", "updatedAt")
                            SELECT uuid_generate_v4(), %(name)s, %(timeStamp)s, %(deviceId)s, NOW(), NOW()
                            RETURNING id
                            )
                            INSERT INTO "Images" (id, name, "timeStamp", uri, "tenantId", "activityId", "thumbnailId", "logId", "createdAt", "updatedAt")
                            SELECT
                            uuid_generate_v4(), %(name)s, %(timeStamp)s, %(uri)s, %(tenantId)s, %(activityId)s,
                            (SELECT id FROM inserted_thumbnail),
                            %(logId)s, NOW(), NOW();
                        """
                        
                # Execute the update statement with the specified values
                cursor.execute(query, {
                    'deviceId': deviceId, 
                    'name':img_name, 
                    'timeStamp': img_timestamp, 
                    'uri': str(gif_cid), 
                    'tenantId': tenantId, 
                    'activityId': None, 
                    'logId': None
                    })
                logger.info("Inserted the Thumbnail column")
            # cursor.execute(f"SELECT pg_advisory_unlock({result_id})")
            # Commit the changes and close the connection
            connection.commit()
            cursor.close()
            connection.close()  
                 
        except psycopg2.errors.SerializationFailure as e:
            # If the transaction encounters a serialization failure, retry with exponential backoff
            logger.error("Transaction serialization failure", exc_info=e)
            connection.rollback()
            cursor.close()
            connection.close()
            #TODO: 
            #move to .env 
            max_retries = 5
            delay = 0.2
            retry_count = 0
            while retry_count < max_retries:
                logger.info(f"Retrying transaction after {str(delay)} seconds...")
                time.sleep(delay)
                try:
                    # Establish a connection to the PostgreSQL database
                    connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
                    # Create a cursor object
                    cursor=connection.cursor()
                    # # Execute the update statement with the specified values
                    # cursor.execute(query, (img_name, img_timestamp, deviceId, img_name, img_timestamp, str(gif_cid), tenantId, None, None))
                    cursor.execute(sql.SQL("SELECT COUNT(*) FROM {} WHERE {} = %s;").format(sql.Identifier("Thumbnails"), sql.Identifier("deviceId")), [deviceId])
                    row_count = cursor.fetchone()[0]
                    
                    logger.info(f"COUNT:{str(row_count)}")
                    
                    if row_count > 0:
                        # Fetch the id of the matching row in Thumbnails table
                        cursor.execute(sql.SQL("SELECT id FROM {} WHERE {} = %s;").format(sql.Identifier("Thumbnails"), sql.Identifier("deviceId")), [deviceId])
                        thumbnail_id = cursor.fetchone()[0]
                        
                        logger.info(f"THUMBNAIL ID: {str(thumbnail_id)}")
                        # Update the uri column in Images table where thumbnailId matches the fetched id
                        cursor.execute(sql.SQL("UPDATE {} SET uri = %s, {} = %s, {} = %s, {} = %s WHERE {} = %s").format(sql.Identifier("Images"),sql.Identifier("timeStamp"),sql.Identifier("createdAt"),sql.Identifier("updatedAt"), sql.Identifier("thumbnailId")), [str(gif_cid), img_timestamp, img_timestamp, img_timestamp, thumbnail_id])
                        logger.info("Updated the Thumbnail column")
                    else:
                        query = """
                                    WITH inserted_thumbnail AS (
                                    INSERT INTO "Thumbnails" (id, name, "timeStamp", "deviceId", "createdAt", "updatedAt")
                                    SELECT uuid_generate_v4(), %(name)s, %(timeStamp)s, %(deviceId)s, NOW(), NOW()
                                    RETURNING id
                                    )
                                    INSERT INTO "Images" (id, name, "timeStamp", uri, "tenantId", "activityId", "thumbnailId", "logId", "createdAt", "updatedAt")
                                    SELECT
                                    uuid_generate_v4(), %(name)s, %(timeStamp)s, %(uri)s, %(tenantId)s, %(activityId)s,
                                    (SELECT id FROM inserted_thumbnail),
                                    %(logId)s, NOW(), NOW();
                                """
                                
                        # Execute the update statement with the specified values
                        cursor.execute(query, {
                            'deviceId': deviceId, 
                            'name':img_name, 
                            'timeStamp': img_timestamp, 
                            'uri': str(gif_cid), 
                            'tenantId': tenantId, 
                            'activityId': None, 
                            'logId': None
                            })
                        logger.info("Inserted the Thumbnail column")
                    connection.commit()
                    cursor.close()
                    connection.close() 
                    logger.info("Transaction succeeded on retry")
                except psycopg2.errors.SerializationFailure as e:
                    logger.error(f"Transaction serialization failure",exc_info=e)
                    connection.rollback()
                    cursor.close()
                    connection.close()
                    delay *= 2
                    retry_count += 1
                    
                except Exception as e:
                    logger.error("Postges error occured: ", exc_info=e)
                    connection.rollback()
                    cursor.close()
                    connection.close()
                    
                    return
                logger.info("Transaction failed after maximum retries")
            
        except Exception as e:
            logger.error("Postges error occured: ", exc_info=e)
            connection.rollback()
            cursor.close()
            connection.close()
            return
            #time.sleep(15)
            

