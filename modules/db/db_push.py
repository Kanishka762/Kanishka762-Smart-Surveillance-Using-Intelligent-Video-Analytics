from modules.components.load_paths import *
import imageio
import subprocess as sp
from os.path import join, dirname
from dotenv import load_dotenv
import time
import os
import psycopg2
from datetime import datetime
from pytz import timezone
import cv2
from psycopg2 import sql

# cwd = os.getcwd()
# data_path = join(cwd, 'data')
# dotenv_path = join(data_path, '.env')
load_dotenv(dotenv_path)

ipfs_url = os.getenv("ipfs")

pg_url = os.getenv("pghost")
pgdb = os.getenv("pgdb")
pgport = os.getenv("pgport")
pguser = os.getenv("pguser")
pgpassword = os.getenv("pgpassword")
ddns_env = os.getenv("DDNS_NAME")

def gst_hls_push(deviceInfo):
    
    for item in deviceInfo:
        
        if item == '':
            continue
        
        device_id = item['deviceId']
        ddns_name = item['ddns']
        if(ddns_name == None):
            hostname = ddns_env
        else:
            hostname = ddns_name
            
        hls_url = f'http://{hostname}/live/{device_id}/{device_id}.m3u8'        
        # Define the update statement
        query='''UPDATE "DeviceMetaData" SET uri=%s WHERE "deviceId"=%s;'''
            
        try:
            # Establish a connection to the PostgreSQL database
            connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
            # Create a cursor object
            cursor=connection.cursor()     
            # Execute the update statement with the specified values
            cursor.execute(query, (hls_url, device_id))           
            # Commit the changes and close the connection
            connection.commit()
            cursor.close()
            connection.close()    
            print("Updated the uri column in device table")
        except psycopg2.errors.SerializationFailure as e:
            # If the transaction encounters a serialization failure, retry with exponential backoff
            print(f"Transaction serialization failure: {e}")
            connection.rollback()
            cursor.close()
            connection.close()
            max_retries = 5
            delay = 0.2
            retry_count = 0
            while retry_count < max_retries:
                print(f"Retrying transaction after {delay} seconds...")
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
                    print("Transaction succeeded on retry")
                except psycopg2.errors.SerializationFailure as e:
                    print(f"Transaction serialization failure: {e}")
                    connection.rollback()
                    cursor.close()
                    connection.close()
                    delay *= 2
                    retry_count += 1
                except Exception as e:
                    print("Postges error occured: ", e)
                    connection.rollback()
                    cursor.close()
                    connection.close()
                    return
            print("Transaction failed after maximum retries")
        
        except Exception as e:
            print("Postges error occured: ", e)
            connection.rollback()
            cursor.close()
            connection.close()
            return
            #time.sleep(15)

def gif_push(file_path, device_info, gifBatch):
    
    deviceId = device_info['deviceId']
    tenantId = device_info['tenantId']
    
    print("LENGTH OF THE BATCH: ", len(gifBatch))
    with imageio.get_writer(file_path, mode="I") as writer:
        for idx, frame in enumerate(gifBatch):
            print("FRAME: ", idx)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb_frame)

    command = f'ipfs --api={ipfs_url} add {file_path} -Q'
    gif_cid = sp.getoutput(command)
    print("GIF CID: ", gif_cid)
    
    img_timestamp = str(datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S.%f'))
    
    img_name = f"THUMBNAIL_{img_timestamp}"
    
    print(f"IMAGE TIMESTAMP : {img_timestamp}")
    
    try:
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
        # Create a cursor object
        cursor=connection.cursor()       
        
        cursor.execute(sql.SQL("SELECT COUNT(*) FROM {} WHERE {} = %s;").format(sql.Identifier("Thumbnails"), sql.Identifier("deviceId")), [deviceId])
        row_count = cursor.fetchone()[0]
        
        print("COUNT: ", row_count)
        
        if row_count > 0:
            # Fetch the id of the matching row in Thumbnails table
            cursor.execute(sql.SQL("SELECT id FROM {} WHERE {} = %s;").format(sql.Identifier("Thumbnails"), sql.Identifier("deviceId")), [deviceId])
            thumbnail_id = cursor.fetchone()[0]
            
            print("THUMBNAIL ID: ", thumbnail_id)
            # Update the uri column in Images table where thumbnailId matches the fetched id
            cursor.execute(sql.SQL("UPDATE {} SET uri = %s, {} = %s, {} = %s, {} = %s WHERE {} = %s").format(sql.Identifier("Images"),sql.Identifier("timeStamp"),sql.Identifier("createdAt"),sql.Identifier("updatedAt"), sql.Identifier("thumbnailId")), [str(gif_cid), img_timestamp, img_timestamp, img_timestamp, thumbnail_id])
            print("Updated the Thumbnail column")
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
            print("Inserted the Thumbnail column")
         
        # Commit the changes and close the connection
        connection.commit()
        cursor.close()
        connection.close()       
    except psycopg2.errors.SerializationFailure as e:
        # If the transaction encounters a serialization failure, retry with exponential backoff
        print(f"Transaction serialization failure: {e}")
        connection.rollback()
        cursor.close()
        connection.close()
        max_retries = 5
        delay = 0.2
        retry_count = 0
        while retry_count < max_retries:
            print(f"Retrying transaction after {delay} seconds...")
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
                
                print("COUNT: ", row_count)
                
                if row_count > 0:
                    # Fetch the id of the matching row in Thumbnails table
                    cursor.execute(sql.SQL("SELECT id FROM {} WHERE {} = %s;").format(sql.Identifier("Thumbnails"), sql.Identifier("deviceId")), [deviceId])
                    thumbnail_id = cursor.fetchone()[0]
                    
                    print("THUMBNAIL ID: ", thumbnail_id)
                    # Update the uri column in Images table where thumbnailId matches the fetched id
                    cursor.execute(sql.SQL("UPDATE {} SET uri = %s, {} = %s, {} = %s, {} = %s WHERE {} = %s").format(sql.Identifier("Images"),sql.Identifier("timeStamp"),sql.Identifier("createdAt"),sql.Identifier("updatedAt"), sql.Identifier("thumbnailId")), [str(gif_cid), img_timestamp, img_timestamp, img_timestamp, thumbnail_id])
                    print("Updated the Thumbnail column")
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
                    print("Inserted the Thumbnail column")
                connection.commit()
                cursor.close()
                connection.close() 
                print("Transaction succeeded on retry")
            except psycopg2.errors.SerializationFailure as e:
                print(f"Transaction serialization failure: {e}")
                connection.rollback()
                cursor.close()
                connection.close()
                delay *= 2
                retry_count += 1
            except Exception as e:
                print("Postges error occured: ", e)
                connection.rollback()
                cursor.close()
                connection.close()
                return
            print("Transaction failed after maximum retries")
        
    except Exception as e:
        print("Postges error occured: ", e)
        connection.rollback()
        cursor.close()
        connection.close()
        return
        #time.sleep(15)
