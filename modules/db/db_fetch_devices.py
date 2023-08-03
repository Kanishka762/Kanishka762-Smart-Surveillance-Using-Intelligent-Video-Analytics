from modules.components.load_paths import *
import psycopg2
from pytz import timezone 
from datetime import datetime
#.env vars loaded
import os
from os.path import join, dirname
from dotenv import load_dotenv
import ast

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
tenant_ids = os.getenv("TENANT_IDS")

print(pg_url, pgdb, pgport, pguser, pgpassword)

def fetch_db():
    try:
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
        # Create a cursor object
        cursor=connection.cursor()
        query =  '''SELECT dev.id, dev."tenantId", metadev.urn, metadev.ddns, metadev.ip, CAST(metadev.port AS INTEGER), metadev."videoEncodingInformation", dev."remoteUsername", metadev.rtsp, dev."remoteDeviceSalt", ARRAY_AGG(DISTINCT feat."name") AS feature_names, ge.latitude, ge.longitude
                FROM "Device" dev
                INNER JOIN "DeviceMetaData" metadev ON dev."deviceId" = metadev."deviceId"
                INNER JOIN "DeviceFeatures" devfeat ON dev."deviceId" = devfeat."deviceId" AND devfeat.enabled = True
                INNER JOIN "Features" feat ON devfeat."featureId" = feat.id
                INNER JOIN "Geo" ge ON ge."deviceMetaDataId" = metadev.id
                WHERE dev.id IS NOT NULL AND dev."tenantId" IS NOT NULL AND dev."remoteUsername" IS NOT NULL AND dev."remoteDeviceSalt" IS NOT NULL AND dev.deleted = False
                GROUP BY dev.id, dev."tenantId", metadev.urn, metadev.ddns, metadev.ip, metadev.port, metadev."videoEncodingInformation", dev."remoteUsername", metadev.rtsp, dev."remoteDeviceSalt", ge.latitude, ge.longitude;
            ;'''
        cursor.execute(query)
        connection.commit()
        print("Selecting rows from device table using cursor.fetchall")
        device_records = cursor.fetchall()
        return(device_records)
    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)
        connection.rollback()  # rollback the transaction if an error occurs
    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

def filter_devices():
    dev_det = fetch_db()
    updated_device_det =[]
    for i in range(len(dev_det)):
        if dev_det[i][1] in tenant_ids:
            updated_device_det.append(dev_det[i])
    return updated_device_det




# device_data = fetch_db()
# print(device_data)