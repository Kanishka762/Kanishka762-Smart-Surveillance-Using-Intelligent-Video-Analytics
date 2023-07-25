import psycopg2
from psycopg2 import OperationalError, Error, DatabaseError
from psycopg2.extras import RealDictCursor
from pytz import timezone 
from datetime import datetime
#.env vars loaded
import os
from os.path import join, dirname
from dotenv import load_dotenv
import ast
import uuid
import reverse_geocode
import json
cwd = os.getcwd()
data_path = join(cwd, 'data')
dotenv_path = join(data_path, '.env')
load_dotenv(dotenv_path)

ipfs_url = os.getenv("ipfs")
nats_urls = os.getenv("nats")
nats_urls = ast.literal_eval(nats_urls)

pg_url = os.getenv("pghost")
pgdb = os.getenv("pgdb")
pgport = os.getenv("pgport")
pguser = os.getenv("pguser")
pgpassword = os.getenv("pgpassword")

track_type = ast.literal_eval(os.getenv("track_type"))

# pg_url='216.48.190.128'
# pgdb='postgres'
# pgport='26257'
# pguser='root'
# pgpassword=''

ack = False


def dbpush_activities(act_out):

    try:
        print("PUSHING THE ACTIVITY CONTENTS TO DB")
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
        # Create a cursor object
        cursor=connection.cursor(cursor_factory=RealDictCursor)

        # Convert the object array to JSON string
        object_array = json.dumps(act_out['metaData']['object'])
            
        # to form the title
        data = act_out['metaData']['object']

        # Convert 'None' to an empty string for the 'activity' field
        for obj in data:
            if obj['activity'] is None:
                obj['activity'] = ''
        
        num_people = len([item for item in data if item['class'] == 'Male' or item['class'] == 'Female'])
        activities = list(set([item['activity'] for item in data if item['activity'] is not None])) or [None]
        if activities is None:
            activity_string = 'detected'
        else:
            activity_string = ' and '.join(activities)
            
        if num_people == 1:
            title = f"1 person {activity_string}"
        else:
            title = f"{num_people} people {activity_string}"
        
        if act_out["type"] == "anomaly":
            category_type = "ANOMALIE"
            anomaly_type = "TRUE"
        else:
            category_type = "DETECTION"
            anomaly_type = "FALSE"
        
        coordinates = [(act_out['geo']['latitude'], act_out['geo']['longitude'])]
        result = reverse_geocode.search(coordinates)
        loc_name = str(result[0]['city'])
        
        # img_name_1 = f"LOG_{act_out['timestamp']}"
        
        query = """
            WITH inserted_activity AS (
            INSERT INTO "Activities" (id, "tenantId", "batchId", "category", "memberId", location, title, timestamp, score, "deviceId", "createdAt", "updatedAt")
            VALUES (uuid_generate_v4(), %(tenantId)s, %(batchId)s, %(category)s, %(memberId)s, %(location)s, %(title)s, %(timestamp)s, %(score)s, %(deviceId)s, NOW(), NOW())
            RETURNING id
            ),
            inserted_images AS (
            INSERT INTO "Images" (id, name, "timeStamp", uri, "tenantId", "activityId", "thumbnailId", "logId", "createdAt", "updatedAt")
            SELECT uuid_generate_v4(), 'ACTIVITY_' || to_char(NOW(), 'YYYYMMDDHH24MISSMS'), %(timestamp)s, %(uri1)s, %(tenantId)s, inserted_activity.id, %(thumbnailId)s, %(logId)s, NOW(), NOW()
            FROM inserted_activity
            RETURNING id
            ),
            inserted_activity_meta AS (
            INSERT INTO "ActivityMeta" (id, "peopleCount", "vehicleCount", anomaly, "activityId", category, "createdAt", "updatedAt")
            SELECT uuid_generate_v4(), %(peopleCount)s, %(vehicleCount)s, %(anomaly)s, inserted_activity.id, %(category)s, NOW(), NOW()
            FROM inserted_activity
            RETURNING id
            ),
            inserted_geo AS (
            INSERT INTO "Geo" (id, latitude, name, longitude, "deviceMetaDataId", "metaId", "createdAt", "updatedAt")
            SELECT uuid_generate_v4(), %(latitude)s, %(location)s, %(longitude)s, (SELECT d.id FROM "DeviceMetaData" d WHERE d."deviceId" = %(deviceId)s), inserted_activity_meta.id, NOW(), NOW()
            FROM inserted_activity_meta
            RETURNING id
            ),
            inserted_logs AS (
            INSERT INTO "Logs" (id, "tenantId", "_id", class, track, activity, cid, "memberId", "activityId", "createdAt", "updatedAt")
            SELECT uuid_generate_v4(), %(tenantId)s, object->>'id', object->>'class', object->>'track', CASE WHEN object->>'activity' <> '' THEN object->>'activity' ELSE NULL END, object->>'cids', %(memberId)s, inserted_activity.id, NOW(), NOW()
            FROM inserted_activity, jsonb_array_elements(%(objects)s) AS object
            RETURNING id
            ),
            inserted_images2 AS (
            INSERT INTO "Images" (id, name, "timeStamp", uri, "tenantId", "activityId", "thumbnailId", "logId", "createdAt", "updatedAt")
            SELECT uuid_generate_v4(), 'LOG_' || to_char(NOW(), 'YYYYMMDDHH24MISSMS'), %(timestamp)s, object->>'cids', %(tenantId)s, %(activityId)s, %(thumbnailId)s, inserted_logs.id, NOW(), NOW()
            FROM inserted_logs, jsonb_array_elements(%(objects)s) AS object
            RETURNING id
            )
            SELECT * 
            FROM (
                SELECT * FROM inserted_activity
                UNION ALL
                SELECT * FROM inserted_images
                UNION ALL
                SELECT * FROM inserted_activity_meta
                UNION ALL
                SELECT * FROM inserted_geo
                UNION ALL
                SELECT * FROM inserted_logs
                UNION ALL
                SELECT * FROM inserted_images2
            )AS inserted_rows;
        """
        
        cursor.execute(query,{
        'tenantId': act_out['tenantId'], 
        'batchId': act_out['batchid'], 
        'memberId': None, 
        'location': str(loc_name), 
        'title': title, 
        'timestamp': act_out['timestamp'], 
        'score': act_out['metaData']['frameAnomalyScore'], 
        'deviceId': act_out['deviceid'],
        'uri1': act_out['metaData']['cid'],
        'thumbnailId': None, 
        'logId': None, 
        'activityId': None,
        'peopleCount': act_out["metaData"]["count"]["peopleCount"], 
        'vehicleCount': act_out["metaData"]["count"]["vehicleCount"], 
        'anomaly': anomaly_type, 
        'category': category_type, 
        'latitude': act_out['geo']['latitude'], 
        'longitude': act_out['geo']['longitude'],
        'objects': object_array
        })
                
        # Fetch all inserted rows
        rows = cursor.fetchall()

        # Print the inserted rows
        for row in rows:
            print(row)
        
        # Commit the changes and close the connection
        connection.commit()
        cursor.close()
        connection.close()
        
        print("Data inserted successfully!")
        return("SUCCESS!!")
        
    except (Exception, psycopg2.Error, OperationalError, Error, DatabaseError) as error:
        # Handle exceptions and rollback the transaction if necessary
        if 'conn' in locals():
            connection.rollback()
            cursor.close()
            connection.close()
        print(f"Error occurred during data insertion: {error}")
        return("FAILURE!!")
        
def dbpush_members(mem_out):
    try:
        print("PUSHING THE MEMBER CONTENTS TO DB")
        
        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(host=pg_url, database=pgdb, port=pgport, user=pguser, password=pgpassword)
        # Create a cursor object
        cursor=connection.cursor(cursor_factory=RealDictCursor)
                
        for item in mem_out['metaData']['object']:
            # if((item['track'] is not None) and (item['memDID'] is not None)):
            if((item['track'] == "100") and (item['memDID'] == None)):
                type_track = track_type[item['track']]
                if(item['track'] == "01"):
                    is_blacklist = True
                else:
                    is_blacklist = False
                query = """
                        WITH inserted_member AS (
                        INSERT INTO "Member" (id, type, "tenantId", track, "blackListed", faceid, "createdAt", "updatedAt")
                        VALUES (uuid_generate_v4(), %(type)s, %(tenantId)s, %(track)s, %(blackListed)s, %(faceid)s, NOW(), NOW())
                        RETURNING id
                        ),
                        inserted_tags AS (
                        INSERT INTO "Tags" (id, "tenantId", name, active, "memberId", "deviceId", "taggableType", "createdAt", "updatedAt")
                        VALUES (uuid_generate_v4(), %(tenantId)s, %(type)s, %(active)s, (SELECT id FROM inserted_member), %(deviceId)s, %(taggableType)s, NOW(), NOW())
                        RETURNING id
                        ),
                        inserted_member_tags AS (
                        INSERT INTO "MemberTags" ("memberId", "tagId", "createdAt", "updatedAt")
                        VALUES ((SELECT id FROM inserted_member), (SELECT id FROM inserted_tags), NOW(), NOW())
                        RETURNING "memberId"
                        ),
                        updated_activities AS (
                        UPDATE "Activities"
                        SET "memberId" = %(memberId)s
                        WHERE "batchId" = %(batchId)s
                        RETURNING id
                        ),
                        updated_logs AS (
                        UPDATE "Logs"
                        SET track = %(track)s, "memberId" = (SELECT id FROM inserted_member)
                        WHERE "activityId" IN (SELECT id FROM updated_activities)
                        RETURNING id
                        )
                        SELECT * 
                        FROM (
                            SELECT * FROM inserted_member
                            UNION ALL
                            SELECT * FROM inserted_tags
                            UNION ALL
                            SELECT * FROM inserted_member_tags
                            UNION ALL
                            SELECT * FROM updated_activities
                            UNION ALL
                            SELECT * FROM updated_logs
                        )AS inserted_rows;
                    """
            
                cursor.execute(query,{
                'type': type_track,
                'tenantId': mem_out['tenantId'], 
                'track': item['track'],
                'blackListed': is_blacklist,
                'faceid': '{' + item['cids'] + '}',
                'active': True,
                'deviceId': mem_out['deviceid'],
                'taggableType': 'Member',
                'memberId': None,
                'batchId': mem_out['batchid'], 
                })                   
                
                # Fetch all inserted rows
                rows = cursor.fetchall()
                
                # print(rows)

                # Print the inserted rows
                for row in rows:
                    print(row)
                        
        # Commit the changes and close the connection
        connection.commit()
        cursor.close()
        connection.close()
        
        print("Data inserted successfully!")
        return("SUCCESS!!")
        
    except (Exception, psycopg2.Error, OperationalError, Error, DatabaseError) as error:
        # Handle exceptions and rollback the transaction if necessary
        if 'conn' in locals():
            connection.rollback()
            cursor.close()
            connection.close()
        print(f"Error occurred during data insertion: {error}")
        return("FAILURE!!")               
    
    
    
# act_dict = {'type': 'anomaly', 'deviceid': '63627047-611a-46fb-81e2-7c6cfe0b32ca', 'batchid': 'd50ab502-c52f-46a1-ac2f-acca357f7997', 'timestamp': '2023-07-22 13:09:41.370124+05:30', 'geo': {'latitude': 28.7, 'longitude': 77.1}, 'metaData': {'detect': 2, 'frameAnomalyScore': 0.0, 'count': {'peopleCount': 2, 'vehicleCount': 0, 'ObjectCount': 0}, 'anomalyIds': [], 'cid': 'QmRBUoyiL5UqJFhAc4EgTQ4MbX9pzqochD3G9heU17zSAm', 'object': [{'class': 'Male', 'detectionScore': 0.0, 'activityScore': 0.0, 'track': None, 'id': '2', 'memDID': 'None', 'activity': 'No Activity', 'detectTime': '2023-07-22 13:09:41.370124+05:30', 'cids': 'QmQKPtUeZbcunpu6XkhvNKmw2aTDZE2KxAJg35sG5gToJr'}, {'class': 'Gun', 'detectionScore': 0.0, 'activityScore': 0.0, 'track': None, 'id': '5', 'memDID': 'None', 'activity': 'No Activity', 'detectTime': '2023-07-22 13:09:41.370124+05:30', 'cids': 'QmTFdL2FnNmwitme6UBhcrxWWTsEYfwaGtHGkyvyBRZe3z'}]}, 'tenantId': '41335e9a-05d9-4cca-99c6-a44dc4217056', 'version': 'v0.0.4'}
# dbpush_activities(act_dict)

# mem_dict = {'type': 'activity', 'deviceid': 'daf2a1b9-a5c7-47f0-a57c-e941f47b670a', 'batchid': 'bf72ff47-d794-4a17-aaad-f3a6733f03b1', 'timestamp': '2023-06-19 10:37:20.284136+05:30', 'geo': {'latitude': 26.25, 'longitude': 88.11}, 'metaData': {'detect': 1, 'frameAnomalyScore': 31.990000000000006, 'count': {'peopleCount': 1, 'vehicleCount': 0, 'ObjectCount': 0}, 'anomalyIds': [], 'cid': 'QmdbLLjUfi5mBoQBxTerYDwEqvq6tS9sbf7wW5szSBxdTM', 'object': [{'class': 'Person', 'detectionScore': 37.388000000000005, 'activityScore': 10.0, 'track': '100', 'id': '4', 'memDID': None, 'activity': 'Standing', 'detectTime': '2023-06-19 10:37:20.284136+05:30', 'cids': 'Qmd2FwoTXZTVNPk2xj1iYiX5MyEaX1SYafwXGhBBZUkFfz'}]}, 'tenantId': 'e410d5e8-9da4-4144-9dae-78066a71be8b', 'version': 'v0.0.3'}
# dbpush_members(mem_dict)