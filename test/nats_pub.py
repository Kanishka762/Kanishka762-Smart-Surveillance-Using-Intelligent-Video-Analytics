import asyncio
import nats
from nats.errors import ConnectionClosedError, TimeoutError, NoServersError
import json
from nanoid import generate
import time

# activity_out = {'type': 'activity', 'deviceid': '40a33680-e95a-11ed-9138-29fd70751e73', 'batchid': '7fdba220-b1b7-4da2-b35a-e7cc74e7dab4', 'timestamp': '2023-05-02 16:21:14.384387', 'geo': {'latitude': 12.918747, 'longitude': 77.431558}, 'metaData': {'detect': 2, 'frameAnomalyScore': [0, 0, 25.12, 25.79, 26.95, 27.329999999999995, 26.660000000000004, 25.51, 24.04, 0, 0, 0, 24.29, 30.160000000000004, 29.339999999999996, 27.625, 28.7, 27.610000000000003, 28.1, 31.5, 30.76, 29.464999999999996, 27.329999999999995, 26.419999999999998, 27.43, 29.080000000000002, 32.75, 33.25, 29.920000000000005, 28.080000000000002], 'count': {'peopleCount': 2, 'vehicleCount': 0, 'ObjectCount': 0}, 'anamolyIds': [], 'cid': 'QmXbCZHPHeuYhwv4JWrGXzkCTLfa4H6aSqykfK52pvLnqu', 'object': [{'class': 'Person', 'detectionScore': 29.899411764705885, 'activityScore': 10.0, 'track': '100', 'id': '4.0', 'activity': ['walking'], 'detectTime': '', 'cids': 'Qmd6EZKo743ciyAg3dndmCUp3a7HxCBVDJvM9cHwF69cvk'}, {'class': 'Person', 'detectionScore': 27.54181818181818, 'activityScore': 10.0, 'track': '100', 'id': '5.0', 'activity': ['walking', 'carrying'], 'detectTime': '', 'cids': 'QmSxBtCBcpLvfeqWfSTR9ydR6TDZCRVRZLNAGdeG75D4J7'}]}, 'memory': '4.663916015625 GB', 'version': 'v0.0.3', 'tenantid': 'a7c4c832-249b-4a81-93da-21f56708f484'}


async def main(i):
    # It is very likely that the demo server will see traffic from clients other than yours.
    # To avoid this, start your own locally and modify the example to use it.
    nc = await nats.connect("nats://216.48.181.154:5222")
    
    # js = nc.jetstream()
    # device_data = {
    #     "username": "test",
    #     "password": "test123456789",
    #     # "ip": "streams.ckdr.co.in",
    #     "port": "5554",
    #     "ddns": "streams.ckdr.co.in",
    #     "rtsp": "rtsp://216.48.184.201:8554//stream1",
    #     "videoEncodingInformation": "MP4",
    #     # "deviceId": "40a33680-e95a-11ed-9138-29fd70751e73",
    #     "deviceId": str(i),
    #     "urn": "uuid:3266ee49-9fc4-d257-0e8d-17a0469a7np9",
    #     "subscriptions" : ["activity"],
    #     "lat": "26.008032",
    #     "long": "74.039402",
    #     "tenantId": "a7c4c832-249b-4a81-93da-21f56708f484"
    # }

#{'id': 'ui75LlKf6gzrfa7LuU2y27Jaqsoumya', 'member': [{'memberId': 'did:ckdr:Ee292mtxIxTyuCi6so8oPZqo4q4p9ebrV8lTJSatAdsowmya', 'type': 'known', 'faceCID': ['QmXcUS7oTCwpGJw9oaksffptH6KSEhW74GUyL1KvVhRBEP'], 'role': 'admin'}]},
    mem = [{'id': 'ui75LlKf6gzrfa7LuU2y27Jaqdarshan', 'member': [{'memberId': 'did:ckdr:Ee292mtxIxTyuCi6so8oPZqo4q4p9ebrV8lTJSatAdarshan', 'type': 'unknown', 'faceCID': ['QmcfXhgV3tYKzPD3Ae6V1XDDVyP87QZhRPDqXq1mirbYyA'], 'role': 'admin'}]},{'id': 'ui75LlKf6gzrfa7LuU2y27Jaqsachin', 'member': [{'memberId': 'did:ckdr:Ee292mtxIxTyuCi6so8oPZqo4q4p9ebrV8lTJSatAdsachin', 'type': 'known', 'faceCID': ['QmVLtwDMJ7AaiNayrKtoe3FLPtsSspruimAMAzvSx7TBXB'], 'role': 'admin'}]},{'id': 'ui75LlKf6gzrfa7LuU2y27Jaqnivetheni', 'member': [{'memberId': 'did:ckdr:Ee292mtxIxTyuCi6so8oPZqo4q4p9ebrV8lTJSatAdnivetheni', 'type': 'unknown', 'faceCID': ['QmVnynu14R4x19eGsSR2ApsuhL2hdzao7m81J3fDy4axBY'], 'role': 'admin'}]},{'id': 'ui75LlKf6gzrfa7LuU2y27Jaqnisrihari', 'member': [{'memberId': 'did:ckdr:Ee292mtxIxTyuCi6so8oPZqo4q4p9ebrV8lTJSatAdsrihari', 'type': 'known', 'faceCID': ['QmbrkxKiwRJVfjxxnZfJ4vSf5DGWRJFQmC4RVmBicHj7jf'], 'role': 'admin'}]},{'id': 'ui75LlKf6gzrfa7LuU2y27Jaqnizahaan', 'member': [{'memberId': 'did:ckdr:Ee292mtxIxTyuCi6so8oPZqo4q4p9ebrV8lTJSatAdzahaan', 'type': 'known', 'faceCID': ['QmV56SYwt5eu2kNbRqz59zjNuXnTSWmTEHhzwBWTo6Z5F9'], 'role': 'admin'}]},{'id': 'ui75LlKf6gzrfa7LuU2y27Jaqnizbarath', 'member': [{'memberId': 'did:ckdr:Ee292mtxIxTyuCi6so8oPZqo4q4p9ebrV8lTJSatAdbarath', 'type': 'unknown', 'faceCID': ['QmTEz8BekEmDsohkGzVMYPpfvmPYDyQf6LfKFYA62pRbWA'], 'role': 'admin'}]},{'id': 'ui75LlKf6gzrfa7LuU2y27Jaqnizkrishani', 'member': [{'memberId': 'did:ckdr:Ee292mtxIxTyuCi6so8oPZqo4q4p9ebrV8lTJSatAkrishani', 'type': 'unknown', 'faceCID': ['QmPsaDnQvyekrfUSHQVMkz9QYEkw1EsBBZD81poJmUfDDv'], 'role': 'admin'}]},{'id': 'ui75LlKf6gzrfa7LuU2y27Jaqnizkrishabd', 'member': [{'memberId': 'did:ckdr:Ee292mtxIxTyuCi6so8oPZqo4q4p9ebrV8lTJSatAkrishabd', 'type': 'unknown', 'faceCID': ['QmNoevhBZRNLvbeQntbKxqv47TjPa3uSFDHd3rXPTQKLhp'], 'role': 'admin'}]}]
    for memberd_ata in mem:
        memberd_ata = {'id': 'ui75LlKf6gzrfa7LuU2y27Jaqnizahaan', 'member': [{'memberId': 'did:ckdr:Ee292mtxIxTyuCi6so8oPZqo4q4p9ebrV8lTJSatAdzahaan', 'type': 'known', 'faceCID': ['QmV56SYwt5eu2kNbRqz59zjNuXnTSWmTEHhzwBWTo6Z5F9'], 'role': 'admin'}]}
        JSONEncoder = json.dumps(memberd_ata)
        json_encoded = JSONEncoder.encode()
        Subject = "service.member_update"
        # Subject = "service.device_discovery"
        ack = await nc.request(Subject, json_encoded)
        time.sleep(5)
        print("Member is getting published")
        

        # Terminate connection to NATS.
    await nc.drain()

if __name__ == '__main__':
    for i in range(1,6):
        asyncio.run(main(i))