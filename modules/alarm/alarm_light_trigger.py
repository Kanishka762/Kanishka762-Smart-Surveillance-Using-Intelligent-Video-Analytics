import socket
import paho.mqtt.client as mqtt

mqtt_broker = "192.168.1.6"
mqtt_port = 1883
mqtt_user = "esp123"
mqtt_password = "Happy@1234"
publish_topic = "alert"

def alarm():
    def get_local_ip():
        try:
            # Create a socket object
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Connect to any public server
            s.connect(("8.8.8.8", 80))
            
            # Get the local IP address
            local_ip = s.getsockname()[0]
            
            # Close the socket
            s.close()
            
            return local_ip
        except Exception as e:
            print("Error occurred:", e)
            return None

    def update_code_file(local_ip):
        try:
            global mqtt_broker
            mqtt_broker = local_ip
            print("mqtt_broker IP = " , mqtt_broker )
            print("IP updated successfully.")

        except Exception as e:
            print("Error occurred while updating the IP:", str(e))

    #call functions for getting and updating IP address
    local_ip = get_local_ip()
    update_code_file(local_ip)

    #For connecting to the MQTT broker
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker")
        else:
            print("Failed to connect, error code:", rc)

    def on_publish(client, userdata, mid):
        print("Message published")


    # Create an MQTT client
    client = mqtt.Client()

    # Set MQTT username and password
    client.username_pw_set(mqtt_user, mqtt_password)

    # Set callbacks
    client.on_connect = on_connect
    client.on_publish = on_publish

    # Connect to the MQTT broker
    client.connect(mqtt_broker, mqtt_port)

    # Run the MQTT client loop in the background
    client.loop_start()

    # Publish a message
    message = "alert"
    client.publish(publish_topic, message)

    # Disconnect the client
    client.loop_stop()
    client.disconnect()

# if __name__ == "__main__":
#     alarm()