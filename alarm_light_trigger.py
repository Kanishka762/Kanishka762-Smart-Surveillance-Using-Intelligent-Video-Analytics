import paho.mqtt.client as mqtt
mqtt_broker = "192.168.43.201"
mqtt_port = 1883
mqtt_user = "esp123"
mqtt_password = "Happy@1234"
publish_topic = "alert"
light_topic = "light"
def alarm():
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
    # Check if the light is on
    message = client.publish(light_topic, "is_on")
    if message.rc == 0:
        print("Light is on")
    else:
        print("Light is off")
    # Turn on the light
    message = "on"
    client.publish(light_topic, message)
    # Disconnect the client
    client.loop_stop()
    client.disconnect()
if __name__ == "__main__":
    alarm()
