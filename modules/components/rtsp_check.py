import cv2

def check_rtsp_stream(rtsp_url):
    try:
        # Open the RTSP stream
        cap = cv2.VideoCapture(rtsp_url)

        # Check if the stream is opened successfully
        if not cap.isOpened():
            print(f"Error: Unable to open the RTSP stream: {rtsp_url}.")
            return False

        # Read a frame from the stream to ensure it's working
        ret, frame = cap.read()

        # Check if a frame was successfully read
        if not ret:
            print(f"Error: Unable to read frames from the RTSP stream: {rtsp_url}")
            return False

        # Release the stream and clean up
        cap.release()
        cv2.destroyAllWindows()

        # If all checks pass, the RTSP stream is working
        print("RTSP stream is working.")
        return True

    except Exception as e:
        print("Error:", e)
        return False
