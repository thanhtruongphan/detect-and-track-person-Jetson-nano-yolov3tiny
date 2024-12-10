import cv2
import numpy as np
import time
from collections import deque


class PersonDetector:
    def __init__(self, input_width=320, input_height=320):
        self.input_width = input_width
        self.input_height = input_height

        # Initialize YOLOv3-Tiny model
        self.net = cv2.dnn.readNet(
            "yolov3-tiny.weights",
            "yolov3-tiny.cfg"
        )

        # Use CUDA backend for acceleration
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Load COCO class names
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Get output layer names
        self.output_layers = self.net.getUnconnectedOutLayersNames()

        # For FPS calculation
        self.fps_counter = deque(maxlen=30)

    def detect(self, frame):
        height, width = frame.shape[:2]
        start_time = time.time()

        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            frame,
            1/255.0,
            (self.input_width, self.input_height),
            swapRB=True,
            crop=False
        )

        # Forward pass
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # Process detections
        boxes = []
        confidences = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter for person class (class_id 0) and confidence
                if confidence > 0.3 and class_id == 0:  # 0 is person class in COCO
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.05)

        # Calculate FPS
        process_time = time.time() - start_time
        self.fps_counter.append(1 / process_time if process_time > 0 else 0)
        current_fps = sum(self.fps_counter) / len(self.fps_counter)

        # Draw detections
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f'Person {confidences[i]:.2f}'
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw FPS
        cv2.putText(frame, f'FPS: {current_fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, current_fps


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


def main():
    # Initialize CSI camera (Pi Camera V2)
    pipeline = gstreamer_pipeline(flip_method=0)
    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not video_capture.isOpened():
        print("Error: Unable to open camera. Check the connection and pipeline.")
        return

    # Initialize YOLOv3 Tiny detector
    detector = PersonDetector(input_width=320, input_height=320)

    print("Starting real-time detection...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Run detection
        detected_frame, fps = detector.detect(frame)

        # Display output
        cv2.imshow("Pi Camera V2 - Person Detection", detected_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
