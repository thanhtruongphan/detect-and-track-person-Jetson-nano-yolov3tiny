import cv2
import numpy as np
import time
from collections import deque
from pylibfreenect2 import Freenect2, OpenGLPacketPipeline, CudaPacketPipeline, FrameType, SyncMultiFrameListener


class PersonDetector:
    def __init__(self, input_width=320, input_height=320):
        self.input_width = input_width
        self.input_height = input_height
        
        # Initialize network with TensorRT optimized model
        self.net = cv2.dnn.readNet(
            "yolov3-tiny.weights",
            "yolov3-tiny.cfg"
        )
        
        # Use CUDA backend
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
        
        # Process each output layer
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter for person class (class_id 0) and confidence
                if confidence > 0.3 and class_id == 0:  # 0 is person class in COCO
                    # Scale boxes back to original image
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
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
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f'Person {confidences[i]:.2f}'
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw FPS
        cv2.putText(frame, f'FPS: {current_fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, current_fps

def main():
    # Kinect v2 initialization
    try:
        pipeline = CudaPacketPipeline()
        print("Using CudaPacketPipeline")
    except Exception:
        pipeline = OpenGLPacketPipeline()
        print("Using OpenGLPacketPipeline")
    
    freenect = Freenect2()
    if freenect.enumerateDevices() == 0:
        print("No device connected!")
        exit(1)
    
    device = freenect.openDefaultDevice(pipeline)
    listener = SyncMultiFrameListener(FrameType.Color)
    
    device.setColorFrameListener(listener)
    device.start()
    
    # Initialize YOLOv3 Tiny detector
    detector = PersonDetector(input_width=320, input_height=320)
    
    print("Starting real-time detection...")
    print("Press 'q' to quit.")
    
    while True:
        frames = listener.waitForNewFrame()
        color_frame = frames["color"]
        color_image = cv2.cvtColor(color_frame.asarray(np.uint8), cv2.COLOR_RGBA2BGR)

        # Giảm kích thước khung hình màu
        color_image = cv2.resize(color_image, (color_image.shape[1] // 3, color_image.shape[0] // 3))
        
        # Run detection
        detected_frame, fps = detector.detect(color_image)
        
        # Display output
        cv2.imshow("Kinect v2 - Person Detection", detected_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        listener.release(frames)
    
    # Clean up
    device.stop()
    device.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
