import cv2
import numpy as np
import time
from collections import deque
from pylibfreenect2 import Freenect2, OpenGLPacketPipeline, FrameType, SyncMultiFrameListener

class PersonDetector:
    def __init__(self, input_width=320, input_height=320):  # Increased input size for better detection
        self.input_width = input_width
        self.input_height = input_height
        
        # Initialize network with TensorRT optimized model
        self.net = cv2.dnn.readNet(
            "yolov3-tiny.weights",
            "yolov3-tiny.cfg"
        )
        
        # Enable CUDA backend if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print("CUDA not available, using CPU")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load COCO class names
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.fps_counter = deque(maxlen=30)

    def detect(self, frame, conf_threshold=0.4, nms_threshold=0.1):
        height, width = frame.shape[:2]
        start_time = time.time()
        
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            (self.input_width, self.input_height), 
            swapRB=True, 
            crop=False
        )
        
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf_threshold and class_id == 0:  # Person class
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = max(0, int(center_x - w/2))
                    y = max(0, int(center_y - h/2))
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        process_time = time.time() - start_time
        self.fps_counter.append(1 / process_time if process_time > 0 else 0)
        current_fps = sum(self.fps_counter) / len(self.fps_counter)
        
        return boxes, confidences, indices, current_fps

def calculate_distance(depth_image, x, y, w, h, kernel_size=5):
    """Calculate the distance to a detected person using depth data."""
    # Get center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Ensure coordinates are within bounds
    center_x = min(max(0, center_x), depth_image.shape[1] - 1)
    center_y = min(max(0, center_y), depth_image.shape[0] - 1)
    
    # Define region for averaging
    x_start = max(0, center_x - kernel_size // 2)
    x_end = min(depth_image.shape[1], center_x + kernel_size // 2 + 1)
    y_start = max(0, center_y - kernel_size // 2)
    y_end = min(depth_image.shape[0], center_y + kernel_size // 2 + 1)
    
    # Get depth region and calculate average
    region = depth_image[y_start:y_end, x_start:x_end]
    valid_region = region[np.isfinite(region) & (region > 0)]
    
    if valid_region.size > 0:
        # Convert to meters and apply median filter to remove outliers
        distance = np.median(valid_region) / 1000.0  # Convert to meters
        return distance
    return None

def main():
    try:
        # Initialize Kinect
        freenect = Freenect2()
        if freenect.enumerateDevices() == 0:
            raise RuntimeError("No Kinect device found!")
        
        try:
            pipeline = OpenGLPacketPipeline()
            print("Using OpenGLPacketPipeline")
        except Exception:
            from pylibfreenect2 import CudaPacketPipeline
            pipeline = CudaPacketPipeline()
            print("Using CudaPacketPipeline")
        
        device = freenect.openDefaultDevice(pipeline)
        listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
        
        device.setColorFrameListener(listener)
        device.setIrAndDepthFrameListener(listener)
        device.start()
        
        # Initialize detector
        detector = PersonDetector()
        
        print("Starting detection... Press 'q' to quit.")
        
        while True:
            frames = listener.waitForNewFrame()
            color_frame = frames["color"]
            depth_frame = frames["depth"]
            
            # Process color frame
            color_image = cv2.cvtColor(color_frame.asarray(np.uint8), cv2.COLOR_RGBA2BGR)
            # Scale down image for faster processing
            scale_factor = 0.5
            color_image = cv2.resize(color_image, None, fx=scale_factor, fy=scale_factor)
            
            # Get depth data
            depth_image = depth_frame.asarray(np.float32)
            depth_image = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]))
            
            # Detect persons
            boxes, confidences, indices, fps = detector.detect(color_image)
            
            # Draw detections and calculate distances
            for i in indices:
                box = boxes[i]
                x, y, w, h = box
                confidence = confidences[i]
                
                # Calculate distance
                distance = calculate_distance(depth_image, x, y, w, h)
                
                # Draw bounding box
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw labels
                label = f'Person {confidence:.2f}'
                if distance is not None:
                    label += f' d: {distance:.2f}m'
                
                cv2.putText(color_image, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Draw FPS
            cv2.putText(color_image, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display output
            cv2.imshow("Kinect v2 Person Detection", color_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            listener.release(frames)
        
        # Cleanup
        device.stop()
        device.close()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
if __name__ == "__main__":
    main()