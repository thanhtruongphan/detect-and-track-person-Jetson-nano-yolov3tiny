# works well, about 4-8 FPS, 
# INFO: YOLO v3tiny, Kinect v2 camera, and HSV for detect person with distance

import cv2
import numpy as np
import time
from collections import deque
from pylibfreenect2 import Freenect2, OpenGLPacketPipeline, FrameType, SyncMultiFrameListener

class PersonDetector:
    def __init__(self, input_width=320, input_height=320):  # Significantly reduced input size
                                                            # usually is 416x416, and can reduce to 192x192 or 160x160...
        self.input_width = input_width
        self.input_height = input_height
        self.skip_frames = 2  # Process every nth frame (change to 3, 4... if FPS so low)
        self.frame_count = 0
        
        # Initialize network
        self.net = cv2.dnn.readNet(
            "yolov3-tiny.weights",
            "yolov3-tiny.cfg"
        )
        
        # Use CUDA if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.fps_counter = deque(maxlen=10)  # Reduced size for more recent FPS calculation
        self.last_detection = None  # Store last detection results

    def detect(self, frame):
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        # Skip frames to improve performance
        if self.frame_count % self.skip_frames != 0 and self.last_detection is not None:
            return self.last_detection
            
        start_time = time.time()
        
        # Create blob
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
        
        # Process detections
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.4 and class_id == 0:  # Person class
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = max(0, int(center_x - w/2))
                    y = max(0, int(center_y - h/2))
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        
        process_time = time.time() - start_time
        self.fps_counter.append(1 / process_time if process_time > 0 else 0)
        current_fps = sum(self.fps_counter) / len(self.fps_counter)
        
        self.last_detection = (boxes, confidences, indices, current_fps)
        return self.last_detection

def calculate_distance(depth_image, x, y, w, h):
    """Optimized distance calculation with smaller kernel"""
    # Get center point only
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Ensure coordinates are within bounds
    if (center_x >= depth_image.shape[1] or center_y >= depth_image.shape[0] or
        center_x < 0 or center_y < 0):
        return None
    
    # Use a small 3x3 kernel around the center point
    x_start = max(0, center_x - 1)
    x_end = min(depth_image.shape[1], center_x + 2)
    y_start = max(0, center_y - 1)
    y_end = min(depth_image.shape[0], center_y + 2)
    
    # Get depth region
    region = depth_image[y_start:y_end, x_start:x_end]
    valid_depths = region[(region > 0) & (region < 10000)]  # Filter invalid values
    
    if len(valid_depths) > 0:
        return np.median(valid_depths) / 1000.0  # Convert to meters
    return None

def main():
    # Initialize Kinect
    freenect = Freenect2()
    pipeline = OpenGLPacketPipeline()
    device = freenect.openDefaultDevice(pipeline)
    
    listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)
    device.start()
    
    # Initialize detector
    detector = PersonDetector()
    
    print("Starting detection... Press 'q' to quit.")
    
    try:
        while True:
            frames = listener.waitForNewFrame()
            
            # Process color frame with reduced resolution
            color_frame = frames["color"]
            color_image = cv2.cvtColor(color_frame.asarray(np.uint8), cv2.COLOR_RGBA2BGR)
            color_image = cv2.resize(color_image, (640, 480))  # Fixed smaller resolution
            
            # Get depth frame with matching resolution
            depth_frame = frames["depth"]
            depth_image = depth_frame.asarray(np.float32)
            depth_image = cv2.resize(depth_image, (640, 480))
            
            # Run detection
            boxes, confidences, indices, fps = detector.detect(color_image)
            
            # Process detections
            if len(indices) > 0:
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
                        label += f' - {distance:.1f}m'
                    
                    cv2.putText(color_image, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Display FPS
            cv2.putText(color_image, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Kinect v2 Person Detection", color_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            listener.release(frames)
        
        device.stop()
        device.close()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()