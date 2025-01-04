import cv2
import numpy as np
import time
import json
from collections import deque
from pylibfreenect2 import Freenect2, OpenGLPacketPipeline, FrameType, SyncMultiFrameListener

class PersonDetector:
    def __init__(self, input_width=320, input_height=320):
        self.input_width = input_width
        self.input_height = input_height
        self.skip_frames = 2
        self.frame_count = 0
        
        # Set the specific HSV range
        self.lower_hsv = np.array([30, 50, 100])
        self.upper_hsv = np.array([90, 255, 255])
        
        self.net = cv2.dnn.readNet(
            "yolov3-tiny.weights",
            "yolov3-tiny.cfg"
        )
        
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.fps_counter = deque(maxlen=10)
        self.last_detection = None

    def check_color_in_roi(self, hsv_frame, x, y, w, h):
        """Check if target color exists in region of interest"""
        roi = hsv_frame[y:y+h, x:x+w]
        mask = cv2.inRange(roi, self.lower_hsv, self.upper_hsv)
        color_ratio = np.sum(mask > 0) / (w * h)
        return color_ratio > 0.1  # Return True if more than 10% of ROI has target color

    def detect(self, frame, hsv_frame):
        self.frame_count += 1
        height, width = frame.shape[:2]
        
        if self.frame_count % self.skip_frames != 0 and self.last_detection is not None:
            return self.last_detection
            
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
        has_target_color = []
        
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
                    
                    # Check if person has target color
                    color_match = self.check_color_in_roi(hsv_frame, x, y, w, h)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    has_target_color.append(color_match)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        
        process_time = time.time() - start_time
        self.fps_counter.append(1 / process_time if process_time > 0 else 0)
        current_fps = sum(self.fps_counter) / len(self.fps_counter)
        
        self.last_detection = (boxes, confidences, indices, has_target_color, current_fps)
        return self.last_detection

def calculate_distance(depth_image, x, y, w, h):
    center_x = x + w // 2
    center_y = y + h // 2
    
    if (center_x >= depth_image.shape[1] or center_y >= depth_image.shape[0] or
        center_x < 0 or center_y < 0):
        return None
    
    x_start = max(0, center_x - 1)
    x_end = min(depth_image.shape[1], center_x + 2)
    y_start = max(0, center_y - 1)
    y_end = min(depth_image.shape[0], center_y + 2)
    
    region = depth_image[y_start:y_end, x_start:x_end]
    valid_depths = region[(region > 0) & (region < 10000)]
    
    if len(valid_depths) > 0:
        return np.median(valid_depths) / 1000.0
    return None

def main():
    # Create shared folder path if it doesn't exist
    import os
    shared_folder = '/home/jetson/Documents/human_detection_yolo3tiny/tmp'
    os.makedirs(shared_folder, exist_ok=True)
    detection_file_path = os.path.join(shared_folder, 'person_detection.json')
    
    freenect = Freenect2()
    pipeline = OpenGLPacketPipeline()
    device = freenect.openDefaultDevice(pipeline)
    
    listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)
    device.start()
    
    detector = PersonDetector()
    
    print("Starting detection... Press 'q' to quit.")
    print(f"Using HSV range: H({detector.lower_hsv[0]}-{detector.upper_hsv[0]}), "
          f"S({detector.lower_hsv[1]}-{detector.upper_hsv[1]}), "
          f"V({detector.lower_hsv[2]}-{detector.upper_hsv[2]})")
    
    try:
        while True:
            frames = listener.waitForNewFrame()
            
            color_frame = frames["color"]
            color_image = cv2.cvtColor(color_frame.asarray(np.uint8), cv2.COLOR_RGBA2BGR)
            color_image = cv2.resize(color_image, (640, 480))
            color_image = cv2.flip(color_image, 1)
            
            # Convert to HSV
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            
            depth_frame = frames["depth"]
            depth_image = depth_frame.asarray(np.float32)
            depth_image = cv2.resize(depth_image, (640, 480))
            depth_image = cv2.flip(depth_image, 1)
            
            boxes, confidences, indices, has_target_color, fps = detector.detect(color_image, hsv_image)
            
            # Create HSV mask for visualization
            hsv_mask = cv2.inRange(hsv_image, detector.lower_hsv, detector.upper_hsv)
            hsv_result = cv2.bitwise_and(color_image, color_image, mask=hsv_mask)
            
            # Initialize detection data with default values
            detection_data = {
                "timestamp": time.time(),
                "detected": False,
                "distance": None,
                "x_center": 0.0
            }
            
            # Find the closest person with target color
            if len(indices) > 0:
                min_distance = float('inf')
                best_detection = None
                
                for i, idx in enumerate(indices):
                    if has_target_color[idx]:
                        box = boxes[idx]
                        x, y, w, h = box
                        distance = calculate_distance(depth_image, x, y, w, h)
                        
                        if distance is not None and distance < min_distance:
                            min_distance = distance
                            # Normalize x_center to [-1, 1] range
                            x_center = ((x + w/2) - 320) / 320
                            best_detection = {
                                "distance": distance,
                                "x_center": x_center
                            }
                            
                            # Draw rectangle for the detected person
                            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f'Person {distance:.1f}m'
                            cv2.putText(color_image, label, (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if best_detection is not None:
                    detection_data.update({
                        "detected": True,
                        "distance": best_detection["distance"],
                        "x_center": best_detection["x_center"]
                    })
            
            # Save detection data
            with open(detection_file_path, 'w') as f:
                json.dump(detection_data, f)
            
            cv2.putText(color_image, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show both original and HSV filtered images
            cv2.imshow("Kinect v2 Person Detection", color_image)
            cv2.imshow("HSV Filter", hsv_result)
            
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
