import cv2
import numpy as np
import time
import json
import os
from collections import deque
from pylibfreenect2 import Freenect2, OpenGLPacketPipeline, FrameType, SyncMultiFrameListener

class PersonTracker:
    def __init__(self):
        self.track_id = None
        self.lost_frames = 0
        self.max_lost_frames = 30  # Frames to wait before considering target lost
        self.tracked_position = None  # (x, y, w, h)
        self.position_history = deque(maxlen=5)  # Store last 5 positions
        self.search_region_scale = 1.5  # Scale factor for search region when target is lost

    def update(self, detected_boxes, has_target_color):
        if not detected_boxes:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.track_id = None
            return None

        if self.track_id is None:
            # Initialize tracking with the first detected person with target color
            for i, box in enumerate(detected_boxes):
                if has_target_color[i]:
                    self.track_id = i
                    self.tracked_position = box
                    self.position_history.clear()
                    self.position_history.append(box)
                    self.lost_frames = 0
                    return box
        else:
            # Continue tracking existing target
            best_iou = 0
            best_box = None
            
            # Predict search region based on previous movements
            predicted_position = self._predict_position()
            search_region = self._get_search_region(predicted_position)

            for i, box in enumerate(detected_boxes):
                if has_target_color[i]:
                    if self._is_in_search_region(box, search_region):
                        iou = self._calculate_iou(predicted_position, box)
                        if iou > best_iou:
                            best_iou = iou
                            best_box = box

            if best_iou > 0.3:  # IOU threshold
                self.tracked_position = best_box
                self.position_history.append(best_box)
                self.lost_frames = 0
                return best_box
            else:
                self.lost_frames += 1
                if self.lost_frames > self.max_lost_frames:
                    self.track_id = None
                return None

    def _predict_position(self):
        if len(self.position_history) < 2:
            return self.tracked_position
        
        # Simple linear motion prediction
        last_pos = np.array(self.position_history[-1])
        prev_pos = np.array(self.position_history[-2])
        velocity = last_pos - prev_pos
        predicted = last_pos + velocity
        return predicted.tolist()

    def _get_search_region(self, box):
        if not box:
            return None
        x, y, w, h = box
        scale = self.search_region_scale + (self.lost_frames * 0.1)  # Increase search area over time
        scale = min(scale, 2.5)  # Cap maximum search area
        
        new_w = w * scale
        new_h = h * scale
        new_x = x - (new_w - w) / 2
        new_y = y - (new_h - h) / 2
        
        return [int(new_x), int(new_y), int(new_w), int(new_h)]

    def _is_in_search_region(self, box, search_region):
        if not search_region:
            return True
        
        box_center = (box[0] + box[2]/2, box[1] + box[3]/2)
        region_x1 = search_region[0]
        region_y1 = search_region[1]
        region_x2 = region_x1 + search_region[2]
        region_y2 = region_y1 + search_region[3]
        
        return (region_x1 <= box_center[0] <= region_x2 and 
                region_y1 <= box_center[1] <= region_y2)

    def _calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

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
        self.tracker = PersonTracker()

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

    def detect_and_track(self, frame, hsv_frame):
        boxes, confidences, indices, has_target_color, fps = self.detect(frame, hsv_frame)
        
        if len(indices) > 0:
            detected_boxes = [boxes[i] for i in indices]
            detected_colors = [has_target_color[i] for i in indices]
            tracked_box = self.tracker.update(detected_boxes, detected_colors)
            
            return boxes, confidences, indices, has_target_color, fps, tracked_box
        else:
            tracked_box = self.tracker.update([], [])
            return boxes, confidences, indices, has_target_color, fps, tracked_box

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
            try:
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
                
                boxes, confidences, indices, has_target_color, fps, tracked_box = detector.detect_and_track(color_image, hsv_image)
                
                # Create HSV mask for visualization
                hsv_mask = cv2.inRange(hsv_image, detector.lower_hsv, detector.upper_hsv)
                hsv_result = cv2.bitwise_and(color_image, color_image, mask=hsv_mask)
                
                # Initialize detection data
                detection_data = {
                    "timestamp": time.time(),
                    "detected": False,
                    "tracking": tracked_box is not None,
                    "distance": None,
                    "x_center": 0.0
                }
                
                # Draw tracking visualization and update detection data
                if tracked_box is not None:
                    x, y, w, h = tracked_box
                    try:
                        distance = calculate_distance(depth_image, x, y, w, h)
                    except:
                        distance = None
                    
                    x_center = ((x + w/2) - 320) / 320
                    
                    # Draw tracked person with different color
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 255), 3)
                    
                    # Safe label creation
                    if distance is not None:
                        label = f'Tracked Person {distance:.1f}m'
                    else:
                        label = 'Tracked Person (No depth)'
                        
                    cv2.putText(color_image, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    detection_data.update({
                        "detected": True,
                        "distance": distance if distance is not None else None,
                        "x_center": x_center
                    })
                
                # Draw other detections
                if len(indices) > 0:
                    for i, idx in enumerate(indices):
                        if has_target_color[idx]:
                            box = boxes[idx]
                            x, y, w, h = box
                            if tracked_box is None or box != tracked_box:
                                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Save detection data
                try:
                    with open(detection_file_path, 'w') as f:
                        json.dump(detection_data, f)
                except Exception as e:
                    print(f"Error saving detection data: {str(e)}")
                
                # Display FPS
                try:
                    cv2.putText(color_image, f'FPS: {fps:.1f}', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except:
                    pass
                
                # Show images
                cv2.imshow("Kinect v2 Person Detection", color_image)
                cv2.imshow("HSV Filter", hsv_result)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                listener.release(frames)
                
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                continue
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        device.stop()
        device.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
