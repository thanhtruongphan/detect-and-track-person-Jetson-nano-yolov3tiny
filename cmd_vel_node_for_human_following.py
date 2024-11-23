#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
from collections import deque   
from geometry_msgs.msg import Twist
from pylibfreenect2 import Freenect2, OpenGLPacketPipeline, FrameType, SyncMultiFrameListener

class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower')
        
        # Parameters
        self.target_distance = 1.0  # Target distance in meters
        self.max_linear_speed = 0.5  # Maximum forward/backward speed
        self.max_angular_speed = 0.5  # Maximum rotation speed
        self.distance_tolerance = 0.1  # Acceptable distance variation
        
        # ROS2 publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Kinect setup
        self.setup_kinect()
        
        # Person detector
        self.detector = self.initialize_detector()
        
    def initialize_detector(self):
        return PersonDetector()
    
    def setup_kinect(self):
        freenect = Freenect2()
        pipeline = OpenGLPacketPipeline()
        self.device = freenect.openDefaultDevice(pipeline)
        
        self.listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()
    
    def calculate_distance(self, depth_image, x, y, w, h):
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
    
    def calculate_speed_commands(self, distance, box_center, image_width):
        twist = Twist()
        
        if distance is not None:
            distance_error = distance - self.target_distance
            
            if abs(distance_error) > self.distance_tolerance:
                linear_speed = np.clip(distance_error, -self.max_linear_speed, self.max_linear_speed)
                twist.linear.x = linear_speed
            
            image_center = image_width / 2
            angular_error = (box_center[0] - image_center) / image_width
            twist.angular.z = np.clip(angular_error, -self.max_angular_speed, self.max_angular_speed)
        
        return twist
    
    def run(self):
        while rclpy.ok():
            try:
                frames = self.listener.waitForNewFrame()
                
                color_frame = frames["color"]
                color_image = cv2.cvtColor(color_frame.asarray(np.uint8), cv2.COLOR_RGBA2BGR)
                color_image = cv2.resize(color_image, (640, 480))
                
                depth_frame = frames["depth"]
                depth_image = depth_frame.asarray(np.float32)
                depth_image = cv2.resize(depth_image, (640, 480))
                
                boxes, confidences, indices, fps = self.detector.detect(color_image)
                
                if len(indices) > 0:
                    i = indices[0]
                    box = boxes[i]
                    x, y, w, h = box
                    
                    box_center = (x + w/2, y + h/2)
                    distance = self.calculate_distance(depth_image, x, y, w, h)
                    
                    if distance is not None:
                        twist = self.calculate_speed_commands(distance, box_center, color_image.shape[1])
                        self.cmd_vel_pub.publish(twist)
                
                self.listener.release(frames)
                
            except Exception as e:
                self.get_logger().error(f"Error in person following: {str(e)}")
            
            rclpy.spin_once(self)

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

def main(args=None):
    rclpy.init(args=args)
    follower = PersonFollower()
    
    try:
        follower.run()
    except KeyboardInterrupt:
        pass
    finally:
        follower.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()