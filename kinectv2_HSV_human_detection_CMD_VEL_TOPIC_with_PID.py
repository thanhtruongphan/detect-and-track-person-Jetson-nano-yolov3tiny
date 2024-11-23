import cv2
import numpy as np
import time
from collections import deque
from pylibfreenect2 import Freenect2, OpenGLPacketPipeline, FrameType, SyncMultiFrameListener
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class PIDController:
    def __init__(self, kp=0.3, ki=0.1, kd=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        
    def compute(self, error, max_output=0.5):
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Avoid division by zero
        if dt <= 0:
            return 0
            
        # Calculate PID terms
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        
        # Calculate output
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        # Update states
        self.last_error = error
        self.last_time = current_time
        
        # Clamp output
        return np.clip(output, -max_output, max_output)

class PersonDetector:
    def __init__(self, input_width=320, input_height=320):
        self.input_width = input_width
        self.input_height = input_height
        self.skip_frames = 2
        self.frame_count = 0
        
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

    def detect(self, frame):
        # ... (keep the existing detect method unchanged)
        # This is the same as in your working code
        pass

class RobotController(Node):
    def __init__(self):
        super().__init__('kinect_robot_controller')
        
        # Create QoS profile for cmd_vel
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create publisher for robot velocity commands
        self.vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            qos_profile
        )
        
        # Initialize PID controller
        self.pid_controller = PIDController()
        
        # Target distance in meters
        self.target_distance = 1.0
        
        # Safety thresholds
        self.min_distance = 0.5  # Minimum safe distance
        self.max_distance = 3.0  # Maximum tracking distance
        
    def calculate_velocity(self, current_distance):
        if current_distance is None:
            # If no person detected, stop the robot
            return self.publish_velocity(0.0)
            
        # Ensure distance is within safe bounds
        if current_distance < self.min_distance:
            return self.publish_velocity(-0.2)  # Back up slowly
        elif current_distance > self.max_distance:
            return self.publish_velocity(0.0)  # Stop if person too far
            
        # Calculate error (positive error means we need to move forward)
        error = current_distance - self.target_distance
        
        # Get velocity from PID controller
        velocity = self.pid_controller.compute(error)
        
        # Publish velocity command
        self.publish_velocity(velocity)
        
        return velocity
        
    def publish_velocity(self, linear_velocity):
        msg = Twist()
        msg.linear.x = linear_velocity
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        
        self.vel_publisher.publish(msg)
        return linear_velocity

def main():
    # Initialize ROS2
    rclpy.init()
    robot_controller = RobotController()
    
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
    
    print("Starting detection and control... Press 'q' to quit.")
    
    try:
        while rclpy.ok():
            frames = listener.waitForNewFrame()
            
            # Process color frame
            color_frame = frames["color"]
            color_image = cv2.cvtColor(color_frame.asarray(np.uint8), cv2.COLOR_RGBA2BGR)
            color_image = cv2.resize(color_image, (640, 480))
            
            # Get depth frame
            depth_frame = frames["depth"]
            depth_image = depth_frame.asarray(np.float32)
            depth_image = cv2.resize(depth_image, (640, 480))
            
            # Run detection
            boxes, confidences, indices, fps = detector.detect(color_image)
            
            # Variables for closest person
            min_distance = float('inf')
            closest_box = None
            closest_confidence = None
            
            # Process detections
            if len(indices) > 0:
                for i in indices:
                    box = boxes[i]
                    x, y, w, h = box
                    confidence = confidences[i]
                    
                    # Calculate distance
                    distance = calculate_distance(depth_image, x, y, w, h)
                    
                    # Update closest person
                    if distance is not None and distance < min_distance:
                        min_distance = distance
                        closest_box = box
                        closest_confidence = confidence
                    
                    # Draw bounding box
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw labels
                    label = f'Person {confidence:.2f}'
                    if distance is not None:
                        label += f' - {distance:.1f}m'
                    
                    cv2.putText(color_image, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Calculate and publish velocity command for closest person
            if closest_box is not None:
                velocity = robot_controller.calculate_velocity(min_distance)
                
                # Draw velocity info
                cv2.putText(color_image, f'Velocity: {velocity:.2f} m/s', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # No person detected, stop the robot
                robot_controller.calculate_velocity(None)
            
            # Display FPS
            cv2.putText(color_image, f'FPS: {fps:.1f}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Kinect v2 Person Detection", color_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            listener.release(frames)
            
            # Process ROS callbacks
            rclpy.spin_once(robot_controller, timeout_sec=0)
        
        # Cleanup
        device.stop()
        device.close()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        rclpy.shutdown()

if __name__ == "__main__":
    main()