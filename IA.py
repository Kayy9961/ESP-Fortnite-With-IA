import cv2
import numpy as np
import pyautogui
import keyboard
import mediapipe as mp
import time
import threading

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)  

sensitivity = 0.1  

frame_duration = 1 / 60  

cv2.setUseOptimized(True)

screen = None

def capture_screen():
    global screen
    while True:
        screenshot = pyautogui.screenshot()
        screen = np.array(screenshot)

def process_frame():
    global screen
    while True:
        if screen is not None:
            start_time = time.time()

            screen_height, screen_width = screen.shape[:2]
            circle_center = (screen_width // 2, screen_height // 2)
            circle_radius = 250  

            screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            screen_bgr = cv2.circle(screen_bgr, circle_center, circle_radius, (255, 255, 255), 2)
            screen_rgb = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(screen_rgb)

            if results.pose_landmarks:
                x_min, y_min = float('inf'), float('inf')
                x_max, y_max = float('-inf'), float('-inf')

                for landmark in results.pose_landmarks.landmark:
                    x = int(landmark.x * screen.shape[1])
                    y = int(landmark.y * screen.shape[0])
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

               
                head_top_extension = 20  
                y_min = max(0, y_min - head_top_extension)

          
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                neck = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] 

               
                left_shoulder_point = (int(left_shoulder.x * screen.shape[1]), int(left_shoulder.y * screen.shape[0]))
                right_shoulder_point = (int(right_shoulder.x * screen.shape[1]), int(right_shoulder.y * screen.shape[0]))
                left_hip_point = (int(left_hip.x * screen.shape[1]), int(left_hip.y * screen.shape[0]))
                right_hip_point = (int(right_hip.x * screen.shape[1]), int(right_hip.y * screen.shape[0]))
                left_ankle_point = (int(left_ankle.x * screen.shape[1]), int(left_ankle.y * screen.shape[0]))
                right_ankle_point = (int(right_ankle.x * screen.shape[1]), int(right_ankle.y * screen.shape[0]))
                left_wrist_point = (int(left_wrist.x * screen.shape[1]), int(left_wrist.y * screen.shape[0]))
                right_wrist_point = (int(right_wrist.x * screen.shape[1]), int(right_wrist.y * screen.shape[0]))
                neck_point = (int(neck.x * screen.shape[1]), int(neck.y * screen.shape[0]))

                
                mid_shoulder_point = (
                    (left_shoulder_point[0] + right_shoulder_point[0]) // 2,
                    (left_shoulder_point[1] + right_shoulder_point[1]) // 2
                )

                screen_bgr = cv2.line(screen_bgr, left_shoulder_point, right_shoulder_point, (0, 0, 255), 2)
                screen_bgr = cv2.line(screen_bgr, mid_shoulder_point, left_ankle_point, (0, 0, 255), 2)
                screen_bgr = cv2.line(screen_bgr, mid_shoulder_point, right_ankle_point, (0, 0, 255), 2)
                screen_bgr = cv2.line(screen_bgr, neck_point, mid_shoulder_point, (0, 0, 255), 2)
                screen_bgr = cv2.line(screen_bgr, left_shoulder_point, left_wrist_point, (0, 0, 255), 2)
                screen_bgr = cv2.line(screen_bgr, right_shoulder_point, right_wrist_point, (0, 0, 255), 2)

                
                if x_min < x_max and y_min < y_max: 
                    screen_bgr = cv2.rectangle(screen_bgr, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                distance_from_center = np.sqrt((center_x - circle_center[0]) ** 2 + (center_y - circle_center[1]) ** 2)

                if distance_from_center <= circle_radius:
                    if keyboard.is_pressed("t"):
                        delta_x = center_x - circle_center[0]
                        delta_y = center_y - circle_center[1]
                        move_x = int(delta_x * sensitivity)
                        move_y = int(delta_y * sensitivity)
                        pyautogui.moveRel(move_x, move_y)

            cv2.imshow("Detección en tiempo real", screen_bgr)

            elapsed_time = time.time() - start_time
            time_to_wait = max(int((frame_duration - elapsed_time) * 1000), 1) 
            if cv2.waitKey(time_to_wait) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

t1 = threading.Thread(target=capture_screen)
t2 = threading.Thread(target=process_frame)

t1.start()
t2.start()

t1.join()
t2.join()
