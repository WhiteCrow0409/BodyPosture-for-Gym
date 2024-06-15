import cv2
import mediapipe as mp 
import numpy as np 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate angle between three points
def cal_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to check posture based on selected exercise
def check_posture(exercise, landmarks, mp_pose):
    if exercise == 'bicep curl':
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        angle = cal_angle(shoulder, elbow, wrist)
        if 85 <= angle <= 95:
            return "Correct starting position", (0, 255, 0)
        else:
            return "Adjust your starting position", (0, 0, 255)
    elif exercise == 'tricep curl':
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = cal_angle(shoulder, elbow, wrist)
        if 165 <= angle <= 180:
            return "Correct starting position", (0, 255, 0)
        else:
            return "Adjust your starting position", (0, 0, 255)
    elif exercise == 'deadlift':
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        angle = cal_angle(hip, knee, ankle)
        if 160 <= angle <= 180:
            return "Correct starting position", (0, 255, 0)
        else:
            return "Adjust your starting position", (0, 0, 255)
    elif exercise == 'shoulder press':
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = cal_angle(shoulder, elbow, wrist)
        if 75 <= angle <= 105:
            return "Correct starting position", (0, 255, 0)
        else:
            return "Adjust your starting position", (0, 0, 255)
    elif exercise == 'chest fly':
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        angle = cal_angle(shoulder, elbow, wrist)
        if 85 <= angle <= 95:
            return "Correct starting position", (0, 255, 0)
        else:
            return "Adjust your starting position", (0, 0, 255)
    else:
        return "Unknown exercise", (0, 0, 255)

# Setup video capture
cap = cv2.VideoCapture(0)

# Setup mediapipe instances for pose detection and drawing
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

exercise = input("Select an exercise (bicep curl, tricep curl, deadlift, shoulder press, chest fly): ").lower()

while cap.isOpened():
    ret, frame = cap.read()

    # Recoloring image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    result = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Getting the landmarks
    try:
        landmarks = result.pose_landmarks.landmark
        feedback, color = check_posture(exercise, landmarks, mp_pose)
        cv2.putText(image, feedback,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    except AttributeError:
        landmarks = None

    # Rendering detection
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    
    cv2.imshow('Mediapipe Feed', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

pose.close()  # Close the pose detection instance
cap.release()
cv2.destroyAllWindows()
