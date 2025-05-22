# code for ientifying the landmark
import cv2
import mediapipe as mp


# initialize mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# load the image
image_path = r'run.jpg'
image = cv2.imread(image_path)  
image_rgb = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

# perform pose estimation 
results = pose.process(image_rgb)

# Draw landmarks only(no lines)

if results.pose_landmarks:
    print("pose landmark detected!")

    #  extract landmark data
    for idx,landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark{idx}: (x: {landmark.x} , y:{landmark.y}, z:{landmark.z})")
    for landmark in results.pose_landmarks.landmark:
        # Get image dimensions
        h , w , c = image.shape
        # convert normalized coordinates to pixel coordinates
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        # draw the keypoints
        cv2.circle(image, (cx, cy), 5,( 255,0, 0), -1)  # coordinate = (cx,cy) ,angle = (5) , color = (255,0,0) , width = -1

        # draw the landmark on the image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

# Display the output image
cv2.imshow("Pose Landmarks",image)
cv2.imshow("pose drawing",annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# release Resources
pose.close()

