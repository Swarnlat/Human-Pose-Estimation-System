# import streamlit as st

# from PIL import Image
# import numpy as np
# import cv2

# DEMO_IMAGE = 'stand.jpg'

# BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
#                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

# POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]


# width = 368
# height = 368
# inWidth = width
# inHeight = height

# net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")




# st.title("Human Pose Estimation OpenCV")

# st.text('Make Sure you have a clear image with all the parts clearly visible')

# img_file_buffer = st.file_uploader("Upload an image, Make sure you have a clear image", type=[ "jpg", "jpeg",'png'])

# if img_file_buffer is not None:
#     image = np.array(Image.open(img_file_buffer))

# else:
#     demo_image = DEMO_IMAGE
#     image = np.array(Image.open(demo_image))
    
# st.subheader('Original Image')
# st.image(
#     image, caption=f"Original Image", use_column_width=True
# ) 

# thres = st.slider('Threshold for detecting the key points',min_value = 0,value = 20, max_value = 100,step = 5)

# thres = thres/100

# @st.cache
# def poseDetector(frame):
#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]
    
#     net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
#     out = net.forward()
#     out = out[:, :19, :, :]
    
#     assert(len(BODY_PARTS) == out.shape[1])
    
#     points = []
#     for i in range(len(BODY_PARTS)):
#         # Slice heatmap of corresponging body's part.
#         heatMap = out[0, i, :, :]

#         _, conf, _, point = cv2.minMaxLoc(heatMap)
#         x = (frameWidth * point[0]) / out.shape[3]
#         y = (frameHeight * point[1]) / out.shape[2]
#         points.append((int(x), int(y)) if conf > thres else None)
        
        
#     for pair in POSE_PAIRS:
#         partFrom = pair[0]
#         partTo = pair[1]
#         assert(partFrom in BODY_PARTS)
#         assert(partTo in BODY_PARTS)

#         idFrom = BODY_PARTS[partFrom]
#         idTo = BODY_PARTS[partTo]

#         if points[idFrom] and points[idTo]:
#             cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
#             cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
#             cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            
            
#     t, _ = net.getPerfProfile()
    
#     return frame


# output = poseDetector(image)


# st.subheader('Positions Estimated')
# st.image(
#        output, caption=f"Positions Estimated", use_column_width=True)
    
# st.markdown('''
#             # 
             
#             ''')

# code for ientifying the landmark
import cv2
import mediapipe as mp


# initialize mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# load the image
image_path = r'WhatsApp Image 2025-01-24 at 21.49.57_b3d51738.jpg'
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
        cv2.circle(image, (cx, cy), 3,( 255,0, 0), -4)  # coordinate = (cx,cy) ,angle = (5) , color = (255,0,0) , width = -1

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