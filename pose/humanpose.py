import cv2
import numpy as np
import matplotlib.pyplot as plt

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

# how the body parts are connected 
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Input size
input_width = 368
input_height = 368

# Load the pre-trained pose estimation model
model_path = "graph_opt.pb"  
net = cv2.dnn.readNetFromTensorflow(model_path)

#for  keypoints detecting
confidence_threshold = 0.2


def detect_pose(frame, threshold=confidence_threshold):
    # dimensions 
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # Prepare the frame for input to the model
    net.setInput(cv2.dnn.blobFromImage(
        frame, 1.0, (input_width, input_height), (127.5, 127.5, 127.5), swapRB=True, crop=False
    ))

    # Perform a forward pass through the model
    output = net.forward()
    output = output[:, :19, :, :]  # Consider only the first 19 keypoints

    # List to store the detected keypoints
    keypoints = []

    # Loop through all body parts to extract keypoints
    for i in range(len(BODY_PARTS)):
        heat_map = output[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heat_map)

        x = (frame_width * point[0]) / output.shape[3]
        y = (frame_height * point[1]) / output.shape[2]

        keypoints.append((int(x), int(y)) if conf > threshold else None)

    # for Drawing connections and keypoints on the frame
    for pair in POSE_PAIRS:
        part_from = pair[0]
        part_to = pair[1]

        id_from = BODY_PARTS[part_from]
        id_to = BODY_PARTS[part_to]

        if keypoints[id_from] and keypoints[id_to]:
            cv2.line(frame, keypoints[id_from], keypoints[id_to], (0, 255, 0), 3)
            cv2.circle(frame, keypoints[id_from], 4, (255, 0, 0), -1)
            cv2.circle(frame, keypoints[id_to], 4, (255, 0, 0), -1)

    # to Display the model's inference time
    t, _ = net.getPerfProfile()
    inference_time = t * 1000.0 / cv2.getTickFrequency()
    cv2.putText(
        frame, f"Inference Time: {inference_time:.2f} ms", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
    )

    return frame

# for displaying an image using Matplotlib
def show_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


input_source = "stand.jpg"  

try:
    if isinstance(input_source, str): 
        frame = cv2.imread(input_source)
        if frame is None:
            raise FileNotFoundError(f"Image file '{input_source}' not found.")
        output_frame = detect_pose(frame)
        show_image(output_frame)
        cv2.imwrite("pose_output.png", output_frame)
        print("Pose estimation completed. Output saved as 'pose_output.png'.")
    else: 
        cap = cv2.VideoCapture(input_source)
        print("Press 'q' to exit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output_frame = detect_pose(frame)
            cv2.imshow("Pose Estimation", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {e}")
