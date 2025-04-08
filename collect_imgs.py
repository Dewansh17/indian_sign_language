import os

import cv2


DATA_DIR = './newdata'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10
dataset_size = 100

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Z" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('z'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()


# import os
# import cv2

# # Configuration
# DATA_DIR = './data'  # Directory where static images are stored
# SEQUENCE_LENGTH = 30  # Number of frames per motion sequence
# DATASET_SIZE = 100    # Number of sequences per motion gesture
# MOTION_GESTURES = ["hello", "thank_you", "goodbye"]  # Add your motion gestures here

# # Create data directory if it doesn't exist
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# for gesture in MOTION_GESTURES:
#     gesture_dir = os.path.join(DATA_DIR, gesture)
#     os.makedirs(gesture_dir, exist_ok=True)

#     print(f'Collecting data for motion gesture: {gesture}')

#     # Wait for 'Z' key press to start capturing
#     while True:
#         ret, frame = cap.read()
#         cv2.putText(frame, f'Ready for "{gesture}"? Press "Z"!', (50, 50), 
#                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(25) == ord('z'):
#             break

#     # Capture sequences for the motion gesture
#     for seq_num in range(DATASET_SIZE):
#         seq_dir = os.path.join(gesture_dir, str(seq_num))
#         os.makedirs(seq_dir, exist_ok=True)

#         print(f'Capturing sequence {seq_num + 1}/{DATASET_SIZE} for "{gesture}"')

#         # Capture SEQUENCE_LENGTH frames
#         for frame_num in range(SEQUENCE_LENGTH):
#             ret, frame = cap.read()
#             cv2.imwrite(os.path.join(seq_dir, f'{frame_num}.jpg'), frame)
#             cv2.imshow('frame', frame)
#             cv2.waitKey(25)  # Adjust delay for smoother capture

# cap.release()
# cv2.destroyAllWindows()