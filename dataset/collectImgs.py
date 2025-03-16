import os  # Importing the OS module for directory handling
import cv2  # Importing OpenCV for image capturing

# Define the directory where the dataset will be stored
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):  # Check if the directory exists
    os.makedirs(DATA_DIR)  # Create the directory if it doesn't exist

# Define the number of classes (different sign language gestures) and dataset size per class
number_of_classes = 38  # Total number of different signs to be captured
dataset_size = 100  # Number of images per sign

# Open the webcam for capturing images
cap = cv2.VideoCapture(0)

# Loop through each class to collect images
for j in range(number_of_classes):
    # Create a directory for the current class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for user input before starting to collect data
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            break  # If no frame is captured, exit loop

        # Display a message on the frame to inform the user
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Show the frame
        
        # Wait for the user to press 'q' to start capturing images
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0  # Initialize image counter
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a frame
        if not ret:
            break
        
        cv2.imshow('frame', frame)  # Display the frame
        cv2.waitKey(25)  # Wait briefly before capturing the next frame
        
        # Save the captured frame as an image in the respective class folder
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)
        
        counter += 1  # Increment the counter

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
