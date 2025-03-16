import cv2 as cv
import numpy as np
import imutils

color = (255, 255, 255)

# Define the lower and upper boundaries for each color in HSV space
colors = {
        'blue': [np.array([101, 50, 38]), np.array([110, 255, 255])],
        'red': [np.array([160, 20, 70]), np.array([190, 255, 255])],
        'yellow': [np.array([16, 0, 99]), np.array([39, 255, 255])],
        'green': [np.array([33, 19, 105]), np.array([77, 255, 255])]}

def find_color(frame, points):
    mask = cv.inRange(frame, points[0], points[1])  # Create mask with boundaries
    cnts = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # Find contours from mask
    cnts = imutils.grab_contours(cnts)
    
    for c in cnts:
        area = cv.contourArea(c)  # Find how big the contour is
        if area > 1000:  # Only if contour is big enough, then
            M = cv.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])  # Calculate X position
                cy = int(M['m01'] / M['m00'])  # Calculate Y position
                return c, cx, cy
    return None  # Return None if no valid color is found

def check_color(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # Convert frame to HSV
    
    for name, clr in colors.items():  # For each color in colors
        result = find_color(hsv, clr)  # Call find_color function above
        if result:
            c, cx, cy = result
            cv.drawContours(frame, [c], -1, color, 3)  # Draw contours
            cv.circle(frame, (cx, cy), 7, color, -1)  # Draw circle
            cv.putText(frame, name, (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 1, color, 1)  # Put text
            
            if name == 'red':
                return 'stop'
            elif name == 'green':
                return 'keep going'
    return 'continue'  # No red or green detected, continue normally

# Capture video from the camera
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture a frame. Check the camera.")
        break
    
    action = check_color(frame)  # Check the frame for a color action
    
    if action == 'stop':
        print("Stop the car!")  
    elif action == 'keep going':
        print("Keep going!")  
    elif action == 'continue':
        print("No red or green detected. Continue normally.")
    
    cv.imshow("Frame: ", frame)  # Show the processed frame
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop when 'q' is pressed

cap.release()  # Release the camera
cv.destroyAllWindows()  # Close all windows
