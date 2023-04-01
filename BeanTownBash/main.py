import cv2
import numpy as np

# Define lower bounds for each color band in HSV
lower_colors = [
    np.array([0, 100, 100]), # red
    np.array([36, 100, 100]), # green
    np.array([100, 100, 100]), # blue
    np.array([0, 60, 20]), # brown
    np.array([20, 100, 100]), # orange
    np.array([26, 100, 100]), # yellow
    np.array([160, 100, 100]), # purple
    np.array([0, 0, 0]), # black
    np.array([0, 0, 255]), # gold
    np.array([0, 255, 255]), # silver
]
# Define upper in HSV
upper_colors = [
    np.array([10, 255, 255]), # red
    np.array([70, 255, 255]), # green
    np.array([130, 255, 255]), # blue
    np.array([20, 255, 70]), # brown
    np.array([30, 255, 255]), # orange
    np.array([32, 255, 255]), # yellow
    np.array([180, 255, 255]), # purple
    np.array([180, 255, 50]), # black
    np.array([60, 255, 255]), # gold
    np.array([30, 50, 255]), # silver
]

color_names = ['red', 'green', 'blue', 'brown', 'orange', 'yellow', 'purple', 'black', 'gold', 'silver']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Read in frame from webcam
    ret, frame = cap.read()

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Loop over color bounds
    band_colors = []
    for i in range(len(lower_colors)):
        # Create mask for current color
        mask = cv2.inRange(hsv, lower_colors[i], upper_colors[i])

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Calculate centroid of largest contour
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w*h < 300:
                continue
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

            # Determine color of band based on centroid position
            band_colors.append(color_names[i])

            # Draw circle at centroid and label with color
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(frame, color_names[i], (cx - 30, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Resistor Color Detection', frame)

    # Exit loop if 'esc' key is pressed
    if cv2.waitKey(1) == 27:
        break


# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
