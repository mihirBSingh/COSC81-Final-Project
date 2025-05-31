import cv2
import numpy as np

# Load the image
image = cv2.imread('dartmouthmap.jpg')

print

# Convert to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color range (example: blue)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create the mask
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Apply the mask
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Mask', mask)
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()