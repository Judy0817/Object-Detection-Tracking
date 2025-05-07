# import cv2
# import numpy as np

# # Set the source video path
# SOURCE_VIDEO_PATH = 'd:\\User Data\\Oshadi\\USJ\\Acedemic\\4th Year\\Sem 7\\FYP\\FYP_Ideas_me\\input_videos\\video1.mp4'

# # Read the first frame from the video
# cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
# ret, frame = cap.read()
# cap.release()

# # Check if the frame was loaded
# if not ret:
#     print("Error: Unable to read video.")
#     exit()

# polygon_points = []  # Store polygon points

# # Mouse callback function to capture polygon points
# def draw_polygon(event, x, y, flags, param):
#     global polygon_points, frame

#     if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse click
#         polygon_points.append((x, y))  # Append clicked point
#         cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Draw point
#         if len(polygon_points) > 1:
#             cv2.line(frame, polygon_points[-2], polygon_points[-1], (255, 0, 0), 2)  # Draw lines

#         cv2.imshow("Select Polygon", frame)

# # Display frame and set mouse callback
# cv2.imshow("Select Polygon", frame)
# cv2.setMouseCallback("Select Polygon", draw_polygon)

# print("Click on the frame to define the polygon. Press 'q' when done.")

# # Wait for user input
# while True:
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') and len(polygon_points) > 2:  # Press 'q' to finish
#         break

# cv2.destroyAllWindows()

# # Convert to NumPy array
# polygon_zone = np.array(polygon_points, np.int32)

# # Print or save the polygon points
# print("Selected Polygon Coordinates:", polygon_zone.tolist())

import cv2
import numpy as np

# Global variables to store polygon points
polygon_points = []
drawing_complete = False  # Flag to indicate if polygon drawing is complete

# Mouse callback function to capture clicks
def draw_polygon(event, x, y, flags, param):
    global polygon_points, drawing_complete

    if event == cv2.EVENT_LBUTTONDOWN and not drawing_complete:
        polygon_points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        if len(polygon_points) > 1:
            cv2.line(frame, polygon_points[-2], polygon_points[-1], (0, 255, 0), 2)
        cv2.imshow("Define Polygon Zone", frame)

# Load the video
# video_path = 'Object-Detection-Tracking\school_gate.mp4'
video_path="d:\\User Data\\Oshadi\\USJ\\Acedemic\\4th Year\\Sem 7\\FYP\\FYP_Data_Lake\\Object-Detection-Tracking\\peoplecounteryolov8-main\\peoplecount1.mp4"
cap = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

# Create a window and set the mouse callback function
cv2.namedWindow("Define Polygon Zone", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Define Polygon Zone", frame.shape[1], frame.shape[0])  # Ensure full image is visible
cv2.setMouseCallback("Define Polygon Zone", draw_polygon)

# Instructions for the user
print("Click on the image to define the polygon vertices. Press 'q' to finish.")

while True:
    cv2.imshow("Define Polygon Zone", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit and save the polygon
    if key == ord('q'):
        drawing_complete = True
        break

# Close all OpenCV windows
cv2.destroyAllWindows()

# Convert polygon points to numpy array
if len(polygon_points) > 2:  # Ensure at least 3 points for a valid polygon
    polygon_zone = np.array(polygon_points, np.int32)
    print("Polygon Points:", polygon_zone)

    # Extract the bottom horizontal line of the polygon
    bottom_points = polygon_zone[np.argsort(polygon_zone[:, 1])[-2:]]  # Get the two points with the highest y-values
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]  # Sort by x-coordinate
    GATE_LINE = [tuple(bottom_points[0]), tuple(bottom_points[1])]
    print("Gate Line:", GATE_LINE)
else:
    print("Invalid polygon. Please define at least 3 points.")
    exit()

# Now you can use `polygon_zone` and `GATE_LINE` in your tracking script