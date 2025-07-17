import cv2
import pickle

VIDEO_PATH = "parking/parking_crop.mp4"
OUTPUT_PATH = "parking/outputs/parking_slot_coords.pkl"

cap = cv2.VideoCapture(VIDEO_PATH)
success, frame = cap.read()
cap.release()

if not success:
    print("Could not read the video.")
    exit()

clone = frame.copy()
drawing = False
ix, iy = -1, -1
slots = []

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, slots, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = clone.copy()
            cv2.rectangle(temp, (ix, iy), (x, y), (255, 0, 0), 2)
            for sx, sy, sw, sh in slots:
                cv2.rectangle(temp, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
            cv2.imshow("Draw Parking Slots", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w = abs(x - ix)
        h = abs(y - iy)
        x1, y1 = min(ix, x), min(iy, y)
        slots.append((x1, y1, w, h))
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        cv2.imshow("Draw Parking Slots", frame)

cv2.namedWindow("Draw Parking Slots")
cv2.setMouseCallback("Draw Parking Slots", draw_rectangle)
cv2.imshow("Draw Parking Slots", frame)

print(" Click and drag to draw parking slots.")
print(" Press 's' to save slots.")
print(" Press 'q' to quit without saving.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        with open(OUTPUT_PATH, 'wb') as f:
            pickle.dump(slots, f)
        print(f" Saved {len(slots)} slot coordinates to {OUTPUT_PATH}")
        break
    elif key == ord("q"):
        print(" Quit without saving.")
        break

cv2.destroyAllWindows()

