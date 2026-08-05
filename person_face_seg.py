import cv2
from ultralytics import YOLO
from deepface import DeepFace

# Load your model
model = YOLO('best.pt')

# Automatically find the right camera index
cap = None
for i in range(3):
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        cap = temp_cap
        print(f"Using Camera Index: {i}")
        break
    temp_cap.release()

if cap is None:
    print("No webcam found! Check your Windows Privacy Settings.")
    exit()

# Load your face database (folder containing folders each named with a person's name and contains several close up images of the person's face)
db_path = 'face_dataset'

# Create the window and allow it to be resized
cv2.namedWindow("Multi-Person Recognition", cv2.WINDOW_NORMAL)

# This line ensures the image stretches to fill the window when you drag it
cv2.setWindowProperty("Multi-Person Recognition", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

# Initialize memory before the loop
id_to_name = {}  # This dictionary will store {track_id: "Name"}
frame_count = 0
line_x = 400            # Vertical line position (pixels from left)
counter = 0             # Total crossing count
previous_x = {}         # Dictionary to track last position: {track_id: cx}
already_counted = set()  # Stores IDs that already crossed

while True:
    success, frame = cap.read()
    if not success: break
    frame_count += 1

    # Use .track() to get persistent IDs for each person
    results = model.track(frame, conf=0.5, persist=True, show=False, device='cpu')
    annotated_frame = results[0].plot()

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Check crossing relative to the vertical X boundary
            if track_id in previous_x:
                prev_cx = previous_x[track_id]

                # Left-to-Right Crossing
                if prev_cx < line_x and cx >= line_x and track_id not in already_counted:
                    counter += 1
                    already_counted.add(track_id)

                # Right-to-Left Crossing
                elif prev_cx > line_x and cx <= line_x and track_id not in already_counted:
                    counter += 1
                    already_counted.add(track_id)

            # Update position for next frame
            previous_x[track_id] = cx

            # Only run DeepFace if we haven't identified this ID yet
            # OR run it every 30 frames to confirm identity
            if track_id not in id_to_name or frame_count % 30 == 0:
                person_crop = frame[y1:y2, x1:x2]

                try:
                    objs = DeepFace.find(img_path=person_crop,
                                         db_path=db_path,
                                         model_name='ArcFace',
                                         detector_backend='opencv',
                                         enforce_detection=False,
                                         silent=True)

                    if len(objs) > 0 and not objs[0].empty:
                        best_match = objs[0].iloc[0]
                        if best_match['distance'] < 0.6:
                            id_to_name[track_id] = best_match['identity'].split('\\')[-2]
                        else:
                            id_to_name[track_id] = "Unknown"
                    else:
                        id_to_name[track_id] = "Unknown"
                except:
                    id_to_name[track_id] = "Searching..."

            # Draw the specific name for THIS box
            display_name = id_to_name.get(track_id, "Scanning...")
            cv2.putText(annotated_frame, f"ID: {display_name}", (x1 + 250, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Draw the vertical boundary line (Green)
    cv2.line(annotated_frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 0), 2)

    # Draw the live counter text (Green)
    cv2.putText(annotated_frame, f"Count: {counter}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # NOW display the fully drawn frame
    cv2.imshow("Multi-Person Recognition", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
