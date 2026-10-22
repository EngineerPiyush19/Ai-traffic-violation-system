import os
import re
from collections import Counter

import cv2
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO

model = YOLO("plates.pt")
ocr = LicensePlateRecognizer("cct-xs-v1-global-model")

SOURCE = "traffic.mp4"               # video name, or 0 for webcam
MIN_PLATE_WIDTH = 60                 # ignore tiny plates (pixels, original video)
PATTERN = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")   # e.g. DL3CCR8958

os.makedirs("crops", exist_ok=True)
votes = Counter()
cap = cv2.VideoCapture(SOURCE)
n = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1
    if n % 2:                         # process every 2nd frame for speed
        continue

    # detect on the full-resolution frame so plate crops stay sharp
    r = model(frame, imgsz=960, conf=0.4, verbose=False)[0]

    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        if (x2 - x1) < MIN_PLATE_WIDTH:
            continue
        plate = frame[y1:y2, x1:x2]
        if plate.size == 0:
            continue

        cv2.imwrite(f"crops/{n}_{x1}.jpg", plate)      # check these by eye

        res = ocr.run(plate)
        pred = res[0] if isinstance(res, (list, tuple)) and res else res
        text = getattr(pred, "plate", str(pred))
        text = "".join(ch for ch in text.upper() if ch.isalnum())

        color = (0, 0, 255)           # red = not a valid plate format
        if PATTERN.match(text):
            votes[text] += 1
            color = (0, 255, 0)       # green = valid format
            print(text)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    show = cv2.resize(frame, (960, 540))   # shrink only for display
    cv2.imshow("Plates", show)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFinal plates (reading, times seen):")
for plate, count in votes.most_common():
    if count >= 2:
        print(plate, count)