# README.md

## Parking Management Project

A parking lot management system using YOLO model to detect and count available parking slots in a video.

---

### Demo

▶️ [Watch the demo video](https://drive.google.com/file/d/1CBrsH-xH2jNB7mFrT_2H96_WJ9wWixtu/view?usp=drive_link)

---

### Requirements

- Python 3.8+
- OpenCV (`cv2`)
- Ultralytics YOLO (`ultralytics`)
- Trained YOLO model at `model_best/runs/detect/train/weights/best.pt`
- Input video named `carPark.mp4`

---

### Installation

```sh
pip install opencv-python ultralytics
```

---

### Usage

1. Place the video to be analyzed in the project folder as `carPark.mp4`.
2. Make sure the trained YOLO model is at `model_best/runs/detect/train/weights/best.pt`.
3. Edit the `total_slots` variable in [main.py](d:\Projects_COMPUTER_VISION\venv\Project_Parking_Management\main.py) to match the actual number of parking slots.
4. Run the program:

```sh
python main.py
```

5. The result will be saved as `parking-management.avi` with the number of free slots displayed on each frame.

---

### Source Code Explanation

- [main.py](d:\Projects_COMPUTER_VISION\venv\Project_Parking_Management\main.py): Reads the video, predicts parking slots using YOLO, draws bounding boxes, and displays the number of free/total slots.
- Uses YOLO to classify regions as "car" or "free".
- Displays the number of free slots on the output video.

---

### Contact

For any questions, please contact via email: nguyenphuongv07@gmail.com.
