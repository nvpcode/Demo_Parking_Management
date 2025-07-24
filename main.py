import cv2
from ultralytics import YOLO

# Load mô hình đã huấn luyện
model = YOLO("model_best/runs/detect/train/weights/best.pt")

# Tổng số ô đỗ xe (đặt thủ công từ trước – theo số box bạn có)
total_slots = 70  # <-- Thay số này theo số lượng ô thực tế

# Mở video
cap = cv2.VideoCapture("carPark.mp4")
assert cap.isOpened(), "Không thể mở video"

# Cấu hình video writer
w, h, fps = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             cap.get(cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("parking-management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Đọc từng frame và xử lý
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán bằng mô hình YOLO
    results = model.predict(frame, conf=0.3, iou=0.5, verbose=True, device=0)
    r = results[0]

    free_count = 0

    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_name == "car":
                color = (0, 0, 255)
            elif cls_name == "free":
                color = (0, 255, 0)
                free_count += 1

                # Ghi chữ "free" tại trung tâm
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.putText(frame, "free", (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            else:
                continue

            # Vẽ khung
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # --- Hiển thị Free/Total ---
    display_text = f"Free: {free_count}/{total_slots}"
    (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    padding = 10
    bg_x2 = w - 10
    bg_x1 = bg_x2 - text_w - 2 * padding
    bg_y1 = 10
    bg_y2 = bg_y1 + text_h + 2 * padding

    # Vẽ nền trắng
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
    # Vẽ text
    cv2.putText(frame, display_text, (bg_x1 + padding, bg_y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

    video_writer.write(frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("Video đã lưu: parking-management.avi")