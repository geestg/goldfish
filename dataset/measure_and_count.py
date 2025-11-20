# measure_and_count.py
from ultralytics import YOLO
import cv2
import numpy as np
import csv
import os

# =======================
# KONFIGURASI AWAL
# =======================
MODEL_PATH = r"D:\goldfish\training\weights\best.pt"  # model YOLOv8 Pose Anda
INPUT_VIDEO = r"D:\goldfish\videos\mas2.mp4"         # video kalibrasi Anda
OUTPUT_VIDEO = r"D:\goldfish\output\annotated_mas2.mp4"
OUTPUT_CSV = r"D:\goldfish\output\measurement_px.csv"

# Nilai sementara (jangan diubah)
A = 1.0
B = 0.0

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


# =======================
# FUNGSI PERHITUNGAN
# =======================
def compute_length_px(head_xy, tail_xy):
    dx = tail_xy[0] - head_xy[0]
    dy = tail_xy[1] - head_xy[1]
    return float(np.sqrt(dx * dx + dy * dy))


def px_to_cm(length_px, a=A, b=B):
    return float(a * length_px + b)


# =======================
# MAIN PIPELINE
# =======================
def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writer video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # File CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    csv_file = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame_index",
        "fish_id",
        "head_x", "head_y",
        "tail_x", "tail_y",
        "length_px",
        "length_cm"
    ])

    frame_idx = 0
    print("Memulai proses pengukuran piksel...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        result = results[0]

        fish_count = 0

        if result.keypoints is not None:
            kpts_xy = result.keypoints.xy.cpu().numpy()

            for i in range(kpts_xy.shape[0]):
                head = kpts_xy[i, 0]
                tail = kpts_xy[i, 1]

                length_px = compute_length_px(head, tail)
                length_cm = px_to_cm(length_px)

                csv_writer.writerow([
                    frame_idx, i,
                    f"{head[0]:.2f}", f"{head[1]:.2f}",
                    f"{tail[0]:.2f}", f"{tail[1]:.2f}",
                    f"{length_px:.4f}",
                    f"{length_cm:.4f}"
                ])

                # Gambar keypoint head & tail
                head_int = (int(head[0]), int(head[1]))
                tail_int = (int(tail[0]), int(tail[1]))

                cv2.circle(frame, head_int, 5, (0, 0, 255), -1)
                cv2.circle(frame, tail_int, 5, (255, 0, 0), -1)
                cv2.line(frame, head_int, tail_int, (0, 255, 0), 2)

                fish_count += 1

        cv2.putText(frame, f"Fish count: {fish_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    csv_file.close()
    print("Selesai! CSV dan video anotasi tersimpan.")


if __name__ == "__main__":
    main()
