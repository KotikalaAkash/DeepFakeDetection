import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm


def extract_faces_yolo(video_path,
                       output_folder,
                       model_path="C:\\Users\\RGUKT\\Desktop\\deepfake\\models\\yolov8n-face.pt",
                       frame_skip=10,
                       max_faces=30,
                       img_size=224):

    os.makedirs(output_folder, exist_ok=True)

    # Load YOLO model
    model = YOLO(model_path)
    # model = YOLO("https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face.pt")
    # model = YOLO("yolov5-face.pt")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video.")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret or saved_count >= max_faces:
            break

        if frame_count % frame_skip == 0:

            results = model(frame, verbose=False)

            for r in results:
                boxes = r.boxes

                if boxes is not None:
                    for box in boxes:

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        face = frame[y1:y2, x1:x2]

                        if face.size == 0:
                            continue

                        face = cv2.resize(face, (img_size, img_size))

                        save_path = os.path.join(
                            output_folder,
                            f"frame_{saved_count:02d}.jpg"
                        )

                        cv2.imwrite(save_path, face)
                        saved_count += 1
                        break

                break

        frame_count += 1

    cap.release()

    print("Extraction finished")
    print("Total faces saved:", saved_count)


input_root = "C:\\Users\\RGUKT\\Desktop\\deepfake\\CelebDF_final"         
output_root = "dataset_faces"  

splits = ["train", "test",  "val"]

for split in splits:

    split_input_path = os.path.join(input_root, split)

    if not os.path.exists(split_input_path):
        print(f"{split} folder not found, skipping...")
        continue

    for class_name in os.listdir(split_input_path):

        class_input_path = os.path.join(split_input_path, class_name)

        if not os.path.isdir(class_input_path):
            continue

        print(f"\n Processing: {split}/{class_name}")

        for video_file in tqdm(os.listdir(class_input_path)):

            if video_file.endswith((".mp4", ".avi", ".mov", ".mkv")):

                video_path = os.path.join(class_input_path, video_file)
                video_name = os.path.splitext(video_file)[0]

                save_folder = os.path.join(
                    output_root,
                    split,
                    class_name,
                    video_name
                )

                extract_faces_yolo(video_path, save_folder)

print("\n All train/val/test face extraction completed successfully!")