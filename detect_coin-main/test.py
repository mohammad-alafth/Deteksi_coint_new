from ultralytics import YOLO

# 1. Load model YOLOv8
model = YOLO("yolov8n.pt")  # Bisa diganti dengan yolov8s.pt, m, l, dll sesuai spek

# 2. Training model
model.train(
    data="dataset/data.yaml",  # Path ke file YAML dataset Anda
    epochs=50,
    imgsz=640
)

# 3. Evaluasi model setelah training
metrics = model.val()  # Menggunakan data validasi dari data.yaml

# 4. Tampilkan hasil evaluasi
print("=== EVALUASI MODEL ===")
print(f"mAP50      : {metrics.box.map50:.4f}")
print(f"mAP50-95   : {metrics.box.map:.4f}")
print(f"Precision  : {metrics.box.precision:.4f}")
print(f"Recall     : {metrics.box.recall:.4f}")
