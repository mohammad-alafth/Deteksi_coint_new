from ultralytics import YOLO

# Load model base
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data="Coin-Detector-2/data.yaml",  # path ke file YAML kamu
    epochs=100,
    imgsz=640
)


# 3. Evaluasi model setelah training
# Evaluasi performa model
metrics = model.val()

# Hasil evaluasi global
print("=== EVALUASI MODEL (MEAN) ===")
print(f"mAP50      : {metrics.map50():.4f}")
print(f"mAP50-95   : {metrics.map():.4f}")
print(f"Precision  : {metrics.mp():.4f}")
print(f"Recall     : {metrics.mr():.4f}")

# Hasil evaluasi per kelas
class_names = [
    "100",
    "seratus",
    "1000-angklung",
    "1000-silver",
    "1000-sawit",
    "1000-SAWIT",
    "200",
    "200-silver",
    "500-kuning",
    "500-silver"
]

print("\n=== EVALUASI PER KELAS ===")
for i, name in enumerate(class_names):
    p, r, ap50, ap = metrics.class_result(i)
    print(f"{name:<15} | Precision: {p:.4f} | Recall: {r:.4f} | mAP50: {ap50:.4f} | mAP50-95: {ap:.4f}")

