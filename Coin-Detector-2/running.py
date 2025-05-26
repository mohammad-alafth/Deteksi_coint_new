from ultralytics import YOLO
import cv2

# Load model hasil training
model = YOLO(r"C:\Projek_CV\Mohammad_allif_alfath_4TID\runs\detect\train20\weights\best.pt")

# Buka webcam (0 untuk webcam utama)
cap = cv2.VideoCapture(1)

# Daftar label sesuai urutan di data.yaml
class_to_nominal = {
    0: 100,   # "100"
    1: 100,   # "seratus"
    2: 1000,  # "1000-angklung"
    3: 1000,  # "1000-silver"
    4: 1000,  # "1000-sawit"
    5: 1000,  # "1000-SAWIT"
    6: 200,   # "200"
    7: 200,   # "200-silver"
    8: 500,   # "500-kuning"
    9: 500    # "500-silver"
}

while True:
    success, frame = cap.read()
    if not success:
        break

    # Jalankan deteksi
    results = model(frame, stream=True)
    total_money = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Hanya proses jika termasuk koin
            if cls in class_to_nominal:
                nominal = class_to_nominal[cls]
                total_money += nominal

                # Gambar bounding box dan label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Rp{nominal}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan total deteksi
    cv2.putText(frame, f'Total: Rp{total_money}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("YOLOv8 Coin Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
