from ultralytics import YOLO
import cv2

# Load model hasil training
model = YOLO(r"C:\Projek_CV\runs\detect\train25\weights\best.pt")

# Buka webcam (0 untuk webcam utama)
cap = cv2.VideoCapture(1)

# Mapping index kelas (sesuai urutan data.yaml) ke nilai uang
class_to_nominal = {
    0: 1000,  # coin_1000_e2010
    1: 1000,  # coin_1000_e2016
    2: 100,   # coin_100_e1999
    3: 100,   # coin_100_e2016
    4: 200,   # coin_200_e2003
    5: 200,   # coin_200_e2016
    6: 500,   # coin_500_e2003
    7: 500,   # coin_500_e2016
    8: 50     # coin_50_1999
}

while True:
    success, frame = cap.read()
    if not success:
        break

    # Jalankan deteksi
    results = model.predict(source=frame, stream=True, verbose=False)
    total_money = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
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
