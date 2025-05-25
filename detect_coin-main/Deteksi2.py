import cv2
from ultralytics import YOLO

# Load model hasil training (yang hanya mendeteksi koin)
model = YOLO(r"C:\Users\yipyi\Downloads\Mohammad_allif_alfath_4TID\Mohammad_allif_alfath_4TID\runs\detect\train2\weights\best.pt")

# Pemetaan kelas ke nominal koin
class_to_nominal = {
    0: 100,
    1: 200,
    2: 500,
    3: 1000
}

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, stream=True)
    total_money = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Hanya deteksi kelas koin (0 sampai 3)
            if cls in class_to_nominal:
                nominal = class_to_nominal[cls]
                total_money += nominal

                # Gambar kotak dan label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Rp{nominal}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Jika bukan koin, jangan gambar apapun (bisa juga ditandai warna merah kalau mau debug)
                pass

    # Tampilkan total uang
    cv2.putText(frame, f'Total: Rp{total_money}', (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("YOLOv8 Coin Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
