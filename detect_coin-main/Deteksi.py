import cv2 as cv
import cvzone
import numpy as np
from cvzone.ColorModule import ColorFinder

# Inisialisasi video capture dari kamera default
cap = cv.VideoCapture(0)  # Ubah dari 1 ke 0

# Cek apakah kamera berhasil dibuka
if not cap.isOpened():
    print("Kamera tidak dapat dibuka!")
    exit()

# Set resolusi kamera (opsional)
cap.set(3, 640)
cap.set(4, 480)

# Objek ColorFinder (tidak digunakan aktif)
myColorFinder = ColorFinder(False)

# Fungsi trackbar
def empty(a):
    pass

# Jendela pengaturan trackbar Canny
cv.namedWindow("Settings")
cv.resizeWindow("Settings", 640, 240)
cv.createTrackbar("Threshold1", "Settings", 100, 255, empty)
cv.createTrackbar("Threshold2", "Settings", 200, 255, empty)

# Fungsi preprocessing
def preProcessing(img):
    imgPre = cv.GaussianBlur(img, (5, 5), 3)
    thresh1 = cv.getTrackbarPos("Threshold1", "Settings")
    thresh2 = cv.getTrackbarPos("Threshold2", "Settings")
    imgPre = cv.Canny(imgPre, thresh1, thresh2)
    kernel = np.ones((3, 3), np.uint8)
    imgPre = cv.dilate(imgPre, kernel, iterations=1)
    imgPre = cv.morphologyEx(imgPre, cv.MORPH_CLOSE, kernel)
    return imgPre

# Fungsi circularity
def calculate_circularity(contour):
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity

# Loop utama
while True:
    success, img = cap.read()
    if not success:
        print("Gagal membaca frame dari kamera.")
        break

    # Tampilkan frame kamera mentah untuk memastikan kamera bekerja
    # cv.imshow("Kamera", img)

    imgPre = preProcessing(img)
    imgContours, conFound = cvzone.findContours(img, imgPre, minArea=500)

    money = 0
    imgCount = np.zeros((480, 640, 3), np.uint8)

    if conFound:
        for contour in conFound:
            peri = cv.arcLength(contour["cnt"], True)
            approx = cv.approxPolyDP(contour["cnt"], 0.02 * peri, True)

            if len(approx) > 5:
                area = contour["area"]
                x, y, w, h = contour['bbox']
                circularity = calculate_circularity(contour['cnt'])

                # Debug area & circularity
                print(f"Area: {area:.2f}, Circularity: {circularity:.2f}")

                nominal = 0

                if circularity > 0.7:
                    if 2100 < area <= 3700:
                        nominal = 100
                    elif 2500 < area <= 4300:
                        nominal = 200
                    elif 2900 < area <= 5100:
                        nominal = 500
                    elif 2300 < area <= 4000:
                        nominal = 1000

                    if nominal > 0:
                        money += nominal
                        cvzone.putTextRect(img, f'Rp{nominal}', (x, y - 10), scale=1, colorR=(0, 255, 0), thickness=2)
                        cv.drawContours(imgContours, [contour['cnt']], -1, (0, 255, 0), 2)

    cvzone.putTextRect(imgCount, f'Total: Rp{money}', (100, 300), scale=5, colorR=(0, 0, 255), thickness=5)
    imageStacked = cvzone.stackImages([img, imgPre, imgContours, imgCount], 2, 0.5)
    cvzone.putTextRect(imageStacked, f'Total: Rp{money}', (50, 50), colorR=(0, 0, 255))
    cv.imshow("Money Counter", imageStacked)

    key = cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
