Deteksi dan Klasifikasi Uang Koin Menggunakan OpenCV Python


Latar Belakang

Seiring dengan kemajuan teknologi pengolahan citra digital, berbagai solusi otomatis berbasis komputer kini banyak diterapkan dalam kehidupan sehari-hari, termasuk dalam mengenali objek fisik seperti uang koin. Dalam konteks keuangan mikro, seperti pada usaha kecil, sistem parkir manual, atau mesin penukar uang, proses identifikasi nilai uang koin sering kali masih dilakukan secara manual. Hal ini tidak hanya menghabiskan waktu, tetapi juga rentan terhadap kesalahan manusia.
Penggunaan teknologi computer vision seperti OpenCV memberikan alternatif yang cepat, efisien, dan dapat diandalkan untuk mengenali dan mengklasifikasikan nominal uang koin secara otomatis. Dengan memanfaatkan citra digital dan teknik deteksi kontur, sistem ini dapat mengidentifikasi koin berdasarkan bentuk, luas area, dan tingkat kebulatan (circularity).
Sistem ini dirancang untuk membaca gambar koin, melakukan proses pendeteksian tepi (Canny), lalu menghitung kontur dari objek yang terdeteksi. Berdasarkan ciri visual dan ukuran area dari kontur tersebut, sistem dapat memperkirakan nominal dari setiap koin, seperti Rp100, Rp200, Rp500, dan Rp1000. Total nilai uang kemudian dihitung secara otomatis dan ditampilkan secara real-time.

Tahapan

Tahapan proses dalam proyek ini dapat dijelaskan sebagai berikut:
- Pengambilan Data Gambar
 Gambar-gambar koin dari berbagai nominal disimpan dalam folder dataset yang telah dikelompokkan berdasarkan nilai koin.


- Preprocessing Citra
 Gambar akan diproses dengan metode Gaussian Blur untuk mengurangi noise, dilanjutkan dengan deteksi tepi menggunakan metode Canny, dan kemudian dilakukan dilasi serta morfologi untuk memperjelas kontur.


- Deteksi Kontur
 Kontur dari objek pada gambar dianalisis untuk menemukan bentuk koin. Kontur yang memiliki tingkat kebulatan tinggi (circularity > 0.7) diproses lebih lanjut untuk dianalisis ukurannya.


- Klasifikasi Nilai Koin
 Berdasarkan luas area kontur, sistem mengelompokkan koin ke dalam nominal tertentu, seperti Rp100, Rp200, Rp500, dan Rp1000.


- Perhitungan Total Nilai
 Semua koin yang dikenali akan dijumlahkan untuk menampilkan total nilai uang yang terdapat dalam gambar.


- Visualisasi Hasil
 Nilai masing-masing koin akan ditampilkan pada posisi koin tersebut, serta total uang akan ditampilkan pada satu frame hasil visualisasi akhir.


Library yang Digunakan
- cv2 → OpenCV, untuk pengolahan citra dan deteksi tepi serta kontur.
- cvzone → Untuk mempermudah visualisasi kontur dan teks pada citra.
- numpy → Untuk perhitungan matriks dan operasi numerik.
- os → Untuk membaca file dan direktori dalam dataset.



