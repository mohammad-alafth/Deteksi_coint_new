from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="Mohammad_allif_alfath_4TID/Coin_rupiah_detection/data.yaml", 
        epochs=50,
        imgsz=896,
        device='cuda', # jika ingin training di GPU
        batch=16  # atau 8 jika RAM GPU terbatas
    )

    # Evaluasi setelah training
    metrics = model.val()
    print("=== EVALUASI MODEL (MEAN) ===")
    print(f"mAP50      : {metrics.map50():.4f}")
    print(f"mAP50-95   : {metrics.map():.4f}")
    print(f"Precision  : {metrics.mp():.4f}")
    print(f"Recall     : {metrics.mr():.4f}")

    class_names = [
        'coin_1000_e2010', 'coin_1000_e2016', 'coin_100_e1999',
        'coin_100_e2016', 'coin_200_e2003', 'coin_200_e2016',
        'coin_500_e2003', 'coin_500_e2016', 'coin_50_1999'
    ]
    print("\n=== EVALUASI PER KELAS ===")
    for i, name in enumerate(class_names):
        p, r, ap50, ap = metrics.class_result(i)
        print(f"{name:<15} | Precision: {p:.4f} | Recall: {r:.4f} | mAP50: {ap50:.4f} | mAP50-95: {ap:.4f}")

if __name__ == '__main__':
    main()
