from ultralytics import YOLO

model = YOLO("D:/goldfish/training/weights/best.pt")

results = model.predict(
    source="D:/goldfish/frame",   # test semua frame sekaligus
    save=True,
    conf=0.5
)

print("Selesai uji banyak gambar.")
