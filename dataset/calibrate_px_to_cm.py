# calibrate_px_to_cm.py
import csv
import numpy as np

CALIB_CSV = r"D:\goldfish\output\measurement_px_calibration.csv"

def main():
    px_list = []
    cm_list = []

    with open(CALIB_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                px = float(row["length_px"])
                cm = float(row["length_cm"])
                px_list.append(px)
                cm_list.append(cm)
            except:
                continue

    if len(px_list) < 2:
        raise RuntimeError("Data kurang. Minimal dua sampel dibutuhkan.")

    x = np.array(px_list)
    y = np.array(cm_list)

    x_mean = x.mean()
    y_mean = y.mean()

    A = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)
    B = y_mean - A*x_mean

    y_pred = A*x + B
    rmse = np.sqrt(np.mean((y - y_pred)**2))

    print("\n=== HASIL KALIBRASI ===")
    print(f"A (slope)      = {A:.6f} cm/pixel")
    print(f"B (intercept)  = {B:.6f} cm")
    print(f"RMSE Error     = {rmse:.4f} cm")
    print(f"Sampel         = {len(x)}")
    print("\nMasukkan nilai ini ke measure_and_count.py pada variabel A dan B.")

if __name__ == "__main__":
    main()
