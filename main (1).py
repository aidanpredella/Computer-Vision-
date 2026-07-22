"""
main.py
OCR plat nomor kendaraan Indonesia menggunakan Visual Language Model (VLM)
yang dijalankan lokal via LM Studio, diintegrasikan dengan Python.

Dataset: juanthomaswijaya/indonesian-license-plate-dataset (folder test)
Format dataset: YOLO detection (images/ + labels/*.txt berisi bbox plat).
Ground truth teks plat TIDAK ada di label YOLO, jadi diambil dari nama file
gambar (asumsi nama file = teks plat, mis. "B1234XYZ.jpg"). Jika struktur
dataset kamu beda, sesuaikan fungsi get_ground_truth().

Requirement: LM Studio jalan di background dengan model vision-capable
yang sudah di-load (lihat KONFIGURASI di bawah).
"""

import os
import csv
import base64
import requests

# =========================
# CER (Character Error Rate)
# =========================

def _normalize(text: str) -> str:
    """Normalisasi teks plat nomor: uppercase, hapus spasi berlebih."""
    if text is None:
        return ""
    return "".join(text.upper().split())


def compute_cer(ground_truth: str, prediction: str):
    """
    Hitung CER antara ground_truth dan prediction.
    CER = (S + D + I) / N
    S = substitusi, D = deletion, I = insertion, N = jumlah karakter ground truth.

    Returns:
        dict dengan keys: cer, S, D, I, N
    """
    gt = _normalize(ground_truth)
    pred = _normalize(prediction)

    n, m = len(gt), len(pred)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
            else:
                sub_cost = dp[i - 1][j - 1] + 1
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1

                best = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = best
                if best == sub_cost:
                    op[i][j] = "S"
                elif best == del_cost:
                    op[i][j] = "D"
                else:
                    op[i][j] = "I"

    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and op[i][j] == "M":
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and op[i][j] == "S":
            S += 1
            i, j = i - 1, j - 1
        elif i > 0 and op[i][j] == "D":
            D += 1
            i -= 1
        elif j > 0 and op[i][j] == "I":
            I += 1
            j -= 1
        else:
            if i > 0:
                D += 1
                i -= 1
            else:
                I += 1
                j -= 1

    N = n if n > 0 else 1
    cer_value = (S + D + I) / N

    return {"cer": round(cer_value, 4), "S": S, "D": D, "I": I, "N": n}


# =========================
# KONFIGURASI
# =========================
LMSTUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "qwen/qwen2.5-vl-7b"

# Path relatif terhadap lokasi file main.py ini.
# Script ini diasumsikan berada LANGSUNG di dalam folder dataset
# (mis. ".../Indonesian License Plate Recognition Dataset/main.py")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = BASE_DIR

IMAGES_DIR = os.path.join(DATASET_ROOT, "images", "test")

# Kemungkinan lokasi label YOLO (dicoba berurutan, dipakai yang pertama ada)
LABELS_DIR_CANDIDATES = [
    os.path.join(DATASET_ROOT, "labels", "test"),
    os.path.join(DATASET_ROOT, "labels"),
]
LABELS_DIR = next((p for p in LABELS_DIR_CANDIDATES if os.path.isdir(p)), LABELS_DIR_CANDIDATES[0])

OUTPUT_CSV = os.path.join(BASE_DIR, "hasil_ocr.csv")

PROMPT = (
    "What is the license plate number shown in this image? "
    "Respond only with the plate number, no explanation, no punctuation."
)

# =========================
# UTIL
# =========================

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def crop_plate_if_labeled(image_path: str, label_path: str):
    """
    Jika ada file label YOLO (class x_center y_center width height, normalized),
    crop area plat nomor supaya VLM fokus. Kalau label tidak ada, pakai gambar asli.
    Return path gambar yang sudah di-crop (disimpan sementara) atau image_path asli.
    """
    if not os.path.exists(label_path):
        return image_path

    try:
        from PIL import Image
    except ImportError:
        return image_path  # PIL tidak ada, skip crop

    with open(label_path, "r") as f:
        line = f.readline().strip()
    if not line:
        return image_path

    parts = line.split()
    if len(parts) < 5:
        return image_path

    _, xc, yc, w, h = parts[:5]
    xc, yc, w, h = float(xc), float(yc), float(w), float(h)

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)

    # sedikit padding
    pad = 5
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)

    cropped = img.crop((x1, y1, x2, y2))

    tmp_path = os.path.join("/tmp", "crop_" + os.path.basename(image_path))
    cropped.save(tmp_path)
    return tmp_path


def get_ground_truth(image_filename: str) -> str:
    """
    Ambil ground truth teks plat dari nama file (tanpa ekstensi).
    Contoh: "B1234XYZ.jpg" -> "B1234XYZ"
    SESUAIKAN fungsi ini kalau dataset kamu punya file mapping/CSV terpisah.
    """
    name, _ = os.path.splitext(image_filename)
    return name


def query_vlm(image_path: str) -> str:
    """Kirim gambar ke LM Studio (local server, OpenAI-compatible API)."""
    b64 = encode_image_base64(image_path)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 50,
    }

    resp = requests.post(LMSTUDIO_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return text.strip()


# =========================
# MAIN PIPELINE
# =========================

def main():
    if not os.path.isdir(IMAGES_DIR):
        print(f"ERROR: folder gambar tidak ditemukan: {IMAGES_DIR}")
        print("Sesuaikan variabel DATASET_ROOT / IMAGES_DIR di main.py")
        return

    image_files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_files:
        print("Tidak ada gambar ditemukan di", IMAGES_DIR)
        return

    rows = []
    total_cer = 0.0

    for idx, fname in enumerate(image_files, 1):
        image_path = os.path.join(IMAGES_DIR, fname)
        label_path = os.path.join(LABELS_DIR, os.path.splitext(fname)[0] + ".txt")

        gt = get_ground_truth(fname)
        infer_path = crop_plate_if_labeled(image_path, label_path)

        try:
            prediction = query_vlm(infer_path)
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] {fname}: GAGAL query VLM ({e})")
            prediction = ""

        result = compute_cer(gt, prediction)
        total_cer += result["cer"]

        rows.append({
            "image": fname,
            "ground_truth": gt,
            "prediction": prediction,
            "CER_score": result["cer"],
        })

        print(f"[{idx}/{len(image_files)}] {fname} | GT: {gt} | Pred: {prediction} | CER: {result['cer']}")

    # simpan CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "prediction", "CER_score"])
        writer.writeheader()
        writer.writerows(rows)

    avg_cer = total_cer / len(rows) if rows else 0
    print(f"\nSelesai. Rata-rata CER: {avg_cer:.4f}")
    print(f"Hasil disimpan di: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()