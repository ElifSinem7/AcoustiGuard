# =============================================================================
# AcoustiGuard — Dataset Auto-Downloader (FIXED)
# =============================================================================
# MIMII  -> Zenodo (0_dB_fan.zip — dogru dosya adi)
# CWRU   -> Case Western Reserve University sunucusu
#
# Kullanim:
#   python download_datasets.py
# =============================================================================

import os
import zipfile
import requests
import scipy.io as sio
from tqdm import tqdm

DATA_DIR  = "data/raw"
MIMII_DIR = os.path.join(DATA_DIR, "mimii", "fan")
CWRU_DIR  = os.path.join(DATA_DIR, "cwru")


def download_file(url, dest_path, desc=""):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        desc=desc, total=total, unit="B", unit_scale=True, ncols=80
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


# =============================================================================
# 1. MIMII — Zenodo record 3384388
# =============================================================================
# Gercek dosya adlari (Zenodo sayfasindan dogrulanmistir):
#   0_dB_fan.zip   6_dB_fan.zip   -6_dB_fan.zip
#   0_dB_pump.zip  0_dB_slider.zip  0_dB_valve.zip  vb.
#
# Sadece fan + 0dB SNR indiriyoruz.
# Diger SNR seviyeleri icin asagidaki yorumlari kaldir.

MIMII_FILES = {
    "0_dB_fan.zip": "https://zenodo.org/record/3384388/files/0_dB_fan.zip",
    # "6_dB_fan.zip":  "https://zenodo.org/record/3384388/files/6_dB_fan.zip",
    # "-6_dB_fan.zip": "https://zenodo.org/record/3384388/files/-6_dB_fan.zip",
}


def download_mimii():
    print("\n" + "=" * 60)
    print("1/2  MIMII Dataset (fan, 0dB SNR)")
    print("=" * 60)

    zip_dir = os.path.join(DATA_DIR, "mimii", "_zips")
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(MIMII_DIR, exist_ok=True)

    for filename, url in MIMII_FILES.items():
        zip_path = os.path.join(zip_dir, filename)

        if not os.path.exists(zip_path):
            print(f"\nDownloading {filename}...")
            try:
                download_file(url, zip_path, desc=filename)
            except Exception as e:
                print(f"  ERROR: {e}")
                print("  Manuel indirme: https://zenodo.org/record/3384388")
                print(f"  '{filename}' dosyasini data/raw/mimii/_zips/ klasorune koy")
                continue
        else:
            print(f"Already downloaded: {filename}")

        print(f"Extracting {filename}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                members = z.namelist()
                wav_files = [m for m in members if m.endswith(".wav")]
                # ZIP icindeki yapi: fan/id_00/normal/x.wav veya id_00/normal/x.wav
                skip = 0
                if wav_files:
                    top = wav_files[0].split("/")[0]
                    skip = 1 if top == "fan" else 0

                for member in members:
                    parts = member.split("/")
                    rel_parts = parts[skip:]
                    if not rel_parts or rel_parts == [""]:
                        continue
                    rel  = os.path.join(*rel_parts)
                    dest = os.path.join(MIMII_DIR, rel)
                    if member.endswith("/"):
                        os.makedirs(dest, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with z.open(member) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
            print(f"  -> {MIMII_DIR}")
        except Exception as e:
            print(f"  Extraction error: {e}")


# =============================================================================
# 2. CWRU — Case Western Reserve University Bearing Data Center
# =============================================================================

CWRU_FILES = {
    "NormalBaseline": {
        "97.mat":  "https://engineering.case.edu/sites/default/files/97.mat",
        "98.mat":  "https://engineering.case.edu/sites/default/files/98.mat",
        "99.mat":  "https://engineering.case.edu/sites/default/files/99.mat",
        "100.mat": "https://engineering.case.edu/sites/default/files/100.mat",
    },
    "12DriveEndFault": {
        "105.mat": "https://engineering.case.edu/sites/default/files/105.mat",
        "118.mat": "https://engineering.case.edu/sites/default/files/118.mat",
        "130.mat": "https://engineering.case.edu/sites/default/files/130.mat",
        "109.mat": "https://engineering.case.edu/sites/default/files/109.mat",
        "122.mat": "https://engineering.case.edu/sites/default/files/122.mat",
        "135.mat": "https://engineering.case.edu/sites/default/files/135.mat",
        "144.mat": "https://engineering.case.edu/sites/default/files/144.mat",
        "156.mat": "https://engineering.case.edu/sites/default/files/156.mat",
        "169.mat": "https://engineering.case.edu/sites/default/files/169.mat",
        "185.mat": "https://engineering.case.edu/sites/default/files/185.mat",
    },
}


def download_cwru():
    print("\n" + "=" * 60)
    print("2/2  CWRU Bearing Dataset")
    print("=" * 60)

    for folder, files in CWRU_FILES.items():
        folder_path = os.path.join(CWRU_DIR, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"\n  [{folder}]")

        for filename, url in files.items():
            dest = os.path.join(folder_path, filename)
            if os.path.exists(dest):
                print(f"    Already exists: {filename}")
                continue
            try:
                download_file(url, dest, desc=f"    {filename}")
            except Exception as e:
                print(f"    ERROR {filename}: {e}")
                print("    Manuel: https://engineering.case.edu/bearingdatacenter/download-data-file")


# =============================================================================
# Verify
# =============================================================================

def verify():
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    wav_count = sum(
        1 for r, _, fs in os.walk(MIMII_DIR) for f in fs if f.endswith(".wav")
    )
    mat_count = sum(
        1 for r, _, fs in os.walk(CWRU_DIR) for f in fs if f.endswith(".mat")
    )

    mimii_ok = wav_count > 0
    cwru_ok  = mat_count > 0

    print(f"MIMII WAV files : {wav_count}  {'OK' if mimii_ok else 'MISSING'}")
    print(f"CWRU .mat files : {mat_count}  {'OK' if cwru_ok else 'MISSING'}")

    if mimii_ok and cwru_ok:
        print("\nAll datasets ready! Next steps:")
        print("  python src/extract_mimii.py")
        print("  python src/extract_cwru.py")
        print("  python src/merge_datasets.py")
        print("  python src/train_isolation_forest.py")
    else:
        if not mimii_ok:
            print("\nMIMII eksik. Manuel indir:")
            print("  https://zenodo.org/record/3384388")
            print("  '0_dB_fan.zip' dosyasini data/raw/mimii/_zips/ klasorune koy")
            print("  Sonra tekrar: python download_datasets.py")
        if not cwru_ok:
            print("\nCWRU eksik. Manuel indir:")
            print("  https://engineering.case.edu/bearingdatacenter/download-data-file")


if __name__ == "__main__":
    print("AcoustiGuard — Dataset Downloader")
    print(f"Hedef: {os.path.abspath(DATA_DIR)}\n")
    download_mimii()
    download_cwru()
    verify()
