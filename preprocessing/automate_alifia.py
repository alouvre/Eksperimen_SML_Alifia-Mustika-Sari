#!/usr/bin/env python3
"""
automate_alifia.py
------------------
• Dapat di‑import: from automate_alifia import preprocess_student_data
• Dapat dijalankan via CLI  : python automate_alifia.py input.csv output.csv
"""

import sys
import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------- FUNGSI PREPROCESSING UTAMA ----------
def preprocess_student_data(
        df_raw,
        selected_features=None,
        is_training: bool = True,
        scaler: StandardScaler | None = None,
        save_path="preprocessing_output",
        save_result=False
        ):
    """
    Fungsi preprocessing otomatis data mahasiswa untuk prediksi dropout.

    Args:
    - df_raw: DataFrame mentah.
    - selected_features: List kolom fitur yang akan digunakan (default: 14 fitur penting).
    - is_training: True jika sedang training (fit scaler), False untuk prediksi.
    - scaler: Jika is_training=False, gunakan scaler yang sudah dilatih.
    - save_path: Folder untuk menyimpan output hasil prediksi (jika save_result=True).
    - save_result: True jika ingin menyimpan hasil X prediksi sebagai CSV.

    Returns:
    - Jika training: (X_scaled_df, y, scaler)
    - Jika prediksi: X_scaled_df
    """
    df = df_raw.copy()

    # 1. Filter hanya Dropout & Graduate
    df = df[df['Status'] != 'Enrolled']
    df = df.reset_index(drop=True)

    # 2. Encode target label
    label_enc = LabelEncoder()
    df['Status'] = label_enc.fit_transform(df['Status'])    # Dropout 0, Graduate 1

    # 3. Fitur default jika tidak diberikan
    if selected_features is None:
        selected_features = [
            'MothersQualification', 'FathersQualification',
            'MothersOccupation', 'FathersOccupation',
            'CurricularUnits1stSemCredited', 'CurricularUnits1stSemEnrolled',
            'CurricularUnits1stSemEvaluations', 'CurricularUnits1stSemApproved',
            'CurricularUnits1stSemGrade', 'CurricularUnits2ndSemCredited',
            'CurricularUnits2ndSemEnrolled', 'CurricularUnits2ndSemEvaluations',
            'CurricularUnits2ndSemApproved', 'CurricularUnits2ndSemGrade'
        ]

    # 4. Pisahkan X dan y
    X = df[selected_features]
    y = df['Status'] if is_training else None

    # 5. Scaling
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=selected_features), y, scaler

    else:
        if scaler is None:
            raise ValueError("❌ Scaler harus diberikan saat is_training=False.")
        X_scaled = scaler.transform(X)
        X_df = pd.DataFrame(X_scaled, columns=selected_features)

        # Simpan ke CSV jika diminta
        if save_result:
            os.makedirs(save_path, exist_ok=True)
            X_df.to_csv(os.path.join(save_path, "data_student_preprocessing.csv"), index=False)
            print(f"✅ Hasil preprocessing disimpan di: {save_path}/data_student_preprocessing.csv")
        else:
            print("❎ Hasil preprocessing tidak disimpan")

        return X_df


# ---------- ENTRY‑POINT UNTUK COMMAND‑LINE ----------
def _cli():
    """
    contoh:
        python automate_alifia.py data/data_student_raw.csv \
                                  preprocessing/preprocessing_output/data_student_preprocessed.csv
    """
    if len(sys.argv) != 3:
        print("Usage: python preprocessing/automate_alifia.py <input_csv> <output_csv>")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    # Baca data mentah
    df_raw = pd.read_csv(in_path)

    # Jalankan preprocessing (mode training)
    X_df, y, scaler = preprocess_student_data(df_raw, is_training=True)

    # Gabungkan X dan y untuk disimpan mudah
    df_preprocessed = pd.concat([X_df, y.rename('Status')], axis=1)

    # Pastikan folder tujuan ada
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Simpan ke CSV
    df_preprocessed.to_csv(out_path, index=False)
    print(f"✅ Dataset bersih disimpan di: {out_path}")

    # Opsional – simpan scaler kalau diperlukan di workflow selanjutnya
    scaler_path = os.path.join(os.path.dirname(out_path), "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler disimpan di : {scaler_path}")


# Jalankan _cli() hanya jika file dipanggil langsung
if __name__ == "__main__":
    _cli()
