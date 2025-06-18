import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def preprocess_student_data(df_raw, selected_features=None, is_training=True, scaler=None, save_path="preprocessing_output", save_result=False):
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

    # Filter hanya Dropout & Graduate
    df = df[df['Status'] != 'Enrolled']
    df = df.reset_index(drop=True)

    # Encode target label
    label_enc = LabelEncoder()
    df['Status'] = label_enc.fit_transform(df['Status'])

    # Fitur default jika tidak diberikan
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

    # Pisahkan X dan y
    X = df[selected_features]
    y = df['Status'] if is_training else None

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
