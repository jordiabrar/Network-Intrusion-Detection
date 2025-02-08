import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset NSL-KDD
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
           "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
           "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
           "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", 
           "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", 
           "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

# Load training & testing data
df_train = pd.read_csv("KDDTrain+.csv", names=columns)
df_test = pd.read_csv("KDDTest+.csv", names=columns)

# 2. Preprocessing Data
# Menghapus kolom yang tidak relevan
drop_columns = ['num_outbound_cmds']  # Tidak memiliki informasi yang berguna
df_train.drop(columns=drop_columns, inplace=True)
df_test.drop(columns=drop_columns, inplace=True)

# Encoding fitur kategorikal
encoder = LabelEncoder()
categorical_features = ['protocol_type', 'service', 'flag']
for col in categorical_features:
    df_train[col] = encoder.fit_transform(df_train[col])
    df_test[col] = encoder.transform(df_test[col])

# Encoding label menjadi biner (Normal: 0, Serangan: 1)
df_train['label'] = df_train['label'].apply(lambda x: 0 if x == 'normal' else 1)
df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Pisahkan fitur dan target
X_train, y_train = df_train.drop(columns=['label']), df_train['label']
X_test, y_test = df_test.drop(columns=['label']), df_test['label']

# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Train Model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 6. Prediksi Serangan Baru
def predict_intrusion(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return "Normal Traffic" if prediction == 0 else "Possible Intrusion Detected"

# Contoh prediksi
test_sample = X_test[0]
result = predict_intrusion(test_sample)
print("Prediksi untuk sampel data baru:", result)
