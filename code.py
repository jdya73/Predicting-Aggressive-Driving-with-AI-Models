import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
train_df = pd.read_csv("train_motion_data.csv")
test_df = pd.read_csv("test_motion_data.csv")

# Normalize sensor data
scaler = MinMaxScaler()
train_df[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']] = scaler.fit_transform(
    train_df[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']]
)
test_df[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']] = scaler.transform(
    test_df[['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']]
)

# Encode target labels
label_encoder = LabelEncoder()
train_df['Class'] = label_encoder.fit_transform(train_df['Class'])
test_df['Class'] = label_encoder.transform(test_df['Class'])

# Increase sequence length to avoid zero-size pooling
SEQ_LENGTH = 10  # Increase time window for better feature extraction

# Function to create sequences from dataset
def create_sequences(df, seq_length):
    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(df.iloc[i:i + seq_length, :-2].values)  # Use all sensor columns
        y.append(df.iloc[i + seq_length, -2])  # Class label
    return np.array(X), np.array(y)

# Create sequences for CNN and LSTM
X_train_seq, y_train_seq = create_sequences(train_df, SEQ_LENGTH)
X_test_seq, y_test_seq = create_sequences(test_df, SEQ_LENGTH)

# Ensure y_test_seq is properly defined
if y_test_seq is None or len(y_test_seq) == 0:
    raise ValueError("y_test_seq is empty. Ensure dataset is correctly loaded and processed.")

# Reshape for models (samples, time_steps, features)
X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], SEQ_LENGTH, 6))
X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], SEQ_LENGTH, 6))

# Define CNN Model
cnn_model = keras.Sequential([
    keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(SEQ_LENGTH, 6)),
    keras.layers.MaxPooling1D(pool_size=2, strides=1),
    keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compile and Train CNN Model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cnn = cnn_model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32, validation_data=(X_test_seq, y_test_seq))

# Define LSTM Model
lstm_model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 6)),
    keras.layers.LSTM(64),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compile and Train LSTM Model
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_lstm = lstm_model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32, validation_data=(X_test_seq, y_test_seq))

# Evaluate Models
y_pred_cnn = cnn_model.predict(X_test_seq).argmax(axis=1)
y_pred_lstm = lstm_model.predict(X_test_seq).argmax(axis=1)

# Ensure y_test_seq is correctly shaped for confusion matrix
y_test_seq = y_test_seq[:len(y_pred_cnn)]  # Adjust size if necessary

cnn_report = classification_report(y_test_seq, y_pred_cnn, target_names=label_encoder.classes_)
lstm_report = classification_report(y_test_seq, y_pred_lstm, target_names=label_encoder.classes_)

# Print Evaluation Metrics
print("\nCNN Classification Report:\n", cnn_report)
print("\nLSTM Classification Report:\n", lstm_report)

# Plot Accuracy Comparison
plt.figure(figsize=(6, 4))
plt.plot(history_cnn.history['accuracy'], label='CNN Train Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Test Accuracy')
plt.plot(history_lstm.history['accuracy'], label='LSTM Train Accuracy', linestyle='dashed')
plt.plot(history_lstm.history['val_accuracy'], label='LSTM Test Accuracy', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('CNN vs LSTM Model Accuracy')
plt.show()

# Additional Graphs: Loss Comparison
plt.figure(figsize=(6, 4))
plt.plot(history_cnn.history['loss'], label='CNN Train Loss')
plt.plot(history_cnn.history['val_loss'], label='CNN Test Loss')
plt.plot(history_lstm.history['loss'], label='LSTM Train Loss', linestyle='dashed')
plt.plot(history_lstm.history['val_loss'], label='LSTM Test Loss', linestyle='dashed')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('CNN vs LSTM Model Loss')
plt.show()

# Confusion Matrices for CNN and LSTM
# Plot CNN Confusion Matrix
plt.figure(figsize=(6, 4))
cnn_cm = confusion_matrix(y_test_seq, y_pred_cnn)
sns.heatmap(cnn_cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('CNN Confusion Matrix')
plt.show()

# Plot LSTM Confusion Matrix
plt.figure(figsize=(6, 4))
lstm_cm = confusion_matrix(y_test_seq, y_pred_lstm)
sns.heatmap(lstm_cm, annot=True, fmt='d', cmap='Reds', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('LSTM Confusion Matrix')
plt.show()
