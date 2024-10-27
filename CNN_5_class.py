import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset
data = pd.read_csv('data.csv')

# Step 2: Remove unwanted "Unnamed" column (if it exists)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Step 3: Preprocess the data
if data.isnull().sum().sum() > 0:
    data = data.fillna(data.median())

categorical_cols = data.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    data = pd.get_dummies(data, drop_first=True)

X = data.drop(columns=['y'])
y = data['y']

# Step 4: Encode labels to categorical (for multi-class problems)
y = to_categorical(y)

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Step 6: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data to fit into Conv1D (CNN requires 3D input: samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 7: Build the CNN model
model = Sequential()

# Add Conv1D layers
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# Add more Conv1D and Pooling layers (optional)
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# Flatten the output before feeding into dense layers
model.add(Flatten())

# Fully connected layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))

# Output layer with softmax activation (for multi-class classification)
model.add(Dense(y.shape[1], activation='softmax'))

# Step 8: Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 9: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Step 10: Evaluate the model
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test_class = np.argmax(y_test, axis=1)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test_class, y_pred)
classification_rep = classification_report(y_test_class, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_rep)

# Step 11: Save results
with open('results_cnn.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Model Accuracy", f"{accuracy * 100:.2f}%"])
    writer.writerow(["Classification Report"])
    writer.writerow(classification_rep.splitlines())
    results_df = pd.DataFrame({'Actual': y_test_class, 'Predicted': y_pred})
    results_df.to_csv(file, index=False, mode='a')

print("\nModel evaluation and predictions saved to 'results_cnn.csv'")