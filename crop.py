import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\2 year\\myprojects\\crop yield\\archive (1)\\Crop_recommendation.csv")


print("Dataset Shape:", data.shape)
print("Columns:", data.columns)
print("First 5 rows:\n", data.head())
print("\nMissing Values:\n", data.isnull().sum())




from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

X = data.drop(columns=['label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(max_depth=10, criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

train_accuracy = clf.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)

indices = np.arange(len(y_test))





y_test_array = np.array(y_test)
y_pred_array = np.array(y_pred)
indices = np.arange(len(y_test_array))

sampled_indices = np.random.choice(indices, size=100, replace=False)

plt.figure(figsize=(12, 6))
plt.scatter(sampled_indices, y_test_array[sampled_indices],
            label='Actual Labels', color='blue', alpha=0.7, marker='o')

plt.scatter(sampled_indices, y_pred_array[sampled_indices],
            label='Predicted Labels', color='red', alpha=0.7, marker='x')

plt.title('Comparison of Actual and Predicted Labels', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Label', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




def predict_crop():
    print("Enter the values for the following parameters:")
    N = float(input("Nitrogen content (N): "))
    P = float(input("Phosphorus content (P): "))
    K = float(input("Potassium content (K): "))
    temperature = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))
    ph = float(input("pH value: "))
    rainfall = float(input("Rainfall (mm): "))

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    encoded_prediction = clf.predict(input_data)
    crop_name = le.inverse_transform(encoded_prediction)

    print(f"\nThe most suitable crop for the given conditions is: {crop_name[0]}")

predict_crop()


