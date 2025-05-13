import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ file CSV
data = pd.read_csv('FINAL.csv')

# Chuyển đổi 'Walc' thành nhị phân
data['Walc_binary'] = data['Walc'].apply(lambda x: 1 if x > 2 else 0)

# Biến độc lập và phụ thuộc
X = data[['freetime']]
y = data['Walc_binary']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Tính toán và in độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác của mô hình là: {accuracy * 100:.2f}%")

# Tạo lưới giá trị 'freetime' từ min đến max và chuyển thành DataFrame
x_min, x_max = X['freetime'].min() - 1, X['freetime'].max() + 1
xx = np.linspace(x_min, x_max, num=300).reshape(-1, 1)
xx_df = pd.DataFrame(xx, columns=['freetime'])

# Dự đoán xác suất và lớp trên lưới giá trị
pred_probs = model.predict_proba(xx_df)[:, 1]
pred_boundary = model.predict(xx_df)

# Vẽ đường quyết định
plt.figure(figsize=(10, 6))
plt.plot(xx_df, pred_boundary, color='red', label='Decision Boundary')
plt.scatter(X_train, y_train, c='blue', marker='o', label='Train Data')
plt.scatter(X_test, y_test, c='green', marker='x', label='Test Data')
plt.xlabel('freetime')
plt.ylabel('Walc_binary (Tiêu thụ rượu thấp/cao)')
plt.title('Decision Boundary for freetime vs Walc_binary with Random Forest')
plt.legend()
plt.grid(True)
plt.show()

# Vẽ bản đồ nhiệt xác suất
plt.figure(figsize=(10, 6))
plt.scatter(xx_df, pred_probs, c=pred_probs, cmap='coolwarm', label='Probability', edgecolor='k')
plt.colorbar(label='Probability of High Alcohol Consumption')
plt.scatter(X_train, y_train, c='blue', marker='o', label='Train Data')
plt.scatter(X_test, y_test, c='green', marker='x', label='Test Data')
plt.xlabel('freetime')
plt.ylabel('Probability of High Alcohol Consumption')
plt.title('Random Forest: Probability Heatmap')
plt.legend()
plt.grid(True)
plt.show()
