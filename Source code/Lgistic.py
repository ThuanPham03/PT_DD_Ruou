import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Đọc dữ liệu từ file CSV
data = pd.read_csv('FINAL.csv')

# Chuyển đổi biến Walc thành nhị phân (1: tiêu thụ cao, 0: tiêu thụ thấp)
data['Walc_binary'] = data['Walc'].apply(lambda x: 0 if x <= 2 else 1)

# Lựa chọn biến độc lập và phụ thuộc
X = data[['freetime']]
y = data['Walc_binary']
#Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Xây dựng và huấn luyện mô hình hồi quy logistic
model = LogisticRegression(class_weight='balanced', C=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Ma trận nhầm lẫn
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Báo cáo phân loại
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Hệ số hồi quy và intercept
print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Tạo dữ liệu cho đường hồi quy
freetime_range = np.linspace(X['freetime'].min(), X['freetime'].max(), 300).reshape(-1, 1)
freetime_range_df = pd.DataFrame(freetime_range, columns=['freetime'])  # Chuyển đổi sang DataFrame có tên cột
# Dự đoán xác suất cho lớp 1 (tiêu thụ cao)
predictions = model.predict_proba(freetime_range_df)[:, 1]
# Tính xác suất dự đoán
y_prob = model.predict_proba(X_test)[:, 1]
# Vẽ biểu đồ scatter
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Thực tế')
plt.scatter(X_test, y_prob, color='red', label='Dự đoán (Xác suất)')
plt.xlabel('thời gian rãnh (freetime)')
plt.ylabel('Xác suất tiêu thụ rượu bia cao')
plt.title('Mối quan hệ giữa freetime và xác suất tiêu thụ rượu')
plt.legend()
plt.grid(True)
plt.show()
# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['freetime'], y=data['Walc_binary'], hue=data['Walc_binary'], palette='coolwarm', s=100)
plt.plot(freetime_range, predictions, color='red', linewidth=2, label='Logistic Regression Curve')
plt.title('Logistic Regression: freetime vs Walc_binary')
plt.xlabel('freetime')
plt.ylabel('Walc (Binary: 0=Low, 1=High)')
plt.legend()
plt.show()

