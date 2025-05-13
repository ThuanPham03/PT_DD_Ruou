import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Đọc dữ liệu từ file CSV
data = pd.read_csv('FINAL.csv')

# Chuyển đổi Walc thành nhị phân (1: tiêu thụ cao, 0: tiêu thụ thấp)
data['Walc_binary'] = data['Walc'].apply(lambda x: 0 if x <= 2 else 1)

# Lựa chọn biến độc lập và phụ thuộc
X = data[['absences']]
y = data['Walc_binary']

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình SVM
model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Ma trận nhầm lẫn
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Báo cáo phân loại
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# In ra độ chính xác của mô hình
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))

 #Tạo lưới các giá trị absences từ min đến max
x_min, x_max = X['absences'].min() - 1, X['absences'].max() + 1
xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)
# Chuyển đổi xx thành DataFrame với tên cột tương ứng
xx_df = pd.DataFrame(xx, columns=['absences'])
y_pred_boundary = model.predict(xx_df)

plt.figure(figsize=(10, 6))
plt.scatter(X['absences'], y, c=y, cmap='viridis', edgecolors='k', label='Actual Data')
plt.plot(xx, y_pred_boundary, color='red', label='Decision Boundary (SVM)')
plt.title('SVM Decision Boundary: absences vs Walc_binary')
plt.xlabel('absences')
plt.ylabel('Walc_binary (Low/High Consumption)')
plt.legend()
plt.show()
