import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
# Đọc dữ liệu từ file CSV
data = pd.read_csv('FINAL.csv')

# Chuyển đổi biến Walc thành nhị phân (1: tiêu thụ cao, 0: tiêu thụ thấp)
data['Walc_binary'] = data['Walc'].apply(lambda x: 0 if x <= 2 else 1)

# Lựa chọn biến độc lập và phụ thuộc
X = data[['freetime']]  # Chỉ sử dụng biến freetime làm độc lập
y = data['Walc_binary']  # Sử dụng Walc_binary làm phụ thuộc

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Xây dựng và huấn luyện mô hình Cây Quyết Định
model = DecisionTreeClassifier(random_state=42,class_weight='balanced')
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

# Vẽ biểu đồ cây quyết định
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=['freetime'], class_names=['Low', 'High'], filled=True, rounded=True, fontsize=8)
plt.title('Decision Tree for freetime vs Walc_binary')
plt.show()
# Tạo lưới các giá trị freetime từ min đến max
x_min, x_max = X['freetime'].min() - 1, X['freetime'].max() + 1
xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)
# Chuyển đổi xx thành DataFrame với tên cột tương ứng
xx_df = pd.DataFrame(xx, columns=['freetime'])
y_pred_boundary = model.predict(xx_df)
# Vẽ sơ đồ đường quyết định
plt.figure(figsize=(10, 6))
plt.plot(xx, y_pred_boundary, color='red', label="Decision Boundary")
plt.scatter(X_train, y_train, c='blue', marker='o', label="Train Data")
plt.scatter(X_test, y_test, c='green', marker='x', label="Test Data")
plt.title('Decision Boundary for freetime vs Walc_binary')
plt.xlabel('freetime')
plt.ylabel('Walc_binary (Low/High)')
plt.legend()
plt.show()
