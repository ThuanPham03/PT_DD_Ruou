import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
data = pd.read_csv('FINAL.csv')
data['Walc_binary'] = data['Walc'].apply(lambda x: 0 if x <= 2 else 1)

# Lựa chọn dữ liệu
X = data[['freetime','absences']]
y = data['Walc_binary']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình
model = SVC(kernel='linear', class_weight='balanced')
model.fit(X_train, y_train)

# Vẽ biểu đồ 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test['freetime'], X_test['absences'], y_test, c=y_test, cmap='viridis', edgecolor='k', label='Test Data')

ax.set_xlabel('Freetime')
ax.set_ylabel('Absences')
ax.set_zlabel('Walc_binary (High/Low)')

plt.title('3D Visualization: Absences vs Freetime vs Walc_binary')
plt.show()



