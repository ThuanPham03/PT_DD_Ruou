import pandas as pd

# Đọc dữ liệu từ file CSV (thay thế 'data.csv' bằng tên file của bạn)
studentalcohol = pd.read_csv('student-mat.csv')

# Hiển thị thông tin về DataFrame
#studentalcohol.info()
# Hiển thị 5 dòng đầu tiên
#print(df.head())

#Lọc các dữ liệu phân loại
categorical=studentalcohol.select_dtypes(include = ["object"]).keys()
print(categorical)
#lọc các dữ liệu số lượng
quantitive=studentalcohol.select_dtypes(include = ["int64","float64"]).keys()
print(quantitive)