import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# baca file csv
data = pd.read_csv('Student_Performance.csv')

# buat variabel durasi waktu belajar dan nilai
x = data['Hours Studied'].values.reshape(-1, 1)
y = data['Performance Index'].values

# Model Linear
model_linear = LinearRegression()
model_linear.fit(x, y)
prediksi_linear_y = model_linear.predict(x)

# Model Eksponensial menggunakan log dari x
log_x = np.log(x + 1)  
model_eksponensial = LinearRegression()
model_eksponensial.fit(log_x, y)
prediksi_eksponensial_y = model_eksponensial.predict(log_x)

# Plot grafik titik data dan hasil regresi dengan matplotlib
plt.figure(figsize=(16, 16))

plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, prediksi_linear_y, color='red', label='Regresi Linear')
plt.xlabel('Lama Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Prediksi Regresi Linear')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, prediksi_eksponensial_y, color='red', label='Regresi Eksponensial')
plt.xlabel('Lama Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Prediksi Regresi Eksponensial')
plt.legend()

plt.show()

# Menghitung galat dengan akar rata2 kuadrat
rms_model_linear = np.sqrt(mean_squared_error(y, prediksi_linear_y))
rms_model_eksponensial = np.sqrt(mean_squared_error(y, prediksi_eksponensial_y))

print(f'Galat dari Regresi Linear: {rms_model_linear}')
print(f'Galat dari Regresi Eksponensial: {rms_model_eksponensial}')