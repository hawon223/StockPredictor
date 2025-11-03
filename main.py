import FinanceDataReader as fdr
import datetime
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from matplotlib import font_manager, rc

base_path = os.path.abspath(os.path.dirname(__file__))
font_path = os.path.join(base_path, 'fonts', 'malgun.ttf')

today = datetime.datetime.today()

# 삼성전자 (005930) 주가 불러오기
df = fdr.DataReader("005930", start="2020-01-01", end=today)

df['Target'] = df['Close'].shift(-1)
df = df.dropna()

X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Change']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

model = XGBRegressor(n_estimators = 10000, learning_rate = 0.01, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8, random_state = 42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(r2)
print(mae)


font_manager.fontManager.addfont(font_path)
rc('font', family='Malgun Gothic')

plt.figure(figsize=(12, 6))

plt.plot(y_test.values, label='실제 주가', color='blue', linewidth=2)
plt.plot(y_pred, label='예측 주가', color='red', linestyle='--', linewidth=2)

plt.title('실제 주가 vs 예측 주가 (XGBoost)', fontsize=16)
plt.xlabel('일자 (테스트 데이터 순서)', fontsize=12)
plt.ylabel('주가 (원)', fontsize=12)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
