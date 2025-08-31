import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. 가상 데이터 생성
np.random.seed(42)
N = 1000

# 연속형 변수로 변경
sleep = np.random.uniform(0, 7.5, N)        # 수면 시간 (0 ~ 7.5 시간)
exercise = np.random.uniform(0, 3, N)       # 운동 시간 (0 ~ 3 시간)
event = np.random.randint(-5, 6, size=N)    # 이벤트 점수 (-5 ~ 5)

X = np.stack([sleep, exercise, event], axis=1)

# 2. 코르티솔 수치 생성 (기준점 유지)
cortisol = (
    15
    - 0.842760 * (sleep - 4.5)
    - 0.822440 * (exercise - 1)
    - 0.511592 * event
    + np.random.normal(0, 1.0, N)
)
cortisol = np.clip(cortisol, 0, None)

# 3. 전처리
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 학습
X_train, X_test, y_train, y_test = train_test_split(X_scaled, cortisol, test_size=0.2, random_state=42)

from tensorflow.keras.layers import Input
model = Sequential([
    Input(shape=(3,)),

    Dense(16, activation='relu', input_shape=(3,)),
    Dense(8, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, validation_split=0.2, epochs=100, verbose=1)

# 5. 저장
model.save("model.h5")
joblib.dump(scaler, "scaler.pkl")

print("✅ 모델과 스케일러 저장 완료!")
