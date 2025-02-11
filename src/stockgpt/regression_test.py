import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------
# 기존 데이터 로딩 관련 함수들 (제공해주신 코드)
# ---------------------------
def load_return_tokens_from_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    
    if 'ReturnToken' not in df.columns:
        print(f"File {file_path} does not have 'ReturnToken' column. Skipping.")
        return None

    tokens = df['ReturnToken'].dropna().tolist()
    return torch.tensor(tokens, dtype=torch.long)

def decode_return(token):
    """
    단일 토큰 인덱스(token: 0 ~ 401)를 해당하는 수익률(소수 형태)로 디코딩(복원)
    
    변환 규칙:
      - token == 0   -> -10000 basis points (-100%)
      - token == 401 -> +10000 basis points (+100%)
      - token 1 ~ 400: 해당 구간의 대표값(중간값)은
                       -10000 + (token - 1) * 50 + 25
                       basis point 단위이며, 이를 10000으로 나누어 소수 형태로 반환
    """
    if token == 0:
        r_bp = -10000
    elif token == 401:
        r_bp = 10000
    else:
        r_bp = -10000 + (token - 1) * 50 + 25
    return r_bp / 100.0  # 백분율 형태로 변환

def decode(token_list):
    """
    token_list: 정수 토큰의 리스트 (예: [196, 200, 200, 210, 210])
    
    각 토큰을 decode_return 함수를 사용해 디코딩하고,
    디코딩된 수익률의 리스트(%)를 반환.
    """
    return [round(decode_return(token), 1) for token in token_list]

# ---------------------------
# 데이터 로딩 및 전처리
# ---------------------------

# 현재 스크립트 기준의 디렉토리 설정
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
data_dir = os.path.join(project_root, "data", "model")
print(f"데이터 디렉토리 절대 경로: {data_dir}")

csv_files = glob.glob(os.path.join(data_dir, "*_with_returns_tokens.csv"))
print(f"발견된 CSV 파일 수: {len(csv_files)}")

train_data_list = []
for file_path in csv_files:
    tokens_tensor = load_return_tokens_from_file(file_path)
    block_size = 256
    if tokens_tensor is not None and len(tokens_tensor) > block_size:
        train_data_list.append(tokens_tensor)

print(f"총 {len(train_data_list)}개의 파일에서 ReturnToken 시퀀스를 로드했습니다.")

# (필요하면) train/val split은 여기서 별도로 할 수 있지만,
# 회귀 모델을 위한 데이터셋은 모든 파일의 데이터를 합쳐서 구성합니다.

# ---------------------------
# 회귀 문제 데이터셋 구성
# ---------------------------

window_size = 40  # 과거 40, 70개 수익률을 이용해 다음 수익률을 예측
X_data = []
y_data = []

# 각 파일의 토큰 시퀀스에 대해
for tokens_tensor in train_data_list:
    # tensor를 리스트로 변환하고, 디코딩하여 수익률 시퀀스로 변환
    tokens_list = tokens_tensor.tolist()
    returns = decode(tokens_list)  # returns는 리스트 형태 (예: [ -95.0, -94.5, ..., 2.5, ... ])
    # 슬라이딩 윈도우 방식으로 특징(feature)와 타깃(target)을 생성
    for i in range(len(returns) - window_size):
        X_data.append(returns[i : i + window_size])
        y_data.append(returns[i + window_size])

X_data = np.array(X_data)  # shape: (n_samples, window_size)
y_data = np.array(y_data)  # shape: (n_samples,)

print("생성된 데이터셋의 shape:", X_data.shape, y_data.shape)

# ---------------------------
# 데이터 분할 및 회귀 모델 학습
# ---------------------------

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# 1. 선형 회귀
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
print("Linear Regression RMSE: {:.4f}".format(rmse_lr))

# 2. Lasso 회귀
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
print("Lasso Regression RMSE: {:.4f}".format(rmse_lasso))
