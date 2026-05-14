# Code-storage
엔지니어링 산업 경진대회 머신러닝 코드 
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 시뮬레이션용 가상 센서 데이터 생성 (Data Generation)
# ==========================================
def generate_synthetic_data(num_samples=10000):
    np.random.seed(42)
    
    # 기본 차량 데이터 생성
    speed = np.random.uniform(0, 110, num_samples) # 차량 속도 (0~110 km/h)
    payload = np.random.choice([0, 10, 20], num_samples) # 적재 하중 (0t, 10t, 20t)
    weather_v2x = np.random.choice([0, 1], num_samples, p=[0.8, 0.2]) # 0: 정상, 1: 악천후(빙결/폭우)
    
    # 카메라 및 레이더 거리 데이터 (목표물 거리 10m ~ 100m 가정)
    D_camera = np.random.uniform(10, 100, num_samples)
    
    # 정상 상태의 노이즈 생성 (문서 기반)
    # 카메라 태생 오차 5~8% + 피칭(Pitching)으로 인한 오차 3~5%
    base_camera_error = np.random.uniform(0.05, 0.08, num_samples)
    pitching_noise = np.random.uniform(0.03, 0.05, num_samples) 
    
    # 실제 오염 발생 여부 (Target Variable)
    # 악천후일수록, 적재 하중이 높을수록(거동 불안정) 오염 확률 증가
    contamination_prob = 0.05 + (weather_v2x * 0.3) + (payload / 100)
    is_contaminated = np.random.binomial(1, np.clip(contamination_prob, 0, 1))
    
    # 레이더 거리 산출 (오염 시 오차율 급증 반영)
    D_radar = D_camera.copy()
    for i in range(num_samples):
        total_noise_margin = base_camera_error[i] + pitching_noise[i]
        if is_contaminated[i] == 1:
            # 오염 발생 시 15%~40%의 무작위 오차 발생
            error_multiplier = np.random.uniform(0.15, 0.40)
        else:
            # 정상 주행 시 물리적 노이즈만 반영
            error_multiplier = total_noise_margin
            
        # 오차를 더하거나 빼서 레이더 값 생성
        D_radar[i] = D_camera[i] * (1 + np.random.choice([-1, 1]) * error_multiplier)
        
    df = pd.DataFrame({
        'Speed': speed,
        'Payload': payload,
        'Weather_V2X': weather_v2x,
        'D_camera': D_camera,
        'D_radar': D_radar,
        'Pitching_Noise': pitching_noise,
        'Target_Contaminated': is_contaminated # 1: 오염됨(클리닝 필요), 0: 정상(노이즈)
    })
    
    # 센서 오차율(E) 실시간 산출 공식 적용
    df['Error_Rate'] = (abs(df['D_camera'] - df['D_radar']) / df['D_camera']) * 100
    
    return df

df = generate_synthetic_data()

# ==========================================
# 2. LightGBM 모델 학습 (Machine Learning)
# ==========================================
# Feature 및 Target 분리
X = df[['Speed', 'Payload', 'Weather_V2X', 'D_camera', 'D_radar', 'Pitching_Noise', 'Error_Rate']]
y = df['Target_Contaminated']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM 분류기 초기화 (경량화 추론 모델)
# 연산량 감소를 위해 num_leaves와 max_depth를 보수적으로 설정
lgbm_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,  # 문서의 Gradient Error Flow 보정률 η=0.1
    max_depth=5,
    num_leaves=31,
    objective='binary',
    random_state=42
)

lgbm_model.fit(X_train, y_train)

# 모델 성능 평가
y_pred = lgbm_model.predict(X_test)
print("--- LightGBM 모델 성능 평가 ---")
print(classification_report(y_test, y_pred))


# ==========================================
# 3. 통합 제어 아키텍처 알고리즘 (Inference & Control)
# ==========================================
def intelligent_cleaning_controller(real_time_data, model):
    """
    10ms 단위로 들어오는 실시간 데이터를 받아 클리닝 작동 여부를 판단하는 폐루프 제어 함수
    """
    error_rate = real_time_data['Error_Rate']
    speed = real_time_data['Speed']
    weather = real_time_data['Weather_V2X']
    
    # 1단계: V2X 기상 데이터 기반 동적 임계치 오버라이드
    if weather == 1:
        dynamic_threshold = 30.0 # 악천후 시 오작동 방지를 위해 30%로 대폭 상향
        retarder_standby = True  # 공주 거리 확보를 위한 리타더 대기
    else:
        # 정상 기상 시, LightGBM 모델이 노이즈(0)인지 오염(1)인지 판단
        # 문서의 "기본 임계치 보수적 10% 최적화" 로직을 모델의 확률로 대체
        prediction = model.predict(real_time_data.to_frame().T)[0]
        dynamic_threshold = 10.0 if prediction == 1 else 100.0 # 오염으로 판단 시 임계치 10%로 민감도 상승
        retarder_standby = False
        
    # 2단계: 오차율 비교 및 속도 구간(Zone) 판별을 통한 분사 지속시간 산출
    if error_rate >= dynamic_threshold:
        if speed <= 50:
            zone = "Zone 1 (서행/도심)"
            delay_time = 4.0
        elif 50 < speed <= 80:
            zone = "Zone 2 (간선도로)"
            delay_time = 2.0
        else:
            zone = "Zone 3 (고속도로)"
            delay_time = 0.2 # Fail-Safe 즉각 작동
            
        print(f"[명령] 오염 감지 확정! (오차율: {error_rate:.1f}%) -> {zone} 판단 기준 적용.")
        print(f"[동작] {delay_time}초 대기 후 솔레노이드 밸브 개방 (공기 분사)")
        if retarder_standby or speed > 80:
            print("[제어] 리타더 선제적 제동 개입 활성화 (Linear 감속 시작)")
            
    else:
        print(f"[상태] 정상 주행 중 (오차율: {error_rate:.1f}%) -> 시스템 대기 (10ms 후 재평가)")

# --- 실시간 제어 시뮬레이션 ---
print("\n--- 실시간 통합 제어 시뮬레이션 ---")
# 테스트 케이스 1: 85km/h 주행, 맑음, 오염 발생 (오차 18%)
test_case_1 = X_test.iloc[0]
test_case_1['Speed'] = 85
test_case_1['Weather_V2X'] = 0
test_case_1['Error_Rate'] = 18.5
test_case_1['D_camera'] = 50.0
test_case_1['D_radar'] = 40.75 # 18.5% 차이

intelligent_cleaning_controller(test_case_1, lgbm_model)
