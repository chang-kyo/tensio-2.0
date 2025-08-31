import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# 모델 및 스케일러 로딩
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

# 코르티솔 → 스트레스 수치 변환 함수
def cortisol_to_stress(cortisol):
    return 10 / (1 + np.exp(-0.5 * (cortisol - 15)))

# 스트레스 권고 문구 반환 함수
def get_stress_advice(stress):
    if stress < 1:
        return "매우 안정적인 상태입니다. 현재 상태를 유지하세요."
    elif stress < 2:
        return "스트레스 수치가 매우 낮습니다. 무리 없이 활동 가능합니다."
    elif stress < 3:
        return "안정적인 상태입니다. 가벼운 운동이나 집중 활동에 적합합니다."
    elif stress < 4:
        return "일상적인 스트레스 수준입니다. 특별한 조치는 필요하지 않습니다."
    elif stress < 5:
        return "다소 긴장된 상태입니다. 짧은 휴식이나 산책이 도움이 될 수 있습니다."
    elif stress < 6:
        return "스트레스가 쌓이기 시작했습니다. 심호흡이나 가벼운 이완을 추천합니다."
    elif stress < 7:
        return "주의가 필요한 스트레스 수준입니다. 30분 정도의 휴식을 고려해 보세요."
    elif stress < 8:
        return "높은 스트레스 상태입니다. 조용한 환경에서 충분한 휴식이 필요합니다."
    elif stress < 9:
        return "심리적 부담이 큰 상태입니다. 명상이나 수면 등 적극적인 이완이 필요합니다."
    else:
        return "매우 높은 스트레스입니다. 1~2시간 이상의 안정과 회복 시간을 확보하세요."

# 제목
st.title("스트레스 예측 프로그램 TENSIO")
st.write("수면, 운동, 이벤트 입력에 따른 코르티솔 농도와 스트레스 수치를 예측합니다.")

# 입력 슬라이더
sleep = st.slider("수면 시간 (시간)", 0.0, 7.5, step=0.5)
exercise = st.slider("운동 시간 (시간)", 0.0, 3.0, step=0.5)
event = st.slider("이벤트 점수 (부정: -5, 긍정: 5)", -5, 5, step=1)

# 입력 데이터 처리
X = np.array([[sleep, exercise, event]])
X_scaled = scaler.transform(X)

# 예측
cortisol = model.predict(X_scaled)[0][0]
cortisol = max(cortisol, 0)  # 코르티솔 음수 방지
stress = cortisol_to_stress(cortisol)

# 출력 결과
st.subheader("예측 결과")
st.write(f"코르티솔 농도: {cortisol:.2f}")
st.write(f"스트레스 수치 (0~10): {stress:.2f}")

# 권고 문구 출력
advice = get_stress_advice(stress)
st.markdown(f"**권장사항:** {advice}")

st.markdown("---")
st.subheader("당신이 모르던 스트레스 이야기, TENSIO")
st.markdown("""
- **TENSIO에 대해**: TENSIO는 딥러닝을 기반으로 한 수치 예측 프로그램입니다. 스트레스에 영향을 미치는 외부 요인들의 입력을 통해 간접적으로 스트레스 상태를 예측하고, 개인이 자기관리에 참고할 수 있도록 도울 수 있습니다.
- **TENSIO의 목적**: 본인에게 채감되는 스트레스와 실제 스트레스 수치는 차이가 날 수 있습니다. TENSIO를 통해 해당 차이를 인지하고 적절한 스트레스 관리 방법을 알 수 있습니다.
- **스트레스 수치 (0~10)**: 코르티솔 수치를 바탕으로 스트레스 수준을 정규화한 값입니다. 수치가 높을수록 심리적, 생리적 긴장도가 크다는 것을 의미합니다.
- **코르티솔(Cortisol)**: 스트레스 상황에서 분비되는 대표적인 호르몬입니다. 적절한 수준은 에너지 조절과 생존 반응에 중요하지만, 과도한 수치는 면역력 저하나 만성 질환으로 이어질 수 있습니다.
- **정상 코르티솔 범위**: 사람에 따라 다르지만, 일반적으로 아침 기준 10~20이 정상 범위로 여겨집니다.
""")

st.markdown(
    """
    <div style='text-align: center; font-size: 12px; color: gray; margin-top: 50px;'>
        배포자: <b>여준호</b> |  
        Instagram: <a href='https://instagram.com/yeojiahp_' target='_blank'>@yeojiahp_</a> |  
        전화: 010-9941-3609 |  
        이메일: tomgraz@naver.com
    </div>
    """,
    unsafe_allow_html=True
)
