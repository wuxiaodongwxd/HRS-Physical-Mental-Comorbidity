import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
import sys

# 忽略常见的无关警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 兼容某些环境里已弃用的 np.bool
if not hasattr(np, 'bool'):
    np.bool = bool

# 页面配置
st.set_page_config(
    page_title="Physical-Mental Comorbidity Risk Assessment Model in IndiaI",
    page_icon="🩺",
    layout="wide"
)

# 训练模型使用的特征（顺序必须与训练一致）
FEATURES = [
    "self_rate_health", "pain", "adl","marriage", "score",
    "household_income", "ph_activities", "age",
    "gender", "working_status"
]

# 英文显示标签（仅影响界面显示，不影响编码）
FEATURE_LABELS = {
    "self_rate_health": "Self-reported health score",
    "pain": "Pain troubles",
    "adl": "Activities of daily living limitations (ADLs)",
    "marriage": "Marital status",
    "score": "Cognitive function score (normalized)",
    "household_income": "Household income",
    "ph_activities": "Physical activity level",
    "age": "Age",
    "gender": "Gender",
    "working_status": "Working status",
}

# 字段说明（侧栏说明文字）
FEATURE_DESC = {
    "self_rate_health": "Likert 1–5: Poor/Fair/Good/Very Good/Excellent (encoded 1–5)",
    "pain": "0/1 map to No/Yes",
    "adl": "0/1 map to No/Yes (any ADLs)",
    "marriage": "0/1 map to Married/Single",
    "score": "Model expects normalized [0,1]; the app converts raw score to normalized",
    "household_income": "0/1/2/3 map to Low/Low-middle/High-middle/High",
    "ph_activities": "0/1/2 map to Inactive/Moderate/Active",
    "age": "Age (Range from 50-106)",
    "gender": "0/1 map to Female/Male",
    "working_status": "0/1 map to No/Yes",
}

# 选项集合与显示格式化函数
YES_NO_OPTIONS = [0, 1]
YES_NO_FMT = lambda x: "No" if x == 0 else "Yes"

MARRIAGE_OPTIONS = [0,1]
MARRIAGE_FMT = lambda x: 'Married' if x == 0 else "Single"

PHA_OPTIONS = [0,1,2]
PHA_FMT = lambda x: {0: "Inactive", 1: "Moderate", 2: "Active"}[x]

LIKERT5_OPTIONS = [1, 2, 3, 4, 5]
LIKERT5_FMT = lambda x: {1: "Poor", 2: "Fair", 3: "Good", 4: "Very good", 5: "Excellent"}[x]

INCOME_OPTIONS = [0, 1, 2, 3]
INCOME_FMT = lambda x: {0: "Low", 1: "Low-middle", 2: "High-middle", 3: "High"}[x]

GENDER_OPTIONS = [0, 1]
GENDER_FMT = lambda x: "Female" if x == 0 else "Male"

# 加载模型；为部分环境提供 numpy._core 兼容兜底
@st.cache_resource
def load_model():
    model_path = 'hrs_result.pkl'
    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as e:
        if 'numpy._core' in str(e):
            import numpy as _np
            sys.modules['numpy._core'] = _np.core
            sys.modules['numpy._core._multiarray_umath'] = _np.core._multiarray_umath
            sys.modules['numpy._core.multiarray'] = _np.core.multiarray
            sys.modules['numpy._core.umath'] = _np.core.umath
            return joblib.load(model_path)
        raise


def main():
    st.sidebar.title("HRS")
    st.sidebar.markdown(
        "- Predicts risk of physical-mental comorbidity using 10 features.\n"
        "- Binary classification model (Random Forest).\n"
        "- Cognitive function score is entered as a raw value and normalized internally."
    )

    # 侧栏：展开的“特征与说明”
    with st.sidebar.expander("Features Notes"):
        for k in FEATURES:
            st.markdown(f"- {FEATURE_LABELS.get(k,k)}: {FEATURE_DESC.get(k,'')}")

    # 归一化用的原始分值范围（固定值；UI 不展示）
    SCORE_RAW_MIN = 0
    SCORE_RAW_MAX = 35

    # Load model
    try:
        model = load_model()
        st.sidebar.success("Model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        return

    # 页面标题与说明
    st.title("Physical-Mental Comorbidity Risk Assessment Model in the United States")
    st.markdown("Enter the inputs below and click Predict.")

    # 三列布局：分组输入控件
    col1, col2, col3 = st.columns(3)

    with col1:
        self_rate_health = st.selectbox(
            FEATURE_LABELS['self_rate_health'], LIKERT5_OPTIONS, format_func=LIKERT5_FMT
        )
        pain = st.selectbox(
            FEATURE_LABELS['pain'], YES_NO_OPTIONS, format_func=YES_NO_FMT
        )
        adl = st.selectbox(
            FEATURE_LABELS['adl'], YES_NO_OPTIONS, format_func=YES_NO_FMT
        )
        marriage = st.selectbox(
            FEATURE_LABELS['marriage'], MARRIAGE_OPTIONS, format_func=MARRIAGE_FMT
        )

    with col2:
        # 认知原始分值（滑块输入）
        score_raw = st.slider(
            "Cognitive function score (raw)",
            min_value=int(SCORE_RAW_MIN),
            max_value=int(SCORE_RAW_MAX),
            value=int(SCORE_RAW_MIN),
            step=1,
        )
        # 转为 [0,1] 的归一化分值
        score_norm = (score_raw - SCORE_RAW_MIN - 1) / (SCORE_RAW_MAX - SCORE_RAW_MIN -1)
        household_income = st.selectbox(
            FEATURE_LABELS['household_income'], INCOME_OPTIONS, format_func=INCOME_FMT
        )
        ph_activities = st.selectbox(
            FEATURE_LABELS['ph_activities'], PHA_OPTIONS, format_func=PHA_FMT
        )

    with col3:
        age = st.number_input(
            FEATURE_LABELS['age'], min_value=50, max_value=106, value=50, step=1
        )
        gender = st.selectbox(
            FEATURE_LABELS['pain'], GENDER_OPTIONS, format_func=GENDER_FMT
        )
        working_status = st.selectbox(
            FEATURE_LABELS['working_status'], YES_NO_OPTIONS, format_func=YES_NO_FMT
        )

    if st.button("Predict"):
        # 按训练顺序组装输入行
        row = [
            self_rate_health, pain, adl, marriage,
            score_norm, household_income, ph_activities,
            age, gender, working_status
        ]
        input_df = pd.DataFrame([row], columns=FEATURES)

        try:
            proba = model.predict_proba(input_df)[0]
            pred = int(model.predict(input_df)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # 提示预测类别与概率
        st.subheader("Prediction Result")
        st.markdown(
            f"Based on feature values, predicted possibility of physical-mental comorbidity is: <span style='color:red;'>{proba[1] * 100:.2f}%</span>  \n"
            "When using this model to evaluate the risk of physical–mental comorbidity, "
            "we recommend that the optimal threshold value be set at 27.590%.  \n"
            "Please note: This prediction is generated by a machine learning model to assist your decision-making. "
            "It should not replace your professional judgment in evaluating the patient.",
            unsafe_allow_html=True
        )

        # SHAP 可解释性
        st.write("---")
        st.subheader("Explainability SHAP Force Plot")
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(input_df)

            # 兼容不同 shap 版本的返回格式
            if isinstance(sv, list):
                shap_value = np.array(sv[1][0])  # class 1 contribution
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            elif isinstance(sv, np.ndarray) and sv.ndim == 2:
                shap_value = sv[0]
                expected_value = explainer.expected_value
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                shap_value = sv[0, :, 1]
                expected_value = explainer.expected_value[1]
            else:
                raise RuntimeError("Unrecognized SHAP output format")

            # 力导向图（Force Plot）
            try:
                force_plot = shap.force_plot(
                    expected_value,
                    shap_value,
                    input_df.iloc[0],
                    feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURES],
                    matplotlib=True,
                    show=False,
                    figsize=(20, 3)
                )
                st.pyplot(force_plot)
            except Exception as e:
                st.error(f"Force plot failed: {e}")
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")




if __name__ == "__main__":
    main()
