import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

st.set_page_config(
    page_title="老年T2DM骨质疏松风险预测",
    page_icon="🦴",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 860px; padding-top: 1.5rem; padding-bottom: 3rem;}
    div.stButton > button {width: 100%; height: 3rem; font-size: 1.05rem; font-weight: 600;}
    .small-note {color: #666; font-size: 0.92rem; line-height: 1.6;}
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = ["BMI", "PTH", "Ca_mmol_L", "VitD25OH", "TC_mmol_L"]
DISPLAY_NAMES = {
    "BMI": "BMI",
    "PTH": "PTH",
    "Ca_mmol_L": "血钙（Ca）",
    "VitD25OH": "25(OH)D",
    "TC_mmol_L": "总胆固醇（TC）",
}
FINAL_THRESHOLD = 0.479535
MODEL_PATH = Path(__file__).resolve().parent / "Base_XGB_model.json"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "未找到 Base_XGB_model.json。请先将最终XGBoost模型导出为JSON并上传到网页项目根目录。"
        )
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    return booster

try:
    booster = load_model()
except Exception as e:
    st.error(f"模型加载失败：{e}")
    st.stop()

st.title("🦴 老年2型糖尿病并发骨质疏松风险预测")
st.caption("基于最终冻结的 Base-XGBoost 模型")

with st.expander("📌 模型信息", expanded=False):
    st.markdown(
        f"""
        - **算法：** XGBoost
        - **预测特征：** BMI、PTH、Ca、25(OH)D、TC
        - **模型阳性阈值：** {FINAL_THRESHOLD:.6f}
        - **训练集 Nested-CV OOF AUC：** 0.858
        - **独立留出测试集 AUC：** 0.839
        - 本模型目前属于**内部验证后的研究型临床辅助工具**，尚不能替代DXA诊断或临床医生综合判断。
        """
    )

st.info("请输入患者当前指标。所有变量必须使用与模型训练数据完全一致的单位。")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        bmi = st.number_input("BMI（kg/m²）", min_value=0.0, value=24.0, step=0.1, format="%.2f")
        pth = st.number_input(
            "PTH（pg/mL）",
            min_value=0.0,
            value=50.0,
            step=0.1,
            format="%.2f",
        )
        ca = st.number_input("血钙 Ca（mmol/L）", min_value=0.0, value=2.30, step=0.01, format="%.3f")

    with col2:
        vitd = st.number_input(
            "25(OH)D（ng/mL）",
            min_value=0.0,
            value=20.0,
            step=0.1,
            format="%.2f",
        )
        tc = st.number_input("总胆固醇 TC（mmol/L）", min_value=0.0, value=5.0, step=0.1, format="%.2f")

    submitted = st.form_submit_button("开始预测")

if submitted:
    input_df = pd.DataFrame([[bmi, pth, ca, vitd, tc]], columns=FEATURES)
    dmatrix = xgb.DMatrix(input_df, feature_names=FEATURES)
    probability = float(booster.predict(dmatrix)[0])

    st.divider()
    st.subheader("预测结果")

    # ============================================================
    # 预测结果展示
    # ============================================================

    st.subheader("预测结果")

    st.metric(
        "预测骨质疏松概率",
        f"{probability * 100:.1f}%"
    )

    st.progress(
        float(np.clip(probability, 0, 1))
    )

    if probability >= FINAL_THRESHOLD:

        st.error(
            "### 🔴 骨质疏松高风险\n\n"
            "该患者的模型预测概率高于风险判定阈值，"
            "建议结合DXA检查、临床症状、既往史及医生综合评估进一步判断。"
        )

    else:

        st.success(
            "### 🟢 骨质疏松低风险\n\n"
            "该患者的模型预测概率低于风险判定阈值。"
            "低风险结果不能完全排除骨质疏松，仍需结合临床情况综合判断。"
        )

    st.caption(
        f"模型风险判定阈值：{FINAL_THRESHOLD * 100:.1f}%"
    )

    with st.expander("🔎 查看本次预测的个体SHAP解释", expanded=False):
        contrib = booster.predict(dmatrix, pred_contribs=True)[0]
        shap_values = contrib[:-1]
        explanation = pd.DataFrame({
            "特征": [DISPLAY_NAMES[x] for x in FEATURES],
            "输入值": [bmi, pth, ca, vitd, tc],
            "SHAP贡献": shap_values,
        })
        explanation["绝对贡献"] = explanation["SHAP贡献"].abs()
        explanation["方向"] = np.where(
            explanation["SHAP贡献"] > 0,
            "推动预测向OP方向",
            np.where(explanation["SHAP贡献"] < 0, "推动预测向非OP方向", "贡献接近0"),
        )
        explanation = explanation.sort_values("绝对贡献", ascending=False).reset_index(drop=True)
        st.dataframe(
            explanation[["特征", "输入值", "SHAP贡献", "方向"]],
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(explanation.set_index("特征")[["SHAP贡献"]])
        st.caption(
            "SHAP>0表示该特征在本次个体预测中推动模型输出向OP方向；"
            "SHAP<0表示推动向非OP方向。SHAP解释的是模型行为，不代表因果关系。"
        )

    with st.expander("📋 输入指标", expanded=False):
        display_input = pd.DataFrame({
            "指标": ["BMI", "PTH", "Ca", "25(OH)D", "TC"],
            "数值": [bmi, pth, ca, vitd, tc],
        })
        st.dataframe(display_input, use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    """
    <div class="small-note">
    <b>重要声明：</b><br>
    本工具基于研究队列建立并完成内部留出验证，目前尚未完成独立外部验证。
    输出结果仅用于科研及临床辅助风险评估，不能作为骨质疏松确诊依据，
    不能替代DXA检查、专业医疗建议或医生综合判断。请勿仅依据本工具结果改变治疗方案。
    </div>
    """,
    unsafe_allow_html=True,
)
