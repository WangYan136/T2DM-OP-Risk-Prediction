# 老年T2DM并发骨质疏松风险预测网页

## 1. 最终模型
- 算法：Base-XGBoost
- 特征：BMI、PTH、Ca、25(OH)D、TC
- 最终分类阈值：0.479535
- 训练集Nested-CV OOF AUC：0.858
- 留出测试集AUC：0.839

## 2. 先在原PyCharm机器学习项目中导出模型

把本压缩包中的 `export_final_model.py` 复制到你的原项目根目录：

```text
T2DM-OP-ML26.9.20/
├── results/
│   └── models/
│       └── Base_XGB.joblib
└── export_final_model.py
```

运行：

```bash
python export_final_model.py
```

成功后项目根目录会生成：

```text
Base_XGB_model.json
```

## 3. 网页项目最终需要的文件

上传到同一个GitHub仓库根目录：

```text
streamlit_app.py
Base_XGB_model.json
requirements.txt
.streamlit/config.toml
```

`export_final_model.py` 不必上传到网页仓库。

## 4. 本地测试

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 5. 部署到Streamlit Community Cloud

1. 新建一个GitHub仓库。
2. 上传 `streamlit_app.py`、`Base_XGB_model.json`、`requirements.txt`、`.streamlit/config.toml`。
3. 打开 https://share.streamlit.io/ 并使用GitHub登录。
4. 点击 `Create app`。
5. 选择仓库和 `main` 分支。
6. Main file path 填 `streamlit_app.py`。
7. 可设置一个容易记忆的自定义子域名。
8. 点击 `Deploy`。

部署后会得到一个公开的 `*.streamlit.app` 链接，手机和电脑都能访问。

## 6. 单位必须一致

网页输入值必须和原训练数据的单位完全一致。

已明确：
- BMI：kg/m²
- Ca：mmol/L
- TC：mmol/L

正式公开前请再次核对：
- PTH的训练数据单位
- 25(OH)D的训练数据单位

## 7. 临床使用限制

当前模型完成的是内部验证，尚无独立外部验证。网页应定位为“科研/临床辅助风险评估原型”，不能宣传为骨质疏松诊断工具，也不能替代DXA。
