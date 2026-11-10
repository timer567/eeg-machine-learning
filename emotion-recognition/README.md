# 脑电情绪识别：特征方案对照

基于 DEAP 预处理 Python 数据，比较 EMD/PSD 特征与原始信号统计特征，在被试内预测 Valence、Arousal、Dominance、Liking 四个维度。

| 文件 | 内容 |
| --- | --- |
| engineered_features.ipynb | 原 v4，8 通道，EMD 差分特征 + 4 个频带 PSD；线性 SVM、RBF SVM、随机森林 |
| statistical_features.ipynb | 原 v5，每通道均值、标准差、最大值、最小值；不是直接输入完整波形 |
| results/ | 原存档 CSV 与图表，本次没有重新生成 |

## 运行

建议使用单独的 Python 3.10/3.11 虚拟环境；下面是准备步骤，不代表已验证的原始环境。

```bash
cd emotion-recognition
python -m venv .venv
# Windows PowerShell：.\.venv\Scripts\Activate.ps1
# macOS / Linux：source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

按 [DEAP 官方页面](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) 的访问流程自行获取 `data_preprocessed_python`，放在本目录下，使 `data_preprocessed_python/s01.dat` 等文件存在。
打开对应 Notebook 后执行全部单元格。运行输出保存在本目录，不覆盖 `results/` 中的历史快照。
原始数据不随仓库发布；只读取可信来源的 pickle 数据文件。

## 原实验协议与限制

- 标签阈值为 5；每个视频按 672 个采样点切成 12 段，在 128 Hz 下每段为 5.25 秒。代码中的“5 秒”注释是近似表述。
- 原代码对完整被试数据执行 StandardScaler 后才做五折预测，存在预处理数据泄漏。
- 交叉验证以切片为样本，同一视频的片段可能同时进入训练与验证；严格评估应按视频分组，标准化只拟合训练折。
- 混淆矩阵可视化中 Best/Worst 的准确率标题索引与对应矩阵有交换问题。存档图保留原样；读数优先核对 CSV，后续修正需要重新绘图。
- 两份 Notebook 保留原实验逻辑。历史结果不能视为独立视频或跨被试泛化能力证明。
- 本地没有 DEAP 原始文件，本次只完成代码与结果文件的静态核查。
