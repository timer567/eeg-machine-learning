# 运动想象分类

使用 MOABB 的 `Lee2019_MI` 数据接口，比较 CSP、CSSP、FBCSP 特征与 LDA、线性 SVM 分类器。
包含被试内五折实验，以及以被试为单位划分的跨被试五折实验。

```text
within-subject/{csp,cssp,fbcsp}/run.py
cross-subject/{csp,cssp,fbcsp}/run.py
results/{within-subject,cross-subject}/  # 历史 CSV 和图表
```

## 运行

为本任务创建独立虚拟环境，并从本目录安装 `python -m pip install -r requirements.txt`。
例如，在 `motor-imagery` 目录下运行：

```bash
cd within-subject/csp
python run.py
```

其他算法与协议同理进入各自目录运行，避免同名输出覆盖。
脚本通过 MOABB 获取数据，首次执行可能联网下载并建立本地缓存；下载前确认数据源使用条件和磁盘空间。
实验默认使用前 10 名被试；通道、滤波、滑窗与并行数见每份脚本的 Config。
新输出保存在运行目录的“结果”“结果_CSSP”或“结果_FBCSP”中，历史快照单独保存在 results。

## 评估范围

- 被试内结果包含各 Session 及其均值；跨被试脚本只使用 Session 1。两者的任务协议不同。
- 记录试次投票准确率与窗口准确率，比较时应使用同一指标。
- 被试内 CSSP 在同一 Session 上预选 tau，再报告相同划分下的五折结果；这不是嵌套交叉验证，存在选择偏差。
- 保留原实验逻辑与历史结果。本次未下载数据、重新训练，依赖清单不是原始环境锁文件。

数据接口与许可请查看 [MOABB](https://github.com/NeuroTechX/moabb)，数据集为 `Lee2019_MI`。
