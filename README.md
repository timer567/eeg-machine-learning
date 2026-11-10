# EEG Machine Learning｜脑电机器学习实践

按研究任务组织的课程实验存档：脑电情绪识别与运动想象分类。

| 子项目 | 数据与任务 | 方法 | 入口 |
| --- | --- | --- | --- |
| 情绪识别 | DEAP，四个情绪维度二分类 | EMD/PSD 与统计特征，SVM / 随机森林 | [emotion-recognition](emotion-recognition/README.md) |
| 运动想象 | Lee2019_MI，被试内与跨被试分类 | CSP / CSSP / FBCSP，LDA / SVM | [motor-imagery](motor-imagery/README.md) |

两个子项目分别提供依赖、运行说明和历史结果。建议分别创建虚拟环境。
原始数据不随仓库分发；实验评估限制见各子项目 README。历史准确率未在本次发布中重新验证。
原始 Notebook 输出已清空，独立 CSV 和图表保留。详见 [来源与发布范围](NOTICE.md)。
