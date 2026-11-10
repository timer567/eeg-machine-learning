import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from scipy.signal import butter, filtfilt
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm
from moabb.datasets import Lee2019_MI
from moabb.paradigms import MotorImagery


# ==========================================
# 1. 配置模块
# ==========================================
class Config:
    CHANNELS = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
    FS = 100
    FMIN, FMAX = 8, 30
    FILTER_ORDER = 5
    T_START, T_WIN_LEN, T_STEP = 1.0, 1.5, 0.25
    N_WINDOWS = 5
    N_FOLDS = 5
    N_JOBS = 5
    OUTPUT_DIR = "结果"


# ==========================================
# 2. 工具模块
# ==========================================
class Utils:
    @staticmethod
    def butter_filter(data):
        nyq = 0.5 * Config.FS
        low, high = Config.FMIN / nyq, Config.FMAX / nyq
        b, a = butter(Config.FILTER_ORDER, [low, high], btype='band')
        return filtfilt(b, a, data, axis=-1)

    @staticmethod
    def get_windows(data):
        start = int(Config.T_START * Config.FS)
        step = int(Config.T_STEP * Config.FS)
        win_size = int(Config.T_WIN_LEN * Config.FS)
        return np.stack([data[:, :, start + w * step: start + w * step + win_size]
                         for w in range(Config.N_WINDOWS)], axis=1)


# ==========================================
# 3. 核心计算类
# ==========================================
class CrossSubjectExperiment:
    def __init__(self):
        self.dataset = Lee2019_MI()
        self.paradigm = MotorImagery(channels=Config.CHANNELS, resample=Config.FS,
                                     fmin=Config.FMIN, fmax=Config.FMAX)
        # 显式转换为原生 int 列表
        self.subjects = [int(s) for s in range(1, 11)]

    def load_all_data(self):
        """ 阶段 1：并行加载 Session 1 数据 """
        print(f">>> 正在并行加载 {len(self.subjects)} 名被试的 Session 1 数据...")

        def _load_single(sid):
            # 这里的 sid 已经是原生 int
            X, y, meta = self.paradigm.get_data(dataset=self.dataset, subjects=[sid])

            # 动态获取第一个 Session 的名称
            all_sessions = meta['session'].unique()
            first_sess = all_sessions[0]

            mask = meta['session'] == first_sess
            X, y = X[mask], y[mask]

            # 先重采样(get_data已做)后滤波
            X = Utils.butter_filter(X)
            return X, y, np.full(len(y), sid)

        results = Parallel(n_jobs=Config.N_JOBS)(
            delayed(_load_single)(s) for s in tqdm(self.subjects, desc="读取数据")
        )

        self.X_all = np.concatenate([r[0] for r in results])
        self.y_all = np.concatenate([r[1] for r in results])
        self.sub_labels = np.concatenate([r[2] for r in results])

    def run_cross_val(self):
        """ 阶段 2：跨被试 5 折交叉验证 """
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
        print(f"\n>>> 启动跨被试 5 折交叉验证 (5核并行)...")

        # 对 5 折进行并行
        # 注意：这里 kf.split 作用在 self.subjects 列表上
        fold_results = Parallel(n_jobs=Config.N_JOBS)(
            delayed(self._process_fold)(train_idx, test_idx)
            for train_idx, test_idx in kf.split(self.subjects)
        )

        self._format_results(fold_results)

    def _process_fold(self, train_idx, test_idx):
        # 获取对应的被试 ID
        train_subs = [self.subjects[i] for i in train_idx]
        test_subs = [self.subjects[i] for i in test_idx]

        tr_mask = np.isin(self.sub_labels, train_subs)
        te_mask = np.isin(self.sub_labels, test_subs)

        win_samples = int(Config.T_WIN_LEN * Config.FS)
        X_tr = Utils.get_windows(self.X_all[tr_mask]).reshape(-1, len(Config.CHANNELS), win_samples)
        y_tr = np.repeat(self.y_all[tr_mask], Config.N_WINDOWS)

        X_te_struct = Utils.get_windows(self.X_all[te_mask])
        y_te = self.y_all[te_mask]
        te_sub_ids = self.sub_labels[te_mask]

        # CSP 特征共享
        csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
        feat_tr = csp.fit_transform(X_tr, y_tr)

        lda = LinearDiscriminantAnalysis().fit(feat_tr, y_tr)
        svm = SVC(kernel='linear', C=1.0).fit(feat_tr, y_tr)

        fold_data = []
        for sid in test_subs:
            sub_mask = (te_sub_ids == sid)
            X_sub, y_sub = X_te_struct[sub_mask], y_te[sub_mask]

            res = {'LDA': {'cv': 0, 'cw': 0}, 'SVM': {'cv': 0, 'cw': 0}}
            for i in range(len(y_sub)):
                feat_te = csp.transform(X_sub[i])
                for name, clf in [('LDA', lda), ('SVM', svm)]:
                    preds = clf.predict(feat_te)
                    res[name]['cw'] += np.sum(preds == y_sub[i])
                    if Counter(preds).most_common(1)[0][0] == y_sub[i]:
                        res[name]['cv'] += 1

            n_t = len(y_sub)
            fold_data.append({
                'Subject': f"S{sid:02d}",
                'LDA_vote': res['LDA']['cv'] / n_t, 'LDA_win': res['LDA']['cw'] / (n_t * 5),
                'SVM_vote': res['SVM']['cv'] / n_t, 'SVM_win': res['SVM']['cw'] / (n_t * 5)
            })
        return fold_data

    def _format_results(self, fold_results):
        all_res = [item for fold in fold_results for item in fold]
        self.df_final = pd.DataFrame(all_res).sort_values('Subject')

        print("\n" + "=" * 60)
        print(f"跨被试 (Session 1) 实验结果汇总 (LDA 平均: {self.df_final['LDA_vote'].mean():.3f})")
        print(self.df_final.to_string(index=False))
        print("=" * 60)

    def visualize(self):
        if not os.path.exists(Config.OUTPUT_DIR): os.makedirs(Config.OUTPUT_DIR)

        plot_df = pd.melt(self.df_final, id_vars=['Subject'],
                          value_vars=['LDA_vote', 'SVM_vote'],
                          var_name='Classifier', value_name='Accuracy')
        plot_df['Classifier'] = plot_df['Classifier'].str.replace('_vote', '')

        plt.figure(figsize=(9, 6))
        sns.set_theme(style="whitegrid")
        sns.boxplot(x='Classifier', y='Accuracy', data=plot_df, palette="Set3", width=0.4)
        sns.stripplot(x='Classifier', y='Accuracy', data=plot_df, color="red", size=8, jitter=True, alpha=0.6)

        plt.title("Cross-Subject Performance: Session 1 (5-Fold CV)", fontsize=14)
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.ylim(0.4, 1.0)

        plt.savefig(f"{Config.OUTPUT_DIR}/跨被试_Session1_分类对比.png", dpi=300)
        self.df_final.to_csv(f"{Config.OUTPUT_DIR}/CrossSubject_Session1_Results.csv", index=False)
        print(f"\n>>> 结果已保存至 '{Config.OUTPUT_DIR}' 文件夹。")
        plt.show()


if __name__ == "__main__":
    exp = CrossSubjectExperiment()
    exp.load_all_data()
    exp.run_cross_val()
    exp.visualize()