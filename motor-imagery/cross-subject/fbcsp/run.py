import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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
    # 8-30Hz 频段切分
    BANDS = [(8, 12), (12, 18), (18, 24), (24, 30)]
    FILTER_ORDER = 5
    T_START, T_WIN_LEN, T_STEP = 1.0, 1.5, 0.25
    N_WINDOWS = 5
    N_FOLDS = 5
    N_JOBS = 5
    K_FEATURES = 6  # 每个频段4个CSP分量，共16个，筛选最强的6个
    OUTPUT_DIR = "结果_FBCSP"


# ==========================================
# 2. 工具模块
# ==========================================
class Utils:
    @staticmethod
    def butter_bandpass(data, lowcut, highcut):
        nyq = 0.5 * Config.FS
        low, high = lowcut / nyq, highcut / nyq
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
# 3. FBCSP 核心计算类
# ==========================================
class FBCSPCrossSubject:
    def __init__(self):
        self.dataset = Lee2019_MI()
        self.paradigm = MotorImagery(channels=Config.CHANNELS, resample=Config.FS, fmin=7, fmax=31)
        self.subjects = [int(s) for s in range(1, 11)]

    def load_session1_data(self):
        """ 并行加载所有被试的 Session 1 数据 (仅降采样，滤波在后续子带处理) """
        print(f">>> 正在加载 Session 1 原始数据...")

        def _load_single(sid):
            X, y, meta = self.paradigm.get_data(dataset=self.dataset, subjects=[sid])
            first_sess = meta['session'].unique()[0]
            mask = meta['session'] == first_sess
            return X[mask], y[mask], np.full(len(y[mask]), sid)

        results = Parallel(n_jobs=Config.N_JOBS)(
            delayed(_load_single)(s) for s in tqdm(self.subjects, desc="读取中")
        )
        self.X_raw = np.concatenate([r[0] for r in results])
        self.y_all = np.concatenate([r[1] for r in results])
        self.sub_labels = np.concatenate([r[2] for r in results])

    def run_experiment(self):
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
        print(f"\n>>> 启动 FBCSP 跨被试 5 折交叉验证 (5核并行)...")

        fold_results = Parallel(n_jobs=Config.N_JOBS)(
            delayed(self._process_fold)(tr_idx, te_idx)
            for tr_idx, te_idx in kf.split(self.subjects)
        )
        self._format_results(fold_results)

    def _process_fold(self, train_idx, test_idx):
        train_subs = [self.subjects[i] for i in train_idx]
        test_subs = [self.subjects[i] for i in test_idx]

        tr_mask = np.isin(self.sub_labels, train_subs)
        te_mask = np.isin(self.sub_labels, test_subs)

        # 1. 准备训练数据 (展平窗口)
        win_samples = int(Config.T_WIN_LEN * Config.FS)
        X_tr_raw = Utils.get_windows(self.X_raw[tr_mask]).reshape(-1, len(Config.CHANNELS), win_samples)
        y_tr = np.repeat(self.y_all[tr_mask], Config.N_WINDOWS)

        # 2. 多频带 CSP 特征提取 (共享特征核心)
        train_feats_list = []
        csp_bank = []
        for low, high in Config.BANDS:
            X_f = Utils.butter_bandpass(X_tr_raw, low, high)
            csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
            train_feats_list.append(csp.fit_transform(X_f, y_tr))
            csp_bank.append(csp)

        # 特征拼接与筛选 (MI 互信息)
        X_tr_concat = np.concatenate(train_feats_list, axis=1)
        selector = SelectKBest(mutual_info_classif, k=Config.K_FEATURES)
        X_tr_final = selector.fit_transform(X_tr_concat, y_tr)

        # 3. 分类器训练 (共享最终特征)
        lda = LinearDiscriminantAnalysis().fit(X_tr_final, y_tr)
        svm = SVC(kernel='linear', C=1.0).fit(X_tr_final, y_tr)

        # 4. 测试集评估
        X_te_struct = Utils.get_windows(self.X_raw[te_mask])
        y_te = self.y_all[te_mask]
        te_sub_ids = self.sub_labels[te_mask]

        fold_data = []
        for sid in test_subs:
            sub_mask = (te_sub_ids == sid)
            X_sub, y_sub = X_te_struct[sub_mask], y_te[sub_mask]
            counts = {'LDA': {'cv': 0, 'cw': 0}, 'SVM': {'cv': 0, 'cw': 0}}

            for i in range(len(y_sub)):
                # 对测试 Trial 进行多频带处理
                test_win_feats = []
                for b_idx, (low, high) in enumerate(Config.BANDS):
                    X_f_te = Utils.butter_bandpass(X_sub[i], low, high)
                    test_win_feats.append(csp_bank[b_idx].transform(X_f_te))

                # 拼接并应用训练好的筛选器
                X_te_final = selector.transform(np.concatenate(test_win_feats, axis=1))

                for name, clf in [('LDA', lda), ('SVM', svm)]:
                    preds = clf.predict(X_te_final)
                    counts[name]['cw'] += np.sum(preds == y_sub[i])
                    if Counter(preds).most_common(1)[0][0] == y_sub[i]:
                        counts[name]['cv'] += 1

            n_t = len(y_sub)
            fold_data.append({
                'Subject': f"S{sid:02d}",
                'LDA_vote': counts['LDA']['cv'] / n_t, 'LDA_win': counts['LDA']['cw'] / (n_t * 5),
                'SVM_vote': counts['SVM']['cv'] / n_t, 'SVM_win': counts['SVM']['cw'] / (n_t * 5)
            })
        return fold_data

    def _format_results(self, fold_results):
        all_res = [item for fold in fold_results for item in fold]
        self.df_final = pd.DataFrame(all_res).sort_values('Subject')
        print("\n" + "=" * 60)
        print(f"FBCSP 跨被试汇总 (LDA Vote Avg: {self.df_final['LDA_vote'].mean():.3f})")
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
        sns.boxplot(x='Classifier', y='Accuracy', data=plot_df, palette="Pastel2", width=0.4)
        sns.stripplot(x='Classifier', y='Accuracy', data=plot_df, color="blue", size=8, jitter=True, alpha=0.5)

        plt.title("FBCSP Cross-Subject: LDA vs SVM (Session 1)", fontsize=14)
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.ylim(0.4, 1.0)

        plt.savefig(f"{Config.OUTPUT_DIR}/FBCSP_跨被试对比.png", dpi=300)
        self.df_final.to_csv(f"{Config.OUTPUT_DIR}/FBCSP_CrossSubject_Results.csv", index=False)
        print(f">>> 结果已保存至 '{Config.OUTPUT_DIR}'")
        plt.show()


if __name__ == "__main__":
    exp = FBCSPCrossSubject()
    exp.load_session1_data()
    exp.run_experiment()
    exp.visualize()