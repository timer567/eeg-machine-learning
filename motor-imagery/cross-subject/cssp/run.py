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
    TAU_LIST = [1, 2, 3]  # 10ms, 20ms, 30ms
    OUTPUT_DIR = "结果_CSSP"


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
    def apply_cssp_transform(data, tau):
        """ 实现 (N, C, T) -> (N, 2C, T-tau) 的空间-光谱拼接 """
        if tau == 0: return data
        # data 形状可能是 (N, C, T) 或 (C, T)
        x_orig = data[..., tau:]
        x_delayed = data[..., :-tau]
        return np.concatenate([x_orig, x_delayed], axis=-2)

    @staticmethod
    def get_windows(data):
        start = int(Config.T_START * Config.FS)
        step = int(Config.T_STEP * Config.FS)
        win_size = int(Config.T_WIN_LEN * Config.FS)
        return np.stack([data[:, :, start + w * step: start + w * step + win_size]
                         for w in range(Config.N_WINDOWS)], axis=1)


# ==========================================
# 3. CSSP 核心计算类
# ==========================================
class CSSPCrossSubject:
    def __init__(self):
        self.dataset = Lee2019_MI()
        self.paradigm = MotorImagery(channels=Config.CHANNELS, resample=Config.FS,
                                     fmin=Config.FMIN, fmax=Config.FMAX)
        self.subjects = [int(s) for s in range(1, 11)]

    def load_data(self):
        """ 阶段 1：并行加载 Session 1 并滤波 """
        print(f">>> 正在加载 Session 1 数据并执行 {Config.FILTER_ORDER} 阶滤波...")

        def _load_single(sid):
            X, y, meta = self.paradigm.get_data(dataset=self.dataset, subjects=[sid])
            first_sess = meta['session'].unique()[0]
            mask = meta['session'] == first_sess
            X_filt = Utils.butter_filter(X[mask])
            return X_filt, y[mask], np.full(len(y[mask]), sid)

        results = Parallel(n_jobs=Config.N_JOBS)(
            delayed(_load_single)(s) for s in tqdm(self.subjects, desc="读取中")
        )
        self.X_all = np.concatenate([r[0] for r in results])
        self.y_all = np.concatenate([r[1] for r in results])
        self.sub_labels = np.concatenate([r[2] for r in results])

    def run_experiment(self):
        """ 阶段 2：跨被试 5 折交叉验证 """
        kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
        print(f"\n>>> 启动 CSSP 跨被试实验 (Tau 内部寻优 + 5核并行)...")

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

        # 准备训练数据
        X_tr_raw = Utils.get_windows(self.X_all[tr_mask])
        y_tr_raw = self.y_all[tr_mask]

        # --- 内部寻优：在训练集内确定最佳 Tau ---
        best_tau = self._find_best_tau(X_tr_raw, y_tr_raw)

        # --- 正式训练 (使用最佳 Tau) ---
        win_samples = int(Config.T_WIN_LEN * Config.FS)
        X_tr_cssp = Utils.apply_cssp_transform(X_tr_raw.reshape(-1, len(Config.CHANNELS), win_samples), best_tau)
        y_tr_long = np.repeat(y_tr_raw, Config.N_WINDOWS)

        csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
        feat_tr = csp.fit_transform(X_tr_cssp, y_tr_long)

        # 共享特征训练 LDA & SVM
        lda = LinearDiscriminantAnalysis().fit(feat_tr, y_tr_long)
        svm = SVC(kernel='linear', C=1.0).fit(feat_tr, y_tr_long)

        # --- 测试集评估 ---
        X_te_raw = Utils.get_windows(self.X_all[te_mask])
        y_te_raw = self.y_all[te_mask]
        te_sub_labels = self.sub_labels[te_mask]

        fold_data = []
        for sid in test_subs:
            s_mask = (te_sub_labels == sid)
            X_sub, y_sub = X_te_raw[s_mask], y_te_raw[s_mask]
            counts = {'LDA': {'cv': 0, 'cw': 0}, 'SVM': {'cv': 0, 'cw': 0}}

            for i in range(len(y_sub)):
                # 应用 Tau 和 CSP
                X_trial_cssp = Utils.apply_cssp_transform(X_sub[i], best_tau)
                feat_te = csp.transform(X_trial_cssp)

                for name, clf in [('LDA', lda), ('SVM', svm)]:
                    preds = clf.predict(feat_te)
                    counts[name]['cw'] += np.sum(preds == y_sub[i])  # Method 2
                    if Counter(preds).most_common(1)[0][0] == y_sub[i]:  # Method 1
                        counts[name]['cv'] += 1

            n_t = len(y_sub)
            fold_data.append({
                'Subject': f"S{sid:02d}", 'Best_Tau': best_tau,
                'LDA_vote': counts['LDA']['cv'] / n_t, 'LDA_win': counts['LDA']['cw'] / (n_t * 5),
                'SVM_vote': counts['SVM']['cv'] / n_t, 'SVM_win': counts['SVM']['cw'] / (n_t * 5)
            })
        return fold_data

    def _find_best_tau(self, X_win, y_trial):
        """ 在训练集内部进行快速验证以选择 Tau """
        best_t, max_acc = Config.TAU_LIST[0], 0
        win_samples = int(Config.T_WIN_LEN * Config.FS)
        # 内部简单 3 折验证
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)

        for t in Config.TAU_LIST:
            acc_list = []
            for itr, ite in inner_kf.split(X_win):
                X_it = Utils.apply_cssp_transform(X_win[itr].reshape(-1, len(Config.CHANNELS), win_samples), t)
                y_it = np.repeat(y_trial[itr], Config.N_WINDOWS)
                csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
                lda = LinearDiscriminantAnalysis()
                lda.fit(csp.fit_transform(X_it, y_it), y_it)

                X_ie = Utils.apply_cssp_transform(X_win[ite].reshape(-1, len(Config.CHANNELS), win_samples), t)
                y_ie = np.repeat(y_trial[ite], Config.N_WINDOWS)
                acc_list.append(lda.score(csp.transform(X_ie), y_ie))

            avg_acc = np.mean(acc_list)
            if avg_acc > max_acc:
                max_acc, best_t = avg_acc, t
        return best_t

    def _format_results(self, fold_results):
        all_res = [item for fold in fold_results for item in fold]
        self.df_final = pd.DataFrame(all_res).sort_values('Subject')
        print("\n" + "=" * 70)
        print(f"CSSP 跨被试汇总 (Session 1, LDA Vote Avg: {self.df_final['LDA_vote'].mean():.3f})")
        print(self.df_final.to_string(index=False))
        print("=" * 70)

    def visualize(self):
        if not os.path.exists(Config.OUTPUT_DIR): os.makedirs(Config.OUTPUT_DIR)
        plot_df = pd.melt(self.df_final, id_vars=['Subject'],
                          value_vars=['LDA_vote', 'SVM_vote'],
                          var_name='Classifier', value_name='Accuracy')
        plot_df['Classifier'] = plot_df['Classifier'].str.replace('_vote', '')

        plt.figure(figsize=(9, 6))
        sns.set_theme(style="whitegrid")
        sns.boxplot(x='Classifier', y='Accuracy', data=plot_df, palette="Pastel1", width=0.4)
        sns.stripplot(x='Classifier', y='Accuracy', data=plot_df, color="darkgreen", size=8, jitter=True, alpha=0.6)

        plt.title("CSSP Cross-Subject Performance (Session 1)", fontsize=14)
        plt.axhline(0.5, color='gray', linestyle='--')
        plt.ylim(0.4, 1.0)

        plt.savefig(f"{Config.OUTPUT_DIR}/CSSP_跨被试对比.png", dpi=300)
        self.df_final.to_csv(f"{Config.OUTPUT_DIR}/CSSP_CrossSubject_Results.csv", index=False)
        print(f">>> 结果已保存至 '{Config.OUTPUT_DIR}'")
        plt.show()


if __name__ == "__main__":
    exp = CSSPCrossSubject()
    exp.load_data()
    exp.run_experiment()
    exp.visualize()