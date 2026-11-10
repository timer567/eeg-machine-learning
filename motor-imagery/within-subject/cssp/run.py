import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
import os


# ==========================================
# 1. 配置模块
# ==========================================
class Config:
    CHANNELS = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
                'CPz', 'CP2', 'CP4', 'CP6']
    FS = 100
    FMIN, FMAX = 8, 30
    FILTER_ORDER = 5
    T_START, T_WIN_LEN, T_STEP = 1.0, 1.5, 0.25
    N_WINDOWS = 5
    N_FOLDS = 5
    RANDOM_STATE = 42
    N_JOBS = 4
    TAU_LIST = [1, 2, 3]  # CSSP 时延参数 (10ms, 20ms, 30ms)


# ==========================================
# 2. 核心算法：CSSP 拼接逻辑
# ==========================================
def apply_cssp_transform(data, tau):
    """ 实现 (N, C, T) -> (N, 2C, T-tau) 的拼接 """
    if tau == 0: return data
    x_orig = data[:, :, tau:]
    x_delayed = data[:, :, :-tau]
    return np.concatenate([x_orig, x_delayed], axis=1)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


# ==========================================
# 3. 单被试 CSSP 处理逻辑 (双分类器共享特征)
# ==========================================
def process_single_subject(sub_id):
    try:
        dataset = Lee2019_MI()
        paradigm = MotorImagery(channels=Config.CHANNELS, resample=Config.FS, fmin=Config.FMIN, fmax=Config.FMAX)
        X_all, y_all, meta = paradigm.get_data(dataset=dataset, subjects=[sub_id])
        X_all = butter_bandpass_filter(X_all, Config.FMIN, Config.FMAX, Config.FS, Config.FILTER_ORDER)

        unique_sessions = np.sort(meta['session'].unique())
        # 初始化存储：特征提取结果将供 LDA 和 SVM 共同使用
        sub_res = {'LDA': {'vote': [], 'win': []}, 'SVM': {'vote': [], 'win': []}}

        for sess_name in unique_sessions:
            mask = meta['session'] == sess_name
            X_sess, y_sess = X_all[mask], y_all[mask]

            start_idx = int(Config.T_START * Config.FS)
            win_samples = int(Config.T_WIN_LEN * Config.FS)
            step_samples = int(Config.T_STEP * Config.FS)

            # 滑窗
            trial_windows = []
            for trial in X_sess:
                windows = [trial[:, start_idx + w * step_samples: start_idx + w * step_samples + win_samples]
                           for w in range(Config.N_WINDOWS)]
                trial_windows.append(np.array(windows))
            X_wins = np.array(trial_windows)

            kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)

            # --- CSSP 寻优：确定该 Session 的最优 tau ---
            best_tau = 1
            max_val_acc = 0
            for t_val in Config.TAU_LIST:
                temp_scores = []
                for tr_idx, te_idx in list(kf.split(X_wins))[:2]:  # 快速评估
                    X_tr = apply_cssp_transform(X_wins[tr_idx].reshape(-1, len(Config.CHANNELS), win_samples), t_val)
                    y_tr = np.repeat(y_sess[tr_idx], Config.N_WINDOWS)
                    csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
                    lda = LinearDiscriminantAnalysis()
                    lda.fit(csp.fit_transform(X_tr, y_tr), y_tr)
                    X_te = apply_cssp_transform(X_wins[te_idx].reshape(-1, len(Config.CHANNELS), win_samples), t_val)
                    temp_scores.append(lda.score(csp.transform(X_te), np.repeat(y_sess[te_idx], Config.N_WINDOWS)))
                if np.mean(temp_scores) > max_val_acc:
                    max_val_acc = np.mean(temp_scores);
                    best_tau = t_val

            # --- 正式 K-Fold (特征共享给两个分类器) ---
            fold_scores = {'LDA': {'v': [], 'w': []}, 'SVM': {'v': [], 'w': []}}
            for train_idx, test_idx in kf.split(X_wins):
                X_train = apply_cssp_transform(X_wins[train_idx].reshape(-1, len(Config.CHANNELS), win_samples),
                                               best_tau)
                y_train = np.repeat(y_sess[train_idx], Config.N_WINDOWS)

                csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
                X_train_feat = csp.fit_transform(X_train, y_train)

                # 训练两个模型
                lda = LinearDiscriminantAnalysis().fit(X_train_feat, y_train)
                svm = SVC(kernel='linear', C=1.0).fit(X_train_feat, y_train)

                counts = {'LDA': {'cv': 0, 'cw': 0}, 'SVM': {'cv': 0, 'cw': 0}}
                for i in test_idx:
                    test_trial_cssp = apply_cssp_transform(X_wins[i], best_tau)
                    test_feat = csp.transform(test_trial_cssp)

                    for name, clf in zip(['LDA', 'SVM'], [lda, svm]):
                        preds = clf.predict(test_feat)
                        counts[name]['cw'] += np.sum(preds == y_sess[i])
                        if Counter(preds).most_common(1)[0][0] == y_sess[i]: counts[name]['cv'] += 1

                for name in ['LDA', 'SVM']:
                    fold_scores[name]['v'].append(counts[name]['cv'] / len(test_idx))
                    fold_scores[name]['w'].append(counts[name]['cw'] / (len(test_idx) * Config.N_WINDOWS))

            for name in ['LDA', 'SVM']:
                sub_res[name]['vote'].append(np.mean(fold_scores[name]['v']))
                sub_res[name]['win'].append(np.mean(fold_scores[name]['w']))
        return sub_res
    except Exception as e:
        print(f"\n[Error] S{sub_id}: {e}")
        return None


# ==========================================
# 4. 数据导出与自动绘图保存
# ==========================================
def export_and_plot(subject_ids, all_data):
    output_dir = "结果"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    summary_final = {}

    for clf_name in ['LDA', 'SVM']:
        storage = {'vote': {'s1': [], 's2': [], 'avg': []}, 'win': {'s1': [], 's2': [], 'avg': []}}
        for res in all_data:
            for k in ['vote', 'win']:
                s1, s2 = res[clf_name][k][0], res[clf_name][k][1]
                storage[k]['s1'].append(s1);
                storage[k]['s2'].append(s2)
                storage[k]['avg'].append((s1 + s2) / 2)

        df = pd.DataFrame({
            'Subject': [f"S{s:02d}" for s in subject_ids],
            'Method1_S1': storage['vote']['s1'], 'Method1_S2': storage['vote']['s2'],
            'Method1_Avg': storage['vote']['avg'],
            'Method2_S1': storage['win']['s1'], 'Method2_S2': storage['win']['s2'], 'Method2_Avg': storage['win']['avg']
        })
        print(f"\n" + "=" * 25 + f" CSSP + {clf_name} 结果 " + "=" * 25)
        pd.options.display.float_format = '{:.3f}'.format
        print(df.to_string(index=False))
        df.to_csv(f"{output_dir}/CSSP_Results_{clf_name}.csv", index=False)

        # 绘图逻辑
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for i, key in enumerate(['vote', 'win']):
            axes[i].plot(subject_ids, storage[key]['s1'], 'o--', alpha=0.5, label='S1')
            axes[i].plot(subject_ids, storage[key]['s2'], 's--', alpha=0.5, label='S2')
            axes[i].plot(subject_ids, storage[key]['avg'], '^-', color='red', lw=2, label='Avg')
            axes[i].set_title(f"{clf_name} - M{i + 1}");
            axes[i].set_ylim(0.4, 1.0);
            axes[i].legend()
        plt.suptitle(f"MI CSSP Performance - Classifier: {clf_name}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = f"{output_dir}/{clf_name}分类结果.png"
        plt.savefig(save_path, dpi=300)
        print(f">>> 图片保存至: {save_path}")
        plt.show()
        summary_final[clf_name] = np.mean(storage['vote']['avg'])

    print("\n" + "=" * 40)
    print("CSSP 最终平均性能对比 (Trial Voting Avg):")
    for name, score in summary_final.items():
        print(f"{name}: {score:.4f}")
    print("=" * 40)


# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    subjects = list(range(1, 11))
    print(f">>> 开始并行 CSSP 计算 (Tau 寻优 + 双分类器对比)...")
    all_results = Parallel(n_jobs=Config.N_JOBS)(delayed(process_single_subject)(s) for s in tqdm(subjects))
    valid_results = [r for r in all_results if r is not None]
    valid_sub_ids = [subjects[i] for i, r in enumerate(all_results) if r is not None]
    if valid_results: export_and_plot(valid_sub_ids, valid_results)