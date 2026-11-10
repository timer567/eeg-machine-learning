import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
import os


# ==========================================
# 1. 配置模块
# ==========================================
class Config:
    CHANNELS = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1',
                'CPz', 'CP2', 'CP4', 'CP6']
    FS = 100
    BANDS = [(8, 12), (12, 18), (18, 24), (24, 30)]  # FBCSP 子频段
    FILTER_ORDER = 5
    T_START, T_WIN_LEN, T_STEP = 1.0, 1.5, 0.25
    N_WINDOWS = 5
    N_FOLDS = 5
    RANDOM_STATE = 42
    N_JOBS = 4  # 针对 i5-13500H 优化
    K_FEATURES = 6  # MI 筛选保留特征数


# ==========================================
# 2. 信号处理核心
# ==========================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


# ==========================================
# 3. 单被试 FBCSP 处理逻辑 (双分类器共享特征)
# ==========================================
def process_single_subject(sub_id):
    try:
        dataset = Lee2019_MI()
        paradigm = MotorImagery(channels=Config.CHANNELS, resample=Config.FS, fmin=7, fmax=31)
        X_all, y_all, meta = paradigm.get_data(dataset=dataset, subjects=[sub_id])

        unique_sessions = np.sort(meta['session'].unique())
        # 初始化存储结构，特征跑一次，喂给两个分类器
        sub_res = {'LDA': {'vote': [], 'win': []}, 'SVM': {'vote': [], 'win': []}}

        for sess_name in unique_sessions:
            mask = meta['session'] == sess_name
            X_sess, y_sess = X_all[mask], y_all[mask]

            start_idx = int(Config.T_START * Config.FS)
            win_samples = int(Config.T_WIN_LEN * Config.FS)
            step_samples = int(Config.T_STEP * Config.FS)

            # --- 滑窗增强 (5个窗口) ---
            trial_windows = []
            for trial in X_sess:
                windows = [trial[:, start_idx + w * step_samples: start_idx + w * step_samples + win_samples]
                           for w in range(Config.N_WINDOWS)]
                trial_windows.append(np.array(windows))
            X_wins = np.array(trial_windows)

            kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
            fold_scores = {'LDA': {'v': [], 'w': []}, 'SVM': {'v': [], 'w': []}}

            for train_idx, test_idx in kf.split(X_wins):
                # --- [特征提取核心：仅跑一次] ---
                X_train_raw = X_wins[train_idx].reshape(-1, len(Config.CHANNELS), win_samples)
                y_train = np.repeat(y_sess[train_idx], Config.N_WINDOWS)

                train_band_feats = []
                csp_models = []
                for low, high in Config.BANDS:
                    X_f = butter_bandpass_filter(X_train_raw, low, high, Config.FS, Config.FILTER_ORDER)
                    csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
                    train_band_feats.append(csp.fit_transform(X_f, y_train))
                    csp_models.append(csp)

                X_train_comb = np.concatenate(train_band_feats, axis=1)
                selector = SelectKBest(mutual_info_classif, k=Config.K_FEATURES)
                X_train_final = selector.fit_transform(X_train_comb, y_train)

                # --- [分类器对比训练：共享特征] ---
                lda = LinearDiscriminantAnalysis()
                lda.fit(X_train_final, y_train)
                svm = SVC(kernel='linear', C=1.0)  # 使用线性核保持公平性
                svm.fit(X_train_final, y_train)

                # --- [评估过程] ---
                counts = {'LDA': {'cv': 0, 'cw': 0}, 'SVM': {'cv': 0, 'cw': 0}}
                for i in test_idx:
                    test_trial_raw = X_wins[i]
                    test_band_feats = []
                    for idx, (low, high) in enumerate(Config.BANDS):
                        X_f_te = butter_bandpass_filter(test_trial_raw, low, high, Config.FS, Config.FILTER_ORDER)
                        test_band_feats.append(csp_models[idx].transform(X_f_te))

                    X_test_final = selector.transform(np.concatenate(test_band_feats, axis=1))

                    for name, clf in zip(['LDA', 'SVM'], [lda, svm]):
                        preds = clf.predict(X_test_final)
                        counts[name]['cw'] += np.sum(preds == y_sess[i])  # 窗口胜率
                        if Counter(preds).most_common(1)[0][0] == y_sess[i]:  # Trial 投票
                            counts[name]['cv'] += 1

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
# 4. 输出、自动保存与汇总模块
# ==========================================
def export_and_plot(subject_ids, all_data):
    output_dir = "结果"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    overall_summary = {}

    for clf_name in ['LDA', 'SVM']:
        storage = {'vote': {'s1': [], 's2': [], 'avg': []}, 'win': {'s1': [], 's2': [], 'avg': []}}
        for res in all_data:
            for m_key in ['vote', 'win']:
                s1, s2 = res[clf_name][m_key][0], res[clf_name][m_key][1]
                storage[m_key]['s1'].append(s1)
                storage[m_key]['s2'].append(s2)
                storage[m_key]['avg'].append((s1 + s2) / 2)

        # 整理 DataFrame
        df = pd.DataFrame({
            'Subject': [f"S{s:02d}" for s in subject_ids],
            'Method1_S1': storage['vote']['s1'], 'Method1_S2': storage['vote']['s2'],
            'Method1_Avg': storage['vote']['avg'],
            'Method2_S1': storage['win']['s1'], 'Method2_S2': storage['win']['s2'], 'Method2_Avg': storage['win']['avg']
        })

        print(f"\n" + "=" * 25 + f" FBCSP + {clf_name} 结果 " + "=" * 25)
        pd.options.display.float_format = '{:.3f}'.format
        print(df.to_string(index=False))
        df.to_csv(f"{output_dir}/FBCSP_Results_{clf_name}.csv", index=False)

        # 绘图并保存
        plot_and_save_fig(subject_ids, storage, clf_name, output_dir)

        # 记录汇总数据
        overall_summary[clf_name] = np.mean(storage['vote']['avg'])

    # 输出最后总的结果对比
    print("\n" + "=" * 35)
    print("分类器最终平均性能对比 (Method 1 Avg)")
    print("-" * 35)
    for name, score in overall_summary.items():
        print(f"{name} Classifier: {score:.4f}")
    print("=" * 35)


def plot_and_save_fig(subject_ids, storage, clf_name, output_dir):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    titles = [f"M1: Voting ({clf_name})", f"M2: Ratio ({clf_name})"]

    for i, key in enumerate(['vote', 'win']):
        axes[i].plot(subject_ids, storage[key]['s1'], 'o--', alpha=0.5, label='Session 1')
        axes[i].plot(subject_ids, storage[key]['s2'], 's--', alpha=0.5, label='Session 2')
        axes[i].plot(subject_ids, storage[key]['avg'], '^-', color='darkred', lw=2, label='Average')
        axes[i].set_title(titles[i], fontsize=13, fontweight='bold')
        axes[i].set_xticks(subject_ids);
        axes[i].set_ylim(0.4, 1.0)
        axes[i].axhline(y=0.5, color='gray', linestyle=':');
        axes[i].legend()

    plt.suptitle(f"FBCSP Performance Analysis - {clf_name}", fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 自动保存图片：命名格式为 “XXX分类结果”
    save_path = os.path.join(output_dir, f"{clf_name}分类结果.png")
    plt.savefig(save_path, dpi=300)
    print(f">>> 图片已保存至: {save_path}")
    plt.show()


# ==========================================
# 5. 执行
# ==========================================
if __name__ == "__main__":
    subjects = list(range(1, 11))
    print(f">>> 启动 FBCSP 对比实验 (并行受试者: {Config.N_JOBS})...")

    all_results = Parallel(n_jobs=Config.N_JOBS)(
        delayed(process_single_subject)(s) for s in tqdm(subjects)
    )

    valid_results = [r for r in all_results if r is not None]
    valid_sub_ids = [subjects[i] for i, r in enumerate(all_results) if r is not None]

    if valid_results:
        export_and_plot(valid_sub_ids, valid_results)
    else:
        print("实验运行失败，未获取到有效结果。")