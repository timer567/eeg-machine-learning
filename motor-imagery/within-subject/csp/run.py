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
import moabb
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
    N_JOBS = 4  # 针对 i5-13500H 优化


# ==========================================
# 2. 信号处理核心逻辑
# ==========================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


def process_single_subject(sub_id, clf_type='LDA'):
    """
    基础逻辑完全不变，增加 clf_type 参数切换分类器
    """
    try:
        dataset = Lee2019_MI()
        paradigm = MotorImagery(channels=Config.CHANNELS, resample=Config.FS, fmin=Config.FMIN, fmax=Config.FMAX)
        X_all, y_all, meta = paradigm.get_data(dataset=dataset, subjects=[sub_id])
        X_all = butter_bandpass_filter(X_all, Config.FMIN, Config.FMAX, Config.FS, Config.FILTER_ORDER)

        unique_sessions = np.sort(meta['session'].unique())
        sub_res = {'vote': [], 'win': []}

        for sess_name in unique_sessions:
            mask = meta['session'] == sess_name
            X_sess, y_sess = X_all[mask], y_all[mask]

            start_idx = int(Config.T_START * Config.FS)
            win_samples = int(Config.T_WIN_LEN * Config.FS)
            step_samples = int(Config.T_STEP * Config.FS)

            trial_windows = []
            for trial in X_sess:
                windows = [trial[:, start_idx + w * step_samples: start_idx + w * step_samples + win_samples] for w in
                           range(Config.N_WINDOWS)]
                trial_windows.append(np.array(windows))
            X_wins = np.array(trial_windows)

            kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)
            f_v, f_w = [], []

            for train_idx, test_idx in kf.split(X_wins):
                X_train = X_wins[train_idx].reshape(-1, len(Config.CHANNELS), win_samples)
                y_train = np.repeat(y_sess[train_idx], Config.N_WINDOWS)

                csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

                # 分类器切换逻辑
                if clf_type == 'LDA':
                    clf = LinearDiscriminantAnalysis()
                else:
                    clf = SVC(kernel='linear', C=1.0)  # 标准线性 SVM

                clf.fit(csp.fit_transform(X_train, y_train), y_train)

                cv, cw = 0, 0
                for i in test_idx:
                    preds = clf.predict(csp.transform(X_wins[i]))
                    cw += np.sum(preds == y_sess[i])
                    if Counter(preds).most_common(1)[0][0] == y_sess[i]: cv += 1
                f_v.append(cv / len(test_idx))
                f_w.append(cw / (len(test_idx) * Config.N_WINDOWS))

            sub_res['vote'].append(np.mean(f_v))
            sub_res['win'].append(np.mean(f_w))
        return sub_res
    except Exception as e:
        print(f"\n[Error] S{sub_id} ({clf_type}): {e}")
        return None


# ==========================================
# 3. 数据导出与数值打印模块
# ==========================================
def export_results(subject_ids, storage, clf_type):
    data_list = []
    for i, sub_id in enumerate(subject_ids):
        res = {
            'Subject': f"S{sub_id:02d}",
            'Method1_S1': storage['vote']['s1'][i],
            'Method1_S2': storage['vote']['s2'][i],
            'Method1_Avg': storage['vote']['avg'][i],
            'Method2_S1': storage['win']['s1'][i],
            'Method2_S2': storage['win']['s2'][i],
            'Method2_Avg': storage['win']['avg'][i]
        }
        data_list.append(res)

    df = pd.DataFrame(data_list)

    print("\n" + "=" * 85)
    print(f"分类器: {clf_type} - 实验结果详细数值记录")
    print("-" * 85)
    pd.options.display.float_format = '{:.3f}'.format
    print(df.to_string(index=False))
    print("=" * 85)

    if not os.path.exists("结果"): os.makedirs("结果")
    filename = f"结果/MI_Results_S1_S10_{clf_type}.csv"
    df.to_csv(filename, index=False, float_format='%.3f')
    print(f">>> {clf_type} 结果已保存至: {filename}")


# ==========================================
# 4. 可视化模块
# ==========================================
def plot_summary_lines(subject_ids, storage, clf_type):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    titles = [f"Method 1 (Voting) - {clf_type}", f"Method 2 (Ratio) - {clf_type}"]
    metrics_keys = ['vote', 'win']

    for i, key in enumerate(metrics_keys):
        axes[i].plot(subject_ids, storage[key]['s1'], 'o--', label='Session 1', alpha=0.6)
        axes[i].plot(subject_ids, storage[key]['s2'], 's--', label='Session 2', alpha=0.6)
        axes[i].plot(subject_ids, storage[key]['avg'], '^-', label='Average', lw=3, markersize=8)

        axes[i].set_title(titles[i], fontsize=14, fontweight='bold')
        axes[i].set_xlabel("Subject ID")
        axes[i].set_ylabel("Accuracy")
        axes[i].set_xticks(subject_ids)
        axes[i].set_ylim(0.4, 1.0)
        axes[i].axhline(y=0.5, color='red', linestyle=':', label='Chance Level')
        axes[i].legend(loc='lower right')

    plt.suptitle(f"MI Performance Analysis - Classifier: {clf_type}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    subjects = list(range(1, 11))
    classifiers = ['LDA', 'SVM']

    # 存储所有分类器的最终平均分，用于简单对比打印
    clf_comparison = {}

    for clf in classifiers:
        print(f"\n\n>>> 启动分类器对比实验: {clf} (n_jobs={Config.N_JOBS})...")

        results = Parallel(n_jobs=Config.N_JOBS, backend="loky")(
            delayed(process_single_subject)(s, clf_type=clf) for s in tqdm(subjects)
        )

        valid_results = [r for r in results if r is not None]
        valid_sub_ids = [subjects[i] for i, r in enumerate(results) if r is not None]

        storage = {
            'vote': {'s1': [], 's2': [], 'avg': []},
            'win': {'s1': [], 's2': [], 'avg': []}
        }

        for res in valid_results:
            for k in ['vote', 'win']:
                s1, s2 = res[k][0], res[k][1]
                storage[k]['s1'].append(s1)
                storage[k]['s2'].append(s2)
                storage[k]['avg'].append((s1 + s2) / 2)

        if valid_results:
            export_results(valid_sub_ids, storage, clf)
            plot_summary_lines(valid_sub_ids, storage, clf)
            # 记录平均准确率用于最后对比
            clf_comparison[clf] = np.mean(storage['vote']['avg'])
        else:
            print(f"{clf} 实验未获取到有效结果。")

    print("\n" + "=" * 40)
    print("分类器最终平均性能对比 (Method 1 Avg):")
    for name, score in clf_comparison.items():
        print(f"{name}: {score:.4f}")
    print("=" * 40)