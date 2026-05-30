import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from affiliation.metrics import pr_from_events
from affiliation.generics import convert_vector_to_events
from sklearn.metrics import f1_score, precision_score, recall_score

def calc_point2point(predict, actual):
    """ 计算普通的 Precision, Recall, F1 """
    precision = precision_score(actual, predict)
    recall = recall_score(actual, predict)
    f1 = f1_score(actual, predict)
    return precision, recall, f1


def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """
    点调整 (Point-Adjust) 逻辑：
    如果在连续的异常段中，只要有一个点被正确检测到，就认为整个异常段都被成功检测。
    """
    if label is None:
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(predict)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
        elif not actual[i]:
            anomaly_state = False

        if anomaly_state:
            predict[i] = True

    return predict


def affiliation_metrics(predict, actual):

    predict_list = np.array(predict, dtype=int).tolist()
    actual_list = np.array(actual, dtype=int).tolist()

    events_pred = convert_vector_to_events(predict_list)
    events_label = convert_vector_to_events(actual_list)

    Trange = (0, len(actual_list))

    results = pr_from_events(events_pred, events_label, Trange)
    precision = results['precision']
    recall = results['recall']

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def get_anomaly_metrics(score, label, target_anomaly_ratio=50.0):
    if len(label.shape) > 1:
        label = (label.sum(axis=1) > 0).astype(int)
    if len(score.shape) > 1:
        score = score.max(axis=1)

    score = score.flatten()
    label = label.flatten()

    auc_roc = roc_auc_score(label, score)
    auc_prc = average_precision_score(label, score)  # 【新增代码】计算 AUC-PRC (Average Precision)

    percentile_val = 100.0 - target_anomaly_ratio
    best_thresh = np.percentile(score, percentile_val)

    pred = (score >= best_thresh).astype(int)
    actual_ratio = np.sum(pred) / len(pred)

    try:
        aff_p, aff_r, aff_f1 = affiliation_metrics(pred, label)
    except Exception as e:
        print(f"Affiliation 计算出错 ({e})")
        aff_p, aff_r, aff_f1 = 0.0, 0.0, 0.0

    p, r, f1 = calc_point2point(pred, label)

    pa_pred = adjust_predicts(score, label, pred=pred.copy())
    pa_p, pa_r, pa_f1 = calc_point2point(pa_pred, label)

    best_raw_metrics = (p, r, f1)
    best_pa_metrics = (pa_p, pa_r, pa_f1)
    best_aff_metrics = (aff_p, aff_r, aff_f1)

    print("\n" + "="*50)
    print(f"目标比例: {target_anomaly_ratio}%")
    print(f"实际切出的异常比例: {actual_ratio*100:.4f}% (阈值: {best_thresh:.6f})")
    print(f"Affiliation F1: {aff_f1:.4f}")
    print("="*50 + "\n")

    return auc_roc, auc_prc, best_thresh, best_raw_metrics, best_pa_metrics, best_aff_metrics