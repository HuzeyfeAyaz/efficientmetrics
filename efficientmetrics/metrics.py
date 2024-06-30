import numpy as np


class EfficientMetrics:
    def __init__(self, y_true, y_preds, classes=None, class_labels=None):
        self.classes = classes
        if classes is None:
            self.classes = np.unique(y_true).tolist()
        self.y_true = self._convert_dtype(y_true)
        self.y_preds = self._convert_dtype(y_preds)
        self.total_sample = len(y_true)
        self.class_labels = class_labels
        self.class_confmats = {c: {} for c in self.classes}
        self.report = ""
        self.confmat = []

    def _convert_dtype(self, array):
        max_class = max(self.classes)
        if max_class <= np.iinfo(np.int8).max:
            return array.astype(np.int8)
        elif max_class <= np.iinfo(np.uint8).max:
            return array.astype(np.uint8)
        elif max_class <= np.iinfo(np.int16).max:
            return array.astype(np.int16)
        else:
            return array.astype(np.int32)

    def calculate_confusion_matrix(self):
        for c in self.classes:
            c_class = self.y_true == c      # current_class
            c_num = int(c_class.sum())

            tp = int((self.y_true[c_class] == self.y_preds[c_class]).sum())
            fp = int((self.y_preds[~c_class] == c).sum())

            self.class_confmats[c] = {'tp': tp, 'fp': fp, 'c_num': c_num}

            confmat_row = []
            for c_oth in self.classes:
                if c_oth == c:
                    confmat_row.append(tp)
                else:
                    confmat_row.append(
                        int((self.y_preds[c_class] == c_oth).sum()))

            self.confmat.append(confmat_row)

    def precision(self, tp, fp):
        try:
            return round(tp / (tp + fp), 2)
        except ZeroDivisionError:
            return 0

    def recall(self, tp, c_num):
        return round(tp / c_num, 2)

    def f1_score(self, tp, fp, c_num):
        precision = self.precision(tp, fp)
        recall = self.recall(tp, c_num)

        try:
            return round((2*precision*recall)/(precision+recall), 2)
        except ZeroDivisionError:
            return 0

    def calc_accuracy(self, t_vals):
        return t_vals / self.total_sample

    def update_report(self, vals):
        self.report += "{: >12} {: >10} {: >10} {: >10} {: >15}\n".format(
            *vals)

    def classification_report(self):
        self.update_report(["", "precision", "recall", "f1-score", "support"])
        self.report += '\n'

        t_vals, precisions, recalls, f1_scores, supports = 0, [], [], [], []
        for k, v in self.class_confmats.items():
            prec = self.precision(v['tp'], v['fp'])
            rec = self.recall(v['tp'], v['c_num'])
            f1 = self.f1_score(v['tp'], v['fp'], v['c_num'])

            k = k if self.class_labels is None else self.class_labels[k]
            self.update_report([k, prec, rec, f1, v['c_num']])

            t_vals += v['tp']
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)
            supports.append(v['c_num'])

        self.report += '\n'
        self.accuracy = self.calc_accuracy(t_vals)
        self.update_report([
            "accuracy", "", "", round(self.accuracy, 2), self.total_sample
        ])
        self.f1_macro = np.mean(f1_scores)
        self.update_report([
            "macro avg", round(np.mean(precisions), 2),
            round(np.mean(recalls), 2),
            round(self.f1_macro, 2), self.total_sample
        ])

        arr_supps = np.array(supports)
        weighted_prec = np.sum(
            np.array(precisions) * arr_supps) / self.total_sample
        weighted_rec = np.sum(
            np.array(recalls) * arr_supps) / self.total_sample
        self.weighted_f1 = np.sum(
            np.array(f1_scores) * arr_supps) / self.total_sample
        self.update_report([
            "weighted avg", round(weighted_prec, 2), round(weighted_rec, 2),
            round(self.weighted_f1, 2), self.total_sample
        ])
