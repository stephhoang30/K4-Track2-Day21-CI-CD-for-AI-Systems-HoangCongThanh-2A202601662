import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Ty le lop duong tham chieu cua bo du lieu goc (Bonus 5)
REFERENCE_POSITIVE_RATE = 0.248
DRIFT_TOLERANCE = 0.05

CLASS_LABELS = {0: "thu_nhap_thap", 1: "thu_nhap_cao"}


def check_positive_rate(y) -> float:
    """
    Tinh ty le lop duong trong tap huan luyen va canh bao neu lech qua nhieu so
    voi ty le tham chieu 24.8% (Bonus 5).

    Ty le lop duong truot di lam F1 truot theo ma accuracy gan nhu khong doi,
    nen phai bat duoc no truoc khi huan luyen chu khong doi den luc xem metric.
    """
    rate = float(np.mean(y))
    delta = rate - REFERENCE_POSITIVE_RATE

    print(f"Ty le lop duong (>50K): {rate:.2%} (tham chieu {REFERENCE_POSITIVE_RATE:.1%})")
    if abs(delta) > DRIFT_TOLERANCE:
        print(
            f"CANH BAO LECH DU LIEU: ty le lop duong lech {delta:+.1%} so voi "
            f"tham chieu, vuot dung sai {DRIFT_TOLERANCE:.0%}."
        )
    return rate


def tune_threshold(y_true, proba) -> tuple:
    """
    Quet nguong quyet dinh tu 0.10 den 0.90 buoc 0.05, tra ve nguong cho F1 cao
    nhat (Bonus 2).

    model.predict() mac dinh cat o 0.5. Voi du lieu lech 75/25 thi 0.5 gan nhu
    khong bao gio la diem tot nhat: ha nguong xuong doi mot chut precision lay
    nhieu recall, va F1 thuong tang theo.
    """
    grid = np.arange(0.10, 0.91, 0.05)
    scores = [(t, f1_score(y_true, (proba >= t).astype(int), zero_division=0)) for t in grid]
    best_t, best_f1 = max(scores, key=lambda s: s[1])
    return round(float(best_t), 2), float(best_f1), scores


def write_detail(y_true, y_pred, params: dict, pos_rate: float, tuning: dict) -> str:
    """
    Ghi bao cao chi tiet dang van ban: confusion matrix va precision/recall cho
    tung lop (Bonus 3). File nay duoc upload lam artifact trong CI.
    """
    names = [CLASS_LABELS[c] for c in sorted(CLASS_LABELS)]
    cm = confusion_matrix(y_true, y_pred, labels=sorted(CLASS_LABELS))

    lines = ["BAO CAO CHI TIET MO HINH", "=" * 64, "", "Sieu tham so:"]
    lines += [f"  {k}: {v}" for k, v in params.items()]

    lines += [
        "",
        f"Ty le lop duong tap huan luyen: {pos_rate:.2%} "
        f"(tham chieu {REFERENCE_POSITIVE_RATE:.1%})",
        "",
        "Confusion matrix (hang = that, cot = du doan):",
        "",
    ]
    header = " " * 17 + "".join(f"{n:>17}" for n in names)
    lines.append(header)
    for name, row in zip(names, cm):
        lines.append(f"{name:>17}" + "".join(f"{v:>17}" for v in row))

    tn, fp, fn, tp = cm.ravel()
    lines += [
        "",
        f"  Bo sot nguoi thu nhap cao (false negative): {fn}",
        f"  Gan nham nguoi thu nhap thap (false positive): {fp}",
        "",
        "Precision / recall / f1 tung lop:",
        "",
        classification_report(
            y_true, y_pred, labels=sorted(CLASS_LABELS),
            target_names=names, digits=4, zero_division=0,
        ),
        "Quet nguong quyet dinh:",
        "",
        f"  nguong mac dinh 0.50 -> f1 {tuning['f1_at_default']:.4f}",
        f"  nguong tot nhat {tuning['best_threshold']:.2f} -> f1 {tuning['f1_at_best']:.4f}",
        f"  chenh lech: {tuning['f1_at_best'] - tuning['f1_at_default']:+.4f}",
        "",
    ]
    lines += [f"    t={t:.2f}  f1={s:.4f}" for t, s in tuning["grid"]]

    os.makedirs("outputs", exist_ok=True)
    detail = "\n".join(lines)
    with open("outputs/detail.txt", "w") as f:
        f.write(detail)
    return detail


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    pos_rate = check_positive_rate(y_train)

    with mlflow.start_run():

        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # f1_score cho LOP DUONG (target = 1), khong dung average
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds, zero_division=0))
        acc = float(accuracy_score(y_eval, preds))

        proba = model.predict_proba(X_eval)[:, 1]
        best_t, best_f1, grid = tune_threshold(y_eval, proba)
        tuning = {
            "best_threshold": best_t,
            "f1_at_best": best_f1,
            "f1_at_default": f1,
            "grid": grid,
        }

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_best_threshold", best_f1)
        mlflow.log_metric("best_threshold", best_t)
        mlflow.log_metric("positive_rate", pos_rate)
        mlflow.sklearn.log_model(model, "model")

        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"Nguong tot nhat {best_t:.2f} -> F1 {best_f1:.4f} "
              f"({best_f1 - f1:+.4f} so voi nguong 0.50)")

        # Bao truoc ngay tren may ca nhan xem quality gate trong CI se cho qua
        # hay chan, khoi phai push roi moi biet.
        verdict = "QUA" if f1 >= F1_THRESHOLD else "BI CHAN"
        print(f"Quality gate (f1 >= {F1_THRESHOLD}): {verdict}")

        detail = write_detail(
            y_eval, preds, params, pos_rate,
            {**tuning, "grid": grid},
        )
        mlflow.log_artifact("outputs/detail.txt")
        print()
        print(detail)

        # File nay duoc doc boi GitHub Actions o Buoc 2
        with open("outputs/report.json", "w") as f:
            json.dump(
                {
                    "f1_score": f1,
                    "accuracy": acc,
                    "positive_rate": round(pos_rate, 4),
                    "best_threshold": best_t,
                    "f1_at_best_threshold": best_f1,
                },
                f,
            )

        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
