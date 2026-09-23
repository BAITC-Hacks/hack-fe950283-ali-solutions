"""Артефакт обрыва: отличаем настоящие «стоки» от узлов, на которых закончился обход.

Идея. У узлов 1–3 колена исходящие переводы выгружены полностью, поэтому для них
известно, передают ли они деньги дальше. Обучаем на них прозрачную логистическую
регрессию «передаёт ли узел дальше» ТОЛЬКО по признакам входящей стороны — они у
узлов 4-го колена наблюдаются так же, как у остальных. Применяем к 444 узлам
4-го колена и получаем P(сток) вместо наивного «out_deg = 0 ⇒ сток».
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

FEATURES = {
    "log_in_kzt": "log(получено ₸)",
    "in_deg": "число плательщиков",
    "log_in_tx": "log(число поступлений)",
    "log_max_in_tx": "log(крупнейшее поступление)",
    "in_days": "дней с поступлениями",
    "last_in_day": "день последнего поступления",
    "from_fanout": "доля входа от веерных отправителей (≥10 получателей)",
    "payer_seed_share": "доля плательщиков-seed",
}


def _design(df: pd.DataFrame, payer_fanout: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["log_in_kzt"] = np.log1p(df.in_kzt)
    X["in_deg"] = df.in_deg
    X["log_in_tx"] = np.log1p(df.in_tx)
    X["log_max_in_tx"] = np.log1p(df.max_in_tx)
    X["in_days"] = df.in_days
    X["last_in_day"] = df.last_in_day
    X["from_fanout"] = payer_fanout
    X["payer_seed_share"] = df.seed_payers / df.in_deg.replace(0, np.nan)
    return X.fillna(0.0)


def fit_predict(G, df: pd.DataFrame):
    """Возвращает df с колонкой p_forward (вероятность, что узел передаёт деньги дальше) и отчёт."""
    fanout = {}
    for g in df.index:
        tot = df.at[g, "in_kzt"]
        if tot <= 0:
            fanout[g] = 0.0
            continue
        s = sum(d["sum_kzt"] for u, _, d in G.in_edges(g, data=True) if G.out_degree(u) >= C.DISTR_MIN_RECIPIENTS)
        fanout[g] = s / tot
    X = _design(df, pd.Series(fanout))

    train = (df.depth.between(1, C.MAX_DEPTH - 1)) & (~df.is_seed)
    y = (df.out_deg[train] > 0).astype(int)
    class_counts = y.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError("Модель обрыва требует минимум по 2 узла со входом и с/без исходящих на коленах 1–3")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    folds = min(5, int(class_counts.min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.RANDOM_SEED)
    oof = cross_val_predict(model, X[train], y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, oof)
    model.fit(X[train], y)

    df["p_forward"] = np.where(df.out_observed, (df.out_deg > 0).astype(float), model.predict_proba(X)[:, 1])
    trunc = df.truncated
    coefs = model[-1].coef_[0]
    report = {
        "train_nodes": int(train.sum()),
        "train_sink_rate": float(1 - y.mean()),
        "cv_auc": float(auc),
        "cv_folds": folds,
        "cv_brier": float(brier_score_loss(y, oof)),
        "baseline_brier": float(brier_score_loss(y, np.full(len(y), y.mean()))),
        "coefficients": {FEATURES[k]: round(float(c), 3) for k, c in zip(X.columns, coefs)},
        "truncated_nodes": int(trunc.sum()),
        "expected_true_sinks": float((1 - df.p_forward[trunc]).sum()),
        "likely_forwarders": int((df.p_forward[trunc] >= C.TRUNC_TERMINAL_P).sum()),
        "likely_sinks": int((df.p_forward[trunc] <= 1 - C.TRUNC_TERMINAL_P).sum()),
        "uncertain_nodes": int(df.p_forward[trunc].between(1 - C.TRUNC_TERMINAL_P, C.TRUNC_TERMINAL_P, inclusive="neither").sum()),
    }
    return df, report
