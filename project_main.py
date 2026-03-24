"""
╔══════════════════════════════════════════════════════════════════════╗
║   NeuroBehavior Clinical Health Risk — Sınıflandırma Projesi        ║
║   Hedef: Mental_Health_Risk (0 = Düşük Risk, 1 = Yüksek Risk)       ║
║   Yöntem: 10-Fold Stratified Cross Validation                       ║
║                                                                      ║
║   Metrikler:                                                         ║
║     • Accuracy                                                       ║
║     • Recall  (Sensitivity / True Positive Rate)                    ║
║     • Specificity (True Negative Rate)                               ║
║     • Precision                                                      ║
║     • F1-Score                                                       ║
║     • Matthews Correlation Coefficient (MCC)                         ║
║                                                                      ║
║   Modeller:                                                          ║
║     kNN | Naive Bayes | LSVM | RBF SVM | Random Forest |            ║
║     MLP | XGBoost | Decision Tree | Logistic Regression             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────
# 0. KÜTÜPHANELER
# ─────────────────────────────────────────────
import os, sys
import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")          # GUI gerektirmeden grafik üret
import matplotlib.pyplot as plt

# ─── Çalışma dizini ayarla ───────────────────
# CSV dosyalarının bulunduğu klasörde ya da bu scriptin klasöründe çalışır.
# Kaggle / Colab için: dosyaları script ile aynı klasöre koy.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_CANDIDATES = [
    _THIS_DIR,                            # script ile aynı klasör
    os.path.join(_THIS_DIR, "data"),      # ./data/ alt klasörü
    "/mnt/user-data/uploads",            # claude.ai upload klasörü
    os.getcwd(),
]
_DATA_DIR = next(
    (d for d in _DATA_CANDIDATES if os.path.exists(os.path.join(d, "train.csv"))),
    _THIS_DIR
)
_OUT_DIR = _THIS_DIR   # çıktılar script ile aynı klasöre

def dp(fname):  return os.path.join(_DATA_DIR, fname)
def op(fname):  return os.path.join(_OUT_DIR,  fname)
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection  import StratifiedKFold, cross_val_predict
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (
    accuracy_score, recall_score, precision_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    roc_curve, auc, RocCurveDisplay
)
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.naive_bayes      import GaussianNB
from sklearn.svm              import SVC
from sklearn.ensemble         import RandomForestClassifier
from sklearn.neural_network   import MLPClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("⚠  XGBoost bulunamadı → pip install xgboost")

warnings.filterwarnings("ignore")

BANNER = "═" * 68

# ─────────────────────────────────────────────
# 1. VERİ YÜKLEME & BİRLEŞTİRME
# ─────────────────────────────────────────────
print(f"\n{BANNER}")
print("  ADIM 1 — Veri Yükleme")
print(BANNER)

# Birincil kaynak: train.csv (etiket mevcut)
train     = pd.read_csv(dp("train.csv"))
test      = pd.read_csv(dp("test.csv"))
main_df   = pd.read_csv(dp("NeuroBehavior-Clinical_Health_Risk_Sample_Dataset.csv"))
submission= pd.read_csv(dp("sample_submission.csv"))

TARGET = "Mental_Health_Risk"

print(f"  train.csv       : {train.shape[0]} satır × {train.shape[1]} sütun")
print(f"  test.csv        : {test.shape[0]} satır × {test.shape[1]} sütun")
print(f"  NeuroBehavior   : {main_df.shape[0]} satır × {main_df.shape[1]} sütun")
print(f"\n  Hedef Sütun     : '{TARGET}'")
print(f"  Sınıf Dağılımı  : {train[TARGET].value_counts().to_dict()}")
print(f"  (0 = Düşük Risk, 1 = Yüksek Risk)")

# ─────────────────────────────────────────────
# 2. ÖN İŞLEME
# ─────────────────────────────────────────────
print(f"\n{BANNER}")
print("  ADIM 2 — Ön İşleme")
print(BANNER)

# Timestamp sütununu dönüştür → saat bilgisini çıkar
def extract_hour(df):
    if "Timestamp" in df.columns:
        df = df.copy()
        df["Hour"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.hour.fillna(0).astype(int)
        df = df.drop(columns=["Timestamp"])
    return df

train = extract_hour(train)
test  = extract_hour(test)

# ID sütununu koru (submission için), modelden çıkar
train_ids = train["ID"].values
test_ids  = test["ID"].values

DROP_COLS = ["ID"]
X_train_full = train.drop(columns=DROP_COLS + [TARGET])
y             = train[TARGET].values
X_test_full  = test.drop(columns=DROP_COLS)

# Eksik değer yoktu, yine de güvenlik için medyan
X_train_full = X_train_full.fillna(X_train_full.median(numeric_only=True))
X_test_full  = X_test_full.fillna(X_train_full.median(numeric_only=True))

feature_names = X_train_full.columns.tolist()
print(f"  Kullanılan özellikler ({len(feature_names)}): {feature_names}")
print(f"  Eksik değer (train): {X_train_full.isnull().sum().sum()}")

# Ölçeklendirme (SVM / MLP / Logistic için)
scaler      = StandardScaler()
X_scaled    = scaler.fit_transform(X_train_full)
X_test_sc   = scaler.transform(X_test_full)
X_raw       = X_train_full.values

SCALED_SET  = {"LSVM", "RBF SVM", "MLP", "Logistic Reg."}

# ─────────────────────────────────────────────
# 3. MODELLERİ TANIMLAMA
# ─────────────────────────────────────────────
print(f"\n{BANNER}")
print("  ADIM 3 — Modeller")
print(BANNER)

models = {
    "kNN"           : KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes"   : GaussianNB(),
    "LSVM"          : SVC(kernel="linear", C=1.0, probability=True, random_state=42),
    "RBF SVM"       : SVC(kernel="rbf",    C=1.0, gamma="scale", probability=True, random_state=42),
    "Random Forest" : RandomForestClassifier(n_estimators=200, random_state=42),
    "MLP"           : MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600,
                                    early_stopping=True, random_state=42),
    "Decision Tree" : DecisionTreeClassifier(max_depth=8, random_state=42),
    "Logistic Reg." : LogisticRegression(max_iter=1000, random_state=42),
}
if XGB_OK:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        eval_metric="logloss", random_state=42, verbosity=0
    )

for name in models:
    print(f"  ✔ {name}")

# ─────────────────────────────────────────────
# 4. YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────

def specificity_score(y_true, y_pred):
    """Binary specificity = TN / (TN + FP)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def evaluate(name, model, X_data, y_true, cv):
    """10-Fold CV ile tüm metrikleri hesapla."""
    y_pred = cross_val_predict(model, X_data, y_true, cv=cv, method="predict")

    acc   = accuracy_score (y_true, y_pred)
    rec   = recall_score   (y_true, y_pred, pos_label=1, zero_division=0)   # Sensitivity
    spec  = specificity_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1    = f1_score       (y_true, y_pred, pos_label=1, zero_division=0)
    mcc   = matthews_corrcoef(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "Model"      : name,
        "Accuracy"   : round(acc  * 100, 2),
        "Recall"     : round(rec  * 100, 2),
        "Specificity": round(spec * 100, 2),
        "Precision"  : round(prec * 100, 2),
        "F1-Score"   : round(f1   * 100, 2),
        "MCC"        : round(mcc,         4),
        "_cm"        : cm,
        "_y_pred"    : y_pred,
    }

# ─────────────────────────────────────────────
# 5. 10-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────
print(f"\n{BANNER}")
print("  ADIM 4 — 10-Fold Stratified Cross Validation")
print(BANNER)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

results    = []
cms        = {}
pred_store = {}

for name, model in models.items():
    print(f"  ⏳ {name:<18} ...", end="\r")
    X_input = X_scaled if name in SCALED_SET else X_raw
    res = evaluate(name, model, X_input, y, cv)
    cms[name]        = res.pop("_cm")
    pred_store[name] = res.pop("_y_pred")
    results.append(res)
    print(f"  ✔ {name:<18}  "
          f"Acc={res['Accuracy']:6.2f}%  "
          f"F1={res['F1-Score']:6.2f}%  "
          f"MCC={res['MCC']:+.4f}")

# ─────────────────────────────────────────────
# 6. SONUÇ TABLOSU
# ─────────────────────────────────────────────
print(f"\n{BANNER}")
print("  ADIM 5 — Sonuç Tablosu")
print(BANNER)

results_df = pd.DataFrame(results).set_index("Model")

# Terminale yazdır
col_fmt = {"Accuracy":"6.2f","Recall":"6.2f","Specificity":"6.2f",
           "Precision":"6.2f","F1-Score":"6.2f","MCC":"+.4f"}

header = f"{'Model':<18}" + "".join(f"  {c:>12}" for c in results_df.columns)
print(f"\n  {header}")
print("  " + "─" * (len(header)))
for model_name, row in results_df.iterrows():
    vals = "".join(f"  {row[c]:>12.2f}" if c != "MCC" else f"  {row[c]:>+12.4f}"
                   for c in results_df.columns)
    print(f"  {model_name:<18}{vals}")

# En iyi modeller
print(f"\n  {'─'*50}")
for metric in ["Accuracy", "F1-Score", "MCC"]:
    best = results_df[metric].idxmax()
    val  = results_df.loc[best, metric]
    unit = "%" if metric != "MCC" else ""
    print(f"  🏆 En İyi {metric:<12}: {best:<18} → {val}{unit}")

# CSV kaydet
results_df.to_csv(op("classification_results.csv"))
print("\n  ✔ Sonuçlar → classification_results.csv")

# ─────────────────────────────────────────────
# 7. GÖRSELLEŞTİRMELER
# ─────────────────────────────────────────────
print(f"\n{BANNER}")
print("  ADIM 6 — Görseller")
print(BANNER)

# ── 7a. Metrik Bar Grafikleri ──────────────────
metric_cols = ["Accuracy", "Recall", "Specificity", "Precision", "F1-Score", "MCC"]
colors = sns.color_palette("tab10", len(results_df))

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle(
    "NeuroBehavior Clinical Health Risk\nModel Karşılaştırması — 10-Fold Stratified CV",
    fontsize=14, fontweight="bold", y=1.01
)

for ax, metric in zip(axes.flatten(), metric_cols):
    vals   = results_df[metric]
    bars   = ax.barh(results_df.index, vals, color=colors, edgecolor="white", height=0.6)
    ax.set_title(metric, fontweight="bold", fontsize=12)
    is_mcc = (metric == "MCC")
    ax.set_xlabel("MCC Skoru" if is_mcc else "%", fontsize=10)

    xlim_max = 100 if not is_mcc else max(1.05, vals.max() * 1.1)
    xlim_min = 0   if not is_mcc else min(-0.1, vals.min() * 1.1)
    ax.set_xlim(xlim_min, xlim_max)
    ax.axvline(0, color="gray", linewidth=0.5)

    best_idx = list(results_df.index).index(vals.idxmax())
    for i, (bar, val) in enumerate(zip(bars, vals)):
        x_pos = bar.get_width() + (0.5 if not is_mcc else 0.01)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8.5)
        if i == best_idx:
            bar.set_edgecolor("gold")
            bar.set_linewidth(2.5)

plt.tight_layout()
plt.savefig(op("01_model_comparison_bars.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ 01_model_comparison_bars.png")

# ── 7b. Isıl Harita ───────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
hm_data = results_df.copy()
# MCC'yi [0,100] normalize et sadece renk skalası için
hm_norm = hm_data.copy()
hm_norm["MCC"] = (hm_norm["MCC"] + 1) / 2 * 100

sns.heatmap(
    hm_norm, annot=hm_data.values, fmt=".2f",
    cmap="YlOrRd", linewidths=0.6, linecolor="white",
    ax=ax, cbar_kws={"label": "Normalize Skor (%)"}
)
ax.set_title(
    "Performans Isıl Haritası — 10-Fold CV\n"
    "(Tablodaki MCC değerleri gerçek, renk skalası normalize edilmiş)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(op("02_heatmap_results.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ 02_heatmap_results.png")

# ── 7c. Karışıklık Matrisleri ──────────────────
n_models = len(models)
cols_cm  = min(3, n_models)
rows_cm  = (n_models + cols_cm - 1) // cols_cm

fig, axes = plt.subplots(rows_cm, cols_cm,
                         figsize=(5.5 * cols_cm, 4.5 * rows_cm))
axes_flat = axes.flatten() if n_models > 1 else [axes]
fig.suptitle("Karışıklık Matrisleri (10-Fold CV — tüm tahminler birleşik)",
             fontsize=14, fontweight="bold", y=1.01)

labels_cm = ["0\n(Düşük Risk)", "1\n(Yüksek Risk)"]
for ax, (name, cm) in zip(axes_flat, cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_cm, yticklabels=labels_cm,
                ax=ax, cbar=False, linewidths=0.5)
    acc_val = results_df.loc[name, "Accuracy"]
    ax.set_title(f"{name}\n(Acc: {acc_val:.2f}%)", fontweight="bold")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")

for ax in axes_flat[len(cms):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(op("03_confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ 03_confusion_matrices.png")

# ── 7d. Radar Grafiği ─────────────────────────
radar_metrics = ["Accuracy", "Recall", "Specificity", "Precision", "F1-Score"]
radar_data    = results_df[radar_metrics].copy()

angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics, fontsize=11)

cmap_radar = plt.cm.tab10
patches    = []
for i, (model_name, row) in enumerate(radar_data.iterrows()):
    vals_r = row.tolist() + [row.tolist()[0]]
    c      = cmap_radar(i / len(radar_data))
    ax.plot(angles, vals_r, linewidth=1.8, linestyle="solid", color=c)
    ax.fill(angles, vals_r, alpha=0.07, color=c)
    patches.append(mpatches.Patch(color=c, label=model_name))

ax.set_ylim(0, 100)
ax.set_title("Model Radar Karşılaştırması (10-Fold CV)\n",
             fontsize=13, fontweight="bold")
ax.legend(handles=patches, loc="lower right",
          bbox_to_anchor=(1.35, -0.05), fontsize=9)

plt.tight_layout()
plt.savefig(op("04_radar_chart.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ 04_radar_chart.png")

# ── 7e. En İyi Model — ROC Eğrisi ────────────
best_by_f1 = results_df["F1-Score"].idxmax()
best_model_obj = models[best_by_f1]
X_best = X_scaled if best_by_f1 in SCALED_SET else X_raw

try:
    y_prob_cv = cross_val_predict(best_model_obj, X_best, y, cv=cv, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob_cv)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2.5, color="steelblue",
            label=f"{best_by_f1}\nAUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Rastgele")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", fontsize=12)
    ax.set_ylabel("Doğru Pozitif Oranı (TPR)", fontsize=12)
    ax.set_title(f"ROC Eğrisi — {best_by_f1}\n(En Yüksek F1-Score Modeli)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(op("05_roc_curve_best.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ 05_roc_curve_best.png  (AUC = {roc_auc:.4f})")
except Exception as e:
    print(f"  ⚠  ROC grafiği oluşturulamadı: {e}")

# ── 7f. Özellik Önemi (Random Forest) ────────
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_raw, y)
importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors_fi = ["#e74c3c" if v >= importances.quantile(0.75) else "#3498db"
             for v in importances.values]
importances.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
ax.set_title("Özellik Önemi — Random Forest\n(Kırmızı: En Önemli %25)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Gini Önem Skoru")
ax.axvline(importances.quantile(0.75), color="crimson", linestyle="--", lw=1.5,
           label="75. Yüzdelik")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(op("06_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✔ 06_feature_importance.png")

# ─────────────────────────────────────────────
# 8. TEST SETİ TAHMİNİ (Submission)
# ─────────────────────────────────────────────
print(f"\n{BANNER}")
print("  ADIM 7 — Test Seti Tahmini (submission.csv)")
print(BANNER)

# En iyi modeli (F1'e göre) tüm train üzerinde yeniden eğit
best_for_sub = results_df["F1-Score"].idxmax()
final_model  = models[best_for_sub]
X_sub_input  = X_scaled  if best_for_sub in SCALED_SET else X_raw
X_test_input = X_test_sc if best_for_sub in SCALED_SET else X_test_full.values

final_model.fit(X_sub_input, y)
test_preds = final_model.predict(X_test_input)

submission_out = pd.DataFrame({"ID": test_ids, "Mental_Health_Risk": test_preds})
submission_out.to_csv(op("submission_predictions.csv"), index=False)
print(f"  ✔ Kullanılan Model   : {best_for_sub}")
print(f"  ✔ Tahmin Dağılımı   : {dict(pd.Series(test_preds).value_counts().sort_index())}")
print(f"  ✔ Kaydedildi         : submission_predictions.csv")

# ─────────────────────────────────────────────
# 9. ÖZET
# ─────────────────────────────────────────────
print(f"\n{BANNER}")
print("  PROJE TAMAMLANDI")
print(BANNER)
print("""
  Üretilen Dosyalar:
  ┌─────────────────────────────────────────────────────┐
  │  📊  classification_results.csv    (tüm metrikler)  │
  │  📈  01_model_comparison_bars.png  (bar grafikleri) │
  │  🔥  02_heatmap_results.png        (ısıl harita)    │
  │  🧩  03_confusion_matrices.png     (karmaşıklık)    │
  │  🕸   04_radar_chart.png            (radar)         │
  │  📉  05_roc_curve_best.png         (ROC eğrisi)     │
  │  🌲  06_feature_importance.png     (özellik önemi)  │
  │  🎯  submission_predictions.csv    (test tahminleri)│
  └─────────────────────────────────────────────────────┘
""")