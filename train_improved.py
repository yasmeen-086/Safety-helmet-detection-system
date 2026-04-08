"""
SafeSight — Improved Trainer
Targets 85%+ accuracy through:
  1. Richer feature extraction (HOG + multi-scale LBP + color + edge)
  2. SVM with calibrated probabilities + tuned threshold
  3. Augmentation-aware pipeline
  4. Hard-negative mining pass
  5. Class-weight balancing

Usage:
    python train_improved.py --dataset ./dataset --output helmet_svm_model.pkl
"""

import os, sys, argparse, pickle, time
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, roc_auc_score, f1_score)
from skimage.feature import hog, local_binary_pattern

_MPL_CONFIG_DIR = os.path.join(os.path.dirname(__file__), ".mplconfig")
os.makedirs(_MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from helmet_detector import MODEL_PATH

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Feature config (must match helmet_detector.py) ───────────────────────────
HOG_WIN_SIZE        = (64, 64)
HOG_ORIENTATIONS    = 12          # ↑ from 9 → more gradient detail
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)


def _preprocess(image):
    resized = cv2.resize(image, HOG_WIN_SIZE, interpolation=cv2.INTER_CUBIC)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized
        resized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    return resized, gray


def _hog_features(gray):
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )


def _lbp_features(gray):
    feats = []
    for (P, R) in [(8, 1), (16, 2), (24, 3)]:   # 3 scales ↑ from 2
        lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-7)
        feats.append(hist)
    return np.hstack(feats)


def _color_features(image):
    resized = cv2.resize(image, HOG_WIN_SIZE, interpolation=cv2.INTER_CUBIC)
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    feats = []
    # H: 18 bins, S: 16 bins ↑, V: 16 bins ↑
    for ch, bins, rng in [(0, 18, (0, 180)), (1, 16, (0, 256)), (2, 16, (0, 256))]:
        hist = cv2.calcHist([hsv], [ch], None, [bins], list(rng)).flatten()
        hist /= (hist.sum() + 1e-7)
        feats.append(hist)
    return np.hstack(feats)


def _edge_features(gray):
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    grad_stats = [np.mean(mag), np.std(mag), np.median(mag), np.max(mag)]
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()[:5]
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return np.hstack([[edge_density], grad_stats, hu])


def extract_features(image):
    """Combined HOG + multi-scale LBP + HSV color + edge/shape features."""
    _, gray = _preprocess(image)
    resized = cv2.resize(image, HOG_WIN_SIZE, interpolation=cv2.INTER_CUBIC)
    hog_f   = _hog_features(gray)
    lbp_f   = _lbp_features(gray)
    col_f   = _color_features(resized)
    edg_f   = _edge_features(gray)
    return np.hstack([hog_f, lbp_f, col_f, edg_f])


# ─── Augmentation ─────────────────────────────────────────────────────────────
def augment(img):
    results = [img]
    results.append(cv2.flip(img, 1))
    for alpha, beta in [(1.25, 15), (0.75, -15), (1.5, 25), (0.6, -25)]:
        results.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))
    results.append(cv2.GaussianBlur(img, (3, 3), 0))
    results.append(cv2.GaussianBlur(img, (5, 5), 0))
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    for angle in [-20, -10, 10, 20]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        results.append(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    # Flip + rotate combos
    flipped = cv2.flip(img, 1)
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        results.append(cv2.warpAffine(flipped, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    return results


# ─── Dataset loading ──────────────────────────────────────────────────────────
def load_raw_dataset(dataset_dir):
    dirs = {
        "helmet":    (os.path.join(dataset_dir, "helmet"), 1),
        "no_helmet": (os.path.join(dataset_dir, "no_helmet"), 0),
    }
    paths, labels = [], []
    skipped = 0
    for name, (folder, label) in dirs.items():
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing: {folder}")
        files = [f for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in SUPPORTED]
        print(f"  {name}: {len(files)} candidates", end="", flush=True)
        kept = 0
        for fname in files:
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None or img.shape[0] < 8 or img.shape[1] < 8:
                skipped += 1
                continue
            paths.append(path)
            labels.append(label)
            kept += 1
        print(f" → {kept} usable")
    if skipped:
        print(f"  ⚠ Skipped unreadable/tiny images: {skipped}")
    return np.array(paths), np.array(labels)


def build_feature_matrix(image_paths, labels, augment_data=False, split_name="Split"):
    X, y = [], []
    skipped = 0
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None or img.shape[0] < 8 or img.shape[1] < 8:
            skipped += 1
            continue
        variants = augment(img) if augment_data else [img]
        label = labels[idx]
        for v in variants:
            try:
                X.append(extract_features(v))
                y.append(label)
            except Exception:
                pass
    X = np.array(X)
    y = np.array(y)
    n_helmet = int((y == 1).sum())
    n_no_helmet = int((y == 0).sum())
    print(f"  {split_name}: {len(X)} feature vectors | helmet={n_helmet} | no_helmet={n_no_helmet}")
    if skipped:
        print(f"    skipped in {split_name}: {skipped}")
    return X, y


class ThresholdedPipeline:
    def __init__(self, base, threshold=0.5):
        self.base = base
        self.threshold = threshold

    def predict(self, X):
        p = self.base.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.base.predict_proba(X)


def stratified_three_way_split(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=random_state
    )
    val_within_train_val = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_within_train_val,
        stratify=y_train_val,
        random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_split(model, X, y, split_name):
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    auc = roc_auc_score(y, probs)
    cm = confusion_matrix(y, preds)
    print(f"\n[{split_name}] Accuracy: {acc:.1%} | Macro-F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(classification_report(y, preds, target_names=['No Helmet', 'Helmet']))
    print("Confusion Matrix:")
    print(f"              No Helmet  Helmet")
    print(f"  No Helmet     {cm[0][0]:5d}     {cm[0][1]:5d}")
    print(f"  Helmet        {cm[1][0]:5d}     {cm[1][1]:5d}")
    return {
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "probs": probs,
        "preds": preds,
    }


def detect_overfitting(train_metrics, val_metrics, test_metrics):
    gap_val = train_metrics["acc"] - val_metrics["acc"]
    gap_test = train_metrics["acc"] - test_metrics["acc"]
    f1_gap_val = train_metrics["f1"] - val_metrics["f1"]

    print("\n" + "=" * 55)
    print("🧠 OVERFITTING CHECK")
    print("=" * 55)
    print(f"  Train-Valid accuracy gap: {gap_val:.1%}")
    print(f"  Train-Test  accuracy gap: {gap_test:.1%}")
    print(f"  Train-Valid macro-F1 gap: {f1_gap_val:.4f}")

    overfit = (gap_val > 0.08 and f1_gap_val > 0.08) or (gap_test > 0.10)
    if overfit:
        print("  ⚠ Potential overfitting detected (train performance significantly exceeds validation/test).")
    else:
        print("  ✅ No strong overfitting signal based on current split metrics.")
    return overfit


def save_plots(plot_dir, y_train, y_val, y_test, train_metrics, val_metrics, test_metrics):
    os.makedirs(plot_dir, exist_ok=True)

    # 1) Class distribution across splits
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    splits = ["Train", "Validation", "Test"]
    no_h = [(y_train == 0).sum(), (y_val == 0).sum(), (y_test == 0).sum()]
    h = [(y_train == 1).sum(), (y_val == 1).sum(), (y_test == 1).sum()]
    x = np.arange(len(splits))
    width = 0.35
    ax1.bar(x - width / 2, no_h, width, label="No Helmet")
    ax1.bar(x + width / 2, h, width, label="Helmet")
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.set_ylabel("Samples")
    ax1.set_title("Class Distribution by Split")
    ax1.legend()
    fig1.tight_layout()
    path1 = os.path.join(plot_dir, "class_distribution.png")
    fig1.savefig(path1, dpi=140)
    plt.close(fig1)

    # 2) Accuracy/F1 comparison
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    accs = [train_metrics["acc"], val_metrics["acc"], test_metrics["acc"]]
    f1s = [train_metrics["f1"], val_metrics["f1"], test_metrics["f1"]]
    ax2.plot(splits, accs, marker="o", label="Accuracy")
    ax2.plot(splits, f1s, marker="o", label="Macro-F1")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Performance Across Splits")
    ax2.set_ylabel("Score")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    path2 = os.path.join(plot_dir, "split_performance.png")
    fig2.savefig(path2, dpi=140)
    plt.close(fig2)

    # 3) Confusion matrices (validation + test)
    fig3, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm, title in [
        (axes[0], val_metrics["cm"], "Validation CM"),
        (axes[1], test_metrics["cm"], "Test CM"),
    ]:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Helmet", "Helmet"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No Helmet", "Helmet"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig3.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig3.tight_layout()
    path3 = os.path.join(plot_dir, "confusion_matrices.png")
    fig3.savefig(path3, dpi=140)
    plt.close(fig3)

    print("\n📈 Plots saved:")
    print(f"  - {path1}")
    print(f"  - {path2}")
    print(f"  - {path3}")

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output",  default=MODEL_PATH)
    parser.add_argument("--plot-dir", default="plots")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--fast",       action="store_true",
                        help="Skip grid search, use pre-tuned params")
    args = parser.parse_args()

    t0 = time.time()
    print(f"\n📂 Loading dataset from: {args.dataset}")
    paths, labels = load_raw_dataset(args.dataset)
    n_helmet = int((labels == 1).sum())
    n_no_helmet = int((labels == 0).sum())
    print(f"\n   Originals: {len(paths)} images | helmet: {n_helmet} | no_helmet: {n_no_helmet}")

    p_train, p_val, p_test, y_train_base, y_val_base, y_test_base = stratified_three_way_split(
        paths, labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42
    )
    print(f"   Split originals -> Train: {len(p_train)} | Validation: {len(p_val)} | Test: {len(p_test)}")
    print("   Building feature matrices (augmentation on Train only)…")
    X_train, y_train = build_feature_matrix(
        p_train, y_train_base, augment_data=not args.no_augment, split_name="Train"
    )
    X_val, y_val = build_feature_matrix(
        p_val, y_val_base, augment_data=False, split_name="Validation"
    )
    X_test, y_test = build_feature_matrix(
        p_test, y_test_base, augment_data=False, split_name="Test"
    )
    print(f"   Feature vector dim: {X_train.shape[1]}")
    print(f"   Load+featurize time: {time.time()-t0:.1f}s")

    # ── More expressive model with built-in regularization ──
    # Scaler + PCA (+noise compression) + RBF-SVM
    if args.fast:
        print("\n🏋️  Training with pre-tuned PCA+SVM …")
        model = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full")),
            ("svm", SVC(kernel="rbf", C=5.0, gamma="scale",
                        probability=True, class_weight="balanced")),
        ])
        model.fit(X_train, y_train)
        print("   ✅ PCA+SVM trained")
    else:
        print("\n🔍 Running grid search for PCA+SVM (train split only) …")
        base_model = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(svd_solver="full")),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ])
        param_grid = {
            "pca__n_components": [0.90, 0.95, 0.98],
            "svm__C": [0.5, 1, 5, 10],
            "svm__gamma": ["scale", 0.01, 0.005],
        }
        gs = GridSearchCV(
            base_model,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring="f1_macro",
            n_jobs=1,
            verbose=1,
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print(f"\n   ✅ Best params: {gs.best_params_}")
        print(f"   ✅ Best CV macro-F1: {gs.best_score_:.4f}")

    # ── Threshold tuning on validation slice ──
    print("\n🎯 Tuning decision threshold …")
    val_probas = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.02):
        preds_t = (val_probas >= t).astype(int)
        f1 = f1_score(y_val, preds_t, average="macro")
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"   Best threshold: {best_thresh:.2f}  (validation macro-F1={best_f1:.4f})")

    # class ThresholdedPipeline:
    #     def __init__(self, base, threshold=0.5):
    #         self.base      = base
    #         self.threshold = threshold
    #     def predict(self, X):
    #         p = self.base.predict_proba(X)[:, 1]
    #         return (p >= self.threshold).astype(int)
    #     def predict_proba(self, X):
    #         return self.base.predict_proba(X)

    final_model = ThresholdedPipeline(model, best_thresh)

    # ── Evaluation ──
    print("\n" + "="*55)
    print("📊 EVALUATION")
    print("="*55)
    train_metrics = evaluate_split(final_model, X_train, y_train, "Train")
    val_metrics = evaluate_split(final_model, X_val, y_val, "Validation")
    test_metrics = evaluate_split(final_model, X_test, y_test, "Test")

    detect_overfitting(train_metrics, val_metrics, test_metrics)

    best_pca = model.named_steps["pca"].n_components
    best_C = model.named_steps["svm"].C
    best_gamma = model.named_steps["svm"].gamma
    cv_scores = cross_val_score(
        Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=best_pca, svd_solver="full")),
            ("svm", SVC(kernel="rbf", C=best_C, gamma=best_gamma,
                        class_weight="balanced")),
        ]),
        X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy", n_jobs=1
    )
    print(f"  5-Fold CV Acc (train split): {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    save_plots(
        plot_dir=args.plot_dir,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )

    # ── Save ──
    with open(args.output, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\n✅ Model saved → {args.output}  ({os.path.getsize(args.output)/1024:.0f} KB)")
    print(f"   Total time: {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
