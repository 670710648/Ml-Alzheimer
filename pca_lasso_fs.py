from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import BisectingKMeans, DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET = "Diagnosis"
IDENTIFIER_COLUMNS = ("PatientID", "DoctorInCharge")
CATEGORICAL_COLUMNS = ("Gender", "Ethnicity", "EducationLevel")
LASSO_ALPHA_CANDIDATES = (0.001, 0.003, 0.01, 0.02, 0.05)
LASSO_THRESHOLD = 0.00001
PCA_VARIANCE = 0.95
CV_SPLITS = 3
TEST_TRAIN_SPLITS = (
    (0.05, "5:95"),
    (0.10, "10:90"),
    (0.20, "20:80"),
    (0.30, "30:70"),
    (0.40, "40:60"),
    (0.50, "50:50"),
    (0.70, "70:30"),
    (0.80, "80:20"),
    (0.90, "90:10"),
    (0.95, "95:5"),
)
MODEL_ORDER = (
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "Stacking Ensemble",
)
MODEL_COLORS = {
    "Logistic Regression": "#E15759",
    "Decision Tree": "#4C78A8",
    "Random Forest": "#59A14F",
    "XGBoost": "#F28E2B",
    "Stacking Ensemble": "#B07AA1",
}
FEATURE_CLUSTER_DISTANCE_THRESHOLD = 0.55
FEATURE_CLUSTER_MIN_CORRELATION = (
    1 - FEATURE_CLUSTER_DISTANCE_THRESHOLD**2 / 2
)
FEATURE_DBSCAN_MIN_SAMPLES = 2
SCORING = {
    "accuracy": "accuracy",
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
}


def engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add a few interpretable clinical measures before feature selection."""
    engineered = features.copy()
    engineered["PulsePressure"] = (
        engineered["SystolicBP"] - engineered["DiastolicBP"]
    )
    engineered["NonHDLCholesterol"] = (
        engineered["CholesterolTotal"] - engineered["CholesterolHDL"]
    )
    engineered["LDL_HDL_Ratio"] = (
        engineered["CholesterolLDL"]
        / engineered["CholesterolHDL"].replace(0, np.nan)
    )
    engineered["Triglyceride_HDL_Ratio"] = (
        engineered["CholesterolTriglycerides"]
        / engineered["CholesterolHDL"].replace(0, np.nan)
    )
    return engineered


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    engineered_columns = list(engineer_features(features).columns)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    numeric = [column for column in engineered_columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def select_generalizable_setting(cv_results: dict[str, object]) -> int:
    """Prefer the smallest train/validation F1 gap within one SE of peak F1."""
    validation_f1 = np.asarray(cv_results["mean_test_f1"], dtype=float)
    validation_std = np.asarray(cv_results["std_test_f1"], dtype=float)
    training_f1 = np.asarray(cv_results["mean_train_f1"], dtype=float)
    peak_index = int(np.argmax(validation_f1))
    lower_bound = validation_f1[peak_index] - (
        validation_std[peak_index] / np.sqrt(CV_SPLITS)
    )
    eligible = np.flatnonzero(validation_f1 >= lower_bound)
    return int(
        min(
            eligible,
            key=lambda index: (
                max(training_f1[index] - validation_f1[index], 0.0),
                -validation_f1[index],
            ),
        )
    )


class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    """Select features with training-fold LassoCV and the supplied threshold."""

    def __init__(
        self,
        alphas: tuple[float, ...] = LASSO_ALPHA_CANDIDATES,
        threshold: float = LASSO_THRESHOLD,
    ):
        self.alphas = alphas
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> LassoFeatureSelector:
        self.n_features_in_ = X.shape[1]
        self.estimator_ = LassoCV(
            alphas=self.alphas, cv=3, max_iter=50000, tol=1e-5
        )
        self.estimator_.fit(X, y)
        coefficients = np.abs(self.estimator_.coef_)
        self.support_ = coefficients > self.threshold
        if not self.support_.any():
            selected = int(np.argmax(coefficients))
            self.support_[selected] = True
            warnings.warn(
                "Lasso selected no coefficients above the threshold; retaining "
                "the largest absolute coefficient so PCA and model fitting remain valid.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.support_]

    def get_support(self) -> np.ndarray:
        return self.support_.copy()

    def get_feature_names_out(
        self, input_features: np.ndarray | None = None
    ) -> np.ndarray:
        if input_features is None:
            input_features = np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )
        input_features = np.asarray(input_features, dtype=object)
        if input_features.size != self.n_features_in_:
            raise ValueError("input_features length does not match fitted input.")
        return input_features[self.support_]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(
        self, X: pd.DataFrame, _y: pd.Series | None = None
    ) -> ClinicalFeatureEngineer:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = np.asarray(
            engineer_features(X).columns, dtype=object
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(X)

    def get_feature_names_out(
        self, input_features: np.ndarray | None = None
    ) -> np.ndarray:
        if input_features is not None and not np.array_equal(
            np.asarray(input_features, dtype=object), self.feature_names_in_
        ):
            raise ValueError("input_features do not match the fitted input columns.")
        return self.feature_names_out_.copy()


class ClusterConsensusFeatureSelector(BaseEstimator, TransformerMixin):
    """Keep one target-associated feature from clusters agreed on by 2 methods."""

    def __init__(
        self,
        distance_threshold: float = FEATURE_CLUSTER_DISTANCE_THRESHOLD,
        dbscan_min_samples: int = FEATURE_DBSCAN_MIN_SAMPLES,
        random_state: int = RANDOM_STATE,
    ):
        self.distance_threshold = distance_threshold
        self.dbscan_min_samples = dbscan_min_samples
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> ClusterConsensusFeatureSelector:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]
        if self.n_features_in_ < 2:
            self.support_ = np.ones(self.n_features_in_, dtype=bool)
            self.method_labels_ = {
                "agglomerative": np.zeros(self.n_features_in_, dtype=int),
                "divisive": np.zeros(self.n_features_in_, dtype=int),
                "dbscan": np.full(self.n_features_in_, -1, dtype=int),
            }
            self.correlation_ = np.eye(self.n_features_in_)
            self.target_association_ = np.zeros(self.n_features_in_)
            self.selected_clusters_ = []
            self.divisive_cluster_count_ = self.n_features_in_
            return self

        correlation = np.corrcoef(X, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        correlation = np.clip(correlation, -1.0, 1.0)
        np.fill_diagonal(correlation, 1.0)
        absolute_correlation = np.abs(correlation)
        distance = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - absolute_correlation)))
        np.fill_diagonal(distance, 0.0)

        hierarchy = linkage(
            squareform(distance, checks=False),
            method="average",
        )
        agglomerative_labels = fcluster(
            hierarchy,
            t=self.distance_threshold,
            criterion="distance",
        )

        similarity_profiles = absolute_correlation
        max_clusters = min(5, self.n_features_in_)
        divisive_models: dict[int, tuple[BisectingKMeans, float]] = {}
        for cluster_count in range(2, max_clusters + 1):
            model = BisectingKMeans(
                n_clusters=cluster_count,
                n_init=3,
                random_state=self.random_state,
                bisecting_strategy="biggest_inertia",
            )
            labels = model.fit_predict(similarity_profiles)
            if np.unique(labels).size < 2:
                continue
            score = float(silhouette_score(similarity_profiles, labels))
            divisive_models[cluster_count] = (model, score)
        if not divisive_models:
            raise ValueError(
                "Bisecting clustering could not form multiple feature groups."
            )
        best_cluster_count = max(
            divisive_models,
            key=lambda count: (divisive_models[count][1], -count),
        )
        divisive_labels = divisive_models[best_cluster_count][0].labels_

        dbscan_labels = DBSCAN(
            eps=self.distance_threshold,
            min_samples=self.dbscan_min_samples,
            metric="precomputed",
        ).fit_predict(distance)
        self.method_labels_ = {
            "agglomerative": agglomerative_labels,
            "divisive": divisive_labels,
            "dbscan": dbscan_labels,
        }
        self.correlation_ = correlation
        self.target_association_ = np.nan_to_num(
            np.abs(np.corrcoef(X, y, rowvar=False)[:-1, -1]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.divisive_cluster_count_ = best_cluster_count
        self.divisive_silhouette_ = divisive_models[best_cluster_count][1]

        parent = list(range(self.n_features_in_))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        for first in range(self.n_features_in_):
            for second in range(first + 1, self.n_features_in_):
                if absolute_correlation[first, second] < (
                    1 - self.distance_threshold**2 / 2
                ):
                    continue
                agreement = sum(
                    labels[first] == labels[second]
                    and not (
                        method == "dbscan" and labels[first] == -1
                    )
                    for method, labels in self.method_labels_.items()
                )
                if agreement >= 2:
                    union(first, second)

        grouped: dict[int, list[int]] = {}
        for index in range(self.n_features_in_):
            grouped.setdefault(find(index), []).append(index)
        self.selected_clusters_ = [
            indices for indices in grouped.values() if len(indices) > 1
        ]
        self.support_ = np.ones(self.n_features_in_, dtype=bool)
        for indices in self.selected_clusters_:
            representative = max(
                indices, key=lambda index: self.target_association_[index]
            )
            for index in indices:
                self.support_[index] = index == representative
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, self.support_]

    def get_support(self) -> np.ndarray:
        return self.support_.copy()


def make_feature_pipeline(
    features: pd.DataFrame, use_feature_selection: bool = True
) -> Pipeline:
    steps = [
        ("engineer", ClinicalFeatureEngineer()),
        ("preprocess", make_preprocessor(features)),
        ("scale", StandardScaler()),
    ]
    if use_feature_selection:
        steps.extend(
            [
                ("cluster_select", ClusterConsensusFeatureSelector()),
                (
                    "lasso",
                    LassoFeatureSelector(
                        alphas=LASSO_ALPHA_CANDIDATES,
                        threshold=LASSO_THRESHOLD,
                    ),
                ),
                ("pca", PCA(n_components=PCA_VARIANCE, svd_solver="full")),
            ]
        )
    return Pipeline(steps=steps)


def make_model_pipeline(
    features: pd.DataFrame,
    classifier: BaseEstimator,
    use_feature_selection: bool = True,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                make_feature_pipeline(features, use_feature_selection),
            ),
            ("classifier", classifier),
        ]
    )


def get_model_searches(
    features: pd.DataFrame,
    y_train: pd.Series,
    use_feature_selection: bool = True,
) -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    positive_weight = negative_count / positive_count

    tree = make_model_pipeline(
        features,
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        use_feature_selection,
    )
    forest = make_model_pipeline(
        features,
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        use_feature_selection,
    )
    xgboost = make_model_pipeline(
        features,
        XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        ),
        use_feature_selection,
    )
    return {
        "Logistic Regression": (
            make_model_pipeline(
                features,
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                use_feature_selection,
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (
            tree,
            {
                "classifier__max_depth": [2, 4, 6, None],
                "classifier__min_samples_leaf": [5, 15],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            forest,
            {
                "classifier__n_estimators": [200],
                "classifier__max_depth": [6, 10],
                "classifier__min_samples_leaf": [8, 15],
                "classifier__max_features": ["sqrt", 0.8],
                "classifier__class_weight": [None, "balanced_subsample"],
            },
        ),
        "XGBoost": (
            xgboost,
            {
                "classifier__n_estimators": [60, 120],
                "classifier__max_depth": [2, 3],
                "classifier__learning_rate": [0.03, 0.07],
                "classifier__scale_pos_weight": [1.0, positive_weight],
            },
        ),
    }


def make_stack_candidate(
    name: str,
    features: pd.DataFrame,
    seed: int,
    use_feature_selection: bool = True,
) -> StackingClassifier:
    base_models: dict[str, BaseEstimator] = {
        "lr": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1500, random_state=seed
        ),
        "svm": SVC(
            C=1.0,
            kernel="rbf",
            class_weight="balanced",
            random_state=seed,
        ),
        "dt": DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=seed,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32,),
            alpha=0.01,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        ),
        "rf": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        ),
    }
    combinations = {
        "LR + RBF-SVM + shallow DT": ("lr", "svm", "dt"),
        "LR + RBF-SVM + MLP": ("lr", "svm", "mlp"),
        "RBF-SVM + MLP + Random Forest": ("svm", "mlp", "rf"),
    }
    if name not in combinations:
        raise ValueError(f"Unknown stacking candidate: {name}")

    estimators = [
        (
            model_name,
            make_model_pipeline(
                features,
                clone(base_models[model_name]),
                use_feature_selection,
            ),
        )
        for model_name in combinations[name]
    ]
    internal_cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=seed
    )
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1500, random_state=seed
        ),
        stack_method="auto",
        cv=internal_cv,
        n_jobs=1,
        passthrough=False,
    )


def fit_and_score_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    test_label: str,
    seed: int,
    use_feature_selection: bool = True,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], list[dict[str, object]]]:
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=seed)
    records: list[dict[str, object]] = []
    matrices: dict[str, np.ndarray] = {}

    for model_name, (pipeline, parameter_grid) in get_model_searches(
        X_train, y_train, use_feature_selection
    ).items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=SCORING,
            refit=select_generalizable_setting,
            cv=cv,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_index = search.best_index_
        predictions = search.best_estimator_.predict(X_test)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        matrices[model_name] = matrix
        record: dict[str, object] = {
            "test_train_split": test_label,
            "test_fraction": len(y_test) / (len(y_train) + len(y_test)),
            "model": model_name,
            "best_parameters": repr(search.best_params_),
            "cv_accuracy": float(
                search.cv_results_["mean_test_accuracy"][best_index]
            ),
            "cv_f1": float(search.cv_results_["mean_test_f1"][best_index]),
            "cv_precision": float(
                search.cv_results_["mean_test_precision"][best_index]
            ),
            "cv_recall": float(search.cv_results_["mean_test_recall"][best_index]),
            "cv_train_f1": float(search.cv_results_["mean_train_f1"][best_index]),
            "cv_f1_gap": float(
                search.cv_results_["mean_train_f1"][best_index]
                - search.cv_results_["mean_test_f1"][best_index]
            ),
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
            "stacking_candidate": "",
        }
        records.append(record)

    candidate_records: list[dict[str, object]] = []
    candidate_names = (
        "LR + RBF-SVM + shallow DT",
        "LR + RBF-SVM + MLP",
        "RBF-SVM + MLP + Random Forest",
    )
    for candidate_name in candidate_names:
        ensemble = make_stack_candidate(
            candidate_name, X_train, seed, use_feature_selection
        )
        scores = cross_validate(
            ensemble,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        candidate_metrics = {
            metric: float(np.mean(scores[f"test_{metric}"]))
            for metric in SCORING
        }
        candidate_train_f1 = float(np.mean(scores["train_f1"]))
        candidate_f1_std = float(np.std(scores["test_f1"], ddof=1))
        candidate_records.append(
            {
                "test_train_split": test_label,
                "candidate": candidate_name,
                **{f"cv_{key}": value for key, value in candidate_metrics.items()},
                "cv_f1_std": candidate_f1_std,
                "cv_train_f1": candidate_train_f1,
                "cv_f1_gap": candidate_train_f1 - candidate_metrics["f1"],
                "selected": False,
            }
        )
    peak_candidate = max(candidate_records, key=lambda row: float(row["cv_f1"]))
    candidate_lower_bound = float(peak_candidate["cv_f1"]) - (
        float(peak_candidate["cv_f1_std"]) / np.sqrt(CV_SPLITS)
    )
    eligible_candidates = [
        row
        for row in candidate_records
        if float(row["cv_f1"]) >= candidate_lower_bound
    ]
    selected_candidate = min(
        eligible_candidates,
        key=lambda row: (
            max(float(row["cv_f1_gap"]), 0.0),
            -float(row["cv_f1"]),
        ),
    )
    best_candidate_name = str(selected_candidate["candidate"])
    best_candidate_metrics = {
        metric: float(selected_candidate[f"cv_{metric}"])
        for metric in SCORING
    }
    selected_candidate["selected"] = True
    ensemble = make_stack_candidate(
        best_candidate_name, X_train, seed, use_feature_selection
    )
    ensemble.fit(X_train, y_train)
    predictions = ensemble.predict(X_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    matrices["Stacking Ensemble"] = matrix
    records.append(
        {
            "test_train_split": test_label,
            "test_fraction": len(y_test) / (len(y_train) + len(y_test)),
            "model": "Stacking Ensemble",
            "best_parameters": "LogisticRegression meta-learner; "
            f"candidate={best_candidate_name}",
            "cv_accuracy": best_candidate_metrics["accuracy"],
            "cv_f1": best_candidate_metrics["f1"],
            "cv_precision": best_candidate_metrics["precision"],
            "cv_recall": best_candidate_metrics["recall"],
            "cv_train_f1": float(selected_candidate["cv_train_f1"]),
            "cv_f1_gap": float(selected_candidate["cv_f1_gap"]),
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
            "stacking_candidate": best_candidate_name,
        }
    )
    return records, matrices, candidate_records


def plot_correlation_heatmap(
    X: pd.DataFrame, y: pd.Series, output_dir: Path
) -> None:
    engineered = engineer_features(X)
    correlation_data = engineered.copy()
    correlation_data[TARGET] = y
    correlations = correlation_data.corr(numeric_only=True)
    figure_size = max(14, len(correlations.columns) * 0.42)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size * 0.82))
    image = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlations.columns)))
    ax.set_yticks(range(len(correlations.columns)))
    ax.set_xticklabels(correlations.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(correlations.columns, fontsize=7)
    ax.set_title("Engineered Feature Correlations (exploratory only)")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_engineering_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def create_feature_selection_diagnostics(
    X: pd.DataFrame, y: pd.Series, output_dir: Path
) -> dict[str, object]:
    engineered = engineer_features(X)
    preprocessor = make_preprocessor(X)
    prepared = preprocessor.fit_transform(engineered)
    scaled = StandardScaler().fit_transform(prepared)
    feature_names = preprocessor.get_feature_names_out()
    cluster_selector, cluster_diagnostics = create_cluster_feature_diagnostics(
        scaled, feature_names, y.to_numpy(), output_dir
    )
    cluster_selected_names = feature_names[cluster_selector.get_support()]
    cluster_selected_data = cluster_selector.transform(scaled)
    selector = LassoFeatureSelector()
    lasso_data = selector.fit_transform(cluster_selected_data, y.to_numpy())
    selected_names = selector.get_feature_names_out(cluster_selected_names)
    selected_coefficients = selector.estimator_.coef_[selector.get_support()]
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    pca_data = pca.fit_transform(lasso_data)

    lasso_frame = pd.DataFrame(
        {
            "feature": selected_names,
            "lasso_coefficient": selected_coefficients,
            "absolute_coefficient": np.abs(selected_coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    lasso_frame.to_csv(output_dir / "lasso_selected_features.csv", index=False)

    component_frame = pd.DataFrame(
        {
            "component": [f"PC{index + 1}" for index in range(pca.n_components_)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca.explained_variance_ratio_
            ),
        }
    )
    component_frame.to_csv(output_dir / "pca_components.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, max(4, len(lasso_frame) * 0.35)))
    ordered = lasso_frame.sort_values("lasso_coefficient")
    colors = np.where(ordered["lasso_coefficient"] >= 0, "#4C78A8", "#E45756")
    ax.barh(ordered["feature"], ordered["lasso_coefficient"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Lasso coefficient (standardized features)")
    ax.set_title(
        f"Lasso-selected features ({len(lasso_frame)} of {len(feature_names)})"
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "lasso_selected_features.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(1, pca.n_components_ + 1)
    ax.bar(
        positions,
        pca.explained_variance_ratio_,
        color="#59A14F",
        label="Explained variance",
    )
    ax.plot(
        positions,
        np.cumsum(pca.explained_variance_ratio_),
        color="#E15759",
        marker="o",
        label="Cumulative variance",
    )
    ax.axhline(PCA_VARIANCE, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(component_frame["component"])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Variance proportion")
    ax.set_title(
        f"PCA-retained components ({pca.n_components_}; "
        f"{np.sum(pca.explained_variance_ratio_):.1%} variance)"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_selected_components.png", dpi=180)
    plt.close(fig)

    return {
        "raw_predictor_count": int(X.shape[1]),
        "engineered_predictor_count": int(engineered.shape[1]),
        "encoded_scaled_feature_count": int(prepared.shape[1]),
        "cluster_selected_count": int(cluster_selected_data.shape[1]),
        "cluster_selected_features": cluster_selected_names.tolist(),
        "lasso_selected_count": int(lasso_data.shape[1]),
        "lasso_alpha": float(selector.estimator_.alpha_),
        "pca_component_count": int(pca_data.shape[1]),
        "pca_cumulative_variance": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "lasso_features": lasso_frame,
        "pca_components": component_frame,
        **cluster_diagnostics,
    }


def create_cluster_feature_diagnostics(
    scaled_features: np.ndarray,
    feature_names: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
) -> tuple[ClusterConsensusFeatureSelector, dict[str, object]]:
    selector = ClusterConsensusFeatureSelector().fit(scaled_features, y)
    labels = selector.method_labels_
    distance = np.sqrt(
        np.maximum(
            0.0,
            2.0 * (1.0 - np.abs(selector.correlation_)),
        )
    )
    np.fill_diagonal(distance, 0.0)
    hierarchy = linkage(
        squareform(distance, checks=False),
        method="average",
    )
    distance_threshold = selector.distance_threshold
    feature_labels = [str(name) for name in feature_names]

    fig, ax = plt.subplots(figsize=(max(14, len(feature_labels) * 0.35), 8))
    dendrogram(
        hierarchy,
        labels=feature_labels,
        color_threshold=distance_threshold,
        above_threshold_color="#555555",
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
    )
    ax.axhline(
        distance_threshold,
        color="#E15759",
        linestyle="--",
        linewidth=2.2,
        label=(
            f"Feature-cluster cut = {distance_threshold:.2f} "
            f"(|correlation| >= {FEATURE_CLUSTER_MIN_CORRELATION:.2f})"
        ),
    )
    ax.set_title("Agglomerative feature clustering — average-linkage correlation distance")
    ax.set_ylabel("Distance: sqrt(2 × (1 − |correlation|))")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "agglomerative_feature_dendrogram.png", dpi=180)
    plt.close(fig)

    profile_projection = PCA(n_components=2).fit_transform(
        np.abs(selector.correlation_)
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    divisive_labels = labels["divisive"]
    for cluster_id in sorted(np.unique(divisive_labels)):
        mask = divisive_labels == cluster_id
        ax.scatter(
            profile_projection[mask, 0],
            profile_projection[mask, 1],
            s=78,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.6,
            label=f"Cluster {cluster_id} (n={int(mask.sum())})",
        )
    ax.set_title(
        "Divisive feature clustering — Bisecting K-Means\n"
        f"{selector.divisive_cluster_count_} groups chosen by silhouette score"
    )
    ax.set_xlabel("Correlation-profile PC1 (visualization only)")
    ax.set_ylabel("Correlation-profile PC2 (visualization only)")
    ax.legend(
        title="Encoded predictors",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "divisive_feature_clusters.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    dbscan_labels = labels["dbscan"]
    for cluster_id in sorted(np.unique(dbscan_labels)):
        mask = dbscan_labels == cluster_id
        if cluster_id == -1:
            color = "#333333"
            label = f"Noise / ungrouped (n={int(mask.sum())})"
        else:
            color = plt.get_cmap("tab10")(int(cluster_id) % 10)
            label = f"Cluster {cluster_id} (n={int(mask.sum())})"
        ax.scatter(
            profile_projection[mask, 0],
            profile_projection[mask, 1],
            s=78,
            alpha=0.82,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            label=label,
        )
    dbscan_cluster_count = len(set(dbscan_labels) - {-1})
    dbscan_noise_count = int(np.sum(dbscan_labels == -1))
    ax.set_title(
        "DBSCAN feature clustering\n"
        f"eps={distance_threshold:.2f}, min_samples="
        f"{selector.dbscan_min_samples}; {dbscan_cluster_count} clusters"
    )
    ax.set_xlabel("Correlation-profile PC1 (visualization only)")
    ax.set_ylabel("Correlation-profile PC2 (visualization only)")
    ax.legend(
        title="Encoded predictors",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=8,
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "dbscan_feature_clusters.png", dpi=180)
    plt.close(fig)

    assignments = pd.DataFrame(
        {
            "feature": feature_labels,
            "absolute_target_association": selector.target_association_,
            "agglomerative_cluster": labels["agglomerative"],
            "divisive_cluster": labels["divisive"],
            "dbscan_cluster": labels["dbscan"],
            "consensus_selected": selector.get_support(),
        }
    )
    assignments.to_csv(output_dir / "cluster_feature_selection.csv", index=False)
    selected_count = int(selector.get_support().sum())
    diagnostics: dict[str, object] = {
        "cluster_feature_count": len(feature_labels),
        "cluster_selected_count": selected_count,
        "cluster_selected_features": [
            feature_labels[index]
            for index in np.flatnonzero(selector.get_support())
        ],
        "agglomerative_cluster_count": len(
            np.unique(labels["agglomerative"])
        ),
        "divisive_cluster_count": int(selector.divisive_cluster_count_),
        "divisive_silhouette_score": float(selector.divisive_silhouette_),
        "dbscan_cluster_count": dbscan_cluster_count,
        "dbscan_noise_count": dbscan_noise_count,
        "consensus_redundant_cluster_count": len(selector.selected_clusters_),
        "feature_cluster_distance_threshold": distance_threshold,
        "feature_cluster_min_correlation": FEATURE_CLUSTER_MIN_CORRELATION,
        "cluster_assignments": assignments,
    }
    return selector, diagnostics


def plot_accuracy_comparison(
    records: list[dict[str, object]], test_label: str, output_dir: Path
) -> None:
    split_records = sorted(
        (row for row in records if row["test_train_split"] == test_label),
        key=lambda row: float(row["accuracy"]),
    )
    names = [str(row["model"]) for row in split_records]
    accuracies = [float(row["accuracy"]) for row in split_records]
    colors = [MODEL_COLORS[name] for name in names]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(names, accuracies, color=colors, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test accuracy")
    ax.set_title(f"Accuracy by model — test:train {test_label}\n(sorted worst to best)")
    ax.grid(axis="x", alpha=0.25)
    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            min(accuracy + 0.012, 0.97),
            bar.get_y() + bar.get_height() / 2,
            f"{accuracy:.3f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    filename = test_label.replace(":", "_")
    fig.savefig(output_dir / f"accuracy_test_{filename}.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrices(
    matrices: dict[str, np.ndarray], test_label: str, output_dir: Path
) -> None:
    color_maps = {
        "Logistic Regression": "Reds",
        "Decision Tree": "Blues",
        "Random Forest": "Greens",
        "XGBoost": "Oranges",
        "Stacking Ensemble": "Purples",
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, model_name in zip(axes.flat, MODEL_ORDER):
        matrix = matrices[model_name]
        row_totals = matrix.sum(axis=1, keepdims=True)
        row_rates = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        ax.imshow(matrix, cmap=color_maps[model_name])
        ax.set_title(model_name)
        ax.set_xticks([0, 1], labels=["0", "1"])
        ax.set_yticks([0, 1], labels=["0", "1"])
        ax.set_xlabel("Predicted diagnosis")
        ax.set_ylabel("Actual diagnosis")
        for row in range(2):
            for column in range(2):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]}\n({row_rates[row, column]:.1%})",
                    ha="center",
                    va="center",
                    color="#111111",
                    fontsize=10,
                    fontweight="bold",
                )
    for ax in axes.flat[len(MODEL_ORDER) :]:
        ax.axis("off")
    fig.suptitle(
        f"Confusion matrices — test:train {test_label}\n"
        "Cells show count and row-normalized percentage",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    filename = test_label.replace(":", "_")
    fig.savefig(output_dir / f"confusion_matrices_test_{filename}.png", dpi=180)
    plt.close(fig)


def write_report(
    report_path: Path,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    diagnostics: dict[str, object] | None,
    records: list[dict[str, object]],
    candidate_records: list[dict[str, object]],
    use_feature_selection: bool = True,
) -> None:
    lines = [
        "ALZHEIMER'S DIAGNOSIS MODEL EVALUATION",
        "=" * 72,
        "The first number in each requested ratio is the test percentage; "
        "the second is the training percentage.",
        "Every ratio uses a stratified holdout split and 3-fold stratified CV "
        "on its training partition.",
        (
            "Feature engineering, imputation, one-hot encoding, scaling, "
            "consensus clustering-based feature selection, Lasso selection, "
            "and PCA are fitted inside each training fold."
            if use_feature_selection
            else "Feature engineering, imputation, one-hot encoding, and "
            "scaling are fitted inside each training fold. No Lasso feature "
            "selection, PCA, or cluster-based feature selection is applied."
        ),
        (
            "The full-data feature-selection preview below is descriptive "
            "only; model selection is refitted inside each training fold."
            if use_feature_selection
            else "Full-data clustering diagnostics are descriptive only; "
            "all encoded predictors are used by models, and preprocessing is "
            "fitted separately inside each CV fold and training split."
        ),
        "",
        "DATA AND FEATURE PROCESSING",
        "-" * 72,
        f"Rows: {len(df)}",
        f"Diagnosis counts: {y.value_counts().sort_index().to_dict()}",
        f"Raw predictors after dropping IDs/placeholders: {X.shape[1]}",
        "Feature engineering: pulse pressure, non-HDL cholesterol, "
        "LDL/HDL ratio, triglyceride/HDL ratio.",
        "Gender, Ethnicity, and EducationLevel are one-hot encoded; other "
        "numeric inputs are median-imputed and all encoded features scaled.",
    ]
    if diagnostics is None:
        raise ValueError("Feature-processing diagnostics are required.")
    lines.append(
        f"Encoded/scaled features used by each model: "
        f"{diagnostics['encoded_scaled_feature_count']}"
    )
    lines.extend(
        [
            "",
            "CLUSTER-BASED FEATURE ANALYSIS",
            "-" * 72,
            f"Agglomerative feature clusters: "
            f"{diagnostics['agglomerative_cluster_count']} "
            "(average linkage on absolute-correlation distance).",
            f"Divisive Bisecting K-Means feature clusters: "
            f"{diagnostics['divisive_cluster_count']} "
            f"(silhouette={diagnostics['divisive_silhouette_score']:.3f}; "
            "cluster count chosen by silhouette score).",
            f"DBSCAN feature clusters: {diagnostics['dbscan_cluster_count']}; "
            f"ungrouped/noise features: {diagnostics['dbscan_noise_count']}.",
            f"Consensus redundant groups: "
            f"{diagnostics['consensus_redundant_cluster_count']}; "
            f"candidate representatives retained: "
            f"{diagnostics['cluster_selected_count']} of "
            f"{diagnostics['cluster_feature_count']} encoded features.",
            "A redundant pair is collapsed only when at least two clustering "
            "methods agree and absolute feature correlation is at least "
            f"{diagnostics['feature_cluster_min_correlation']:.3f}; the "
            "representative is chosen by absolute Diagnosis association.",
            "Candidate representatives from the full-data descriptive pass:",
        ]
    )
    for feature in diagnostics["cluster_selected_features"]:
        lines.append(f"  {feature}")
    if use_feature_selection:
        lines.extend(
            [
                "The consensus screening is applied inside each training fold "
                "before Lasso and PCA; the descriptive list above is not used "
                "as a precomputed input to holdout or CV scoring.",
                f"Lasso-selected features after consensus screening: "
                f"{diagnostics['lasso_selected_count']} "
                f"(full-data descriptive alpha={diagnostics['lasso_alpha']:.4g}; "
                f"fold-local alpha candidates={LASSO_ALPHA_CANDIDATES}; "
                f"absolute coefficient threshold>{LASSO_THRESHOLD}).",
                f"PCA components retained: {diagnostics['pca_component_count']} "
                f"(at least {PCA_VARIANCE:.0%} variance; "
                f"observed {diagnostics['pca_cumulative_variance']:.1%}).",
                "Lasso-selected columns and coefficients:",
            ]
        )
        lasso_frame = diagnostics["lasso_features"]
        for row in lasso_frame.itertuples(index=False):
            lines.append(
                f"  {row.feature}: coefficient={row.lasso_coefficient:.6f}"
            )
        lines.append("PCA retained component variance:")
        component_frame = diagnostics["pca_components"]
        for row in component_frame.itertuples(index=False):
            lines.append(
                f"  {row.component}: {row.explained_variance_ratio:.2%} "
                f"(cumulative {row.cumulative_explained_variance:.2%})"
            )
    else:
        lines.append(
            "Clustering-based feature selection and Lasso: not applied; all "
            "engineered, encoded predictors are used."
        )
        lines.append("PCA: not applied.")
    lines.extend(
        [
            "",
            "ENGINEERING RATIONALE",
            "-" * 72,
            "The added blood-pressure and lipid measures are clinically "
            "interpretable combinations. The heatmap is exploratory; the "
            "cluster selectors use absolute feature correlation inside the "
            "training folds as described above.",
            "",
            "STACKING DESIGN",
            "-" * 72,
            "Each candidate stacks three base learners; "
            "a logistic-regression meta-learner combines out-of-fold "
            "outputs (probabilities where available and SVM decision scores "
            "otherwise). Candidate combinations are compared by 3-fold "
            "validation F1 on each training partition, using the same "
            "one-standard-error/gap rule as the individual model searches; "
            "only the selected candidate is scored on that holdout.",
            "For both model hyperparameters and stack candidates, choose the "
            "smallest nonnegative train/CV F1 gap among choices within one "
            "standard error of the peak CV F1; ties prefer higher CV F1.",
            "The candidates introduce different decision patterns: a linear "
            "logistic baseline, RBF-SVM margins, shallow-tree interactions, "
            "MLP nonlinearities, and (in one candidate) bagged random trees. "
            "This is a practical diversity rationale; the reported CV scores "
            "show which combination is actually preferred for each split.",
            "",
        ]
    )

    for _, test_label in TEST_TRAIN_SPLITS:
        lines.append(f"TEST:TRAIN {test_label}")
        lines.append("-" * 72)
        for model_name in MODEL_ORDER:
            row = next(
                item
                for item in records
                if item["test_train_split"] == test_label
                and item["model"] == model_name
            )
            lines.extend(
                [
                    f"{model_name}"
                    + (
                        f" — stacking candidate: {row['stacking_candidate']}"
                        if model_name == "Stacking Ensemble"
                        else ""
                    ),
                    "  CV (3-fold) "
                    f"accuracy={row['cv_accuracy']:.4f}, "
                    f"F1={row['cv_f1']:.4f}, "
                    f"precision={row['cv_precision']:.4f}, "
                    f"recall={row['cv_recall']:.4f}, "
                    f"train-F1={row['cv_train_f1']:.4f}, "
                    f"train/CV F1 gap={row['cv_f1_gap']:.4f}",
                    "  Holdout "
                    f"accuracy={row['accuracy']:.4f}, "
                    f"F1={row['f1']:.4f}, "
                    f"precision={row['precision']:.4f}, "
                    f"recall={row['recall']:.4f}",
                    "  Confusion matrix (rows=actual [0,1], "
                    "columns=predicted [0,1]): "
                    f"[[{row['tn']}, {row['fp']}], "
                    f"[{row['fn']}, {row['tp']}]]",
                    f"  Best settings: {row['best_parameters']}",
                ]
            )
        lines.append("Stacking candidate CV comparison (one-SE/gap rule):")
        for candidate in (
            row
            for row in candidate_records
            if row["test_train_split"] == test_label
        ):
            selected = " [SELECTED]" if candidate["selected"] else ""
            lines.append(
                f"  {candidate['candidate']}: "
                f"F1={candidate['cv_f1']:.4f}, "
                f"F1 SD={candidate['cv_f1_std']:.4f}, "
                f"accuracy={candidate['cv_accuracy']:.4f}, "
                f"precision={candidate['cv_precision']:.4f}, "
                f"recall={candidate['cv_recall']:.4f}, "
                f"train/CV F1 gap={candidate['cv_f1_gap']:.4f}{selected}"
            )
        lines.append("")

    lines.extend(
        [
            "INTERPRETATION NOTES",
            "-" * 72,
            "The CV train-F1 minus validation-F1 gap is included as an overfit "
            "warning signal; it is not a guarantee of generalization. A 95% "
            "test split leaves very little training data, so its estimates "
            "are expected to be less stable than those from larger training "
            "fractions. The repeated holdout ratios are comparisons, not "
            "independent external validation datasets.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Alzheimer's diagnosis models."
    )
    parser.add_argument(
        "--without-feature-selection",
        "--without-pca-lasso",
        dest="without_feature_selection",
        action="store_true",
        help=(
            "Evaluate all models on every engineered, encoded, and scaled "
            "predictor without cluster, Lasso, or PCA feature selection."
        ),
    )
    args = parser.parse_args()
    use_feature_selection = not args.without_feature_selection

    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / "alzheimers_disease_data.csv"
    output_dir = (
        script_dir / "model_results"
        if use_feature_selection
        else script_dir / "model_results_without_pca_lasso"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if TARGET not in df.columns:
        raise ValueError(f"Required target column {TARGET!r} was not found.")
    if df[TARGET].isna().any():
        raise ValueError(f"Target column {TARGET!r} contains missing values.")
    classes = set(df[TARGET].unique())
    if classes != {0, 1}:
        raise ValueError(f"Expected binary Diagnosis labels {{0, 1}}, got {classes}.")

    X = df.drop(
        columns=[TARGET, *[column for column in IDENTIFIER_COLUMNS if column in df]],
    )
    y = df[TARGET].astype(int)
    if not set(CATEGORICAL_COLUMNS).issubset(X.columns):
        missing_categories = set(CATEGORICAL_COLUMNS) - set(X.columns)
        raise ValueError(f"Missing expected categorical columns: {missing_categories}")

    plot_correlation_heatmap(X, y, output_dir)
    if use_feature_selection:
        diagnostics = create_feature_selection_diagnostics(X, y, output_dir)
    else:
        engineered = engineer_features(X)
        preprocessor = make_preprocessor(X)
        prepared = preprocessor.fit_transform(engineered)
        scaled = StandardScaler().fit_transform(prepared)
        _, cluster_diagnostics = create_cluster_feature_diagnostics(
            scaled,
            preprocessor.get_feature_names_out(),
            y.to_numpy(),
            output_dir,
        )
        diagnostics = {
            "encoded_scaled_feature_count": int(prepared.shape[1]),
            **cluster_diagnostics,
        }
    all_records: list[dict[str, object]] = []
    all_candidate_records: list[dict[str, object]] = []

    for test_fraction, test_label in TEST_TRAIN_SPLITS:
        print(f"Evaluating test:train {test_label}...", flush=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_fraction,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        records, matrices, candidate_records = fit_and_score_models(
            X_train,
            X_test,
            y_train,
            y_test,
            test_label,
            RANDOM_STATE,
            use_feature_selection,
        )
        all_records.extend(records)
        all_candidate_records.extend(candidate_records)
        plot_accuracy_comparison(all_records, test_label, output_dir)
        plot_confusion_matrices(matrices, test_label, output_dir)

    model_metrics = pd.DataFrame(all_records)
    model_metrics["model_order"] = model_metrics["model"].map(
        {name: index for index, name in enumerate(MODEL_ORDER)}
    )
    model_metrics["split_order"] = model_metrics["test_train_split"].map(
        {label: index for index, (_, label) in enumerate(TEST_TRAIN_SPLITS)}
    )
    model_metrics = model_metrics.sort_values(
        ["split_order", "model_order"]
    ).drop(columns=["model_order", "split_order"])
    model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(all_candidate_records).to_csv(
        output_dir / "stacking_candidate_cv.csv", index=False
    )
    write_report(
        output_dir / "model_report.txt",
        df,
        X,
        y,
        diagnostics,
        all_records,
        all_candidate_records,
        use_feature_selection,
    )
    print(f"Reports and plots saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
