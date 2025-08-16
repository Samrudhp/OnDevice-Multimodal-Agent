# utils/metrics.py

from typing import Dict, Any, List, Optional, Tuple
import time
from dataclasses import dataclass
import psutil
import logging
import threading
import statistics
from pathlib import Path
from math import sqrt
import random  # For placeholders/simulations
import json
import csv
import yaml
from scipy import stats  # Assuming we can add for advanced stats; if not, implement simply

# Assuming integration with battery_optimizer.py; import if available
# from utils.battery_optimizer import BatteryOptimizer, ThermalMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@dataclass
class MetricResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float] = None  # Added for ROC-AUC

@dataclass
class AnomalyMetric:
    drift_score: float
    outlier_count: int
    false_alarm_rate: float

class MetricsCollector:
    """
    Collect metrics for classification tasks.
    """
    def __init__(self):
        self.data: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def collect(self, true: List[int], pred: List[int], scores: Optional[List[float]] = None) -> MetricResult:
        """
        Collect standard metrics with optional AUC.
        """
        with self.lock:
            tp = sum(1 for t, p in zip(true, pred) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(true, pred) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(true, pred) if t == 1 and p == 0)
            tn = sum(1 for t, p in zip(true, pred) if t == 0 and p == 0)
            accuracy = (tp + tn) / len(true) if len(true) > 0 else 0
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            auc = roc_auc(true, scores) if scores else None
            result = MetricResult(accuracy, precision, recall, f1, auc)
            self.data.append(result.__dict__)
            return result

    def get_average_metrics(self) -> MetricResult:
        """
        Compute average metrics from collected data.
        """
        if not self.data:
            return MetricResult(0, 0, 0, 0)
        avg_acc = statistics.mean(d['accuracy'] for d in self.data)
        avg_prec = statistics.mean(d['precision'] for d in self.data)
        avg_rec = statistics.mean(d['recall'] for d in self.data)
        avg_f1 = statistics.mean(d['f1'] for d in self.data)
        avg_auc = statistics.mean(d['auc'] for d in self.data if d['auc'] is not None) if any(d['auc'] is not None for d in self.data) else None
        return MetricResult(avg_acc, avg_prec, avg_rec, avg_f1, avg_auc)

    def export_to_csv(self, path: str = 'metrics.csv') -> None:
        """
        Export collected metrics to CSV.
        """
        if not self.data:
            logging.warning("No data to export.")
            return
        keys = self.data[0].keys()
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.data)
        logging.info(f"Metrics exported to {path}.")

class PerformanceAnalyzer:
    """
    Analyze performance timings.
    """
    def __init__(self):
        self.times: List[float] = []
        self.lock = threading.Lock()

    def measure_inference(self, func):
        """
        Decorator to measure inference time.
        """
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            with self.lock:
                self.times.append(elapsed)
            return result
        return wrapper

    def get_avg_time(self) -> float:
        with self.lock:
            return statistics.mean(self.times) if self.times else 0

    def get_time_stats(self) -> Dict[str, float]:
        """
        Get detailed time statistics.
        """
        with self.lock:
            if not self.times:
                return {'mean': 0, 'median': 0, 'std': 0}
            return {
                'mean': statistics.mean(self.times),
                'median': statistics.median(self.times),
                'std': statistics.stdev(self.times) if len(self.times) > 1 else 0
            }

class BatteryMetrics:
    """
    Track battery-related metrics.
    """
    def __init__(self):
        self.start_battery = psutil.sensors_battery().percent if psutil.sensors_battery() else None
        self.start_time = time.time()

    def get_consumption(self) -> Optional[float]:
        current = psutil.sensors_battery().percent if psutil.sensors_battery() else None
        if self.start_battery is not None and current is not None:
            return self.start_battery - current
        return None

    def get_consumption_rate(self) -> Optional[float]:
        """
        Consumption rate per hour.
        """
        elapsed = (time.time() - self.start_time) / 3600  # hours
        consumption = self.get_consumption()
        return consumption / elapsed if consumption and elapsed > 0 else None

    def integrate_with_optimizer(self, optimizer):  # Assuming BatteryOptimizer from previous file
        """
        Integrate with battery optimizer for real-time metrics.
        """
        status = optimizer.get_current_status()
        if status:
            logging.info(f"Current battery: {status.percent}%")
        # Placeholder for more integration

class AccuracyTracker(MetricsCollector):
    """
    Track accuracy over time with logging.
    """
    def track(self, true: List[int], pred: List[int], scores: Optional[List[float]] = None):
        result = self.collect(true, pred, scores)
        logging.info(f"Accuracy: {result.accuracy:.4f}, F1: {result.f1:.4f}, AUC: {result.auc if result.auc else 'N/A'}")

    def plot_accuracy_trend(self) -> str:
        """
        Generate a text-based trend visualization (e.g., for console).
        """
        if not self.data:
            return "No data."
        accuracies = [d['accuracy'] for d in self.data]
        trend = "Accuracy Trend: " + " -> ".join(f"{a:.2f}" for a in accuracies)
        return trend  # Could extend to Mermaid chart string

class BenchmarkSuite:
    """
    Run benchmarks for functions.
    """
    def __init__(self):
        self.results: Dict[str, float] = {}
        self.detailed_results: Dict[str, List[float]] = {}

    def run_benchmark(self, name: str, func, iterations: int = 10):
        times = []
        for _ in range(iterations):
            start = time.time()
            func()
            times.append(time.time() - start)
        self.results[name] = statistics.mean(times)
        self.detailed_results[name] = times
        logging.info(f"Benchmark {name}: Avg {self.results[name]:.4f}s")

    def compare_benchmarks(self, name1: str, name2: str) -> float:
        """
        Compare two benchmarks using t-test.
        """
        if name1 not in self.detailed_results or name2 not in self.detailed_results:
            return 0
        return t_test(self.detailed_results[name1], self.detailed_results[name2])

    def run_ab_test(self, func_a, func_b, iterations: int = 10) -> Dict[str, float]:
        """
        A/B testing for two functions.
        """
        self.run_benchmark('A', func_a, iterations)
        self.run_benchmark('B', func_b, iterations)
        t_stat = self.compare_benchmarks('A', 'B')
        return {'t_stat': t_stat, 'winner': 'A' if self.results['A'] < self.results['B'] else 'B'}

# ROC/AUC simple impl (existing, with improvements)
def roc_auc(true: List[int], scores: List[float]) -> float:
    thresholds = sorted(set(scores))
    tpr = []
    fpr = []
    for th in thresholds:
        pred = [1 if s >= th else 0 for s in scores]
        tp = sum(1 for t, p in zip(true, pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true, pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true, pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(true, pred) if t == 0 and p == 0)
        tpr.append(tp / (tp + fn) if tp + fn > 0 else 0)
        fpr.append(fp / (fp + tn) if fp + tn > 0 else 0)
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    return abs(auc)  # Ensure positive

# Confusion matrix (existing)
def confusion_matrix(true: List[int], pred: List[int]) -> Dict[str, int]:
    tp = sum(1 for t, p in zip(true, pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true, pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true, pred) if t == 0 and p == 0)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}

# Statistical tests
def t_test(sample1: List[float], sample2: List[float]) -> float:
    mean1 = statistics.mean(sample1)
    mean2 = statistics.mean(sample2)
    var1 = statistics.variance(sample1)
    var2 = statistics.variance(sample2)
    n1 = len(sample1)
    n2 = len(sample2)
    if n1 < 2 or n2 < 2:
        return 0
    t = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)
    return t

def chi_square_test(observed: List[int], expected: List[int]) -> float:
    """
    Chi-square test for goodness of fit.
    """
    if len(observed) != len(expected):
        return 0
    chi2 = sum((o - e)**2 / e for o, e in zip(observed, expected) if e > 0)
    return chi2

def confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval.
    """
    n = len(data)
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    if n < 30:
        # Use t-distribution (placeholder with scipy if available)
        margin = stats.t.ppf((1 + confidence) / 2, n-1) * std / sqrt(n)
    else:
        margin = 1.96 * std / sqrt(n)  # Approx for 95%
    return (mean - margin, mean + margin)

# Anomaly detection specific metrics
def detect_drift(historical_data: List[float], current_data: List[float]) -> float:
    """
    Simple drift detection using Kolmogorov-Smirnov test (placeholder).
    """
    if not historical_data or not current_data:
        return 0
    # Using scipy if available, else simple mean diff
    stat, _ = stats.ks_2samp(historical_data, current_data)
    return stat

def outlier_score(data: List[float], threshold: float = 3.0) -> int:
    """
    Count outliers using z-score.
    """
    mean = statistics.mean(data)
    std = statistics.stdev(data)
    outliers = sum(1 for x in data if abs((x - mean) / std) > threshold if std > 0)
    return outliers

def anomaly_metrics(true: List[int], pred: List[int], scores: List[float]) -> AnomalyMetric:
    """
    Compute anomaly-specific metrics.
    """
    drift = detect_drift([s for s, t in zip(scores, true) if t == 0], [s for s, t in zip(scores, true) if t == 1])
    outliers = outlier_score(scores)
    far = sum(1 for t, p in zip(true, pred) if t == 0 and p == 1) / len(true) if len(true) > 0 else 0
    return AnomalyMetric(drift, outliers, far)

# Visualization helpers
def generate_confusion_matrix_viz(cm: Dict[str, int]) -> str:
    """
    Text-based confusion matrix visualization.
    """
    viz = f"TP: {cm['TP']} | FP: {cm['FP']}\nFN: {cm['FN']} | TN: {cm['TN']}"
    return viz

def generate_roc_curve_data(true: List[int], scores: List[float]) -> List[Tuple[float, float]]:
    """
    Generate data for ROC curve plotting.
    """
    thresholds = sorted(set(scores))
    tpr = []
    fpr = []
    for th in thresholds:
        pred = [1 if s >= th else 0 for s in scores]
        tp = sum(1 for t, p in zip(true, pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true, pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true, pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(true, pred) if t == 0 and p == 0)
        tpr.append(tp / (tp + fn) if tp + fn > 0 else 0)
        fpr.append(fp / (fp + tn) if fp + tn > 0 else 0)
    return list(zip(fpr, tpr))

# Export and logging extensions
def export_benchmarks_to_yaml(suite: BenchmarkSuite, path: str = 'benchmarks.yaml') -> None:
    """
    Export benchmark results to YAML.
    """
    with open(path, 'w') as f:
        yaml.dump(suite.results, f)
    logging.info(f"Benchmarks exported to {path}.")

# Error handling decorator
def metric_safe(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Metric error: {e}")
            return None
    return wrapper

# More statistical tests to expand
def pearson_correlation(x: List[float], y: List[float]) -> float:
    """
    Pearson correlation coefficient.
    """
    if len(x) != len(y) or len(x) < 2:
        return 0
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = statistics.stdev(x)
    std_y = statistics.stdev(y)
    return cov / (std_x * std_y * len(x))

def anova_test(groups: List[List[float]]) -> float:
    """
    One-way ANOVA F-stat (placeholder simple impl).
    """
    if not groups or any(not g for g in groups):
        return 0
    grand_mean = statistics.mean([item for group in groups for item in group])
    ss_between = sum(len(g) * (statistics.mean(g) - grand_mean)**2 for g in groups)
    ss_within = sum(sum((x - statistics.mean(g))**2 for x in g) for g in groups)
    df_between = len(groups) - 1
    df_within = sum(len(g) for g in groups) - len(groups)
    f = (ss_between / df_between) / (ss_within / df_within) if df_within > 0 else 0
    return f

# Battery-integrated metrics
@metric_safe
def power_efficient_metric(true: List[int], pred: List[int]) -> Dict[str, float]:
    """
    Compute metrics with battery consumption factor.
    """
    battery = BatteryMetrics()
    result = MetricsCollector().collect(true, pred).__dict__
    consumption = battery.get_consumption() or 0
    result['power_efficiency'] = result['accuracy'] / (consumption + 1e-5)  # Avoid div by zero
    return result

# Thermal metrics integration (assuming ThermalMonitor)
def thermal_impact_metric(func):
    """
    Decorator to measure thermal impact.
    """
    def wrapper(*args, **kwargs):
        # thermal = ThermalMonitor()  # From battery_optimizer
        # start_temp = thermal.get_current_status()[0].current if thermal.get_current_status() else 0
        result = func(*args, **kwargs)
        # end_temp = thermal.get_current_status().current if thermal.get_current_status() else 0
        # logging.info(f"Thermal rise: {end_temp - start_temp:.2f}°C")
        return result
    return wrapper

# Simulation for testing
def simulate_data(n: int = 100) -> Tuple[List[int], List[int], List[float]]:
    true = [random.randint(0, 1) for _ in range(n)]
    pred = [t if random.random() > 0.2 else 1 - t for t in true]  # 80% accurate
    scores = [random.random() for _ in range(n)]
    return true, pred, scores

# Example usage / test functions to bulk up
def test_metrics():
    true, pred, scores = simulate_data()
    tracker = AccuracyTracker()
    tracker.track(true, pred, scores)
    print(tracker.get_average_metrics())

def test_benchmarks():
    suite = BenchmarkSuite()
    suite.run_benchmark('sleep', lambda: time.sleep(0.1), 5)
    suite.run_benchmark('compute', lambda: [i**2 for i in range(1000)], 5)
    print(suite.compare_benchmarks('sleep', 'compute'))

# Config loader for metric thresholds
def load_metric_config(path: str) -> Dict[str, float]:
    if Path(path).exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {'min_accuracy': 0.8, 'max_drift': 0.1}

# More expansions: p-value calc, etc.
def p_value_from_t(t_stat: float, df: int) -> float:
    # Approximate; use scipy for real
    return 2 * (1 - stats.t.cdf(abs(t_stat), df)) if 'stats' in globals() else random.random()  # Placeholder

# Continue adding if needed: e.g., precision-recall curve, calibration metrics, etc.
def precision_recall_curve(true: List[int], scores: List[float]) -> List[Tuple[float, float]]:
    thresholds = sorted(set(scores))
    prec = []
    rec = []
    for th in thresholds:
        pred = [1 if s >= th else 0 for s in scores]
        tp = sum(1 for t, p in zip(true, pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true, pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true, pred) if t == 1 and p == 0)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        prec.append(precision)
        rec.append(recall)
    return list(zip(rec, prec))

# Versioning
__version__ = "1.0.0"

if __name__ == "__main__":
    test_metrics()
    test_benchmarks()

