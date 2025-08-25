"""
RAG Quality Metrics and Feedback Loop System.

This module implements comprehensive metrics for RAG system quality assessment
and a feedback loop for continuous improvement.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import statistics
from pathlib import Path
from enum import Enum
import threading
import time

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.metrics import precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MetricType(Enum):
    """Types of RAG metrics."""
    # Retrieval metrics
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MRR = "mean_reciprocal_rank"  # Mean Reciprocal Rank
    MAP = "mean_average_precision"  # Mean Average Precision
    NDCG = "normalized_discounted_cumulative_gain"
    
    # Quality metrics
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    COVERAGE = "coverage"
    DIVERSITY = "diversity"
    NOVELTY = "novelty"
    
    # Performance metrics
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CACHE_HIT_RATE = "cache_hit_rate"
    
    # User feedback metrics
    USER_SATISFACTION = "user_satisfaction"
    CLICK_THROUGH_RATE = "click_through_rate"
    DWELL_TIME = "dwell_time"


@dataclass
class QueryFeedback:
    """Feedback for a single query."""
    query_id: str
    query: str
    retrieved_docs: List[Dict[str, Any]]
    selected_docs: List[int]  # Indices of actually useful docs
    relevance_scores: List[float]  # User-provided or automated scores
    response_quality: float  # Overall quality score (0-1)
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    """Configuration for RAG metrics system."""
    # Metric calculation
    enable_all_metrics: bool = True
    metrics_to_track: List[MetricType] = field(default_factory=lambda: list(MetricType))
    
    # Feedback collection
    collect_implicit_feedback: bool = True
    collect_explicit_feedback: bool = False
    feedback_sample_rate: float = 1.0  # Percentage of queries to sample
    
    # Thresholds
    relevance_threshold: float = 0.7
    quality_threshold: float = 0.6
    latency_threshold_ms: float = 1000.0
    
    # Window sizes for moving averages
    metric_window_size: int = 1000
    trend_window_size: int = 100
    
    # Feedback loop
    enable_feedback_loop: bool = True
    feedback_update_interval: int = 100  # Update model every N feedbacks
    min_feedback_for_update: int = 50
    
    # Persistence
    persist_metrics: bool = True
    metrics_path: str = ".rag/metrics"
    checkpoint_interval: int = 1000


class RAGMetricsCollector:
    """
    Collects and analyzes RAG system metrics.
    
    Features:
    - Comprehensive metric calculation
    - Real-time performance monitoring
    - Trend detection and alerting
    - Feedback incorporation
    - A/B testing support
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics: Dict[MetricType, deque] = {
            metric: deque(maxlen=self.config.metric_window_size)
            for metric in self.config.metrics_to_track
        }
        
        # Feedback storage
        self.feedback_history: deque = deque(maxlen=self.config.metric_window_size)
        self.pending_feedback: List[QueryFeedback] = []
        
        # Aggregated metrics
        self.aggregated_metrics: Dict[str, float] = {}
        self.metric_trends: Dict[MetricType, float] = {}
        
        # Performance tracking
        self.query_times: deque = deque(maxlen=1000)
        self.retrieval_times: deque = deque(maxlen=1000)
        
        # A/B testing
        self.ab_tests: Dict[str, Dict] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._last_update = datetime.now()
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "total_feedback": 0,
            "model_updates": 0,
            "alerts_triggered": 0
        }
        
        # Load existing metrics if available
        if self.config.persist_metrics:
            self._load_metrics()
    
    def record_query(self, query_id: str, query: str, 
                    retrieved_docs: List[Dict[str, Any]],
                    latency_ms: float,
                    metadata: Optional[Dict] = None) -> str:
        """
        Record a RAG query for metrics collection.
        
        Args:
            query_id: Unique query identifier
            query: The query text
            retrieved_docs: Documents retrieved by RAG
            latency_ms: Query latency in milliseconds
            metadata: Optional metadata
        
        Returns:
            Query ID for feedback reference
        """
        with self._lock:
            self.stats["total_queries"] += 1
            
            # Record latency
            self.metrics[MetricType.LATENCY].append(latency_ms)
            self.query_times.append((datetime.now(), latency_ms))
            
            # Check if we should sample this query
            if self.config.feedback_sample_rate < 1.0:
                import random
                if random.random() > self.config.feedback_sample_rate:
                    return query_id
            
            # Create placeholder feedback entry
            feedback = QueryFeedback(
                query_id=query_id,
                query=query,
                retrieved_docs=retrieved_docs,
                selected_docs=[],
                relevance_scores=[],
                response_quality=0.0,
                latency_ms=latency_ms,
                metadata=metadata or {}
            )
            
            self.pending_feedback.append(feedback)
            
            # Calculate immediate metrics if possible
            self._calculate_immediate_metrics(feedback)
        
        return query_id
    
    def record_feedback(self, query_id: str,
                       selected_docs: Optional[List[int]] = None,
                       relevance_scores: Optional[List[float]] = None,
                       response_quality: Optional[float] = None):
        """
        Record user feedback for a query.
        
        Args:
            query_id: Query identifier
            selected_docs: Indices of documents user found useful
            relevance_scores: Relevance scores for retrieved docs
            response_quality: Overall response quality (0-1)
        """
        with self._lock:
            # Find pending feedback entry
            feedback_entry = None
            for feedback in self.pending_feedback:
                if feedback.query_id == query_id:
                    feedback_entry = feedback
                    break
            
            if not feedback_entry:
                self.logger.warning(f"No pending feedback found for query {query_id}")
                return
            
            # Update feedback
            if selected_docs is not None:
                feedback_entry.selected_docs = selected_docs
            if relevance_scores is not None:
                feedback_entry.relevance_scores = relevance_scores
            if response_quality is not None:
                feedback_entry.response_quality = response_quality
            
            # Move to history
            self.pending_feedback.remove(feedback_entry)
            self.feedback_history.append(feedback_entry)
            self.stats["total_feedback"] += 1
            
            # Calculate metrics
            self._calculate_retrieval_metrics(feedback_entry)
            
            # Trigger feedback loop if needed
            if (self.config.enable_feedback_loop and 
                len(self.feedback_history) >= self.config.min_feedback_for_update and
                self.stats["total_feedback"] % self.config.feedback_update_interval == 0):
                self._trigger_feedback_loop()
    
    def _calculate_immediate_metrics(self, feedback: QueryFeedback):
        """Calculate metrics that don't require user feedback."""
        # Throughput
        current_time = datetime.now()
        recent_queries = [
            t for t, _ in self.query_times
            if (current_time - t).total_seconds() < 60
        ]
        throughput = len(recent_queries) / 60.0  # Queries per second
        self.metrics[MetricType.THROUGHPUT].append(throughput)
        
        # Document diversity (based on retrieved docs)
        if feedback.retrieved_docs:
            diversity = self._calculate_diversity(feedback.retrieved_docs)
            self.metrics[MetricType.DIVERSITY].append(diversity)
    
    def _calculate_retrieval_metrics(self, feedback: QueryFeedback):
        """Calculate retrieval performance metrics."""
        if not feedback.selected_docs or not feedback.retrieved_docs:
            return
        
        num_retrieved = len(feedback.retrieved_docs)
        num_relevant = len(feedback.selected_docs)
        
        # Precision: relevant docs / retrieved docs
        precision = num_relevant / num_retrieved if num_retrieved > 0 else 0
        self.metrics[MetricType.PRECISION].append(precision)
        
        # For recall, we'd need to know total relevant docs (not available)
        # Using a proxy: assume selected docs are minimum relevant set
        estimated_recall = min(1.0, num_relevant / max(1, num_relevant))
        self.metrics[MetricType.RECALL].append(estimated_recall)
        
        # F1 Score
        if precision + estimated_recall > 0:
            f1 = 2 * (precision * estimated_recall) / (precision + estimated_recall)
            self.metrics[MetricType.F1_SCORE].append(f1)
        
        # Mean Reciprocal Rank (MRR)
        if feedback.selected_docs:
            first_relevant = min(feedback.selected_docs)
            mrr = 1.0 / (first_relevant + 1)
            self.metrics[MetricType.MRR].append(mrr)
        
        # NDCG (Normalized Discounted Cumulative Gain)
        if feedback.relevance_scores:
            ndcg = self._calculate_ndcg(feedback.relevance_scores)
            self.metrics[MetricType.NDCG].append(ndcg)
        
        # Overall relevance
        if feedback.relevance_scores:
            avg_relevance = statistics.mean(feedback.relevance_scores)
            self.metrics[MetricType.RELEVANCE].append(avg_relevance)
    
    def _calculate_diversity(self, documents: List[Dict[str, Any]]) -> float:
        """Calculate diversity score of retrieved documents."""
        if len(documents) <= 1:
            return 0.0
        
        # Simple diversity: unique terms / total terms
        all_terms = set()
        total_terms = 0
        
        for doc in documents:
            text = doc.get("text", "")
            terms = set(text.lower().split())
            all_terms.update(terms)
            total_terms += len(terms)
        
        if total_terms == 0:
            return 0.0
        
        return len(all_terms) / total_terms
    
    def _calculate_ndcg(self, relevance_scores: List[float], k: Optional[int] = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not relevance_scores:
            return 0.0
        
        k = k or len(relevance_scores)
        
        # Calculate DCG
        dcg = 0.0
        for i, score in enumerate(relevance_scores[:k]):
            dcg += (2 ** score - 1) / np.log2(i + 2)
        
        # Calculate ideal DCG
        ideal_scores = sorted(relevance_scores, reverse=True)[:k]
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += (2 ** score - 1) / np.log2(i + 2)
        
        # NDCG
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def _trigger_feedback_loop(self):
        """Trigger feedback loop to update the RAG model."""
        self.logger.info("Triggering feedback loop for model update")
        
        try:
            # Analyze recent feedback
            recent_feedback = list(self.feedback_history)[-self.config.min_feedback_for_update:]
            
            # Calculate improvement areas
            improvements = self._analyze_feedback(recent_feedback)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(improvements)
            
            # Log recommendations (actual model update would happen here)
            self.logger.info(f"Feedback loop recommendations: {recommendations}")
            
            self.stats["model_updates"] += 1
            
        except Exception as e:
            self.logger.error(f"Feedback loop failed: {e}")
    
    def _analyze_feedback(self, feedback_list: List[QueryFeedback]) -> Dict[str, Any]:
        """Analyze feedback to identify improvement areas."""
        analysis = {
            "avg_quality": 0.0,
            "low_quality_queries": [],
            "common_failures": [],
            "performance_issues": []
        }
        
        if not feedback_list:
            return analysis
        
        # Average quality
        quality_scores = [f.response_quality for f in feedback_list if f.response_quality > 0]
        if quality_scores:
            analysis["avg_quality"] = statistics.mean(quality_scores)
        
        # Find low quality queries
        for feedback in feedback_list:
            if feedback.response_quality < self.config.quality_threshold:
                analysis["low_quality_queries"].append({
                    "query": feedback.query,
                    "quality": feedback.response_quality
                })
        
        # Performance issues
        for feedback in feedback_list:
            if feedback.latency_ms > self.config.latency_threshold_ms:
                analysis["performance_issues"].append({
                    "query": feedback.query,
                    "latency": feedback.latency_ms
                })
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on feedback analysis."""
        recommendations = []
        
        # Quality recommendations
        if analysis["avg_quality"] < self.config.quality_threshold:
            recommendations.append(f"Improve retrieval quality (current: {analysis['avg_quality']:.2f})")
        
        # Performance recommendations
        if analysis["performance_issues"]:
            avg_latency = statistics.mean([p["latency"] for p in analysis["performance_issues"]])
            recommendations.append(f"Optimize query performance (avg latency: {avg_latency:.0f}ms)")
        
        # Specific query patterns
        if len(analysis["low_quality_queries"]) > 5:
            recommendations.append("Review and improve handling of failing query patterns")
        
        return recommendations
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            "current_metrics": {},
            "trends": {},
            "alerts": [],
            "stats": self.stats.copy()
        }
        
        with self._lock:
            # Current metric values
            for metric_type, values in self.metrics.items():
                if values:
                    summary["current_metrics"][metric_type.value] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else 0,
                        "min": min(values),
                        "max": max(values),
                        "latest": values[-1]
                    }
            
            # Trends
            for metric_type in self.metrics:
                trend = self._calculate_trend(metric_type)
                if trend:
                    summary["trends"][metric_type.value] = trend
            
            # Check for alerts
            alerts = self._check_alerts()
            summary["alerts"] = alerts
        
        return summary
    
    def _calculate_trend(self, metric_type: MetricType) -> Optional[Dict[str, float]]:
        """Calculate trend for a metric."""
        values = self.metrics[metric_type]
        if len(values) < self.config.trend_window_size:
            return None
        
        # Compare recent window to previous window
        window_size = self.config.trend_window_size // 2
        recent = list(values)[-window_size:]
        previous = list(values)[-2*window_size:-window_size]
        
        if not recent or not previous:
            return None
        
        recent_mean = statistics.mean(recent)
        previous_mean = statistics.mean(previous)
        
        change = (recent_mean - previous_mean) / max(0.001, previous_mean)
        
        return {
            "direction": "up" if change > 0.05 else "down" if change < -0.05 else "stable",
            "change_percent": change * 100,
            "recent_value": recent_mean,
            "previous_value": previous_mean
        }
    
    def _check_alerts(self) -> List[Dict[str, Any]]:
        """Check for metric alerts."""
        alerts = []
        
        # Check latency
        if MetricType.LATENCY in self.metrics and self.metrics[MetricType.LATENCY]:
            recent_latency = list(self.metrics[MetricType.LATENCY])[-10:]
            if recent_latency:
                avg_latency = statistics.mean(recent_latency)
                if avg_latency > self.config.latency_threshold_ms:
                    alerts.append({
                        "type": "performance",
                        "message": f"High latency detected: {avg_latency:.0f}ms",
                        "severity": "warning"
                    })
        
        # Check quality
        if MetricType.RELEVANCE in self.metrics and self.metrics[MetricType.RELEVANCE]:
            recent_relevance = list(self.metrics[MetricType.RELEVANCE])[-10:]
            if recent_relevance:
                avg_relevance = statistics.mean(recent_relevance)
                if avg_relevance < self.config.relevance_threshold:
                    alerts.append({
                        "type": "quality",
                        "message": f"Low relevance detected: {avg_relevance:.2f}",
                        "severity": "warning"
                    })
        
        if alerts:
            self.stats["alerts_triggered"] += len(alerts)
        
        return alerts
    
    def run_ab_test(self, test_name: str, variant: str, 
                    query_id: str, metric_value: float):
        """
        Record A/B test results.
        
        Args:
            test_name: Name of the A/B test
            variant: Variant identifier (A or B)
            query_id: Query identifier
            metric_value: Metric value to track
        """
        with self._lock:
            if test_name not in self.ab_tests:
                self.ab_tests[test_name] = {
                    "variants": {},
                    "start_time": datetime.now()
                }
            
            if variant not in self.ab_tests[test_name]["variants"]:
                self.ab_tests[test_name]["variants"][variant] = {
                    "values": [],
                    "count": 0
                }
            
            self.ab_tests[test_name]["variants"][variant]["values"].append(metric_value)
            self.ab_tests[test_name]["variants"][variant]["count"] += 1
    
    def get_ab_test_results(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results with statistical significance."""
        if test_name not in self.ab_tests:
            return None
        
        test = self.ab_tests[test_name]
        results = {
            "test_name": test_name,
            "duration": (datetime.now() - test["start_time"]).total_seconds(),
            "variants": {}
        }
        
        for variant, data in test["variants"].items():
            if data["values"]:
                results["variants"][variant] = {
                    "count": data["count"],
                    "mean": statistics.mean(data["values"]),
                    "std": statistics.stdev(data["values"]) if len(data["values"]) > 1 else 0,
                    "confidence": self._calculate_confidence(data["values"])
                }
        
        # Calculate statistical significance if two variants
        if len(results["variants"]) == 2:
            variants = list(results["variants"].keys())
            values_a = test["variants"][variants[0]]["values"]
            values_b = test["variants"][variants[1]]["values"]
            
            if values_a and values_b:
                # Simple t-test approximation
                mean_diff = abs(statistics.mean(values_a) - statistics.mean(values_b))
                pooled_std = np.sqrt(
                    (statistics.variance(values_a) + statistics.variance(values_b)) / 2
                ) if len(values_a) > 1 and len(values_b) > 1 else 0
                
                if pooled_std > 0:
                    t_stat = mean_diff / (pooled_std * np.sqrt(2 / min(len(values_a), len(values_b))))
                    # Rough p-value approximation
                    p_value = 2 * (1 - min(0.99, abs(t_stat) / 3))
                    results["statistical_significance"] = {
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
        
        return results
    
    def _calculate_confidence(self, values: List[float]) -> float:
        """Calculate confidence interval for values."""
        if len(values) < 2:
            return 0.0
        
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        confidence = 1.96 * std / np.sqrt(len(values))  # 95% confidence interval
        
        return confidence
    
    def save_metrics(self):
        """Save metrics to disk."""
        if not self.config.persist_metrics:
            return
        
        try:
            metrics_path = Path(self.config.metrics_path)
            metrics_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            metrics_data = {
                "metrics": {k.value: list(v) for k, v in self.metrics.items()},
                "feedback_history": [asdict(f) for f in self.feedback_history],
                "stats": self.stats,
                "ab_tests": self.ab_tests,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(metrics_path / "metrics.json", "w") as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            self.logger.debug("Metrics saved to disk")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def _load_metrics(self) -> bool:
        """Load metrics from disk."""
        metrics_path = Path(self.config.metrics_path)
        if not metrics_path.exists():
            return False
        
        try:
            metrics_file = metrics_path / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_data = json.load(f)
                
                # Restore metrics
                for metric_name, values in metrics_data.get("metrics", {}).items():
                    metric_type = MetricType(metric_name)
                    if metric_type in self.metrics:
                        self.metrics[metric_type].extend(values)
                
                self.stats.update(metrics_data.get("stats", {}))
                self.ab_tests = metrics_data.get("ab_tests", {})
                
                self.logger.info("Metrics loaded from disk")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")
            return False