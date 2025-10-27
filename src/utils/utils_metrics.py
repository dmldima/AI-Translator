"""
Metrics tracking and cost estimation.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class TranslationMetrics:
    """Metrics for translation operations."""
    
    # Counters
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cached_requests: int = 0
    
    # Content metrics
    total_characters: int = 0
    total_words: int = 0
    total_segments: int = 0
    
    # Token usage (OpenAI)
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # Timing
    total_duration_seconds: float = 0.0
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    
    # Cost tracking
    estimated_cost_usd: float = 0.0
    
    # By language pair
    by_language_pair: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # By domain
    by_domain: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def record_request(
        self,
        success: bool,
        cached: bool,
        characters: int,
        words: int,
        segments: int,
        duration: float,
        tokens: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        cost: Optional[float] = None,
        language_pair: Optional[str] = None,
        domain: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Record a translation request."""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error:
                self.errors.append(error)
        
        if cached:
            self.cached_requests += 1
        
        self.total_characters += characters
        self.total_words += words
        self.total_segments += segments
        
        if tokens:
            self.total_tokens += tokens
        if prompt_tokens:
            self.prompt_tokens += prompt_tokens
        if completion_tokens:
            self.completion_tokens += completion_tokens
        
        self.total_duration_seconds += duration
        
        if self.min_duration is None or duration < self.min_duration:
            self.min_duration = duration
        if self.max_duration is None or duration > self.max_duration:
            self.max_duration = duration
        
        if cost:
            self.estimated_cost_usd += cost
        
        if language_pair:
            self.by_language_pair[language_pair] += 1
        
        if domain:
            self.by_domain[domain] += 1
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.cached_requests / self.total_requests) * 100
    
    @property
    def avg_duration(self) -> float:
        """Calculate average duration."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_duration_seconds / self.successful_requests
    
    @property
    def avg_characters_per_request(self) -> float:
        """Calculate average characters per request."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_characters / self.successful_requests
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'cached_requests': self.cached_requests,
            'success_rate': f"{self.success_rate:.1f}%",
            'cache_hit_rate': f"{self.cache_hit_rate:.1f}%",
            'total_characters': self.total_characters,
            'total_words': self.total_words,
            'total_segments': self.total_segments,
            'total_tokens': self.total_tokens,
            'total_duration_seconds': round(self.total_duration_seconds, 2),
            'avg_duration_seconds': round(self.avg_duration, 2),
            'min_duration_seconds': round(self.min_duration, 2) if self.min_duration else None,
            'max_duration_seconds': round(self.max_duration, 2) if self.max_duration else None,
            'estimated_cost_usd': round(self.estimated_cost_usd, 4),
            'by_language_pair': dict(self.by_language_pair),
            'by_domain': dict(self.by_domain),
            'error_count': len(self.errors)
        }


class MetricsTracker:
    """Global metrics tracker."""
    
    def __init__(self):
        self.metrics = TranslationMetrics()
        self._start_times = {}
    
    def start_request(self, request_id: str):
        """Start timing a request."""
        self._start_times[request_id] = time.time()
    
    def end_request(
        self,
        request_id: str,
        success: bool,
        cached: bool = False,
        characters: int = 0,
        words: int = 0,
        segments: int = 1,
        tokens: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        cost: Optional[float] = None,
        language_pair: Optional[str] = None,
        domain: Optional[str] = None,
        error: Optional[str] = None
    ):
        """End timing and record metrics."""
        duration = 0.0
        if request_id in self._start_times:
            duration = time.time() - self._start_times[request_id]
            del self._start_times[request_id]
        
        self.metrics.record_request(
            success=success,
            cached=cached,
            characters=characters,
            words=words,
            segments=segments,
            duration=duration,
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            language_pair=language_pair,
            domain=domain,
            error=error
        )
    
    def get_metrics(self) -> TranslationMetrics:
        """Get current metrics."""
        return self.metrics
    
    def reset(self):
        """Reset metrics."""
        self.metrics = TranslationMetrics()
        self._start_times = {}


# Global instance
_tracker = MetricsTracker()


def get_tracker() -> MetricsTracker:
    """Get global metrics tracker."""
    return _tracker


if __name__ == "__main__":
    # Test metrics
    tracker = MetricsTracker()
    
    # Simulate some requests
    tracker.start_request("req1")
    time.sleep(0.1)
    tracker.end_request(
        "req1",
        success=True,
        characters=100,
        words=20,
        language_pair="en-ru",
        domain="general"
    )
    
    tracker.start_request("req2")
    time.sleep(0.2)
    tracker.end_request(
        "req2",
        success=True,
        cached=True,
        characters=150,
        words=30,
        language_pair="en-ru",
        domain="legal"
    )
    
    # Print metrics
    metrics = tracker.get_metrics()
    print("Metrics:")
    for key, value in metrics.to_dict().items():
        print(f"  {key}: {value}")
