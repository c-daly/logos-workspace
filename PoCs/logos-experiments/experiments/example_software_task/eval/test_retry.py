"""
Eval for the retry mechanism ticket.

Tests the real logos_events.EventBus.publish() for retry behavior
on transient Redis errors. These tests fail until retry logic is
added to logos/logos_events/event_bus.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from logos_events.event_bus import EventBus
from logos_config.settings import RedisConfig


@pytest.fixture
def bus():
    """Create an EventBus with a mock Redis client injected."""
    config = RedisConfig(url="redis://localhost:6379")
    b = EventBus(redis_config=config)
    b._redis = MagicMock()
    return b


def _event(**overrides):
    """Build a standard LOGOS event dict."""
    e = {"event_type": "test", "source": "eval", "payload": {"key": "value"}}
    e.update(overrides)
    return e


class TestRetryBehavior:
    def test_publish_succeeds_on_first_try(self, bus):
        """Normal publish works without retry."""
        bus.publish("logos:test:event", _event())
        bus._redis.publish.assert_called_once()

    def test_retries_on_transient_error(self, bus):
        """Publish retries when Redis raises a connection error."""
        bus._redis.publish.side_effect = [
            ConnectionError("Connection lost"),
            ConnectionError("Connection lost"),
            None,
        ]

        bus.publish("logos:test:event", _event())

        assert bus._redis.publish.call_count == 3

    def test_raises_after_max_retries(self, bus):
        """After 3 retries (4 total calls), the error propagates."""
        bus._redis.publish.side_effect = ConnectionError("Connection lost")

        with pytest.raises(ConnectionError):
            bus.publish("logos:test:event", _event())

        assert bus._redis.publish.call_count == 4

    def test_exponential_backoff(self, bus):
        """Retry delays increase exponentially."""
        bus._redis.publish.side_effect = [
            ConnectionError("Connection lost"),
            ConnectionError("Connection lost"),
            None,
        ]

        delays = []
        with patch("time.sleep", side_effect=lambda d: delays.append(d)):
            with patch("logos_events.event_bus.time.sleep",
                       side_effect=lambda d: delays.append(d), create=True):
                bus.publish("logos:test:event", _event())

        assert len(delays) == 2
        assert delays[1] > delays[0], "Backoff should increase"

    def test_no_retry_on_value_error(self, bus):
        """Non-transient errors raise immediately without retry."""
        bus._redis.publish.side_effect = ValueError("Bad message format")

        with pytest.raises(ValueError):
            bus.publish("logos:test:event", _event())

        assert bus._redis.publish.call_count == 1
