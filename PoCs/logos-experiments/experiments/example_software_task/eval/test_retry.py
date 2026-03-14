"""
Eval for the retry mechanism ticket.

These tests define what success looks like. They fail until
the agent builds a correct implementation in workspace/.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The agent's implementation lives in workspace/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workspace"))


def _get_event_bus():
    """Import the agent's EventBus implementation."""
    try:
        from event_bus import EventBus
        return EventBus
    except ImportError:
        pytest.fail(
            "No implementation found. Create workspace/event_bus.py "
            "with an EventBus class that has a publish(channel, message) method."
        )


def _inject_mock_redis(bus, mock_redis):
    """Inject a mock Redis client into an EventBus instance.

    Scans for the attribute holding the Redis client regardless of name,
    so the agent isn't locked into a specific private attribute convention.
    """
    # Try common attribute names agents might use
    for attr in ("_redis", "_client", "_conn", "_connection", "_redis_client", "redis", "client"):
        if hasattr(bus, attr):
            setattr(bus, attr, mock_redis)
            return
    # Fallback: set all None-valued private attrs (likely the unconnected client)
    for attr in vars(bus):
        if attr.startswith("_") and getattr(bus, attr) is None:
            setattr(bus, attr, mock_redis)
            return
    raise AttributeError(
        "Could not find a Redis client attribute on EventBus. "
        "Store your Redis client as an instance attribute (e.g. self._redis)."
    )


class TestRetryBehavior:
    def test_publish_succeeds_on_first_try(self):
        """Normal publish works without retry."""
        EventBus = _get_event_bus()
        bus = EventBus(redis_url="redis://localhost:6379")
        mock_redis = MagicMock()
        _inject_mock_redis(bus, mock_redis)

        bus.publish("events.test", {"key": "value"})

        mock_redis.publish.assert_called_once()

    def test_retries_on_transient_error(self):
        """Publish retries when Redis raises a connection error."""
        EventBus = _get_event_bus()
        bus = EventBus(redis_url="redis://localhost:6379")
        mock_redis = MagicMock()
        # Fail twice, then succeed
        mock_redis.publish.side_effect = [
            ConnectionError("Connection lost"),
            ConnectionError("Connection lost"),
            None,
        ]
        _inject_mock_redis(bus, mock_redis)

        bus.publish("events.test", {"key": "value"})

        assert mock_redis.publish.call_count == 3

    def test_raises_after_max_retries(self):
        """After 3 retries (4 total calls), the error propagates."""
        EventBus = _get_event_bus()
        bus = EventBus(redis_url="redis://localhost:6379")
        mock_redis = MagicMock()
        mock_redis.publish.side_effect = ConnectionError("Connection lost")
        _inject_mock_redis(bus, mock_redis)

        with pytest.raises(ConnectionError):
            bus.publish("events.test", {"key": "value"})

        assert mock_redis.publish.call_count == 4

    def test_exponential_backoff(self):
        """Retry delays increase exponentially."""
        EventBus = _get_event_bus()
        bus = EventBus(redis_url="redis://localhost:6379")
        mock_redis = MagicMock()
        mock_redis.publish.side_effect = [
            ConnectionError("Connection lost"),
            ConnectionError("Connection lost"),
            None,
        ]
        _inject_mock_redis(bus, mock_redis)

        delays = []
        with patch("time.sleep", side_effect=lambda d: delays.append(d)):
            with patch("event_bus.sleep", side_effect=lambda d: delays.append(d), create=True):
                bus.publish("events.test", {"key": "value"})

        assert len(delays) == 2
        assert delays[1] > delays[0], "Backoff should increase"

    def test_no_retry_on_value_error(self):
        """Non-transient errors raise immediately without retry."""
        EventBus = _get_event_bus()
        bus = EventBus(redis_url="redis://localhost:6379")
        mock_redis = MagicMock()
        mock_redis.publish.side_effect = ValueError("Bad message format")
        _inject_mock_redis(bus, mock_redis)

        with pytest.raises(ValueError):
            bus.publish("events.test", {"key": "value"})

        assert mock_redis.publish.call_count == 1
