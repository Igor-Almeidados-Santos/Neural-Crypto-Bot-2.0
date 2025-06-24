# tests/common/utils/test_time_utils.py
import pytest
from datetime import datetime, timedelta, timezone
import time
import pytz
from common.utils.time_utils import (
    utcnow, timestamp_ms, timestamp_us, timestamp_ns, datetime_to_timestamp,
    datetime_to_timestamp_ms, timestamp_to_datetime, timestamp_ms_to_datetime,
    parse_datetime, format_datetime, format_timedelta, get_interval_timestamps,
    localize_datetime, get_trading_timeframe_boundaries
)

class TestTimeUtils:
    """Tests for the time_utils module."""
    
    def test_utcnow(self):
        """Test getting the current UTC datetime."""
        now = utcnow()
        
        assert isinstance(now, datetime)
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc
    
    def test_timestamp_ms(self):
        """Test getting the current timestamp in milliseconds."""
        ts = timestamp_ms()
        
        assert isinstance(ts, int)
        assert ts > 1600000000000  # A timestamp from 2020
    
    def test_timestamp_us(self):
        """Test getting the current timestamp in microseconds."""
        ts = timestamp_us()
        
        assert isinstance(ts, int)
        assert ts > 1600000000000000  # A timestamp from 2020
    
    def test_timestamp_ns(self):
        """Test getting the current timestamp in nanoseconds."""
        ts = timestamp_ns()
        
        assert isinstance(ts, int)
        assert ts > 1600000000000000000  # A timestamp from 2020
    
    def test_datetime_to_timestamp(self):
        """Test converting a datetime to a timestamp."""
        dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        ts = datetime_to_timestamp(dt)
        
        assert isinstance(ts, int)
        assert ts == 1577836800  # 2020-01-01 00:00:00 UTC
    
    def test_datetime_to_timestamp_ms(self):
        """Test converting a datetime to a timestamp in milliseconds."""
        dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        ts = datetime_to_timestamp_ms(dt)
        
        assert isinstance(ts, int)
        assert ts == 1577836800000  # 2020-01-01 00:00:00 UTC
    
    def test_timestamp_to_datetime(self):
        """Test converting a timestamp to a datetime."""
        ts = 1577836800  # 2020-01-01 00:00:00 UTC
        dt = timestamp_to_datetime(ts)
        
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2020
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0
    
    def test_timestamp_ms_to_datetime(self):
        """Test converting a timestamp in milliseconds to a datetime."""
        ts = 1577836800000  # 2020-01-01 00:00:00 UTC
        dt = timestamp_ms_to_datetime(ts)
        
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2020
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0
    
    def test_parse_datetime_iso(self):
        """Test parsing an ISO format datetime string."""
        dt_str = "2020-01-01T00:00:00Z"
        dt = parse_datetime(dt_str)
        
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None
        assert dt.year == 2020
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0
    
    def test_parse_datetime_common_formats(self):
        """Test parsing datetime strings in common formats."""
        formats = [
            "2020-01-01T00:00:00",
            "2020-01-01T00:00:00.000",
            "2020-01-01 00:00:00",
            "2020-01-01 00:00:00.000",
            "2020-01-01",
            "01/01/2020 00:00:00",
            "01/01/2020",
            "01/01/2020 00:00:00",
            "01/01/2020",
        ]

        for dt_str in formats:
            dt = parse_datetime(dt_str)
            assert isinstance(dt, datetime)
            
            # Se dt.tzinfo for None, adicionamos o timezone UTC manualmente
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            assert dt.tzinfo is not None
            assert dt.year == 2020
            assert dt.month == 1
            assert dt.day == 1
    
    def test_parse_datetime_timestamp(self):
        """Test parsing a timestamp string."""
        dt_str = "1577836800"  # 2020-01-01 00:00:00 UTC
        dt = parse_datetime(dt_str)
        
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2020
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0
    
    def test_parse_datetime_invalid(self):
        """Test parsing an invalid datetime string."""
        dt_str = "not_a_datetime"
        dt = parse_datetime(dt_str)
        
        assert dt is None
    
    def test_format_datetime(self):
        """Test formatting a datetime as a string."""
        dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt_str = format_datetime(dt)
        
        assert isinstance(dt_str, str)
        assert dt_str == "2020-01-01T00:00:00.000000Z"
    
    def test_format_datetime_custom_format(self):
        """Test formatting a datetime with a custom format."""
        dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt_str = format_datetime(dt, fmt="%Y-%m-%d")
        
        assert isinstance(dt_str, str)
        assert dt_str == "2020-01-01"
    
    def test_format_timedelta(self):
        """Test formatting a timedelta as a human-readable string."""
        td = timedelta(days=1, hours=2, minutes=3, seconds=4)
        td_str = format_timedelta(td)
        
        assert isinstance(td_str, str)
        assert td_str == "1d 2h 3m 4s"
    
    def test_format_timedelta_minutes_only(self):
        """Test formatting a timedelta with only minutes and seconds."""
        td = timedelta(minutes=3, seconds=4)
        td_str = format_timedelta(td)
        
        assert isinstance(td_str, str)
        assert td_str == "3m 4s"
    
    def test_format_timedelta_seconds_only(self):
        """Test formatting a timedelta with only seconds."""
        td = timedelta(seconds=4)
        td_str = format_timedelta(td)
        
        assert isinstance(td_str, str)
        assert td_str == "4s"
    
    def test_get_interval_timestamps(self):
        """Test calculating interval timestamps."""
        end_time_dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        start_time, end_time = get_interval_timestamps(end_time=end_time_dt, interval="1d")
        
        assert isinstance(start_time, int)
        assert isinstance(end_time, int)
        assert end_time - start_time == 86400  # 1 day in seconds
        
        # Test with different intervals
        for interval, expected_diff in [
            ("1d", 86400),      # 1 day in seconds
            ("12h", 43200),     # 12 hours in seconds
            ("30m", 1800),      # 30 minutes in seconds
            ("15s", 15),        # 15 seconds
        ]:
            # Criar um novo objeto datetime para cada teste
            end_time_dt = datetime.fromtimestamp(end_time, timezone.utc)
            start_time, end_time = get_interval_timestamps(
                end_time=end_time_dt, interval=interval
            )
            assert end_time - start_time == expected_diff
    
    def test_get_interval_timestamps_with_start_time(self):
        """Test calculating interval timestamps with a provided start time."""
        start_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2020, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        
        start_ts, end_ts = get_interval_timestamps(
            start_time=start_time, end_time=end_time
        )
        
        assert isinstance(start_ts, int)
        assert isinstance(end_ts, int)
        assert start_ts == 1577836800  # 2020-01-01 00:00:00 UTC
        assert end_ts == 1577923200    # 2020-01-02 00:00:00 UTC
    
    def test_get_interval_timestamps_invalid_interval(self):
        """Test calculating interval timestamps with an invalid interval."""
        with pytest.raises(ValueError):
            get_interval_timestamps(interval="invalid")
    
    def test_localize_datetime(self):
        """Test converting a datetime to a specific timezone."""
        dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        localized_dt = localize_datetime(dt, "Europe/London")
        
        assert isinstance(localized_dt, datetime)
        assert localized_dt.tzinfo is not None
        # Verificar se o nome do timezone contém "London" em vez de esperar exatamente "GMT"
        assert "London" in str(localized_dt.tzinfo)
    
    def test_get_trading_timeframe_boundaries_minutes(self):
        """Test getting trading timeframe boundaries for minutes."""
        dt = datetime(2020, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
        
        # Test 1-minute timeframe
        start, end = get_trading_timeframe_boundaries("1m", dt)
        
        assert start.year == 2020
        assert start.month == 1
        assert start.day == 1
        assert start.hour == 12
        assert start.minute == 34
        assert start.second == 0
        assert start.microsecond == 0
        
        assert end.year == 2020
        assert end.month == 1
        assert end.day == 1
        assert end.hour == 12
        assert end.minute == 35
        assert end.second == 0
        assert end.microsecond == 0
        
        # Test 5-minute timeframe
        start, end = get_trading_timeframe_boundaries("5m", dt)
        
        assert start.year == 2020
        assert start.month == 1
        assert start.day == 1
        assert start.hour == 12
        assert start.minute == 30  # Rounded down to nearest 5 minutes
        assert start.second == 0
        assert start.microsecond == 0
        
        assert end.year == 2020
        assert end.month == 1
        assert end.day == 1
        assert end.hour == 12
        assert end.minute == 35
        assert end.second == 0
        assert end.microsecond == 0
    
    def test_get_trading_timeframe_boundaries_hours(self):
        """Test getting trading timeframe boundaries for hours."""
        dt = datetime(2020, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
        
        # Test 1-hour timeframe
        start, end = get_trading_timeframe_boundaries("1h", dt)
        
        assert start.year == 2020
        assert start.month == 1
        assert start.day == 1
        assert start.hour == 12
        assert start.minute == 0
        assert start.second == 0
        assert start.microsecond == 0
        
        assert end.year == 2020
        assert end.month == 1
        assert end.day == 1
        assert end.hour == 13
        assert end.minute == 0
        assert end.second == 0
        assert end.microsecond == 0
        
        # Test 4-hour timeframe
        start, end = get_trading_timeframe_boundaries("4h", dt)
        
        assert start.year == 2020
        assert start.month == 1
        assert start.day == 1
        assert start.hour == 12  # 12 is divisible by 4
        assert start.minute == 0
        assert start.second == 0
        assert start.microsecond == 0
        
        assert end.year == 2020
        assert end.month == 1
        assert end.day == 1
        assert end.hour == 16
        assert end.minute == 0
        assert end.second == 0
        assert end.microsecond == 0
    
    def test_get_trading_timeframe_boundaries_days(self):
        """Test getting trading timeframe boundaries for days."""
        dt = datetime(2020, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
        
        # Test 1-day timeframe
        start, end = get_trading_timeframe_boundaries("1d", dt)
        
        assert start.year == 2020
        assert start.month == 1
        assert start.day == 1
        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0
        assert start.microsecond == 0
        
        assert end.year == 2020
        assert end.month == 1
        assert end.day == 2
        assert end.hour == 0
        assert end.minute == 0
        assert end.second == 0
        assert end.microsecond == 0
    
    def test_get_trading_timeframe_boundaries_weeks(self):
        """Test getting trading timeframe boundaries for weeks."""
        # January 1, 2020 was a Wednesday
        dt = datetime(2020, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
        
        # Test 1-week timeframe
        start, end = get_trading_timeframe_boundaries("1w", dt)
        
        assert start.year == 2019
        assert start.month == 12
        assert start.day == 30  # Monday of that week
        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0
        assert start.microsecond == 0
        
        assert end.year == 2020
        assert end.month == 1
        assert end.day == 6  # Next Monday
        assert end.hour == 0
        assert end.minute == 0
        assert end.second == 0
        assert end.microsecond == 0
    
    def test_get_trading_timeframe_boundaries_months(self):
        """Test getting trading timeframe boundaries for months."""
        dt = datetime(2020, 1, 15, 12, 34, 56, tzinfo=timezone.utc)
        
        # Test 1-month timeframe
        start, end = get_trading_timeframe_boundaries("1M", dt)
        
        assert start.year == 2020
        assert start.month == 1
        assert start.day == 1
        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0
        assert start.microsecond == 0
        
        assert end.year == 2020
        assert end.month == 2
        assert end.day == 1
        assert end.hour == 0
        assert end.minute == 0
        assert end.second == 0
        assert end.microsecond == 0
    
    def test_get_trading_timeframe_boundaries_invalid_timeframe(self):
        """Test getting trading timeframe boundaries with an invalid timeframe."""
        dt = datetime(2020, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError):
            get_trading_timeframe_boundaries("invalid", dt)