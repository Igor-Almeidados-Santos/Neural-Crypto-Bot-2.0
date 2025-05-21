# src/common/utils/time_utils.py
from datetime import datetime, timedelta, timezone
from typing import Optional, Union, Tuple
import time
import pytz
import re

def utcnow() -> datetime:
    """Get the current UTC datetime.
    
    Returns:
        A datetime object representing the current UTC time.
    """
    return datetime.now(timezone.utc)

def timestamp_ms() -> int:
    """Get the current timestamp in milliseconds.
    
    Returns:
        The current timestamp in milliseconds.
    """
    return int(time.time() * 1000)

def timestamp_us() -> int:
    """Get the current timestamp in microseconds.
    
    Returns:
        The current timestamp in microseconds.
    """
    return int(time.time() * 1000000)

def timestamp_ns() -> int:
    """Get the current timestamp in nanoseconds.
    
    Returns:
        The current timestamp in nanoseconds.
    """
    return time.time_ns()

def datetime_to_timestamp(dt: datetime) -> int:
    """Convert a datetime object to a Unix timestamp in seconds.
    
    Args:
        dt: The datetime object to convert.
        
    Returns:
        The Unix timestamp in seconds.
    """
    if dt.tzinfo is None:
        # Assume UTC if no timezone is specified
        dt = dt.replace(tzinfo=timezone.utc)
    
    return int(dt.timestamp())

def datetime_to_timestamp_ms(dt: datetime) -> int:
    """Convert a datetime object to a Unix timestamp in milliseconds.
    
    Args:
        dt: The datetime object to convert.
        
    Returns:
        The Unix timestamp in milliseconds.
    """
    return datetime_to_timestamp(dt) * 1000

def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """Convert a Unix timestamp to a datetime object.
    
    Args:
        timestamp: The Unix timestamp in seconds.
        
    Returns:
        A datetime object representing the timestamp.
    """
    return datetime.fromtimestamp(timestamp, timezone.utc)

# src/common/utils/time_utils.py (continuação)
def timestamp_ms_to_datetime(timestamp_ms: int) -> datetime:
    """Convert a Unix timestamp in milliseconds to a datetime object.
    
    Args:
        timestamp_ms: The Unix timestamp in milliseconds.
        
    Returns:
        A datetime object representing the timestamp.
    """
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)

def parse_datetime(date_string: str) -> Optional[datetime]:
    """Parse a datetime string in various formats.
    
    Args:
        date_string: The datetime string to parse.
        
    Returns:
        A datetime object representing the parsed string, or None if parsing failed.
    """
    # Try to parse ISO format
    try:
        return datetime.fromisoformat(date_string.replace('Z', '+00:00'))
    except ValueError:
        pass
    
    # Try common formats
    formats = [
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d',
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y',
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_string, fmt)
            return dt.replace(tzinfo=timezone.utc)  # Adiciona timezone UTC
        except ValueError:
            continue
    
    # Try to parse Unix timestamp
    try:
        timestamp = float(date_string)
        return timestamp_to_datetime(timestamp)
    except ValueError:
        pass
    
    return None

def format_datetime(dt: datetime, fmt: str = '%Y-%m-%dT%H:%M:%S.%fZ') -> str:
    """Format a datetime object as a string.
    
    Args:
        dt: The datetime object to format.
        fmt: The format string to use.
        
    Returns:
        The formatted datetime string.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.strftime(fmt)

def format_timedelta(td: timedelta) -> str:
    """Format a timedelta object as a human-readable string.
    
    Args:
        td: The timedelta object to format.
        
    Returns:
        A human-readable representation of the timedelta.
    """
    # Get total seconds
    total_seconds = int(td.total_seconds())
    
    # Calculate days, hours, minutes and seconds
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Format the string
    parts = []
    
    if days > 0:
        parts.append(f"{days}d")
    
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")
    
    parts.append(f"{seconds}s")
    
    return " ".join(parts)

def get_interval_timestamps(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    interval: str = '1d',
) -> Tuple[int, int]:
    """Calculate start and end timestamps for a given interval.
    
    Args:
        start_time: The start datetime. If None, it will be calculated based on end_time and interval.
        end_time: The end datetime. If None, the current time will be used.
        interval: The interval string, e.g. '1d', '12h', '30m', '15s'.
        
    Returns:
        A tuple of (start_timestamp, end_timestamp) in seconds.
    """
    if end_time is None:
        end_time = utcnow()
    
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    
    # Parse the interval
    match = re.match(r'^(\d+)([dhms])$', interval)
    if not match:
        raise ValueError(f"Invalid interval format: {interval}. Expected format: '1d', '12h', '30m', '15s'")
    
    value, unit = match.groups()
    value = int(value)
    
    # Calculate the timedelta
    if unit == 'd':
        delta = timedelta(days=value)
    elif unit == 'h':
        delta = timedelta(hours=value)
    elif unit == 'm':
        delta = timedelta(minutes=value)
    elif unit == 's':
        delta = timedelta(seconds=value)
    else:
        raise ValueError(f"Invalid interval unit: {unit}")
    
    # Calculate the start time if not provided
    if start_time is None:
        start_time = end_time - delta
    
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    
    # Return timestamps
    return (datetime_to_timestamp(start_time), datetime_to_timestamp(end_time))

def localize_datetime(dt: datetime, timezone_str: str) -> datetime:
    """Convert a datetime object to a specific timezone.
    
    Args:
        dt: The datetime object to convert.
        timezone_str: The timezone string, e.g. 'Europe/London'.
        
    Returns:
        The datetime object in the specified timezone.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    tz = pytz.timezone(timezone_str)
    return dt.astimezone(tz)

def get_trading_timeframe_boundaries(
    timeframe: str,
    dt: Optional[datetime] = None,
) -> Tuple[datetime, datetime]:
    """Get the start and end datetimes for a trading timeframe.
    
    Args:
        timeframe: The timeframe string, one of '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'.
        dt: The reference datetime. If None, the current time will be used.
        
    Returns:
        A tuple of (start_datetime, end_datetime) for the timeframe containing the reference datetime.
    """
    if dt is None:
        dt = utcnow()
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # Parse the timeframe
    match = re.match(r'^(\d+)([mhdwM])$', timeframe)
    if not match:
        raise ValueError(
            f"Invalid timeframe format: {timeframe}. "
            f"Expected format: '1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'"
        )
    
    value, unit = match.groups()
    value = int(value)
    
    # Calculate the boundaries
    if unit == 'm':
        # Minutes
        minute = (dt.minute // value) * value
        start = dt.replace(minute=minute, second=0, microsecond=0)
        end = start + timedelta(minutes=value)
    elif unit == 'h':
        # Hours
        hour = (dt.hour // value) * value
        start = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=value)
    elif unit == 'd':
        # Days
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=value)
    elif unit == 'w':
        # Weeks (starting on Monday)
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        days_since_monday = dt.weekday()
        start = start - timedelta(days=days_since_monday)
        end = start + timedelta(weeks=value)
    elif unit == 'M':
        # Months
        start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Calculate end date (handling different month lengths)
        if dt.month + value > 12:
            year_offset = (dt.month + value - 1) // 12
            month = (dt.month + value - 1) % 12 + 1
            end = dt.replace(year=dt.year + year_offset, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            end = dt.replace(month=dt.month + value, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"Invalid timeframe unit: {unit}")
    
    return (start, end)