"""
Utilitários de validação para o Neural Crypto Bot.

Este módulo contém funções para validar diversos tipos de dados
usados em todo o sistema.
"""
# src/common/utils/validation.py
import re
from typing import Any, Dict, List, Optional, Union, TypeVar, Type, Callable, Set
from datetime import datetime, date
import json
import uuid
from decimal import Decimal
import ipaddress

T = TypeVar('T')

class ValidationError(Exception):
    """Exception raised when validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        """Initialize a new ValidationError.
        
        Args:
            message: The error message.
            field: The name of the field that failed validation.
            value: The value that failed validation.
        """
        self.field = field
        self.value = value
        super().__init__(message)

class Validator:
    """Utility class for validating values."""
    
    @staticmethod
    def is_string(value: Any) -> bool:
        """Check if a value is a string.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a string, False otherwise.
        """
        return isinstance(value, str)
    
    @staticmethod
    def is_not_empty_string(value: Any) -> bool:
        """Check if a value is a non-empty string.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a non-empty string, False otherwise.
        """
        return isinstance(value, str) and value.strip() != ""
    
    @staticmethod
    def is_number(value: Any) -> bool:
        """Check if a value is a number (int, float, Decimal).
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a number, False otherwise.
        """
        return isinstance(value, (int, float, Decimal))
    
    @staticmethod
    def is_integer(value: Any) -> bool:
        """Check if a value is an integer.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is an integer, False otherwise.
        """
        return isinstance(value, int)
    
    @staticmethod
    def is_positive(value: Any) -> bool:
        """Check if a value is a positive number.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a positive number, False otherwise.
        """
        return Validator.is_number(value) and value > 0
    
    @staticmethod
    def is_non_negative(value: Any) -> bool:
        """Check if a value is a non-negative number.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a non-negative number, False otherwise.
        """
        return Validator.is_number(value) and value >= 0
    
    @staticmethod
    def is_in_range(value: Any, min_value: Optional[float] = None, max_value: Optional[float] = None) -> bool:
        """Check if a value is in the given range.
        
        Args:
            value: The value to check.
            min_value: The minimum value (inclusive). If None, there is no minimum.
            max_value: The maximum value (inclusive). If None, there is no maximum.
            
        Returns:
            True if the value is in the range, False otherwise.
        """
        if not Validator.is_number(value):
            return False
        
        if min_value is not None and value < min_value:
            return False
        
        if max_value is not None and value > max_value:
            return False
        
        return True
    
    @staticmethod
    def is_boolean(value: Any) -> bool:
        """Check if a value is a boolean.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a boolean, False otherwise.
        """
        return isinstance(value, bool)
    
    @staticmethod
    def is_list(value: Any) -> bool:
        """Check if a value is a list.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a list, False otherwise.
        """
        return isinstance(value, list)
    
    @staticmethod
    def is_dict(value: Any) -> bool:
        """Check if a value is a dictionary.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a dictionary, False otherwise.
        """
        return isinstance(value, dict)
    
    @staticmethod
    def is_datetime(value: Any) -> bool:
        """Check if a value is a datetime object.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a datetime object, False otherwise.
        """
        return isinstance(value, datetime)
    
    @staticmethod
    def is_date(value: Any) -> bool:
        """Check if a value is a date object.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a date object, False otherwise.
        """
        return isinstance(value, date)
    
    @staticmethod
    def is_uuid(value: Any) -> bool:
        """Check if a value is a valid UUID.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a valid UUID, False otherwise.
        """
        if isinstance(value, uuid.UUID):
            return True
        
        if not isinstance(value, str):
            return False
        
        try:
            uuid.UUID(value)
            return True
        except (ValueError, AttributeError):
            return False
    
    @staticmethod
    def is_email(value: Any) -> bool:
        """Check if a value is a valid email address.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a valid email address, False otherwise.
        """
        if not isinstance(value, str):
            return False
        
        # Simple email validation regex
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_regex, value))
    
    @staticmethod
    def is_url(value: Any) -> bool:
        """Check if a value is a valid URL.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a valid URL, False otherwise.
        """
        if not isinstance(value, str):
            return False
        
        # Simple URL validation regex
        url_regex = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
        return bool(re.match(url_regex, value))
    
    @staticmethod
    def is_ip_address(value: Any) -> bool:
        """Check if a value is a valid IP address (IPv4 or IPv6).
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a valid IP address, False otherwise.
        """
        if not isinstance(value, str):
            return False
        
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_ipv4_address(value: Any) -> bool:
        """Check if a value is a valid IPv4 address.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a valid IPv4 address, False otherwise.
        """
        if not isinstance(value, str):
            return False
        
        try:
            ip = ipaddress.ip_address(value)
            return isinstance(ip, ipaddress.IPv4Address)
        except ValueError:
            return False
    
    @staticmethod
    def is_ipv6_address(value: Any) -> bool:
        """Check if a value is a valid IPv6 address.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a valid IPv6 address, False otherwise.
        """
        if not isinstance(value, str):
            return False
        
        try:
            ip = ipaddress.ip_address(value)
            return isinstance(ip, ipaddress.IPv6Address)
        except ValueError:
            return False
    
    @staticmethod
    def is_json(value: Any) -> bool:
        """Check if a value is a valid JSON string.
        
        Args:
            value: The value to check.
            
        Returns:
            True if the value is a valid JSON string, False otherwise.
        """
        if not isinstance(value, str):
            return False
        
        try:
            json.loads(value)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def matches_pattern(value: Any, pattern: str) -> bool:
        """Check if a value matches a regular expression pattern.
        
        Args:
            value: The value to check.
            pattern: The regular expression pattern to match.
            
        Returns:
            True if the value matches the pattern, False otherwise.
        """
        if not isinstance(value, str):
            return False
        
        return bool(re.match(pattern, value))
    
    @staticmethod
    def is_valid_length(value: Any, min_length: Optional[int] = None, max_length: Optional[int] = None) -> bool:
        """Check if a value has a valid length.
        
        Args:
            value: The value to check.
            min_length: The minimum length (inclusive). If None, there is no minimum.
            max_length: The maximum length (inclusive). If None, there is no maximum.
            
        Returns:
            True if the value has a valid length, False otherwise.
        """
        if not hasattr(value, '__len__'):
            return False
        
        length = len(value)
        
        if min_length is not None and length < min_length:
            return False
        
        if max_length is not None and length > max_length:
            return False
        
        return True
    
    @staticmethod
    def is_in_set(value: Any, valid_values: Set[Any]) -> bool:
        """Check if a value is in a set of valid values.
        
        Args:
            value: The value to check.
            valid_values: The set of valid values.
            
        Returns:
            True if the value is in the set, False otherwise.
        """
        return value in valid_values
    
    @staticmethod
    def validate(
        value: Any, 
        validators: List[Callable[[Any], bool]], 
        error_message: str,
        field: Optional[str] = None,
    ) -> None:
        """Validate a value using multiple validators.
        
        Args:
            value: The value to validate.
            validators: A list of validator functions.
            error_message: The error message to raise if validation fails.
            field: The name of the field being validated.
            
        Raises:
            ValidationError: If any validator returns False.
        """
        for validator in validators:
            if not validator(value):
                raise ValidationError(error_message, field, value)
    
    @staticmethod
    def validate_required(value: Any, field: str) -> None:
        """Validate that a value is not None.
        
        Args:
            value: The value to validate.
            field: The name of the field being validated.
            
        Raises:
            ValidationError: If the value is None.
        """
        if value is None:
            raise ValidationError(f"Field '{field}' is required", field, value)
    
    @staticmethod
    def validate_type(value: Any, field: str, expected_type: Type[T]) -> None:
        """Validate that a value is of the expected type.
        
        Args:
            value: The value to validate.
            field: The name of the field being validated.
            expected_type: The expected type of the value.
            
        Raises:
            ValidationError: If the value is not of the expected type.
        """
        if value is not None and not isinstance(value, expected_type):
            raise ValidationError(
                f"Field '{field}' must be of type {expected_type.__name__}",
                field,
                value
            )
    
    @staticmethod
    def validate_string(
        value: Any, 
        field: str, 
        min_length: Optional[int] = None, 
        max_length: Optional[int] = None, 
        pattern: Optional[str] = None,
        required: bool = True,
    ) -> None:
        """Validate that a value is a string with specific constraints.
        
        Args:
            value: The value to validate.
            field: The name of the field being validated.
            min_length: The minimum length of the string.
            max_length: The maximum length of the string.
            pattern: A regular expression pattern that the string must match.
            required: Whether the field is required.
            
        Raises:
            ValidationError: If validation fails.
        """
        if value is None:
            if required:
                raise ValidationError(f"Field '{field}' is required", field, value)
            return
        
        if not isinstance(value, str):
            raise ValidationError(f"Field '{field}' must be a string", field, value)
        
        if min_length is not None and len(value) < min_length:
            raise ValidationError(
                f"Field '{field}' must be at least {min_length} characters long",
                field,
                value
            )
        
        if max_length is not None and len(value) > max_length:
            raise ValidationError(
                f"Field '{field}' must be at most {max_length} characters long",
                field,
                value
            )
        
        if pattern is not None and not re.match(pattern, value):
            raise ValidationError(
                f"Field '{field}' must match the pattern {pattern}",
                field,
                value
            )
    
    @staticmethod
    def validate_number(
        value: Any,
        field: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        required: bool = True,
    ) -> None:
        """Validate that a value is a number with specific constraints.
        
        Args:
            value: The value to validate.
            field: The name of the field being validated.
            min_value: The minimum value.
            max_value: The maximum value.
            required: Whether the field is required.
            
        Raises:
            ValidationError: If validation fails.
        """
        if value is None:
            if required:
                raise ValidationError(f"Field '{field}' is required", field, value)
            return
        
        if not isinstance(value, (int, float, Decimal)):
            raise ValidationError(f"Field '{field}' must be a number", field, value)
        
        if min_value is not None and value < min_value:
            raise ValidationError(
                f"Field '{field}' must be greater than or equal to {min_value}",
                field,
                value
            )
        
        if max_value is not None and value > max_value:
            raise ValidationError(
                f"Field '{field}' must be less than or equal to {max_value}",
                field,
                value
            )

    def validate_trading_pair(pair: str) -> bool:
        """
        Validates if a trading pair has the correct format (e.g. 'BTC/USDT').
        
        Args:
            pair: The trading pair to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(pair, str):
            return False
            
        # Check if the pair has the base/quote format
        parts = pair.split('/')
        if len(parts) != 2:
            return False
            
        base, quote = parts
        
        # Check if base and quote are non-empty strings
        if not base or not quote:
            return False
            
        # Check if base and quote contain only valid characters
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if not all(c in valid_chars for c in base) or not all(c in valid_chars for c in quote):
            return False
            
        return True