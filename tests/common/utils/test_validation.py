# tests/common/utils/test_validation.py
import pytest
import uuid
from decimal import Decimal
from datetime import datetime, date
from common.utils.validation import Validator, ValidationError

class TestValidator:
    """Tests for the Validator class."""
    
    def test_is_string(self):
        """Test checking if a value is a string."""
        assert Validator.is_string("test") is True
        assert Validator.is_string("") is True
        assert Validator.is_string(123) is False
        assert Validator.is_string(None) is False
    
    def test_is_not_empty_string(self):
        """Test checking if a value is a non-empty string."""
        assert Validator.is_not_empty_string("test") is True
        assert Validator.is_not_empty_string("") is False
        assert Validator.is_not_empty_string(" ") is False
        assert Validator.is_not_empty_string(123) is False
        assert Validator.is_not_empty_string(None) is False
    
    def test_is_number(self):
        """Test checking if a value is a number."""
        assert Validator.is_number(123) is True
        assert Validator.is_number(123.45) is True
        assert Validator.is_number(Decimal("123.45")) is True
        assert Validator.is_number("123") is False
        assert Validator.is_number(None) is False
    
    def test_is_integer(self):
        """Test checking if a value is an integer."""
        assert Validator.is_integer(123) is True
        assert Validator.is_integer(0) is True
        assert Validator.is_integer(-123) is True
        assert Validator.is_integer(123.45) is False
        assert Validator.is_integer("123") is False
        assert Validator.is_integer(None) is False
    
    def test_is_positive(self):
        """Test checking if a value is a positive number."""
        assert Validator.is_positive(123) is True
        assert Validator.is_positive(0.1) is True
        assert Validator.is_positive(0) is False
        assert Validator.is_positive(-123) is False
        assert Validator.is_positive("123") is False
        assert Validator.is_positive(None) is False
    
    def test_is_non_negative(self):
        """Test checking if a value is a non-negative number."""
        assert Validator.is_non_negative(123) is True
        assert Validator.is_non_negative(0) is True
        assert Validator.is_non_negative(-0.1) is False
        assert Validator.is_non_negative(-123) is False
        assert Validator.is_non_negative("123") is False
        assert Validator.is_non_negative(None) is False
    
    def test_is_in_range(self):
        """Test checking if a value is in a range."""
        assert Validator.is_in_range(5, 1, 10) is True
        assert Validator.is_in_range(1, 1, 10) is True
        assert Validator.is_in_range(10, 1, 10) is True
        assert Validator.is_in_range(0, 1, 10) is False
        assert Validator.is_in_range(11, 1, 10) is False
        assert Validator.is_in_range(5, None, 10) is True
        assert Validator.is_in_range(5, 1, None) is True
        assert Validator.is_in_range("5", 1, 10) is False
        assert Validator.is_in_range(None, 1, 10) is False
    
    def test_is_boolean(self):
        """Test checking if a value is a boolean."""
        assert Validator.is_boolean(True) is True
        assert Validator.is_boolean(False) is True
        assert Validator.is_boolean(1) is False
        assert Validator.is_boolean(0) is False
        assert Validator.is_boolean("True") is False
        assert Validator.is_boolean(None) is False
    
    def test_is_list(self):
        """Test checking if a value is a list."""
        assert Validator.is_list([]) is True
        assert Validator.is_list([1, 2, 3]) is True
        assert Validator.is_list(()) is False
        assert Validator.is_list({}) is False
        assert Validator.is_list("list") is False
        assert Validator.is_list(None) is False
    
    def test_is_dict(self):
        """Test checking if a value is a dictionary."""
        assert Validator.is_dict({}) is True
        assert Validator.is_dict({"key": "value"}) is True
        assert Validator.is_dict([]) is False
        assert Validator.is_dict(()) is False
        assert Validator.is_dict("dict") is False
        assert Validator.is_dict(None) is False
    
    def test_is_datetime(self):
        """Test checking if a value is a datetime object."""
        assert Validator.is_datetime(datetime.now()) is True
        assert Validator.is_datetime(date.today()) is False
        assert Validator.is_datetime("2020-01-01") is False
        assert Validator.is_datetime(None) is False
    
    def test_is_date(self):
        """Test checking if a value is a date object."""
        assert Validator.is_date(date.today()) is True
        assert Validator.is_date(datetime.now()) is False
        assert Validator.is_date("2020-01-01") is False
        assert Validator.is_date(None) is False
    
    def test_is_uuid(self):
        """Test checking if a value is a valid UUID."""
        uid = uuid.uuid4()
        assert Validator.is_uuid(uid) is True
        assert Validator.is_uuid(str(uid)) is True
        assert Validator.is_uuid("not-a-uuid") is False
        assert Validator.is_uuid(123) is False
        assert Validator.is_uuid(None) is False
    
    def test_is_email(self):
        """Test checking if a value is a valid email address."""
        assert Validator.is_email("test@example.com") is True
        assert Validator.is_email("test.name@example.co.uk") is True
        assert Validator.is_email("test@example") is False
        assert Validator.is_email("test") is False
        assert Validator.is_email("@example.com") is False
        assert Validator.is_email(123) is False
        assert Validator.is_email(None) is False
    
    def test_is_url(self):
        """Test checking if a value is a valid URL."""
        assert Validator.is_url("http://example.com") is True
        assert Validator.is_url("https://example.com/path?query=value") is True
        assert Validator.is_url("ftp://example.com") is True
        assert Validator.is_url("example.com") is False
        assert Validator.is_url("not-a-url") is False
        assert Validator.is_url(123) is False
        assert Validator.is_url(None) is False
    
    def test_is_ip_address(self):
        """Test checking if a value is a valid IP address."""
        assert Validator.is_ip_address("192.168.1.1") is True
        assert Validator.is_ip_address("::1") is True
        assert Validator.is_ip_address("2001:0db8:85a3:0000:0000:8a2e:0370:7334") is True
        assert Validator.is_ip_address("not-an-ip") is False
        assert Validator.is_ip_address(123) is False
        assert Validator.is_ip_address(None) is False
    
    def test_is_ipv4_address(self):
        """Test checking if a value is a valid IPv4 address."""
        assert Validator.is_ipv4_address("192.168.1.1") is True
        assert Validator.is_ipv4_address("::1") is False
        assert Validator.is_ipv4_address("not-an-ip") is False
        assert Validator.is_ipv4_address(123) is False
        assert Validator.is_ipv4_address(None) is False
    
    def test_is_ipv6_address(self):
        """Test checking if a value is a valid IPv6 address."""
        assert Validator.is_ipv6_address("::1") is True
        assert Validator.is_ipv6_address("2001:0db8:85a3:0000:0000:8a2e:0370:7334") is True
        assert Validator.is_ipv6_address("192.168.1.1") is False
        assert Validator.is_ipv6_address("not-an-ip") is False
        assert Validator.is_ipv6_address(123) is False
        assert Validator.is_ipv6_address(None) is False
    
    def test_is_json(self):
        """Test checking if a value is a valid JSON string."""
        assert Validator.is_json('{"key": "value"}') is True
        assert Validator.is_json('[1, 2, 3]') is True
        assert Validator.is_json('"string"') is True
        assert Validator.is_json('not-json') is False
        assert Validator.is_json(123) is False
        assert Validator.is_json(None) is False
    
    def test_matches_pattern(self):
        """Test checking if a value matches a pattern."""
        assert Validator.matches_pattern("abc123", r"^[a-z]+\d+$") is True
        assert Validator.matches_pattern("abc", r"^[a-z]+\d+$") is False
        assert Validator.matches_pattern(123, r"^[a-z]+\d+$") is False
        assert Validator.matches_pattern(None, r"^[a-z]+\d+$") is False
    
    def test_is_valid_length(self):
        """Test checking if a value has a valid length."""
        assert Validator.is_valid_length("test", 1, 10) is True
        assert Validator.is_valid_length([1, 2, 3], 1, 10) is True
        assert Validator.is_valid_length("", 1, 10) is False
        assert Validator.is_valid_length("too long string", 1, 10) is False
        assert Validator.is_valid_length("test", None, 10) is True
        assert Validator.is_valid_length("test", 1, None) is True
        assert Validator.is_valid_length(123, 1, 10) is False
        assert Validator.is_valid_length(None, 1, 10) is False
    
    def test_is_in_set(self):
        """Test checking if a value is in a set."""
        assert Validator.is_in_set("a", {"a", "b", "c"}) is True
        assert Validator.is_in_set(1, {1, 2, 3}) is True
        assert Validator.is_in_set("d", {"a", "b", "c"}) is False
        assert Validator.is_in_set(4, {1, 2, 3}) is False
        assert Validator.is_in_set(None, {1, 2, 3}) is False
        assert Validator.is_in_set(None, {1, 2, 3, None}) is True
    
    def test_validate(self):
        """Test validating a value with multiple validators."""
        # Test with all validators passing
        Validator.validate(
            "test",
            [Validator.is_string, lambda x: len(x) > 0],
            "Value must be a non-empty string"
        )
        
        # Test with one validator failing
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate(
                "",
                [Validator.is_string, lambda x: len(x) > 0],
                "Value must be a non-empty string",
                "test_field"
            )
        
        assert str(excinfo.value) == "Value must be a non-empty string"
        assert excinfo.value.field == "test_field"
        assert excinfo.value.value == ""
    
    def test_validate_required(self):
        """Test validating that a value is required."""
        # Test with a non-None value
        Validator.validate_required("test", "test_field")
        
        # Test with None
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate_required(None, "test_field")
        
        assert str(excinfo.value) == "Field 'test_field' is required"
        assert excinfo.value.field == "test_field"
        assert excinfo.value.value is None
    
    def test_validate_type(self):
        """Test validating that a value is of a specific type."""
        # Test with the correct type
        Validator.validate_type("test", "test_field", str)
        
        # Test with an incorrect type
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate_type(123, "test_field", str)
        
        assert str(excinfo.value) == "Field 'test_field' must be of type str"
        assert excinfo.value.field == "test_field"
        assert excinfo.value.value == 123
        
        # Test with None (should not raise if the value is optional)
        Validator.validate_type(None, "test_field", str)
    
    def test_validate_string(self):
        """Test validating that a value is a string with specific constraints."""
        # Test with a valid string
        Validator.validate_string("test", "test_field", 1, 10, r"^[a-z]+$")
        
        # Test with an invalid type
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate_string(123, "test_field")
        
        assert str(excinfo.value) == "Field 'test_field' must be a string"
        
        # Test with a string that is too short
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate_string("", "test_field", 1)
        
        assert str(excinfo.value) == "Field 'test_field' must be at least 1 characters long"
        
        # Test with a string that is too long
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate_string("too long string", "test_field", 1, 10)
        
        assert str(excinfo.value) == "Field 'test_field' must be at most 10 characters long"
        
        # Test with a string that doesn't match the pattern
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate_string("Test123", "test_field", 1, 10, r"^[a-z]+$")
        
        assert str(excinfo.value) == "Field 'test_field' must match the pattern ^[a-z]+$"
        
        # Test with None (required)
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate_string(None, "test_field")
        
        assert str(excinfo.value) == "Field 'test_field' is required"
        
        # Test with None (not required)
        Validator.validate_string(None, "test_field", required=False)
    
    def test_validate_number(self):
        """Test validating that a value is a number with specific constraints."""
        # Test with valid numbers
        Validator.validate_number(123, "test_field", 1, 1000)
        Validator.validate_number(123.45, "test_field", 1, 1000)
        Validator.validate_number(Decimal("123.45"), "test_field", 1, 1000)
        
        # Test with an invalid type
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate_number("123", "test_field")
        
        assert str(excinfo.value) == "Field 'test_field' must be a number"
        
        # Test with a number that is too small
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate_number(0, "test_field", 1)
        
        assert str(excinfo.value) == "Field 'test_field' must be greater than or equal to 1"
        
        # Test with a number that is too large
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate_number(2000, "test_field", 1, 1000)
        
        assert str(excinfo.value) == "Field 'test_field' must be less than or equal to 1000"
        
        # Test with None (required)
        with pytest.raises(ValidationError) as excinfo:
            Validator.validate_number(None, "test_field")
        
        assert str(excinfo.value) == "Field 'test_field' is required"
        
        # Test with None (not required)
        Validator.validate_number(None, "test_field", required=False)