# tests/common/utils/test_config.py
import pytest
import os
import json
import tempfile
from common.utils.config import (
    get_config, get_config_value, get_config_int, 
    get_config_float, get_config_bool, get_config_list,
    get_config_dict, refresh_config
)

class TestConfig:
    """Tests for the config module."""
    
    def setup_method(self):
        """Set up the test environment."""
        # Clear any existing environment variables used in tests
        for key in [
            'TEST_STRING', 'TEST_INT', 'TEST_FLOAT', 'TEST_BOOL_TRUE', 
            'TEST_BOOL_FALSE', 'TEST_LIST', 'TEST_DICT', 'CONFIG_FILE'
        ]:
            if key in os.environ:
                del os.environ[key]
        
        # Reset the config cache
        refresh_config()
    
    def teardown_method(self):
        """Clean up after tests."""
        # Reset the config cache
        refresh_config()
    
    def test_get_config(self):
        """Test getting the config."""
        os.environ['TEST_KEY'] = 'test_value'
        
        config = get_config()
        
        assert 'TEST_KEY' in config
        assert config['TEST_KEY'] == 'test_value'
    
    def test_get_config_with_json_file(self):
        """Test getting the config with a JSON file."""
        # Create a temporary JSON config file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp:
            config_data = {
                'FILE_KEY': 'file_value',
                'NESTED': {
                    'KEY': 'nested_value'
                }
            }
            temp.write(json.dumps(config_data).encode('utf-8'))
            temp_name = temp.name
        
        try:
            # Set the CONFIG_FILE environment variable
            os.environ['CONFIG_FILE'] = temp_name
            
            # Force a refresh of the config
            refresh_config()
            
            config = get_config()
            
            assert 'FILE_KEY' in config
            assert config['FILE_KEY'] == 'file_value'
            assert 'NESTED' in config
            assert config['NESTED'] == {'KEY': 'nested_value'}
        finally:
            # Clean up the temporary file
            os.unlink(temp_name)
    
    def test_get_config_value(self):
        """Test getting a config value."""
        os.environ['TEST_STRING'] = 'test_value'
        
        value = get_config_value('TEST_STRING')
        
        assert value == 'test_value'
    
    def test_get_config_value_with_default(self):
        """Test getting a config value with a default."""
        value = get_config_value('NONEXISTENT_KEY', 'default_value')
        
        assert value == 'default_value'
    
    def test_get_config_int(self):
        """Test getting an integer config value."""
        os.environ['TEST_INT'] = '42'
        
        value = get_config_int('TEST_INT')
        
        assert value == 42
        assert isinstance(value, int)
    
    def test_get_config_int_with_default(self):
        """Test getting an integer config value with a default."""
        value = get_config_int('NONEXISTENT_KEY', 42)
        
        assert value == 42
        assert isinstance(value, int)
    
    def test_get_config_int_with_invalid_value(self):
        """Test getting an integer config value with an invalid value."""
        os.environ['TEST_INT'] = 'not_an_int'
        
        value = get_config_int('TEST_INT', 42)
        
        assert value == 42
        assert isinstance(value, int)
    
    def test_get_config_float(self):
        """Test getting a float config value."""
        os.environ['TEST_FLOAT'] = '3.14'
        
        value = get_config_float('TEST_FLOAT')
        
        assert value == 3.14
        assert isinstance(value, float)
    
    def test_get_config_float_with_default(self):
        """Test getting a float config value with a default."""
        value = get_config_float('NONEXISTENT_KEY', 3.14)
        
        assert value == 3.14
        assert isinstance(value, float)
    
    def test_get_config_float_with_invalid_value(self):
        """Test getting a float config value with an invalid value."""
        os.environ['TEST_FLOAT'] = 'not_a_float'
        
        value = get_config_float('TEST_FLOAT', 3.14)
        
        assert value == 3.14
        assert isinstance(value, float)
    
    def test_get_config_bool_true(self):
        """Test getting a boolean config value that is true."""
        for truth_value in ['true', 'True', 'TRUE', 'yes', 'Yes', 'YES', '1', 'T', 't', 'Y', 'y']:
            os.environ['TEST_BOOL_TRUE'] = truth_value
            
            value = get_config_bool('TEST_BOOL_TRUE')
            
            assert value is True
            assert isinstance(value, bool)
    
    def test_get_config_bool_false(self):
        """Test getting a boolean config value that is false."""
        for false_value in ['false', 'False', 'FALSE', 'no', 'No', 'NO', '0', 'F', 'f', 'N', 'n']:
            os.environ['TEST_BOOL_FALSE'] = false_value
            
            value = get_config_bool('TEST_BOOL_FALSE')
            
            assert value is False
            assert isinstance(value, bool)
    
    def test_get_config_bool_with_default(self):
        """Test getting a boolean config value with a default."""
        value = get_config_bool('NONEXISTENT_KEY', True)
        
        assert value is True
        assert isinstance(value, bool)
    
    def test_get_config_bool_with_invalid_value(self):
        """Test getting a boolean config value with an invalid value."""
        os.environ['TEST_BOOL_TRUE'] = 'not_a_bool'
        
        value = get_config_bool('TEST_BOOL_TRUE', True)
        
        assert value is True
        assert isinstance(value, bool)
    
    def test_get_config_list(self):
        """Test getting a list config value."""
        os.environ['TEST_LIST'] = 'item1,item2,item3'
        
        value = get_config_list('TEST_LIST')
        
        assert value == ['item1', 'item2', 'item3']
        assert isinstance(value, list)
    
    def test_get_config_list_with_custom_separator(self):
        """Test getting a list config value with a custom separator."""
        os.environ['TEST_LIST'] = 'item1;item2;item3'
        
        value = get_config_list('TEST_LIST', separator=';')
        
        assert value == ['item1', 'item2', 'item3']
        assert isinstance(value, list)
    
    def test_get_config_list_with_default(self):
        """Test getting a list config value with a default."""
        value = get_config_list('NONEXISTENT_KEY', ['default1', 'default2'])
        
        assert value == ['default1', 'default2']
        assert isinstance(value, list)
    
    def test_get_config_dict(self):
        """Test getting a dictionary config value."""
        os.environ['TEST_DICT'] = '{"key1": "value1", "key2": "value2"}'
        
        value = get_config_dict('TEST_DICT')
        
        assert value == {'key1': 'value1', 'key2': 'value2'}
        assert isinstance(value, dict)
    
    def test_get_config_dict_with_default(self):
        """Test getting a dictionary config value with a default."""
        value = get_config_dict('NONEXISTENT_KEY', {'default_key': 'default_value'})
        
        assert value == {'default_key': 'default_value'}
        assert isinstance(value, dict)
    
    def test_get_config_dict_with_invalid_value(self):
        """Test getting a dictionary config value with an invalid value."""
        os.environ['TEST_DICT'] = 'not_a_dict'
        
        value = get_config_dict('TEST_DICT', {'default_key': 'default_value'})
        
        assert value == {'default_key': 'default_value'}
        assert isinstance(value, dict)
    
    def test_refresh_config(self):
        """Test refreshing the config cache."""
        os.environ['TEST_KEY'] = 'initial_value'
        
        # Get the initial config
        config1 = get_config()
        assert config1['TEST_KEY'] == 'initial_value'
        
        # Change the environment variable
        os.environ['TEST_KEY'] = 'updated_value'
        
        # Get the config again (should still have the old value due to caching)
        config2 = get_config()
        assert config2['TEST_KEY'] == 'initial_value'
        
        # Refresh the config cache
        refresh_config()
        
        # Get the config again (should now have the updated value)
        config3 = get_config()
        assert config3['TEST_KEY'] == 'updated_value'