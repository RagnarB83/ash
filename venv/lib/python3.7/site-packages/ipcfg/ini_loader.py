"""Load .ini config files in the IPython config system.
"""
import ast

try:
    import ConfigParser as configparser
except ImportError:
    # python3
    import configparser

from .IPython.loader import FileConfigLoader, ConfigFileNotFound, Config
from .IPython.path import filefind


class IniFileConfigLoader(FileConfigLoader):
    """Load configuration files from .ini format files, using
    ConfigParser
    """
    def __init__(self, filename, path=None):
        super(IniFileConfigLoader, self).__init__()
        self.filename = filename
        self.path = path
        self.full_filename = ''
        self.parser = configparser.RawConfigParser()

    def _find_file(self):
        """Try to find the file by searching the paths."""
        self.full_filename = filefind(self.filename, self.path)

    def load_config(self):
        """Load the config from a file and return it as a Struct."""
        self.clear()
        try:
            self._find_file()
        except IOError as e:
            raise ConfigFileNotFound(str(e))
        self._read_file_as_config()

        return self.config


    def _read_file_as_config(self):
        with open(self.full_filename) as f:
            self.parser.readfp(f)

        for section in self.parser.sections():
            items = {}
            for key, value in self.parser.items(section):

                # Filter out inline comments
                pound_index = value.find('#')
                if pound_index > -1:
                    value = value[:pound_index]

                # Filter any trailing whitespace
                value = value.strip()

                try:
                    value = ast.literal_eval(value)
                except ValueError:
                    pass
                items[key] = value

            self.config[section].update(Config(items))
