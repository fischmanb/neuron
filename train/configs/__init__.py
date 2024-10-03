import configparser
import json
import os
import pathlib


class Config(configparser.ConfigParser):
    def __init__(self, configspec):
        """
        configfile str full path for ini file https://docs.python.org/3/library/configparser.html
        """
        super(Config, self).__init__()
        configdir = pathlib.Path(__file__).parent
        self.configdir = configdir
        configfile = configdir.joinpath(pathlib.Path(configspec).name)
        if os.path.exists(configfile):
            self.sections()
            self.read(configfile)
        else:
            raise Exception(f"Config file not found {configfile}")
        self.configfile = configfile.name

    def getlist(self, item, **kwargs):
        """
        Provide a python typed object from a configparser file.

        Parameters
        ----------
        item  :  str the attribute being selected from the config file

        Returns
        -------
        list of dict
            value from config file

        """
        try:
            if self["TRAIN"].get(item):
                return json.loads(self["TRAIN"][item])
            return kwargs.get("default")
        except json.decoder.JSONDecodeError as err:
            raise Exception(f"Problem decoding configparser file {self.configfile} - {err}")

    def getdict(self, item, **kwargs):
        """
        Provide a python typed object from a configparser file.

        Parameters
        ----------
        item  :  str the attribute being selected from the config file

        Returns
        -------
        dict
            value from config file

        """
        try:
            if self["TRAIN"].get(item):
                return json.loads(self["TRAIN"][item])
            return kwargs.get("default")
        except json.decoder.JSONDecodeError as err:
            raise Exception(f"Problem decoding configparser file {self.configfile} - {err}")
