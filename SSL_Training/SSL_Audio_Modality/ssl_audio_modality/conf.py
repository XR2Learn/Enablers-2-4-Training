"""
File to include global variables across the python package and configuration.
All the other files inside the python package can access this variables.
"""
from decouple import config

TESTING = 'Importing variable from conf.py'
