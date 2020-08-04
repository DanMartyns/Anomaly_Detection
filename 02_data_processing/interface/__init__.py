from flask import Flask

interface = Flask(__name__, template_folder='template', static_folder='static')

from interface import interface