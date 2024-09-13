import sys
from enum import Enum, unique

from . import print_supported_models
from .api import run_api
from .extras.env import VERSION, print_env
from .extras.logging import get_logger
from .train.tuner import run_exp
from .webui.interface import run_web_demo

USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   fastie-cli api -h: launch an API server                          |\n"
    + "|   fastie-cli train -h: train models                                |\n"
    + "|   fastie-cli demo -h: launch a interface in Web UI                 |\n"
    + "|   fastie-cli version: show version info                            |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 54
    + "\n"
    + "| Welcome to Fast IE, version {}".format(VERSION)
    + " " * (23 - len(VERSION))
    + "|\n|"
    + " " * 52
    + "|\n"
    + "| Project page: https://github.com/xusenlinzy/fastie |\n"
    + "-" * 54
)

logger = get_logger(__name__)


@unique
class Command(str, Enum):
    API = "api"
    ENV = "env"
    TRAIN = "train"
    DEMO = "demo"
    VER = "version"
    HELP = "help"
    MODELS = "models"


def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.API:
        run_api()
    elif command == Command.ENV:
        print_env()
    elif command == Command.TRAIN:
        run_exp()
    elif command == Command.DEMO:
        run_web_demo()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    elif command == Command.MODELS:
        print_supported_models()
    else:
        raise NotImplementedError("Unknown command: {}".format(command))
