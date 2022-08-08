#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import colossalai
import logging
import inspect, os.path as osp, os
from pathlib import Path
from typing import Union, List

from colossalai.context.parallel_mode import ParallelMode


from rich.logging import RichHandler

_FORMAT = "%(name)s - %(levelname)s - %(asctime)s - %(message)s"
logging.basicConfig(
    level=logging.INFO, format=_FORMAT, handlers=[RichHandler(show_path=False, markup=True, rich_tracebacks=True, locals_max_string=180)]
)


class DistributedLogger:
    """This is a distributed event logger class essentially based on :class:`logging`.

    Args:
        name (str): The name of the logger.

    Note:
        The parallel_mode used in ``info``, ``warning``, ``debug`` and ``error``
        should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """

    __instances = dict()

    @staticmethod
    def get_instance(name: str):
        """Get the unique single logger instance based on name.

        Args:
            name (str): The name of the logger.

        Returns:
            DistributedLogger: A DistributedLogger object
        """
        if name in DistributedLogger.__instances:
            return DistributedLogger.__instances[name]
        else:
            logger = DistributedLogger(name=name)
            return logger

    def __init__(self, name):
        if name in DistributedLogger.__instances:
            raise Exception("Logger with the same name has been created, you should use colossalai.logging.get_dist_logger")
        else:
            self._name = name
            self._logger = logging.getLogger(name)
            for handler in self._logger.handlers:
                formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
                handler.setFormatter(formatter)
            DistributedLogger.__instances[name] = self

    @staticmethod
    def __get_call_info():
        stack = inspect.stack()

        # stack[1] gives previous function ('info' in our case)
        # stack[2] gives before previous function and so on

        fn = osp.basename(stack[2][1])
        ln = stack[2][2]
        func = stack[2][3]

        return fn, ln, func

    @staticmethod
    def _check_valid_logging_level(level: str):
        assert level in ["INFO", "DEBUG", "WARNING", "ERROR"], "found invalid logging level"

    def set_level(self, level: str) -> None:
        """Set the logging level

        Args:
            level (str): Can only be INFO, DEBUG, WARNING and ERROR.
        """
        self._check_valid_logging_level(level)
        self._logger.setLevel(getattr(logging, level))

    def log_to_file(self, path: Union[str, Path], mode: str = "a", level: str = "INFO", suffix: str = None) -> None:
        """Save the logs to file

        Args:
            path (A string or pathlib.Path object): The file to save the log.
            mode (str): The mode to write log into the file.
            level (str): Can only be INFO, DEBUG, WARNING and ERROR.
            suffix (str): The suffix string of log's name.
        """
        self._check_valid_logging_level(level)
        # set the default file name if path is a directory
        if not colossalai.core.global_context.is_initialized(ParallelMode.GLOBAL):
            rank = 0
        else:
            rank = colossalai.core.global_context.get_global_rank()
        if rank != 0:
            return
        if isinstance(path, Path):
            path = str(path)
        assert isinstance(path, str), f"expected argument path to be type str, but got {type(path)}"

        # create log directory
        os.makedirs(osp.dirname(path), exist_ok=True)
        file_handler = logging.FileHandler(path, mode)
        file_handler.setLevel(getattr(logging, level))
        formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d,%H:%M:%S")
        file_handler.setFormatter(formatter)
        self._logger.propagate = False
        self._logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

    def flush(self):
        for handler in self._logger.handlers:
            handler.flush()

    def _log(self, level, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: List[int] = None) -> None:
        if ranks is None:
            getattr(self._logger, level)(message)
            self.flush()
        else:
            local_rank = colossalai.core.global_context.get_local_rank(parallel_mode)
            if local_rank in ranks:
                getattr(self._logger, level)(message)
                self.flush()

    def info(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: List[int] = None) -> None:
        """Log an info message.
        Args:
            message (str): The message to be logged.
            parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`):
                The parallel mode used for logging. Defaults to ParallelMode.GLOBAL.
            ranks (List[int]): List of parallel ranks.
        """
        message = "[{}, {}, {}] {}".format(*self.__get_call_info(), message)
        self._log("info", message, parallel_mode, ranks)

    def warning(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: List[int] = None) -> None:
        """Log a warning message.

        Args:
            message (str): The message to be logged.
            parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`):
                The parallel mode used for logging. Defaults to ParallelMode.GLOBAL.
            ranks (List[int]): List of parallel ranks.
        """
        message = "[{}, {}, {}] {}".format(*self.__get_call_info(), message)
        self._log("warning", message, parallel_mode, ranks)

    def debug(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: List[int] = None) -> None:
        """Log a debug message.

        Args:
            message (str): The message to be logged.
            parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`):
                The parallel mode used for logging. Defaults to ParallelMode.GLOBAL.
            ranks (List[int]): List of parallel ranks.
        """
        message = "[{}, {}, {}] {}".format(*self.__get_call_info(), message)
        self._log("debug", message, parallel_mode, ranks)

    def error(self, message: str, parallel_mode: ParallelMode = ParallelMode.GLOBAL, ranks: List[int] = None) -> None:
        """Log an error message.

        Args:
            message (str): The message to be logged.
            parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`):
                The parallel mode used for logging. Defaults to ParallelMode.GLOBAL.
            ranks (List[int]): List of parallel ranks.
        """
        message = "[{}, {}, {}] {}".format(*self.__get_call_info(), message)
        self._log("error", message, parallel_mode, ranks)
