from typing import Callable
from rich.pretty import pretty_repr

from rich.console import Console
console = Console()

from loguru import logger

def create_logger(level: str = 'DEBUG', remove= True) -> str:
    """"""
    
    level = level.upper()

    if remove:
        logger.remove()

    def _log_formatter(record: dict) -> str:
        """Log message formatter"""
        
        lvl: str = record['level'].name.upper()

        color_map = {
            'TRACE': 'dim blue',
            'DEBUG': 'cyan',
            'INFO': 'bold',
            'SUCCESS': 'bold green',
            'WARNING': 'yellow',
            'ERROR': 'bold red',
            'CRITICAL': 'bold white on red'
        }
        
        lvl_color = color_map.get(lvl, 'cyan')
        return ('{level.icon}' + f' [{lvl_color}]{{message}}[/{lvl_color}]')
        # [not bold green]{time: HH:mm:ss}[/not bold green] | 

    handler_id = logger.add(
        console.print,
        level       = level,
        format      = _log_formatter,
        colorize    = True,
    )
    return handler_id # from loguru import logger to use logger


def create_log_io(log) -> Callable:

    def log_io(func: Callable) -> Callable:
        # import inspect
        # func_out = inspect.signature(func).return_annotation

        def wrapper(*args, **kwargs):
            
            args_pr = pretty_repr(args, expand_all= True)
            kwargs_pr = pretty_repr(kwargs, expand_all= True)
            logger.info(
                f"\tcall {func.__name__}\
\ninputs\n\
args: {args_pr}\n\
kwargs: {kwargs_pr}"
)

            result = func(*args, **kwargs)

            result_pr = pretty_repr(result, expand_all= True)
            logger.info(f"\treturn\n{str(result_pr)}\n", extra= {'markup': True})
            
            return result
        return wrapper
    
    return log_io
