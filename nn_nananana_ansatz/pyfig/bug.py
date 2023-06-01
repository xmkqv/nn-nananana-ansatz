from typing import Callable, Any

def try_this(fn: Callable):
    try:
        return fn()
    except Exception as e:
        return e

def try_this_wrap(fn: Callable):
    def inner(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return e
    return inner