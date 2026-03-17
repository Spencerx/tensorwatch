# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Defense-in-depth RestrictedUnpickler for TensorWatch.

Blocks modules commonly exploited in pickle deserialization attacks while
allowing the data types that TensorWatch legitimately serializes (numpy
arrays, torch tensors, TensorWatch data classes, built-in collections, etc.).

WARNING: This is NOT a complete sandbox. A determined attacker may still find
bypass techniques. Do not load pickle data from untrusted sources.
"""

import io
import pickle
import logging

_BLOCKED_MODULES = frozenset({
    # OS / filesystem access
    'os', 'posix', 'nt', 'os.path',
    'shutil', 'pathlib',
    'tempfile', 'glob', 'fnmatch',
    # Process / subprocess execution
    'subprocess', 'multiprocessing',
    'pty', 'commands',
    # Code compilation / execution
    'code', 'codeop', 'compileall',
    'importlib', 'runpy', 'pkgutil',
    # Network
    'socket', 'http', 'urllib', 'ftplib', 'smtplib', 'xmlrpc',
    'socketserver', 'asyncio',
    # Low-level / FFI
    'ctypes', 'mmap',
    # Interactive / debug
    'pdb', 'profile', 'webbrowser',
    # Signal handling
    'signal',
})

# Specific names blocked from builtins module
_BLOCKED_BUILTINS = frozenset({
    'eval', 'exec', 'compile', '__import__',
    'open', 'input', 'breakpoint',
    'exit', 'quit',
    'globals', 'locals', 'vars',
    'getattr', 'setattr', 'delattr',
})


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that blocks known-dangerous modules and callables.

    Allowed: numpy, torch, tensorwatch, collections, standard data types.
    Blocked: os, subprocess, socket, ctypes, importlib, etc.
    """

    def find_class(self, module, name):
        top_module = module.split('.')[0]

        if top_module in _BLOCKED_MODULES:
            raise pickle.UnpicklingError(
                "Blocked: unpickling {}.{} is not allowed "
                "(module '{}' is restricted)".format(module, name, top_module))

        if top_module == 'builtins' and name in _BLOCKED_BUILTINS:
            raise pickle.UnpicklingError(
                "Blocked: unpickling builtins.{} is not allowed".format(name))

        return super().find_class(module, name)


def restricted_loads(data: bytes):
    """Deserialize bytes using RestrictedUnpickler."""
    return RestrictedUnpickler(io.BytesIO(data)).load()


def restricted_load(f):
    """Deserialize from a file object using RestrictedUnpickler."""
    return RestrictedUnpickler(f).load()
