"""Constrain a compute process and descendants to the supervisor-owned group.

Linux uses an inherited seccomp filter, including across exec and native calls.
Darwin API guards cover local Python integration tests only; they are explicitly
reported as weaker and are not a kernel containment guarantee.
"""

from __future__ import annotations

import errno
import os
import platform
import subprocess
import sys


def _deny_group_change(*args, **kwargs):
    raise PermissionError(
        errno.EPERM, "tuning compute cannot change process group/session"
    )


def _python_guards():
    os.setsid = _deny_group_change
    os.setpgid = _deny_group_change
    original = subprocess.Popen.__init__

    def init(self, *args, **kwargs):
        if kwargs.get("start_new_session") or kwargs.get("process_group") not in (
            None,
            -1,
        ):
            _deny_group_change()
        return original(self, *args, **kwargs)

    subprocess.Popen.__init__ = init
    for name in ("posix_spawn", "posix_spawnp"):
        if not hasattr(os, name):
            continue
        original_spawn = getattr(os, name)

        def spawn(*args, _original=original_spawn, **kwargs):
            if kwargs.get("setsid") or kwargs.get("setpgroup") is not None:
                _deny_group_change()
            return _original(*args, **kwargs)

        setattr(os, name, spawn)


def install_process_group_guard():
    if sys.platform.startswith("linux"):
        _install_linux_filter()
        level = "linux-inherited-seccomp-v1"
    elif sys.platform == "darwin":
        level = "darwin-python-api-only-local-validation"
    else:
        raise RuntimeError(f"unsupported tuning containment platform: {sys.platform}")
    _python_guards()
    return {
        "process_group_guard": level,
        "kernel_group_containment": sys.platform.startswith("linux"),
        "scope": "compute and descendants; session/group changes and namespace escapes denied",
    }


def _install_linux_filter():
    import ctypes

    machine = platform.machine().lower()
    # audit arch, setsid, setpgid, unshare, setns, clone, clone3
    layouts = {
        "x86_64": (0xC000003E, 112, 109, 272, 308, 56, 435),
        "aarch64": (0xC00000B7, 157, 154, 97, 268, 220, 435),
    }
    if machine not in layouts:
        raise RuntimeError(f"unsupported Linux seccomp architecture: {machine}")
    arch, setsid, setpgid, unshare, setns, clone, clone3 = layouts[machine]

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    # Classic BPF over struct seccomp_data; unknown compat architecture fails
    # closed. x32 syscall numbers must not bypass the x86_64 syscall checks.
    instructions = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, arch),
        (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0),
    ]
    if machine == "x86_64":
        instructions.extend(
            [(0x35, 0, 1, 0x40000000), (0x06, 0, 0, 0x50000 | errno.ENOSYS)]
        )
    for syscall in (setsid, setpgid, unshare, setns):
        instructions.extend(
            [(0x15, 0, 1, syscall), (0x06, 0, 0, 0x50000 | errno.EPERM)]
        )
    # glibc can fall back from clone3 to clone for ordinary threads/processes.
    instructions.extend(
        [
            (0x15, 0, 1, clone3),
            (0x06, 0, 0, 0x50000 | errno.ENOSYS),
            (0x15, 0, 3, clone),
            (0x20, 0, 0, 16),
            (0x45, 0, 1, 0x7E020000),
            (0x06, 0, 0, 0x50000 | errno.EPERM),
            (0x06, 0, 0, 0x7FFF0000),
        ]
    )
    array = (Filter * len(instructions))(
        *(Filter(*instruction) for instruction in instructions)
    )
    program = Program(len(instructions), array)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(
            ctypes.get_errno(), "cannot enable no_new_privs for tuning compute"
        )
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:  # PR_SET_SECCOMP, FILTER
        raise OSError(
            ctypes.get_errno(), "cannot install inherited tuning seccomp filter"
        )
