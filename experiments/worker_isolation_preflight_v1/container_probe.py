"""Runs inside the synthetic container only; no research imports or inputs."""
import ctypes
import errno
import json
import os
import platform
import socket
from pathlib import Path


def restrict_syscalls():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This reviewed seccomp filter requires Linux x86_64')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'PR_SET_NO_NEW_PRIVS failed')

    class Filter(ctypes.Structure):
        _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte),
                    ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ushort),
                    ('filters', ctypes.POINTER(Filter))]

    deny = 0x00050000 | errno.EPERM
    # Check audit architecture and reject the x32 alternate syscall ABI.
    instructions = [Filter(0x20, 0, 0, 4), Filter(0x15, 1, 0, 0xc000003e),
                    Filter(0x06, 0, 0, 0x80000000), Filter(0x20, 0, 0, 0),
                    Filter(0x35, 0, 1, 0x40000000), Filter(0x06, 0, 0, deny)]
    # Socket creation/access, exec, ptrace, namespaces, mount, process memory,
    # pidfd_getfd, BPF and io_uring. No Landlock or user namespace is required.
    blocked = set(range(41, 56)) | {
        59, 101, 155, 161, 165, 166, 272, 288, 299, 304, 307, 308,
        310, 311, 321, 322, 425, 426, 427, 438,
    }
    for number in sorted(blocked):
        instructions += [Filter(0x15, 0, 1, number), Filter(0x06, 0, 0, deny)]
    instructions.append(Filter(0x06, 0, 0, 0x7fff0000))
    array = (Filter * len(instructions))(*instructions)
    program = Program(len(instructions), array)
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0):
        raise OSError(ctypes.get_errno(), 'seccomp filter installation failed')


def probe():
    checks = {}

    def denied(name, action, allowed=(errno.EPERM, errno.EACCES, errno.EROFS)):
        try:
            action()
        except OSError as error:
            if error.errno not in allowed:
                raise
            checks[name] = True
        else:
            checks[name] = False

    restrict_syscalls()
    checks['nonroot_user'] = os.getuid() == os.geteuid() == 65534
    checks['nonroot_group'] = os.getgid() == os.getegid() == 65534
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    checks['capabilities_dropped'] = all(int(status[k].strip(), 16) == 0 for k in ['CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb'])
    checks['no_new_privileges'] = status['NoNewPrivs'].strip() == '1'
    checks['seccomp_installed'] = status['Seccomp'].strip() == '2'
    checks['no_host_credentials'] = not any('TOKEN' in k or 'SECRET' in k or k.endswith('API_KEY') for k in os.environ)
    checks['source_read_allowed'] = Path('/worker/protected.txt').read_text() == 'UNCHANGED_SOURCE'
    denied('source_write_denied', lambda: Path('/worker/protected.txt').write_text('WRONG'))
    denied('source_truncate_denied', lambda: os.truncate('/worker/protected.txt', 0))
    denied('source_unlink_denied', lambda: Path('/worker/protected.txt').unlink())
    denied('root_write_denied', lambda: Path('/forbidden.txt').write_text('WRONG'))
    denied('private_read_denied', lambda: Path('/private/secret.txt').read_text())
    symlink = Path('/output/private_symlink')
    symlink.symlink_to('/private/secret.txt')
    denied('symlink_read_denied', symlink.read_text)
    symlink.unlink()
    denied('hardlink_escape_denied', lambda: os.link('/private/secret.txt', '/output/private_hardlink'),
           (errno.EPERM, errno.EACCES, errno.EROFS, errno.EXDEV))
    # Both internet and local sockets must be denied, not merely disconnected.
    denied('inet_socket_denied', lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM))
    denied('unix_socket_denied', lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))
    denied('exec_denied', lambda: os.execve('/usr/local/bin/python', ['python', '-c', 'raise SystemExit(71)'], {}))
    Path('/output/allowed.txt').write_text('ALLOWED_OUTPUT')
    checks['output_write_allowed'] = Path('/output/allowed.txt').read_text() == 'ALLOWED_OUTPUT'
    Path('/tmp/allowed.txt').write_text('ALLOWED_TEMP')
    checks['temp_write_allowed'] = True
    result = dict(passed=all(checks.values()), checks=checks,
                  platform=dict(system=platform.system(), architecture=platform.machine(), python=platform.python_version()),
                  scientific_inputs_read=0, new_model_forwards=0, new_optimizer_steps=0,
                  parent_agent_isolation_certified=False)
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result['passed']:
        raise SystemExit(3)


if __name__ == '__main__':
    try:
        probe()
    except Exception as error:
        print(json.dumps(dict(passed=False, error_type=type(error).__name__,
                              errno=getattr(error, 'errno', None), reason=str(error))), flush=True)
        raise SystemExit(3)
