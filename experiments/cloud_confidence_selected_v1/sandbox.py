"""Linux/x86_64 worker restrictions; never certifies the parent agent's sandbox.

Kernel references: docs.kernel.org/userspace-api/landlock.html and
kernel.org/doc/html/latest/userspace-api/seccomp_filter.html.
"""
import ctypes,errno,os,platform,sys
from pathlib import Path


def lockdown(root,out,tmp):
    if platform.system()!='Linux' or platform.machine()!='x86_64':raise RuntimeError('Linux x86_64 required for actual worker isolation; no permissive fallback')
    root,out,tmp=map(lambda p:Path(p).resolve(),[root,out,tmp])
    libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
    def call(n,*args):
        r=libc.syscall(n,*args)
        if r<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
        return r
    abi=call(444,None,0,1)
    if abi<3:raise RuntimeError('Landlock ABI>=3 required, including truncate protection')
    if libc.prctl(38,1,0,0,0):raise OSError(ctypes.get_errno(),'no_new_privs failed')
    access=ctypes.c_uint64((1<<15)-1);fd=call(444,ctypes.byref(access),ctypes.sizeof(access),0)
    class Beneath(ctypes.Structure):
        _pack_=1
        _fields_=[('allowed',ctypes.c_uint64),('parent',ctypes.c_int32)]
    def rule(path,rights):
        p=Path(path).resolve()
        if not p.exists():return
        if not p.is_dir():rights&=(1|2|4|(1<<14))
        handle=os.open(p,os.O_PATH|os.O_CLOEXEC)
        try:call(445,fd,1,ctypes.byref(Beneath(rights,handle)),0)
        finally:os.close(handle)
    try:
        # Read-only program, libraries and data. Home/other tmp directories are
        # not granted access. Private confirmation inputs must not be present.
        for path in [root,'/usr','/lib','/lib64','/bin','/opt',sys.prefix,sys.base_prefix,'/etc/fonts','/etc/ld.so.cache','/proc/self','/proc/cpuinfo','/proc/meminfo','/sys/devices/system/cpu','/dev']:
            rule(path,1|4|8)
        for path in [out,tmp]:rule(path,(1<<15)-1)
        rule('/dev/null',2|4)
        call(446,fd,0)
    finally:os.close(fd)
    class Filter(ctypes.Structure):_fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint32)]
    class Program(ctypes.Structure):_fields_=[('length',ctypes.c_ushort),('filters',ctypes.POINTER(Filter))]
    deny=0x00050000|errno.EPERM
    ins=[Filter(0x20,0,0,4),Filter(0x15,1,0,0xc000003e),Filter(0x06,0,0,0x80000000),Filter(0x20,0,0,0),Filter(0x35,0,1,0x40000000),Filter(0x06,0,0,deny)]
    # No sockets/network, new executables, namespace/mount manipulation,
    # cross-process memory/fd access, or io_uring network bypass.
    for number in list(range(41,56))+[59,101,155,161,165,272,288,299,304,307,308,310,311,321,322,425,426,427,438]:
        ins.extend([Filter(0x15,0,1,number),Filter(0x06,0,0,deny)])
    ins.append(Filter(0x06,0,0,0x7fff0000));array=(Filter*len(ins))(*ins);program=Program(len(ins),array)
    if libc.prctl(22,2,ctypes.byref(program),0,0):raise OSError(ctypes.get_errno(),'seccomp filter installation failed')
    return dict(Landlock_ABI=abi,filesystem_restriction_installed=True,network_and_exec_filter_installed=True,parent_agent_isolation_certified=False)


def proof(root_fixture,private_fixture,out):
    import socket
    out=Path(out);private_fixture=Path(private_fixture);flags={}
    def denied(name,action,errors=(errno.EACCES,errno.EPERM)):
        try:action();flags[name]=False
        except OSError as ex:
            if ex.errno not in errors:raise
            flags[name]=True
    denied('source_write_denied',lambda:Path(root_fixture).write_text('MUST_BE_DENIED'))
    denied('private_read_denied',private_fixture.read_text)
    link=out/'private_symlink';link.symlink_to(private_fixture)
    denied('symlink_read_denied',link.read_text)
    link.unlink()  # Keep the result bundle free of outward fixture symlinks.
    denied('hardlink_escape_denied',lambda:os.link(private_fixture,out/'private_hardlink'),(errno.EACCES,errno.EPERM,errno.EXDEV))
    denied('network_socket_denied',lambda:socket.socket(socket.AF_INET,socket.SOCK_STREAM))
    (out/'allowed_write.txt').write_text('fixture');flags['output_write_allowed']=True
    flags['no_credential_environment']=not any(k in os.environ for k in ['OPENAI_API_KEY','CODEX_API_KEY','GITHUB_TOKEN','GH_TOKEN'])
    if not all(flags.values()):raise RuntimeError('Worker isolation fixture failed: '+str(flags))
    return flags
