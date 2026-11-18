import os
#!/usr/bin/env python3
"""SSH to VM 10.10.120.13 via paramiko with password auth (user/password from env VM_USER/VM_PASS)."""
import paramiko
import shlex
import sys

HOST = "10.10.120.13"
USER = "sysadm"
PASS = os.environ.get("VM_PASS", "")

def ssh_run(cmd: str, timeout: int = 300) -> tuple:
    """Run cmd on VM. Returns (rc, stdout, stderr)."""
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(HOST, username=USER, password=PASS, timeout=15, look_for_keys=False, allow_agent=False)
        # wrap in bash -lc so PATH/sudo work
        remote = f"bash -lc {shlex.quote(cmd)}"
        stdin, stdout, stderr = c.exec_command(remote, timeout=timeout, get_pty=False)
        out = stdout.read().decode()
        err = stderr.read().decode()
        rc = stdout.channel.recv_exit_status()
        c.close()
        return rc, out, err
    except Exception as e:
        return -1, "", f"EXC: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: ssh_vm.py <remote cmd>", file=sys.stderr)
        sys.exit(2)
    rc, out, err = ssh_run(sys.argv[1], timeout=int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    sys.stdout.write(out)
    sys.stderr.write(err)
    sys.exit(rc if rc >= 0 else 1)
