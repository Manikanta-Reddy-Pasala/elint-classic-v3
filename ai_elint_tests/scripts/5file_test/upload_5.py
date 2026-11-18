#!/usr/bin/env python3
"""Push runner scripts to VM, then launch the 5-file both-mode test in background."""
import os
import paramiko
import shlex
import sys

HOST, USER, PASS = "10.10.120.13", "sysadm", os.environ.get("VM_PASS", "")
SCRATCH = "/mnt/c/Users/Manikanta.Pasala/Documents/coderepo/ai_elint_tests/.aiforge-worktrees/chat-283-e465/scratch"

def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=15, look_for_keys=False, allow_agent=False)
    sftp = c.open_sftp()
    sftp.put(f"{SCRATCH}/run_one.sh", "/home/sysadm/run_one.sh")
    sftp.put(f"{SCRATCH}/run_5_both.sh", "/home/sysadm/run_5_both.sh")
    sftp.close()

    def run(cmd, timeout=600):
        remote = f"bash -lc {shlex.quote(cmd)}"
        stdin, stdout, stderr = c.exec_command(remote, timeout=timeout)
        out = stdout.read().decode()
        err = stderr.read().decode()
        rc = stdout.channel.recv_exit_status()
        print(f"rc={rc}")
        print(out)
        if err:
            print("STDERR:", err, file=sys.stderr)
        return rc

    # kill any stale engine/batch jobs from earlier sessions
    run("pkill -9 -f 'timeout 900 ./target/release/ai_processor' 2>/dev/null; pkill -9 -f run_batch.sh 2>/dev/null; sleep 2; echo KILLED", 30)
    # launch the 5-file both-mode run in background
    rc = run("nohup bash /home/sysadm/run_5_both.sh > /home/sysadm/5files_both_run.log 2>&1 & echo PID=$!", 30)
    c.close()
    sys.exit(0 if rc == 0 else 1)

if __name__ == "__main__":
    main()
