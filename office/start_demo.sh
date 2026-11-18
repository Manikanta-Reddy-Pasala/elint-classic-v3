#!/bin/bash
# Start (or restart) the ELINT demo: databases, model service, UI. They also start by themselves at boot.
docker start elint_pg elint_ch >/dev/null
sudo systemctl restart elint-model elint-ui
sleep 5
systemctl --no-pager --lines=0 status elint-model elint-ui | grep -E "●|Active"
echo "Open: http://$(hostname -I | cut -d" " -f1):8800"
