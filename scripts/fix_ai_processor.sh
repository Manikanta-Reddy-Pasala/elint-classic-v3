#!/usr/bin/env bash
# fix_ai_processor.sh — run ON THE VM (10.10.120.13) as sysadm.
# Deletes the stale ai-processor consumer group and recreates the service from
# scratch with rdkafka debug logging, then verifies it is actually consuming.
set -u
cd ~/ai_elint || { echo "no ~/ai_elint"; exit 1; }

echo "=== 1. delete stale consumer group ==="
docker exec elint-redpanda rpk group delete ai-processor --brokers 127.0.0.1:9092 2>&1 || true

echo "=== 2. remove + recreate ai-processor container ==="
docker compose rm -f ai-processor 2>&1
# make sure debug logging is on (idempotent)
grep -q 'RUST_LOG=info,rdkafka=debug' docker-compose.yml || \
  sed -i '/CH_URL=tcp:\/\//a\      - RUST_LOG=info,rdkafka=debug' docker-compose.yml
docker compose up -d ai-processor 2>&1

echo "=== 3. wait 25s for group join + first fetch ==="
sleep 25

echo "=== 4. group state ==="
docker exec elint-redpanda rpk group describe ai-processor --brokers 127.0.0.1:9092 2>&1 | head -25

echo "=== 5. last 80 lines of ai-processor log ==="
docker logs elint-ai-processor --tail 80 2>&1

echo "=== 6. verdict ==="
if docker logs elint-ai-processor --tail 200 2>&1 | grep -q 'got [0-9]\+ PDW messages'; then
  echo "PASS: ai-processor is consuming"
else
  echo "CHECK: no 'got N PDW messages' yet — look at the rdkafka debug lines above"
fi
