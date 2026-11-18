#!/bin/bash
# Print a compact fingerprint of the PG result tables: emitters by name,
# and per signal: emitter name, cluster rows, total pdw_count.
export LC_ALL=C PGPASSWORD=elint
q(){ psql -h 127.0.0.1 -p 15432 -U elint -d sigint -t -A -F' ' -c "$1" 2>/dev/null; }
echo "emitters: $(q "select string_agg(name||'x'||c, ',' order by name) from (select name, count(*) c from emitters group by name) t")"
echo "signals: $(q "select count(*) from signals") clusters: $(q "select count(*) from clusters") pdw: $(q "select coalesce(sum(pdw_count),0) from clusters") orphan: $(q "select count(*) from clusters where signal_id is null")"
q "select e.name, count(c.id), coalesce(sum(c.pdw_count),0) from signals s join emitters e on e.id=s.emitter_id left join clusters c on c.signal_id=s.id group by e.name, s.id order by e.name, 3 desc" | tr '\n' ';'; echo
