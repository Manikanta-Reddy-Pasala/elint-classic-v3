import sys, pandas as pd, numpy as np
for f in sys.argv[1:]:
    d = pd.read_csv(f)
    print("==", f.split("/")[-1], len(d), "rows; cols:", ",".join(d.columns))
    d["f_mhz"] = d.center_frequency_hz / 1e6; d["pw_us"] = d.duration_ns / 1e3; d["bw_mhz"] = d.bandwidth_hz / 1e6
    d["pri_us"] = d.pulse_repetition_interval_ns / 1e3
    g = d.groupby("emitter_id")
    out = pd.DataFrame({
        "n": g.size(),
        "f_min": g.f_mhz.min().round(1), "f_max": g.f_mhz.max().round(1),
        "pw": g.pw_us.agg(lambda x: f"{x.min():.1f}-{x.max():.1f}"),
        "bw": g.bw_mhz.agg(lambda x: f"{x.min():.2f}-{x.max():.2f}"),
        "doa": g.direction_of_arrival_deg.agg(lambda x: f"{x.quantile(.05):.0f}-{x.quantile(.95):.0f}"),
        "pri": g.pri_us.agg(lambda x: f"{x.min():.0f}-{x.max():.0f} ({x.round().nunique()})"),
        "type": g.type.agg(lambda x: ",".join(sorted(x.unique()))[:30]),
        "pow": g.power_dbm.agg(lambda x: f"{x.min():.0f}..{x.max():.0f}"),
        "toa_s": g.time_of_arrival_ns.agg(lambda x: f"{x.min()/1e9:.1f}-{x.max()/1e9:.1f}"),
        "blank": g.sidelobe_blanking_flag_as_int8.agg(lambda x: ",".join(map(str,sorted(x.unique())))),
    })
    print(out.to_string())
