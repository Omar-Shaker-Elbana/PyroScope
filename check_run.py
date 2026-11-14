import json, pandas as pd, numpy as np
d = "data/processed_24h/"
j = json.load(open(d + "frp_mapping.json"))
print("k per satellite:", j["k_by_modis_satellite"])
print("scale basis:", j["scale_basis"])
print("k window:", j["scale_policy"].get("k_window"))
for s, i in j["modis_satellites"].items():
    print(s, "| trust:", i["k_trust"], "| pairs:", i["n_pairs"], "| ci:", i.get("k_ci95"),
          "| median dt (min):", i.get("median_abs_dt_min"))
    print("   problems:", i["k_trust_problems"])
c = pd.read_csv(d + "calendar_daily.csv")
print(c[["date", "viirs_detections", "viirs_frp_mw", "harmonized_status",
         "harmonized_frp_mw_viirs_equiv", "modis_frp_viirs_equiv_mw"]].to_string(index=False))
m = pd.read_csv(d + "matched_pairs.csv")
print("matched MODIS pixels:", len(m), "| with a VIIRS match:", int((m.n_viirs > 0).sum()))
print("cell_lon range:", m.cell_lon.min(), m.cell_lon.max())
for sat in m.source_satellite.unique():
    a = m[(m.source_satellite == sat) & (m.n_viirs > 0) & (m.abs_dt_min <= 90)]
    if len(a):
        print(sat, "independent k (ratio of sums, <=90 min):", round(a.viirs_frp_sum.sum() / a.frp.sum(), 3), "n =", len(a))
