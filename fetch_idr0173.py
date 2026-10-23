"""
Download the idr0173 sample images listed in idr0173_sample_manifest.csv and
record per-image metadata (shape, dtype, pixel size, channel names).

Setup (once):
    pip install zarr fsspec aiohttp requests numpy pandas tifffile

Usage:
    python fetch_idr0173.py                     # the 60 main images
    python fetch_idr0173.py --set reserve       # the 20 reserves
    python fetch_idr0173.py --limit 2           # quick test on 2 images first
    python fetch_idr0173.py --keep-stacks       # also save full z-stacks (~140 MB each)

Output (in ./data):
    mip/<sample_id>_MAP2.tif, mip/<sample_id>_pS129.tif   max-intensity projections
    stacks/<sample_id>.tif                               full ZCYX stack (only with --keep-stacks)
    image_metadata.csv                                   one row per image
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, zarr, tifffile


def read_ome_zarr(url):
    # Works with zarr 2 and zarr 3; remote URLs are read through fsspec
    root = zarr.open_group(url, mode="r")
    attrs = dict(root.attrs)
    ms = attrs["multiscales"][0]
    axes = [a["name"] if isinstance(a, dict) else a for a in ms["axes"]]
    ds0 = ms["datasets"][0]
    scale = None
    for t in ds0.get("coordinateTransformations", []):
        if t.get("type") == "scale":
            scale = t["scale"]
    arr = root[ds0["path"]]
    chans = [c.get("label", f"ch{i}") for i, c in enumerate(attrs.get("omero", {}).get("channels", []))]
    return arr, axes, scale, chans, ms["axes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="idr0173_sample_manifest.csv")
    ap.add_argument("--set", default="main", choices=["main", "reserve", "all"])
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-stacks", action="store_true")
    a = ap.parse_args()

    m = pd.read_csv(a.manifest)
    if a.set != "all":
        m = m[m["set"] == a.set]
    if a.limit:
        m = m.head(a.limit)
    os.makedirs(f"{a.out}/mip", exist_ok=True)
    if a.keep_stacks:
        os.makedirs(f"{a.out}/stacks", exist_ok=True)

    meta_path = f"{a.out}/image_metadata.csv"
    prev = pd.read_csv(meta_path) if os.path.exists(meta_path) else pd.DataFrame(columns=["sample_id"])
    if "error" in prev:  # retry previously failed images
        prev = prev[prev["error"].isna()]
    done = set(prev["sample_id"])
    rows = []
    for i, r in enumerate(m.itertuples(), 1):
        if r.sample_id in done:
            print(f"[{i}/{len(m)}] {r.sample_id} already done, skipping")
            continue
        print(f"[{i}/{len(m)}] {r.sample_id} {r.disease} {r.patient} ...", flush=True)
        try:
            arr, axes, scale, chans, axes_full = read_ome_zarr(r.url)
            data = arr[:]  # level 0, full resolution
            # Reorder to (C, Z, Y, X), dropping a singleton T if present
            order = [axes.index(x) for x in ("c", "z", "y", "x")]
            if "t" in axes:
                data = np.take(data, 0, axis=axes.index("t"))
                order = [o - (1 if o > axes.index("t") else 0) for o in order]
            czyx = np.transpose(data, order)
            # Manifest says channel order is "MAP2; pS129"; prefer labels stored in the file if present
            names = chans if len(chans) == czyx.shape[0] else [c.strip() for c in r.channels.split(";")]
            names = [n.replace(" ", "_") for n in names]
            for ci, name in enumerate(names):
                tifffile.imwrite(f"{a.out}/mip/{r.sample_id}_{name}.tif", czyx[ci].max(axis=0))
            if a.keep_stacks:
                tifffile.imwrite(f"{a.out}/stacks/{r.sample_id}.tif", czyx.transpose(1, 0, 2, 3),
                                 imagej=True, metadata={"axes": "ZCYX"})
            units = {ax["name"]: ax.get("unit") for ax in axes_full if isinstance(ax, dict)}
            sc = dict(zip(axes, scale)) if scale else {}
            row = dict(sample_id=r.sample_id, disease=r.disease, patient=r.patient,
                       axes="".join(axes), shape="x".join(map(str, arr.shape)), dtype=str(arr.dtype),
                       n_channels=czyx.shape[0], n_z=czyx.shape[1], height=czyx.shape[2], width=czyx.shape[3],
                       pixel_size_x=sc.get("x"), pixel_size_y=sc.get("y"), pixel_size_z=sc.get("z"),
                       unit_xy=units.get("x"), unit_z=units.get("z"), channel_names=";".join(names))
            for ci, name in enumerate(names):
                ch = czyx[ci]
                row[f"{name}_min"] = int(ch.min()); row[f"{name}_max"] = int(ch.max())
                row[f"{name}_p99"] = float(np.percentile(ch, 99))
                # Saturation check: fraction of voxels at the dtype ceiling
                if np.issubdtype(ch.dtype, np.integer):
                    row[f"{name}_frac_saturated"] = float((ch == np.iinfo(ch.dtype).max).mean())
            rows.append(row)
        except Exception as e:
            print(f"   FAILED: {e!r}", file=sys.stderr)
            rows.append(dict(sample_id=r.sample_id, disease=r.disease, patient=r.patient, error=repr(e)))
        # write as we go so an interrupted run can resume
        pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(meta_path, index=False)
    print(f"Done. Metadata in {meta_path}")


if __name__ == "__main__":
    main()