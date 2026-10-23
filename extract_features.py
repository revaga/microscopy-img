"""
Week 5: normalize, segment, and extract morphometric features from the idr0173 max projections.

Setup (once):
    pip install cellpose scikit-image tifffile pandas matplotlib

Usage (from the microscopy-img folder):
    python extract_features.py --limit 3          # quick test
    python extract_features.py                    # all images in data/mip
    python extract_features.py --no-cellpose      # skip soma segmentation (fast, classical features only)

Inputs:  data/mip/<id>_cell.tif (MAP2), data/mip/<id>_protein.tif (pS129), idr0173_sample_manifest.csv
Outputs: features/per_image.csv   one row per image  (the morphometric table)
         features/per_soma.csv    one row per Cellpose-segmented soma
         features/masks/<id>_*.tif label/binary masks
         features/qc/<id>.png      overlay to check segmentation by eye

What is measured (pixel size 0.11 um; camera is 12-bit, so 4095 = saturated):
  QC          tissue fraction, saturation, focus, illumination unevenness, autofluorescence (lipofuscin)
  MAP2        dendrite area fraction, skeleton length density, branch-point density, mean width,
              orientation coherence, MAP2 intensity
  Somata      Cellpose-SAM on MAP2 (downsampled 4x): count, area, shape, intensity
  pS129       puncta (oligomer-like spots): count, density, intensity, fraction inside MAP2 / somata
"""
import argparse, os, glob, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np, pandas as pd, tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, feature, transform

PX = 0.11            # um per pixel (from OME-Zarr metadata)
SAT = 4095           # 12-bit camera ceiling
DS = 4               # downsample factor for Cellpose (-> 0.44 um/px)
MIN_SOMA_UM2 = 40    # smaller Cellpose objects are dendrite fragments, not somata
MAX_SOMA_UM2 = 1500
MAX_SOMA_ASPECT = 1.6  # major/minor axis; real somata here are ~1.3, blood vessels 1.7-7+


# ---------------------------------------------------------------- normalization
def normalize(img, sigma_bg=60):
    """Subtract a smooth background (large Gaussian, as in the authors' DoG), then
    scale the 1st-99.8th percentile to 0-1. Returns float32 image and background."""
    img = img.astype(np.float32)
    bg = filters.gaussian(img, sigma_bg, preserve_range=True)
    hp = img - bg
    lo, hi = np.percentile(hp, [1, 99.8])
    return np.clip((hp - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32), bg


def rescale(img):
    lo, hi = np.percentile(img, [1, 99.8])
    return np.clip((img.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)


# ---------------------------------------------------------------- QC masks
def tissue_mask(cell, prot, offset=100):
    """Tissue vs. empty space (holes, vessel lumen, section edge). The 561 nm channel has a
    fairly even tissue autofluorescence floor, while empty space sits at the camera offset
    (~100 counts), so threshold the smoothed pS129 channel just above the offset."""
    s = filters.gaussian(prot.astype(np.float32), 15, preserve_range=True)
    thr = offset + max(40.0, 0.3 * (np.percentile(s, 95) - offset))
    m = s > thr
    m = morphology.remove_small_objects(m, 5000)
    m = morphology.remove_small_holes(m, 5000)
    return m


def autofluor_mask(cell, prot, tissue):
    """Lipofuscin and other autofluorescence: blobs that are very bright and large in the
    561 nm channel (real pS129 puncta are diffraction-limited) and also above background in
    the MAP2 channel (lipofuscin emits in both)."""
    if tissue.sum() < 1000:
        return np.zeros_like(tissue)
    p = filters.gaussian(prot.astype(np.float32), 3, preserve_range=True)
    c = filters.gaussian(cell.astype(np.float32), 3, preserve_range=True)
    pv = p[tissue]; med = np.median(pv); mad = np.median(np.abs(pv - med)) * 1.4826
    m = (p > med + 10 * mad) & (c > np.percentile(c[tissue], 60)) & tissue
    m = morphology.remove_small_objects(m, int(3 / PX**2))      # >3 um^2
    return morphology.binary_dilation(m, morphology.disk(6))


def focus_score(img):
    """Variance of the Laplacian on the rescaled image: lower = blurrier."""
    return float(ndi.laplace(filters.gaussian(rescale(img), 1)).var() * 1e4)


def illumination_unevenness(img, tissue):
    """Coefficient of variation of a heavily smoothed image within tissue."""
    s = filters.gaussian(img.astype(np.float32), 100, preserve_range=True)
    v = s[tissue] if tissue.any() else s.ravel()
    return float(v.std() / (v.mean() + 1e-6))


# ---------------------------------------------------------------- MAP2 dendrites
def dendrite_features(cell_n, tissue, excl):
    dog = filters.gaussian(cell_n, 1) - filters.gaussian(cell_n, 8)
    valid = tissue & ~excl
    thr = filters.threshold_otsu(dog[valid]) if valid.sum() > 1000 else np.inf
    m = (dog > thr) & valid
    m = morphology.remove_small_objects(m, int(2 / PX**2))
    sk = morphology.skeletonize(m)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3)), mode="constant") - 1
    branch = sk & (nb >= 3)
    branch = measure.label(morphology.binary_dilation(branch, morphology.disk(1))).max()
    tissue_um2 = valid.sum() * PX**2
    sk_um = sk.sum() * PX
    # orientation coherence from the structure tensor (0 = isotropic, 1 = all parallel)
    Axx, Axy, Ayy = feature.structure_tensor(cell_n, sigma=8, order="rc")
    w = m.astype(float)
    jxx, jxy, jyy = (Axx * w).sum(), (Axy * w).sum(), (Ayy * w).sum()
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + 1e-9)
    return m, {
        "map2_area_frac": m.sum() / max(valid.sum(), 1),
        "map2_skeleton_um_per_100um2": 100 * sk_um / max(tissue_um2, 1e-6),
        "map2_branchpoints_per_100um2": 100 * branch / max(tissue_um2, 1e-6),
        "map2_mean_width_um": (m.sum() * PX**2) / max(sk_um, 1e-6),
        "map2_orientation_coherence": float(coh),
        "map2_mean_norm_intensity_in_mask": float(cell_n[m].mean()) if m.any() else np.nan,
    }


# ---------------------------------------------------------------- somata (Cellpose)
def load_cellpose(use_gpu=True):
    from cellpose import models
    from importlib.metadata import version
    major = int(version("cellpose").split(".")[0])
    if major >= 4:
        return models.CellposeModel(gpu=use_gpu)             # Cellpose-SAM ("cpsam")
    return models.CellposeModel(gpu=use_gpu, model_type="cyto3")


def segment_somata(model, cell_raw, tissue, excl):
    small = transform.downscale_local_mean(cell_raw.astype(np.float32), (DS, DS))
    out = model.eval(small, diameter=30, channel_axis=None)   # ~13 um soma at 0.44 um/px
    lab_small = out[0]
    lab = np.kron(lab_small, np.ones((DS, DS), dtype=lab_small.dtype))[: cell_raw.shape[0], : cell_raw.shape[1]]
    keep = np.zeros(lab.max() + 1, bool)
    for r in measure.regionprops(lab):
        a = r.area * PX**2
        rr, cc = r.coords[:, 0], r.coords[:, 1]
        aspect = r.major_axis_length / max(r.minor_axis_length, 1e-6)
        if MIN_SOMA_UM2 <= a <= MAX_SOMA_UM2 and aspect <= MAX_SOMA_ASPECT and tissue[rr, cc].mean() > 0.8 and excl[rr, cc].mean() < 0.3:
            keep[r.label] = True
    lab = np.where(keep[lab], lab, 0)
    from skimage.segmentation import relabel_sequential
    return relabel_sequential(lab)[0] if lab.any() else lab  # keep touching somata separate


# ---------------------------------------------------------------- pS129 puncta
def detect_puncta(prot_raw, tissue, excl, k=6.0):
    """High-pass + LoG-like spot detection (after the authors' RASP approach, simplified):
    DoG(1, 3 px), local maxima above median + k * MAD within tissue."""
    img = prot_raw.astype(np.float32)
    dog = filters.gaussian(img, 1, preserve_range=True) - filters.gaussian(img, 3, preserve_range=True)
    valid = tissue & ~excl
    v = dog[valid] if valid.sum() > 1000 else dog.ravel()
    med = np.median(v); mad = np.median(np.abs(v - med)) * 1.4826
    pk = feature.peak_local_max(dog, min_distance=3, threshold_abs=med + k * mad, labels=valid.astype(int))
    return pk, dog


# ---------------------------------------------------------------- per image
def process(sid, mipdir, model, outdir):
    cell = tifffile.imread(f"{mipdir}/{sid}_cell.tif")
    prot = tifffile.imread(f"{mipdir}/{sid}_protein.tif")
    tissue = tissue_mask(cell, prot)
    af = autofluor_mask(cell, prot, tissue) & tissue
    cell_n, _ = normalize(cell); prot_n, _ = normalize(prot)
    row = {
        "sample_id": sid,
        "qc_tissue_frac": tissue.mean(),
        "qc_autofluor_frac": af.sum() / max(tissue.sum(), 1),
        "qc_autofluor_n_objects": int(measure.label(af).max()),
        "qc_map2_sat_frac": float((cell >= SAT).mean()),
        "qc_ps129_sat_frac": float((prot >= SAT).mean()),
        "qc_map2_focus": focus_score(cell),
        "qc_map2_illum_cv": illumination_unevenness(cell, tissue),
        "qc_map2_p99_raw": float(np.percentile(cell, 99)),
        "qc_ps129_p99_raw": float(np.percentile(prot, 99)),
        "qc_ps129_background_raw": float(np.median(prot[tissue])) if tissue.any() else np.nan,
    }
    dend, dfeat = dendrite_features(cell_n, tissue, af)
    row.update(dfeat)

    somata = np.zeros(cell.shape, np.int32)
    soma_rows = []
    if model is not None:
        somata = segment_somata(model, cell, tissue, af)
        props = measure.regionprops_table(
            somata, intensity_image=np.stack([cell_n, prot_n], -1),
            properties=("label", "area", "perimeter", "eccentricity", "solidity",
                        "major_axis_length", "minor_axis_length", "intensity_mean"))
        sp = pd.DataFrame(props)
        if len(sp):
            sp = sp.rename(columns={"intensity_mean-0": "map2_mean_norm", "intensity_mean-1": "ps129_mean_norm"})
            sp["area_um2"] = sp["area"] * PX**2
            sp["perimeter_um"] = sp["perimeter"] * PX
            sp["major_axis_um"] = sp["major_axis_length"] * PX
            sp["minor_axis_um"] = sp["minor_axis_length"] * PX
            sp["aspect_ratio"] = sp["major_axis_length"] / np.maximum(sp["minor_axis_length"], 1e-6)
            sp["circularity"] = 4 * np.pi * sp["area"] / np.maximum(sp["perimeter"], 1) ** 2
            sp.insert(0, "sample_id", sid)
            soma_rows = sp.drop(columns=["area", "perimeter", "major_axis_length", "minor_axis_length"])
    n_soma = int(somata.max())
    row["soma_n"] = n_soma if model is not None else np.nan
    row["soma_per_10000um2"] = (1e4 * n_soma / max(tissue.sum() * PX**2, 1)) if model is not None else np.nan
    if len(soma_rows):
        for c in ["area_um2", "eccentricity", "aspect_ratio", "solidity", "circularity", "map2_mean_norm", "ps129_mean_norm"]:
            row[f"soma_mean_{c}"] = soma_rows[c].mean()

    pk, _ = detect_puncta(prot, tissue, af)
    tissue_um2 = (tissue & ~af).sum() * PX**2
    row["ps129_puncta_n"] = len(pk)
    row["ps129_puncta_per_100um2"] = 100 * len(pk) / max(tissue_um2, 1e-6)
    if len(pk):
        r, c = pk[:, 0], pk[:, 1]
        row["ps129_puncta_mean_norm_intensity"] = float(prot_n[r, c].mean())
        row["ps129_puncta_frac_in_map2"] = float(dend[r, c].mean())
        row["ps129_puncta_frac_in_somata"] = float((somata[r, c] > 0).mean()) if model is not None else np.nan
        if len(soma_rows):
            counts = np.bincount(somata[r, c], minlength=somata.max() + 1)
            soma_rows["ps129_puncta_n"] = counts[soma_rows["label"].values]
            soma_rows["ps129_puncta_per_100um2"] = 100 * soma_rows["ps129_puncta_n"] / soma_rows["area_um2"]

    # save masks + QC overlay
    os.makedirs(f"{outdir}/masks", exist_ok=True); os.makedirs(f"{outdir}/qc", exist_ok=True)
    tifffile.imwrite(f"{outdir}/masks/{sid}_somata.tif", somata.astype(np.uint16))
    tifffile.imwrite(f"{outdir}/masks/{sid}_map2.tif", dend.astype(np.uint8) * 255)
    tifffile.imwrite(f"{outdir}/masks/{sid}_excluded.tif", ((~tissue) | af).astype(np.uint8) * 255)
    qc_figure(sid, cell, prot, tissue, af, dend, somata, pk, row, f"{outdir}/qc/{sid}.png")
    return row, soma_rows


def qc_figure(sid, cell, prot, tissue, af, dend, somata, pk, row, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    c, p = rescale(cell), rescale(prot)
    f, ax = plt.subplots(1, 3, figsize=(18, 6.4))
    ax[0].imshow(c, cmap="gray"); ax[0].set_title(f"{sid} MAP2 (raw, rescaled)")
    ov = np.dstack([c * 0.6] * 3)
    ov[dend] = [0.1, 0.8, 1.0]
    ov[find_boundaries(somata)] = [1, 1, 0]
    ov[af] = ov[af] * 0.4 + np.array([1, 0, 1]) * 0.6
    ov[~tissue] = ov[~tissue] * 0.4 + np.array([1, 0.3, 0]) * 0.6
    ax[1].imshow(ov); ax[1].set_title(f"cyan MAP2 mask | yellow somata (n={row.get('soma_n')}) | magenta autofluor | orange no tissue", fontsize=9)
    ax[2].imshow(p, cmap="gray")
    if len(pk): ax[2].scatter(pk[:, 1], pk[:, 0], s=6, facecolors="none", edgecolors="r", linewidths=0.6)
    ax[2].set_title(f"pS129 + puncta (n={row['ps129_puncta_n']})")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=70); plt.close(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mip", default="data/mip")
    ap.add_argument("--manifest", default="idr0173_sample_manifest.csv")
    ap.add_argument("--out", default="features")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", nargs="*", help="only these sample ids, e.g. S01 S35")
    ap.add_argument("--no-cellpose", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force Cellpose onto the CPU")
    ap.add_argument("--max-aspect", type=float, default=1.6,
                    help="drop Cellpose objects more elongated than this (vessels); default 1.6")
    a = ap.parse_args()

    ids = sorted({os.path.basename(p).split("_")[0] for p in glob.glob(f"{a.mip}/*_cell.tif")})
    if a.ids: ids = [i for i in ids if i in a.ids]
    if a.limit: ids = ids[: a.limit]
    global MAX_SOMA_ASPECT
    MAX_SOMA_ASPECT = a.max_aspect
    model = None if a.no_cellpose else load_cellpose(use_gpu=not a.cpu)
    os.makedirs(a.out, exist_ok=True)

    rows, somas = [], []
    for i, sid in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] {sid}", flush=True)
        r, s = process(sid, a.mip, model, a.out)
        rows.append(r)
        if len(s): somas.append(s)
    df = pd.DataFrame(rows)
    if os.path.exists(a.manifest):
        man = pd.read_csv(a.manifest)[["sample_id", "disease", "patient", "braak", "oligomer_count", "oligomer_intensity"]]
        df = man.merge(df, on="sample_id", how="right")
    df.to_csv(f"{a.out}/per_image.csv", index=False)
    if somas: pd.concat(somas).to_csv(f"{a.out}/per_soma.csv", index=False)
    print(f"Wrote {a.out}/per_image.csv ({len(df)} images, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
