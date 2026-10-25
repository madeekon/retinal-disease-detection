"""Dataset audit for the ODIR-5K "Ocular Disease Recognition" export.

The audit answers one question before any model is trained:
"Which image files can be used, with which image-level label, for which patient?"

Key decisions (see README):
* One row of full_df.csv describes ONE image (column `filename`).
* The image-level label comes from `labels` / `target` (they must agree).
* The N/D/G/C/A/H/M/O columns are PATIENT-level flags (both eyes combined) and are
  never used as labels. They are only compared with the image labels for reporting.
* Diagnostic keywords are used only inside this audit for label quality control.
  They are never written to the training manifests.
* Images are matched by filename; the patient ID encoded in the filename must equal
  the `ID` column.
"""

from __future__ import annotations

import ast
import hashlib
import io
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

log = logging.getLogger("retina")

CLASS_CODES = ["N", "D", "G", "C", "A", "H", "M", "O"]
CLASS_NAMES = {
    "N": "Normal", "D": "Diabetic retinopathy", "G": "Glaucoma", "C": "Cataract",
    "A": "AMD", "H": "Hypertensive retinopathy", "M": "Pathological myopia", "O": "Other abnormality",
}
REQUIRED_COLUMNS = [
    "ID", "Patient Age", "Patient Sex", "Left-Fundus", "Right-Fundus",
    "Left-Diagnostic Keywords", "Right-Diagnostic Keywords",
    *CLASS_CODES, "filepath", "labels", "target", "filename",
]
PATIENT_COLUMNS = [
    "Patient Age", "Patient Sex", "Left-Fundus", "Right-Fundus",
    "Left-Diagnostic Keywords", "Right-Diagnostic Keywords", *CLASS_CODES,
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FILENAME_RE = re.compile(r"^(\d+)_(left|right)\.(jpe?g|png)$", re.IGNORECASE)
KEYWORD_SPLIT_RE = re.compile(r"[，,;；]")
NORMAL_TERM = "normal fundus"
# Keywords that describe image acquisition, not retinal findings.
NON_DIAGNOSTIC_TERMS = {
    "lens dust", "low image quality", "image offset", "optic disk photographically invisible",
    "no fundus image", "anterior segment image",
}


# -----------------------------------------------------------------------------
# 1. Tabular checks
# -----------------------------------------------------------------------------
def load_full_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"full_df.csv not found: {path}")
    return pd.read_csv(path)


def check_schema(df: pd.DataFrame) -> dict[str, Any]:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"full_df.csv is missing required columns: {missing_cols}")
    per_patient = df.groupby("ID").size()
    constant = df.groupby("ID")[PATIENT_COLUMNS].nunique(dropna=False).le(1).all(axis=1)
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "missing_values_per_column": df.isna().sum().to_dict(),
        "n_fully_duplicated_rows": int(df.duplicated().sum()),
        "n_unique_patient_ids": int(df["ID"].nunique()),
        "n_unique_filenames": int(df["filename"].nunique()),
        "rows_per_patient_distribution": per_patient.value_counts().sort_index().to_dict(),
        "patients_with_inconsistent_patient_fields": int((~constant).sum()),
        "filepath_directories": df["filepath"].astype(str).str.rsplit("/", n=1).str[0].value_counts().to_dict(),
        "interpretation": (
            "Each row is one image. Patient-level columns (age, sex, both keywords, N..O flags) "
            "are repeated on the left- and right-eye rows of the same patient."
        ),
    }


def _split_keywords(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []
    return [t.strip().lower() for t in KEYWORD_SPLIT_RE.split(text) if t.strip()]


def _parse_label(value: Any) -> tuple[str | None, str | None]:
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None, "labels_not_parseable"
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 1:
        return None, "labels_not_single_class"
    code = str(parsed[0]).strip().upper()
    if code not in CLASS_CODES:
        return None, "labels_unknown_class"
    return code, None


def _parse_target(value: Any) -> tuple[str | None, str | None]:
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None, "target_not_parseable"
    if not isinstance(parsed, (list, tuple)) or len(parsed) != len(CLASS_CODES):
        return None, "target_wrong_length"
    arr = np.asarray(parsed, dtype=float)
    if not (np.isin(arr, [0, 1]).all() and arr.sum() == 1):
        return None, "target_not_one_hot"
    return CLASS_CODES[int(arr.argmax())], None


def build_image_records(df: pd.DataFrame) -> pd.DataFrame:
    """Create one audited record per row of full_df.csv (= per labeled image)."""
    rows = []
    filename_counts = df["filename"].astype(str).value_counts()
    for idx, r in df.iterrows():
        issues: list[str] = []
        notes: list[str] = []
        filename = str(r["filename"]).strip()
        m = FILENAME_RE.match(filename)
        fn_pid, eye = (int(m.group(1)), m.group(2).lower()) if m else (None, None)
        if m is None:
            issues.append("filename_pattern_invalid")
        elif fn_pid != int(r["ID"]):
            issues.append("filename_patient_id_mismatch")
        if eye == "left" and filename != str(r["Left-Fundus"]).strip():
            issues.append("filename_not_equal_left_fundus")
        if eye == "right" and filename != str(r["Right-Fundus"]).strip():
            issues.append("filename_not_equal_right_fundus")
        if filename_counts.get(filename, 0) > 1:
            issues.append("duplicate_filename")
        if os.path.basename(str(r["filepath"])) != filename:
            notes.append("filepath_basename_differs")

        label_code, lab_err = _parse_label(r["labels"])
        target_code, tgt_err = _parse_target(r["target"])
        issues += [e for e in (lab_err, tgt_err) if e]
        if label_code and target_code and label_code != target_code:
            issues.append("labels_target_disagree")

        eye_kw = r["Left-Diagnostic Keywords"] if eye == "left" else r["Right-Diagnostic Keywords"]
        terms = set(_split_keywords(eye_kw))
        findings = terms - {NORMAL_TERM} - NON_DIAGNOSTIC_TERMS
        kw_conflict = None
        if label_code == "N" and findings:
            kw_conflict = "normal_label_with_findings_in_eye_keywords"
        elif label_code and label_code != "N" and NORMAL_TERM in terms:
            kw_conflict = "abnormal_label_with_normal_fundus_keyword"
        elif label_code and label_code != "N" and not findings:
            kw_conflict = "abnormal_label_without_findings_in_eye_keywords"
        if kw_conflict:
            issues.append(kw_conflict)
        if terms & NON_DIAGNOSTIC_TERMS:
            notes.append("acquisition_note:" + "|".join(sorted(terms & NON_DIAGNOSTIC_TERMS)))

        patient_flag_mismatch = bool(label_code and label_code != "N" and int(r[label_code]) == 0)
        if patient_flag_mismatch:
            notes.append("disease_type_not_flagged_at_patient_level")

        rows.append({
            "row_index": int(idx),
            "image_id": Path(filename).stem,
            "filename": filename,
            "patient_id": int(r["ID"]),
            "eye": eye,
            "label_code": label_code,
            "label_name": CLASS_NAMES.get(label_code) if label_code else None,
            "binary_target": (0 if label_code == "N" else 1) if label_code else None,
            "patient_age": r["Patient Age"],
            "patient_sex": r["Patient Sex"],
            # audit-only columns (never copied to training manifests)
            "audit_eye_keywords": eye_kw,
            "audit_patient_flags": "".join(c for c in CLASS_CODES if int(r[c]) == 1),
            "audit_patient_N_flag": int(r["N"]),
            "audit_patient_flag_mismatch": patient_flag_mismatch,
            "label_issues": ";".join(issues),
            "notes": ";".join(notes),
        })
    return pd.DataFrame(rows)


def label_report(records: pd.DataFrame) -> dict[str, Any]:
    valid = records[records["label_code"].notna()]
    n_normal_in_abnormal_patient = int(((valid["label_code"] == "N") & (valid["audit_patient_N_flag"] == 0)).sum())
    issue_counts: dict[str, int] = defaultdict(int)
    for s in records["label_issues"]:
        for item in filter(None, str(s).split(";")):
            issue_counts[item] += 1
    return {
        "image_level_class_counts": valid["label_code"].value_counts().reindex(CLASS_CODES, fill_value=0).to_dict(),
        "binary_counts": {
            "normal_0": int((valid["binary_target"] == 0).sum()),
            "abnormal_1": int((valid["binary_target"] == 1).sum()),
        },
        "patient_level_flag_sums_over_rows": {
            c: int(records["audit_patient_flags"].str.contains(c).sum()) for c in CLASS_CODES
        },
        "normal_images_of_patients_with_patient_N_flag_0": n_normal_in_abnormal_patient,
        "abnormal_images_whose_disease_type_is_not_flagged_at_patient_level": int(records["audit_patient_flag_mismatch"].sum()),
        "label_issue_counts": dict(issue_counts),
        "explanation": (
            "A patient with one diseased eye has N=0 at patient level, but the healthy eye still has "
            "image-level label N. Using patient flags as image labels would mislabel those healthy eyes."
        ),
    }


def _read_xlsx_stdlib(path: Path) -> pd.DataFrame:
    """Minimal reader for the first worksheet of an .xlsx file (used only if openpyxl is missing)."""
    import zipfile
    import xml.etree.ElementTree as ET

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.find("m:sheetData", ns).findall("m:row", ns):
        values: dict[int, Any] = {}
        for c in row.findall("m:c", ns):
            letters = re.match(r"[A-Z]+", c.get("r")).group(0)
            col = 0
            for ch in letters:
                col = col * 26 + (ord(ch) - 64)
            t, v = c.get("t"), c.find("m:v", ns)
            if t == "s" and v is not None:
                values[col - 1] = shared[int(v.text)]
            elif t == "inlineStr":
                values[col - 1] = "".join(x.text or "" for x in c.iter(f"{{{ns['m']}}}t"))
            elif v is not None:
                values[col - 1] = v.text
        rows.append(values)
    width = max(max(r) for r in rows if r) + 1
    table = [[r.get(i) for i in range(width)] for r in rows]
    out = pd.DataFrame(table[1:], columns=table[0])
    for col in out.columns:
        try:
            out[col] = pd.to_numeric(out[col])
        except (ValueError, TypeError):
            pass
    return out


def cross_check_xlsx(df: pd.DataFrame, xlsx_path: Path | None) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Compare full_df.csv with the original ODIR data.xlsx (patient level)."""
    if xlsx_path is None or not xlsx_path.exists():
        return {"status": "skipped", "reason": f"file not found: {xlsx_path}"}, None
    reader = "pandas/openpyxl"
    try:
        x = pd.read_excel(xlsx_path)
    except ImportError:
        x = _read_xlsx_stdlib(xlsx_path)
        reader = "stdlib zipfile/xml fallback (openpyxl not installed)"
    patients = df.drop_duplicates("ID")
    merged = patients.merge(x, on="ID", how="left", suffixes=("", "_xlsx"), indicator=True)
    mismatches = {
        c: int((merged[c].astype(str).str.strip() != merged[f"{c}_xlsx"].astype(str).str.strip()).sum())
        for c in PATIENT_COLUMNS if f"{c}_xlsx" in merged.columns
    }
    eyes = []
    for _, r in x.iterrows():
        for side in ("Left", "Right"):
            eyes.append({"patient_id": int(r["ID"]), "filename": str(r[f"{side}-Fundus"]).strip(),
                         "eye": side.lower(), "audit_eye_keywords": r[f"{side}-Diagnostic Keywords"]})
    eyes_df = pd.DataFrame(eyes)
    labeled = set(df["filename"].astype(str))
    eyes_df["has_image_level_label"] = eyes_df["filename"].isin(labeled)
    unl = eyes_df[~eyes_df["has_image_level_label"]]
    return {
        "status": "ok",
        "reader": reader,
        "xlsx_rows_patients": int(len(x)),
        "full_df_patients_missing_in_xlsx": int((merged["_merge"] != "both").sum()),
        "xlsx_patients_absent_from_full_df": int(len(set(x["ID"]) - set(df["ID"]))),
        "patient_field_mismatches": mismatches,
        "xlsx_eye_images": int(len(eyes_df)),
        "xlsx_eye_images_with_image_level_label": int(eyes_df["has_image_level_label"].sum()),
        "xlsx_eye_images_without_image_level_label": int(len(unl)),
        "top_keywords_of_unlabeled_eyes": unl["audit_eye_keywords"].value_counts().head(15).to_dict(),
    }, eyes_df


# -----------------------------------------------------------------------------
# 2. Image folder checks
# -----------------------------------------------------------------------------
def scan_image_dir(folder: Path | None) -> dict[str, list[Path]]:
    """Map basename -> list of paths for every image file under folder (recursive)."""
    index: dict[str, list[Path]] = defaultdict(list)
    if folder is None or not folder.exists():
        return index
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                index[name].append(Path(root) / name)
    return index


def inspect_image(path: Path, decode: bool = True, hashes: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {"readable": False, "read_error": None, "width": None, "height": None,
                           "mode": None, "mean_gray": None, "md5": None, "file_bytes": None}
    try:
        data = path.read_bytes()
        out["file_bytes"] = len(data)
        if hashes:
            out["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as img:
            out["width"], out["height"], out["mode"] = img.width, img.height, img.mode
            if decode:
                img.load()  # full decode: catches truncated or corrupt JPEG files
                small = img.convert("L")
                small.thumbnail((128, 128))
                out["mean_gray"] = float(np.asarray(small, dtype=np.float32).mean())
        out["readable"] = True
    except Exception as exc:  # noqa: BLE001 - any failure makes the image unusable
        out["read_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _progress(it, total: int, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(it, total=total, desc=desc, ncols=90)
    except ImportError:  # pragma: no cover
        return it


def inspect_many(paths: list[Path], decode: bool, hashes: bool, workers: int = 8,
                 desc: str = "images") -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(_progress(pool.map(lambda p: inspect_image(p, decode, hashes), paths), len(paths), desc))


def md5_many(paths: list[Path], workers: int = 8) -> list[str | None]:
    def _md5(p: Path) -> str | None:
        try:
            return hashlib.md5(p.read_bytes()).hexdigest()
        except OSError:
            return None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_md5, paths))


# -----------------------------------------------------------------------------
# 3. Legacy embeddings
# -----------------------------------------------------------------------------
LEGACY_RECIPE = "DenseNet121 IMAGENET1K_V1 .features -> adaptive_avg_pool2d (no ReLU), Resize(224,224)"


def _legacy_priority_label(row: pd.Series) -> str:
    names = {"N": "Normal", "D": "Diabetes", "G": "Glaucoma", "C": "Cataract", "A": "AMD",
             "H": "Hypertension", "M": "Myopia", "O": "Other"}
    for c in CLASS_CODES:
        if int(row[c]) == 1:
            return names[c]
    return "Unknown"


def _legacy_encoder():
    import torch
    from torchvision import models, transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features.eval().to(device)
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return net, tf, device


def compute_legacy_descriptors(paths: list[Path], batch_size: int = 32) -> np.ndarray:
    """Re-create descriptors with the legacy recipe (used if legacy files are absent or unverified)."""
    import torch
    import torch.nn.functional as F

    net, tf, device = _legacy_encoder()
    out = []
    with torch.no_grad():
        for start in _progress(range(0, len(paths), batch_size), (len(paths) + batch_size - 1) // batch_size,
                               "Computing descriptors"):
            batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[start:start + batch_size]]).to(device)
            out.append(F.adaptive_avg_pool2d(net(batch), 1).flatten(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def audit_legacy_embeddings(df: pd.DataFrame, records: pd.DataFrame, emb_dir: Path | None,
                            n_verify: int, seed: int) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Check whether embeddings.npy rows can be mapped to images of full_df.csv."""
    if emb_dir is None or not (emb_dir / "embeddings.npy").exists():
        return {"status": "skipped", "reason": f"embeddings.npy not found in {emb_dir}", "verified": False}, None
    emb = np.load(emb_dir / "embeddings.npy", mmap_mode="r")
    rep: dict[str, Any] = {"status": "ok", "path": str(emb_dir), "shape": list(emb.shape),
                           "dtype": str(emb.dtype), "verified": False}
    meta_path = emb_dir / "metadata.csv"
    rep["metadata_columns"] = list(pd.read_csv(meta_path, nrows=1).columns) if meta_path.exists() else None
    rep["stored_filenames"] = bool(rep["metadata_columns"] and "filename" in rep["metadata_columns"])
    if emb.shape[0] != len(df):
        rep["row_count_matches_full_df"] = False
        rep["conclusion"] = "Row count differs from full_df.csv; alignment cannot be inferred. Do not use."
        return rep, None
    rep["row_count_matches_full_df"] = True
    sample = np.asarray(emb[: min(len(emb), 2000)], dtype=np.float32)
    rep["fraction_negative_values_first_2000_rows"] = float((sample < 0).mean())
    rep["note_on_negative_values"] = (
        "Negative values mean the vectors were pooled from densenet.features WITHOUT the final ReLU "
        "used by torchvision's DenseNet forward(). They differ from the classifier input of the "
        "pretrained network used in the CNN experiments."
    )
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        legacy = df.apply(_legacy_priority_label, axis=1).to_numpy()
        rep["metadata_label_equals_patient_priority_label_in_row_order"] = float((meta["label"].to_numpy() == legacy).mean())
        if "Patient Age" in meta.columns:
            rep["metadata_age_equals_full_df_age_in_row_order"] = float(
                (meta["Patient Age"].to_numpy() == df["Patient Age"].to_numpy()).mean())
        legacy_binary = (meta["label"].to_numpy() != "Normal").astype(int)
        image_binary = records["binary_target"].fillna(-1).astype(int).to_numpy()
        rep["legacy_vs_image_level_binary_disagreements"] = int((legacy_binary != image_binary).sum())

    rep["reencode_check"] = {"status": "skipped"}
    if n_verify > 0:
        try:
            usable = records[records["readable"]].reset_index(drop=True)
            rep["reencode_check"] = _reencode_check(emb, usable, n_verify, seed)
        except Exception as exc:  # noqa: BLE001
            rep["reencode_check"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    rc = rep["reencode_check"]
    if rc.get("status") == "ok" and rc.get("aligned"):
        rep["verified"] = True
        rep["conclusion"] = ("Rows are aligned with full_df.csv row order (verified by re-encoding). "
                             "Labels stored with them are patient-level and must not be used.")
    elif rc.get("status") == "ok":
        rep["conclusion"] = "Re-encoding did NOT reproduce stored vectors. Do not use legacy embeddings."
    else:
        rep["conclusion"] = ("Alignment only inferred from labels/ages in row order (re-encoding not run). "
                             "Prefer re-extracting embeddings keyed by filename.")
    index = records[["row_index", "filename", "patient_id", "binary_target"]].rename(columns={"row_index": "embedding_row"})
    return rep, index


def _reencode_check(emb: np.ndarray, records: pd.DataFrame, n: int, seed: int) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    rows = rng.choice(len(records), size=min(n, len(records)), replace=False)
    net, tf, device = _legacy_encoder()
    stored = np.asarray(emb, dtype=np.float32)
    stored_n = stored / (np.linalg.norm(stored, axis=1, keepdims=True) + 1e-12)
    matched, best_other, argmax_ok = [], [], 0
    with torch.no_grad():
        for i in rows:
            rec = records.iloc[i]
            x = tf(Image.open(rec["image_path"]).convert("RGB")).unsqueeze(0).to(device)
            v = F.adaptive_avg_pool2d(net(x), 1).flatten().cpu().numpy()
            v = v / (np.linalg.norm(v) + 1e-12)
            sims = stored_n @ v
            row = int(rec["row_index"])
            matched.append(float(sims[row]))
            best_other.append(float(np.delete(sims, row).max()))
            argmax_ok += int(int(np.argmax(sims)) == row)
    aligned = min(matched) > 0.99 and argmax_ok == len(rows)
    return {"status": "ok", "n_checked": len(rows), "cosine_same_row_min": min(matched),
            "cosine_same_row_mean": float(np.mean(matched)),
            "cosine_best_other_row_max": max(best_other),
            "same_row_is_nearest_count": argmax_ok, "aligned": bool(aligned), "recipe": LEGACY_RECIPE}


# -----------------------------------------------------------------------------
# 4. Duplicates, near duplicates and split groups
# -----------------------------------------------------------------------------
def exact_duplicate_groups(records: pd.DataFrame) -> pd.DataFrame:
    cols = ["md5", "n_images", "n_patients", "cross_patient", "filenames", "label_codes", "binary_targets"]
    sub = records[records["md5"].notna()]
    groups = []
    for value, g in sub.groupby("md5"):
        if len(g) > 1:
            groups.append({"md5": value, "n_images": len(g), "n_patients": g["patient_id"].nunique(),
                           "cross_patient": g["patient_id"].nunique() > 1,
                           "filenames": ";".join(g["filename"]),
                           "label_codes": ";".join(g["label_code"].astype(str)),
                           "binary_targets": ";".join(g["binary_target"].astype(str))})
    return pd.DataFrame(groups, columns=cols)


NEAR_DUPLICATE_COLUMNS = [
    "filename_a", "filename_b", "patient_a", "patient_b", "cosine", "label_a", "label_b", "binary_a", "binary_b",
    "age_a", "age_b", "sex_a", "sex_b", "merge_into_one_split_group", "same_retina_high_confidence",
    "binary_label_conflict"]


def find_near_duplicates(records: pd.DataFrame, desc: np.ndarray, review_threshold: float,
                         merge_threshold: float, identity_threshold: float, same_side_only: bool = True,
                         chunk: int = 1024) -> pd.DataFrame:
    """Cross-patient image pairs whose descriptor cosine similarity is >= review_threshold.

    merge_threshold    : pairs at or above it are put into one split group (precaution against leakage).
    identity_threshold : pairs at or above it are treated as the same retina; a normal/abnormal label
                         disagreement within such a pair is a label conflict.
    same_side_only     : a left eye cannot be the same retina as a right eye, so mixed pairs are ignored.
    """
    side = records["eye"].to_numpy()
    dn = desc / (np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12)
    pid = records["patient_id"].to_numpy()
    pairs = []
    for start in range(0, len(dn), chunk):
        sims = dn[start:start + chunk] @ dn.T
        ii, jj = np.nonzero(sims >= review_threshold)
        for i_local, j in zip(ii, jj):
            i = start + i_local
            if j <= i or pid[i] == pid[j] or (same_side_only and side[i] != side[j]):
                continue
            pairs.append((i, j, float(sims[i_local, j])))
    cols = NEAR_DUPLICATE_COLUMNS
    rows = []
    for i, j, s in pairs:
        a, b = records.iloc[i], records.iloc[j]
        merge = s >= merge_threshold
        same = s >= identity_threshold
        rows.append({"filename_a": a["filename"], "filename_b": b["filename"],
                     "patient_a": int(a["patient_id"]), "patient_b": int(b["patient_id"]), "cosine": round(s, 5),
                     "label_a": a["label_code"], "label_b": b["label_code"],
                     "binary_a": a["binary_target"], "binary_b": b["binary_target"],
                     "age_a": a["patient_age"], "age_b": b["patient_age"],
                     "sex_a": a["patient_sex"], "sex_b": b["patient_sex"],
                     "merge_into_one_split_group": bool(merge),
                     "same_retina_high_confidence": bool(same),
                     "binary_label_conflict": bool(same and a["binary_target"] != b["binary_target"])})
    return pd.DataFrame(rows, columns=cols).sort_values("cosine", ascending=False, ignore_index=True)


def assign_split_groups(records: pd.DataFrame, patient_links: list[tuple[int, int]]) -> pd.Series:
    """Union-find over patient IDs. Patients are merged when they share an identical image file
    or when a near-duplicate pair links them."""
    parent: dict[int, int] = {int(p): int(p) for p in records["patient_id"].unique()}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    for _, g in records[records["md5"].notna()].groupby("md5"):
        pids = [int(p) for p in g["patient_id"].unique()]
        for other in pids[1:]:
            union(pids[0], other)
    for a, b in patient_links:
        union(int(a), int(b))
    return records["patient_id"].map(lambda p: f"G{find(int(p))}")


def plot_pair_montage(pairs: pd.DataFrame, path_of: dict[str, str], out_path: Path, title: str,
                      max_pairs: int = 36, ncols: int = 6) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = pairs.head(max_pairs)
    if pairs.empty:
        return
    nblocks = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(2 * nblocks, ncols, figsize=(2.3 * ncols, 2.6 * 2 * nblocks), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (_, r) in enumerate(pairs.iterrows()):
        blk, col = divmod(k, ncols)
        for row_off, fname, lab, age, sex in ((0, r["filename_a"], r["label_a"], r["age_a"], r["sex_a"]),
                                              (1, r["filename_b"], r["label_b"], r["age_b"], r["sex_b"])):
            ax = axes[2 * blk + row_off, col]
            with Image.open(path_of[fname]) as im:
                ax.imshow(im.convert("RGB").resize((192, 192)))
            head = f"cos {r['cosine']:.4f}\n" if row_off == 0 else ""
            ax.set_title(f"{head}{fname} [{lab}] {age} {str(sex)[:1]}", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_size_inches()[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. Orchestration
# -----------------------------------------------------------------------------
def run_audit(cfg: dict[str, Any], paths) -> dict[str, Any]:
    acfg = cfg["audit"]
    policy = acfg["exclusion_policy"]
    ndcfg = acfg.get("near_duplicates", {"enabled": False})
    report: dict[str, Any] = {"paths": {
        "full_df_csv": paths.full_df_csv, "odir_xlsx": paths.odir_xlsx, "image_dir": paths.image_dir,
        "reference_image_dirs": paths.reference_image_dirs, "legacy_embeddings_dir": paths.legacy_embeddings_dir,
    }}
    steps = 8

    log.info("Step 1/%d  Reading %s", steps, paths.full_df_csv)
    df = load_full_df(paths.full_df_csv)
    report["schema"] = check_schema(df)
    log.info("          rows=%d  patients=%d  unique filenames=%d", report["schema"]["n_rows"],
             report["schema"]["n_unique_patient_ids"], report["schema"]["n_unique_filenames"])

    log.info("Step 2/%d  Building image-level labels from `labels` / `target`", steps)
    records = build_image_records(df)
    report["labels"] = label_report(records)
    log.info("          image-level classes: %s", report["labels"]["image_level_class_counts"])
    log.info("          normal eyes of patients with patient N flag = 0: %d",
             report["labels"]["normal_images_of_patients_with_patient_N_flag_0"])

    log.info("Step 3/%d  Cross-checking with data.xlsx", steps)
    report["xlsx_cross_check"], eyes_df = cross_check_xlsx(df, paths.odir_xlsx)
    log.info("          %s (%s)", report["xlsx_cross_check"].get("status"), report["xlsx_cross_check"].get("reader", ""))

    log.info("Step 4/%d  Scanning image folders", steps)
    primary = scan_image_dir(paths.image_dir)
    if not primary:
        raise FileNotFoundError(f"No image files found in primary image_dir: {paths.image_dir}")
    refs = {name: scan_image_dir(p) for name, p in paths.reference_image_dirs.items()}
    ambiguous = {k: len(v) for k, v in primary.items() if len(v) > 1}
    labeled_names = set(records["filename"])
    testing_names = set(refs.get("raw_testing_images", {}).keys())
    training_names = set(refs.get("raw_training_images", {}).keys())
    xlsx_unlabeled = set(eyes_df.loc[~eyes_df["has_image_level_label"], "filename"]) if eyes_df is not None else set()
    csv_referenced = set(df["Left-Fundus"].astype(str)) | set(df["Right-Fundus"].astype(str))

    def categorize(name: str) -> str:
        if name in labeled_names:
            return "labeled_in_full_df"
        if name in testing_names:
            return "odir_testing_image_no_label"
        if name in xlsx_unlabeled or name in csv_referenced or name in training_names:
            return "odir_training_image_without_image_level_label"
        return "unknown_origin"

    extra = pd.DataFrame([{"filename": n, "category": categorize(n), "n_copies": len(v), "path": str(v[0])}
                          for n, v in primary.items() if n not in labeled_names],
                         columns=["filename", "category", "n_copies", "path"])
    report["image_dir_scan"] = {
        "image_dir": paths.image_dir,
        "n_image_files": int(sum(len(v) for v in primary.values())),
        "n_unique_basenames": len(primary),
        "basenames_found_more_than_once": len(ambiguous),
        "files_without_image_level_label": int(len(extra)),
        "unlabeled_files_by_category": extra["category"].value_counts().to_dict(),
        "reference_dirs_file_counts": {name: int(sum(len(v) for v in idx.values())) for name, idx in refs.items()},
    }
    log.info("          primary dir files=%d  unlabeled files=%d", report["image_dir_scan"]["n_image_files"], len(extra))

    log.info("Step 5/%d  Matching labeled rows to image files by filename and decoding them", steps)
    records["image_found"] = records["filename"].map(lambda n: n in primary)
    records["image_path"] = records["filename"].map(lambda n: str(primary[n][0]) if n in primary else None)
    records["matched_filename_patient_id_ok"] = records.apply(
        lambda r: bool(r["image_found"]) and FILENAME_RE.match(Path(r["image_path"]).name) is not None
        and int(FILENAME_RE.match(Path(r["image_path"]).name).group(1)) == int(r["patient_id"]), axis=1)
    for name, idx in refs.items():
        records[f"in_{name}"] = records["filename"].map(lambda n, idx=idx: n in idx)

    found = records[records["image_found"]]
    stats_df = pd.DataFrame(inspect_many([Path(p) for p in found["image_path"]], decode=acfg["decode_images"],
                                         hashes=acfg["hash_images"], desc="Decoding labeled images"),
                            index=found.index)
    for col in stats_df.columns:
        records.loc[found.index, col] = stats_df[col]
    if "readable" not in records:
        records["readable"] = False
    records["readable"] = records["readable"].fillna(False).astype(bool)
    dark_thr = float(acfg["dark_image_mean_threshold"])
    records["flag_dark_image"] = records["mean_gray"].astype(float) < dark_thr

    ref_compare = {}
    pp = refs.get("preprocessed_images")
    if acfg.get("compare_reference_dirs") and acfg.get("hash_reference_preprocessed") and pp:
        both = records[records["image_found"] & records["filename"].map(lambda n: n in pp)]
        ref_md5 = md5_many([pp[n][0] for n in both["filename"]])
        same = (pd.Series(ref_md5, index=both.index) == both["md5"])
        ref_compare = {"labeled_images_present_in_both": int(len(both)),
                       "byte_identical_to_preprocessed_images": int(same.sum())}

    sizes = records.loc[records["readable"], ["width", "height"]].astype(int).astype(str).agg("x".join, axis=1)
    report["image_checks"] = {
        "labeled_rows": int(len(records)),
        "labeled_rows_with_image_found": int(records["image_found"].sum()),
        "labeled_rows_missing_image": int((~records["image_found"]).sum()),
        "matched_filename_patient_id_ok": int(records["matched_filename_patient_id_ok"].sum()),
        "readable_images": int(records["readable"].sum()),
        "unreadable_images": int((records["image_found"] & ~records["readable"]).sum()),
        "image_size_counts": sizes.value_counts().head(10).to_dict(),
        "image_mode_counts": records.loc[records["readable"], "mode"].value_counts().to_dict(),
        "dark_images_flagged": int(records["flag_dark_image"].sum()),
        "dark_images": records.loc[records["flag_dark_image"], "filename"].tolist(),
        "dark_threshold_mean_gray": dark_thr,
        "coverage_in_reference_dirs": {name: int(records[f"in_{name}"].sum()) for name in refs},
        "primary_vs_preprocessed_images": ref_compare,
    }
    log.info("          found=%d missing=%d unreadable=%d", report["image_checks"]["labeled_rows_with_image_found"],
             report["image_checks"]["labeled_rows_missing_image"], report["image_checks"]["unreadable_images"])

    log.info("Step 6/%d  Legacy DenseNet121 embeddings", steps)
    report["legacy_embeddings"], legacy_index = audit_legacy_embeddings(
        df, records, paths.legacy_embeddings_dir, int(acfg.get("verify_legacy_embeddings_n", 0)), int(cfg["seed"]))
    log.info("          %s", report["legacy_embeddings"].get("conclusion", report["legacy_embeddings"].get("reason")))

    log.info("Step 7/%d  Exact and near duplicates across patients", steps)
    exact = exact_duplicate_groups(records)
    near = pd.DataFrame(columns=NEAR_DUPLICATE_COLUMNS)
    nd_rep: dict[str, Any] = {"enabled": bool(ndcfg.get("enabled"))}
    if ndcfg.get("enabled"):
        usable = records[records["readable"]]
        desc, source = None, None
        if ndcfg.get("source", "legacy_if_verified") == "legacy_if_verified" and report["legacy_embeddings"].get("verified"):
            emb = np.load(paths.legacy_embeddings_dir / "embeddings.npy")
            desc, source = emb[usable["row_index"].to_numpy()].astype(np.float32), "verified legacy embeddings"
        else:
            try:
                desc = compute_legacy_descriptors([Path(p) for p in usable["image_path"]])
                source = "descriptors recomputed with the legacy recipe"
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Near-duplicate check failed (descriptors could not be computed: "
                    f"{type(exc).__name__}: {exc}). Install torch/torchvision or verify the legacy embeddings. "
                    "Disabling audit.near_duplicates would allow patient leakage between subsets.") from exc
        if desc is not None:
            near = find_near_duplicates(usable.reset_index(drop=True), desc,
                                        float(ndcfg["review_threshold"]), float(ndcfg["merge_threshold"]),
                                        float(ndcfg["identity_threshold"]), bool(ndcfg.get("same_side_only", True)))
            merged = near[near["merge_into_one_split_group"]]
            nd_rep.update({
                "descriptor_source": source, "descriptor_recipe": LEGACY_RECIPE,
                "review_threshold": float(ndcfg["review_threshold"]),
                "merge_threshold": float(ndcfg["merge_threshold"]),
                "identity_threshold": float(ndcfg["identity_threshold"]),
                "same_side_only": bool(ndcfg.get("same_side_only", True)),
                "pairs_at_or_above_review_threshold": int(len(near)),
                "pairs_merged": int(len(merged)),
                "patients_involved_in_merged_pairs": int(len(set(merged["patient_a"]) | set(merged["patient_b"]))),
                "pairs_same_retina_high_confidence": int(near["same_retina_high_confidence"].sum()),
                "same_retina_pairs_with_binary_label_conflict": int(near["binary_label_conflict"].sum()),
                "merged_pairs_with_different_8class_label": int((merged["label_a"] != merged["label_b"]).sum()),
                "review_only_pairs": int(len(near) - len(merged)),
            })
            path_of = dict(zip(records["filename"], records["image_path"]))
            plot_pair_montage(merged, path_of, paths.figures_dir / "near_duplicates_merged.png",
                              f"Cross-patient near duplicates merged into one split group (cos >= {ndcfg['merge_threshold']})")
            plot_pair_montage(near[~near["merge_into_one_split_group"]], path_of,
                              paths.figures_dir / "near_duplicates_review_band.png",
                              f"Pairs between review ({ndcfg['review_threshold']}) and merge ({ndcfg['merge_threshold']}) "
                              "threshold: NOT merged, for manual review")
    report["duplicates"] = {
        "exact_md5_groups": int(len(exact)),
        "exact_md5_groups_cross_patient": int(exact["cross_patient"].sum()) if len(exact) else 0,
        "near_duplicates": nd_rep,
    }
    log.info("          exact groups=%d  near-duplicate pairs merged=%s", len(exact), nd_rep.get("pairs_merged"))

    log.info("Step 8/%d  Exclusions and split groups", steps)
    conflict_files: set[str] = set()
    if policy.get("near_duplicate_label_conflict") and len(near):
        c = near[near["binary_label_conflict"]]
        conflict_files |= set(c["filename_a"]) | set(c["filename_b"])
    if policy.get("near_duplicate_label_conflict") and len(exact):
        for _, g in exact.iterrows():
            if len(set(g["binary_targets"].split(";"))) > 1:
                conflict_files |= set(g["filenames"].split(";"))

    invalid = {"filename_pattern_invalid", "filename_patient_id_mismatch", "filename_not_equal_left_fundus",
               "filename_not_equal_right_fundus", "labels_not_parseable", "labels_not_single_class",
               "labels_unknown_class", "target_not_parseable", "target_wrong_length",
               "target_not_one_hot", "labels_target_disagree"}
    conflicts = {"normal_label_with_findings_in_eye_keywords", "abnormal_label_with_normal_fundus_keyword",
                 "abnormal_label_without_findings_in_eye_keywords"}

    def reasons(r: pd.Series) -> str:
        out = []
        issues = set(filter(None, str(r["label_issues"]).split(";")))
        if policy.get("missing_image") and not r["image_found"]:
            out.append("missing_image")
        if policy.get("unreadable_image") and r["image_found"] and not r["readable"]:
            out.append("unreadable_image")
        if policy.get("invalid_label") and issues & invalid:
            out.append("invalid_label")
        if policy.get("duplicate_filename") and "duplicate_filename" in issues:
            out.append("duplicate_filename")
        if policy.get("contradictory_eye_keywords") and issues & conflicts:
            out.append("contradictory_eye_keywords")
        if r["filename"] in conflict_files:
            out.append("near_duplicate_label_conflict")
        if r["image_found"] and r["readable"] and not r["matched_filename_patient_id_ok"]:
            out.append("matched_file_patient_id_mismatch")
        return ";".join(out)

    records["exclusion_reasons"] = records.apply(reasons, axis=1)
    records["included"] = records["exclusion_reasons"] == ""
    clean = records[records["included"]].copy()
    links = []
    if len(near):
        m = near[near["merge_into_one_split_group"]]
        links = list(zip(m["patient_a"], m["patient_b"]))
    clean["split_group"] = assign_split_groups(clean, links)
    excl_counts: dict[str, int] = defaultdict(int)
    for s in records.loc[~records["included"], "exclusion_reasons"]:
        for item in s.split(";"):
            excl_counts[item] += 1
    report["exclusions"] = {"n_excluded": int((~records["included"]).sum()), "by_reason": dict(excl_counts),
                            "excluded_files": records.loc[~records["included"], "filename"].tolist(),
                            "policy": policy}
    group_sizes = clean.groupby("split_group")["patient_id"].nunique()
    report["clean_dataset"] = {
        "n_images": int(len(clean)),
        "n_patients": int(clean["patient_id"].nunique()),
        "n_split_groups": int(clean["split_group"].nunique()),
        "split_groups_with_more_than_one_patient": int((group_sizes > 1).sum()),
        "images_per_patient": clean.groupby("patient_id").size().value_counts().sort_index().to_dict(),
        "binary_counts": {"normal_0": int((clean["binary_target"] == 0).sum()),
                          "abnormal_1": int((clean["binary_target"] == 1).sum())},
        "abnormal_fraction": float(clean["binary_target"].mean()) if len(clean) else None,
        "class_counts": clean["label_code"].value_counts().reindex(CLASS_CODES, fill_value=0).to_dict(),
        "eye_counts": clean["eye"].value_counts().to_dict(),
    }
    log.info("          clean images=%d patients=%d groups=%d excluded=%d", len(clean), clean["patient_id"].nunique(),
             clean["split_group"].nunique(), report["exclusions"]["n_excluded"])

    # ---------------- write outputs ----------------
    out = paths.audit_dir
    records.to_csv(out / "labeled_images_audit.csv", index=False)
    records[~records["included"]].to_csv(out / "excluded_records.csv", index=False)
    extra.to_csv(out / "unlabeled_image_files.csv", index=False)
    exact.to_csv(out / "exact_duplicate_groups.csv", index=False)
    near.to_csv(out / "near_duplicate_pairs.csv", index=False)
    if eyes_df is not None:
        eyes_df.to_csv(out / "xlsx_all_eyes.csv", index=False)
    if legacy_index is not None:
        legacy_index.to_csv(out / "legacy_embeddings_index.csv", index=False)
    manifest_cols = ["image_id", "filename", "patient_id", "split_group", "eye", "label_code", "label_name",
                     "binary_target", "image_path", "width", "height", "md5", "flag_dark_image",
                     "patient_age", "patient_sex", "row_index"]
    clean[manifest_cols].to_csv(out / "clean_manifest.csv", index=False)
    return report


def summary_text(report: dict[str, Any]) -> str:
    """Human-readable summary of the most important audit numbers."""
    s, lab, img, cd = report["schema"], report["labels"], report["image_checks"], report["clean_dataset"]
    scan, ex, du, le = report["image_dir_scan"], report["exclusions"], report["duplicates"], report["legacy_embeddings"]
    nd = du.get("near_duplicates", {})
    lines = [
        "DATA AUDIT SUMMARY",
        "=" * 70,
        f"full_df.csv rows (one per image)        : {s['n_rows']}",
        f"Unique patient IDs                      : {s['n_unique_patient_ids']}",
        f"Rows per patient                        : {s['rows_per_patient_distribution']}",
        f"Fully duplicated rows                   : {s['n_fully_duplicated_rows']}",
        f"Patients with inconsistent fields       : {s['patients_with_inconsistent_patient_fields']}",
        "",
        "Image-level labels (labels/target)",
        f"  8-class counts                        : {lab['image_level_class_counts']}",
        f"  Binary counts                         : {lab['binary_counts']}",
        f"  Normal eyes in patients with N flag 0 : {lab['normal_images_of_patients_with_patient_N_flag_0']}",
        f"  Label issues                          : {lab['label_issue_counts'] or 'none'}",
        "",
        f"data.xlsx cross-check                   : {report['xlsx_cross_check'].get('status')} "
        f"({report['xlsx_cross_check'].get('reader', report['xlsx_cross_check'].get('reason', ''))})",
    ]
    if report["xlsx_cross_check"].get("status") == "ok":
        x = report["xlsx_cross_check"]
        lines += [f"  Patient field mismatches (total)      : {sum(x['patient_field_mismatches'].values())}",
                  f"  Eyes in xlsx with / without label     : {x['xlsx_eye_images_with_image_level_label']} / "
                  f"{x['xlsx_eye_images_without_image_level_label']}"]
    lines += [
        "",
        f"Primary image folder                    : {scan['image_dir']}",
        f"  Image files in folder                 : {scan['n_image_files']}",
        f"  Files without image-level label       : {scan['files_without_image_level_label']} "
        f"{scan['unlabeled_files_by_category']}",
        f"  Labeled rows matched to a file        : {img['labeled_rows_with_image_found']} / {img['labeled_rows']}",
        f"  Missing / unreadable                  : {img['labeled_rows_missing_image']} / {img['unreadable_images']}",
        f"  Image sizes                           : {img['image_size_counts']}",
        f"  Dark images flagged (kept)            : {img['dark_images_flagged']} {img.get('dark_images', '')}",
        f"  Coverage in reference folders         : {img['coverage_in_reference_dirs']}",
        f"  Primary vs preprocessed_images        : {img['primary_vs_preprocessed_images']}",
        "",
        f"Exact duplicate groups (MD5)            : {du['exact_md5_groups']} (cross-patient {du['exact_md5_groups_cross_patient']})",
    ]
    if nd.get("enabled") and "pairs_merged" in nd:
        lines += [
            f"Near-duplicate pairs (cos >= {nd['review_threshold']})      : {nd['pairs_at_or_above_review_threshold']}",
            f"  Merged into one split group (>= {nd['merge_threshold']}) : {nd['pairs_merged']} pairs, "
            f"{nd['patients_involved_in_merged_pairs']} patients",
            f"  Same retina, high confidence (>= {nd['identity_threshold']}): {nd['pairs_same_retina_high_confidence']} pairs, "
            f"{nd['same_retina_pairs_with_binary_label_conflict']} with normal/abnormal conflict (excluded)",
            f"  Merged pairs with other 8-class label : {nd['merged_pairs_with_different_8class_label']}",
            f"  Descriptor source                     : {nd['descriptor_source']}",
        ]
    elif nd.get("enabled"):
        lines.append(f"Near-duplicate check                    : FAILED {nd.get('error')}")
    lines += [
        f"Excluded records                        : {ex['n_excluded']} {ex['by_reason']}",
        "",
        "CLEAN DATASET",
        f"  Images                                : {cd['n_images']}",
        f"  Patients / split groups               : {cd['n_patients']} / {cd['n_split_groups']} "
        f"({cd['split_groups_with_more_than_one_patient']} groups contain >1 patient ID)",
        f"  Binary counts                         : {cd['binary_counts']}  (abnormal fraction {cd['abnormal_fraction']:.4f})",
        f"  8-class counts                        : {cd['class_counts']}",
        "",
        f"Legacy embeddings                       : {le.get('status')} {le.get('shape', '')}",
        f"  Conclusion                            : {le.get('conclusion', le.get('reason', ''))}",
    ]
    if le.get("reencode_check", {}).get("status") == "ok":
        rc = le["reencode_check"]
        lines.append(f"  Re-encode check                       : n={rc['n_checked']} min cos same row="
                     f"{rc['cosine_same_row_min']:.5f}, max cos other rows={rc['cosine_best_other_row_max']:.5f}")
    if "legacy_vs_image_level_binary_disagreements" in le:
        lines.append(f"  Legacy vs image-level binary disagree  : {le['legacy_vs_image_level_binary_disagreements']}")
    return "\n".join(lines)
