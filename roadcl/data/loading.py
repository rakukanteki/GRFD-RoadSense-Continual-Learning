from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from roadcl.config import AXES, CLASSES, DataConfig
from roadcl.utils.runtime import file_sha256

ID_COLUMNS = ["source_file", "segment_id", "seconds_elapsed"]


def _trace_signature(recording: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(recording[["seconds_elapsed", *AXES]], index=False).to_numpy()
    boundaries = recording.segment_id.ne(recording.segment_id.shift()).to_numpy()
    return hashlib.sha256(values.tobytes() + boundaries.tobytes()).hexdigest()


def _distinct_recordings(frame: pd.DataFrame, name: str, traces: list[dict]) -> list[str]:
    seen: dict[str, str] = {}
    keep = []
    for record, recording in frame.groupby("record", sort=True):
        signature = _trace_signature(recording)
        if signature not in seen:
            keep.append(record)
            seen[signature] = record
        traces.append({"record": record, "class_name": name, "retained_record": seen[signature],
                       "sensor_trace_sha256": signature, "rows": len(recording),
                       "segments": recording.segment_id.nunique()})
    return keep


def _check_groups_disjoint(frame: pd.DataFrame, valid: pd.Series, message: str) -> None:
    row_hash = pd.util.hash_pandas_object(frame.loc[valid, ["seconds_elapsed", *AXES]], index=False)
    if (frame.loc[valid, "source_group"].groupby(row_hash).nunique() > 1).any():
        raise ValueError(message)


def _apply_manifest(metadata: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Replace the filename proxy with verified session ids, one per record."""
    mapping = pd.read_csv(path, dtype=str)
    if mapping.record.duplicated().any() or mapping.session_id.isna().any():
        raise ValueError("the session manifest needs one nonempty session_id per record")
    sessions = metadata.record.map(mapping.set_index("record").session_id)
    if sessions.isna().any() or sessions.groupby(metadata.source_group).nunique().max() > 1:
        raise ValueError("the session manifest is incomplete or splits a known source group")
    metadata["source_group"] = sessions
    return metadata


def load_recordings(cfg: DataConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the per-class files, drop duplicate traces and return samples plus recording metadata."""
    parts, audit, traces = [], [], []
    for name in CLASSES:
        path = Path(cfg.data_dir) / cfg.files[name]
        if not path.is_file():
            raise FileNotFoundError(f"required input not found: {path}; set data.data_dir and data.files")
        frame = pd.read_csv(path, usecols=[*ID_COLUMNS, "label", *AXES])
        if not frame.label.eq(name).all():
            raise ValueError(f"label mismatch in the {name} file")
        if frame[ID_COLUMNS].isna().any().any():
            raise ValueError(f"missing recording identifiers or timestamps in the {name} file")

        frame["record"] = name + "/" + frame.source_file.astype(str)
        frame["source_group"] = frame.source_file.astype(str).replace(cfg.aliases)
        keep = _distinct_recordings(frame, name, traces)
        _check_groups_disjoint(frame, frame[AXES].notna().all(axis=1),
                               "identical sensor rows appear in distinct source groups")
        audit.append({
            "class_name": name,
            "filename": path.name,
            "original_rows": len(frame),
            "original_recordings": frame.record.nunique(),
            "distinct_traces": len(keep),
            "sha256": file_sha256(path),
            "nonfinite_sensor_rows": int((~np.isfinite(frame[AXES]).all(axis=1)).sum()),
        })
        parts.append(frame[frame.record.isin(keep)].copy())

    data = pd.concat(parts, ignore_index=True)
    _check_groups_disjoint(data, np.isfinite(data[AXES]).all(axis=1),
                           "identical timed sensor rows cross canonical source groups")
    metadata = (
        data[["record", "source_group", "label"]]
        .drop_duplicates()
        .sort_values("record")
        .reset_index(drop=True)
    )
    if cfg.session_manifest:
        metadata = _apply_manifest(metadata, cfg.session_manifest)
    if metadata.groupby("record").size().max() != 1:
        raise ValueError("recording ids are not unique")
    pd.DataFrame(traces).to_csv(out_dir / "trace_provenance.csv", index=False)
    pd.DataFrame(audit).to_csv(out_dir / "input_audit.csv", index=False)
    return data, metadata
