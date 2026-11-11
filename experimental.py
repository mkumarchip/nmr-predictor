import io
import math
import re
import tempfile
import zipfile
from typing import Any

from multiplicity import CARBON_TOLERANCE, PROTON_TOLERANCE


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?")
MNOVA_BLOCK_RE = re.compile(
    r"(?is)(?P<nucleus>(?:\^?1\s*H)|(?:\^?13\s*C))\s*NMR\s*"
    r"(?:\((?P<meta>[^)]*)\))?\s*(?:δ|delta)?\s*(?P<body>.*?)"
    r"(?=(?:\^?1\s*H|\^?13\s*C)\s*NMR|\Z)"
)
SHIFT_RE = re.compile(
    r"^\s*(?:δ\s*)?(?P<first>[-+]?\d+(?:\.\d+)?)"
    r"(?:\s*(?:-|–|—|to)\s*(?P<second>[-+]?\d+(?:\.\d+)?))?\s*(?:ppm)?",
    re.IGNORECASE,
)
MULTIPLICITY_RE = re.compile(
    r"(?i)(?<![A-Za-z])(?:br\s*)?(?:s|d|t|q|quint|sext|sept|m|dd|dt|td|dq|qd|tt|ddd|dddd)(?![A-Za-z])"
)
MNOVA_UNSUPPORTED_MESSAGE = (
    "Native .mnova files aren't supported — in Mnova, export to JCAMP-DX (.jdx) "
    "or copy the Multiplet Analysis report text, then import that."
)


def _numbers(line: str) -> list[float]:
    return [float(value) for value in NUMBER_RE.findall(line)]


def _normalize_report_text(text: str) -> str:
    return (
        text.replace("¹³C", "13C")
        .replace("¹H", "1H")
        .replace("^13C", "13C")
        .replace("^1H", "1H")
        .replace("\u00a0", " ")
    )


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]" and depth:
            depth -= 1
        elif char in ",;\n" and depth == 0:
            part = text[start:index].strip().strip(".;")
            if part:
                parts.append(part)
            start = index + 1
    tail = text[start:].strip().strip(".;")
    if tail:
        parts.append(tail)
    return parts


def _parse_metadata(meta: str) -> dict[str, Any]:
    frequency = None
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*MHz", meta, re.IGNORECASE)
    if match:
        frequency = float(match.group(1))
    fields = [field.strip() for field in meta.split(",") if field.strip()]
    solvent = next((field for field in fields if "MHz" not in field), None)
    return {"frequency_mhz": frequency, "solvent": solvent}


def _parse_mnova_peak(segment: str, nucleus: str) -> dict[str, Any] | None:
    match = SHIFT_RE.match(segment)
    if not match:
        return None
    first = float(match.group("first"))
    second = float(match.group("second")) if match.group("second") else None
    shift = (first + second) / 2.0 if second is not None else first
    peak: dict[str, Any] = {"shift_ppm": round(shift, 6)}
    if second is not None:
        peak["range_ppm"] = [round(min(first, second), 6), round(max(first, second), 6)]

    details_match = re.search(r"\((.*)\)", segment, re.DOTALL)
    details = details_match.group(1).strip() if details_match else ""
    if not details:
        return peak

    multiplicity = MULTIPLICITY_RE.search(details)
    if multiplicity:
        peak["multiplicity"] = re.sub(r"\s+", "", multiplicity.group(0).lower())
    j_match = re.search(r"(?i)\bJ(?:\s*[A-Za-z0-9-]+)?\s*=\s*([^;)]*?)\s*Hz\b", details)
    if j_match:
        peak["J_hz"] = [abs(value) for value in _numbers(j_match.group(1))]
    integration = re.search(r"(?i)(\d+(?:\.\d+)?)\s*H\b", details)
    if integration and nucleus == "H":
        value = float(integration.group(1))
        peak["integration"] = int(value) if value.is_integer() else value
    return peak


def _parse_mnova_report(text: str, requested_nucleus: str) -> dict[str, Any] | None:
    normalized = _normalize_report_text(text)
    blocks: dict[str, dict[str, Any]] = {}
    for match in MNOVA_BLOCK_RE.finditer(normalized):
        nucleus = "C" if "13" in re.sub(r"\s+", "", match.group("nucleus")) else "H"
        metadata = _parse_metadata(match.group("meta") or "")
        peaks = []
        for segment in _split_top_level(match.group("body")):
            peak = _parse_mnova_peak(segment, nucleus)
            if peak:
                peaks.append(peak)
        if not peaks:
            continue
        block = blocks.setdefault(
            nucleus,
            {"nucleus": nucleus, "peaks": [], "trace": None, **metadata},
        )
        block["peaks"].extend(peaks)
        block["peaks"].sort(key=lambda row: row["shift_ppm"], reverse=True)

    if not blocks:
        return None
    nucleus = requested_nucleus if requested_nucleus in blocks else ("H" if "H" in blocks else "C")
    selected = dict(blocks[nucleus])
    selected.update(
        {
            "format": "mnova-report",
            "parser": "Mnova Multiplet Analysis",
            "spectra": blocks,
        }
    )
    return selected


def _normalize_trace(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    pairs = sorted(zip(x, y), reverse=True)
    if not pairs:
        return [], []
    xs, ys = zip(*pairs)
    scale = max((abs(value) for value in ys), default=0.0) or 1.0
    return [round(value, 6) for value in xs], [round(value / scale, 6) for value in ys]


def _prepare_continuous_trace(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    import numpy as np

    pairs = sorted(zip(x, y), reverse=True)
    if not pairs:
        return [], []
    xs, raw_y = zip(*pairs)
    values = np.asarray(raw_y, dtype=float)
    values -= np.median(values)
    if abs(float(values.min())) > abs(float(values.max())):
        values = -values
    scale = max((abs(float(value)) for value in values), default=0.0) or 1.0
    return (
        [round(float(value), 6) for value in xs],
        [round(float(value) / scale, 6) for value in values],
    )


def _local_maxima(x: list[float], y: list[float], nucleus: str) -> list[dict[str, float]]:
    if len(x) < 3:
        return []
    import numpy as np
    from scipy.signal import find_peaks

    values = np.asarray(y, dtype=float)
    values = values - np.median(values)
    if abs(float(values.min())) > abs(float(values.max())):
        values = -values
    maximum = float(values.max())
    if maximum <= 0:
        return []
    values /= maximum
    tolerance = PROTON_TOLERANCE if nucleus == "H" else CARBON_TOLERANCE
    resolution = abs(float(x[1]) - float(x[0])) or tolerance
    distance = max(1, int(round(tolerance / resolution)))
    indices, _ = find_peaks(values, height=0.05, prominence=0.02, distance=distance)
    peaks = [(float(x[index]), float(values[index])) for index in indices]
    return [
        {"shift_ppm": round(ppm, 6), "intensity": round(intensity, 6)}
        for ppm, intensity in sorted(peaks, reverse=True)
    ]


def _jcamp_nucleus(data: dict[str, Any], requested: str) -> str:
    if requested in {"H", "C"}:
        return requested
    observed = str(data.get(".observe nucleus", data.get("observe nucleus", ""))).upper()
    if "13" in observed and "C" in observed:
        return "C"
    if "1" in observed and "H" in observed:
        return "H"
    raise ValueError("JCAMP nucleus is missing; choose 1H or 13C before importing")


def _parse_jcamp(text: str, requested_nucleus: str) -> dict[str, Any]:
    try:
        import jcamp

        data = jcamp.read(io.StringIO(text))
        if "x" not in data or "y" not in data:
            raise ValueError("JCAMP contains no readable XY data")
        nucleus = _jcamp_nucleus(data, requested_nucleus)
        x = [float(value) for value in data["x"]]
        y = [float(value) for value in data["y"]]
        frequency = data.get(".observe frequency", data.get("observe frequency"))
        frequency = float(frequency) if frequency is not None else None
        x_units = str(data.get("xunits", "PPM")).strip().upper()
        if x_units in {"HZ", "HERTZ"}:
            if not frequency:
                raise ValueError("JCAMP x-axis is in Hz but observation frequency is missing")
            x = [value / frequency for value in x]
            x_units = "PPM"
        peak_mode = "peak table" in data or "peaktable" in data or "xypoints" in data
        x, y = _normalize_trace(x, y) if peak_mode else _prepare_continuous_trace(x, y)
        peaks = (
            [{"shift_ppm": ppm, "intensity": intensity} for ppm, intensity in zip(x, y)]
            if peak_mode
            else _local_maxima(x, y, nucleus)
        )
        return {
            "format": "jcamp",
            "parser": "jcamp 1.3.2",
            "nucleus": nucleus,
            "peaks": peaks,
            "trace": None if peak_mode else {"x": x, "y": y},
            "frequency_mhz": frequency,
            "x_units": x_units,
            "title": data.get("title"),
        }
    except ImportError as exc:
        raise ValueError("JCAMP import requires the jcamp package: pip install jcamp") from exc


def _jdf_nucleus(label: str, requested: str) -> str:
    if requested in {"H", "C"}:
        return requested
    normalized = re.sub(r"[^A-Z0-9]", "", label.upper())
    if "CARBON" in normalized or "13C" in normalized or "C13" in normalized:
        return "C"
    if "PROTON" in normalized or "1H" in normalized or normalized == "H":
        return "H"
    raise ValueError("JEOL nucleus could not be inferred; choose 1H or 13C before importing")


def _downsample_trace(x: list[float], y: list[float], max_points: int = 32768) -> tuple[list[float], list[float]]:
    if len(x) <= max_points:
        return x, y
    stride = math.ceil(len(x) / max_points)
    return x[::stride], y[::stride]


def _parse_jdf(data_bytes: bytes, requested_nucleus: str, filename: str) -> dict[str, Any]:
    if not data_bytes:
        raise ValueError("JEOL JDF upload is empty")
    try:
        import nmrglue as ng
        import numpy as np
    except ImportError as exc:
        raise ValueError("JEOL import requires the nmrglue JEOL development reader") from exc
    if not hasattr(ng, "jeol"):
        raise ValueError("installed nmrglue build has no JEOL reader; install nmrglue 0.12-dev")

    try:
        with tempfile.NamedTemporaryFile(suffix=".jdf") as handle:
            handle.write(data_bytes)
            handle.flush()
            dictionary, raw_data = ng.jeol.read(handle.name)
    except Exception as exc:
        raise ValueError(
            f"could not read JEOL JDF file; it may be invalid or use an unsupported JDF variant (reader detail: {exc})"
        ) from exc

    data = np.asarray(raw_data).squeeze()
    if data.ndim != 1:
        raise ValueError(f"JEOL import currently supports 1D spectra only; file has {data.ndim} dimensions")
    universal = ng.jeol.guess_udic(dictionary, data)
    axis = universal[0]
    nucleus = _jdf_nucleus(str(axis.get("label", "")), requested_nucleus)
    sweep_hz = float(axis.get("sw", 0.0))
    observe_mhz = float(axis.get("obs", 0.0))
    carrier_hz = float(axis.get("car", 0.0))
    if sweep_hz <= 0 or observe_mhz <= 0:
        raise ValueError("JEOL file is missing sweep-width or observation-frequency metadata")

    units = dictionary.get("header", {}).get("units", [])
    unit_base = str(units[0][2]).lower() if units and len(units[0]) >= 3 else "second"
    time_domain = unit_base == "second"
    phase_method = "not required"
    if time_domain:
        line_broadening_hz = 0.5 if nucleus == "H" else 1.0
        processed = ng.proc_base.em(data, lb=line_broadening_hz / sweep_hz)
        target_size = 1 << math.ceil(math.log2(max(2, processed.size * 2)))
        target_size = min(target_size, 262144)
        processed = ng.proc_base.zf_size(processed, target_size)
        spectrum = ng.proc_base.fft(processed)
        try:
            phased = ng.proc_autophase.autops(spectrum, "acme", disp=False)
            intensity = np.real(phased)
            phase_method = "ACME auto-phase"
        except Exception:
            intensity = np.abs(spectrum)
            phase_method = "magnitude fallback"
        intensity = intensity[::-1]
        conversion = ng.fileio.fileiobase.unit_conversion(
            intensity.size, False, sweep_hz, observe_mhz, carrier_hz
        )
        ppm = conversion.ppm_scale()
    else:
        intensity = np.real(data) if np.iscomplexobj(data) else data.astype(float)
        starts = dictionary.get("header", {}).get("data_axis_start", [0.0])
        stops = dictionary.get("header", {}).get("data_axis_stop", [float(data.size - 1)])
        ppm = np.linspace(float(starts[0]), float(stops[0]), intensity.size)
        if unit_base == "hertz":
            ppm = ppm / observe_mhz
        elif unit_base != "ppm":
            conversion = ng.fileio.fileiobase.unit_conversion(
                intensity.size, False, sweep_hz, observe_mhz, carrier_hz
            )
            ppm = conversion.ppm_scale()

    full_x, full_y = _prepare_continuous_trace(ppm.tolist(), np.asarray(intensity, dtype=float).tolist())
    peaks = _local_maxima(full_x, full_y, nucleus)
    trace_x, trace_y = _downsample_trace(full_x, full_y)
    return {
        "format": "jeol-jdf",
        "parser": "nmrglue JEOL 0.12-dev",
        "nucleus": nucleus,
        "peaks": peaks,
        "trace": {"x": trace_x, "y": trace_y},
        "frequency_mhz": round(observe_mhz, 6),
        "title": dictionary.get("header", {}).get("title") or filename,
        "auto_processed": True,
        "processing_note": (
            f"JEOL raw data auto-processed ({phase_method}); phasing and peak picks may need review."
            if time_domain
            else "JEOL processed spectrum imported automatically; peak picks may need review."
        ),
    }


def _parse_mnova_container(data_bytes: bytes, requested_nucleus: str) -> dict[str, Any]:
    if not data_bytes:
        raise ValueError(MNOVA_UNSUPPORTED_MESSAGE)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data_bytes))
    except (zipfile.BadZipFile, OSError):
        raise ValueError(MNOVA_UNSUPPORTED_MESSAGE) from None

    entries = archive.infolist()
    if len(entries) > 10000:
        archive.close()
        raise ValueError(MNOVA_UNSUPPORTED_MESSAGE)
    candidates = []
    for info in entries:
        name = info.filename.lower()
        if info.is_dir() or info.file_size > 20 * 1024 * 1024:
            continue
        if name.endswith((".jdx", ".dx", ".jcamp")):
            priority = 0
        elif name.endswith((".txt", ".csv")):
            priority = 1
        else:
            continue
        candidates.append((priority, info))

    for _, info in sorted(candidates, key=lambda item: (item[0], item[1].file_size)):
        try:
            payload = archive.read(info)
            text = payload.decode("utf-8", errors="replace")
            result = parse_experimental(text, requested_nucleus, info.filename)
            result["embedded_format"] = result["format"]
            result["format"] = "mnova-container"
            result["parser"] = f"Mnova embedded {result.get('parser', result['embedded_format'])}"
            result["embedded_file"] = info.filename
            archive.close()
            return result
        except Exception:
            continue
    archive.close()
    raise ValueError(MNOVA_UNSUPPORTED_MESSAGE)


def _parse_peak_list(text: str) -> dict[str, Any]:
    rows = [_numbers(line) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    rows = [row for row in rows if row]
    peaks: list[dict[str, float]] = []
    if len(rows) == 1 and len(rows[0]) > 2:
        peaks = [{"shift_ppm": value, "intensity": 1.0} for value in rows[0]]
    else:
        for row in rows:
            peaks.append({"shift_ppm": row[0], "intensity": row[1] if len(row) > 1 else 1.0})
    scale = max((abs(row["intensity"]) for row in peaks), default=0.0) or 1.0
    for row in peaks:
        row["shift_ppm"] = round(float(row["shift_ppm"]), 6)
        row["intensity"] = round(float(row["intensity"]) / scale, 6)
    peaks.sort(key=lambda row: row["shift_ppm"], reverse=True)
    return {"format": "peaks", "parser": "plain peak list", "peaks": peaks, "trace": None}


def parse_experimental(
    text: str, nucleus: str, filename: str = "", data_bytes: bytes | None = None
) -> dict[str, Any]:
    nucleus = nucleus.upper().strip()
    if nucleus not in {"H", "C", "AUTO"}:
        raise ValueError("nucleus must be H, C, or AUTO")
    extension = filename.lower()
    if extension.endswith(".mnova"):
        result = _parse_mnova_container(data_bytes or b"", nucleus)
        result["filename"] = filename or None
        return result
    if extension.endswith(".jdf"):
        result = _parse_jdf(data_bytes or b"", nucleus, filename)
        result["filename"] = filename or None
        return result
    if not text.strip():
        raise ValueError("experimental data is empty")
    mnova = _parse_mnova_report(text, nucleus)
    if mnova:
        mnova["filename"] = filename or None
        return mnova
    is_jcamp = "##TITLE=" in text.upper() or filename.lower().endswith((".dx", ".jdx"))
    if nucleus == "AUTO" and not is_jcamp:
        raise ValueError("could not detect a 1H or 13C Mnova report; choose a nucleus for a plain peak list")
    result = _parse_jcamp(text, nucleus) if is_jcamp else _parse_peak_list(text)
    if not result["peaks"] and not result.get("trace"):
        raise ValueError("no numeric spectrum data found")
    result.setdefault("nucleus", nucleus)
    result.update({"filename": filename or None})
    return result
