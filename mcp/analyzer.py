"""
MERRA2 data analysis tools.

Contains:
  - MERRA2Analyzer  — load, slice, plot, and analyse MERRA2/PRISM NetCDF data
  - get_dataset_metadata
  - list_available_files
"""

import json
import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import FancyBboxPatch
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    _CARTOPY_AVAILABLE = True
except Exception:
    ccrs = None
    cfeature = None
    shpreader = None
    _CARTOPY_AVAILABLE = False
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
import textwrap
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from config import PINS, _SCRIPT_DIR, _DEFAULT_CONFIG, allowed_roots_message, path_is_allowed, state_dir

class MERRA2Analyzer:
    """Analyze MERRA2 NetCDF data with xarray."""
    
    # Known data directories — preprocessed output from preproc_merra_prism.py
    #
    # These used to default to a stale path (/data/granite-wxc/.../merra_prism_LA_county)
    # that no longer exists on this host and doesn't match the current case_name
    # ("merra_prism_with_H"). That silently broke every analysis tool for anyone who
    # didn't have MERRA2_*_DATA_DIR set, even though the agent picked the right tool.
    # Fixed to derive from _SCRIPT_DIR (the actual repo root, config.py / GRANITE_WXC_SCRIPT_DIR)
    # and the real case_name subdirectory.
    _PREPROCESSED_ROOT = _SCRIPT_DIR / "preprocessed"
    _DEFAULT_INFERENCE_DIR = Path(
        os.getenv(
            "MERRA2_INFERENCE_DATA_DIR",
            str(_SCRIPT_DIR / "experiments" / "inference_output" / "merra_prism_with_H"),
        )
    )
    _DEFAULT_TRAINING_DIR = Path(
        os.getenv(
            "MERRA2_TRAINING_DATA_DIR",
            str(_SCRIPT_DIR / "preprocessed" / "training" / "merra_prism_with_H"),
        )
    )
    DATA_DIRS = {
        "inference": str(_DEFAULT_INFERENCE_DIR),
        "training":  str(_DEFAULT_TRAINING_DIR),
    }
    
    # Year threshold: before 2016 = training, 2016+ = inference
    YEAR_THRESHOLD = 2016
    ARTIFACTS_DIR = state_dir() / "artifacts"
    DEFAULT_REPO = PINS["code"]["repo"].removeprefix("https://github.com/").removesuffix(".git")
    DEFAULT_REF = PINS["code"]["variants"][PINS["code"]["default_variant"]]["commit"]
    DEFAULT_PATH = "examples/MERRA_PRISM"

    # Named region presets — (lat_min, lat_max, lon_min, lon_max)
    KNOWN_REGIONS: Dict[str, Tuple[float, float, float, float]] = {
        "la_county":            (33.3, 34.9, -119.0, -117.6),
        "los_angeles_county":   (33.3, 34.9, -119.0, -117.6),
        "los_angeles":          (33.3, 34.9, -119.0, -117.6),
        "la":                   (33.3, 34.9, -119.0, -117.6),
        "socal":                (32.5, 35.5, -121.0, -116.5),
        "southern_california":  (32.5, 35.5, -121.0, -116.5),
        "california":           (32.5, 42.0, -124.5, -114.0),
    }

    @classmethod
    def _resolve_region(
        cls,
        region: Optional[str],
        lat_min: Optional[float],
        lat_max: Optional[float],
        lon_min: Optional[float],
        lon_max: Optional[float],
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """If a named region is given and no explicit bounds are set, look up the preset bounds."""
        if region is not None and all(x is None for x in (lat_min, lat_max, lon_min, lon_max)):
            key = region.lower().replace(" ", "_").replace("-", "_")
            if key in cls.KNOWN_REGIONS:
                lat_min, lat_max, lon_min, lon_max = cls.KNOWN_REGIONS[key]
        return lat_min, lat_max, lon_min, lon_max

    
    def __init__(self):
        self.ds = None
        self.filepath = None
        self.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _http_get_text(url: str, timeout: int = 30) -> str:
        """Fetch text content over HTTP using the standard library."""
        req = urlrequest.Request(
            url,
            headers={
                "User-Agent": "merra2-agent/1.0",
                "Accept": "application/vnd.github+json",
            },
        )
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")

    @staticmethod
    def _extract_snippets(content: str, tokens: List[str], max_snippets: int = 3, context: int = 2) -> List[str]:
        """Extract concise code/document snippets around token matches."""
        lines = content.splitlines()
        if not lines:
            return []

        matches: List[int] = []
        lowered = [ln.lower() for ln in lines]
        for idx, line in enumerate(lowered):
            if any(tok in line for tok in tokens):
                matches.append(idx)

        if not matches:
            preview = "\n".join(lines[: min(20, len(lines))])
            return [preview]

        snippets: List[str] = []
        used_ranges: List[Tuple[int, int]] = []
        for idx in matches:
            if len(snippets) >= max_snippets:
                break
            start = max(0, idx - context)
            end = min(len(lines), idx + context + 1)
            overlaps = any(not (end <= s or start >= e) for s, e in used_ranges)
            if overlaps:
                continue
            used_ranges.append((start, end))
            block = "\n".join(lines[start:end])
            snippets.append(block)

        return snippets

    def github_repo_context(
        self,
        query: str,
        repo: str = DEFAULT_REPO,
        ref: str = DEFAULT_REF,
        path_prefix: str = DEFAULT_PATH,
        max_files: int = 6,
    ) -> str:
        """Search the local granite-wxc repo for relevant snippets grounded in the query.

        The ``repo``, ``ref``, and ``path_prefix`` parameters are accepted for
        interface compatibility but are ignored — the local checkout at
        ``_SCRIPT_DIR`` is always used as the source of truth.
        """
        query = (query or "").strip()
        if not query:
            return json.dumps({"status": "error", "message": "query is required"})

        tokens = [
            t for t in re.findall(r"[a-zA-Z0-9_]{3,}", query.lower())
            if t not in {"what", "which", "with", "from", "that", "this", "then", "into"}
        ]
        if not tokens:
            tokens = [query.lower()]

        allowed_ext = (".md", ".yaml", ".yml", ".py", ".json", ".txt")
        search_root = _SCRIPT_DIR

        # Walk the local directory tree and score each candidate file
        blobs: List[Tuple[int, Path]] = []
        try:
            for fpath in search_root.rglob("*"):
                if not fpath.is_file():
                    continue
                if fpath.suffix.lower() not in allowed_ext:
                    continue
                # Skip hidden dirs / __pycache__ / .git
                if any(part.startswith((".","__pycache__")) for part in fpath.parts):
                    continue
                rel = str(fpath.relative_to(search_root))
                path_score = sum(3 for tok in tokens if tok in rel.lower())
                if fpath.name.lower() == "readme.md":
                    path_score += 2
                if fpath.suffix.lower() in (".yaml", ".yml"):
                    path_score += 1
                blobs.append((path_score, fpath))
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Failed to walk local repo: {e}"})

        if not blobs:
            return json.dumps({
                "status": "error",
                "message": f"No candidate files found under {search_root}",
            })

        blobs.sort(key=lambda x: x[0], reverse=True)
        candidates = [fp for _, fp in blobs[: max(max_files * 4, 12)]]

        highlights = []
        for fpath in candidates:
            if len(highlights) >= max_files:
                break
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lower_content = content.lower()
            content_score = sum(lower_content.count(tok) for tok in tokens)
            if content_score == 0 and len(highlights) > 0:
                continue

            snippets = self._extract_snippets(content, tokens)
            highlights.append({
                "path": str(fpath.relative_to(search_root)),
                "local_path": str(fpath),
                "score": int(content_score),
                "snippets": snippets,
            })

        if not highlights:
            return json.dumps({
                "status": "error",
                "message": "No relevant snippets found for query in candidate files",
                "search_root": str(search_root),
            })

        return json.dumps({
            "status": "success",
            "source": "local",
            "search_root": str(search_root),
            "query": query,
            "highlights": highlights,
            "note": "Snippets are sourced from the local granite-wxc checkout. Cite 'local_path' when referencing files.",
        })
    
    @staticmethod
    def parse_natural_date(date_str: str) -> Optional[str]:
        """Parse natural language date formats into yyyymmdd format.
        
        Examples:
            "Jan 1, 2024" -> "20240101"
            "January 1, 2024" -> "20240101"
            "2024-01-01" -> "20240101"
            "01/01/2024" -> "20240101"
            "20240101" -> "20240101"
        """
        date_str = date_str.strip()
        
        # Try common formats
        formats = [
            "%b %d, %Y",      # Jan 1, 2024
            "%B %d, %Y",      # January 1, 2024
            "%b %d %Y",       # Jan 1 2024
            "%B %d %Y",       # January 1 2024
            "%Y-%m-%d",       # 2024-01-01
            "%m/%d/%Y",       # 01/01/2024
            "%d/%m/%Y",       # 01/01/2024 (try second)
            "%Y%m%d",         # 20240101
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y%m%d")
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def determine_dataset_type(date_str: str) -> str:
        """Determine if a date belongs to training or inference dataset.
        
        Before 2016 = training
        2016 and after = inference
        """
        # Parse to get year
        normalized_date = MERRA2Analyzer.parse_natural_date(date_str)
        if not normalized_date:
            return "inference"  # default
        
        year = int(normalized_date[:4])
        return "training" if year < MERRA2Analyzer.YEAR_THRESHOLD else "inference"
    
    def find_file_by_date(self, date_str: str, dataset_type: Optional[str] = None) -> str:
        """Find a file by date string (supports natural language dates and yyyymmdd).
        
        Args:
            date_str: Date in any natural format or yyyymmdd (e.g., 'Jan 1, 2024' or '20240101')
            dataset_type: 'inference' or 'training'. If None, auto-determined by year.
        """
        # Parse the date
        normalized_date = self.parse_natural_date(date_str)
        if not normalized_date:
            return f"Could not parse date: '{date_str}'. Try formats like 'Jan 1, 2024' or '20240101'"
        
        # Auto-determine dataset type if not specified
        if dataset_type is None:
            dataset_type = self.determine_dataset_type(normalized_date)
        
        if dataset_type not in self.DATA_DIRS:
            return f"Dataset type must be 'inference' or 'training', got '{dataset_type}'"
        
        directory = Path(self.DATA_DIRS[dataset_type])
        if not directory.exists():
            return f"Directory not found: {directory}"
        
        # Search for file matching yyyymmdd.nc pattern
        pattern = f"{normalized_date}.nc"
        matching_files = list(directory.glob(f"*{pattern}*"))
        
        if not matching_files:
            return f"No file found for date {normalized_date} ({date_str}) in {dataset_type}. Pattern: {pattern}"
        
        return str(matching_files[0])
    
    def load_by_date(self, date_str: str, dataset_type: Optional[str] = None) -> str:
        """Load dataset by date string (natural language or yyyymmdd format).
        
        Args:
            date_str: Date like 'Jan 1, 2024', 'January 1, 2024', or '20240101'
            dataset_type: 'inference' or 'training'. If None, auto-determined by year.
        """
        filepath = self.find_file_by_date(date_str, dataset_type)
        
        # If filepath is an error message, return it
        if filepath.startswith("No file") or filepath.startswith("Could not parse") or filepath.startswith("Dataset type"):
            return filepath
        
        return self.load_dataset(filepath)
    
    def load_dataset(self, filepath: str) -> str:
        """Load a NetCDF file."""
        try:
            self.filepath = filepath
            self.ds = xr.open_dataset(filepath)
            return f"✓ Loaded: {Path(filepath).name}\nDimensions: {dict(self.ds.dims)}\nVariables: {list(self.ds.data_vars)}"
        except Exception as e:
            return f"✗ Error loading {filepath}: {e}"
    
    def dataset_summary(self) -> str:
        """Get overall dataset summary."""
        if self.ds is None:
            return "No dataset loaded"
        
        lines = [
            f"FILE: {self.filepath}",
            f"DIMENSIONS: {dict(self.ds.dims)}",
            f"\nVARIABLES ({len(self.ds.data_vars)}):",
        ]
        
        for var in self.ds.data_vars:
            shape = self.ds[var].shape
            dtype = self.ds[var].dtype
            long_name = self.ds[var].attrs.get('long_name', 'N/A')
            units = self.ds[var].attrs.get('units', 'N/A')
            lines.append(f"  • {var:8} {str(shape):30} {long_name:25} [{units}]")
        
        lines.append(f"\nCOORDINATES ({len(self.ds.coords)}):")
        for coord in self.ds.coords:
            lines.append(f"  • {coord:8} length={len(self.ds[coord])}")
        
        return "\n".join(lines)
    
    def variable_info(
        self,
        var_name: str,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        region: Optional[str] = None,
    ) -> str:
        """Get detailed info on a specific variable, optionally restricted to a spatial bounding box."""
        if self.ds is None:
            return "No dataset loaded"
        if var_name not in self.ds.data_vars:
            return f"Variable '{var_name}' not found. Available: {list(self.ds.data_vars)}"
        
        lat_min, lat_max, lon_min, lon_max = self._resolve_region(region, lat_min, lat_max, lon_min, lon_max)
        var = self._apply_spatial_subset(self.ds[var_name], lat_min, lat_max, lon_min, lon_max)
        subset_note = ""
        if any(x is not None for x in (lat_min, lat_max, lon_min, lon_max)):
            subset_note = f" [lat {lat_min}–{lat_max}, lon {lon_min}–{lon_max}]"
        lines = [f"VARIABLE: {var_name}{subset_note}", f"Shape: {var.shape}", f"Dtype: {var.dtype}"]
        
        # Stats (skip NaN/invalid)
        raw = var.values.reshape(-1)
        valid_data = raw[np.isfinite(raw) & (np.abs(raw) < 1e10)]
        if len(valid_data) > 0:
            lines.append(f"\nStats (valid values only):")
            lines.append(f"  Min: {np.nanmin(valid_data):.6g}")
            lines.append(f"  Max: {np.nanmax(valid_data):.6g}")
            lines.append(f"  Mean: {np.nanmean(valid_data):.6g}")
            lines.append(f"  Std: {np.nanstd(valid_data):.6g}")
        
        # Attributes
        if var.attrs:
            lines.append(f"\nAttributes:")
            for k, v in var.attrs.items():
                lines.append(f"  {k}: {v}")
        
        return "\n".join(lines)
    
    def slice_variable(
        self,
        var_name: str,
        time_idx: int = 0,
        lev_idx: int = 0,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        region: Optional[str] = None,
    ) -> str:
        """Slice and summarize a variable at specific indices, optionally restricted to a bounding box."""
        if self.ds is None:
            return "No dataset loaded"
        if var_name not in self.ds.data_vars:
            return f"Variable '{var_name}' not found"
        
        var = self.ds[var_name]
        lat_min, lat_max, lon_min, lon_max = self._resolve_region(region, lat_min, lat_max, lon_min, lon_max)
        
        # Basic slicing
        sliced = var.isel(time=time_idx, lev=lev_idx) if 'time' in var.dims and 'lev' in var.dims else var
        sliced = self._apply_spatial_subset(sliced, lat_min, lat_max, lon_min, lon_max)
        
        subset_note = ""
        if any(x is not None for x in (lat_min, lat_max, lon_min, lon_max)):
            subset_note = f" [lat {lat_min}–{lat_max}, lon {lon_min}–{lon_max}]"
        lines = [
            f"SLICE: {var_name}{subset_note}",
            f"Shape: {sliced.shape}",
            f"Range: [{np.nanmin(sliced.values):.6g}, {np.nanmax(sliced.values):.6g}]"
        ]
        
        # ASCII heatmap for 2D data
        if sliced.ndim == 2:
            data = sliced.values
            valid = data[~np.isnan(data) & (data < 1e10)]
            if len(valid) > 0:
                normalized = (data - np.nanmin(valid)) / (np.nanmax(valid) - np.nanmin(valid))
                lines.append(f"\nHeatmap (lat x lon):")
                chars = " ░▒▓█"
                for row in normalized:
                    row_str = "".join(chars[int(min(4, max(0, v * 5)))] if not np.isnan(v) else "?" for v in row)
                    lines.append(row_str[:80])  # Limit width
        
        return "\n".join(lines)
    
    def list_variables(self) -> str:
        """Simple list of all variables."""
        if self.ds is None:
            return "No dataset loaded"
        return "Variables:\n" + "\n".join(f"  • {v}" for v in self.ds.data_vars)
    
    def list_coordinates(self) -> str:
        """Simple list of all coordinates."""
        if self.ds is None:
            return "No dataset loaded"
        lines = ["Coordinates:"]
        for coord in self.ds.coords:
            size = len(self.ds[coord])
            min_val, max_val = self.ds[coord].values[0], self.ds[coord].values[-1]
            try:
                range_str = f"{float(min_val):.2f} to {float(max_val):.2f}"
            except (TypeError, ValueError):
                range_str = f"{min_val} to {max_val}"
            lines.append(f"  • {coord:8} (n={size:4}) [{range_str}]")
        return "\n".join(lines)

    def _slice_for_plot(self, var_name: str, time_idx: int = 0, lev_idx: int = 0) -> Tuple[Optional[xr.DataArray], Optional[str]]:
        """Return a plotted slice and error string (if any)."""
        if self.ds is None:
            return None, "No dataset loaded"
        resolved = self._resolve_var_name(self.ds, var_name)
        if resolved is None:
            available = ", ".join(sorted(str(v) for v in self.ds.data_vars))
            return None, f"Variable '{var_name}' not found. Available: {available}"

        var = self.ds[resolved]
        sliced = var
        if "time" in var.dims:
            sliced = sliced.isel(time=time_idx)
        if "lev" in var.dims:
            sliced = sliced.isel(lev=lev_idx)
        return sliced, None

    @staticmethod
    def _apply_spatial_subset(
        da: xr.DataArray,
        lat_min: Optional[float],
        lat_max: Optional[float],
        lon_min: Optional[float],
        lon_max: Optional[float],
        region: Optional[str] = None,
    ) -> xr.DataArray:
        """Subset a DataArray to the given lat/lon bounding box.

        If *region* is provided and no explicit bounds are set, the bounds are
        looked up from ``MERRA2Analyzer.KNOWN_REGIONS``.
        """
        lat_min, lat_max, lon_min, lon_max = MERRA2Analyzer._resolve_region(
            region, lat_min, lat_max, lon_min, lon_max
        )
        sel_kwargs: Dict[str, Any] = {}
        if "lat" in da.coords and (lat_min is not None or lat_max is not None):
            lat_arr = da.coords["lat"].values
            lo = float(lat_min) if lat_min is not None else float(np.nanmin(lat_arr))
            hi = float(lat_max) if lat_max is not None else float(np.nanmax(lat_arr))
            # xarray sel with slice requires ascending order
            if lat_arr[0] <= lat_arr[-1]:
                sel_kwargs["lat"] = slice(lo, hi)
            else:
                sel_kwargs["lat"] = slice(hi, lo)
        if "lon" in da.coords and (lon_min is not None or lon_max is not None):
            lon_arr = da.coords["lon"].values
            lo = float(lon_min) if lon_min is not None else float(np.nanmin(lon_arr))
            hi = float(lon_max) if lon_max is not None else float(np.nanmax(lon_arr))
            if lon_arr[0] <= lon_arr[-1]:
                sel_kwargs["lon"] = slice(lo, hi)
            else:
                sel_kwargs["lon"] = slice(hi, lo)
        if sel_kwargs:
            da = da.sel(**sel_kwargs)
        return da

    @staticmethod
    def _create_map_axes(
        figsize: Tuple[float, float] = (10, 7),
        constrained_layout: bool = False,
    ) -> Tuple[plt.Figure, Any]:
        """Create map axes; prefer Cartopy GeoAxes when available."""
        if _CARTOPY_AVAILABLE:
            fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
            return fig, ax
        return plt.subplots(figsize=figsize, constrained_layout=constrained_layout)

    @staticmethod
    def _map_plot_kwargs() -> Dict[str, Any]:
        if _CARTOPY_AVAILABLE:
            return {"transform": ccrs.PlateCarree()}
        return {}

    @staticmethod
    def _set_map_bounds(ax, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
        if _CARTOPY_AVAILABLE and hasattr(ax, "set_extent"):
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            return
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

    @staticmethod
    def _draw_map_grid(ax, linewidth: float = 0.4) -> None:
        if _CARTOPY_AVAILABLE and hasattr(ax, "gridlines"):
            # Avoid draw_labels=True — it triggers a Cartopy/Shapely LinearRing
            # crash on small extents (e.g. LA County). Use manual ticks instead.
            ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=False,
                linestyle="--",
                linewidth=linewidth,
                alpha=0.5,
            )
            try:
                from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
                import numpy as _np
                extent = ax.get_extent(crs=ccrs.PlateCarree())
                lon0, lon1, lat0, lat1 = extent
                n_lon = min(6, max(3, int(abs(lon1 - lon0) / 0.5) + 1))
                n_lat = min(6, max(3, int(abs(lat1 - lat0) / 0.5) + 1))
                lon_ticks = _np.round(_np.linspace(lon0, lon1, n_lon), 2)
                lat_ticks = _np.round(_np.linspace(lat0, lat1, n_lat), 2)
                ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
                ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                ax.xaxis.set_major_formatter(LongitudeFormatter())
                ax.yaxis.set_major_formatter(LatitudeFormatter())
            except Exception:
                pass
            return
        ax.grid(True, linestyle="--", linewidth=linewidth, alpha=0.5)

    @staticmethod
    def _draw_borders(ax, lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> None:
        """Draw coastlines, country borders, and state lines.

        Uses Cartopy features on GeoAxes when available; otherwise falls back
        to Natural Earth shapefile rendering on plain matplotlib axes.
        """
        if _CARTOPY_AVAILABLE and hasattr(ax, "add_feature"):
            try:
                ax.coastlines(resolution="10m", linewidth=0.8, color="black")
                ax.add_feature(
                    cfeature.BORDERS.with_scale("10m"),
                    edgecolor="black",
                    linewidth=0.6,
                )
                ax.add_feature(
                    cfeature.STATES.with_scale("10m"),
                    edgecolor="#444444",
                    linewidth=0.4,
                    linestyle="--",
                )
                return
            except Exception:
                pass

        if shpreader is None:
            return

        def _plot_geom(geom, **kw):
            gtype = geom.geom_type
            if gtype in ("LineString", "LinearRing"):
                xy = np.asarray(geom.coords)
                if len(xy) >= 2:
                    ax.plot(xy[:, 0], xy[:, 1], **kw)
            elif gtype == "Polygon":
                xy = np.asarray(geom.exterior.coords)
                if len(xy) >= 2:
                    ax.plot(xy[:, 0], xy[:, 1], **kw)
            elif hasattr(geom, "geoms"):
                for part in geom.geoms:
                    _plot_geom(part, **kw)

        layers = [
            ("physical",  "coastline",                        "10m", dict(color="black",   linewidth=0.8, linestyle="-")),
            ("cultural",  "admin_0_boundary_lines_land",       "10m", dict(color="black",   linewidth=0.6, linestyle="-")),
            ("cultural",  "admin_1_states_provinces_lines",    "10m", dict(color="#444444", linewidth=0.4, linestyle="--")),
        ]
        for category, name, resolution, kw in layers:
            try:
                path = shpreader.natural_earth(resolution=resolution, category=category, name=name)
                for geom in shpreader.Reader(path).geometries():
                    # Quick bounding-box filter to skip faraway geometries
                    b = geom.bounds  # (minx, miny, maxx, maxy)
                    if b[2] < lon_min or b[0] > lon_max or b[3] < lat_min or b[1] > lat_max:
                        continue
                    _plot_geom(geom, **kw)
            except Exception:
                pass

    def _finalize_plot(self, fig: plt.Figure, filename_prefix: str) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe_prefix = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename_prefix)
        filename = f"{safe_prefix}_{timestamp}.png"
        output_path = self.ARTIFACTS_DIR / filename
        if not path_is_allowed(output_path):
            plt.close(fig)
            return json.dumps({
                "status": "error",
                "message": (
                    f"Artifact path {output_path} is outside allowed roots "
                    f"({allowed_roots_message()})."
                ),
            })
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        url = f"http://localhost:8001/artifacts/{filename}"
        return json.dumps(
            {
                "status": "success",
                "filename": filename,
                "path": str(output_path),
                "plot_url": url,
                "download_url": url,
            }
        )

    @staticmethod
    def _infer_detail_level(detail_level: Optional[str], request_text: Optional[str]) -> str:
        """Infer detail level from explicit setting or free-text request."""
        valid = {"high", "moderate", "detailed"}
        explicit = (detail_level or "").strip().lower()
        if explicit in valid:
            return explicit

        text = (request_text or "").lower()
        if any(k in text for k in ["high level", "high-level", "overview", "simple", "brief"]):
            return "high"
        if any(k in text for k in ["detailed", "deep", "comprehensive", "full", "step by step", "step-by-step"]):
            return "detailed"
        return "moderate"

    @staticmethod
    def _infer_focus(focus: Optional[str], request_text: Optional[str]) -> Optional[str]:
        """Infer focus area from explicit setting or free-text request."""
        allowed = {
            "data",
            "preprocessing",
            "training",
            "inference",
            "evaluation",
            "deployment",
            "model",
        }
        explicit = (focus or "").strip().lower()
        if explicit in allowed:
            return explicit

        text = (request_text or "").lower()
        mapping = {
            "preprocessing": ["preprocess", "regrid", "alignment", "tiling"],
            "training": ["train", "finetune", "fine-tune", "checkpoint"],
            "inference": ["infer", "prediction", "predict"],
            "evaluation": ["evaluate", "metric", "validation", "compare"],
            "data": ["data", "input", "target", "dataset"],
            "deployment": ["deploy", "serving", "production", "api"],
            "model": ["model", "architecture", "backbone"],
        }
        for key, hints in mapping.items():
            if any(h in text for h in hints):
                return key
        return None

    @staticmethod
    def _resolve_config_path(config_path: Optional[str]) -> Optional[Path]:
        """Resolve a config path safely to an existing YAML file."""
        candidates: List[Path] = []
        if config_path:
            p = Path(config_path).expanduser()
            if p.is_absolute():
                candidates.append(p)
            else:
                candidates.append((_SCRIPT_DIR / p).resolve())
                candidates.append((_SCRIPT_DIR / p.name).resolve())

        candidates.append(Path(_DEFAULT_CONFIG).resolve())

        for c in candidates:
            if c.exists() and c.is_file() and c.suffix.lower() in {".yaml", ".yml"}:
                return c
        return None

    @staticmethod
    def _load_workflow_context(config_path: Optional[str]) -> Dict[str, Any]:
        """Load compact workflow context from YAML config when available."""
        ctx: Dict[str, Any] = {
            "config_path": None,
            "predictors": [],
            "levels": [],
            "targets": [],
            "training_start": None,
            "training_end": None,
            "inference_start": None,
            "inference_end": None,
            "epochs": None,
            "batch_size": None,
            "weights": None,
        }

        resolved = MERRA2Analyzer._resolve_config_path(config_path)
        if not resolved:
            return ctx

        try:
            import yaml as _yaml
            cfg = _yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
            data = cfg.get("data", {}) if isinstance(cfg, dict) else {}
            predictor_map = data.get("predictor_variables", {}) if isinstance(data, dict) else {}
            targets = data.get("target_variables") or data.get("output_vars") or []
            dates = cfg.get("dates", {}) if isinstance(cfg, dict) else {}
            train_dates = dates.get("training", {}) if isinstance(dates, dict) else {}
            infer_dates = dates.get("inference", {}) if isinstance(dates, dict) else {}

            levels = []
            for _, lv in predictor_map.items():
                if isinstance(lv, list):
                    levels.extend(lv)

            ctx.update(
                {
                    "config_path": str(resolved),
                    "predictors": sorted([str(k) for k in predictor_map.keys()]),
                    "levels": sorted({str(l) for l in levels}),
                    "targets": [str(t) for t in targets],
                    "training_start": train_dates.get("start"),
                    "training_end": train_dates.get("end"),
                    "inference_start": infer_dates.get("start"),
                    "inference_end": infer_dates.get("end"),
                    "epochs": cfg.get("num_epochs"),
                    "batch_size": cfg.get("batch_size"),
                    "weights": cfg.get("path_model_weights"),
                }
            )
        except Exception:
            pass

        return ctx

    @staticmethod
    def _auto_generate_steps(
        process_type: str,
        detail_level: str,
        focus: Optional[str],
        ctx: Dict[str, Any],
    ) -> List[str]:
        """Generate robust workflow steps from process intent and config context."""
        predictors = ", ".join(ctx.get("predictors", [])[:5])
        targets = ", ".join(ctx.get("targets", [])[:4])
        levels = ", ".join(ctx.get("levels", [])[:4])
        train_range = None
        if ctx.get("training_start") and ctx.get("training_end"):
            train_range = f"{ctx['training_start']} to {ctx['training_end']}"
        infer_range = None
        if ctx.get("inference_start") and ctx.get("inference_end"):
            infer_range = f"{ctx['inference_start']} to {ctx['inference_end']}"

        base: Dict[str, List[str]] = {
            "preprocessing": [
                "Identify predictor and target sources and domain coverage",
                "Align MERRA2 and PRISM grids to a shared tiling strategy",
                "Write preprocessed NetCDF outputs and index files",
                "Validate sample tiles, masks, and metadata completeness",
            ],
            "training": [
                "Load preprocessed training set and normalization scalars",
                "Initialize backbone weights and training hyperparameters",
                "Run fine-tuning loop with periodic checkpointing",
                "Validate metrics and select best checkpoint",
                "Persist final artifacts and training logs",
            ],
            "inference": [
                "Load selected checkpoint and inference configuration",
                "Read inference inputs for the requested date range",
                "Run batched prediction over all tiles",
                "Write predicted outputs and diagnostics",
                "Review map-level quality and summary statistics",
            ],
            "general": [
                "Confirm data, domain, and run configuration",
                "Preprocess and normalize model inputs",
                "Train or select an existing checkpoint",
                "Run inference on target period",
                "Evaluate outputs and package artifacts",
            ],
        }

        ptype = process_type if process_type in base else "general"
        steps = list(base[ptype])

        # Inject concrete context when available.
        if predictors:
            steps[0] = f"Identify predictor sources ({predictors}) and target variables ({targets or 'configured outputs'})"
        if levels and "preprocess" in steps[1].lower():
            steps[1] = f"Align grids and pressure levels ({levels}) to a shared tiling strategy"
        if ptype == "training" and ctx.get("weights"):
            steps[1] = "Initialize from configured pretrained backbone and training hyperparameters"
        if ptype == "training" and train_range:
            steps.insert(1, f"Select training window: {train_range}")
        if ptype == "inference" and infer_range:
            steps[1] = f"Read inference inputs for configured period ({infer_range})"

        # Detail level controls granularity.
        if detail_level == "high":
            steps = steps[:4]
        elif detail_level == "detailed":
            detailed_tail = [
                "Track run-time logs and failure checks during execution",
                "Archive outputs, checkpoints, and reproducibility metadata",
            ]
            if ptype in ("training", "general") and ctx.get("epochs"):
                steps.insert(3, f"Train for configured epochs (num_epochs={ctx['epochs']}) with checkpoint cadence")
            if ptype in ("inference", "general") and ctx.get("batch_size"):
                steps.insert(3, f"Run inference with configured batch size (batch_size={ctx['batch_size']})")
            steps.extend(detailed_tail)

        # Focus filter keeps the requested concern central while preserving flow.
        if focus:
            focus_map = {
                "data": ["data", "predictor", "target", "window", "inputs"],
                "preprocessing": ["preprocess", "align", "tile", "mask", "normalize"],
                "training": ["train", "checkpoint", "epoch", "fine-tuning", "weights"],
                "inference": ["inference", "predict", "output", "batch"],
                "evaluation": ["evaluate", "metric", "quality", "diagnostic", "validate"],
                "deployment": ["artifact", "package", "serve", "api"],
                "model": ["model", "backbone", "weights", "hyperparameter"],
            }
            hints = focus_map.get(focus, [])
            match_idx = [i for i, s in enumerate(steps) if any(h in s.lower() for h in hints)]
            if match_idx:
                keep_idx = {0, len(steps) - 1}
                # Keep one neighbor around each focused step for context continuity.
                for i in match_idx:
                    for j in range(max(0, i - 1), min(len(steps), i + 2)):
                        keep_idx.add(j)

                focused_steps = [steps[i] for i in sorted(keep_idx)]
                min_target = 4 if detail_level == "high" else (6 if detail_level == "detailed" else 5)

                if len(focused_steps) < min_target:
                    for s in steps:
                        if s not in focused_steps:
                            focused_steps.append(s)
                        if len(focused_steps) >= min_target:
                            break

                steps = focused_steps

        # De-duplicate while preserving order.
        deduped: List[str] = []
        seen = set()
        for s in steps:
            key = s.strip().lower()
            if key and key not in seen:
                deduped.append(s)
                seen.add(key)

        return deduped

    def generate_workflow_image(
        self,
        process_type: str = "general",
        title: Optional[str] = None,
        steps: Optional[List[str]] = None,
        detail_level: Optional[str] = None,
        focus: Optional[str] = None,
        request_text: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> str:
        """Generate a workflow diagram image (PNG) and return artifact URLs.

        Args:
            process_type: One of training, inference, preprocessing, or general.
            title: Optional custom title shown on top of the image.
            steps: Optional ordered list of workflow steps. Overrides auto generation when provided.
            detail_level: Optional high/moderate/detailed granularity.
            focus: Optional focus area such as data, preprocessing, training, inference, evaluation.
            request_text: Optional free-text user intent used to infer detail/focus automatically.
            config_path: Optional YAML config path to inject concrete project-specific context.
        """
        ptype = (process_type or "general").strip().lower()
        if ptype not in {"preprocessing", "training", "inference", "general"}:
            ptype = "general"

        inferred_detail = self._infer_detail_level(detail_level, request_text)
        inferred_focus = self._infer_focus(focus, request_text)
        context = self._load_workflow_context(config_path)
        auto_steps = self._auto_generate_steps(ptype, inferred_detail, inferred_focus, context)

        normalized_steps = [str(s).strip() for s in (steps or auto_steps) if str(s).strip()]
        if len(normalized_steps) < 2:
            return json.dumps({
                "status": "error",
                "message": "At least 2 non-empty steps are required to generate a workflow image.",
            })

        title_suffix = f" ({inferred_detail})" if inferred_detail else ""
        workflow_title = (title or f"{ptype.title()} Workflow{title_suffix}").strip()
        n = len(normalized_steps)

        fig_h = max(4.5, 1.35 * n + 2.3)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        fig.patch.set_facecolor("#f6fbf8")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, n + 1.4)
        ax.axis("off")

        # Header band
        ax.add_patch(
            FancyBboxPatch(
                (0.4, n + 0.55), 9.2, 0.7,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                linewidth=0,
                facecolor="#1f7a4f",
            )
        )
        ax.text(
            0.7,
            n + 0.9,
            workflow_title,
            fontsize=15,
            color="white",
            weight="bold",
            va="center",
        )

        x0, w = 1.0, 8.0
        box_h = 0.75
        for i, step in enumerate(normalized_steps, start=1):
            y = n - i + 1
            face = "#e8f3ff" if i % 2 else "#edf8ef"
            edge = "#9db7c8"
            ax.add_patch(
                FancyBboxPatch(
                    (x0, y), w, box_h,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    linewidth=1.2,
                    edgecolor=edge,
                    facecolor=face,
                )
            )
            ax.text(
                x0 + 0.22,
                y + box_h / 2,
                f"{i}.",
                fontsize=11,
                weight="bold",
                color="#1f3a4a",
                va="center",
            )
            wrapped = textwrap.fill(step, width=68)
            ax.text(
                x0 + 0.65,
                y + box_h / 2,
                wrapped,
                fontsize=10.5,
                color="#1f3a4a",
                va="center",
            )

            if i < n:
                ax.annotate(
                    "",
                    xy=(x0 + w / 2, y - 0.06),
                    xytext=(x0 + w / 2, y - 0.35),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#2f5f7a"),
                )

        ax.text(
            0.6,
            0.32,
            "Generated by MERRA2 agent MCP tool",
            fontsize=9,
            color="#496270",
        )
        fig.tight_layout()

        plot_meta = json.loads(self._finalize_plot(fig, f"workflow_{ptype}"))
        result = {
            "status": "success",
            "message": f"Generated workflow image for '{ptype}' process",
            "process_type": ptype,
            "title": workflow_title,
            "steps": normalized_steps,
            "detail_level": inferred_detail,
            "focus": inferred_focus,
            "auto_generated": steps is None,
            "config_path": context.get("config_path"),
        }
        result.update(plot_meta)
        return json.dumps(result)

    @staticmethod
    def _resolve_var_name(ds: xr.Dataset, var_name: str) -> Optional[str]:
        """Resolve a variable name against a dataset, with common aliases.

        Inference outputs use ``ppt`` / ``tmax`` / ``tmin``, while preprocessed
        files use ``target_ppt`` / ``target_tmax`` / ``target_tmin``. Accept either.
        """
        if var_name in ds.data_vars:
            return var_name
        candidates: List[str] = []
        if var_name.startswith("target_"):
            candidates.append(var_name[len("target_"):])
        else:
            candidates.append(f"target_{var_name}")
        if var_name.startswith("output_"):
            candidates.append(var_name[len("output_"):])
        else:
            candidates.append(f"output_{var_name}")
        for cand in candidates:
            if cand in ds.data_vars:
                return cand
        return None

    @staticmethod
    def _slice_from_dataset(ds: xr.Dataset, var_name: str, time_idx: int = 0, lev_idx: int = 0) -> xr.DataArray:
        """Slice a variable from a given dataset by time/level when available."""
        resolved = MERRA2Analyzer._resolve_var_name(ds, var_name)
        if resolved is None:
            available = ", ".join(sorted(str(v) for v in ds.data_vars))
            raise KeyError(f"Variable '{var_name}' not found. Available: {available}")
        var = ds[resolved]
        sliced = var
        if "time" in var.dims:
            sliced = sliced.isel(time=time_idx)
        if "lev" in var.dims:
            sliced = sliced.isel(lev=lev_idx)
        return sliced

    def plot_variable_2d(self, var_name: str, time_idx: int = 0, lev_idx: int = 0, cmap: str = "viridis",
                          vmin: Optional[float] = None, vmax: Optional[float] = None,
                          lat_min: Optional[float] = None, lat_max: Optional[float] = None,
                          lon_min: Optional[float] = None, lon_max: Optional[float] = None,
                          region: Optional[str] = None) -> str:
        """Create a publication-ready 2D map plot with lat/lon labels and save as PNG."""
        sliced, err = self._slice_for_plot(var_name, time_idx, lev_idx)
        if err:
            return json.dumps({"status": "error", "message": err})

        lat_min, lat_max, lon_min, lon_max = self._resolve_region(region, lat_min, lat_max, lon_min, lon_max)

        sliced = self._apply_spatial_subset(sliced, lat_min, lat_max, lon_min, lon_max)

        if sliced.ndim != 2:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Expected 2D slice after indexing, got shape {sliced.shape} for '{var_name}'"
                }
            )

        lat = sliced.coords["lat"].values if "lat" in sliced.coords else np.arange(sliced.shape[0])
        lon = sliced.coords["lon"].values if "lon" in sliced.coords else np.arange(sliced.shape[1])
        data = sliced.values

        finite = np.isfinite(data) & (np.abs(data) < 1e10)
        if not np.any(finite):
            return json.dumps({"status": "error", "message": f"No finite values to plot for '{var_name}'"})

        plt.style.use("default")
        fig, ax = self._create_map_axes(figsize=(10, 7))

        plot_lon_min, plot_lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        plot_lat_min, plot_lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))

        mesh = ax.pcolormesh(
            lon,
            lat,
            data,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **self._map_plot_kwargs(),
        )
        self._draw_borders(ax, plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)

        self._set_map_bounds(ax, plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        self._draw_map_grid(ax, linewidth=0.4)

        units = sliced.attrs.get("units", "")
        cbar_label = var_name if not units else f"{var_name} ({units})"
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        level_text = ""
        if "lev" in sliced.coords and self.ds is not None and "lev" in self.ds.coords:
            lev_val = self.ds["lev"].values[lev_idx]
            level_text = f" | level={lev_val}"
        title = f"{var_name} map (LA County)\ntime_idx={time_idx}{level_text}"
        ax.set_title(title, fontsize=12)
        fig.tight_layout()

        return self._finalize_plot(fig, f"map_{var_name}_t{time_idx}_l{lev_idx}")

    def plot_data(
        self,
        var_name: str,
        plot_type: str = "map",
        time_idx: int = 0,
        lev_idx: int = 0,
        cmap: str = "viridis",
        bins: int = 40,
        line_axis: str = "lon",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        region: Optional[str] = None,
    ) -> str:
        """Create map/heatmap/line/histogram plots and return downloadable PNG URL."""
        sliced, err = self._slice_for_plot(var_name, time_idx, lev_idx)
        if err:
            return json.dumps({"status": "error", "message": err})

        lat_min, lat_max, lon_min, lon_max = self._resolve_region(region, lat_min, lat_max, lon_min, lon_max)
        sliced = self._apply_spatial_subset(sliced, lat_min, lat_max, lon_min, lon_max)

        plot_type = (plot_type or "map").lower()
        units = sliced.attrs.get("units", "")
        units_suffix = f" ({units})" if units else ""

        if plot_type in {"map", "heatmap"}:
            if sliced.ndim != 2:
                return json.dumps({"status": "error", "message": f"{plot_type} requires 2D data, got shape {sliced.shape}"})
            lat = sliced.coords["lat"].values if "lat" in sliced.coords else np.arange(sliced.shape[0])
            lon = sliced.coords["lon"].values if "lon" in sliced.coords else np.arange(sliced.shape[1])
            fig, ax = self._create_map_axes(figsize=(10, 7))
            plot_lon_min, plot_lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
            plot_lat_min, plot_lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
            mesh = ax.pcolormesh(
                lon,
                lat,
                sliced.values,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **self._map_plot_kwargs(),
            )
            self._draw_borders(ax, plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)
            self._set_map_bounds(ax, plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)
            ax.set_xlabel("Longitude", fontsize=11)
            ax.set_ylabel("Latitude", fontsize=11)
            self._draw_map_grid(ax, linewidth=0.4)
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label(f"{var_name}{units_suffix}", fontsize=11)
            cbar.ax.tick_params(labelsize=10)
            ax.set_title(f"{var_name} {plot_type} (LA County)", fontsize=12)
            fig.tight_layout()

        elif plot_type == "line":
            fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
            if sliced.ndim == 2:
                if line_axis == "lat":
                    y = np.nanmean(sliced.values, axis=1)
                    x = sliced.coords["lat"].values if "lat" in sliced.coords else np.arange(sliced.shape[0])
                    x_label = "Latitude"
                else:
                    y = np.nanmean(sliced.values, axis=0)
                    x = sliced.coords["lon"].values if "lon" in sliced.coords else np.arange(sliced.shape[1])
                    x_label = "Longitude"
            elif sliced.ndim == 1:
                y = sliced.values
                x = sliced.coords[sliced.dims[0]].values if sliced.dims else np.arange(len(y))
                x_label = sliced.dims[0] if sliced.dims else "Index"
            else:
                plt.close(fig)
                return json.dumps({"status": "error", "message": f"line plot requires 1D or 2D data, got shape {sliced.shape}"})
            ax.plot(x, y, linewidth=2.0, color="#1f77b4")
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel(f"{var_name}{units_suffix}", fontsize=11)
            ax.set_title(f"{var_name} line profile", fontsize=12)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

        elif plot_type == "histogram":
            fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
            values = sliced.values.reshape(-1)
            values = values[np.isfinite(values) & (np.abs(values) < 1e10)]
            if len(values) == 0:
                plt.close(fig)
                return json.dumps({"status": "error", "message": f"No finite values to plot histogram for '{var_name}'"})
            ax.hist(values, bins=bins, color="#4c72b0", edgecolor="black", alpha=0.85)
            ax.set_xlabel(f"{var_name}{units_suffix}", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(f"{var_name} histogram", fontsize=12)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        else:
            return json.dumps({"status": "error", "message": f"Unsupported plot_type '{plot_type}'"})

        ax.tick_params(axis="both", labelsize=10)
        result = json.loads(self._finalize_plot(fig, f"{plot_type}_{var_name}_t{time_idx}_l{lev_idx}"))
        result.update({"plot_type": plot_type, "variable": var_name, "time_idx": time_idx, "lev_idx": lev_idx})
        return json.dumps(result)

    def compute_statistics(
        self,
        var_name: str,
        time_idx: int = 0,
        lev_idx: int = 0,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        region: Optional[str] = None,
    ) -> str:
        """Compute descriptive statistics for requested variable slice, optionally within a bounding box."""
        sliced, err = self._slice_for_plot(var_name, time_idx, lev_idx)
        if err:
            return json.dumps({"status": "error", "message": err})

        lat_min, lat_max, lon_min, lon_max = self._resolve_region(region, lat_min, lat_max, lon_min, lon_max)
        sliced = self._apply_spatial_subset(sliced, lat_min, lat_max, lon_min, lon_max)

        values = sliced.values.reshape(-1)
        values = values[np.isfinite(values) & (np.abs(values) < 1e10)]
        if len(values) == 0:
            return json.dumps({"status": "error", "message": f"No finite values found for '{var_name}'"})

        stats = {
            "status": "success",
            "variable": var_name,
            "shape": list(sliced.shape),
            "spatial_subset": {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max},
            "count": int(values.size),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
        }
        return json.dumps(stats, indent=2)

    def compare_dates_difference_map(
        self,
        date_1: str,
        date_2: str,
        var_name: str,
        time_idx: int = 0,
        lev_idx: int = 0,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        region: Optional[str] = None,
    ) -> str:
        """Compute and plot date_2 - date_1 difference map with summary stats."""
        lat_min, lat_max, lon_min, lon_max = self._resolve_region(region, lat_min, lat_max, lon_min, lon_max)
        path_1 = self.find_file_by_date(date_1)
        if path_1.startswith("No file") or path_1.startswith("Could not parse") or path_1.startswith("Dataset type"):
            return json.dumps({"status": "error", "message": path_1})
        path_2 = self.find_file_by_date(date_2)
        if path_2.startswith("No file") or path_2.startswith("Could not parse") or path_2.startswith("Dataset type"):
            return json.dumps({"status": "error", "message": path_2})

        try:
            with xr.open_dataset(path_1) as ds1, xr.open_dataset(path_2) as ds2:
                r1 = self._resolve_var_name(ds1, var_name)
                r2 = self._resolve_var_name(ds2, var_name)
                if r1 is None or r2 is None:
                    return json.dumps(
                        {
                            "status": "error",
                            "message": f"Variable '{var_name}' not found in both datasets",
                        }
                    )

                s1 = self._apply_spatial_subset(
                    self._slice_from_dataset(ds1, r1, time_idx, lev_idx),
                    lat_min, lat_max, lon_min, lon_max,
                )
                s2 = self._apply_spatial_subset(
                    self._slice_from_dataset(ds2, r2, time_idx, lev_idx),
                    lat_min, lat_max, lon_min, lon_max,
                )
                if s1.ndim != 2 or s2.ndim != 2:
                    return json.dumps(
                        {
                            "status": "error",
                            "message": f"Difference map requires 2D slices, got {s1.shape} and {s2.shape}",
                        }
                    )

                diff = s2.values - s1.values
                finite = np.isfinite(diff) & (np.abs(diff) < 1e10)
                if not np.any(finite):
                    return json.dumps({"status": "error", "message": "No finite values in difference array"})

                lat = s1.coords["lat"].values if "lat" in s1.coords else np.arange(s1.shape[0])
                lon = s1.coords["lon"].values if "lon" in s1.coords else np.arange(s1.shape[1])
                vmax = float(np.nanmax(np.abs(diff[finite])))
                norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

                fig, ax = self._create_map_axes(figsize=(8.5, 6.0), constrained_layout=True)
                lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
                lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
                mesh = ax.pcolormesh(
                    lon,
                    lat,
                    diff,
                    shading="auto",
                    cmap="RdBu_r",
                    norm=norm,
                    **self._map_plot_kwargs(),
                )
                self._draw_borders(ax, lon_min, lon_max, lat_min, lat_max)
                self._set_map_bounds(ax, lon_min, lon_max, lat_min, lat_max)
                units = s1.attrs.get("units", "")
                cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
                label = f"{var_name} difference ({date_2} - {date_1})"
                if units:
                    label += f" [{units}]"
                cbar.set_label(label, fontsize=11)
                cbar.ax.tick_params(labelsize=10)

                ax.set_xlabel("Longitude", fontsize=11)
                ax.set_ylabel("Latitude", fontsize=11)
                ax.tick_params(axis="both", labelsize=10)
                self._draw_map_grid(ax, linewidth=0.5)
                ax.set_title(f"{var_name} difference map (LA County)\n{date_2} - {date_1}", fontsize=12)

                plot_meta = json.loads(self._finalize_plot(fig, f"diff_{var_name}_{date_1}_vs_{date_2}"))
                vals = diff[finite]
                pos_frac = float(np.mean(vals > 0))
                neg_frac = float(np.mean(vals < 0))
                summary = {
                    "status": "success",
                    "message": f"Computed and plotted difference map for {var_name}: {date_2} - {date_1}",
                    "variable": var_name,
                    "date_1": date_1,
                    "date_2": date_2,
                    "difference_definition": "date_2 - date_1",
                    "stats": {
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "median": float(np.median(vals)),
                        "p05": float(np.percentile(vals, 5)),
                        "p95": float(np.percentile(vals, 95)),
                        "positive_fraction": pos_frac,
                        "negative_fraction": neg_frac,
                    },
                    "analysis_hint": (
                        "Positive values indicate warming/increase on date_2 relative to date_1; "
                        "negative values indicate cooling/decrease."
                    ),
                }
                summary.update(plot_meta)
                return json.dumps(summary)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Difference computation failed: {e}"})

    @staticmethod
    def _to_datetime(date_str: str) -> Optional[datetime]:
        normalized = MERRA2Analyzer.parse_natural_date(date_str)
        if not normalized:
            return None
        return datetime.strptime(normalized, "%Y%m%d")

    @staticmethod
    def _extract_date_from_filename(path: Path) -> Optional[datetime]:
        match = re.search(r"(\d{8})\.nc", path.name)
        if not match:
            return None
        try:
            return datetime.strptime(match.group(1), "%Y%m%d")
        except ValueError:
            return None

    def _files_in_range(self, start_dt: datetime, end_dt: datetime) -> List[Tuple[datetime, Path]]:
        """Return sorted files in [start_dt, end_dt], scanning only the relevant directory."""
        # Choose which directories to scan based on year boundary (2016)
        dirs_to_scan: List[str] = []
        if start_dt.year < self.YEAR_THRESHOLD:
            dirs_to_scan.append(self.DATA_DIRS["training"])
        if end_dt.year >= self.YEAR_THRESHOLD:
            dirs_to_scan.append(self.DATA_DIRS["inference"])

        files: List[Tuple[datetime, Path]] = []
        for directory in dirs_to_scan:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue
            for p in dir_path.glob("*.nc*"):
                file_dt = self._extract_date_from_filename(p)
                if file_dt and start_dt <= file_dt <= end_dt:
                    files.append((file_dt, p))
        files.sort(key=lambda x: x[0])
        return files

    @staticmethod
    def _aggregate_points(
        dates: List[datetime], values: List[float], frequency: str
    ) -> Tuple[List[datetime], List[float]]:
        if frequency == "daily":
            return dates, values

        buckets: Dict[datetime, List[float]] = {}
        for d, v in zip(dates, values):
            if frequency == "weekly":
                key = datetime(d.year, d.month, d.day) - timedelta(days=d.weekday())
            elif frequency == "monthly":
                key = datetime(d.year, d.month, 1)
            else:
                key = datetime(d.year, d.month, d.day)
            buckets.setdefault(key, []).append(v)

        out_dates = sorted(buckets.keys())
        out_vals = [float(np.mean(buckets[k])) for k in out_dates]
        return out_dates, out_vals

    def compute_trend_timeseries(
        self,
        var_name: str,
        start_date: str,
        end_date: str,
        frequency: str = "monthly",
        lev_idx: int = 0,
        spatial_stat: str = "mean",
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        region: Optional[str] = None,
    ) -> str:
        """Compute long-term trend and generate timeseries + trend-line plot."""
        lat_min, lat_max, lon_min, lon_max = self._resolve_region(region, lat_min, lat_max, lon_min, lon_max)
        frequency = (frequency or "monthly").lower()
        spatial_stat = (spatial_stat or "mean").lower()
        if frequency not in {"daily", "weekly", "monthly"}:
            return json.dumps({"status": "error", "message": "frequency must be one of: daily, weekly, monthly"})
        if spatial_stat not in {"mean", "median", "min", "max"}:
            return json.dumps({"status": "error", "message": "spatial_stat must be one of: mean, median, min, max"})

        start_dt = self._to_datetime(start_date)
        end_dt = self._to_datetime(end_date)
        if not start_dt or not end_dt:
            return json.dumps({"status": "error", "message": "Could not parse start_date or end_date"})
        if end_dt < start_dt:
            return json.dumps({"status": "error", "message": "end_date must be on or after start_date"})

        files = self._files_in_range(start_dt, end_dt)
        if not files:
            return json.dumps({"status": "error", "message": "No files found in requested date range"})

        # Resolve variable name once from the first readable file (ppt ↔ target_ppt).
        resolved_var: Optional[str] = None
        available_vars: List[str] = []
        for _, probe_path in files[:5]:
            try:
                with xr.open_dataset(probe_path, mask_and_scale=False) as probe_ds:
                    available_vars = sorted(str(v) for v in probe_ds.data_vars)
                    resolved_var = self._resolve_var_name(probe_ds, var_name)
                    if resolved_var:
                        break
            except Exception:
                continue
        if not resolved_var:
            return json.dumps({
                "status": "error",
                "message": (
                    f"Variable '{var_name}' not found in data files. "
                    f"Available: {available_vars or 'unknown'}. "
                    "Tip: inference outputs use ppt/tmax/tmin; preprocessed files use target_ppt/…"
                ),
            })

        region_label = region.replace("_", " ") if region else "domain"
        if lat_min is not None or lon_min is not None:
            if not region:
                region_label = "spatial subset"

        dates: List[datetime] = []
        values: List[float] = []
        skipped = 0
        for file_dt, file_path in files:
            try:
                with xr.open_dataset(file_path, mask_and_scale=False) as ds:
                    use_var = self._resolve_var_name(ds, var_name) or resolved_var
                    if use_var not in ds.data_vars:
                        skipped += 1
                        continue
                    sliced = self._slice_from_dataset(ds, use_var, time_idx=0, lev_idx=lev_idx)
                    sliced = self._apply_spatial_subset(sliced, lat_min, lat_max, lon_min, lon_max)
                    arr = sliced.values.reshape(-1)
                    arr = arr[np.isfinite(arr) & (np.abs(arr) < 1e10)]
                    if arr.size == 0:
                        skipped += 1
                        continue
                    if spatial_stat == "mean":
                        v = float(np.mean(arr))
                    elif spatial_stat == "median":
                        v = float(np.median(arr))
                    elif spatial_stat == "min":
                        v = float(np.min(arr))
                    else:
                        v = float(np.max(arr))
                    dates.append(file_dt)
                    values.append(v)
            except Exception:
                skipped += 1

        if len(values) < 2:
            return json.dumps({
                "status": "error",
                "message": (
                    f"Insufficient valid data points for trend "
                    f"(got {len(values)} from {len(files)} files, skipped={skipped}, "
                    f"requested='{var_name}', resolved='{resolved_var}')"
                ),
            })

        agg_dates, agg_vals = self._aggregate_points(dates, values, frequency)
        if len(agg_vals) < 2:
            return json.dumps({"status": "error", "message": "Insufficient aggregated points for trend"})

        x_days = np.array([(d - agg_dates[0]).days for d in agg_dates], dtype=float)
        y = np.array(agg_vals, dtype=float)
        slope, intercept = np.polyfit(x_days, y, 1)
        yhat = slope * x_days + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
        ax.plot(agg_dates, y, color="#1f77b4", linewidth=1.8, marker="o", markersize=3.5, label=f"{frequency} {spatial_stat}")
        ax.plot(agg_dates, yhat, color="#d62728", linewidth=2.0, label="linear trend")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel(f"{resolved_var} ({spatial_stat} over {region_label})", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.legend(fontsize=10, loc="best")
        ax.set_title(f"{resolved_var} trend ({frequency}) from {start_dt.date()} to {end_dt.date()}", fontsize=12)

        plot_meta = json.loads(
            self._finalize_plot(
                fig,
                f"trend_{var_name}_{frequency}_{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}",
            )
        )
        result = {
            "status": "success",
            "message": f"Computed {frequency} trend for {resolved_var} over {len(agg_vals)} points",
            "variable": var_name,
            "resolved_variable": resolved_var,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "frequency": frequency,
            "spatial_stat": spatial_stat,
            "n_raw_points": len(values),
            "n_points": len(agg_vals),
            "n_skipped_files": skipped,
            "trend": {
                "slope_per_day": float(slope),
                "slope_per_year": float(slope * 365.25),
                "intercept": float(intercept),
                "r2": r2,
            },
            "summary_stats": {
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "median": float(np.median(y)),
            },
            "analysis_hint": (
                "Positive slope_per_year indicates increasing trend; negative indicates decreasing trend."
            ),
        }
        result.update(plot_meta)
        return json.dumps(result)


    def compute_monthly_climatology(
        self,
        var_name: str,
        month: int,
        lev_idx: int = 0,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        region: Optional[str] = None,
    ) -> str:
        """Compute and plot monthly climatology (average of all days in a month across years)."""
        lat_min, lat_max, lon_min, lon_max = self._resolve_region(region, lat_min, lat_max, lon_min, lon_max)
        if not (1 <= month <= 12):
            return json.dumps({"status": "error", "message": "Month must be 1-12"})
        
        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        # Collect all files for this month across both training and inference dirs
        all_files = []
        for dataset_type in ["training", "inference"]:
            dir_path = Path(self.DATA_DIRS[dataset_type])
            if not dir_path.exists():
                continue
            for p in dir_path.glob("*.nc*"):
                file_dt = self._extract_date_from_filename(p)
                if file_dt and file_dt.month == month:
                    # Check year range if specified
                    if start_year is not None and file_dt.year < start_year:
                        continue
                    if end_year is not None and file_dt.year > end_year:
                        continue
                    all_files.append((file_dt, p))
        
        if not all_files:
            year_str = ""
            if start_year or end_year:
                year_str = f" for {start_year or '?'}-{end_year or '?'}"
            return json.dumps({
                "status": "error",
                "message": f"No files found for month {month} ({month_names[month]}){year_str} across requested years"
            })
        
        all_files.sort(key=lambda x: x[0])
        
        # Load and aggregate data
        data_slices = []
        failed_count = 0
        for file_dt, file_path in all_files:
            try:
                with xr.open_dataset(file_path, mask_and_scale=False) as ds:
                    if self._resolve_var_name(ds, var_name) is None:
                        failed_count += 1
                        continue
                    sliced = self._slice_from_dataset(ds, var_name, time_idx=0, lev_idx=lev_idx)
                    sliced = self._apply_spatial_subset(sliced, lat_min, lat_max, lon_min, lon_max)
                    if sliced.ndim != 2:
                        failed_count += 1
                        continue
                    data_slices.append(sliced.values)
            except Exception:
                failed_count += 1
        
        if len(data_slices) == 0:
            return json.dumps({
                "status": "error",
                "message": f"Could not extract 2D data for {var_name} from any {month_names[month]} files"
            })
        
        # Compute climatological mean
        stacked = np.stack(data_slices, axis=0)
        valid_mask = np.isfinite(stacked) & (np.abs(stacked) < 1e10)
        clim_data = np.nanmean(np.where(valid_mask, stacked, np.nan), axis=0)
        
        if not np.any(np.isfinite(clim_data)):
            return json.dumps({
                "status": "error",
                "message": "All climatological data is NaN or invalid"
            })
        
        # Get coordinates from first valid file (apply same subset for consistent coords)
        coords_found = False
        lat, lon = None, None
        with xr.open_dataset(all_files[0][1], mask_and_scale=False) as ds:
            sliced = self._apply_spatial_subset(
                self._slice_from_dataset(ds, var_name, time_idx=0, lev_idx=lev_idx),
                lat_min, lat_max, lon_min, lon_max,
            )
            lat = sliced.coords["lat"].values if "lat" in sliced.coords else np.arange(sliced.shape[0])
            lon = sliced.coords["lon"].values if "lon" in sliced.coords else np.arange(sliced.shape[1])
            units = sliced.attrs.get("units", "")
            coords_found = True
        
        if not coords_found:
            return json.dumps({"status": "error", "message": "Could not extract coordinate information"})
        
        # Plot climatology
        plt.style.use("default")
        fig, ax = self._create_map_axes(figsize=(10, 7))
        
        plot_lon_min, plot_lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        plot_lat_min, plot_lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        
        mesh = ax.pcolormesh(
            lon,
            lat,
            clim_data,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **self._map_plot_kwargs(),
        )
        self._draw_borders(ax, plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)
        
        self._set_map_bounds(ax, plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        self._draw_map_grid(ax, linewidth=0.4)
        
        cbar_label = var_name if not units else f"{var_name} ({units})"
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=10)
        
        year_range_str = ""
        if start_year or end_year:
            year_range_str = f" ({start_year or '?'}-{end_year or '?'})"
        title = f"{var_name} {month_names[month]} climatology{year_range_str} (LA County)\n{len(data_slices)} days averaged"
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        
        plot_meta = json.loads(self._finalize_plot(fig, f"climatology_{var_name}_month{month:02d}_{start_year or 'all'}_{end_year or 'all'}"))
        
        # Compute statistics
        valid_clim = clim_data[np.isfinite(clim_data) & (np.abs(clim_data) < 1e10)]
        result = {
            "status": "success",
            "message": f"Computed and plotted {month_names[month]} climatology for {var_name}",
            "variable": var_name,
            "month": month,
            "month_name": month_names[month],
            "start_year": start_year,
            "end_year": end_year,
            "n_days_averaged": len(data_slices),
            "n_failed_files": failed_count,
            "stats": {
                "min": float(np.min(valid_clim)),
                "max": float(np.max(valid_clim)),
                "mean": float(np.mean(valid_clim)),
                "std": float(np.std(valid_clim)),
                "median": float(np.median(valid_clim)),
                "p05": float(np.percentile(valid_clim, 5)),
                "p25": float(np.percentile(valid_clim, 25)),
                "p75": float(np.percentile(valid_clim, 75)),
                "p95": float(np.percentile(valid_clim, 95)),
            },
        }
        result.update(plot_meta)
        return json.dumps(result)

    def compute_seasonal_climatology(
        self,
        var_name: str,
        season: str,
        lev_idx: int = 0,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        region: Optional[str] = None,
    ) -> str:
        """Compute and plot seasonal climatology (average of all days in a season across years).
        Seasons: 'DJF' (winter), 'MAM' (spring), 'JJA' (summer), 'SON' (fall).
        """
        lat_min, lat_max, lon_min, lon_max = self._resolve_region(region, lat_min, lat_max, lon_min, lon_max)
        season_map = {
            'DJF': (12, 1, 2),  # Winter
            'MAM': (3, 4, 5),   # Spring
            'JJA': (6, 7, 8),   # Summer
            'SON': (9, 10, 11), # Fall
        }
        
        season = season.upper()
        if season not in season_map:
            return json.dumps({
                "status": "error",
                "message": f"Season must be one of: {', '.join(season_map.keys())}"
            })
        
        # Collect all files for this season across both dirs
        all_files = []
        for dataset_type in ["training", "inference"]:
            dir_path = Path(self.DATA_DIRS[dataset_type])
            if not dir_path.exists():
                continue
            for p in dir_path.glob("*.nc*"):
                file_dt = self._extract_date_from_filename(p)
                if file_dt and file_dt.month in season_map[season]:
                    # Check year range if specified
                    if start_year is not None and file_dt.year < start_year:
                        continue
                    if end_year is not None and file_dt.year > end_year:
                        continue
                    all_files.append((file_dt, p))
        
        if not all_files:
            return json.dumps({
                "status": "error",
                "message": f"No files found for {season} season across requested years"
            })
        
        all_files.sort(key=lambda x: x[0])
        
        # Load and aggregate data
        data_slices = []
        failed_count = 0
        for file_dt, file_path in all_files:
            try:
                with xr.open_dataset(file_path, mask_and_scale=False) as ds:
                    if self._resolve_var_name(ds, var_name) is None:
                        failed_count += 1
                        continue
                    sliced = self._slice_from_dataset(ds, var_name, time_idx=0, lev_idx=lev_idx)
                    sliced = self._apply_spatial_subset(sliced, lat_min, lat_max, lon_min, lon_max)
                    if sliced.ndim != 2:
                        failed_count += 1
                        continue
                    data_slices.append(sliced.values)
            except Exception:
                failed_count += 1
        
        if len(data_slices) == 0:
            return json.dumps({
                "status": "error",
                "message": f"Could not extract 2D data for {var_name} from any {season} files"
            })
        
        # Compute climatological mean
        stacked = np.stack(data_slices, axis=0)
        valid_mask = np.isfinite(stacked) & (np.abs(stacked) < 1e10)
        clim_data = np.nanmean(np.where(valid_mask, stacked, np.nan), axis=0)
        
        if not np.any(np.isfinite(clim_data)):
            return json.dumps({
                "status": "error",
                "message": "All climatological data is NaN or invalid"
            })
        
        # Get coordinates (apply same subset for consistent coords)
        lat, lon = None, None
        units = ""
        with xr.open_dataset(all_files[0][1], mask_and_scale=False) as ds:
            sliced = self._apply_spatial_subset(
                self._slice_from_dataset(ds, var_name, time_idx=0, lev_idx=lev_idx),
                lat_min, lat_max, lon_min, lon_max,
            )
            lat = sliced.coords["lat"].values if "lat" in sliced.coords else np.arange(sliced.shape[0])
            lon = sliced.coords["lon"].values if "lon" in sliced.coords else np.arange(sliced.shape[1])
            units = sliced.attrs.get("units", "")
        
        # Plot
        plt.style.use("default")
        fig, ax = self._create_map_axes(figsize=(10, 7))
        
        plot_lon_min, plot_lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        plot_lat_min, plot_lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        
        mesh = ax.pcolormesh(
            lon,
            lat,
            clim_data,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **self._map_plot_kwargs(),
        )
        self._draw_borders(ax, plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)
        
        self._set_map_bounds(ax, plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        self._draw_map_grid(ax, linewidth=0.4)
        
        cbar_label = var_name if not units else f"{var_name} ({units})"
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=10)
        
        year_range_str = ""
        if start_year or end_year:
            year_range_str = f" ({start_year or '?'}-{end_year or '?'})"
        title = f"{var_name} {season} climatology{year_range_str} (LA County)\n{len(data_slices)} days averaged"
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        
        plot_meta = json.loads(self._finalize_plot(fig, f"climatology_{var_name}_{season}_{start_year or 'all'}_{end_year or 'all'}"))
        
        # Statistics
        valid_clim = clim_data[np.isfinite(clim_data) & (np.abs(clim_data) < 1e10)]
        result = {
            "status": "success",
            "message": f"Computed and plotted {season} climatology for {var_name}",
            "variable": var_name,
            "season": season,
            "start_year": start_year,
            "end_year": end_year,
            "n_days_averaged": len(data_slices),
            "n_failed_files": failed_count,
            "stats": {
                "min": float(np.min(valid_clim)),
                "max": float(np.max(valid_clim)),
                "mean": float(np.mean(valid_clim)),
                "std": float(np.std(valid_clim)),
                "median": float(np.median(valid_clim)),
                "p05": float(np.percentile(valid_clim, 5)),
                "p25": float(np.percentile(valid_clim, 25)),
                "p75": float(np.percentile(valid_clim, 75)),
                "p95": float(np.percentile(valid_clim, 95)),
            },
        }
        result.update(plot_meta)
        return json.dumps(result)

    def compute_period_composite(
        self,
        var_name: str,
        start_date: str,
        end_date: str,
        lev_idx: int = 0,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None,
        region: Optional[str] = None,
    ) -> str:
        """Compute composite map by averaging all days in a date range across years."""
        lat_min, lat_max, lon_min, lon_max = self._resolve_region(region, lat_min, lat_max, lon_min, lon_max)
        # Parse dates (without year for climatological context)
        def parse_month_day(date_str: str) -> Optional[Tuple[int, int]]:
            """Parse month-day from various formats."""
            date_str = date_str.strip()
            formats = [
                "%b %d",      # Jun 15
                "%m-%d",      # 06-15
                "%m/%d",      # 06/15
                "%B %d",      # June 15
                "%d %b",      # 15 Jun
                "%d %B",      # 15 June
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return (dt.month, dt.day)
                except ValueError:
                    continue
            return None
        
        start_md = parse_month_day(start_date)
        end_md = parse_month_day(end_date)
        
        if not start_md or not end_md:
            return json.dumps({
                "status": "error",
                "message": f"Could not parse dates. Use formats like 'Jun 15', '06-15', or 'June 15'"
            })
        
        # Collect files in date range across years
        all_files = []
        for dataset_type in ["training", "inference"]:
            dir_path = Path(self.DATA_DIRS[dataset_type])
            if not dir_path.exists():
                continue
            for p in dir_path.glob("*.nc*"):
                file_dt = self._extract_date_from_filename(p)
                if not file_dt:
                    continue
                
                # Check year range if specified
                if start_year is not None and file_dt.year < start_year:
                    continue
                if end_year is not None and file_dt.year > end_year:
                    continue
                
                # Check if file's month-day falls within range
                file_md = (file_dt.month, file_dt.day)
                if start_md[0] <= end_md[0]:  # Same or sequential months
                    in_range = (start_md <= file_md <= end_md)
                else:  # Wraps around year (e.g., Nov 1 to Feb 28)
                    in_range = (file_md >= start_md or file_md <= end_md)
                
                if in_range:
                    all_files.append((file_dt, p))
        
        if not all_files:
            return json.dumps({
                "status": "error",
                "message": f"No files found for date range {start_date} to {end_date} across all years"
            })
        
        all_files.sort(key=lambda x: x[0])
        
        # Load and aggregate
        data_slices = []
        failed_count = 0
        for file_dt, file_path in all_files:
            try:
                with xr.open_dataset(file_path, mask_and_scale=False) as ds:
                    if self._resolve_var_name(ds, var_name) is None:
                        failed_count += 1
                        continue
                    sliced = self._slice_from_dataset(ds, var_name, time_idx=0, lev_idx=lev_idx)
                    sliced = self._apply_spatial_subset(sliced, lat_min, lat_max, lon_min, lon_max)
                    if sliced.ndim != 2:
                        failed_count += 1
                        continue
                    data_slices.append(sliced.values)
            except Exception:
                failed_count += 1
        
        if len(data_slices) == 0:
            return json.dumps({
                "status": "error",
                "message": f"Could not extract 2D data for {var_name} from any files in range"
            })
        
        # Compute composite
        stacked = np.stack(data_slices, axis=0)
        valid_mask = np.isfinite(stacked) & (np.abs(stacked) < 1e10)
        composite_data = np.nanmean(np.where(valid_mask, stacked, np.nan), axis=0)
        
        if not np.any(np.isfinite(composite_data)):
            return json.dumps({
                "status": "error",
                "message": "All composite data is NaN or invalid"
            })
        
        # Get coordinates (apply same subset for consistent coords)
        lat, lon = None, None
        units = ""
        with xr.open_dataset(all_files[0][1], mask_and_scale=False) as ds:
            sliced = self._apply_spatial_subset(
                self._slice_from_dataset(ds, var_name, time_idx=0, lev_idx=lev_idx),
                lat_min, lat_max, lon_min, lon_max,
            )
            lat = sliced.coords["lat"].values if "lat" in sliced.coords else np.arange(sliced.shape[0])
            lon = sliced.coords["lon"].values if "lon" in sliced.coords else np.arange(sliced.shape[1])
            units = sliced.attrs.get("units", "")
        
        # Plot
        plt.style.use("default")
        fig, ax = self._create_map_axes(figsize=(10, 7))
        
        plot_lon_min, plot_lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        plot_lat_min, plot_lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        
        mesh = ax.pcolormesh(
            lon,
            lat,
            composite_data,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **self._map_plot_kwargs(),
        )
        self._draw_borders(ax, plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)
        
        self._set_map_bounds(ax, plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        self._draw_map_grid(ax, linewidth=0.4)
        
        cbar_label = var_name if not units else f"{var_name} ({units})"
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label(cbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=10)
        
        title = f"{var_name} composite {start_date}–{end_date} (LA County)\n{len(data_slices)} days averaged"
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        
        plot_meta = json.loads(self._finalize_plot(fig, f"composite_{var_name}_{start_md[0]:02d}{start_md[1]:02d}_{end_md[0]:02d}{end_md[1]:02d}"))
        
        # Statistics
        valid_comp = composite_data[np.isfinite(composite_data) & (np.abs(composite_data) < 1e10)]
        result = {
            "status": "success",
            "message": f"Computed and plotted composite map for {var_name} ({start_date}–{end_date})",
            "variable": var_name,
            "date_range": f"{start_date} to {end_date}",
            "n_days_averaged": len(data_slices),
            "n_failed_files": failed_count,
            "stats": {
                "min": float(np.min(valid_comp)),
                "max": float(np.max(valid_comp)),
                "mean": float(np.mean(valid_comp)),
                "std": float(np.std(valid_comp)),
                "median": float(np.median(valid_comp)),
                "p05": float(np.percentile(valid_comp, 5)),
                "p25": float(np.percentile(valid_comp, 25)),
                "p75": float(np.percentile(valid_comp, 75)),
                "p95": float(np.percentile(valid_comp, 95)),
            },
        }
        result.update(plot_meta)
        return json.dumps(result)


    def get_available_variables(self) -> str:
        """
        Get list of variables that would be available after loading a dataset.
        Scans first file from each dataset to determine variables.
        """
        variables_info = {
            "status": "success",
            "message": "Available variables in datasets",
            "variables_by_dataset": {}
        }
        
        for dataset_type in ["training", "inference"]:
            dir_path = Path(self.DATA_DIRS[dataset_type])
            if not dir_path.exists():
                continue
            
            nc_files = sorted(dir_path.glob("*.nc"))
            if not nc_files:
                continue
            
            # Try to read first file to get variable names
            try:
                with xr.open_dataset(nc_files[0], mask_and_scale=False) as ds:
                    var_list = list(ds.data_vars)
                    coord_list = list(ds.coords)
                    variables_info["variables_by_dataset"][dataset_type] = {
                        "sample_file": nc_files[0].name,
                        "variables": var_list,
                        "coordinates": coord_list,
                        "n_variables": len(var_list)
                    }
            except Exception as e:
                variables_info["variables_by_dataset"][dataset_type] = {
                    "status": "error",
                    "message": f"Could not read file: {e}"
                }
        
        return json.dumps(variables_info, indent=2)


def get_dataset_metadata() -> str:
    """
    Get comprehensive metadata about available datasets: date ranges, file counts, variables, etc.
    This helps the agent understand what data is available without loading specific files.
    """
    metadata = {
        "status": "success",
        "datasets": {},
        "year_threshold": MERRA2Analyzer.YEAR_THRESHOLD,
        "note": f"Training: before {MERRA2Analyzer.YEAR_THRESHOLD}, Inference: {MERRA2Analyzer.YEAR_THRESHOLD}+"
    }
    
    for dataset_type in ["training", "inference"]:
        dir_path = Path(MERRA2Analyzer.DATA_DIRS[dataset_type])
        
        dataset_info = {
            "path": str(dir_path),
            "exists": dir_path.exists(),
        }
        
        if dir_path.exists():
            nc_files = sorted(dir_path.glob("*.nc*"))
            dataset_info["n_files"] = len(nc_files)
            
            if nc_files:
                # Extract date range
                dates = []
                for p in nc_files:
                    file_dt = MERRA2Analyzer._extract_date_from_filename(p)
                    if file_dt:
                        dates.append(file_dt)
                
                if dates:
                    dates.sort()
                    dataset_info["date_range"] = {
                        "start": dates[0].strftime("%Y-%m-%d"),
                        "end": dates[-1].strftime("%Y-%m-%d"),
                        "years": sorted(set(d.year for d in dates))
                    }
                    dataset_info["n_dates"] = len(dates)
                    
                    # Sample file info (variables, etc.)
                    try:
                        with xr.open_dataset(nc_files[0], mask_and_scale=False) as ds:
                            dataset_info["sample_file"] = nc_files[0].name
                            dataset_info["variables"] = list(ds.data_vars)
                            dataset_info["coordinates"] = list(ds.coords)
                            dataset_info["dimensions"] = dict(ds.dims)
                    except Exception as e:
                        dataset_info["sample_error"] = str(e)
        
        metadata["datasets"][dataset_type] = dataset_info
    
    return json.dumps(metadata, indent=2)


def list_available_files(dataset_type: str = "inference") -> str:
    """List all available files in a directory."""
    if dataset_type not in MERRA2Analyzer.DATA_DIRS:
        return f"Dataset type must be 'inference' or 'training'"
    
    directory = Path(MERRA2Analyzer.DATA_DIRS[dataset_type])
    if not directory.exists():
        return f"Directory not found: {directory}"
    
    nc_files = sorted(directory.glob("*.nc"))
    if not nc_files:
        return f"No .nc files found in {directory}"
    
    lines = [f"Available files in {dataset_type} ({len(nc_files)} total):"]
    for f in nc_files[-10:]:  # Show last 10 files
        lines.append(f"  • {f.name}")
    if len(nc_files) > 10:
        lines.append(f"  ... and {len(nc_files) - 10} more")
    
    return "\n".join(lines)

