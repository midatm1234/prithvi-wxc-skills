#!/usr/bin/env python3
"""
MERRA2 MCP tools — thin orchestrator.

Imports from the focused sub-modules and wires them together via
process_tool_call(), which is the single entry-point used by mcp_server.py.

Sub-modules
-----------
config.py       Shared path constants (_SCRIPT_DIR, _DEFAULT_CONFIG, _GRANITE_PYTHON)
job_manager.py  JobManager, start_*_job helpers, _job_manager singleton
yaml_tools.py   read_yaml_config, create_custom_yaml, list_available_configs
analyzer.py     MERRA2Analyzer, get_dataset_metadata, list_available_files
"""

import json
import subprocess
import sys
from typing import Any, Dict

from config import _SCRIPT_DIR, _DEFAULT_CONFIG
from job_manager import (
    _job_manager,
    start_inference_job,
    start_preprocessing_job,
    start_compute_scalars_job,
    start_download_job,
    check_raw_data_status,
    run_training_pipeline,
    start_evaluation_job,
    create_refinement_config,
    start_refinement_inference_job,
    start_refinement_evaluation_job,
    setup_code,
    setup_training_env,
    preflight_check,
)
from provenance import get_run_manifest, list_run_manifests, replay_run
from setup_tools import check_environment
from yaml_tools import read_yaml_config, create_custom_yaml, list_available_configs
from analyzer import MERRA2Analyzer, get_dataset_metadata, list_available_files

def process_tool_call(tool_name: str, arguments: Dict[str, Any], analyzer: MERRA2Analyzer) -> str:
    """Process MCP tool calls."""
    
    if tool_name == "get_dataset_metadata":
        return get_dataset_metadata()
    
    elif tool_name == "get_available_variables":
        return analyzer.get_available_variables()
    
    elif tool_name == "load_by_date":
        date_str = arguments.get("date", "")
        # dataset_type is optional - will auto-determine based on year
        dataset_type = arguments.get("dataset_type")
        return analyzer.load_by_date(date_str, dataset_type)
    
    elif tool_name == "find_file_by_date":
        date_str = arguments.get("date", "")
        dataset_type = arguments.get("dataset_type")
        return analyzer.find_file_by_date(date_str, dataset_type)
    
    elif tool_name == "load_dataset":
        filepath = arguments.get("filepath", "")
        return analyzer.load_dataset(filepath)
    
    elif tool_name == "list_files":
        dataset_type = arguments.get("dataset_type", "inference")
        return list_available_files(dataset_type)
    
    elif tool_name == "dataset_summary":
        return analyzer.dataset_summary()
    
    elif tool_name == "variable_info":
        return analyzer.variable_info(
            arguments.get("var_name", ""),
            arguments.get("lat_min"),
            arguments.get("lat_max"),
            arguments.get("lon_min"),
            arguments.get("lon_max"),
            arguments.get("region"),
        )
    
    elif tool_name == "slice_variable":
        return analyzer.slice_variable(
            arguments.get("var_name", ""),
            arguments.get("time_idx", 0),
            arguments.get("lev_idx", 0),
            arguments.get("lat_min"),
            arguments.get("lat_max"),
            arguments.get("lon_min"),
            arguments.get("lon_max"),
            arguments.get("region"),
        )
    
    elif tool_name == "list_variables":
        return analyzer.list_variables()
    
    elif tool_name == "list_coordinates":
        return analyzer.list_coordinates()

    elif tool_name == "plot_variable_2d":
        return analyzer.plot_variable_2d(
            arguments.get("var_name", ""),
            arguments.get("time_idx", 0),
            arguments.get("lev_idx", 0),
            arguments.get("cmap", "viridis"),
            arguments.get("vmin"),
            arguments.get("vmax"),
            arguments.get("lat_min"),
            arguments.get("lat_max"),
            arguments.get("lon_min"),
            arguments.get("lon_max"),
            arguments.get("region"),
        )

    elif tool_name == "plot_data":
        return analyzer.plot_data(
            arguments.get("var_name", ""),
            arguments.get("plot_type", "map"),
            arguments.get("time_idx", 0),
            arguments.get("lev_idx", 0),
            arguments.get("cmap", "viridis"),
            arguments.get("bins", 40),
            arguments.get("line_axis", "lon"),
            arguments.get("vmin"),
            arguments.get("vmax"),
            arguments.get("lat_min"),
            arguments.get("lat_max"),
            arguments.get("lon_min"),
            arguments.get("lon_max"),
            arguments.get("region"),
        )

    elif tool_name == "generate_workflow_image":
        return analyzer.generate_workflow_image(
            arguments.get("process_type", "general"),
            arguments.get("title"),
            arguments.get("steps"),
            arguments.get("detail_level"),
            arguments.get("focus"),
            arguments.get("request_text"),
            arguments.get("config_path"),
        )

    elif tool_name == "compute_statistics":
        return analyzer.compute_statistics(
            arguments.get("var_name", ""),
            arguments.get("time_idx", 0),
            arguments.get("lev_idx", 0),
            arguments.get("lat_min"),
            arguments.get("lat_max"),
            arguments.get("lon_min"),
            arguments.get("lon_max"),
            arguments.get("region"),
        )

    elif tool_name == "compare_dates_difference_map":
        return analyzer.compare_dates_difference_map(
            arguments.get("date_1", ""),
            arguments.get("date_2", ""),
            arguments.get("var_name", ""),
            arguments.get("time_idx", 0),
            arguments.get("lev_idx", 0),
            arguments.get("lat_min"),
            arguments.get("lat_max"),
            arguments.get("lon_min"),
            arguments.get("lon_max"),
            arguments.get("region"),
        )

    elif tool_name == "compute_trend_timeseries":
        return analyzer.compute_trend_timeseries(
            arguments.get("var_name", ""),
            arguments.get("start_date", ""),
            arguments.get("end_date", ""),
            arguments.get("frequency", "monthly"),
            arguments.get("lev_idx", 0),
            arguments.get("spatial_stat", "mean"),
            arguments.get("lat_min"),
            arguments.get("lat_max"),
            arguments.get("lon_min"),
            arguments.get("lon_max"),
            arguments.get("region"),
        )

    elif tool_name == "compute_monthly_climatology":
        return analyzer.compute_monthly_climatology(
            arguments.get("var_name", ""),
            arguments.get("month", 1),
            arguments.get("lev_idx", 0),
            arguments.get("cmap", "viridis"),
            arguments.get("vmin"),
            arguments.get("vmax"),
            arguments.get("start_year"),
            arguments.get("end_year"),
            arguments.get("lat_min"),
            arguments.get("lat_max"),
            arguments.get("lon_min"),
            arguments.get("lon_max"),
            arguments.get("region"),
        )

    elif tool_name == "compute_seasonal_climatology":
        return analyzer.compute_seasonal_climatology(
            arguments.get("var_name", ""),
            arguments.get("season", "JJA"),
            arguments.get("lev_idx", 0),
            arguments.get("cmap", "viridis"),
            arguments.get("vmin"),
            arguments.get("vmax"),
            arguments.get("start_year"),
            arguments.get("end_year"),
            arguments.get("lat_min"),
            arguments.get("lat_max"),
            arguments.get("lon_min"),
            arguments.get("lon_max"),
            arguments.get("region"),
        )

    elif tool_name == "compute_period_composite":
        return analyzer.compute_period_composite(
            arguments.get("var_name", ""),
            arguments.get("start_date", ""),
            arguments.get("end_date", ""),
            arguments.get("lev_idx", 0),
            arguments.get("cmap", "viridis"),
            arguments.get("vmin"),
            arguments.get("vmax"),
            arguments.get("start_year"),
            arguments.get("end_year"),
            arguments.get("lat_min"),
            arguments.get("lat_max"),
            arguments.get("lon_min"),
            arguments.get("lon_max"),
            arguments.get("region"),
        )

    elif tool_name == "github_repo_context":
        return analyzer.github_repo_context(
            arguments.get("query", ""),
            arguments.get("repo", MERRA2Analyzer.DEFAULT_REPO),
            arguments.get("ref", MERRA2Analyzer.DEFAULT_REF),
            arguments.get("path_prefix", MERRA2Analyzer.DEFAULT_PATH),
            int(arguments.get("max_files", 6)),
        )

    # ------------------------------------------------------------------
    # Job management tools
    # ------------------------------------------------------------------

    elif tool_name == "start_inference_job":
        return start_inference_job(
            config_path=arguments.get("config_path", _DEFAULT_CONFIG),
            checkpoint=arguments.get("checkpoint"),
            batch_size=int(arguments.get("batch_size", 1)),
            depends_on=arguments.get("depends_on"),
        )

    elif tool_name == "start_preprocessing_job":
        return start_preprocessing_job(
            config_path=arguments.get("config_path", _DEFAULT_CONFIG),
            mode=arguments.get("mode", "both"),
            depends_on=arguments.get("depends_on"),
        )

    elif tool_name == "start_compute_scalars_job":
        return start_compute_scalars_job(
            config_path=arguments.get("config_path", _DEFAULT_CONFIG),
            depends_on=arguments.get("depends_on"),
        )

    elif tool_name == "start_download_job":
        return start_download_job(
            dataset=arguments.get("dataset", "all"),
            start_date=arguments.get("start_date"),
            end_date=arguments.get("end_date"),
            config_path=arguments.get("config_path"),
            output_dir=arguments.get("output_dir"),
            elevation_file=arguments.get("elevation_file"),
            variables=arguments.get("variables"),
            overwrite=bool(arguments.get("overwrite", False)),
            domains=arguments.get("domains"),
            variant=arguments.get("variant"),
            depends_on=arguments.get("depends_on"),
        )

    elif tool_name == "check_raw_data_status":
        return check_raw_data_status()

    elif tool_name == "check_environment":
        return check_environment(bool(arguments.get("include_training_python", True)))

    elif tool_name == "setup_code":
        return setup_code(arguments.get("variant"), arguments.get("depends_on"))

    elif tool_name == "setup_training_env":
        return setup_training_env(arguments.get("depends_on"))

    elif tool_name == "preflight_check":
        return preflight_check(arguments.get("config_path", _DEFAULT_CONFIG), arguments.get("stage", "training"))

    elif tool_name == "get_run_manifest":
        return get_run_manifest(arguments.get("job_id", ""), arguments.get("path", ""),
                                bool(arguments.get("include_config_text", False)))

    elif tool_name == "list_run_manifests":
        return list_run_manifests(int(arguments.get("limit", 20)))

    elif tool_name == "replay_run":
        return replay_run(arguments.get("manifest_path", ""), arguments.get("job_id", ""),
                          bool(arguments.get("dry_run", False)))

    elif tool_name == "get_job_status":
        return _job_manager.get_status(arguments.get("job_id", ""))

    elif tool_name == "list_jobs":
        return _job_manager.list_jobs()

    elif tool_name == "cancel_job":
        return _job_manager.cancel_job(arguments.get("job_id", ""))

    elif tool_name == "read_yaml_config":
        return read_yaml_config(
            config_path=arguments.get("config_path", _DEFAULT_CONFIG),
        )

    elif tool_name == "list_available_configs":
        return json.dumps({"status": "ok", "configs": list_available_configs()})

    elif tool_name == "create_custom_yaml":
        return create_custom_yaml(
            base_config=arguments.get("base_config", _DEFAULT_CONFIG),
            output_name=arguments.get("output_name"),
            lat_min=arguments.get("lat_min"),
            lat_max=arguments.get("lat_max"),
            lon_min=arguments.get("lon_min"),
            lon_max=arguments.get("lon_max"),
            num_epochs=arguments.get("num_epochs"),
            batch_size=arguments.get("batch_size"),
            learning_rate=arguments.get("learning_rate"),
            training_start=arguments.get("training_start"),
            training_end=arguments.get("training_end"),
            inference_start=arguments.get("inference_start"),
            inference_end=arguments.get("inference_end"),
            validation_start=arguments.get("validation_start"),
            validation_end=arguments.get("validation_end"),
            num_gpus=arguments.get("num_gpus"),
            predictor_variables=arguments.get("predictor_variables"),
            target_variables=arguments.get("target_variables"),
            case_name=arguments.get("case_name"),
            extra_overrides=arguments.get("extra_overrides"),
            localize_paths=bool(arguments.get("localize_paths", True)),
        )

    elif tool_name == "run_training_pipeline":
        return run_training_pipeline(
            config_path=arguments.get("config_path", _DEFAULT_CONFIG),
            num_gpus=int(arguments.get("num_gpus", 1)),
            save_every=int(arguments.get("save_every", 5)),
            queue_inference=bool(arguments.get("queue_inference", True)),
            evaluate=bool(arguments.get("evaluate", True)),
            train_phase1=bool(arguments.get("train_phase1", True)),
            refinement_type=arguments.get("refinement_type") or None,
            refiner_attention=bool(arguments.get("refiner_attention", True)),
            ensemble_size=arguments.get("ensemble_size"),
            refinement_epochs=arguments.get("refinement_epochs"),
        )

    elif tool_name == "start_evaluation_job":
        return start_evaluation_job(
            config_path=arguments.get("config_path", _DEFAULT_CONFIG),
            run_label=arguments.get("run_label"),
            depends_on=arguments.get("depends_on"),
        )

    elif tool_name == "create_refinement_config":
        return create_refinement_config(
            base_config=arguments.get("base_config", ""),
            refinement_type=arguments.get("refinement_type", ""),
            refiner_attention=bool(arguments.get("refiner_attention", True)),
        )

    elif tool_name == "start_refinement_inference_job":
        return start_refinement_inference_job(
            config_path=arguments.get("config_path", ""),
            ensemble_size=arguments.get("ensemble_size"),
            num_gpus=int(arguments.get("num_gpus", 1)),
            split=arguments.get("split", "inference"),
            depends_on=arguments.get("depends_on"),
        )

    elif tool_name == "start_refinement_evaluation_job":
        return start_refinement_evaluation_job(
            config_path=arguments.get("config_path", ""),
            split=arguments.get("split", "inference"),
            depends_on=arguments.get("depends_on"),
        )

    elif tool_name == "get_gpu_status":
        return get_gpu_status()

    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


def get_gpu_status() -> str:
    """Return GPU utilization, VRAM, temperature and process info via nvidia-smi."""
    try:
        gpu_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in gpu_out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "util_pct": int(parts[2].replace(" %", "")),
                    "mem_used_mib": int(parts[3].replace(" MiB", "")),
                    "mem_total_mib": int(parts[4].replace(" MiB", "")),
                    "temp_c": int(parts[5]),
                })

        uuid_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        uuid_map = {}
        for line in uuid_out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                uuid_map[parts[1]] = int(parts[0])

        proc_out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        processes = []
        for line in proc_out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                processes.append({
                    "gpu_index": uuid_map.get(parts[0], -1),
                    "pid": int(parts[1]),
                    "process": parts[2].split("/")[-1],
                    "mem_mib": int(parts[3].replace(" MiB", "")),
                })

        summary_lines = []
        for g in gpus:
            mem_pct = round(g["mem_used_mib"] / g["mem_total_mib"] * 100)
            summary_lines.append(
                f"GPU {g['index']} ({g['name']}): util={g['util_pct']}%, "
                f"VRAM={g['mem_used_mib']}/{g['mem_total_mib']} MiB ({mem_pct}%), "
                f"temp={g['temp_c']}°C"
            )

        return json.dumps({
            "status": "ok",
            "gpus": gpus,
            "processes": processes,
            "summary": "\n".join(summary_lines),
        })
    except FileNotFoundError:
        return json.dumps({"status": "error", "message": "nvidia-smi not found"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


def main():
    """Main MCP server - reads JSON from stdin."""
    analyzer = MERRA2Analyzer()
    
    while True:
        try:
            line = input()
            if not line.strip():
                continue
            
            request = json.loads(line)
            tool_name = request.get("tool")
            arguments = request.get("arguments", {})
            
            result = process_tool_call(tool_name, arguments, analyzer)
            
            response = {
                "success": True,
                "result": result
            }
            print(json.dumps(response))
            sys.stdout.flush()
        
        except json.JSONDecodeError as e:
            print(json.dumps({"success": False, "error": f"JSON parse error: {e}"}))
        except Exception as e:
            print(json.dumps({"success": False, "error": str(e)}))


if __name__ == "__main__":
    main()
