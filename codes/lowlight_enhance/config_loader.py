"""
Configuration loader for HDR Gaussian Splatting training.
Generates cartesian product of all list combinations from TOML config.
"""

import hashlib
import itertools
from typing import Any, Dict, List

import toml


def generate_experiment_name(cfg: Dict[str, Any]) -> str:
    """Generate a unique experiment name from config.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Experiment name string
    """
    # Handle nested dict structure - extract data_dir from data.data_dirs
    data_dir = cfg.get("data_dir", "")
    if not data_dir:
        # Try nested structure
        data_section = cfg.get("data", {})
        if isinstance(data_section, dict):
            data_dirs = data_section.get("data_dirs", [])
            if isinstance(data_dirs, list) and len(data_dirs) > 0:
                data_dir = data_dirs[0]
            else:
                data_dir = "unknown"
        else:
            data_dir = "unknown"
    
    # Extract other params (handle nested structure)
    max_steps = cfg.get("max_steps", 0)
    if not max_steps:
        training = cfg.get("training", {})
        if isinstance(training, dict):
            max_steps = training.get("max_steps", [0])
            if isinstance(max_steps, list) and len(max_steps) > 0:
                max_steps = max_steps[0]
    
    enable_intrinsic_decomp = cfg.get("enable_intrinsic_decomp", False)
    if not isinstance(enable_intrinsic_decomp, bool):
        hdr = cfg.get("hdr", {})
        if isinstance(hdr, dict):
            enable_intrinsic_decomp = hdr.get("enable_intrinsic_decomp", [False])
            if isinstance(enable_intrinsic_decomp, list) and len(enable_intrinsic_decomp) > 0:
                enable_intrinsic_decomp = enable_intrinsic_decomp[0]
    
    curriculum_enabled = cfg.get("curriculum_enabled", False)
    if not isinstance(curriculum_enabled, bool):
        hdr = cfg.get("hdr", {})
        if isinstance(hdr, dict):
            curriculum_enabled = hdr.get("curriculum_enabled", [False])
            if isinstance(curriculum_enabled, list) and len(curriculum_enabled) > 0:
                curriculum_enabled = curriculum_enabled[0]
    
    use_nll_loss = cfg.get("use_nll_loss", False)
    if not isinstance(use_nll_loss, bool):
        loss = cfg.get("loss", {})
        if isinstance(loss, dict):
            use_nll_loss = loss.get("use_nll_loss", [False])
            if isinstance(use_nll_loss, list) and len(use_nll_loss) > 0:
                use_nll_loss = use_nll_loss[0]
    
    # Create a hash from key config parameters
    key_params = {
        "data_dir": str(data_dir),
        "max_steps": str(max_steps),
        "enable_intrinsic_decomp": str(enable_intrinsic_decomp),
        "curriculum_enabled": str(curriculum_enabled),
        "use_nll_loss": str(use_nll_loss),
    }
    
    # Create hash string
    hash_str = "_".join(f"{k}_{v}" for k, v in sorted(key_params.items()))
    hash_obj = hashlib.md5(hash_str.encode())
    hash_hex = hash_obj.hexdigest()[:8]
    
    # Extract dataset name from path
    dataset_name = data_dir.split("/")[-1] if "/" in str(data_dir) else str(data_dir)
    
    return f"{dataset_name}_{hash_hex}"


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Unflatten a dictionary.
    
    Args:
        d: Flattened dictionary
        sep: Separator for nested keys
        
    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def load_configs(config_path: str) -> List[Dict[str, Any]]:
    """Load configurations from TOML file and generate all combinations.
    
    Args:
        config_path: Path to TOML config file
        
    Returns:
        List of configuration dictionaries, one for each experiment
    """
    # Load TOML file
    with open(config_path, "r") as f:
        config_data = toml.load(f)
    
    # Flatten the config
    flat_config = flatten_dict(config_data)
    
    # Separate list parameters from scalar parameters
    # Only data_dirs should participate in cartesian product (for multiple datasets)
    # Other lists (like eval_steps, save_steps) are kept as lists but don't generate experiments
    list_params = {}
    scalar_params = {}
    
    # Parameters that should be lists but NOT participate in cartesian product
    # Check both flattened keys (e.g., 'training.eval_steps') and simple keys
    non_experiment_list_keys = {
        'eval_steps', 'save_steps', 'ply_steps',
        'training.eval_steps', 'training.save_steps',
        'other.ply_steps'
    }
    
    for key, value in flat_config.items():
        if isinstance(value, list):
            # Check if it's a nested list (like [[7000, 30000]])
            if len(value) > 0 and isinstance(value[0], list):
                # Unwrap nested lists - treat as scalar (single checkpoint list)
                scalar_params[key] = value[0]
            elif key == 'data.data_dirs' or key == 'data_dirs':
                # Only data_dirs participates in cartesian product (for multiple datasets)
                list_params[key] = value
            elif key in non_experiment_list_keys or key.endswith('.eval_steps') or key.endswith('.save_steps') or key.endswith('.ply_steps'):
                # These are lists but don't generate experiments - keep as scalar
                scalar_params[key] = value
            else:
                # Other lists - treat as scalar (shouldn't happen with new config format)
                scalar_params[key] = value
        else:
            scalar_params[key] = value
    
    # Generate cartesian product only for data_dirs
    if not list_params:
        # No data_dirs list, return single config
        configs = [scalar_params]
    else:
        # Get all combinations (only for data_dirs)
        keys = list(list_params.keys())
        values = [list_params[k] for k in keys]
        combinations = list(itertools.product(*values))
        
        # Create configs from combinations
        configs = []
        for combo in combinations:
            cfg = scalar_params.copy()
            for key, value in zip(keys, combo):
                cfg[key] = value
            configs.append(cfg)
    
    # Unflatten each config
    unflattened_configs = [unflatten_dict(cfg) for cfg in configs]
    
    # Post-process configs: unwrap single-item lists and handle special cases
    base_result_dir = config_data.get("results", {}).get("base_result_dir", "results")
    
    def unwrap_lists(d):
        """Recursively unwrap single-item lists."""
        if isinstance(d, dict):
            return {k: unwrap_lists(v) for k, v in d.items()}
        elif isinstance(d, list) and len(d) == 1:
            return unwrap_lists(d[0])
        else:
            return d
    
    processed_configs = []
    for cfg in unflattened_configs:
        # Unwrap single-item lists
        cfg = unwrap_lists(cfg)
        
        # Flatten common nested structures for easier access
        # Extract data_dir from data.data_dirs
        if "data" in cfg and isinstance(cfg["data"], dict):
            data_section = cfg["data"]
            if "data_dirs" in data_section:
                data_dirs = data_section["data_dirs"]
                if isinstance(data_dirs, list) and len(data_dirs) > 0:
                    cfg["data_dir"] = data_dirs[0]
                elif isinstance(data_dirs, str):
                    cfg["data_dir"] = data_dirs
            if "data_factors" in data_section:
                data_factors = data_section["data_factors"]
                if isinstance(data_factors, list) and len(data_factors) > 0:
                    cfg["data_factor"] = data_factors[0]
                elif isinstance(data_factors, (int, float)):
                    cfg["data_factor"] = data_factors
        
        # Extract training parameters
        if "training" in cfg and isinstance(cfg["training"], dict):
            training = cfg["training"]
            for key in ["max_steps", "batch_size", "batch_sizes", "means_lr", "scales_lr", 
                       "opacities_lr", "quats_lr", "sh0_lr", "shN_lr", "eval_steps", 
                       "save_steps", "steps_scaler"]:
                if key in training:
                    cfg[key] = training[key]
            # Handle batch_sizes -> batch_size conversion
            if "batch_sizes" in training and "batch_size" not in cfg:
                batch_sizes = training["batch_sizes"]
                if isinstance(batch_sizes, list) and len(batch_sizes) > 0:
                    cfg["batch_size"] = batch_sizes[0]
                elif isinstance(batch_sizes, (int, float)):
                    cfg["batch_size"] = batch_sizes
        
        # Extract model parameters
        if "model" in cfg and isinstance(cfg["model"], dict):
            model = cfg["model"]
            for key in ["init_type", "init_num_pts", "init_extent", "init_opa", 
                       "init_scale", "sh_degree", "sh_degree_interval", 
                       "opacity_reg", "scale_reg"]:
                if key in model:
                    cfg[key] = model[key]
        
        # Extract HDR parameters
        if "hdr" in cfg and isinstance(cfg["hdr"], dict):
            hdr = cfg["hdr"]
            for key in ["enable_intrinsic_decomp", "albedo_reg_lambda", "illum_reg_lambda",
                       "enable_synthetic_exposures", "num_synthetic_exposures",
                       "curriculum_enabled", "curriculum_max_exposure",
                       "enable_exposure_opt", "exposure_lr"]:
                if key in hdr:
                    cfg[key] = hdr[key]
        
        # Extract loss parameters
        if "loss" in cfg and isinstance(cfg["loss"], dict):
            loss = cfg["loss"]
            for key in ["use_nll_loss", "use_log_space_loss", "log_space_eps",
                       "ssim_lambda", "ratio_lambda", "sh_reg_lambda", 
                       "ratio_loss_freq", "enable_ratio_loss"]:
                if key in loss:
                    cfg[key] = loss[key]
        
        # Extract rendering parameters
        if "rendering" in cfg and isinstance(cfg["rendering"], dict):
            rendering = cfg["rendering"]
            for key in ["camera_model", "near_plane", "far_plane", "packed", 
                       "antialiased", "virtual_gain"]:
                if key in rendering:
                    cfg[key] = rendering[key]
        
        # Extract results parameters
        if "results" in cfg and isinstance(cfg["results"], dict):
            results = cfg["results"]
            for key in ["test_every", "normalize_world_space", "global_scale"]:
                if key in results:
                    cfg[key] = results[key]
        
        # Extract pruning parameters
        if "pruning" in cfg and isinstance(cfg["pruning"], dict):
            pruning = cfg["pruning"]
            for key in ["min_opacity_threshold", "min_gs_count", 
                       "pruning_protection_steps", "enable_opacity_clamping"]:
                if key in pruning:
                    cfg[key] = pruning[key]
        
        # Extract other parameters
        if "other" in cfg and isinstance(cfg["other"], dict):
            other = cfg["other"]
            for key in ["disable_viewer", "port", "save_ply", "ply_steps",
                       "disable_video", "render_traj_path", "tb_every",
                       "tb_save_image", "lpips_net"]:
                if key in other:
                    cfg[key] = other[key]
        
        # Extract logging parameters
        if "logging" in cfg and isinstance(cfg["logging"], dict):
            logging = cfg["logging"]
            for key in ["csv_columns"]:
                if key in logging:
                    cfg[key] = logging[key]
        
        # Generate experiment name
        exp_name = generate_experiment_name(cfg)
        cfg["experiment_name"] = exp_name
        cfg["result_dir"] = f"{base_result_dir}/{exp_name}"
        
        processed_configs.append(cfg)
    
    return processed_configs

