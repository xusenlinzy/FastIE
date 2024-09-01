import json
import os
import signal
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psutil
from transformers.utils.import_utils import _is_package_available
from yaml import safe_dump, safe_load

if _is_package_available("gradio"):
    import gradio as gr


DEFAULT_CONFIG_DIR = "config"
TRAINER_LOG = "trainer_log.jsonl"
TRAINING_ARGS = "training_args.yaml"
RUNNING_LOG = "running_log.txt"


def abort_process(pid: int) -> None:
    r"""
    Aborts the processes recursively in a bottom-up way.
    """
    try:
        children = psutil.Process(pid).children()
        if children:
            for child in children:
                abort_process(child.pid)

        os.kill(pid, signal.SIGABRT)
    except Exception:
        pass


def clean_cmd(args: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Removes args with NoneType or False or empty string value.
    """
    no_skip_keys = ["packing"]
    return {k: v for k, v in args.items() if (k in no_skip_keys) or (v is not None and v is not False and v != "")}


def gen_cmd(args: Dict[str, Any]) -> str:
    r"""
    Generates arguments for previewing.
    """
    cmd_lines = ["python -m fastie.cli "]
    for k, v in clean_cmd(args).items():
        cmd_lines.append("    --{} {} ".format(k, str(v)))

    if os.name == "nt":
        cmd_text = "`\n".join(cmd_lines)
    else:
        cmd_text = "\\\n".join(cmd_lines)

    cmd_text = "```bash\n{}\n```".format(cmd_text)
    return cmd_text


def save_cmd(args: Dict[str, Any]) -> str:
    r"""
    Saves arguments to launch training.
    """
    output_dir = args["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, TRAINING_ARGS), "w", encoding="utf-8") as f:
        safe_dump(clean_cmd(args), f)

    return os.path.join(output_dir, TRAINING_ARGS)


def get_eval_results(path: os.PathLike) -> str:
    r"""
    Gets scores after evaluation.
    """
    with open(path, "r", encoding="utf-8") as f:
        result = json.dumps(json.load(f), indent=4)
    return "```json\n{}\n```\n".format(result)


def get_time() -> str:
    r"""
    Gets current date and time.
    """
    return datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")


def get_trainer_info(output_path: os.PathLike) -> Tuple[str, "gr.Slider", Optional["gr.Plot"]]:
    r"""
    Gets training infomation for monitor.
    """
    running_log = ""
    running_progress = gr.Slider(visible=False)
    running_loss = None

    running_log_path = os.path.join(output_path, RUNNING_LOG)
    if os.path.isfile(running_log_path):
        with open(running_log_path, "r", encoding="utf-8") as f:
            running_log = f.read()

    trainer_log_path = os.path.join(output_path, TRAINER_LOG)
    if os.path.isfile(trainer_log_path):
        trainer_log: List[Dict[str, Any]] = []
        with open(trainer_log_path, "r", encoding="utf-8") as f:
            for line in f:
                trainer_log.append(json.loads(line))

        if len(trainer_log) != 0:
            latest_log = trainer_log[-1]
            percentage = latest_log["percentage"]
            label = "Running {:d}/{:d}: {} < {}".format(
                latest_log["current_steps"],
                latest_log["total_steps"],
                latest_log["elapsed_time"],
                latest_log["remaining_time"],
            )
            running_progress = gr.Slider(label=label, value=percentage, visible=True)

    return running_log, running_progress, running_loss


def load_args(config_path: str) -> Optional[Dict[str, Any]]:
    r"""
    Loads saved arguments.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return safe_load(f)
    except Exception:
        return None


def save_args(config_path: str, config_dict: Dict[str, Any]):
    r"""
    Saves arguments.
    """
    with open(config_path, "w", encoding="utf-8") as f:
        safe_dump(config_dict, f)


def list_config_paths(current_time: str) -> "gr.Dropdown":
    r"""
    Lists all the saved configuration files.
    """
    config_files = ["{}.yaml".format(current_time)]
    if os.path.isdir(DEFAULT_CONFIG_DIR):
        for file_name in os.listdir(DEFAULT_CONFIG_DIR):
            if file_name.endswith(".yaml") and file_name not in config_files:
                config_files.append(file_name)

    return gr.Dropdown(choices=config_files)
