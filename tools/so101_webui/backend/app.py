import json
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import datasets
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
DATA_DIR = ROOT / "data"
TASKS_FILE = DATA_DIR / "tasks.json"

HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "lerobot"
OUTPUTS_DIR = Path.cwd() / "outputs"


@dataclass
class Job:
    id: str
    name: str
    command: list[str]
    created_at: float
    process: subprocess.Popen | None = None
    status: str = "pending"
    logs: list[str] = field(default_factory=list)


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
MAX_LOG_LINES = 800


def _repo_id_to_local_root(repo_id: str) -> Path:
    return HF_CACHE_DIR / repo_id


def _local_dataset_exists(repo_id: str) -> bool:
    if not repo_id:
        return False
    root = _repo_id_to_local_root(repo_id)
    return (root / "meta" / "info.json").exists()


def _load_local_dataset(repo_id: str) -> tuple[Path, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = _repo_id_to_local_root(repo_id)
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    episodes_glob = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    data_glob = sorted((root / "data").glob("*/*.parquet"))

    if not info_path.exists():
        raise FileNotFoundError(f"Dataset not found at {root}")
    if not tasks_path.exists():
        raise FileNotFoundError(f"Task parquet not found at {tasks_path}")
    if len(episodes_glob) == 0:
        raise FileNotFoundError("Episode metadata files not found")
    if len(data_glob) == 0:
        raise FileNotFoundError("Data parquet files not found")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    tasks_df = pd.read_parquet(tasks_path)
    episodes_df = pd.concat([pd.read_parquet(p) for p in episodes_glob], ignore_index=True)
    data_ds = datasets.concatenate_datasets([datasets.Dataset.from_parquet(str(p)) for p in data_glob])
    data_df = data_ds.to_pandas()
    return root, info, tasks_df, episodes_df, data_df


def _write_local_dataset(
    root: Path,
    info: dict,
    tasks_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    data_df: pd.DataFrame,
) -> None:
    # Backup current files once before overwrite
    backup_root = root / "_webui_backup"
    backup_root.mkdir(exist_ok=True, parents=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / ts
    backup_dir.mkdir(exist_ok=True, parents=True)

    for rel in ["meta/info.json", "meta/tasks.parquet", "meta/episodes", "data"]:
        src = root / rel
        dst = backup_dir / rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        elif src.is_dir():
            if dst.exists():
                continue
            import shutil

            shutil.copytree(src, dst)

    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # Overwrite data parquet as a single chunk/file for deterministic edits
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data_ds = datasets.Dataset.from_pandas(data_df, preserve_index=False)
    data_ds.to_parquet(str(data_path))

    # Remove old extra data chunks/files
    for p in sorted((root / "data").glob("*/*.parquet")):
        if p != data_path:
            p.unlink()

    ep_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_ds = datasets.Dataset.from_pandas(episodes_df, preserve_index=False)
    ep_ds.to_parquet(str(ep_path))
    for p in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        if p != ep_path:
            p.unlink()

    tasks_df.to_parquet(root / "meta" / "tasks.parquet")
    (root / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def _episode_video_entries(info: dict, episodes_df: pd.DataFrame, root: Path, ep_idx: int) -> list[dict]:
    out: list[dict] = []
    if ep_idx < 0 or ep_idx >= len(episodes_df):
        return out
    row = episodes_df.iloc[ep_idx]

    video_path_tpl = info.get("video_path")
    if not video_path_tpl:
        return out

    cam_keys = [k for k, ft in info["features"].items() if ft["dtype"] in ("video", "image")]
    for cam in cam_keys:
        chunk_key = f"videos/{cam}/chunk_index"
        file_key = f"videos/{cam}/file_index"
        from_ts_key = f"videos/{cam}/from_timestamp"
        to_ts_key = f"videos/{cam}/to_timestamp"
        if chunk_key not in row or file_key not in row:
            continue
        rel = video_path_tpl.format(
            video_key=cam,
            chunk_index=int(row[chunk_key]),
            file_index=int(row[file_key]),
        )
        abs_path = root / rel
        out.append(
            {
                "camera": cam,
                "path": str(abs_path),
                "start_s": float(row[from_ts_key]) if from_ts_key in row else 0.0,
                "end_s": float(row[to_ts_key]) if to_ts_key in row else None,
            }
        )

    return out


def _rebuild_after_edit(
    root: Path,
    info: dict,
    tasks_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    data_df: pd.DataFrame,
    deleted_episode_indices: set[int] | None = None,
    task_updates: dict[int, str] | None = None,
) -> dict:
    deleted_episode_indices = deleted_episode_indices or set()
    task_updates = task_updates or {}

    if len(data_df) == 0:
        raise ValueError("Empty dataset cannot be edited")

    task_by_index = {int(v): str(k) for k, v in tasks_df["task_index"].to_dict().items()}
    data_df = data_df.copy()
    data_df["_task_name"] = data_df["task_index"].map(lambda x: task_by_index.get(int(x), "unknown_task"))

    keep_mask = ~data_df["episode_index"].astype(int).isin(deleted_episode_indices)
    data_df = data_df.loc[keep_mask].copy()
    if len(data_df) == 0:
        raise ValueError("All episodes would be deleted. At least one episode must remain.")

    if task_updates:
        for ep_idx, new_task in task_updates.items():
            data_df.loc[data_df["episode_index"].astype(int) == int(ep_idx), "_task_name"] = new_task

    original_episode_ids = sorted(data_df["episode_index"].astype(int).unique().tolist())
    ep_map = {old: new for new, old in enumerate(original_episode_ids)}
    data_df["episode_index"] = data_df["episode_index"].astype(int).map(ep_map)
    data_df = data_df.sort_values(["episode_index", "frame_index", "index"]).reset_index(drop=True)
    data_df["frame_index"] = data_df.groupby("episode_index").cumcount().astype("int64")
    data_df["index"] = range(len(data_df))

    used_tasks = sorted(data_df["_task_name"].astype(str).unique().tolist())
    task_name_to_index = {name: i for i, name in enumerate(used_tasks)}
    data_df["task_index"] = data_df["_task_name"].map(task_name_to_index).astype("int64")

    task_table = pd.DataFrame({"task_index": [task_name_to_index[k] for k in used_tasks]}, index=used_tasks)
    task_table.index.name = "task"

    episodes_df = episodes_df.copy()
    episodes_df["episode_index"] = episodes_df["episode_index"].astype(int)
    episodes_df = episodes_df[~episodes_df["episode_index"].isin(deleted_episode_indices)].copy()
    episodes_df["episode_index"] = episodes_df["episode_index"].map(ep_map)
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)

    lengths = data_df.groupby("episode_index").size().to_dict()
    from_to: dict[int, tuple[int, int]] = {}
    cursor = 0
    for ep in sorted(lengths):
        ln = int(lengths[ep])
        from_to[ep] = (cursor, cursor + ln)
        cursor += ln

    episodes_df["length"] = episodes_df["episode_index"].map(lambda x: int(lengths[int(x)]))
    episodes_df["dataset_from_index"] = episodes_df["episode_index"].map(lambda x: int(from_to[int(x)][0]))
    episodes_df["dataset_to_index"] = episodes_df["episode_index"].map(lambda x: int(from_to[int(x)][1]))
    episodes_df["data/chunk_index"] = 0
    episodes_df["data/file_index"] = 0
    episodes_df["tasks"] = episodes_df["episode_index"].map(
        lambda ep: sorted(data_df.loc[data_df["episode_index"] == int(ep), "_task_name"].unique().tolist())
    )

    info["total_episodes"] = int(len(lengths))
    info["total_frames"] = int(len(data_df))
    info["total_tasks"] = int(len(used_tasks))
    info["splits"] = {"train": f"0:{int(len(lengths))}"}

    data_df = data_df.drop(columns=["_task_name"])
    _write_local_dataset(root, info, task_table, episodes_df, data_df)
    return {
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "total_tasks": info["total_tasks"],
    }


def _detect_lerobot_version() -> str:
    try:
        return metadata.version("lerobot")
    except Exception:  # noqa: BLE001
        pass

    try:
        out = subprocess.check_output(
            [sys.executable, "-c", "import lerobot; print(lerobot.__version__)"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=4,
        )
        version = out.strip()
        if version:
            return version
    except Exception:  # noqa: BLE001
        pass

    version_file = Path.cwd() / "src" / "lerobot" / "__version__.py"
    if version_file.exists():
        text = version_file.read_text(encoding="utf-8")
        marker = "__version__ = "
        for line in text.splitlines():
            if line.startswith(marker):
                return line.replace(marker, "").strip().strip("'\"")

    return "unknown"


def _train_presets(version: str) -> dict:
    presets = [
        {
            "id": "debug_quick",
            "label": "빠른 디버그",
            "description": "설정/파이프라인 점검용 짧은 학습",
            "train_mode": "standard",
            "policy_type": "act",
            "train_extra_args": "--batch_size=4 --steps=2000 --save_freq=500 --log_freq=20 --eval_freq=0 --num_workers=2 --policy.push_to_hub=false",
            "peft_args": "",
        },
        {
            "id": "so101_act_stable",
            "label": "SO101 ACT 안정형",
            "description": "SO101 실기 데이터에 무난한 ACT 기본 튜닝",
            "train_mode": "standard",
            "policy_type": "act",
            "train_extra_args": "--batch_size=16 --steps=120000 --save_freq=20000 --eval_freq=0 --num_workers=4 --policy.use_amp=true --policy.push_to_hub=false",
            "peft_args": "",
        },
        {
            "id": "so101_diffusion_balanced",
            "label": "SO101 Diffusion 균형형",
            "description": "Diffusion 정책 기준 기본 안정 설정",
            "train_mode": "standard",
            "policy_type": "diffusion",
            "train_extra_args": "--batch_size=8 --steps=180000 --save_freq=30000 --eval_freq=0 --num_workers=4 --policy.use_amp=true --policy.push_to_hub=false",
            "peft_args": "",
        },
        {
            "id": "pi0_partial_ft",
            "label": "PI0 부분 미세조정",
            "description": "0.3.4에서 LoRA 대신 freeze 기반 부분 미세조정",
            "train_mode": "peft",
            "policy_type": "pi0",
            "train_extra_args": "--batch_size=8 --steps=60000 --save_freq=10000 --eval_freq=0 --num_workers=4 --policy.push_to_hub=false",
            "peft_args": "--policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.train_state_proj=true",
        },
        {
            "id": "smolvla_partial_ft",
            "label": "SmolVLA 부분 미세조정",
            "description": "0.3.4에서 LoRA 대신 freeze 기반 부분 미세조정",
            "train_mode": "peft",
            "policy_type": "smolvla",
            "train_extra_args": "--batch_size=8 --steps=60000 --save_freq=10000 --eval_freq=0 --num_workers=4 --policy.push_to_hub=false",
            "peft_args": "--policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.train_state_proj=true",
        },
    ]

    peft_supported = False
    if version != "unknown":
        # lerobot 0.3.4 기준 PEFT 전용 LoRA 인자는 기본 제공하지 않음.
        peft_supported = False

    return {
        "lerobot_version": version,
        "peft_lora_builtin_supported": peft_supported,
        "notes": [
            "lerobot 0.3.4에서는 LoRA 전용 CLI 인자가 기본 제공되지 않습니다.",
            "PEFT 프리셋은 freeze/train_expert_only 기반 부분 미세조정으로 구성됩니다.",
        ],
        "presets": presets,
    }


def _ensure_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TASKS_FILE.exists():
        TASKS_FILE.write_text("[]", encoding="utf-8")


def _read_tasks() -> list[dict]:
    _ensure_files()
    return json.loads(TASKS_FILE.read_text(encoding="utf-8"))


def _write_tasks(tasks: list[dict]) -> None:
    TASKS_FILE.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_command(name: str, command: list[str]) -> Job:
    job_id = str(uuid.uuid4())
    job = Job(id=job_id, name=name, command=command, created_at=time.time())
    with JOBS_LOCK:
        JOBS[job_id] = job

    def worker() -> None:
        job.status = "running"
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(Path.cwd()),
            )
            job.process = proc
            if proc.stdout is not None:
                for line in proc.stdout:
                    job.logs.append(line.rstrip("\n"))
                    if len(job.logs) > MAX_LOG_LINES:
                        job.logs = job.logs[-MAX_LOG_LINES:]
            code = proc.wait()
            job.status = "success" if code == 0 else f"failed({code})"
        except Exception as exc:  # noqa: BLE001
            job.logs.append(f"[launcher-error] {exc}")
            job.status = "failed(launch-error)"

    threading.Thread(target=worker, daemon=True).start()
    return job


def _list_local_datasets() -> list[dict]:
    rows: list[dict] = []

    if HF_CACHE_DIR.exists():
        for p in sorted(HF_CACHE_DIR.rglob("meta/info.json")):
            try:
                info = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            repo_id = info.get("repo_id")
            dataset_root = p.parents[1]
            rows.append(
                {
                    "name": dataset_root.name,
                    "repo_id": repo_id,
                    "path": str(dataset_root),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(dataset_root.stat().st_mtime)),
                }
            )

    if OUTPUTS_DIR.exists():
        for p in sorted(OUTPUTS_DIR.glob("**/checkpoints/last/pretrained_model")):
            rows.append(
                {
                    "name": p.parent.parent.parent.name,
                    "repo_id": None,
                    "path": str(p),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
                }
            )

    return rows


def _build_record_command(payload: dict) -> list[str]:
    dataset_repo_id = str(payload.get("dataset_repo_id", "")).strip()
    robot_port = str(payload.get("robot_port", "")).strip()
    teleop_port = str(payload.get("teleop_port", "")).strip()
    if not dataset_repo_id:
        raise ValueError("dataset_repo_id is required (예: yourname/so101_task)")
    if not robot_port:
        raise ValueError("robot_port is required")
    if not teleop_port:
        raise ValueError("teleop_port is required")

    cameras: dict[str, dict] = {}
    if payload.get("top_camera_enabled", True):
        cameras["top"] = {
            "type": payload.get("top_camera_type", "opencv"),
            "index_or_path": payload.get("top_camera_index_or_path", 0),
            "width": int(payload.get("top_camera_width", 640)),
            "height": int(payload.get("top_camera_height", 480)),
            "fps": int(payload.get("top_camera_fps", 30)),
        }
    if payload.get("front_camera_enabled", True):
        cameras["front"] = {
            "type": payload.get("front_camera_type", "opencv"),
            "index_or_path": payload.get("front_camera_index_or_path", 1),
            "width": int(payload.get("front_camera_width", 640)),
            "height": int(payload.get("front_camera_height", 480)),
            "fps": int(payload.get("front_camera_fps", 30)),
        }
    cameras_json = json.dumps(cameras, ensure_ascii=False)

    resume_flag = bool(payload.get("record_append", False))
    if resume_flag and not _local_dataset_exists(dataset_repo_id):
        # Safety: prevent 404 on first-time dataset creation.
        resume_flag = False

    cmd = [
        "lerobot-record",
        f"--dataset.repo_id={dataset_repo_id}",
        "--robot.type=so101_follower",
        f"--robot.port={robot_port}",
        f"--robot.id={payload['robot_id']}",
        f"--robot.cameras={cameras_json}",
        "--teleop.type=so101_leader",
        f"--teleop.port={teleop_port}",
        f"--teleop.id={payload['teleop_id']}",
        "--dataset.single_task=web_collected_task",
        f"--dataset.num_episodes={payload.get('num_episodes', 20)}",
        f"--dataset.episode_time_s={payload.get('episode_time_s', 30)}",
        f"--dataset.fps={payload.get('fps', 30)}",
        f"--resume={str(resume_flag).lower()}",
    ]
    extra = str(payload.get("extra_args", "")).strip()
    if extra:
        cmd.extend(shlex.split(extra, posix=False))
    return cmd


def _build_train_command(payload: dict) -> list[str]:
    dataset = str(payload.get("dataset_repo_id", "")).strip()
    if not dataset:
        raise ValueError("dataset_repo_id is required (예: yourname/so101_task)")
    job_name = payload.get("job_name", f"so101_{int(time.time())}")
    output_dir = payload.get("output_dir", f"outputs/train/{job_name}")
    policy_type = payload.get("policy_type", "act")
    device = payload.get("device", "cuda")

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={dataset}",
        f"--policy.type={policy_type}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--device={device}",
    ]

    if payload.get("train_mode", "standard") == "peft":
        peft_args = str(payload.get("peft_args", "")).strip()
        if peft_args:
            cmd.extend(shlex.split(peft_args, posix=False))

    extra = str(payload.get("extra_args", "")).strip()
    if extra:
        cmd.extend(shlex.split(extra, posix=False))
    return cmd


def _build_run_command(payload: dict) -> list[str]:
    dataset_repo_id = str(payload.get("dataset_repo_id", "")).strip()
    robot_port = str(payload.get("robot_port", "")).strip()
    policy_path = str(payload.get("policy_path", "")).strip()
    if not dataset_repo_id:
        raise ValueError("dataset_repo_id is required (예: yourname/eval_so101_task)")
    if not robot_port:
        raise ValueError("robot_port is required")
    if not policy_path:
        raise ValueError("policy_path is required")

    cameras: dict[str, dict] = {}
    if payload.get("top_camera_enabled", True):
        cameras["top"] = {
            "type": payload.get("top_camera_type", "opencv"),
            "index_or_path": payload.get("top_camera_index_or_path", 0),
            "width": int(payload.get("top_camera_width", 640)),
            "height": int(payload.get("top_camera_height", 480)),
            "fps": int(payload.get("top_camera_fps", 30)),
        }
    if payload.get("front_camera_enabled", True):
        cameras["front"] = {
            "type": payload.get("front_camera_type", "opencv"),
            "index_or_path": payload.get("front_camera_index_or_path", 1),
            "width": int(payload.get("front_camera_width", 640)),
            "height": int(payload.get("front_camera_height", 480)),
            "fps": int(payload.get("front_camera_fps", 30)),
        }
    cameras_json = json.dumps(cameras, ensure_ascii=False)

    resume_flag = bool(payload.get("record_append", False))
    if resume_flag and not _local_dataset_exists(dataset_repo_id):
        resume_flag = False

    cmd = [
        "lerobot-record",
        f"--dataset.repo_id={dataset_repo_id}",
        "--robot.type=so101_follower",
        f"--robot.port={robot_port}",
        f"--robot.id={payload['robot_id']}",
        f"--robot.cameras={cameras_json}",
        f"--control.policy.path={policy_path}",
        "--dataset.single_task=web_policy_run",
        f"--dataset.num_episodes={payload.get('num_episodes', 5)}",
        f"--dataset.episode_time_s={payload.get('episode_time_s', 30)}",
        f"--dataset.fps={payload.get('fps', 30)}",
        f"--resume={str(resume_flag).lower()}",
    ]
    extra = str(payload.get("extra_args", "")).strip()
    if extra:
        cmd.extend(shlex.split(extra, posix=False))
    return cmd


class ApiHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def _send_json(self, payload: dict | list, status: int = 200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _not_found(self):
        self._send_json({"error": "not found"}, status=404)

    def _send_file(self, path: Path):
        if not path.exists() or not path.is_file():
            return self._not_found()
        if path.suffix.lower() == ".mp4":
            ctype = "video/mp4"
        elif path.suffix.lower() == ".json":
            ctype = "application/json; charset=utf-8"
        else:
            ctype = "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_api_get(self, path: str):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if path == "/api/health":
            return self._send_json({"ok": True, "cwd": str(Path.cwd())})

        if path == "/api/meta":
            return self._send_json(_train_presets(_detect_lerobot_version()))

        if path == "/api/tasks":
            return self._send_json(_read_tasks())

        if path == "/api/datasets":
            return self._send_json(_list_local_datasets())

        if path == "/api/find-cameras":
            cmd = ["lerobot-find-cameras"]
            cam_type = query.get("type", [None])[0]
            if cam_type:
                cmd.append(cam_type)
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=30)
                return self._send_json({"ok": True, "raw": out})
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"ok": False, "error": str(exc)}, status=500)

        if path == "/api/dataset/episodes":
            repo_id = query.get("repo_id", [None])[0]
            if not repo_id:
                return self._send_json({"error": "repo_id is required"}, status=400)
            try:
                root, info, tasks_df, episodes_df, _ = _load_local_dataset(repo_id)
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"error": str(exc)}, status=400)

            task_by_index = {int(v): str(k) for k, v in tasks_df["task_index"].to_dict().items()}
            rows = []
            for _, r in episodes_df.sort_values("episode_index").iterrows():
                ep_idx = int(r["episode_index"])
                tasks = r["tasks"]
                if isinstance(tasks, str):
                    tasks = [tasks]
                rows.append(
                    {
                        "episode_index": ep_idx,
                        "length": int(r.get("length", 0)),
                        "tasks": tasks if isinstance(tasks, list) else [],
                        "videos": _episode_video_entries(info, episodes_df, root, ep_idx),
                        "primary_task": tasks[0] if isinstance(tasks, list) and len(tasks) > 0 else None,
                        "task_suggestions": sorted(task_by_index.values()),
                    }
                )
            return self._send_json(rows)

        if path == "/api/files":
            p = query.get("path", [None])[0]
            if not p:
                return self._send_json({"error": "path is required"}, status=400)
            target = Path(p).resolve()
            if HF_CACHE_DIR not in target.parents and target != HF_CACHE_DIR:
                return self._send_json({"error": "access denied"}, status=403)
            return self._send_file(target)

        if path == "/api/jobs":
            with JOBS_LOCK:
                out = [
                    {
                        "id": j.id,
                        "name": j.name,
                        "status": j.status,
                        "created_at": j.created_at,
                        "command": j.command,
                        "log_tail": j.logs[-120:],
                    }
                    for j in sorted(JOBS.values(), key=lambda x: x.created_at, reverse=True)
                ]
            return self._send_json(out)

        return self._not_found()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            return self._handle_api_get(path)

        # SPA fallback
        requested = FRONTEND_DIR / path.lstrip("/")
        if path == "/" or not requested.exists() or requested.is_dir():
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/tasks":
            body = self._read_json()
            body["id"] = str(uuid.uuid4())
            body["created_at"] = int(time.time())
            tasks = _read_tasks()
            tasks.append(body)
            _write_tasks(tasks)
            return self._send_json(body, status=201)

        if path == "/api/actions/record":
            body = self._read_json()
            try:
                cmd = _build_record_command(body)
            except ValueError as exc:
                return self._send_json({"error": str(exc)}, status=400)
            job = _run_command("record", cmd)
            return self._send_json({"job_id": job.id, "command": cmd})

        if path == "/api/actions/train":
            body = self._read_json()
            try:
                cmd = _build_train_command(body)
            except ValueError as exc:
                return self._send_json({"error": str(exc)}, status=400)
            job = _run_command("train", cmd)
            return self._send_json({"job_id": job.id, "command": cmd})

        if path == "/api/actions/run":
            body = self._read_json()
            try:
                cmd = _build_run_command(body)
            except ValueError as exc:
                return self._send_json({"error": str(exc)}, status=400)
            job = _run_command("run", cmd)
            return self._send_json({"job_id": job.id, "command": cmd})

        if path == "/api/dataset/episodes/task":
            body = self._read_json()
            repo_id = body.get("repo_id")
            ep = body.get("episode_index")
            task = str(body.get("task", "")).strip()
            if not repo_id or ep is None or not task:
                return self._send_json({"error": "repo_id, episode_index, task are required"}, status=400)
            try:
                root, info, tasks_df, episodes_df, data_df = _load_local_dataset(repo_id)
                result = _rebuild_after_edit(
                    root=root,
                    info=info,
                    tasks_df=tasks_df,
                    episodes_df=episodes_df,
                    data_df=data_df,
                    task_updates={int(ep): task},
                )
                return self._send_json({"ok": True, **result})
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"error": str(exc)}, status=500)

        if path == "/api/dataset/episodes/delete":
            body = self._read_json()
            repo_id = body.get("repo_id")
            episodes = body.get("episode_indices", [])
            if not repo_id or not isinstance(episodes, list) or len(episodes) == 0:
                return self._send_json({"error": "repo_id and episode_indices are required"}, status=400)
            try:
                root, info, tasks_df, episodes_df, data_df = _load_local_dataset(repo_id)
                result = _rebuild_after_edit(
                    root=root,
                    info=info,
                    tasks_df=tasks_df,
                    episodes_df=episodes_df,
                    data_df=data_df,
                    deleted_episode_indices={int(x) for x in episodes},
                )
                return self._send_json({"ok": True, **result})
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"error": str(exc)}, status=500)

        if path.startswith("/api/jobs/") and path.endswith("/stop"):
            job_id = path.split("/")[3]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                return self._send_json({"error": "job not found"}, status=404)
            if job.process and job.process.poll() is None:
                job.process.terminate()
                job.status = "stopped"
                job.logs.append("[system] process terminated by user")
            return self._send_json({"ok": True, "status": job.status})

        return self._not_found()

    def do_PUT(self):
        path = urlparse(self.path).path
        if not path.startswith("/api/tasks/"):
            return self._not_found()

        task_id = path.split("/")[-1]
        body = self._read_json()
        tasks = _read_tasks()
        for i, t in enumerate(tasks):
            if t.get("id") == task_id:
                body["id"] = task_id
                body["updated_at"] = int(time.time())
                tasks[i] = body
                _write_tasks(tasks)
                return self._send_json(body)

        return self._send_json({"error": "task not found"}, status=404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if not path.startswith("/api/tasks/"):
            return self._not_found()

        task_id = path.split("/")[-1]
        tasks = _read_tasks()
        next_tasks = [t for t in tasks if t.get("id") != task_id]
        if len(next_tasks) == len(tasks):
            return self._send_json({"error": "task not found"}, status=404)
        _write_tasks(next_tasks)
        return self._send_json({"ok": True})


def main() -> None:
    _ensure_files()
    host = os.getenv("SO101_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("SO101_WEB_PORT", "7070"))

    server = ThreadingHTTPServer((host, port), ApiHandler)
    print(f"SO101 web UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
