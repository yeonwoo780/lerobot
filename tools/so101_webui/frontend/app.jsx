const { useEffect, useMemo, useRef, useState } = React;

const defaultForm = {
  name: "",
  single_task: "",
  dataset_repo_id: "",
  robot_port: "",
  teleop_port: "",
  robot_id: "so101_follower_1",
  teleop_id: "so101_leader_1",
  fps: 30,
  episode_time_s: 30,
  dataset_reset_time_s: 5,
  num_episodes: 20,
  policy_type: "act",
  device: "cuda",
  peft_args: "",
  train_args: "--batch_size=8 --steps=60000 --save_freq=10000 --eval_freq=0 --num_workers=4 --policy.push_to_hub=false",
  record_args: "--dataset.single_task=web_collected_task --dataset.num_episodes=20 --dataset.episode_time_s=30 --dataset.reset_time_s=5 --dataset.fps=30 --resume=false",
  run_args: "",
  async_server_args: "--fps=30 --inference_latency=0.1 --obs_queue_timeout=1.0",
  async_client_args: "",
  async_server_host: "127.0.0.1",
  async_server_port: 8080,
  async_server_address: "127.0.0.1:8080",
  async_policy_type: "act",
  async_pretrained_name_or_path: "",
  async_policy_device: "cuda",
  async_actions_per_chunk: 200,
  async_chunk_size_threshold: 0.7,
  async_verify_robot_cameras: true,
  async_task: "",
  policy_path: "",
  record_append: false,

  top_camera_enabled: true,
  top_camera_type: "opencv",
  top_camera_index_or_path: "0",
  top_camera_width: 640,
  top_camera_height: 480,
  top_camera_fps: 30,

  front_camera_enabled: true,
  front_camera_type: "opencv",
  front_camera_index_or_path: "1",
  front_camera_width: 640,
  front_camera_height: 480,
  front_camera_fps: 30,

  wrist_camera_enabled: false,
  wrist_camera_type: "opencv",
  wrist_camera_index_or_path: "2",
  wrist_camera_width: 640,
  wrist_camera_height: 480,
  wrist_camera_fps: 30,
};

function upsertCliArg(args, key, value) {
  const safeValue = String(value);
  const quoted = /\s/.test(safeValue) ? `"${safeValue}"` : safeValue;
  const arg = `--${key}=${quoted}`;
  const text = String(args || "").trim();
  const escapedKey = String(key).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(`--${escapedKey}=(?:\"[^\"]*\"|'[^']*'|.*?)(?=\\s--[A-Za-z0-9_.-]+(?:=|\\b)|$)`);
  if (!text) return arg;
  if (re.test(text)) return text.replace(re, arg);
  return `${text} ${arg}`.trim();
}

function syncRecordArgs(recordArgs, form) {
  let out = String(recordArgs || "").trim();
  const recordTask = String(form.single_task || "").trim() || "web_collected_task";
  out = upsertCliArg(out, "dataset.single_task", recordTask);
  out = upsertCliArg(out, "dataset.num_episodes", Number(form.num_episodes));
  out = upsertCliArg(out, "dataset.episode_time_s", Number(form.episode_time_s));
  out = upsertCliArg(out, "dataset.reset_time_s", Number(form.dataset_reset_time_s));
  out = upsertCliArg(out, "dataset.fps", Number(form.fps));
  out = upsertCliArg(out, "resume", String(Boolean(form.record_append)).toLowerCase());
  return out;
}

function syncRunArgs(runArgs, form) {
  let out = String(runArgs || "").trim();
  const runTask = String(form.single_task || "").trim() || String(form.name || "").trim() || "web_policy_run";
  out = upsertCliArg(out, "dataset.single_task", runTask);
  out = upsertCliArg(out, "dataset.num_episodes", Number(form.num_episodes));
  out = upsertCliArg(out, "dataset.episode_time_s", Number(form.episode_time_s));
  out = upsertCliArg(out, "dataset.fps", Number(form.fps));
  out = upsertCliArg(out, "resume", String(Boolean(form.record_append)).toLowerCase());
  if (String(form.policy_path || "").trim()) {
    out = upsertCliArg(out, "policy.path", String(form.policy_path).trim());
  }
  return out;
}

function App() {
  const [activePage, setActivePage] = useState("dashboard");

  const [tasks, setTasks] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [form, setForm] = useState(() => ({
    ...defaultForm,
    record_args: syncRecordArgs(defaultForm.record_args, defaultForm),
    run_args: syncRunArgs(defaultForm.run_args, defaultForm),
  }));
  const [trainMode, setTrainMode] = useState("standard");
  const [lastCommand, setLastCommand] = useState("");

  const [meta, setMeta] = useState({
    lerobot_version: "unknown",
    notes: [],
    presets: [],
  });
  const [presetId, setPresetId] = useState("");
  const [realtimeTaskId, setRealtimeTaskId] = useState("");
  const [realtimeCheckpointPath, setRealtimeCheckpointPath] = useState("");

  const [datasetRepoId, setDatasetRepoId] = useState("");
  const [episodes, setEpisodes] = useState([]);
  const [selectedEpisodeIdx, setSelectedEpisodeIdx] = useState(null);
  const [episodeTaskDrafts, setEpisodeTaskDrafts] = useState({});
  const [integrityReport, setIntegrityReport] = useState(null);

  const [cameraRaw, setCameraRaw] = useState("");
  const jobListRef = useRef(null);

  const selectedTask = useMemo(() => tasks.find((t) => t.id === selectedId) || null, [tasks, selectedId]);
  const selectedPreset = useMemo(() => meta.presets.find((p) => p.id === presetId), [meta, presetId]);
  const selectedEpisode = useMemo(
    () => episodes.find((e) => e.episode_index === selectedEpisodeIdx) || null,
    [episodes, selectedEpisodeIdx]
  );
  const checkpointOptions = useMemo(
    () => datasets.filter((d) => String(d.path || "").includes("pretrained_model")),
    [datasets]
  );
  const asyncJobs = useMemo(
    () => jobs.filter((j) => j.name === "async-server" || j.name === "async-client"),
    [jobs]
  );

  function buildEpisodeVideoSrc(v, epIdx) {
    const base = `/api/files?path=${encodeURIComponent(v.path)}&ep=${encodeURIComponent(String(epIdx))}&cam=${encodeURIComponent(String(v.camera))}`;
    const start = Number(v.start_s || 0);
    if (v.end_s != null) {
      return `${base}#t=${start},${Number(v.end_s)}`;
    }
    return `${base}#t=${start}`;
  }

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  async function refreshAll() {
    const [taskRows, dataRows, jobRows, metaRow] = await Promise.all([
      api("/api/tasks"),
      api("/api/datasets"),
      api("/api/jobs"),
      api("/api/meta"),
    ]);
    setTasks(taskRows);
    setDatasets(dataRows);
    setJobs(jobRows);
    setMeta(metaRow);
  }

  useEffect(() => {
    refreshAll().catch((e) => alert(e.message));
    const timer = setInterval(() => api("/api/jobs").then(setJobs).catch(() => {}), 3000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const el = jobListRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [jobs]);

  function updateForm(patch) {
    setForm((prev) => {
      const next = { ...prev, ...patch };
      const patchKeys = Object.keys(patch || {});
      const directArgEdit = patchKeys.includes("record_args") || patchKeys.includes("run_args");
      if (!directArgEdit) {
        next.record_args = syncRecordArgs(next.record_args, next);
        next.run_args = syncRunArgs(next.run_args, next);
      }
      return next;
    });
  }

  function loadTask(t) {
    setSelectedId(t.id);
    const loaded = {
      ...defaultForm,
      ...t,
      train_args: t.train_args || t.train_extra_args || defaultForm.train_args,
      record_args: t.record_args || t.record_extra_args || defaultForm.record_args,
      run_args: t.run_args || t.run_extra_args || "",
      peft_args: t.peft_args || "",
    };
    loaded.record_args = syncRecordArgs(loaded.record_args, loaded);
    loaded.run_args = syncRunArgs(loaded.run_args, loaded);
    setForm(loaded);
    if (t.dataset_repo_id) setDatasetRepoId(t.dataset_repo_id);
  }

  function resetForm() {
    setSelectedId(null);
    setForm({
      ...defaultForm,
      record_args: syncRecordArgs(defaultForm.record_args, defaultForm),
      run_args: syncRunArgs(defaultForm.run_args, defaultForm),
    });
  }

  async function saveTask() {
    const body = {
      ...form,
      fps: Number(form.fps),
      episode_time_s: Number(form.episode_time_s),
      dataset_reset_time_s: Number(form.dataset_reset_time_s),
      num_episodes: Number(form.num_episodes),
      top_camera_width: Number(form.top_camera_width),
      top_camera_height: Number(form.top_camera_height),
      top_camera_fps: Number(form.top_camera_fps),
      front_camera_width: Number(form.front_camera_width),
      front_camera_height: Number(form.front_camera_height),
      front_camera_fps: Number(form.front_camera_fps),
      wrist_camera_width: Number(form.wrist_camera_width),
      wrist_camera_height: Number(form.wrist_camera_height),
      wrist_camera_fps: Number(form.wrist_camera_fps),
      async_server_port: Number(form.async_server_port),
      async_actions_per_chunk: Number(form.async_actions_per_chunk),
      async_chunk_size_threshold: Number(form.async_chunk_size_threshold),
    };

    if (!body.name || !body.dataset_repo_id) {
      alert("Task 이름과 dataset repo_id는 필수입니다.");
      return;
    }

    if (selectedTask) {
      await api(`/api/tasks/${selectedTask.id}`, { method: "PUT", body: JSON.stringify(body) });
    } else {
      await api("/api/tasks", { method: "POST", body: JSON.stringify(body) });
    }

    await refreshAll();
    resetForm();
  }

  async function removeTask(taskId) {
    if (!confirm("정말 삭제할까요?")) return;
    await api(`/api/tasks/${taskId}`, { method: "DELETE" });
    await refreshAll();
    if (selectedId === taskId) resetForm();
  }

  function ensureActionInputs(kind) {
    if ((kind === "record" || kind === "train" || kind === "run") && !String(form.dataset_repo_id || "").trim()) {
      alert("dataset_repo_id를 입력하세요. 예: yourname/so101_task");
      return false;
    }
    if ((kind === "record" || kind === "run" || kind === "async-client") && !String(form.robot_port || "").trim()) {
      alert("robot_port를 입력하세요.");
      return false;
    }
    if (kind === "record" && !String(form.teleop_port || "").trim()) {
      alert("teleop_port를 입력하세요.");
      return false;
    }
    if (kind === "async-client" && !String(form.async_pretrained_name_or_path || form.policy_path || "").trim()) {
      alert("async pretrained 경로 또는 policy path를 입력하세요.");
      return false;
    }
    return true;
  }

  async function startAction(kind) {
    if (!ensureActionInputs(kind)) return;

    let policyPath = String(form.policy_path || "").trim();
    if (kind === "run" && !policyPath) {
      const latest = [...datasets]
        .filter((d) => String(d.path || "").includes("checkpoints") && String(d.path || "").includes("pretrained_model"))
        .sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")))[0];
      if (!latest?.path) {
      alert("policy_path를 입력하세요.");
        return;
      }
      policyPath = latest.path;
      updateForm({ policy_path: latest.path });
    }
    if (kind === "async-client" && !String(form.async_pretrained_name_or_path || "").trim()) {
      const candidate = policyPath || ([...datasets]
        .filter((d) => String(d.path || "").includes("checkpoints") && String(d.path || "").includes("pretrained_model"))
        .sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")))[0]?.path || "");
      if (candidate) {
        updateForm({ async_pretrained_name_or_path: candidate, policy_path: policyPath || candidate });
      }
    }

    const payload = {
      ...form,
      policy_path: kind === "run" ? policyPath : form.policy_path,
      fps: Number(form.fps),
      episode_time_s: Number(form.episode_time_s),
      dataset_reset_time_s: Number(form.dataset_reset_time_s),
      num_episodes: Number(form.num_episodes),
      top_camera_width: Number(form.top_camera_width),
      top_camera_height: Number(form.top_camera_height),
      top_camera_fps: Number(form.top_camera_fps),
      front_camera_width: Number(form.front_camera_width),
      front_camera_height: Number(form.front_camera_height),
      front_camera_fps: Number(form.front_camera_fps),
      wrist_camera_width: Number(form.wrist_camera_width),
      wrist_camera_height: Number(form.wrist_camera_height),
      wrist_camera_fps: Number(form.wrist_camera_fps),
      async_server_port: Number(form.async_server_port),
      async_actions_per_chunk: Number(form.async_actions_per_chunk),
      async_chunk_size_threshold: Number(form.async_chunk_size_threshold),
    };

    if (kind === "train") {
      payload.train_mode = trainMode;
      payload.extra_args = form.train_args || "";
    }
    if (kind === "record") payload.extra_args = form.record_args || "";
    if (kind === "run") payload.extra_args = form.run_args || "";
    if (kind === "async-server") payload.extra_args = form.async_server_args || "";
    if (kind === "async-client") payload.extra_args = form.async_client_args || "";

    const res = await api(`/api/actions/${kind}`, { method: "POST", body: JSON.stringify(payload) });
    setLastCommand(res.command.join(" "));
    await refreshAll();
  }

  async function stopJob(jobId) {
    await api(`/api/jobs/${jobId}/stop`, { method: "POST" });
    await refreshAll();
  }

  function useAsPolicyPath(path) {
    if (!path) return;
    const quotedPath = String(path).includes(" ") ? `"${path}"` : String(path);
    const policyArg = `--policy.path=${quotedPath}`;
    const currentExtra = String(form.run_args || "").trim();
    const nextExtra = /--policy\.path=(?:"[^"]*"|'[^']*'|\S+)/.test(currentExtra)
      ? currentExtra.replace(/--policy\.path=(?:"[^"]*"|'[^']*'|\S+)/, policyArg)
      : `${currentExtra} ${policyArg}`.trim();

    updateForm({ policy_path: path, run_args: nextExtra });
    setLastCommand(`policy_path set: ${path}`);
  }

  function applyRealtimeTask(taskId) {
    if (!taskId) return;
    const t = tasks.find((x) => x.id === taskId);
    if (!t) return;
    loadTask(t);
    updateForm({
      async_task: String(t.single_task || "").trim() || String(t.name || "").trim(),
      async_policy_type: t.policy_type || form.async_policy_type || "act",
    });
  }

  function applyRealtimeCheckpoint(path) {
    if (!path) return;
    updateForm({
      policy_path: path,
      async_pretrained_name_or_path: path,
    });
    setLastCommand(`async pretrained set: ${path}`);
  }

  async function deleteDatasetPath(path) {
    if (!path) return;
    if (!confirm(`해당 경로를 삭제할까요?\n${path}`)) return;
    await api(`/api/datasets/path?path=${encodeURIComponent(path)}`, { method: "DELETE" });
    await refreshAll();
  }

  function applyPreset() {
    const preset = meta.presets.find((p) => p.id === presetId);
    if (!preset) return;
    setTrainMode(preset.train_mode || "standard");
    updateForm({
      policy_type: preset.policy_type || form.policy_type,
      train_args: preset.train_extra_args || form.train_args,
      peft_args: preset.peft_args || "",
    });
  }

  async function findCameras() {
    const res = await api("/api/find-cameras");
    setCameraRaw(res.raw || JSON.stringify(res));
  }

  async function loadEpisodes(repoId) {
    if (!repoId) {
      alert("repo_id를 입력하세요.");
      return;
    }
    const rows = await api(`/api/dataset/episodes?repo_id=${encodeURIComponent(repoId)}`);
    setEpisodes(rows);

    const drafts = {};
    rows.forEach((r) => {
      drafts[r.episode_index] = r.primary_task || "";
    });
    setEpisodeTaskDrafts(drafts);

    if (rows.length > 0) {
      setSelectedEpisodeIdx(rows[0].episode_index);
    } else {
      setSelectedEpisodeIdx(null);
    }
  }

  async function checkDatasetIntegrity() {
    if (!datasetRepoId) {
      alert("repo_id를 먼저 입력하세요.");
      return;
    }
    const report = await api(`/api/dataset/check-integrity?repo_id=${encodeURIComponent(datasetRepoId)}`);
    setIntegrityReport(report);
  }

  async function updateEpisodeTask(episodeIndex) {
    const task = String(episodeTaskDrafts[episodeIndex] || "").trim();
    if (!task) {
      alert("task를 입력하세요.");
      return;
    }
    await api("/api/dataset/episodes/task", {
      method: "POST",
      body: JSON.stringify({ repo_id: datasetRepoId, episode_index: episodeIndex, task }),
    });
    await loadEpisodes(datasetRepoId);
    setSelectedEpisodeIdx(episodeIndex);
  }

  async function deleteEpisode(episodeIndex) {
    if (!confirm(`episode ${episodeIndex}를 삭제할까요?`)) return;
    await api("/api/dataset/episodes/delete", {
      method: "POST",
      body: JSON.stringify({ repo_id: datasetRepoId, episode_indices: [episodeIndex] }),
    });
    await loadEpisodes(datasetRepoId);
  }

  function renderTaskPanel() {
    return (
      <section className="grid">
        <div className="panel">
          <h2>Task 설정</h2>
          <div className="form-grid">
            <label>Task 이름<input value={form.name} onChange={(e) => updateForm({ name: e.target.value })} /></label>
            <label>Single Task Prompt<input value={form.single_task} onChange={(e) => updateForm({ single_task: e.target.value })} placeholder="예: Grasp a lego block and put it in the bin." /></label>
            <label>Dataset Repo ID<input value={form.dataset_repo_id} onChange={(e) => updateForm({ dataset_repo_id: e.target.value })} /></label>
            <label>Robot Port<input value={form.robot_port} onChange={(e) => updateForm({ robot_port: e.target.value })} /></label>
            <label>Teleop Port<input value={form.teleop_port} onChange={(e) => updateForm({ teleop_port: e.target.value })} /></label>
            <label>Robot ID<input value={form.robot_id} onChange={(e) => updateForm({ robot_id: e.target.value })} /></label>
            <label>Teleop ID<input value={form.teleop_id} onChange={(e) => updateForm({ teleop_id: e.target.value })} /></label>
            <label>FPS<input type="number" value={form.fps} onChange={(e) => updateForm({ fps: e.target.value })} /></label>
            <label>Episode Time (s)<input type="number" value={form.episode_time_s} onChange={(e) => updateForm({ episode_time_s: e.target.value })} /></label>
            <label>Reset Time (s)<input type="number" value={form.dataset_reset_time_s} onChange={(e) => updateForm({ dataset_reset_time_s: e.target.value })} /></label>
            <label>Num Episodes<input type="number" value={form.num_episodes} onChange={(e) => updateForm({ num_episodes: e.target.value })} /></label>
            <label>Policy Type
              <select value={form.policy_type} onChange={(e) => updateForm({ policy_type: e.target.value })}>
                <option value="act">act</option>
                <option value="diffusion">diffusion</option>
                <option value="vqbet">vqbet</option>
                <option value="pi0">pi0</option>
                <option value="smolvla">smolvla</option>
              </select>
            </label>
            <label>Device
              <select value={form.device} onChange={(e) => updateForm({ device: e.target.value })}>
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
              </select>
            </label>
            <label>Policy Path<input value={form.policy_path} onChange={(e) => updateForm({ policy_path: e.target.value })} /></label>
          </div>

          <h2 style={{ marginTop: 12 }}>카메라 설정 (top/front/wrist)</h2>
          <div className="form-grid">
            <label>Top 사용
              <select value={String(form.top_camera_enabled)} onChange={(e) => updateForm({ top_camera_enabled: e.target.value === "true" })}>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </label>
            <label>Top Type
              <select value={form.top_camera_type} onChange={(e) => updateForm({ top_camera_type: e.target.value })}>
                <option value="opencv">opencv</option>
                <option value="realsense">realsense</option>
              </select>
            </label>
            <label>Top Index/Path<input value={form.top_camera_index_or_path} onChange={(e) => updateForm({ top_camera_index_or_path: e.target.value })} /></label>
            <label>Top W<input type="number" value={form.top_camera_width} onChange={(e) => updateForm({ top_camera_width: e.target.value })} /></label>
            <label>Top H<input type="number" value={form.top_camera_height} onChange={(e) => updateForm({ top_camera_height: e.target.value })} /></label>
            <label>Top FPS<input type="number" value={form.top_camera_fps} onChange={(e) => updateForm({ top_camera_fps: e.target.value })} /></label>

            <label>Front 사용
              <select value={String(form.front_camera_enabled)} onChange={(e) => updateForm({ front_camera_enabled: e.target.value === "true" })}>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </label>
            <label>Front Type
              <select value={form.front_camera_type} onChange={(e) => updateForm({ front_camera_type: e.target.value })}>
                <option value="opencv">opencv</option>
                <option value="realsense">realsense</option>
              </select>
            </label>
            <label>Front Index/Path<input value={form.front_camera_index_or_path} onChange={(e) => updateForm({ front_camera_index_or_path: e.target.value })} /></label>
            <label>Front W<input type="number" value={form.front_camera_width} onChange={(e) => updateForm({ front_camera_width: e.target.value })} /></label>
            <label>Front H<input type="number" value={form.front_camera_height} onChange={(e) => updateForm({ front_camera_height: e.target.value })} /></label>
            <label>Front FPS<input type="number" value={form.front_camera_fps} onChange={(e) => updateForm({ front_camera_fps: e.target.value })} /></label>

            <label>Wrist 사용
              <select value={String(form.wrist_camera_enabled)} onChange={(e) => updateForm({ wrist_camera_enabled: e.target.value === "true" })}>
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </label>
            <label>Wrist Type
              <select value={form.wrist_camera_type} onChange={(e) => updateForm({ wrist_camera_type: e.target.value })}>
                <option value="opencv">opencv</option>
                <option value="realsense">realsense</option>
              </select>
            </label>
            <label>Wrist Index/Path<input value={form.wrist_camera_index_or_path} onChange={(e) => updateForm({ wrist_camera_index_or_path: e.target.value })} /></label>
            <label>Wrist W<input type="number" value={form.wrist_camera_width} onChange={(e) => updateForm({ wrist_camera_width: e.target.value })} /></label>
            <label>Wrist H<input type="number" value={form.wrist_camera_height} onChange={(e) => updateForm({ wrist_camera_height: e.target.value })} /></label>
            <label>Wrist FPS<input type="number" value={form.wrist_camera_fps} onChange={(e) => updateForm({ wrist_camera_fps: e.target.value })} /></label>
          </div>

          <div className="row" style={{ marginTop: 10 }}>
            <button onClick={saveTask}>{selectedTask ? "Task 수정" : "Task 저장"}</button>
            <button className="secondary" onClick={resetForm}>초기화</button>
          </div>
        </div>

        <div className="panel">
          <h2>Task 목록</h2>
          <div className="task-list">
            {tasks.map((t) => (
              <div key={t.id} className="card">
                <h3>{t.name}</h3>
                <div className="meta">{t.dataset_repo_id}</div>
                <div className="meta">robot: {t.robot_port || "-"} / teleop: {t.teleop_port || "-"}</div>
                <div className="row" style={{ marginTop: 8 }}>
                  <button className="secondary" onClick={() => loadTask(t)}>불러오기</button>
                  <button className="danger" onClick={() => removeTask(t.id)}>삭제</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    );
  }

  function renderControlPanel() {
    return (
      <section className="grid" style={{ marginTop: 14 }}>
        <div className="panel">
          <h2>실행 제어</h2>

          <label>버전 맞춤 프리셋
            <select value={presetId} onChange={(e) => setPresetId(e.target.value)}>
              <option value="">선택하세요</option>
              {meta.presets.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </label>
          <div className="row" style={{ marginTop: 8 }}>
            <button className="secondary" onClick={applyPreset} disabled={!presetId}>프리셋 적용</button>
            <button className="secondary" onClick={findCameras}>카메라 검색</button>
          </div>
          {cameraRaw && <pre>{cameraRaw}</pre>}
          {selectedPreset && <div className="meta" style={{ marginTop: 6 }}>{selectedPreset.description}</div>}
          {meta.notes.map((n) => <div key={n} className="meta">- {n}</div>)}

          <label>기존 데이터셋 이어쓰기
            <select value={String(form.record_append)} onChange={(e) => updateForm({ record_append: e.target.value === "true" })}>
              <option value="false">false</option>
              <option value="true">true</option>
            </select>
          </label>

          <label>Record 인자<textarea value={form.record_args} onChange={(e) => updateForm({ record_args: e.target.value })} /></label>
          <button onClick={() => startAction("record")}>데이터 수집 시작</button>

          <label style={{ marginTop: 8 }}>학습 모드
            <select value={trainMode} onChange={(e) => setTrainMode(e.target.value)}>
              <option value="standard">Standard</option>
              <option value="peft">PEFT(부분 미세조정)</option>
            </select>
          </label>
          {trainMode === "peft" && (
            <label>PEFT 인자<textarea value={form.peft_args} onChange={(e) => updateForm({ peft_args: e.target.value })} /></label>
          )}
          <label>Train 인자<textarea value={form.train_args} onChange={(e) => updateForm({ train_args: e.target.value })} /></label>
          <button onClick={() => startAction("train")}>학습 시작</button>

          <label style={{ marginTop: 8 }}>Run 인자<textarea value={form.run_args} onChange={(e) => updateForm({ run_args: e.target.value })} /></label>
          <div className="meta">Run 인자 샘플</div>
          <pre>{"--display_data=true\n--dataset.num_episodes=3 --dataset.episode_time_s=20\n--policy.device=cuda --policy.use_amp=true"}</pre>
          <button onClick={() => startAction("run")}>학습 모델 실행</button>

          {lastCommand && <><div className="meta" style={{ marginTop: 8 }}>마지막 실행 명령</div><pre>{lastCommand}</pre></>}
        </div>

        <div className="panel">
          <h2>데이터셋/체크포인트</h2>
          <div className="data-list">
            {datasets.map((d, i) => (
              <div key={`${d.path}-${i}`} className="card">
                <h3>{d.name}</h3>
                <div className="meta">repo_id: {d.repo_id || "(local only)"}</div>
                <div className="meta">path: {d.path}</div>
                <div className="meta">updated: {d.updated_at}</div>
                <div className="row" style={{ marginTop: 8 }}>
                  {String(d.path || "").includes("pretrained_model") && (
                    <button className="secondary" onClick={() => useAsPolicyPath(d.path)}>Run 경로로 사용</button>
                  )}
                  <button className="danger" onClick={() => deleteDatasetPath(d.path)}>삭제</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    );
  }

  function renderJobs() {
    return (
      <section className="panel" style={{ marginTop: 14 }}>
        <h2>Job 로그</h2>
        <div className="job-list" ref={jobListRef}>
          {jobs.map((j) => (
            <div key={j.id} className="card">
              <div className="row">
                <h3>{j.name}</h3>
                <span className={`status ${j.status.split("(")[0]}`}>{j.status}</span>
                {j.status === "running" && <button className="danger" onClick={() => stopJob(j.id)}>중지</button>}
              </div>
              <div className="meta">cmd: {j.command.join(" ")}</div>
              <pre>{(j.log_tail || []).join("\n") || "(no logs yet)"}</pre>
            </div>
          ))}
        </div>
      </section>
    );
  }

  function renderVisualizePage() {
    return (
      <div className="viz-layout">
        <div className="panel viz-left">
          <h2>Dataset 선택</h2>
          <div className="row">
            <input value={datasetRepoId} onChange={(e) => setDatasetRepoId(e.target.value)} placeholder="repo_id (user/so101_task)" />
            <button className="secondary" onClick={() => loadEpisodes(datasetRepoId)}>불러오기</button>
            <button className="secondary" onClick={checkDatasetIntegrity}>무결성 체크</button>
          </div>

          {integrityReport && (
            <div className="card" style={{ marginTop: 8 }}>
              <div className="meta">integrity: {integrityReport.ok ? "OK" : "FAIL"}</div>
              {integrityReport.summary && (
                <div className="meta">
                  episodes={integrityReport.summary.episodes_meta_rows}, frames={integrityReport.summary.frames_data_rows}, fps={integrityReport.summary.fps}
                </div>
              )}
              {(integrityReport.issues || []).map((x, i) => (
                <div key={`issue-${i}`} className="meta" style={{ color: "#bf1d28" }}>- {x}</div>
              ))}
              {(integrityReport.warnings || []).map((x, i) => (
                <div key={`warn-${i}`} className="meta" style={{ color: "#905100" }}>- {x}</div>
              ))}
            </div>
          )}

          <h2 style={{ marginTop: 12 }}>Episodes</h2>
          <div className="episode-list">
            {episodes.map((ep) => (
              <button
                key={ep.episode_index}
                className={`episode-item ${selectedEpisodeIdx === ep.episode_index ? "active" : ""}`}
                onClick={() => setSelectedEpisodeIdx(ep.episode_index)}
              >
                <div>Episode {ep.episode_index}</div>
                <div className="meta">{ep.length} frames</div>
              </button>
            ))}
          </div>
        </div>

        <div className="panel viz-right">
          {!selectedEpisode && <div className="meta">왼쪽에서 에피소드를 선택하세요.</div>}
          {selectedEpisode && (
            <>
              <h2>Episode {selectedEpisode.episode_index}</h2>
              <div className="meta">length: {selectedEpisode.length}</div>
              <div className="meta">tasks: {(selectedEpisode.tasks || []).join(", ")}</div>

              <div className="row" style={{ marginTop: 10 }}>
                <input
                  value={episodeTaskDrafts[selectedEpisode.episode_index] || ""}
                  onChange={(e) => setEpisodeTaskDrafts((prev) => ({ ...prev, [selectedEpisode.episode_index]: e.target.value }))}
                  placeholder="task 변경"
                />
                <button className="secondary" onClick={() => updateEpisodeTask(selectedEpisode.episode_index)}>Task 변경</button>
                <button className="danger" onClick={() => deleteEpisode(selectedEpisode.episode_index)}>Episode 삭제</button>
              </div>

              {(selectedEpisode.videos || []).map((v) => (
                <div key={`${selectedEpisode.episode_index}-${v.camera}`} style={{ marginTop: 12 }}>
                  <div className="meta">camera: {v.camera}</div>
                  <video
                    controls
                    style={{ width: "100%", maxHeight: "420px", borderRadius: "8px", border: "1px solid #d8e3f0" }}
                    src={buildEpisodeVideoSrc(v, selectedEpisode.episode_index)}
                    onLoadedMetadata={(e) => {
                      try { e.currentTarget.currentTime = Number(v.start_s || 0); } catch (_) {}
                    }}
                    onTimeUpdate={(e) => {
                      if (v.end_s == null) return;
                      if (e.currentTarget.currentTime >= Number(v.end_s)) {
                        e.currentTarget.pause();
                      }
                    }}
                  />
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    );
  }

  function renderRealtimePage() {
    return (
      <section className="grid">
        <div className="panel">
          <h2>실시간 추론 제어</h2>
          <label>Task 선택
            <select value={realtimeTaskId} onChange={(e) => { setRealtimeTaskId(e.target.value); applyRealtimeTask(e.target.value); }}>
              <option value="">선택하세요</option>
              {tasks.map((t) => <option key={t.id} value={t.id}>{t.name} ({t.dataset_repo_id})</option>)}
            </select>
          </label>
          <label>Checkpoint 선택
            <select value={realtimeCheckpointPath} onChange={(e) => { setRealtimeCheckpointPath(e.target.value); applyRealtimeCheckpoint(e.target.value); }}>
              <option value="">선택하세요</option>
              {checkpointOptions.map((d, i) => <option key={`${d.path}-${i}`} value={d.path}>{d.name} :: {d.path}</option>)}
            </select>
          </label>
          <label>Server Host<input value={form.async_server_host} onChange={(e) => updateForm({ async_server_host: e.target.value })} /></label>
          <label>Server Port<input type="number" value={form.async_server_port} onChange={(e) => updateForm({ async_server_port: e.target.value })} /></label>
          <label>Server 인자<textarea value={form.async_server_args} onChange={(e) => updateForm({ async_server_args: e.target.value })} /></label>
          <div className="row" style={{ marginTop: 8 }}>
            <button onClick={() => startAction("async-server")}>Policy Server 시작</button>
          </div>

          <label style={{ marginTop: 10 }}>Server Address<input value={form.async_server_address} onChange={(e) => updateForm({ async_server_address: e.target.value })} placeholder="127.0.0.1:8080" /></label>
          <label>Async Task<input value={form.async_task} onChange={(e) => updateForm({ async_task: e.target.value })} placeholder="Pick up the eraser and place it in the box." /></label>
          <label>Policy Type<input value={form.async_policy_type} onChange={(e) => updateForm({ async_policy_type: e.target.value })} /></label>
          <label>Pretrained Name/Path<input value={form.async_pretrained_name_or_path} onChange={(e) => updateForm({ async_pretrained_name_or_path: e.target.value })} placeholder="hub repo id or local path" /></label>
          <label>Policy Device<input value={form.async_policy_device} onChange={(e) => updateForm({ async_policy_device: e.target.value })} /></label>
          <label>Actions Per Chunk<input type="number" value={form.async_actions_per_chunk} onChange={(e) => updateForm({ async_actions_per_chunk: e.target.value })} /></label>
          <label>Chunk Size Threshold<input type="number" step="0.01" value={form.async_chunk_size_threshold} onChange={(e) => updateForm({ async_chunk_size_threshold: e.target.value })} /></label>
          <label>Verify Robot Cameras
            <select value={String(form.async_verify_robot_cameras)} onChange={(e) => updateForm({ async_verify_robot_cameras: e.target.value === "true" })}>
              <option value="true">true</option>
              <option value="false">false</option>
            </select>
          </label>
          <label>Client 인자<textarea value={form.async_client_args} onChange={(e) => updateForm({ async_client_args: e.target.value })} /></label>
          <div className="row" style={{ marginTop: 8 }}>
            <button onClick={() => startAction("async-client")}>Robot Client 시작</button>
          </div>
          {lastCommand && <><div className="meta" style={{ marginTop: 8 }}>마지막 실행 명령</div><pre>{lastCommand}</pre></>}
        </div>

        <div className="panel">
          <h2>실시간 추론 로그</h2>
          <div className="job-list">
            {asyncJobs.length === 0 && <div className="meta">(no async jobs)</div>}
            {asyncJobs.map((j) => (
              <div key={j.id} className="card">
                <div className="row">
                  <h3>{j.name}</h3>
                  <span className={`status ${j.status.split("(")[0]}`}>{j.status}</span>
                  {j.status === "running" && <button className="danger" onClick={() => stopJob(j.id)}>중지</button>}
                </div>
                <div className="meta">cmd: {j.command.join(" ")}</div>
                <pre>{(j.log_tail || []).join("\n") || "(no logs yet)"}</pre>
              </div>
            ))}
          </div>
        </div>
      </section>
    );
  }

  return (
    <div className="page">
      <header className="header">
        <h1>SO101 LeRobot Control Center</h1>
        <p>lerobot version: {meta.lerobot_version}</p>
      </header>

      <div className="app-shell">
        <aside className="nav-sidebar panel">
          <button className={`nav-btn ${activePage === "dashboard" ? "active" : ""}`} onClick={() => setActivePage("dashboard")}>Dashboard</button>
          <button className={`nav-btn ${activePage === "visualize" ? "active" : ""}`} onClick={() => setActivePage("visualize")}>Dataset Visualize + Edit</button>
          <button className={`nav-btn ${activePage === "realtime" ? "active" : ""}`} onClick={() => setActivePage("realtime")}>Realtime Inference</button>
        </aside>

        <main className="main-content">
          {activePage === "dashboard" && (
            <>
              {renderTaskPanel()}
              {renderControlPanel()}
              {renderJobs()}
            </>
          )}
          {activePage === "visualize" && renderVisualizePage()}
          {activePage === "realtime" && renderRealtimePage()}
        </main>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);


