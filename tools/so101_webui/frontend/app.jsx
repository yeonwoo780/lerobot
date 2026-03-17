const { useEffect, useMemo, useState } = React;

const defaultForm = {
  name: "",
  dataset_repo_id: "",
  robot_port: "",
  teleop_port: "",
  robot_id: "so101_follower_1",
  teleop_id: "so101_leader_1",
  fps: 30,
  episode_time_s: 30,
  num_episodes: 20,
  policy_type: "act",
  device: "cuda",
  peft_args: "",
  train_extra_args: "",
  record_extra_args: "",
  run_extra_args: "",
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
};

function App() {
  const [tasks, setTasks] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [form, setForm] = useState(defaultForm);
  const [trainMode, setTrainMode] = useState("standard");
  const [lastCommand, setLastCommand] = useState("");
  const [meta, setMeta] = useState({
    lerobot_version: "unknown",
    peft_lora_builtin_supported: false,
    notes: [],
    presets: [],
  });
  const [presetId, setPresetId] = useState("");

  const [datasetRepoId, setDatasetRepoId] = useState("");
  const [episodes, setEpisodes] = useState([]);
  const [episodeTaskDrafts, setEpisodeTaskDrafts] = useState({});
  const [cameraRaw, setCameraRaw] = useState("");

  const selectedTask = useMemo(
    () => tasks.find((t) => t.id === selectedId) || null,
    [tasks, selectedId]
  );

  async function api(path, options = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) {
      throw new Error(await res.text());
    }
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
    const timer = setInterval(() => {
      api("/api/jobs").then(setJobs).catch(() => {});
    }, 3000);
    return () => clearInterval(timer);
  }, []);

  function loadTask(t) {
    setSelectedId(t.id);
    setForm({
      ...defaultForm,
      ...t,
      train_extra_args: t.train_extra_args || "",
      record_extra_args: t.record_extra_args || "",
      run_extra_args: t.run_extra_args || "",
      peft_args: t.peft_args || "",
    });
    if (t.dataset_repo_id) setDatasetRepoId(t.dataset_repo_id);
  }

  function resetForm() {
    setSelectedId(null);
    setForm(defaultForm);
  }

  async function saveTask() {
    const body = {
      ...form,
      fps: Number(form.fps),
      episode_time_s: Number(form.episode_time_s),
      num_episodes: Number(form.num_episodes),
      top_camera_width: Number(form.top_camera_width),
      top_camera_height: Number(form.top_camera_height),
      top_camera_fps: Number(form.top_camera_fps),
      front_camera_width: Number(form.front_camera_width),
      front_camera_height: Number(form.front_camera_height),
      front_camera_fps: Number(form.front_camera_fps),
    };

    if (!body.name || !body.dataset_repo_id) {
      alert("Task 이름과 dataset repo_id는 필수입니다.");
      return;
    }

    if (selectedTask) {
      await api(`/api/tasks/${selectedTask.id}`, {
        method: "PUT",
        body: JSON.stringify(body),
      });
    } else {
      await api("/api/tasks", {
        method: "POST",
        body: JSON.stringify(body),
      });
    }

    await refreshAll();
    resetForm();
  }

  async function removeTask(taskId) {
    if (!confirm("정말 삭제하시겠습니까?")) return;
    await api(`/api/tasks/${taskId}`, { method: "DELETE" });
    await refreshAll();
    if (selectedId === taskId) resetForm();
  }

  async function startAction(kind) {
    if (!String(form.dataset_repo_id || "").trim()) {
      alert("dataset_repo_id를 입력하세요. 예: yourname/so101_task");
      return;
    }
    if ((kind === "record" || kind === "run") && !String(form.robot_port || "").trim()) {
      alert("robot_port를 입력하세요.");
      return;
    }
    if (kind === "record" && !String(form.teleop_port || "").trim()) {
      alert("teleop_port를 입력하세요.");
      return;
    }
    if (kind === "run" && !String(form.policy_path || "").trim()) {
      alert("policy_path를 입력하세요.");
      return;
    }

    const payload = {
      ...form,
      fps: Number(form.fps),
      episode_time_s: Number(form.episode_time_s),
      num_episodes: Number(form.num_episodes),
      top_camera_width: Number(form.top_camera_width),
      top_camera_height: Number(form.top_camera_height),
      top_camera_fps: Number(form.top_camera_fps),
      front_camera_width: Number(form.front_camera_width),
      front_camera_height: Number(form.front_camera_height),
      front_camera_fps: Number(form.front_camera_fps),
    };

    if (kind === "train") {
      payload.train_mode = trainMode;
      payload.extra_args = form.train_extra_args || "";
    }
    if (kind === "record") payload.extra_args = form.record_extra_args || "";
    if (kind === "run") payload.extra_args = form.run_extra_args || "";

    const res = await api(`/api/actions/${kind}`, {
      method: "POST",
      body: JSON.stringify(payload),
    });

    setLastCommand(res.command.join(" "));
    await refreshAll();
  }

  async function stopJob(jobId) {
    await api(`/api/jobs/${jobId}/stop`, { method: "POST" });
    await refreshAll();
  }

  function applyPreset() {
    const preset = meta.presets.find((p) => p.id === presetId);
    if (!preset) return;
    setTrainMode(preset.train_mode || "standard");
    setForm((prev) => ({
      ...prev,
      policy_type: preset.policy_type || prev.policy_type,
      train_extra_args: preset.train_extra_args || "",
      peft_args: preset.peft_args || "",
    }));
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
  }

  async function updateEpisodeTask(episodeIndex) {
    const task = (episodeTaskDrafts[episodeIndex] || "").trim();
    if (!task) {
      alert("task를 입력하세요.");
      return;
    }
    await api("/api/dataset/episodes/task", {
      method: "POST",
      body: JSON.stringify({
        repo_id: datasetRepoId,
        episode_index: episodeIndex,
        task,
      }),
    });
    await loadEpisodes(datasetRepoId);
  }

  async function deleteEpisode(episodeIndex) {
    if (!confirm(`episode ${episodeIndex} 를 삭제할까요?`)) return;
    await api("/api/dataset/episodes/delete", {
      method: "POST",
      body: JSON.stringify({
        repo_id: datasetRepoId,
        episode_indices: [episodeIndex],
      }),
    });
    await loadEpisodes(datasetRepoId);
  }

  async function findCameras() {
    const res = await api("/api/find-cameras");
    setCameraRaw(res.raw || JSON.stringify(res));
  }

  const selectedPreset = meta.presets.find((p) => p.id === presetId);

  return (
    <div className="page">
      <header className="header">
        <h1>SO101 LeRobot 학습 컨트롤 센터</h1>
        <p>Task 저장, 데이터셋 확인, Record/Train/Run(PEFT 옵션 포함)을 한 화면에서 관리합니다.</p>
        <p style={{ marginTop: 6, fontSize: "0.84rem" }}>감지된 lerobot 버전: {meta.lerobot_version}</p>
      </header>

      <section className="grid">
        <div className="panel">
          <h2>Task 설정</h2>
          <div className="form-grid">
            <label>
              Task 이름
              <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </label>
            <label>
              Dataset Repo ID (`user/so101_task`)
              <input
                value={form.dataset_repo_id}
                onChange={(e) => setForm({ ...form, dataset_repo_id: e.target.value })}
              />
            </label>
            <label>
              Robot Port
              <input value={form.robot_port} onChange={(e) => setForm({ ...form, robot_port: e.target.value })} />
            </label>
            <label>
              Teleop Port
              <input value={form.teleop_port} onChange={(e) => setForm({ ...form, teleop_port: e.target.value })} />
            </label>
            <label>
              Robot ID
              <input value={form.robot_id} onChange={(e) => setForm({ ...form, robot_id: e.target.value })} />
            </label>
            <label>
              Teleop ID
              <input value={form.teleop_id} onChange={(e) => setForm({ ...form, teleop_id: e.target.value })} />
            </label>
            <label>
              FPS
              <input type="number" value={form.fps} onChange={(e) => setForm({ ...form, fps: e.target.value })} />
            </label>
            <label>
              Episode Time (s)
              <input
                type="number"
                value={form.episode_time_s}
                onChange={(e) => setForm({ ...form, episode_time_s: e.target.value })}
              />
            </label>
            <label>
              Num Episodes
              <input
                type="number"
                value={form.num_episodes}
                onChange={(e) => setForm({ ...form, num_episodes: e.target.value })}
              />
            </label>
            <label>
              Policy Type
              <select value={form.policy_type} onChange={(e) => setForm({ ...form, policy_type: e.target.value })}>
                <option value="act">act</option>
                <option value="diffusion">diffusion</option>
                <option value="vqbet">vqbet</option>
                <option value="pi0">pi0</option>
                <option value="smolvla">smolvla</option>
              </select>
            </label>
            <label>
              Device
              <select value={form.device} onChange={(e) => setForm({ ...form, device: e.target.value })}>
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
              </select>
            </label>
            <label>
              Policy Path (Run 시)
              <input
                value={form.policy_path}
                onChange={(e) => setForm({ ...form, policy_path: e.target.value })}
                placeholder="outputs/train/.../pretrained_model"
              />
            </label>
          </div>

          <h2 style={{ marginTop: 12 }}>카메라 설정 (top/front)</h2>
          <div className="form-grid">
            <label>
              Top 사용
              <select
                value={String(form.top_camera_enabled)}
                onChange={(e) => setForm({ ...form, top_camera_enabled: e.target.value === "true" })}
              >
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </label>
            <label>
              Top Type
              <select value={form.top_camera_type} onChange={(e) => setForm({ ...form, top_camera_type: e.target.value })}>
                <option value="opencv">opencv</option>
                <option value="realsense">realsense</option>
              </select>
            </label>
            <label>
              Top Index/Path
              <input
                value={form.top_camera_index_or_path}
                onChange={(e) => setForm({ ...form, top_camera_index_or_path: e.target.value })}
              />
            </label>
            <label>
              Top WxH/FPS
              <input
                value={`${form.top_camera_width}x${form.top_camera_height}@${form.top_camera_fps}`}
                onChange={(e) => {
                  const m = e.target.value.match(/(\d+)x(\d+)@(\d+)/);
                  if (!m) return;
                  setForm({
                    ...form,
                    top_camera_width: Number(m[1]),
                    top_camera_height: Number(m[2]),
                    top_camera_fps: Number(m[3]),
                  });
                }}
              />
            </label>

            <label>
              Front 사용
              <select
                value={String(form.front_camera_enabled)}
                onChange={(e) => setForm({ ...form, front_camera_enabled: e.target.value === "true" })}
              >
                <option value="true">true</option>
                <option value="false">false</option>
              </select>
            </label>
            <label>
              Front Type
              <select
                value={form.front_camera_type}
                onChange={(e) => setForm({ ...form, front_camera_type: e.target.value })}
              >
                <option value="opencv">opencv</option>
                <option value="realsense">realsense</option>
              </select>
            </label>
            <label>
              Front Index/Path
              <input
                value={form.front_camera_index_or_path}
                onChange={(e) => setForm({ ...form, front_camera_index_or_path: e.target.value })}
              />
            </label>
            <label>
              Front WxH/FPS
              <input
                value={`${form.front_camera_width}x${form.front_camera_height}@${form.front_camera_fps}`}
                onChange={(e) => {
                  const m = e.target.value.match(/(\d+)x(\d+)@(\d+)/);
                  if (!m) return;
                  setForm({
                    ...form,
                    front_camera_width: Number(m[1]),
                    front_camera_height: Number(m[2]),
                    front_camera_fps: Number(m[3]),
                  });
                }}
              />
            </label>
          </div>

          <div className="row" style={{ marginTop: 10 }}>
            <button onClick={saveTask}>{selectedTask ? "Task 수정" : "Task 저장"}</button>
            <button className="secondary" onClick={resetForm}>
              초기화
            </button>
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
                <div className="meta">cam: top={String(t.top_camera_enabled)} front={String(t.front_camera_enabled)}</div>
                <div className="row" style={{ marginTop: 8 }}>
                  <button className="secondary" onClick={() => loadTask(t)}>
                    불러오기
                  </button>
                  <button className="danger" onClick={() => removeTask(t.id)}>
                    삭제
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="grid" style={{ marginTop: 14 }}>
        <div className="panel">
          <h2>실행 제어</h2>

          <label>
            버전 맞춤 프리셋
            <select value={presetId} onChange={(e) => setPresetId(e.target.value)}>
              <option value="">선택하세요</option>
              {meta.presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>
          <div className="row" style={{ marginTop: 8 }}>
            <button className="secondary" onClick={applyPreset} disabled={!presetId}>
              프리셋 적용
            </button>
            <button className="secondary" onClick={findCameras}>
              카메라 검색
            </button>
          </div>
          {cameraRaw && <pre>{cameraRaw}</pre>}
          {selectedPreset && (
            <div className="meta" style={{ marginTop: 6 }}>
              {selectedPreset.description}
            </div>
          )}
          {meta.notes.map((n) => (
            <div key={n} className="meta" style={{ marginTop: 4 }}>
              - {n}
            </div>
          ))}

          <div style={{ height: 10 }} />

          <label>
            기존 데이터셋에 이어서 기록(append)
            <select
              value={String(form.record_append)}
              onChange={(e) => setForm({ ...form, record_append: e.target.value === "true" })}
            >
              <option value="false">false</option>
              <option value="true">true</option>
            </select>
          </label>
          <div className="meta">처음 수집이면 false, 기존 데이터셋 이어쓰기만 true</div>

          <label>
            Record 추가 인자
            <textarea
              value={form.record_extra_args}
              onChange={(e) => setForm({ ...form, record_extra_args: e.target.value })}
              placeholder="예: --display_data=true"
            />
          </label>
          <button onClick={() => startAction("record")}>데이터 수집 시작 (lerobot-record)</button>

          <div style={{ height: 8 }} />

          <label>
            학습 모드
            <select value={trainMode} onChange={(e) => setTrainMode(e.target.value)}>
              <option value="standard">Standard</option>
              <option value="peft">PEFT(부분 미세조정)</option>
            </select>
          </label>

          {trainMode === "peft" && (
            <label>
              PEFT 인자 (0.3.4: freeze 기반)
              <textarea
                value={form.peft_args}
                onChange={(e) => setForm({ ...form, peft_args: e.target.value })}
                placeholder="예: --policy.freeze_vision_encoder=true --policy.train_expert_only=true"
              />
            </label>
          )}

          <label>
            Train 추가 인자
            <textarea
              value={form.train_extra_args}
              onChange={(e) => setForm({ ...form, train_extra_args: e.target.value })}
              placeholder="예: --batch_size=32 --steps=100000"
            />
          </label>
          <button onClick={() => startAction("train")}>학습 시작 (lerobot-train)</button>

          <div style={{ height: 8 }} />

          <label>
            Run 추가 인자
            <textarea
              value={form.run_extra_args}
              onChange={(e) => setForm({ ...form, run_extra_args: e.target.value })}
              placeholder="예: --control.policy.use_amp=true"
            />
          </label>
          <button onClick={() => startAction("run")}>학습 모델 실행 (lerobot-record + policy)</button>

          {lastCommand && (
            <>
              <div className="meta" style={{ marginTop: 10 }}>
                마지막 실행 명령
              </div>
              <pre>{lastCommand}</pre>
            </>
          )}
        </div>

        <div className="panel">
          <h2>데이터셋/체크포인트 목록</h2>
          <div className="data-list">
            {datasets.map((d, i) => (
              <div key={`${d.path}-${i}`} className="card">
                <h3>{d.name}</h3>
                <div className="meta">repo_id: {d.repo_id || "(local only)"}</div>
                <div className="meta">path: {d.path}</div>
                <div className="meta">updated: {d.updated_at}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="panel" style={{ marginTop: 14 }}>
        <h2>Dataset Visualize + Edit</h2>
        <div className="row">
          <input
            placeholder="repo_id (예: yourname/so101_task)"
            value={datasetRepoId}
            onChange={(e) => setDatasetRepoId(e.target.value)}
          />
          <button className="secondary" onClick={() => loadEpisodes(datasetRepoId)}>
            에피소드 불러오기
          </button>
        </div>
        <div className="job-list" style={{ marginTop: 10 }}>
          {episodes.map((ep) => (
            <div key={ep.episode_index} className="card">
              <h3>Episode {ep.episode_index}</h3>
              <div className="meta">length: {ep.length} frames</div>
              <div className="meta">tasks: {(ep.tasks || []).join(", ")}</div>
              <div className="row" style={{ marginTop: 8 }}>
                <input
                  value={episodeTaskDrafts[ep.episode_index] || ""}
                  onChange={(e) =>
                    setEpisodeTaskDrafts({
                      ...episodeTaskDrafts,
                      [ep.episode_index]: e.target.value,
                    })
                  }
                  placeholder="새 task 명"
                />
                <button className="secondary" onClick={() => updateEpisodeTask(ep.episode_index)}>
                  Task 변경
                </button>
                <button className="danger" onClick={() => deleteEpisode(ep.episode_index)}>
                  Episode 삭제
                </button>
              </div>
              {(ep.videos || []).map((v) => (
                <div key={`${ep.episode_index}-${v.camera}`} style={{ marginTop: 8 }}>
                  <div className="meta">
                    cam: {v.camera} (start {Number(v.start_s || 0).toFixed(2)}s
                    {v.end_s != null ? ` / end ${Number(v.end_s).toFixed(2)}s` : ""})
                  </div>
                  <video
                    controls
                    style={{ width: "100%", maxHeight: "340px", borderRadius: "8px", border: "1px solid #d8e3f0" }}
                    src={`/api/files?path=${encodeURIComponent(v.path)}`}
                    onLoadedMetadata={(e) => {
                      try {
                        e.currentTarget.currentTime = Number(v.start_s || 0);
                      } catch (_) {}
                    }}
                  />
                </div>
              ))}
            </div>
          ))}
        </div>
      </section>

      <section className="panel" style={{ marginTop: 14 }}>
        <h2>Job 로그</h2>
        <div className="job-list">
          {jobs.map((j) => (
            <div key={j.id} className="card">
              <div className="row">
                <h3>{j.name}</h3>
                <span className={`status ${j.status.split("(")[0]}`}>{j.status}</span>
                {j.status === "running" && (
                  <button className="danger" onClick={() => stopJob(j.id)}>
                    중지
                  </button>
                )}
              </div>
              <div className="meta">cmd: {j.command.join(" ")}</div>
              <pre>{(j.log_tail || []).join("\n") || "(no logs yet)"}</pre>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
