**1. Find the USB ports associated with each arm**

```bash
uv run lerobot-find-port
```

**2. connect follower and leader**

```bash
uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=COM5
```

- gripper 연결시 파워 어댑터 기준 좌좌우우좌좌 연결 필요

```bash
uv run lerobot-setup-motors --teleop.type=so101_leader --teleop.port=COM6
```

- gripper 연결시 파워 어댑터 기준 좌좌우우좌좌 연결 필요

**3. calibrate**

반드시 영상과 같은 위치에서 시작필요

```bash
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=follower
```

```bash
uv run lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM6 --teleop.id=leader
```

**4. teleoperate**

```bash
uv run lerobot-teleoperate --robot.type=so101_follower --robot.port=COM5 --robot.id=follower --robot.cameras="{ top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=COM6 --teleop.id=leader --display_data=true
```

**record dataset**

```bash
uv run lerobot-record --robot.type=so101_follower --robot.port=COM5 --robot.cameras="{front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30 }}" --robot.id=follower --teleop.type=so101_leader --teleop.port=COM6 --teleop.id=leader --dataset.repo_id=yeonwoo780/record_m --dataset.num_episodes=5 --dataset.single_task="Grab the erazer"
```

**replay dataset**

```bash
uv run lerobot-replay --robot.type=so101_follower --robot.port=COM4 --robot.id=follower --dataset.repo_id=yeonwoo780/record_m --dataset.episode=0
```

**training**

uv run lerobot-train `
   --dataset.repo_id yeonwoo780/so101-table-cleanup `
   --policy.type act `
   --output_dir outputs/train/table_clean `
   --job_name lerobot_training `
   --policy.device cuda `
   --policy.repo_id yeonwoo780/so101-table-clean-act `
   --wandb.enable false

uv run python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080

```bash
uv run python -m lerobot.async_inference.robot_client --server_address=127.0.0.1:8080 --robot.type=so101_follower --robot.port=COM4 --robot.id=follower --robot.cameras="{wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30 }, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30 }}" --task="Grab the pen and place it in the box." --policy_type=act --pretrained_name_or_path=yeonwoo780/so101-table-clean-act --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5
```

```bash
uv run python -m lerobot.async_inference.robot_client --server_address=127.0.0.1:8080 --robot.type=so101_follower --robot.port=COM4 --robot.id=follower --robot.cameras="{wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30 }, front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30 }}" --task="Grab the pen and place it in the box." --policy_type=act --pretrained_name_or_path=yeonwoo780/so101-table-clean-act --policy_device=cuda --actions_per_chunk=200 --chunk_size_threshold=0.7
```    