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
uv run lerobot-setup-motors --teleop.type=so101_leader --teleop.port=COM4
```

- gripper 연결시 파워 어댑터 기준 좌좌우우좌좌 연결 필요

**3. calibrate**

반드시 영상과 같은 위치에서 시작필요

```bash
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=COM5 --robot.id=follower
```

```bash
uv run lerobot-calibrate --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=leader
```

**4. teleoperate**

```bash
uv run lerobot-teleoperate --robot.type=so101_follower --robot.port=COM5 --robot.id=follower --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=leader --display_data=true
```

**record dataset**

```bash
uv run lerobot-record --robot.type=so101_follower --robot.port=COM5 --robot.cameras="{front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30 }}" --robot.id=follower --teleop.type=so101_leader --teleop.port=COM4 --teleop.id=leader --dataset.repo_id=yeonwoo780/record-test --dataset.num_episodes=4 --dataset.single_task="Grab the erazer"
```

**replay dataset**

```bash
uv run lerobot-replay --robot.type=so101_follower --robot.port=COM5 --robot.id=follower --dataset.repo_id=yeonwoo780/record-test --dataset.episode=0
```