# SO101 LeRobot Web UI

SO101 데이터 수집/관리/학습/실행을 브라우저에서 제어하기 위한 웹앱입니다.

## 기능

- Task CRUD: `dataset_repo_id`, 포트, episode/fps, policy 설정 저장/수정/삭제
- 카메라 설정: Task 단위로 `top`, `front` 카메라 type/index/path/해상도/fps 설정
- 카메라 검색: `lerobot-find-cameras` 실행 결과 확인
- 데이터 확인: 로컬 Hugging Face cache 및 학습 체크포인트 목록
- Record 실행: `lerobot-record` 실행 (`--robot.cameras` 자동 구성)
- Train 실행: `lerobot-train` 실행
- 버전 맞춤 프리셋: 설치된 `lerobot` 버전 감지 후 학습 프리셋 제공
- PEFT 모드: 학습 버튼에서 `PEFT 인자` 전달
- Run 실행: `lerobot-record --control.policy.path=...` 형태로 학습 모델 실행
- Dataset Visualize + Edit:
  - 에피소드 단위 영상 확인(top/front)
  - 에피소드 task 변경
  - 에피소드 삭제
- Job 모니터링: 실행 명령/상태/로그 tail 확인, 실행중 프로세스 중지

## 실행

### 방법 1

```powershell
python tools/so101_webui/backend/app.py
```

### 방법 2

```powershell
.\start_so101_webui.ps1
```

브라우저에서 `http://127.0.0.1:7070` 접속.

## 참고한 LeRobot 공식 흐름

- SO101 teleoperate/record/train/eval 흐름: <https://huggingface.co/docs/lerobot/il_robots>
- SO101 가이드: <https://huggingface.co/docs/lerobot/so101>
- Dataset 시각화 CLI: <https://huggingface.co/docs/lerobot/lerobot-dataset-viz>

## 주의

- 본 UI의 PEFT는 `lerobot-train`에 사용자가 입력한 인자를 그대로 추가 전달하는 방식입니다.
- 현재 환경 감지 버전이 `0.3.4`인 경우, LoRA 전용 CLI 인자는 기본 제공되지 않아 freeze 기반 부분 미세조정 프리셋을 사용합니다.
- 에피소드 삭제/수정 시 데이터/메타를 재작성하며, 기존 파일은 `dataset_root/_webui_backup/<timestamp>`에 백업됩니다.
