# MonoDOR — 공개 코드 및 환경 재현 안내

**MonoDOR**는 *Depth- and Occupancy-Aware Representation Learning for Monocular 3D Object Detection*을 나타내는 코드 공개명입니다.

논문: **Monocular 3D Detection under Occlusion via Depth and Occupancy-Aware Representation Learning**

저자: Heetae Han, Sang Min Yoon

이 저장소는 VMamba 기반 monocular 3D detection의 DAE(depth-aware enhancement), OAE(occupancy-aware enhancement) 구현과 실행 환경을 제공합니다. **IEEE Transactions on Intelligent Vehicles(T-IV) 게재 승인** 상태이며, 온라인 논문 페이지와 최종 서지 정보는 공개 후 추가합니다. 이번 배포에는 논문 PDF를 포함하지 않습니다.

## 제공 범위

- 실제 개발 환경을 참고한 Python 3.10 / PyTorch 2.1.0 / CUDA 11.8 구성
- Docker 이미지 빌드 파일과 GPU 실행용 Compose 설정
- 주요 의존성 버전 고정 및 CUDA extension 설치 스크립트
- 절대경로를 제거한 공개용 모델 YAML
- 데이터 파일 검사, train/val 메타데이터 생성, 학습·평가 실행 스크립트
- CUDA 연산 검사 및 실제 KITTI 프레임의 모델 순전파 검사
- 논문 결과 요약 표와 검증 범위를 구분한 설명
- 원고·리뷰 자료·데이터·가중치를 제외하는 소스 내보내기 도구

코드의 학습 supervision에는 LiDAR-derived depth와 occupancy target이 사용됩니다. 모델 추론 입력은 RGB와 calibration이지만, 현재 검증용 데이터 로더도 point/depth 파일을 읽으므로 데이터 준비 시 함께 제공해야 합니다.

## 빠른 시작

저장소 루트에서 실행합니다.

```bash
cp docker/.env.example docker/.env
```

`docker/.env`의 세 경로를 실제 호스트 절대경로로 수정합니다.

| 변수 | 의미 |
|---|---|
| `KITTI_ROOT` | `ImageSets`, `training`, 메타데이터 pickle이 있는 KITTI 경로 |
| `CHECKPOINT_ROOT` | ADE20K 초기화 checkpoint를 넣은 경로 |
| `OUTPUT_ROOT` | 학습 checkpoint와 로그를 저장할 경로 |
| `TORCH_CUDA_ARCH_LIST` | 빌드 대상 GPU 아키텍처, 기본 `8.0`은 A100 |
| `MAX_JOBS` | CUDA extension 빌드 병렬 작업 수, 기본 2 |

```bash
docker compose --env-file docker/.env -f docker/compose.yaml build
docker compose --env-file docker/.env -f docker/compose.yaml run --rm research \
  python scripts/release/check_environment.py --cuda
docker compose --env-file docker/.env -f docker/compose.yaml run --rm research \
  python scripts/release/prepare_kitti.py
docker compose --env-file docker/.env -f docker/compose.yaml run --rm research \
  bash scripts/release/run.sh train --extra_tag release_run --fix_random_seed
```

Docker 자체 빌드는 이번 서버에 Docker 실행 파일이 없어 수행하지 못했습니다. 현재 서버에서 수행한 GPU·모델 검증은 [검증 기록](VALIDATION.md)에 별도로 적었습니다.

## 데이터와 checkpoint

KITTI `training/` 아래에 `image_2`, `calib`, `label_2`, `velodyne`, `depth_2`, `lidar_depth`가 필요합니다. `ImageSets/train.txt`, `val.txt`와 `kitti_infos_train.pkl`, `kitti_infos_val.pkl`도 필요합니다.

두 depth 폴더의 PNG는 로더에서 256으로 나누어 미터 단위로 읽습니다. 원 실험에서 사용한 전처리 결과를 준비해야 하며, 동일한 결과를 만드는 전처리 스크립트까지 이번 배포에 확보된 것은 아닙니다. 임의의 sparse projection이나 depth completion 결과를 원 실험과 같다고 간주하지 않습니다.

초기화용 파일명은 다음과 같습니다.

```text
upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth
```

이 파일은 ADE20K segmentation checkpoint입니다. 검증 AP를 계산할 때 필요한 학습 완료 3D detector checkpoint와 다릅니다. 둘 다 저장소에 포함하지 않습니다. 자세한 준비 과정은 [영문 재현 가이드](REPRODUCING.md)를 참조하세요.

## 공개 설정

| 항목 | 값 |
|---|---|
| 공개 YAML | `configs/release/vmamba_kitti.yaml` |
| 학습 | 80 epochs, batch size 2, single GPU |
| optimizer | `adam_onecycle`, LR 설정값 0.001 |
| backbone 초기화 | ADE20K 학습 VMamba-B/UPerNet |
| depth | 150 LID bins + invalid 1개, 2.0–46.8 m |
| voxel | 0.16 m × 0.16 m × 0.16 m |
| 데이터 분할 | train 3,712 / val 3,769 |

기존 실험 YAML과 작업 중인 모델 수정사항을 보존하면서 공개용 설정을 별도로 만들었습니다. 이 설정은 현재 결합 모델을 실행합니다. 과거 ablation은 소스를 수정하며 실행된 기록이므로, YAML 스위치만으로 네 행을 재현할 수 있다고 안내하지 않습니다.

## 논문 표와 공개 상태

[결과 요약](RESULTS.md)에 test-set Pedestrian/Cyclist 성능, Car ablation, 효율성 표를 간단히 정리했습니다. 수치는 보관 원고의 보고값이며 이번에 전체 학습을 다시 수행한 결과가 아닙니다. 과거 Full 결과와 최근 Full 실행의 차이, 일부 행의 checkpoint/log 연결 미완료도 구분했습니다.

논문 온라인 공개 후 DOI, 최종 제목·저자 표기, BibTeX를 추가하면 됩니다. 현재는 임의의 출판일이나 DOI를 넣지 않았습니다.

## GitHub 업로드용 소스 만들기

```bash
python scripts/release/export_source.py
```

`release-dist/source/`에 배포용 파일과 SHA-256 manifest가 생성됩니다. 기존 출력 폴더를 덮어쓰지 않습니다. 논문 PDF, 리뷰 대응 자료, 데이터, checkpoint, output, `.git` 이력, IDE 설정은 내보내지 않습니다. 작업 트리의 코드 수정은 반영되므로 업로드할 실제 소스를 확인할 수 있습니다.

내보내기 자체는 Git commit이나 push를 수행하지 않습니다. 새 저장소 또는 기존 저장소에 반영하는 절차는 [배포 안내](RELEASE.md)에 있습니다.
