#!/bin/bash

# 에러 발생 시 즉시 종료
set -e


pip install --user pipx
pipx install uv
uv -V
uv venv stockgpt-env

echo "가상환경 활성화"
source stockgpt-env/bin/activate

echo "pyproject.toml의 패키지 설치"
uv pip install --requirements pyproject.toml

echo "CUDA 지원 여부 확인"
python -c "import torch; print(torch.cuda.is_available())"

echo "모든 작업 완료"
