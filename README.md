# StockGPT
make a small gpt from scratch trained on U.S stock return

# UV Installation Guide

pipx 없을 경우 pip install pipx -> then pipx install uv  

uv -V 해서 인스톨 확인하고 pyproject.toml이 있는 스크린에서 uv sync

파이썬 버전 설정: uv python install 버전넘버 / uv python pin 버전넘버

패키지 추가하는 방법: uv add 패키지이름

검색창에 uv python 치면 다른 명령어들도 나올거임!

poetry나 venv로 하는 것보다 훨씬 나아. 일단 패키지 인스톨도 병렬처리 되고 rust 써서 훨씬 빠름