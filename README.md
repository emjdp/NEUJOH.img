# NEUJOH.img

NEUJOH.img는 이미지의 색과 구조를 보존하면서 문자로 다시 그리는 컬러 ASCII 아트 렌더러입니다.

> **N**eural **E**dge **U**nderstanding for **J**oint **O**ptical **H**alftoning, using **I**mage-**M**apped **G**lyphs

<p align="center">
  <img src="docs/full_wide.png" width="720" alt="NEUJOH.img로 변환한 컬러 ASCII 아트 예시">
</p>

## 무엇이 다른가요?

보통의 ASCII 아트가 밝기에 따라 문자를 고른다면, NEUJOH.img는 글리프의 실제 모양과 이미지의 국소 구조를 함께 비교합니다. 그래서 어두운 영역도 빈 공간이 되지 않고, 윤곽과 질감이 문자 안에 남습니다.

- 글리프를 4×8 잉크 커버리지 벡터로 바꾸고 형태가 가까운 문자를 선택합니다.
- DoG와 Sobel 방향을 이용해 강한 경계를 `-`, `/`, `|`, `\\`로 보강합니다.
- 선형 광과 OKLab에서 색을 처리해 컬러 ASCII 특유의 탁함을 줄입니다.
- 인물 매트를 선택적으로 사용해 피사체와 배경의 디테일을 따로 조절합니다.
- PNG, SVG, 일반 텍스트, 24비트 ANSI 출력을 지원합니다.

## 설치

Python 3.12 기준입니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`rembg` 모델은 첫 실행 때 내려받을 수 있습니다. 빠르게 시험하려면 `--no-matte`를 사용하세요.

## 사용법

```bash
python ascii_art.py input.jpg \
  --cols 100 \
  --aspect 1:1 \
  --look film \
  --scale 2 \
  --svg \
  --ansi \
  -o result
```

이 명령은 `result.png`와 `result.txt`를 만들고, 옵션에 따라 `result.svg`와 `result.ans`도 만듭니다.

자주 쓰는 옵션:

| 옵션 | 설명 |
|---|---|
| `--cols` | 문자 격자의 가로 칸 수 |
| `--charset` | `ascii`, `code`, `blocks`, `mixed`, `minimal` |
| `--look` | `neutral`, `film`, `sunlit`, `neon`, `cold`, `mono` |
| `--aspect` | `1:1`, `16:9` 같은 출력 비율 |
| `--zoom`, `--focus-y` | 크롭 배율과 세로 초점 |
| `--detail`, `--structure` | 국소 디테일과 문자 구조 강도 |
| `--edges` | 방향성 경계 보강 강도 |
| `--bg-gain`, `--energy` | 셀 배경색과 광량 보존 정도 |
| `--invert` | 밝은 종이에 어두운 글자 스타일 |
| `--no-matte` | 피사체 분리 없이 변환 |

전체 옵션은 다음 명령으로 확인할 수 있습니다.

```bash
python ascii_art.py --help
```

## 파이프라인

1. 폰트의 각 글리프를 래스터라이즈하고 작은 커버리지 벡터로 압축합니다.
2. 선택적으로 피사체 매트를 계산해 전경과 배경을 분리합니다.
3. CLAHE, 언샤프, 국소 대비 정규화로 문자 선택용 디테일 채널을 만듭니다.
4. 방향성 경계를 검출해 실루엣과 주요 선을 보강합니다.
5. 선형 광에서 셀 색을 평균내고 OKLab에서 색감을 조정합니다.
6. 글리프 커버리지를 고려해 잉크와 배경의 광량을 보존하며 합성합니다.

핵심 합성식은 다음과 같습니다.

```text
coverage × ink + (1 − coverage) × wash = cell color
```

글자가 셀의 일부만 채우는 점을 보정하기 때문에 밝은 영역이 지나치게 어두워지지 않습니다.

## 구성

- `ascii_art.py`: 변환 파이프라인과 CLI
- `make_profile.py`: 여러 크롭과 해상도를 한 번에 만드는 배치 예제
- `sweep.py`: 주요 파라미터를 비교하는 콘택트 시트 생성기
- `fonts/`: 렌더링에 사용하는 JetBrains Mono 폰트
