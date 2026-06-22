# tiny char-level GPT (Go, raw JSON weights)

순수 Go + 직접 구현한 역전파(autograd 없음)로 학습하는 작은 char-level 트랜스포머입니다.
가중치(=텐서)는 `tiny_english_gpt.json` 한 파일에 그대로 저장됩니다.

## 구조

- **char-level 토크나이저**: vocab = 글자 단위, `"\n"` = 스토리 끝(EOS) 토큰
- **pre-norm 트랜스포머 블록 × N**: 각 블록 = `x + Attn(LN1(x))` → `x + FFN(LN2(x))`
  - FFN = `Linear(d, d_ff)` → GELU → `Linear(d_ff, d)`
- 마지막에 `LN_f` → `lm_head` 로 로짓 계산
- 학습: 손으로 유도한 gradient + Adam, 배치는 goroutine으로 병렬 처리

JSON 스키마: `config`(d_model/n_heads/n_layers/d_ff/max_seq_len), `vocab`,
`token_embed`, `pos_embed`, `blocks[]`(블록별 가중치), `ln_f`, `lm_head`.

## 사용법

### 1. 데이터 준비 (TinyStories 일부)

```bash
curl -L -r 0-3000000 \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt" \
  -o tinystories_raw.txt
```

데이터 파일이 없으면 자동으로 작은 toy corpus로 대체됩니다.

### 2. 학습 (`tiny_english_gpt.json` 생성)

```bash
go run ./train
```

환경변수로 설정 조절 가능 (기본값):

| 변수 | 기본 | 설명 |
|------|------|------|
| `D_MODEL` | 64 | 임베딩 차원 |
| `N_HEADS` | 4 | 어텐션 헤드 수 |
| `N_LAYERS` | 3 | 블록 수 |
| `D_FF` | 256 | FFN 내부 차원 |
| `MAX_SEQ` | 128 | 최대 시퀀스 길이 |
| `BATCH` | 16 | 배치 크기 (goroutine 병렬) |
| `STEPS` | 3000 | 학습 스텝 수 |
| `LR` | 5e-4 | 학습률 |
| `CKPT_EVERY` | 200 | 체크포인트 주기 (JSON 저장 + 샘플 생성) |
| `DATA` | tinystories_raw.txt | 학습 텍스트 |

검증된 "큰" 설정 (~600k params, CPU 약 3시간, loss ~0.9까지 내려감):

```bash
D_MODEL=128 N_LAYERS=4 N_HEADS=8 D_FF=512 BATCH=24 STEPS=20000 LR=4e-4 go run ./train
```

학습 중 `CKPT_EVERY` 마다 JSON을 저장하므로, 학습 도중에도 아래 추론을 실행해
중간 결과를 바로 확인할 수 있습니다.

### 3. 추론 / 쿼리

```bash
go run .        # 샘플 프롬프트 몇 개 생성
go run . -i     # 대화형: 프롬프트를 입력하면 이어서 생성
```

생성은 autoregressive (다음 글자 샘플링 → 이어붙이기 반복)이며,
temperature + top-k 샘플링을 쓰고 EOS(`"\n"`)를 만나면 멈춥니다.
샘플링은 환경변수로 조절합니다:

| 변수 | 기본 | 설명 |
|------|------|------|
| `GEN_TEMP` | 0.8 | temperature. 낮을수록(예 0.5) 안정적·반복적, 높을수록 다양·위험 |
| `GEN_TOPK` | 10 | top-k. 후보 글자 수 |

예: `GEN_TEMP=0.5 GEN_TOPK=8 go run . -i`

(주의: `TEMP`은 Windows 예약 환경변수라 쓰면 안 됨 → `GEN_TEMP` 사용)
