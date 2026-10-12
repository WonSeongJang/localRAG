# localRAG

> AI 코딩 세션 기록과 개인 노트를 검색하는 로컬 하이브리드 RAG.
> 임베딩부터 저장까지 전부 로컬에서 돌아가 내용이 외부로 나가지 않는다.

Claude Code / Codex로 작업하다 보면 "이거 예전에 어떻게 했더라"가 계속 생깁니다. 세션 기록은
JSONL로 쌓여 있지만 grep으로는 못 찾습니다 — 표현이 그때와 다르기 때문입니다. 그렇다고 세션
내용을 외부 임베딩 API에 올릴 수도 없습니다. 그래서 만들었습니다.

## 구조

```
세션 JSONL (~/.claude/projects, ~/.codex/sessions) + 마크다운 노트
  → parsers.py   실행 지침·승인 검토(guardian) 제외, Codex 이중 기록 결속, 반복 질문 보존
  → chunk.py     대화 교환 단위 / 노트 제목·상위 절을 보존한 청킹 (최대 2,500자)
  → embed.py     Ollama bge-m3 임베딩 (1024차원)
  → db.py        SQLite 하나에 chunks + FTS5(BM25) + sqlite-vec
  → search.py    중복 후보 정리 → 관련성 필터 → BM25 + 벡터 RRF 융합
```

## 설계에서 신경 쓴 것

**하이브리드 검색.** 벡터 검색만 쓰면 정확한 단어를 놓칩니다. 함수명이나 에러 메시지처럼 글자
그대로 찾아야 하는 것들이 그렇습니다. 반대로 키워드 검색만으로는 표현이 다른 과거 기록을 못
찾습니다. 그래서 BM25(FTS5)와 벡터 검색을 각각 돌리고 RRF로 합칩니다.

**RRF로 융합하는 이유.** BM25 점수와 코사인 거리는 스케일이 달라서 그냥 더할 수 없습니다.
정규화해서 더하려 해도 BM25 점수 범위가 질의마다 달라 기준이 흔들립니다. RRF는 점수를 버리고
순위만 합칩니다. 다만 RRF 점수는 관련성이나 정답 확률이 아닙니다. 결과를 내보내기 전에
질의어 가중 일치율과 코사인 유사도를 별도로 검사하며, 근거가 약하면 결과를 채우지 않습니다.
현재 기준은 가중 일치율 0.65 이상 또는 코사인 유사도 0.68 이상입니다. 희귀어·미등장어는
높은 가중치를 유지해 질문의 대상을 조용히 버리지 않습니다. 이는 회귀 사례에 맞춘
보수적 휴리스틱이며, 모든 질문의 정확도를 보장하는 수치가 아닙니다.

**한국어 질의.** 원형을 보존하면서 실제 색인에 존재하는 조사 제거형을 추가합니다.
예를 들어 `옵시디언에서`는 `옵시디언`도 찾습니다. 완전한 형태소 분석이나 임의 동의어
확장은 하지 않으며, 질의 표현에 따라 오거부가 생기면 구체어·소스·프로젝트를 바꿔 검색합니다.

**한 파일 안에서 끝낸다.** 데이터 규모가 크지 않아 별도 벡터DB를 띄울 이유가 없었습니다.
sqlite-vec을 쓰면 SQLite 파일 하나로 끝나고, 프로젝트·소스 필터를 SQL 조인으로 벡터 검색에
그대로 걸 수 있습니다. 필터가 있으면 필터 내부에서 코사인 거리를 계산합니다.
필터가 없으면 빠른 KNN을 쓰되, 중복 때문에 후보 범위가 최대 k(4096)를 넘으면
전체 거리 계산으로 전환해 고유 후보가 잘리지 않게 합니다.

**결과 다양성.** 한 세션에서 결과가 도배되면 쓸모가 없어서, 중복 텍스트를 걸러내고 세션당
최대 2건으로 제한합니다. 순위를 합치기 전에 공백 정규화 후 동일한 내용의 후보를 묶고,
고유 내용 80개를 확보하거나 전체 후보를 소진할 때까지 범위를 확장합니다. 표현이 조금씩
다른 복제 문서까지 의미적으로 통합하지는 않습니다.

**증분 인덱싱.** 새 파일과 수정된 파일만 다시 처리하고, 삭제되거나 대상에서 빠진 파일은 DB에서도
정리합니다. 단 Claude·Codex 세션은 아카이브로 취급해, 도구가 원본을 자동 삭제해도 색인에 남깁니다.
임베딩 모델·차원이 바뀌어 재구축할 때도 원본이 없는 세션은 저장된 본문으로 다시 임베딩합니다. 임베딩 모델이나 청킹 설정이 바뀌면 인덱스 서명이 달라지므로, 검색을 중단하고
`index --rebuild`를 안내해 서로 다른 형식의 벡터가 섞이지 않게 합니다.
재구축은 `index.rebuild.db`에서 진행하며, 같은 모델·차원의 동일 본문 벡터를 재사용합니다.
검증을 통과한 뒤 운영 DB를 교체하고 `index.backup-<시각>.db`를 보존합니다.
중단된 작업은 운영 DB가 바뀌지 않았으면 같은 명령으로 재개할 수 있습니다.
`--no-refresh`도 형식 검사를 생략하지 않습니다.

## 사용법

```bash
# 검색 (실행 시 새/변경된 파일을 자동으로 증분 인덱싱)
uv run localrag search "토큰 갱신 로직 어떻게 처리했지"
uv run localrag search "crawling retry" --project MyProject --source claude -n 5 --full

# 수동 인덱싱 / 통계
uv run localrag index
uv run localrag index --source obsidian
uv run localrag index --rebuild   # 파서·청킹·임베딩 설정을 바꾼 뒤
uv run localrag stats

# 백업: index.db + Claude/Codex 세션 원본을 Google Drive 동기화 폴더로 (최근 3회분 유지)
scripts/backup.sh
```

다른 디렉터리에서 쓰려면 `uv run --project /path/to/localRAG localrag search "..."`.

## 요구사항

- [Ollama](https://ollama.com) 실행 중 (`ollama serve`) + `bge-m3` 모델 (`ollama pull bge-m3`)
- [uv](https://docs.astral.sh/uv/) — 의존성은 `sqlite-vec`, `requests` 둘뿐입니다

## 색인 대상 경로

기본값은 아래와 같고, 환경변수나 프로젝트 루트의 `.env`(`KEY=경로` 형식, gitignore 대상)로 바꿀 수 있습니다.
없는 경로의 소스는 건너뜁니다.

| 소스 | 기본 경로 | 환경변수 |
|---|---|---|
| claude | `~/.claude/projects` | `LOCALRAG_CLAUDE_ROOT` |
| codex | `~/.codex/sessions` | `LOCALRAG_CODEX_ROOT` |
| obsidian | `~/obsidian` | `LOCALRAG_OBSIDIAN_ROOT` |
| docs | `~/docs` (추가 마크다운 폴더) | `LOCALRAG_DOCS_ROOT` |
| memory | `~/.claude/projects/*/memory` | (claude와 동일) |

인덱스 파일은 `index.db`이며 `.gitignore`에 있습니다. 원본이 삭제된 세션은 이 파일에만 남으므로
지우지 말고 `scripts/backup.sh`로 백업하세요.

## 결과 진단값

검색 결과의 `match`, `coverage`, `similarity`는 매칭 이유를 확인하는 진단값이며 확신도나 사실 검증 결과가 아닙니다.

## 라이선스

MIT
