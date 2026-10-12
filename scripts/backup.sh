#!/bin/bash
# localRAG 백업: index.db 스냅샷 + Claude/Codex 세션 원본을 Google Drive 동기화 폴더로 복사한다.
# 필요할 때 수동 실행한다. 최근 KEEP 회분만 남긴다.
set -euo pipefail

KEEP=3
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DRIVE="$(ls -d "$HOME"/Library/CloudStorage/GoogleDrive-*/{"My Drive","내 드라이브"} 2>/dev/null | head -1)"
DEST_ROOT="${LOCALRAG_BACKUP_DIR:-$DRIVE/localRAG-backup}"
[ -n "$DRIVE" ] || [ -n "${LOCALRAG_BACKUP_DIR:-}" ] || { echo "Google Drive 폴더를 찾을 수 없음" >&2; exit 1; }

stamp=$(date +%Y-%m-%d_%H%M)
partial="$DEST_ROOT/.partial-$stamp"
mkdir -p "$partial"

# 사용 중인 SQLite는 그냥 복사하면 깨질 수 있어 온라인 백업 API로 스냅샷을 뜬다.
tmp=$(mktemp -d)
/usr/bin/sqlite3 "$REPO/index.db" ".backup '$tmp/index.db'"
gzip -c "$tmp/index.db" > "$partial/index.db.gz"
rm -rf "$tmp"

tar -czf "$partial/claude-projects.tar.gz" -C "$HOME" .claude/projects
tar -czf "$partial/codex-sessions.tar.gz" -C "$HOME" .codex/sessions

mv "$partial" "$DEST_ROOT/$stamp"

# 완성된 백업만 날짜순으로 세어 오래된 것부터 정리
backups=$(ls -1d "$DEST_ROOT"/20* | sort)
excess=$(( $(echo "$backups" | wc -l) - KEEP ))
if [ "$excess" -gt 0 ]; then
  echo "$backups" | head -n "$excess" | while read -r old; do rm -rf "$old"; done
fi

echo "ok $DEST_ROOT/$stamp ($(du -sh "$DEST_ROOT/$stamp" | cut -f1))"
