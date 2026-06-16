#!/usr/bin/env bash
set -e

BUMP="${1:---minor}"

CURRENT=$(grep '^version' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
MAJOR=$(echo "$CURRENT" | cut -d. -f1)
MINOR=$(echo "$CURRENT" | cut -d. -f2)
PATCH=$(echo "$CURRENT" | cut -d. -f3)

case "$BUMP" in
  --major) VERSION="$((MAJOR + 1)).0.0" ;;
  --minor) VERSION="${MAJOR}.$((MINOR + 1)).0" ;;
  --patch) VERSION="${MAJOR}.${MINOR}.$((PATCH + 1))" ;;
  *)       VERSION="${BUMP#v}" ;;  # explicit version like 1.2.3 or v1.2.3
esac

echo "Releasing $CURRENT → $VERSION"

sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
cargo update --workspace

git add Cargo.toml Cargo.lock
git commit -m "Release v$VERSION"
git tag "v$VERSION"
git push origin master
git push origin "v$VERSION"
