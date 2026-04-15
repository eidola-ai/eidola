#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

swift build --product MarkdownEditorDemo

APP_DIR=".build/MarkdownEditorDemo.app/Contents"
mkdir -p "$APP_DIR/MacOS"

cat > "$APP_DIR/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>MarkdownEditorDemo</string>
    <key>CFBundleIdentifier</key>
    <string>dev.eidola.MarkdownEditorDemo</string>
    <key>CFBundleName</key>
    <string>MarkdownEditorDemo</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
</dict>
</plist>
PLIST

cp -f .build/debug/MarkdownEditorDemo "$APP_DIR/MacOS/MarkdownEditorDemo"
open .build/MarkdownEditorDemo.app
