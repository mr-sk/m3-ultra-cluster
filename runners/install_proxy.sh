#!/bin/bash
# Run the recording proxy as a login service on this Mac (LaunchAgent com.m3cluster.proxy):
# started at login, restarted if it exits, with the everyday rules and edit-path filling.
# OpenCode's provider points at it (http://127.0.0.1:$PROXY_PORT/v1), in any project folder.
# No sudo. Uninstall: install_proxy.sh --remove
# Settings: see cluster_env.sh (SERVER_URL, EVERYDAY_RULES, PROXY_SCRIPT, PROXY_PORT, PROXY_LOG).
. "$(dirname "$0")/cluster_env.sh"
LABEL=com.m3cluster.proxy
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist
if [ "$1" = "--remove" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null; rm -f "$PLIST"; echo "removed $LABEL"; exit 0
fi
: "${EVERYDAY_RULES:?set EVERYDAY_RULES in $CONF}"
PY=$(command -v python3); [ -n "$PY" ] || { echo "python3 not found"; exit 1; }
# a proxy started by hand or by a test runner holds the port: stop it first
pid=$(lsof -nP -t -iTCP:"$PROXY_PORT" -sTCP:LISTEN); [ -n "$pid" ] && kill $pid && sleep 1
mkdir -p "$HOME/Library/LaunchAgents" "$(dirname "$PROXY_LOG")"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$PY</string><string>$PROXY_SCRIPT</string>
    <string>--port</string><string>$PROXY_PORT</string>
    <string>--upstream</string><string>$SERVER_URL</string>
    <string>--rules</string><string>$EVERYDAY_RULES</string>
    <string>--fill-edit-path</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$PROXY_LOG</string>
  <key>StandardErrorPath</key><string>$PROXY_LOG</string>
</dict></plist>
EOF
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
launchctl bootstrap "gui/$(id -u)" "$PLIST" && sleep 2 && echo "loaded $LABEL: $(tail -1 "$PROXY_LOG")"
