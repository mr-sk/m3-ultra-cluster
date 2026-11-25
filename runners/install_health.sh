#!/bin/bash
# Install the health watch: copy node_status.sh to each node, and load a LaunchAgent on this Mac
# that runs health.sh every 2 minutes. No sudo. Uninstall: install_health.sh --remove
. "$(dirname "$0")/cluster_env.sh"
LABEL=com.m3cluster.health
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist
if [ "$1" = "--remove" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null; rm -f "$PLIST"; echo "removed $LABEL"; exit 0
fi
scp -q "$RUNNERS/node_status.sh" "$HEAD:m3_node_status.sh" && ssh "$HEAD" "chmod +x ~/m3_node_status.sh && scp -q ~/m3_node_status.sh $PEER:m3_node_status.sh && ssh $PEER 'chmod +x ~/m3_node_status.sh' && echo node script on both nodes"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/dsv4-logs"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$RUNNERS/health.sh</string></array>
  <key>StartInterval</key><integer>120</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$HOME/dsv4-logs/health.out</string>
  <key>StandardErrorPath</key><string>$HOME/dsv4-logs/health.out</string>
</dict></plist>
EOF
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
launchctl bootstrap "gui/$(id -u)" "$PLIST" && echo "loaded $LABEL (every 2 min); log: ~/dsv4-logs/health.log"
