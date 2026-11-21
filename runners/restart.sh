#!/bin/bash
# Restart the cluster server with the given flags and wait until it serves.
# usage: runners/restart.sh [SERVER FLAGS...]     e.g. runners/restart.sh --concurrent 2
# Settings: see cluster_env.sh.
. "$(dirname "$0")/cluster_env.sh"
restart_server "$*"
ssh "$HEAD" 'grep -E "indexer:|runs layers" ~/dsv4-server.log | sort -u | cut -c1-120'
