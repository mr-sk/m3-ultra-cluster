#!/bin/bash
# One line of key=value facts about this node, for health.sh. Copied to ~/m3_node_status.sh on
# each node by install_health.sh. Read-only; no sudo.
RDMA_DEVS=${RDMA_DEVS:-"rdma_en4 rdma_en5"}
wired=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null)
rdma=$(ibv_devinfo 2>/dev/null | awk -v devs=" $RDMA_DEVS " '
  /hca_id:/ {d=$2} /state:/ {if (index(devs, " " d " ") && $2 == "PORT_ACTIVE") n++} END {print n+0}')
tbip=$(ifconfig en4 2>/dev/null | awk '/inet / {print $2; exit}')
server=$(pgrep -f "python.*vlm_cluster_server.py" | wc -l | tr -d ' ')
bench=$(pgrep -f "mtp-bench/" | wc -l | tr -d ' ')
free=$(memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%", "", $NF); print $NF}')
echo "host=$(hostname -s) wired=${wired:-?} rdma=$rdma rdma_want=$(echo $RDMA_DEVS | wc -w | tr -d ' ') tbip=${tbip:-none} server=$server bench=$bench free=${free:-?}"
