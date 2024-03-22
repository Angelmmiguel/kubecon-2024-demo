#!/usr/bin/env bash

# All this is based on the following article:
# https://www.substratus.ai/blog/kind-with-gpus/
echo "Create cluster"
kind create cluster --name kubeconeu24 --config - <<EOF
apiVersion: kind.x-k8s.io/v1alpha4
kind: Cluster
nodes:
- role: control-plane
  image: kindest/node:v1.27.3
  # required for GPU workaround
  extraMounts:
    - hostPath: /dev/null
      containerPath: /var/run/nvidia-container-devices/all
EOF

docker exec -ti kubeconeu24-control-plane ln -s /sbin/ldconfig /sbin/ldconfig.real

echo "Install Nvidia operator"
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia || true
helm repo update
helm install --wait --generate-name \
     -n gpu-operator --create-namespace \
     nvidia/gpu-operator --set driver.enabled=false
