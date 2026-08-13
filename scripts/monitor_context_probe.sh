#!/usr/bin/env bash
set -u

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${repo_dir}/outputs/dnacorpus_context_length_probe_orsa_v2"

echo "DNACorpus 上下文长度实验监控"
date '+时间: %Y-%m-%d %H:%M:%S %Z'
echo

echo "GPU 状态（显存 MiB）"
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>&1 || true
echo

echo "实验进程"
ps -eo pid,etimes,pcpu,pmem,args \
  | awk 'NR==1 || /run_dnacorpus_context_length_probe|run_probability_trace/ {print}' \
  | awk '!/awk/' || true
echo

carbon_done=$(find "${output_dir}/traces_position_major" -path '*/carbon/manifest.json' -type f 2>/dev/null | wc -l)
evo2_done=$(find "${output_dir}/traces_position_major" -path '*/evo2/manifest.json' -type f 2>/dev/null | wc -l)
evo2_optimized_done=$(find "${output_dir}/traces_position_major" -path '*/evo2_optimized/manifest.json' -type f 2>/dev/null | wc -l)
stap_done=$(find "${output_dir}/traces_position_major" -path '*/stap/manifest.json' -type f 2>/dev/null | wc -l)
printf '完整 position-major trace: Carbon %s/18 | Evo2-ref %s/18 | Evo2-opt %s/18 | STAP %s/18\n' \
  "${carbon_done}" "${evo2_done}" "${evo2_optimized_done}" "${stap_done}"
echo

echo "最近完成的模型 trace"
find "${output_dir}/traces_position_major" \
  \( -path '*/carbon/manifest.json' -o -path '*/evo2/manifest.json' -o -path '*/evo2_optimized/manifest.json' \) \
  -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' 2>/dev/null \
  | sort | tail -8 | sed "s#${repo_dir}/##"
echo

echo "当前/最近模型日志（修改时间、大小）"
find "${output_dir}/logs" \
  \( -name '*_carbon.log' -o -name '*_evo2.log' -o -name '*_evo2_optimized.log' \) \
  -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %10s %p\n' 2>/dev/null \
  | sort | tail -8 | sed "s#${repo_dir}/##"
