#!/bin/bash

set -e
set -x

# 定义一个清理函数，用于终止所有后台任务
cleanup() {
  echo "中断所有后台进程..."
  kill $(jobs -p)
  exit
}

# 当脚本收到SIGINT信号时，调用cleanup函数
trap cleanup SIGINT

export PYTHONPATH=.

#!/bin/bash

#!/bin/bash
#!/bin/bash

# 定义 CUDA 设备数组
CUDA_DEVICES=("0" "1" "2" "3" "4" "5" "6" "7")
#CUDA_DEVICES=("BIOLOGY_img" "GEOGRAPHY_img" "HISTORY_img" "POLITICAL_img")
thread_data_num=2086
# 并行运行8个进程
for i in {0..55}
do
    device_index=$((i % ${#CUDA_DEVICES[@]}))
    start=$((i * $thread_data_num))
    end=$((start + $thread_data_num))
    CUDA_VISIBLE_DEVICES=${device_index} python data_process/parse_ocr.py ${start} ${end} &
done
# 等待所有后台进程完成
wait
echo "所有文件下载和处理完成"

