
set -x
set -e
#!/bin/bash
#"chinese" "history" "political" 'geography' 'math' 'chemistry' 'biology' 'geography'

keywords=("chinese" "history" "political")  # 假设有多个关键词
data_prefix=/mnt/data/xujinlingbj/raw_data/llava_sft/stage2_train_1116_arts_wo_img
data_path=${data_prefix}.json

common_info_save_data_prefix=${data_prefix}_with_common_info_
filter_test_save_path_prefix=${data_prefix}_filter_
save_path_prefix=${data_prefix}_filter_in_file_
# 合并生成的文件
merged_file=${data_prefix}_filter.json
clean_train_file=${data_prefix}_filter_clean.json


# 保存生成的 save_path 的数组
declare -a save_path_list=()

# 并行执行
for keyword in "${keywords[@]}"; do
  common_info_save_data_path=${common_info_save_data_prefix}${keyword}.json
  filter_test_save_path=${filter_test_save_path_prefix}${keyword}.json
  save_path=${save_path_prefix}${keyword}.json
  echo $keyword
  echo $common_info_save_data_path
  echo $save_path
  save_path_list+=("$save_path")

  python data_process/remove_duplicate.py "$data_path" "$common_info_save_data_path" "$filter_test_save_path" "$save_path" "$keyword" &
done

wait  # 等待所有后台任务完成
# 打印所有保存的 save_paths
# 使用空格将数组元素连接起来
save_path_list="${save_path_list[*]}"
echo "生成的 save_paths:" save_path_list

python data_process/make_final_train_data.py "${save_path_list}" "$merged_file"
#
python data_process/clean_train_data.py $merged_file $clean_train_file

