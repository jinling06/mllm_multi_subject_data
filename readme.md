# 多模态多学科解题数据集项目介绍

## 数据情况介绍
本项目包含了高中7个学科的多模态解题数据，大约50w+「实际可能更多」，其中包含较多的纯文本解题数据。
数据按照获取批次进行保存了多个文件夹，每个批次包含了多个学科的数据。

全量数据可在[mllm_multi_subject_data](https://modelscope.cn/datasets/callMeWhy/mllm_multi_subject_data/files) 获取。
里面包含了一些清洗后的json文件，大多原始数据为tsv和html的格式，具体每个来源的数据格式可参考下面的清洗代码。

## 数据清洗代码说明
此项目包括mllm_multi_subject_data清洗代码、一些公开多模态解题数据集清洗代码、数据去重代码。为了方便大家使用，下面针对每一部分做简单的介绍。


### 公开数据集
下面的数据都可以直接在网络上搜索到，此处不再给出每个数据的链接。公开数据集的格式都较为规整，代码相对于简洁一些。

| 数据名称                               | 数据处理代码                      |
|------------------------------------|----------------------------------|
| agieval                            | data_process/prepare_agieval.py   |
| CMMLU-Clinical-Knowledge-Benchmark | data_process/prepare_CMMLU-Clinical-Knowledge-Benchmark.py   |
| cmmu                               | data_process/prepare_cmmu.py   |
| EduChat_Math                       |  data_process/prepare_EduChat_Math.py  |
| lava-med-zh-instruct-60k           |  data_process/prepare_lava-med-zh-instruct-60k.py  |
| llava_mid_stage                    |  data_process/prepare_llava_mid_stage.py  |
| m3ke                               |  data_process/prepare_m3ke.py  |
| mathv360k                          | data_process/prepare_mathv360k.py   |
| mathverse                          |  data_process/prepare_mathverse.py  |
| mathvision                         | data_process/prepare_mathvision.py   |
| MathVista                          | data_process/prepare_MathVista.py   |
| mavis    不带答案解释                    | data_process/prepare_mavis.py   |
| mavis 带答案解释                        | data_process/prepare_mavis_with_explain.py   |
| mmcu                               |  data_process/prepare_mmcu.py  |
| mmlu                               | data_process/prepare_mmlu.py   |
| sciq                               |  data_process/prepare_sciq.py  |
| SEED-Bench-H                       |  data_process/prepare_SEED-Bench-H.py  |
| cmmlu                              |  data_process/preprare_cmmlu.py  |



### 私有数据集
为了便于大家快速了解数据的格式以及快速清洗数据，下面给出了每个来源的数据清洗代码。\
下述表格中的数据名称均是在 [mllm_multi_subject_data](https://modelscope.cn/datasets/callMeWhy/mllm_multi_subject_data) 中存储的名称，
个别来源由于数据太多，进行了分开存储，使用时需要更改具体代码中的数据文件夹名称。



| 数据名称                                                          | 数据处理代码                        |
|---------------------------------------------------------------|------------------------------------|
| MLLM_data_4_v1                                                | data_process/prepare_mllm_4_v1.py   |
| MLLM_data_2_v2 理科                                             |  data_process/prepare_mllm_data_2.py  |
| MLLM_data_2_v2 文科                                             |  data_process/prepare_mllm_data_2_arts.py  |
| MLLM_data_3_v1_part1  MLLM_data_3_v1_part2 的理科数据              | data_process/prepare_mllm_data_3_v1.py   |
| MLLM_data_3_v1_part1  MLLM_data_3_v1_part2 的语文之外的文科数据         |  data_process/prepare_mllm_data_3_v1_arts.py  |
| MLLM_data_3_v1_part1  MLLM_data_3_v1_part2 的语文数据              |  data_process/prepare_mllm_data_3_v1_chinese.py  |
| MLLM_data_3_v2_part1 MLLM_data_3_v2_part2                     |  data_process/prepare_mllm_data_3_v2.py  |
| MLLM_data_3_v3                                                |  data_process/prepare_mllm_data_3_v3.py  |
| MLLM_data_3_v4 理科数据                                           |  data_process/prepare_mllm_data_3_v4.py  |
| MLLM_data_3_v4 文科数据                                           |   data_process/prepare_mllm_data_3_v4_arts.py |
| MLLM_data_3_v5                                                |  data_process/prepare_mllm_data_3_v5.py  |
| MLLM_data_1_v1   MLLM_data_1_v2                               |   data_process/prepare_mllm_data_v1.py |
| MLLM_data_1_v3_tsv_part1 MLLM_data_1_v3_tsv_part2 不带解释的答案     | data_process/preprare_mllm_data_1_v3.py   |
| MLLM_data_1_v3_tsv_part1 MLLM_data_1_v3_tsv_part2 带解释的答案      |  data_process/preprare_mllm_data_1_v3_with_explain.py  |

由于此部分代码较为繁杂，下面给出几个注意事项：

1. **文理科分开处理**: 由于之前处理数据时文理科是分开处理的，因此下方存在相同的代码，但拆成了两个文件。例如 `MLLM_data_3_v1_part1` 和 `MLLM_data_3_v1_part2` 对应的数据。
2. **答案解析**: 有的数据源会有答案的解析过程，所以下方有的也分为了两个代码文件进行处理。不做解释备注的都是只有答案，无解析的数据。
3. **数据量**: 此数据原始是用于一个多模态解题比赛，相关清洗代码中都去除了不带图片的数据，并且由于个别来源的数据量较大，有些可能在代码中设置了只清洗前 x 个文件夹，所以具体数据量未知。
4. **路径处理**: 代码在上传的时候，将一些图片的绝对路径改为了相对路径，应该不能直接运行。



###数据去重代码
mllm_multi_subject_data 中的数据是来源于多个获取批次，因此存在数据重复的情况，为了进行较好的去重，可以参考下方的多线程去重代码。
```python
bash scripts/remove_duplicate.sh
```
去重算法是两个字符串的交并比，具体如下：
```python
def jaccard_similarity(set_a, set_b):

    intersection = sum(1 for char in set_a if char in set_b)
    union = len(set_a) + len(set_b) - intersection
    return intersection / union
```
可以根据具体的数据设置不同的阈值。