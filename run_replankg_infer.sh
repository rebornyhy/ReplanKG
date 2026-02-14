#!/usr/bin/env bash
set -e

############################################
# ========== 必填参数区域（用户填写） ==========
############################################

# ===== 通用参数 =====
LLM_PATH="./replankg_llm/"                 # 必填
MID2NAME_PATH="./mid2name/mid2name.pkl"            # 必填
EMB_MODEL_PATH="./emb_model/bge-large-en-v1.5"           # 必填
DATA_PATH=""                # 必填
EXP_NAME=""                 # 必填
OUTPUT_DIR="./output/"               # 必填
RETRY=3                     # 必填

# ===== 数据集选择 =====
# 可选: simpleques / webqsp / cwq
KG_CLASS=""

# ===== 可选参数（有默认值）=====
WIDTH=3
ITER_N=1
# HOP 不需要填，会由 python 内部自动校验


############################################
# ========== SimpleQuestions 路径 ==========
############################################
SQ_TEST_JSONL="./data_kg/simpleques/simpleques_test.jsonl"
SQ_TEST_SUBGRAPH_JSONL="./data_kg/simpleques/simpleques_test_subgraph.jsonl"

############################################
# ========== WebQSP 路径 ==========
############################################
WQ_TEST_JSONL="./data_kg/webqsp/webqsp_test.jsonl"
WQ_TEST_SIMPLE_JSONL="./data_kg/webqsp/test_simple.jsonl"
WQ_TEST_COM_JSONL="./data_kg/webqsp/webqsp_test_com.jsonl"
WQ_DIR="./data_kg/webqsp/"

############################################
# ========== CWQ 路径 ==========
############################################
CWQ_TEST_JSONL="./data_kg/cwq/cwq_test.jsonl"
CWQ_TEST_SIMPLE_JSONL="./data_kg/cwq/test_simple.jsonl"
CWQ_TEST_COM_JSONL="./data_kg/cwq/cwq_test_com.jsonl"
CWQ_DIR="./data_kg/cwq/"


#################################################
# ========== 参数合法性检查 ==========
#################################################

function check_required() {
  if [ -z "$1" ]; then
    echo "ERROR: $2 不能为空"
    exit 1
  fi
}

check_required "$LLM_PATH" "LLM_PATH"
check_required "$MID2NAME_PATH" "MID2NAME_PATH"
check_required "$EMB_MODEL_PATH" "EMB_MODEL_PATH"
check_required "$DATA_PATH" "DATA_PATH"
check_required "$EXP_NAME" "EXP_NAME"
check_required "$OUTPUT_DIR" "OUTPUT_DIR"
check_required "$KG_CLASS" "KG_CLASS"

mkdir -p "$OUTPUT_DIR"


#################################################
# ========== 组装基础命令 ==========
#################################################

BASE_CMD="python kgqa_infer_args.py \
  --llm_path $LLM_PATH \
  --mid2name_path $MID2NAME_PATH \
  --emb_model_path $EMB_MODEL_PATH \
  --data_path $DATA_PATH \
  --exp_name $EXP_NAME \
  --output_dir $OUTPUT_DIR \
  --retry $RETRY \
  --width $WIDTH \
  --iter_n $ITER_N"


#################################################
# ========== 按数据集拼接 ==========
#################################################

if [ "$KG_CLASS" == "simpleques" ]; then

  check_required "$SQ_TEST_JSONL" "SQ_TEST_JSONL"
  check_required "$SQ_TEST_SUBGRAPH_JSONL" "SQ_TEST_SUBGRAPH_JSONL"

  CMD="$BASE_CMD simpleques \
    --sq_test_jsonl $SQ_TEST_JSONL \
    --sq_test_subgraph_jsonl $SQ_TEST_SUBGRAPH_JSONL"

elif [ "$KG_CLASS" == "webqsp" ]; then

  check_required "$WQ_TEST_JSONL" "WQ_TEST_JSONL"
  check_required "$WQ_TEST_SIMPLE_JSONL" "WQ_TEST_SIMPLE_JSONL"
  check_required "$WQ_TEST_COM_JSONL" "WQ_TEST_COM_JSONL"
  check_required "$WQ_DIR" "WQ_DIR"

  CMD="$BASE_CMD webqsp \
    --wq_test_jsonl $WQ_TEST_JSONL \
    --wq_test_simple_jsonl $WQ_TEST_SIMPLE_JSONL \
    --wq_test_com_jsonl $WQ_TEST_COM_JSONL \
    --wq_dir $WQ_DIR"

elif [ "$KG_CLASS" == "cwq" ]; then

  check_required "$CWQ_TEST_JSONL" "CWQ_TEST_JSONL"
  check_required "$CWQ_TEST_SIMPLE_JSONL" "CWQ_TEST_SIMPLE_JSONL"
  check_required "$CWQ_TEST_COM_JSONL" "CWQ_TEST_COM_JSONL"
  check_required "$CWQ_DIR" "CWQ_DIR"

  CMD="$BASE_CMD cwq \
    --cwq_test_jsonl $CWQ_TEST_JSONL \
    --cwq_test_simple_jsonl $CWQ_TEST_SIMPLE_JSONL \
    --cwq_test_com_jsonl $CWQ_TEST_COM_JSONL \
    --cwq_dir $CWQ_DIR"

else
  echo "ERROR: KG_CLASS 必须为 simpleques / webqsp / cwq"
  exit 1
fi


#################################################
# ========== 执行 ==========
#################################################

echo "======================================"
echo "Running RePlanKG Inference..."
echo "KG_CLASS: $KG_CLASS"
echo "EXP_NAME: $EXP_NAME"
echo "======================================"
echo ""
echo "$CMD"
echo ""

eval $CMD
