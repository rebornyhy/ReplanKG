#!/usr/bin/env bash
set -e

############################################
# ========== Required Parameters (User Input) ==========
############################################

# ===== General Parameters =====
LLM_PATH="./replankg_llm/"                 # Required
MID2NAME_PATH="./mid2name/mid2name.pkl"    # Required
EMB_MODEL_PATH="./emb_model/bge-large-en-v1.5"  # Required

EXP_NAME=""                 # Required
OUTPUT_DIR="./output/"      # Required
RETRY=3                     # Required

# ===== Dataset Selection =====
# Options: simpleques / webqsp / cwq
KG_CLASS=""

############################################
# ========== Automatically Match DATA_PATH ==========
############################################

if [ "$KG_CLASS" == "simpleques" ]; then
  DATA_PATH="./simpleques_testdata.jsonl"

elif [ "$KG_CLASS" == "webqsp" ]; then
  DATA_PATH="./webqsp_testdata.jsonl"

elif [ "$KG_CLASS" == "cwq" ]; then
  DATA_PATH="./cwq_testdata.jsonl"

else
  echo "ERROR: KG_CLASS must be one of simpleques / webqsp / cwq"
  exit 1
fi

# Check whether the test data file exists
if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: DATA_PATH file does not exist: $DATA_PATH"
  echo "Please ensure the corresponding test data has been downloaded and placed in the project root directory."
  exit 1
fi

# ===== Optional Parameters (with default values) =====
WIDTH=3
ITER_N=1
# HOP does not need to be specified; it will be validated internally in Python


############################################
# ========== SimpleQuestions Paths ==========
############################################
SQ_TEST_JSONL="./data_kg/simpleques/simpleques_test.jsonl"
SQ_TEST_SUBGRAPH_JSONL="./data_kg/simpleques/simpleques_test_subgraph.jsonl"

############################################
# ========== WebQSP Paths ==========
############################################
WQ_TEST_JSONL="./data_kg/webqsp/webqsp_test.jsonl"
WQ_TEST_SIMPLE_JSONL="./data_kg/webqsp/test_simple.jsonl"
WQ_TEST_COM_JSONL="./data_kg/webqsp/webqsp_test_com.jsonl"
WQ_DIR="./data_kg/webqsp/"

############################################
# ========== CWQ Paths ==========
############################################
CWQ_TEST_JSONL="./data_kg/cwq/cwq_test.jsonl"
CWQ_TEST_SIMPLE_JSONL="./data_kg/cwq/test_simple.jsonl"
CWQ_TEST_COM_JSONL="./data_kg/cwq/cwq_test_com.jsonl"
CWQ_DIR="./data_kg/cwq/"


#################################################
# ========== Parameter Validation ==========
#################################################

function check_required() {
  if [ -z "$1" ]; then
    echo "ERROR: $2 cannot be empty"
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
# ========== Build Base Command ==========
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
# ========== Append Dataset-Specific Arguments ==========
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
  echo "ERROR: KG_CLASS must be one of simpleques / webqsp / cwq"
  exit 1
fi


#################################################
# ========== Execute ==========
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
