import argparse
import json
import pickle
import random
import time
from pathlib import Path

import jsonlines
from FlagEmbedding import FlagAutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from kg_process import *
from get_sim_rels_list import *
from check_modify_plan import *
from execute_plan import *

# NOTE:
# - width defaults to 3
# - iter_n defaults to 1
# - hop is determined by kg_class -> hop mapping (user may optionally pass --hop; if passed, we validate)
# - all other args are REQUIRED (missing -> argparse error)

def load_model_and_tokenizer(model_dir: str):
    model_dir = Path(model_dir).expanduser().resolve()
    if (model_dir / 'adapter_config.json').exists():
        # If you actually use PEFT adapters, ensure peft installed.
        try:
            from peft import AutoPeftModelForCausalLM
        except Exception as e:
            raise RuntimeError(
                "Detected adapter_config.json but 'peft' is not available. "
                "Install peft or provide a non-adapter model_dir."
            ) from e

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto"
        )
        tokenizer_dir = model.peft_config["default"].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map="auto"
        )
        tokenizer_dir = model_dir

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return model, tokenizer


def infer(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    start_time = time.time()
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    end_time = time.time()
    infer_time = end_time - start_time

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response, infer_time


def get_infer_main(model, tokenizer, iter_n, data_path, save_path_tmpl, width=3, retry=1, expected_kg_class=None):
    data = list(jsonlines.open(data_path))
    if 'simpleques' in data_path:
        data = random.sample(data, min(1000, len(data)))

    data_cate = []
    data_stat_info = {
        'infer_num': 0,
        'w/o_stage3_hit1_num': 0,
        'stage3_hit1_num': 0,
        'stage3_iter_hit1_num': 0,
        'w/o_stage3_f1_sum': 0,
        'stage3_f1_sum': 0,
        'stage3_iter_f1_sum': 0,
    }

    for d_idx, dd in tqdm(enumerate(data), total=len(data)):
        exe_num = 0
        exe_time = 0

        if 'parses' in dd:
            d = dd['parses']['0']
        else:
            d = dd

        data_source = d['source']
        data_class = data_source.split('-')[0]

        if expected_kg_class is not None and data_class != expected_kg_class:
            raise ValueError(
                f"[data mismatch] Expected kg_class='{expected_kg_class}' from CLI, "
                f"but found sample data_class='{data_class}' (source='{data_source}'). "
                f"Please check your data_path or kg_class."
            )

        if data_class not in data_cate:
            data_cate.append(data_class)
            print(data_class, 'begin infer!!!')

        raw_idx = d['raw_idx']
        data_kg = data_class_2_data_kg[data_class]

        # stage 1
        prompt = d["prompt"]
        retry_n = retry
        while retry_n > 0:
            try:
                raw_output, infer_time = infer(prompt, model, tokenizer)
                check, whole_plan = check_extract_modify_get_plan(raw_output)
                if check is False:
                    retry_n -= 1
                    continue

                retr_hop = data_class_2_hop[data_class]
                sim_rels_dict = get_sim_rels_list(
                    data_kg, d, whole_plan.split('\n'), data_class, raw_idx,
                    mid2name, emb_model, retr_hop, 5
                )
                break
            except Exception as e:
                print(e, 'something wrong!')
                retry_n -= 1
                continue

        if retry_n == 0:
            print(d_idx, 'infer failed! give up!')
            continue

        d['stage1_input'] = prompt
        d['stage1_raw_output'] = raw_output
        d['stage1_plan'] = whole_plan
        d['stage1_infer_time'] = infer_time
        d['sim_n_hop_rels'] = sim_rels_dict

        if 'parses' in dd:
            dd['parses']['0'] = d
        else:
            dd = d

        if 'cwq' in data_class:
            prompt = get_ai_input_stage2(PROMPT_2_shot, dd, width=width, few_shot=True)
        else:
            prompt = get_ai_input_stage2(PROMPT_2, dd, width=width)

        # stage 2
        retry_n = retry
        while retry_n > 0:
            try:
                raw_output, infer_time = infer(prompt, model, tokenizer)
                check, whole_plan = check_extract_modify_get_plan(raw_output)
                if check is False:
                    retry_n -= 1
                    continue

                start_time = time.time()
                raw_plan_list, ret, return_ent, wrong_dict = execute_whole_plan(
                    data_kg, dd, data_class, whole_plan, mid2name, emb_model
                )
                end_time = time.time()
                exe_time2 = end_time - start_time
                report = get_retr_result_report(raw_plan_list, ret, return_ent, wrong_dict)
                f1_score = cal_pre_f1_score(dd, ret, return_ent, mid2name, data_class)
                break
            except Exception as e:
                print(e, 'something wrong!')
                retry_n -= 1
                continue

        if retry_n == 0:
            print(d_idx, 'infer failed! give up!')
            continue

        d['stage2_input'] = prompt
        d['stage2_output'] = raw_output
        d['stage2_plan'] = whole_plan
        d['stage2_f1_score'] = f1_score
        d['stage2_infer_time'] = infer_time
        d['stage2_wrong_dict'] = wrong_dict
        d['stage2_exe_report'] = report
        exe_num += 1
        exe_time += exe_time2

        if 'parses' in dd:
            dd['parses']['0'] = d
        else:
            dd = d

        if 'cwq' in data_class:
            prompt = get_ai_input_stage3(PROMPT_3_shot, dd, few_shot=True)
        else:
            prompt = get_ai_input_stage3(PROMPT_3, dd)

        # stage 3 (reflection)
        d['reflection_list'] = []
        for _ in range(iter_n):
            cur_ref_dict = dict()
            retry_n = retry
            while retry_n > 0:
                try:
                    raw_output, infer_time = infer(prompt, model, tokenizer)

                    if '<answer>' in raw_output and '</answer>' in raw_output:
                        if is_valid_ai_output_stage3(raw_output) is False:
                            retry_n -= 1
                            continue

                        cur_ref_dict['stage3_input'] = prompt
                        cur_ref_dict['stage3_output'] = raw_output
                        cur_ref_dict['stage3_f1_score'] = (
                            d['stage2_f1_score'] if d['reflection_list'] == []
                            else d['reflection_list'][-1]['stage3_f1_score']
                        )
                        cur_ref_dict['stage3_infer_time'] = infer_time
                    else:
                        check, whole_plan = check_extract_modify_get_plan(raw_output)
                        if check is False:
                            retry_n -= 1
                            continue

                        start_time = time.time()
                        raw_plan_list, ret, return_ent, wrong_dict = execute_whole_plan(
                            data_kg, dd, data_class, whole_plan, mid2name, emb_model
                        )
                        end_time = time.time()
                        exe_time3 = end_time - start_time
                        exe_num += 1
                        exe_time += exe_time3

                        report = get_retr_result_report(raw_plan_list, ret, return_ent, wrong_dict)
                        f1_score = cal_pre_f1_score(dd, ret, return_ent, mid2name, data_class)

                        cur_ref_dict['stage3_input'] = prompt
                        cur_ref_dict['stage3_output'] = raw_output
                        cur_ref_dict['plan'] = whole_plan
                        cur_ref_dict['stage3_wrong_dict'] = wrong_dict
                        cur_ref_dict['stage3_exe_report'] = report
                        cur_ref_dict['stage3_f1_score'] = f1_score
                        cur_ref_dict['stage3_infer_time'] = infer_time

                    break
                except Exception as e:
                    print(e, 'something wrong!')
                    retry_n -= 1
                    continue

            if retry_n == 0:
                print(d_idx, 'infer failed! give up!')
                continue

            d['reflection_list'].append(cur_ref_dict)
            d['avg_exe_time'] = exe_time / max(exe_num, 1)

            if 'plan' not in cur_ref_dict:
                break

            if 'parses' in dd:
                dd['parses']['0'] = d
            else:
                dd = d

            if 'cwq' in data_class:
                prompt = get_ai_input_stage3_ref(PROMPT_3_shot, dd, few_shot=True)
            else:
                prompt = get_ai_input_stage3_ref(PROMPT_3, dd)

        if d['reflection_list'] == []:
            continue

        if 'parses' in dd:
            dd['parses']['0'] = d
        else:
            dd = d

        cur_save_path = save_path_tmpl.format(data_class, exp_name)
        with open(cur_save_path, 'a', encoding='utf-8') as save:
            save.write(json.dumps(dd, ensure_ascii=False) + '\n')

    return


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="RePlanKG inference script with CLI args (dataset-specific KG loaders via subcommands)."
    )

    parser.add_argument("--llm_path", type=str, required=True, help="Path to the LLM (HF model dir).")
    parser.add_argument("--mid2name_path", type=str, required=True, help="Path to mid2name.pkl.")
    parser.add_argument("--emb_model_path", type=str, required=True, help="Path to embedding model (FlagEmbedding finetuned dir).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to inference jsonl (prompted test data).")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for output naming.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference outputs.")

    parser.add_argument("--width", type=int, default=3, help="Exploration breadth (default: 3).")
    parser.add_argument("--iter_n", type=int, default=1, help="Reflection iterations (default: 1).")
    parser.add_argument("--hop", type=int, default=None, help="Optional override for hop; must match kg_class mapping if provided.")
    parser.add_argument("--retry", type=int, required=True, help="Retry times for each stage (required).")

    subparsers = parser.add_subparsers(dest="kg_class", required=True)

    p_sq = subparsers.add_parser("simpleques", help="Use SimpleQuestions KG loader.")
    p_sq.add_argument("--sq_test_jsonl", type=str, required=True, help="SimpleQuestions test jsonl.")
    p_sq.add_argument("--sq_test_subgraph_jsonl", type=str, required=True, help="SimpleQuestions test subgraph jsonl.")

    p_wq = subparsers.add_parser("webqsp", help="Use WebQSP KG loader.")
    p_wq.add_argument("--wq_test_jsonl", type=str, required=True, help="WebQSP test jsonl.")
    p_wq.add_argument("--wq_test_simple_jsonl", type=str, required=True, help="WebQSP test simple jsonl (e.g., test_simple1.jsonl).")
    p_wq.add_argument("--wq_test_com_jsonl", type=str, required=True, help="WebQSP test compositional jsonl (e.g., webqsp_test_com2.jsonl).")
    p_wq.add_argument("--wq_dir", type=str, required=True, help="WebQSP dataset directory (used by loader).")

    p_cwq = subparsers.add_parser("cwq", help="Use CWQ KG loader.")
    p_cwq.add_argument("--cwq_test_jsonl", type=str, required=True, help="CWQ test jsonl.")
    p_cwq.add_argument("--cwq_test_simple_jsonl", type=str, required=True, help="CWQ test simple jsonl (e.g., test_simple1.jsonl).")
    p_cwq.add_argument("--cwq_test_com_jsonl", type=str, required=True, help="CWQ test compositional jsonl (e.g., cwq_test_com5.jsonl).")
    p_cwq.add_argument("--cwq_dir", type=str, required=True, help="CWQ dataset directory (used by loader).")

    return parser


def validate_args(args):
    if args.width <= 0:
        raise ValueError("--width must be positive.")
    if args.iter_n <= 0:
        raise ValueError("--iter_n must be positive.")
    if args.retry <= 0:
        raise ValueError("--retry must be positive.")

    kg_class_2_hop = {"simpleques": 1, "webqsp": 2, "cwq": 4}
    expected_hop = kg_class_2_hop[args.kg_class]
    if args.hop is None:
        args.hop = expected_hop

    must_exist = [
        ("--llm_path", args.llm_path),
        ("--mid2name_path", args.mid2name_path),
        ("--emb_model_path", args.emb_model_path),
        ("--data_path", args.data_path),
        ("--output_dir", args.output_dir),
    ]
    if args.kg_class == "simpleques":
        must_exist += [
            ("--sq_test_jsonl", args.sq_test_jsonl),
            ("--sq_test_subgraph_jsonl", args.sq_test_subgraph_jsonl),
        ]
    elif args.kg_class == "webqsp":
        must_exist += [
            ("--wq_test_jsonl", args.wq_test_jsonl),
            ("--wq_test_simple_jsonl", args.wq_test_simple_jsonl),
            ("--wq_test_com_jsonl", args.wq_test_com_jsonl),
            ("--wq_dir", args.wq_dir),
        ]
    elif args.kg_class == "cwq":
        must_exist += [
            ("--cwq_test_jsonl", args.cwq_test_jsonl),
            ("--cwq_test_simple_jsonl", args.cwq_test_simple_jsonl),
            ("--cwq_test_com_jsonl", args.cwq_test_com_jsonl),
            ("--cwq_dir", args.cwq_dir),
        ]

    for flag, p in must_exist:
        if not Path(p).expanduser().exists():
            raise FileNotFoundError(f"{flag} path not found: {p}")

    Path(args.output_dir).expanduser().mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    args = validate_args(args)

    exp_name = args.exp_name  # used inside get_infer_main

    print("#" * 10, "import mid2name")
    with open(Path(args.mid2name_path).expanduser(), "rb") as f:
        mid2name = pickle.load(f)

    print("#" * 10, "import embedding model")
    emb_model = FlagAutoModel.from_finetuned(
        str(Path(args.emb_model_path).expanduser()),
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
        use_fp16=True
    )

    print("#" * 10, "import data_kg")
    if args.kg_class == "simpleques":
        simpleques = get_simpleques(
            str(Path(args.sq_test_jsonl).expanduser()),
            str(Path(args.sq_test_subgraph_jsonl).expanduser())
        )
        data_class_2_data_kg = {"simpleques": simpleques}
        data_class_2_hop = {"simpleques": args.hop}

    elif args.kg_class == "webqsp":
        webqsp = get_webqsp(
            str(Path(args.wq_test_jsonl).expanduser()),
            str(Path(args.wq_test_simple_jsonl).expanduser()),
            str(Path(args.wq_test_com_jsonl).expanduser()),
            str(Path(args.wq_dir).expanduser())
        )
        data_class_2_data_kg = {"webqsp": webqsp}
        data_class_2_hop = {"webqsp": args.hop}

    elif args.kg_class == "cwq":
        cwq = get_cwq(
            str(Path(args.cwq_test_jsonl).expanduser()),
            str(Path(args.cwq_test_simple_jsonl).expanduser()),
            str(Path(args.cwq_test_com_jsonl).expanduser()),
            str(Path(args.cwq_dir).expanduser())
        )
        data_class_2_data_kg = {"cwq": cwq, "cwq_train": cwq}
        data_class_2_hop = {"cwq": args.hop, "cwq_train": args.hop}

    else:
        raise ValueError(f"Unsupported kg_class: {args.kg_class}")

    print("#" * 10, "load llm")
    model, tokenizer = load_model_and_tokenizer(args.llm_path)

    save_path_tmpl = str(Path(args.output_dir).expanduser() / "{}_{}_infer.jsonl")

    get_infer_main(
        model=model,
        tokenizer=tokenizer,
        iter_n=args.iter_n,
        data_path=str(Path(args.data_path).expanduser()),
        save_path_tmpl=save_path_tmpl,
        width=args.width,
        retry=args.retry,
        expected_kg_class=args.kg_class
    )
