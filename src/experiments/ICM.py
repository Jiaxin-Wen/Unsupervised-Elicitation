import asyncio
import json
import math
import os
import random
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import argparse

from core.llm_api.llm import ModelAPI
from core.utils import setup_environment
from src.experiments.ICM_tools import (
    propose_consistencyfix,
    run_consistencyfix,
    pick_two_inconsistent_claims,
    update_assign_based_on_decision,
)
from src.model_querying.prompt_creation import (
    get_decision_prompt,
    get_judge_prompt_fewshot,
)
from src.model_querying.solution_extraction import (
    extract_claim_logprobs,
    extract_decision_logprobs,
)
from src.pipeline.pipeline import Pipeline, PipelineConfig
from src.tools.dataloaders import (
    load_assignments,
    load_problems_from_json,
    load_problems_from_json_ids,
)
from src.tools.path_utils import get_default_results_directory, get_root_directory


def calculate_accuracy(train_data, inconsistent_pairs):
    train_probs = []
    for i in train_data.values():
        if i["label"] is None:
            continue
        if i["label"] == 1:
            train_probs.append(i["score"])
        else:
            train_probs.append(-i["score"])
    if len(train_probs) == 0:
        train_prob = 0
    else:
        train_prob = np.mean(train_probs)

    return {
        "train_accuracy": 0
        if len(train_data) == 0
        else np.mean([i["label"] == i["vanilla_label"] for i in train_data.values()]),
        "train_label_distribution": Counter(
            [i["vanilla_label"] for i in train_data.values()]
        ),
        "train_predict_distribution": Counter(
            [i["label"] for i in train_data.values()]
        ),
        "train_prob": train_prob,
        "train_size": len(train_data),
        "inconsistent_num": len(inconsistent_pairs),
    }


def update_assign(data):
    for key, value in data.items():
        if value["score"] > 0:
            value["label"] = 1
        else:
            value["label"] = 0
    return data


def fix_inconsistency(demonstrations, cur_metric, name, alpha, iter=0, K=20):
    backup_metric = deepcopy(cur_metric)
    if cur_metric["inconsistent_num"] == 0:
        return demonstrations, cur_metric

    cur_pool = {k: v for k, v in demonstrations.items() if v["label"] is not None}
    assignment = cur_pool
    
    best_metric = cur_metric
    best_assignment = assignment
    best_decision_id = None
    for k in range(K):
        pipeline = propose_consistencyfix(
            args.model,
            name=name,
            iter=f"{iter}-{k}",
            assignment=assignment,
        )
        results = asyncio.run(pipeline.run())
        decisions = results["decisions"]
        assignment = results["get_assign"]
        for decision_id, decision in enumerate(decisions.values()):
            tmp_decision_metric_list = []
            tmp_decision_assignment_list = []
            for score_idx, score in enumerate([0, 1]):
                tmp_decision = deepcopy(decision)
                tmp_decision["score"] = score
                tmp_assignment = update_assign_based_on_decision(
                    deepcopy(assignment), tmp_decision
                )
                tmp_pipeline = run_consistencyfix(
                    model=args.model,
                    name=name,
                    iter=f"{iter}-{k}-{decision_id}-{score_idx}",
                    assignment=tmp_assignment,
                )
                tmp_results = asyncio.run(tmp_pipeline.run())
                tmp_metric = tmp_results["evaluate"]
                tmp_decision_metric_list.append(tmp_metric)
                tmp_decision_assignment_list.append(tmp_assignment)
            tmp_best_decision_id = np.argmax(
                [get_energy(i, args.alpha) for i in tmp_decision_metric_list]
            )
            tmp_assignment = tmp_decision_assignment_list[tmp_best_decision_id]
            tmp_metric = tmp_decision_metric_list[tmp_best_decision_id]

            if get_energy(tmp_metric, args.alpha) >= get_energy(best_metric, args.alpha):
                best_decision_id = decision_id
                best_metric = tmp_metric
                best_assignment = tmp_assignment
                break
        if best_decision_id is None:
            break
        elif best_metric["inconsistent_num"] == 0:
            assignment = best_assignment
            break
        else:
            assignment = best_assignment

    for k in assignment:
        demonstrations[k] = assignment[k]

    return demonstrations, best_metric


def get_pipeline(
    model,
    name=None,
    use_cache=True,
    num_problems=None,
    decision_id=None,
    iter=None,
    assignment=None,
):
    pipeline_name = f"iterative-truth-assign-iter-{iter}"
    if decision_id is not None:
        pipeline_name += f"-{decision_id}"
    if name is not None:
        pipeline_name += "-" + name

    ROOT_DIR = get_root_directory()
    DATA_DIR = ROOT_DIR / "data"


    pipeline_config = PipelineConfig(
        pipeline_name,
        anthropic_num_threads=40,
        openai_fraction_rate_limit=0.99,
        num_problems=num_problems,
        use_cache=use_cache,
    )
    pipeline = Pipeline(pipeline_config)

    assert assignment is not None
    initial_assign = pipeline.add_load_data_step(
        "get_assign", load_assignments, assignment
    )

    def add_train_demonstrations(train_data):
        copy_data = deepcopy(train_data)
        copy_data = {k: v for k, v in copy_data.items() if v["label"] is not None}
        keys = list(copy_data.keys())
        values = list(copy_data.values())
        saved_keys = [
            "prompt",
            "question",
            "choice",
            "choice_2",
            "consistency_id",
            "consistency_key",
            "source",
            "label",
            "vanilla_label",
        ]
        values = []
        for i in copy_data.values():
            values.append({saved_key: i[saved_key] for saved_key in saved_keys if saved_key in i})

        for idx, key in enumerate(keys):
            tmp_keys, tmp_values = [], []
            for j, (prev_key, prev_value) in enumerate(zip(keys, values)):
                if j != idx:
                    tmp_keys.append(prev_key)
                    tmp_values.append(prev_value)

            demos = {
                prev_key: prev_value
                for j, (prev_key, prev_value) in enumerate(zip(tmp_keys, tmp_values))
            }

            sorted_demos = {}
            for k, v in demos.items():
                q = v["consistency_id"]
                if q not in sorted_demos:
                    sorted_demos[q] = []
                sorted_demos[q].append((k, v))

            out_sorted_demos = {}
            for group in sorted_demos.values():
                for k, v in group:
                    out_sorted_demos[k] = v
                    
            copy_data[key]["demonstration"] = out_sorted_demos

        return copy_data

    merged_train_data = pipeline.add_transformation_step(
        "add_train_demonstration",
        add_train_demonstrations,
        dependencies=[initial_assign],
    )

    get_train_preds = pipeline.add_query_step(
        "get_train_preds",
        model,
        get_judge_prompt_fewshot,
        extract_claim_logprobs,
        dependencies=[merged_train_data],
        logprobs=20,
        max_tokens=1,
        use_cache=use_cache,
    )

    pick_claims = pipeline.add_transformation_step(
        "pick_two_inconsistent_claims",
        pick_two_inconsistent_claims,
        dependencies=[initial_assign],
    )

    eval_preds = pipeline.add_eval_step(
        "evaluate",
        calculate_accuracy,
        dependencies=[get_train_preds, pick_claims],
    )
    return pipeline


async def predict_assignment(model, example, demonstrations):
    demos = [
        v
        for k, v in demonstrations.items()
        if k != example["uid"] and v["label"] is not None
    ]
    
    # Use call_single for cleaner single response handling
    try:
        completion = await model_api.call_single(
            model,
            get_judge_prompt_fewshot(
                example,
                demos,
                pipeline=False,
            ),
            logprobs=20,
            max_tokens=1,
        )
        
        # Create a response structure compatible with extract_claim_logprobs
        response_dict = {
            "response": completion,
            "metadata": {"uid": example.get("uid", 0)}
        }
        
        # Extract score using the existing parse function
        parsed_result = extract_claim_logprobs(response_dict)
        score = parsed_result.get("score", 0.1 if "true" in str(completion).lower() else -0.1)
            
        new_label = score > 0
        return int(new_label)
        
    except Exception as e:
        print(f"Error in predict_assignment: {e}")
        # Fallback to random assignment
        return random.choice([0, 1])


def get_temperature(
    iteration, initial_temp, final_temp, decay_rate, schedule="exp"
):
    """
    Calculate the temperature for simulated annealing.

    Parameters:
    - iteration: Current iteration number.
    - initial_temp: Initial temperature.
    - decay_rate: Rate at which the temperature decreases.

    Returns:
    - Current temperature.
    """
    if schedule == "exp":
        return max(final_temp, initial_temp * (decay_rate**iteration))
    elif schedule == "log":
        return max(final_temp, initial_temp / (1 + 2 * np.log(1 + iteration)))
    else:
        assert False


def get_energy(metric, alpha):
    return alpha * metric["train_prob"] - metric["inconsistent_num"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=30)
    parser.add_argument("--seed", type=int, default=27565976)
    parser.add_argument("--testbed", type=str, default="gsm8k")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_seed", type=int, default=8)
    parser.add_argument("--K", type=int, default=3000)
    parser.add_argument("--consistency_fix_K", type=int, default=10)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--initial_T", type=float, default=10)
    parser.add_argument("--final_T", type=float, default=0.01)
    parser.add_argument("--scheduler", type=str, default="log")
    # Add Ollama-specific arguments
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434", 
                       help="Ollama server host URL")
    parser.add_argument("--ollama_temperature", type=float, default=0.0,
                       help="Temperature for Ollama models")
    parser.add_argument("--use_ollama", action="store_true",
                       help="Force use of Ollama for the specified model")
    args = parser.parse_args()
    return args

def load_data(args):
    data_dir = get_root_directory() / "data"
    
    if args.testbed == "alpaca":
        data_file = data_dir / "train_alpaca.json"
        if not data_file.exists():
            print(f"Data file not found: {data_file}")
            print("Please run: python -m src.tools.data_downloader")
            print("Or create the file manually.")
            raise FileNotFoundError(f"Required data file not found: {data_file}")
            
        with open(data_file) as f:
            train = json.load(f)
        template = """Human: {question}
Response 1: {choice}
Response 2: {choice_2}
Claim: Response 1 is more helpful and harmless than Response 2.
I think this claim is """
        for i in train:
            i['source'] = 'alpaca'
            i['consistency_key'] = 'A>B' if i['label'] else 'B>A'
            i['prompt'] = template.format(question=i['question'], choice=i['choice'].replace("\n", ""), choice_2=i['choice_2'].replace("\n", ""))
        
        args.GROUP_SIZE = 2
        
    elif args.testbed == 'gsm8k':
        data_file = data_dir / "train_gsm8k.json"
        if not data_file.exists():
            print(f"Data file not found: {data_file}")
            print("Please run: python -m src.tools.data_downloader")
            print("Or create the file manually.")
            raise FileNotFoundError(f"Required data file not found: {data_file}")
            
        with open(data_file) as f:
            train = json.load(f)
        template = """Question: {question}
Claim: {answer}
I think this claim is """

        for i in train:
            i['source'] = 'gsm8k'
            i['consistency_key'] = i['answer']
            i['prompt'] = template.format(question=i['question'], answer=i['choice'])

        args.GROUP_SIZE = 4
        
    elif args.testbed == 'truthfulQA':
        data_file = data_dir / "train_truthfulqa.json"
        if not data_file.exists():
            print(f"Data file not found: {data_file}")
            print("Please run: python -m src.tools.data_downloader")
            print("Or create the file manually.")
            raise FileNotFoundError(f"Required data file not found: {data_file}")
            
        with open(data_file) as f:
            train = json.load(f)
        template = """Question: {question}
Claim: {answer}
I think this claim is """

        for i in train:
            i['source'] = 'truthfulQA'
            i['consistency_key'] = 'A' if i['label'] else 'B'
            i['prompt'] = template.format(question=i['question'], answer=i['choice'])
        args.GROUP_SIZE = 4
        
    elif args.testbed == 'truthfulQA-preference':
        data_file = data_dir / "train_truthfulqa_preference.json"
        if not data_file.exists():
            print(f"Data file not found: {data_file}")
            print("Please run: python -m src.tools.data_downloader")
            print("Or create the file manually.")
            raise FileNotFoundError(f"Required data file not found: {data_file}")
            
        with open(data_file) as f:
            train = json.load(f)
        template = """Question: {question}
Answer 1: {choice}
Answer 2: {choice_2}
Claim: Answer 1 is more truthful than Answer 2.
I think this claim is """

        for i in train:
            i['source'] = 'truthfulQA-preference'
            i['consistency_key'] = 'A>B' if i['label'] else 'B>A'
            i['prompt'] = template.format(question=i['question'], choice=i['choice'], choice_2=i['choice_2'])
        args.GROUP_SIZE = 2
        
    train_map = {}
    for i in train:
        if i['consistency_id'] not in train_map:
            train_map[i['consistency_id']] = []
        train_map[i['consistency_id']].append(i)
    
    out = []
    for key in train_map:
        out += train_map[key]
    train = out
    
    # sample a batch of batch_size datapoints
    fewshot_ids = random.sample(
        list(range(len(train)// args.GROUP_SIZE)), args.batch_size // args.GROUP_SIZE
    )
    fewshot_ids = [
        i * args.GROUP_SIZE + j for i in fewshot_ids for j in range(args.GROUP_SIZE)
    ]

    return train, fewshot_ids

def initialize(train, fewshot_ids, args):
    demonstrations = {}
    unlabeled_ids = []
    whole_ids = []
    seed_ids = []

    random_init_labels = [1] * (args.num_seed // 2) + [0] * (args.num_seed // 2)
    random.shuffle(random_init_labels)
    
    for id, i in enumerate(fewshot_ids):
        item = train[i]
        item["vanilla_label"] = item["label"] # store dataset labels to measure agreement during the searching process
        item["uid"] = id
        whole_ids.append(item["uid"])
        if id >= args.num_seed:  # set labels to None
            item["label"] = None
            item["type"] = "predict"
            unlabeled_ids.append(item["uid"])
        else: # set random labels
            item["type"] = "seed"
            item["label"] = random_init_labels[id]
            seed_ids.append(item["uid"])
        demonstrations[id] = item
        
    return demonstrations, unlabeled_ids, whole_ids, seed_ids


def main(args):
    # Validate model selection for Ollama
    if args.use_ollama or _is_likely_ollama_model(args.model):
        print(f"Using Ollama model: {args.model}")
        print(f"Ollama host: {args.ollama_host}")
        # Verify Ollama is available
        if not _check_ollama_available():
            print("Warning: Ollama service may not be available")
            print("Start Ollama with: ollama serve")
            print("Pull model with: ollama pull " + args.model.split(':')[0])
    
    train, fewshot_ids = load_data(args)

    demonstrations, unlabeled_ids, whole_ids, seed_ids = initialize(train, fewshot_ids, args)
    
    cur_metric = {
        "train_prob": -1e6,
        "inconsistent_num": 100000,
        "train_accuracy": 1.0,
        "train_predict_distribution": {"0": 0, "1": 0},
        "train_label_distribution": {"0": 0, "1": 0},
    }
    
    print('init random labels = ', Counter([i['label'] for i in demonstrations.values() if i['type'] == 'seed']), 'init label acc = ', np.mean([i['label'] == i['vanilla_label'] for i in demonstrations.values() if i['type'] == 'seed']))
    
    # Update name to include model type
    model_name_safe = args.model.replace('/', '_').replace(':', '_')
    name = f"{args.testbed}-{model_name_safe}-K{args.K}-bc{args.batch_size}_seed{args.seed}-initialsize{args.num_seed}-weighted{args.alpha}-decay{args.decay}-initialT{args.initial_T}-finalT{args.final_T}-scheduler{args.scheduler}"

    iter = 0
    flip_cnt = 0
    example_id = 0

    for _ in tqdm(range(args.K), desc="searching"):
        cur_pool = {
            k: v for k, v in demonstrations.items() if v["label"] is not None
        }
        initial_demos = deepcopy(demonstrations)
        if iter == 0:
            pipeline = get_pipeline(
                args.model,
                name=name,
                num_problems=None,
                iter=iter,
                assignment=cur_pool,
            )
            results = asyncio.run(pipeline.run())
            cur_metric = results["evaluate"]
            
            demonstrations, cur_metric = fix_inconsistency(
                demonstrations, cur_metric, name, args.alpha, iter=iter, K=args.consistency_fix_K
            )
            
        cur_pool = {
            k: v for k, v in demonstrations.items() if v["label"] is not None
        }
       
        while True: # weighted sampling
            candidates_ids = whole_ids
            weights = [1 for _ in range(len(candidates_ids))]
            for i in candidates_ids:
                if i in cur_pool:
                    same_consistency_group_ids = [j for j in candidates_ids if demonstrations[j]["consistency_id"] == demonstrations[i]["consistency_id"]]
                    for j in same_consistency_group_ids:
                        if j not in cur_pool:
                            weights[j] = 100

            example_id = random.choices(candidates_ids, k=1, weights=weights)[0]
            break

        new_label = asyncio.run(
            predict_assignment(
                args.model,
                demonstrations[example_id],
                cur_pool,
            )
        )

        if demonstrations[example_id]["label"] != new_label:
            tmp_demonstrations = deepcopy(demonstrations)
            tmp_demonstrations[example_id]["label"] = new_label
            dummy_metric = {
                "train_prob": -1e6,
                "inconsistent_num": 100000,
                "train_accuracy": 1.0,
                "train_predict_distribution": {"0": 0, "1": 0},
                "train_label_distribution": {"0": 0, "1": 0},
            }

            tmp_demonstrations, _ = fix_inconsistency(
                tmp_demonstrations,
                dummy_metric,
                name + "newlabelexplore",
                args.alpha,
                iter=iter,
                K=10,
            )
            
            tmp_pool = {
                k: v
                for k, v in tmp_demonstrations.items()
                if v["label"] is not None
            }
            pipeline = get_pipeline(
                model=args.model,
                name=name,
                num_problems=None,
                iter=iter,
                assignment=tmp_pool,
            )
            results = asyncio.run(pipeline.run())
            metric = results["evaluate"]
            T = get_temperature(
                flip_cnt, args.initial_T, args.final_T, args.decay, schedule=args.scheduler
            )
            print(f"iter = {iter}, pool size = {len(cur_pool)}, cur acc = {cur_metric['train_accuracy']}, new acc = {metric['train_accuracy']}, cur score = {get_energy(cur_metric, args.alpha)}, new score = {get_energy(metric, args.alpha)}, cur inconsistent num = {cur_metric['inconsistent_num']}, new inconsistent num = {metric['inconsistent_num']}")
            print('cur label distribution = ', Counter([i['label'] for i in demonstrations.values() if i['label'] is not None]))
            print('new label distribution = ', Counter([i['label'] for i in tmp_demonstrations.values() if i['label'] is not None]))

            accept_prob = math.exp((get_energy(metric, args.alpha) - get_energy(cur_metric, args.alpha)) / T)
            print("accept prob = ", accept_prob)
            if random.random() < accept_prob:
                print("accept")
                demonstrations = tmp_demonstrations
                flip_cnt += 1
                cur_metric = metric
                with open(f"log_{name}.jsonl", "a") as f:
                    f.write(json.dumps({
                        "iter": iter,
                        "flip_cnt": flip_cnt,
                        "acc": cur_metric['train_accuracy'],
                        "score": get_energy(cur_metric, args.alpha),
                    }) + "\n")
            else:
                print("reject")
        
        print("=" * 100)
        iter += 1


def _is_likely_ollama_model(model_id: str) -> bool:
    """Check if a model_id looks like an Ollama model."""
    if any(prefix in model_id.lower() for prefix in ["gpt", "claude", "text-", "ft:"]):
        return False
    
    if ":" in model_id and "/" not in model_id:
        return True
        
    common_ollama_names = [
        "llama", "mistral", "mixtral", "qwen", "gemma", "phi", "vicuna", 
        "orca", "wizard", "neural", "starling", "codellama", "deepseek"
    ]
    return any(name in model_id.lower() for name in common_ollama_names)


def _check_ollama_available() -> bool:
    """Check if Ollama service is available."""
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    setup_environment(logger_level="error")
    args = get_args()
    
    # Initialize ModelAPI with Ollama settings
    model_api = ModelAPI(
        anthropic_num_threads=20, 
        openai_fraction_rate_limit=0.99,
        ollama_host=args.ollama_host,
        ollama_temperature=args.ollama_temperature
    )
    
    print("task: ", args.testbed)
    print("model: ", args.model)
    random.seed(args.seed)
    main(args)