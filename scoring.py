import json
from vllm_models import Llama3VLLM
import tqdm
import argparse
import os
import multiprocessing
import math
import random
import re
from copy import deepcopy
import tempfile
from utils import is_port_in_use

random.seed(42)

HOME_DIR = os.environ["HOME"]


def __pairwise(args):
    if args.num_workers > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpuids))
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.port)
    output_dir = args.output_dir
    with open(args.input_dir) as f:
        data = [json.loads(x) for x in tqdm.tqdm(f, desc="loading data")]

    with open(args.src_dir) as f:
        prompts = [json.loads(x) for x in tqdm.tqdm(f, desc="loading source")]

    assert len(data) == len(prompts)
    print("loading model")
    if args.model_type == "offsetbias-lm":
        model = Llama3VLLM(
            model_pt=args.model_pt,
            tensor_parallel_size=len(args.gpuids),
            gpu_memory_utilization=0.9,
            download_dir=os.path.join(HOME_DIR, ".cache/huggingface/hub"),
            quantization=None,
            swap_space=8,
            max_input_len=5000,
            max_model_len=5120,
        )
        with open("prompts/offsetbias_lm.txt", encoding="utf-8") as f:
            prompt_template = f.read().strip()
    else:
        raise NotImplementedError(f"model_type {args.model_type} not implemented")

    pairs = []
    inputs = []
    for d, p in zip(data, prompts):
        prompt = p["prompt"]
        # create pairwise inputs
        for i in range(len(d)):
            for j in range(len(d)):
                if i != j:
                    pairs.append((d[i]["text"], d[j]["text"]))
                    inputs.append(
                        [
                            {
                                "role": "user",
                                "content": prompt_template.format_map(
                                    {
                                        "instruction": prompt,
                                        "output_1": d[i]["text"],
                                        "output_2": d[j]["text"],
                                    }
                                ),
                            }
                        ]
                    )

    print("Number of inputs", len(inputs))
    print("Number of prompts", len(prompts))

    batch_size = args.batch_size

    def parse_output(text, verbose=False):
        pattern = r"Output \((\S+)\)"
        match = re.search(pattern, text)
        if match:
            answer = match.group(1)
            if answer == "a":
                result = 0
            elif answer == "b":
                result = 1
            else:
                result = random.randint(0, 1)
                if verbose:
                    print(f"Invalid answer {answer}: {text}")
        else:
            result = random.randint(0, 1)
            if verbose:
                print(f"No matching pattern: {text}")
        return result

    predictions = []
    with open(output_dir, "w") as f:
        for i in tqdm.tqdm(range(0, len(inputs), batch_size), desc="scoring pairs", disable=not args.is_master):
            batch = inputs[i : min(i + batch_size, len(inputs))]
            results = model.generate(
                batch,
                n=1,
                max_tokens=16,
                temperature=0.0,
                logprobs=4,
                use_tqdm=False,
            )
            for x in results:
                winner = parse_output(x[0]["text"])
                x = x[0]
                x["winner"] = winner
                print(json.dumps(x), file=f, flush=True)
                predictions.append(x)

    results = []
    pos = 0
    for d, p in zip(data, prompts):
        prompt = p["prompt"]
        num_candidates = len(d)
        num_pairs = num_candidates * (num_candidates - 1)
        _predictions = predictions[pos : pos + num_pairs]
        _pairs = pairs[pos : pos + num_pairs]
        pos += num_pairs
        results.append({"prompt": prompt, "predictions": _predictions, "pairs": _pairs})

    print("Number of results", len(results))
    with open(output_dir, "w") as f:
        for x in results:
            print(json.dumps(x), file=f)


def pairwise(args):
    if args.num_workers == 1:
        __pairwise(args)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_workers = args.num_workers
            data = []
            with open(args.input_dir) as f:
                for line in tqdm.tqdm(f, desc="loading data"):
                    d = json.loads(line)
                    data.append(d)
            trunk_size = math.ceil(len(data) / num_workers)
            # split data
            for i in range(num_workers):
                with open(os.path.join(tmpdir, f"input_dir_{i}.jsonl"), "w") as f:
                    for d in data[i * trunk_size : min((i + 1) * trunk_size, len(data))]:
                        print(json.dumps(d), file=f)
            prompts = []
            with open(args.src_dir) as f:
                for line in tqdm.tqdm(f, desc="loading source"):
                    d = json.loads(line)
                    prompts.append(d)
            trunk_size = math.ceil(len(prompts) / num_workers)
            # split prompts
            for i in range(num_workers):
                with open(os.path.join(tmpdir, f"src_dir_{i}.jsonl"), "w") as f:
                    for d in prompts[i * trunk_size : min((i + 1) * trunk_size, len(prompts))]:
                        print(json.dumps(d), file=f)
            processes = []
            num_gpus = len(args.gpuids)
            port = args.port
            assert num_gpus % num_workers == 0
            # start processes
            for i in range(num_workers):
                _args = deepcopy(args)
                _args.input_dir = os.path.join(tmpdir, f"input_dir_{i}.jsonl")
                _args.output_dir = args.output_dir.replace(".jsonl", f"_{i}.jsonl")
                _args.src_dir = os.path.join(tmpdir, f"src_dir_{i}.jsonl")
                _num_gpus = num_gpus // num_workers
                _args.gpuids = args.gpuids[i * _num_gpus : (i + 1) * _num_gpus]
                while is_port_in_use(port):
                    port += 1
                _args.port = port
                port += 1
                if i != 0:
                    _args.is_master = False
                p = multiprocessing.Process(
                    target=__pairwise,
                    args=(_args,),
                )
                p.start()
                processes.append(p)
            # join
            for p in processes:
                p.join()
            # merge
            with open(args.output_dir, "w") as f:
                for i in range(num_workers):
                    _output_dir = args.output_dir.replace(".jsonl", f"_{i}.jsonl")
                    with open(_output_dir) as f_in:
                        for line in f_in:
                            data = json.loads(line)
                            print(json.dumps(data), file=f)
                    os.remove(_output_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--input_dir", type=str, help="input directory")
    parser.add_argument("--src_dir", type=str, help="source directory")
    parser.add_argument("--gpuids", type=int, nargs="+", help="gpu ids")
    parser.add_argument("--output_dir", type=str, help="output directory")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["offsetbias-lm"],
        default="offsetbias-lm",
        help="model type",
    )
    parser.add_argument("--model_pt", type=str, help="model path")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--score_mode", type=str, choices=["pairwise"], default="pairwise"
    )
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--is_master", type=bool, default=True)
    parser.add_argument("--port", type=int, default=28500)
    args = parser.parse_args()
    if args.score_mode == "pairwise":
        pairwise(args)
    else:
        raise NotImplementedError(f"score_mode {args.score_mode} not implemented")
