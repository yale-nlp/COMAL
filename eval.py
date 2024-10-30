from datasets import load_dataset
from sampling import sampling
import tempfile
import json
import argparse
import tqdm
import os
from vllm_models import Llama3VLLM
import random
import re
from transformers import AutoTokenizer
from scipy.special import logsumexp
import math

HOME_DIR = os.environ["HOME"]


def generate(prompts, args):
    with tempfile.NamedTemporaryFile("w") as f:
        for prompt in prompts:
            print(json.dumps({"prompt": prompt}), file=f)
        f.flush()
        args.extend(["--input_dir", f.name])
        sampling(args)


def pairwise_comparison_offsetbias(args, prompts, output_pairs, output_dir):
    parser = argparse.ArgumentParser("eval_alpaca_offsetbias")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args(args)
    num_gpus = args.num_gpus
    batch_size = args.batch_size
    assert len(prompts) == len(output_pairs)

    model = Llama3VLLM(
        model_pt="NCSOFT/Llama-3-OffsetBias-8B",
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.9,
        download_dir=os.path.join(HOME_DIR, ".cache/huggingface/hub"),
        quantization=None,
        swap_space=8,
        max_input_len=5000,
        max_model_len=5120,
    )
    with open("prompts/offsetbias_lm.txt", encoding="utf-8") as f:
        prompt_template = f.read().strip()

    inputs = []

    def build_input(prompt, output_1, output_2):
        return [
            {
                "role": "user",
                "content": prompt_template.format_map(
                    {
                        "instruction": prompt,
                        "output_1": output_1,
                        "output_2": output_2,
                    }
                ),
            }
        ]

    for prompt, output_pair in zip(prompts, output_pairs):
        inputs.append(build_input(prompt, output_pair[0], output_pair[1]))
        inputs.append(build_input(prompt, output_pair[1], output_pair[0]))

    tokenizer = AutoTokenizer.from_pretrained("NCSOFT/Llama-3-OffsetBias-8B")

    def parse_response(
        response: dict,
        sys1_marker: str = "a",
        sys2_marker: str = "b",
        pattern: str = r"Output \((\S+)\)",
        verbose: bool = False,
    ) -> tuple[dict, bool]:
        """
        Parse the response from the model.

        Args:
            response: The response from the model.
            sys1_marker: The marker for system 1.
            sys2_marker: The marker for system 2.
            verbose: Whether to print verbose output.
            tokenizer: The tokenizer to use.
            pattern: The pattern to match the response.

        Returns:
            tuple[dict, bool]: The parsed response and whether the parsing failed.
        """
        text = response["text"]
        match = re.search(pattern, text)
        if match:
            start_index, end_index = match.span(1)
            found_token = text[start_index:end_index]
            prefix_index = len(tokenizer.tokenize(text[:start_index])) - 1
            label_index = len(tokenizer.tokenize(text[:end_index])) - 1
            if label_index - prefix_index > 1:
                # raise ValueError("More than one token in the label")
                print("Warning: More than one token in the label")
                label_index = 0
            token = response["tokens"][label_index]
            if token != found_token:
                print(f"Warning: Token {token} does not match found {found_token}")
                label_index = 0
            logprobs = response["logprobs"][label_index]
        else:
            # no mathing pattern, use the first token
            logprobs = response["logprobs"][0]
            if verbose:
                print(f"No matching pattern for {response['text']}")
        tokens = logprobs.keys()
        if sys1_marker in tokens and sys2_marker in tokens:
            logsum = logsumexp([logprobs[sys1_marker], logprobs[sys2_marker]])
            score_1 = math.exp(logprobs[sys1_marker] - logsum)
            score_2 = math.exp(logprobs[sys2_marker] - logsum)
            if logprobs[sys1_marker] > logprobs[sys2_marker]:
                result = 1
            elif logprobs[sys1_marker] < logprobs[sys2_marker]:
                result = 2
            else:
                result = random.randint(1, 2)
        elif sys1_marker in tokens:
            result = 1
            score_1 = 1
            score_2 = 0
        elif sys2_marker in tokens:
            result = 2
            score_1 = 0
            score_2 = 1
        else:
            if verbose:
                print(f"Empty logprobs for {response['text']}")
            result = random.randint(1, 2)
            score_1 = 0.5
            score_2 = 0.5

        result = {"winner": result}
        result["logprobs_1"] = logprobs[sys1_marker] if sys1_marker in tokens else None
        result["logprobs_2"] = logprobs[sys2_marker] if sys2_marker in tokens else None
        result["score_1"] = score_1
        result["score_2"] = score_2
        return result

    predictions = []

    for i in tqdm.tqdm(range(0, len(inputs), batch_size), desc="scoring pairs"):
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
            prediction = parse_response(x[0])
            x = x[0]
            x["prediction"] = prediction
            predictions.append(x)

    results = []
    avg_score_1 = 0
    avg_score_2 = 0
    win_1 = 0
    win_2 = 0
    for i in range(0, len(predictions), 2):
        score_1 = (
            predictions[i]["prediction"]["score_1"]
            + predictions[i + 1]["prediction"]["score_2"]
        ) / 2
        score_2 = (
            predictions[i]["prediction"]["score_2"]
            + predictions[i + 1]["prediction"]["score_1"]
        ) / 2
        if score_1 > score_2:
            winner = 1
        elif score_1 < score_2:
            winner = 2
        else:
            winner = random.randint(1, 2)
        results.append(
            {
                "prompt": prompts[i // 2],
                "output_1": output_pairs[i // 2][0],
                "output_2": output_pairs[i // 2][1],
                "prediction_1": predictions[i]["prediction"],
                "prediction_2": predictions[i + 1]["prediction"],
                "score_1": score_1,
                "score_2": score_2,
                "winner": winner,
            }
        )
        avg_score_1 += score_1
        avg_score_2 += score_2
        if winner == 1:
            win_1 += 1
        elif winner == 2:
            win_2 += 1

    avg_score_1 /= len(results)
    avg_score_2 /= len(results)
    win_1 /= len(results)
    win_2 /= len(results)

    length1, length2 = 0, 0

    for pair in output_pairs:
        length1 += len(pair[0])
        length2 += len(pair[1])

    length1 /= len(output_pairs)
    length2 /= len(output_pairs)

    with open(output_dir, "w") as f:
        json.dump(
            {
                "score_1": avg_score_1,
                "score_2": avg_score_2,
                "win_1": win_1,
                "win_2": win_2,
                "length_1": length1,
                "length_2": length2,
            },
            f,
            indent=4,
        )

    with open(output_dir.replace(".json", "_details.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(
        json.dumps(
            {
                "score_1": avg_score_1,
                "score_2": avg_score_2,
                "win_1": win_1,
                "win_2": win_2,
                "length_1": length1,
                "length_2": length2,
            },
            indent=4,
        )
    )


def gen_alpaca(args):
    dataset = load_dataset("tatsu-lab/alpaca_eval")["eval"]
    prompts = [x["instruction"] for x in dataset]
    generate(prompts, args)


def eval_alpaca(args):
    parser = argparse.ArgumentParser("eval_alpaca")
    parser.add_argument(
        "--evaluator",
        type=str,
        choices=["offsetbias"],
        default="offsetbias",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sys1_dir", type=str, required=True)
    parser.add_argument("--sys2_dir", type=str, required=True)
    args, remaining_args = parser.parse_known_args(args)

    dataset = load_dataset("tatsu-lab/alpaca_eval")["eval"]
    prompts = [x["instruction"] for x in dataset]

    if args.sys1_dir == "ref":
        print("Using the default reference system")
        sys1_output = load_dataset(
            "tatsu-lab/alpaca_eval", "alpaca_eval_gpt4_baseline"
        )["eval"]
        sys1_output = [x["output"] for x in sys1_output]
    else:
        with open(args.sys1_dir) as f:
            sys1_output = [json.loads(line)["text"] for line in f]
    with open(args.sys2_dir) as f:
        sys2_output = [json.loads(line)["text"] for line in f]

    assert len(sys1_output) == len(prompts)
    assert len(sys2_output) == len(prompts)

    print(len(sys1_output), len(sys2_output), len(prompts))

    output_pairs = list(zip(sys1_output, sys2_output))

    if args.evaluator == "offsetbias":
        pairwise_comparison_offsetbias(
            remaining_args, prompts, output_pairs, args.output_dir
        )
    else:
        raise NotImplementedError(f"evaluator {args.evaluator} not implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("main")
    parser.add_argument(
        "--task", type=str, choices=["gen_alpaca", "eval_alpaca"], required=True
    )
    args, remaining_args = parser.parse_known_args()
    if args.task == "gen_alpaca":
        gen_alpaca(remaining_args)
    elif args.task == "eval_alpaca":
        eval_alpaca(remaining_args)
    else:
        raise NotImplementedError(f"task {args.task} not implemented")
