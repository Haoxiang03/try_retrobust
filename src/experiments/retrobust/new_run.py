import ast

import numpy as np
from tqdm import tqdm
import argparse
import dataclasses
import json
import os
import random
from collections import Counter
from datetime import datetime
from typing import Dict, List
import pandas as pd
import sys
import torch

sys.path.append(os.path.join(os.getcwd()))
from src.common.config import Config
from src.common.logger import get_logger
from src.consts import (
    FULL_MTE_FIELD,
    PRED_MTE_FIELD,
    ACC_MTE_FIELD,
    ACC_AT_1_FIELD,
    ACC_AT_3_FIELD,
    ACC_AT_MAJORITY_FIELD,
    NUM_EXAMPLES_FIELD,
    NUM_ABSTAINS_FIELD,
    CONTEXT_ANSWER_SEP,
)
from src.dataclasses import (
    QuestionV1,
    QuestionWithAnswer,
    format_decompsition_string,
    format_ir_decomposition,
    QuestionV1Retrobust,
)
from src.dataset_readers.dataset_readers_factory import DatasetReadersFactory
from src.dataset_readers.readers.dataset_reader import DatasetReader
from src.pred_evaluators.evaluators_factory import EvaluatorsFactory
from src.pred_evaluators.pred_evaluators.base_evaluator import Evaluator
from src.prompting.prompt_factory import PromptFactoryDict
from src.serpapi.serpapi import get_string_hash, google
from src.serpapi.serpapi import get_question_wiki_snippet

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="/hpc_stor03/sjtu_home/haoxiang.jiang/RAG/cots/src/config/retrobust/nq/with_retrieval_top_5.json",
        help="Config file path",
    )
    return parser.parse_args()


def local_model_generate(prompts: List[str], model_path: str, temperature=0):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=512, temperature=temperature, num_return_sequences=1)

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


class LocalModelAccessor:

    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.model_path = "/hpc_stor03/sjtu_home/haoxiang.jiang/models/Qwen2.5-1.5B-Instruct"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    def _preprocess_input(self, prompt: str, additional_input: str = "") -> str:

        return prompt + additional_input

    def _postprocess_output(self, output: torch.Tensor) -> str:

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def call_local_model(self, prompt: str, additional_input: str = "", temperature: float = 1) -> str:

        full_input = self._preprocess_input(prompt, additional_input)

        inputs = self.tokenizer(full_input, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=1024,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self._postprocess_output(output)
        return generated_text

    def call_local_model_with_steps(self, prompt: str, additional_input: str = "", temperature: float = 1.0, num_steps: int = 5) -> list:

        decompositions = []
        for i in range(num_steps):
            step = self.call_local_model(prompt, additional_input, temperature)
            decompositions.append(step)
        return decompositions
    

def _call_local_model_for_entailment_with_batching(
    questions: List[Dict],
    evaluator: Evaluator,
    entailment_values: Dict,
    prefix="mte",
    batch_size: int = 10,
) -> List[Dict]:
    def batch_iterable(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx : min(ndx + n, l)]

    batches = [x for x in batch_iterable(list(questions), batch_size)]

    logger.info(f"Running entailment with batching using local model.")
    for batch in tqdm(batches):
        input_prompts = [question[f"{prefix}_entailment_input"] for question in batch]
        
        gpt_trace_entailment = local_model_generate(
            input_prompts, model_path=entailment_values.get("model"), temperature=0
        )

        for i, question in enumerate(batch):
            gpt_res = gpt_trace_entailment[i]
            question[f"{prefix}_{FULL_MTE_FIELD}"] = gpt_res

            if "the answer is:" in question[f"{prefix}_{FULL_MTE_FIELD}"]:
                final_answer = gpt_res.split("the answer is:")[-1].strip()

                if len(final_answer) and final_answer[-1] == ".":
                    final_answer = final_answer[:-1]
                question[f"{prefix}_{PRED_MTE_FIELD}"] = final_answer.replace(
                    "\n", ""
                ).strip()
                if "unknown" in question[f"{prefix}_{PRED_MTE_FIELD}"].lower():
                    mte_acc = None
                else:
                    mte_acc = evaluator.evaluate(
                        question[f"{prefix}_{PRED_MTE_FIELD}"]
                        .strip()
                        .replace("\n", ""),
                        question["metadata_gold_answer"]
                        if "metadata" not in question
                        else question["metadata"]["gold_answer"],
                    )
            else:
                mte_acc = None
            question[f"{prefix}_{ACC_MTE_FIELD}"] = mte_acc
    return list(questions)


def _report_results(suffix: str, examples: List[Dict]):
    """ """
    questions_with_entailment_df = pd.DataFrame(examples)
    res = {
        f"{NUM_EXAMPLES_FIELD}_{suffix}": questions_with_entailment_df.shape[0],
        f"{NUM_ABSTAINS_FIELD}_{suffix}": questions_with_entailment_df[
            f"mte_{ACC_MTE_FIELD}"
        ]
        .isna()
        .sum(),
        f"{ACC_AT_1_FIELD}_{suffix}": questions_with_entailment_df[
            ACC_AT_1_FIELD
        ].mean(),
        f"{ACC_AT_3_FIELD}_{suffix}": questions_with_entailment_df[
            ACC_AT_3_FIELD
        ].mean(),
        f"{ACC_AT_MAJORITY_FIELD}_{suffix}": questions_with_entailment_df[
            ACC_AT_MAJORITY_FIELD
        ].mean(),
        f"mte_{ACC_MTE_FIELD}_{suffix}": questions_with_entailment_df[
            f"mte_{ACC_MTE_FIELD}"
        ].mean(),
    }
    for report_func in [logger.info]:
        report_func(res)


def _save_examples(examples: List[Dict], output_path: str):
    """ """
    # format
    for ex in examples:
        # flatten metadata
        if "metadata" in ex:
            for k, v in ex["metadata"].items():
                ex[f"metadata_{k}"] = v
            del ex["metadata"]

        # flatten decompositions
        num_decompositions = len(ex["question"]["decompositions"])
        for i in range(num_decompositions):
            ex[f"decomposition_{i}"] = ex["question"]["decompositions"][i]

    # save
    pd.DataFrame(examples).to_csv(output_path)


def _populate_decompositions(
    example: Dict,
    prompt: str,
    dataset_reader: DatasetReader,
    evaluator: Evaluator,
    decomposition_cache_dir: str,
    num_decompositions: int,
) -> Dict:
    """ """
    example = dataset_reader.parse_example(example)
    model_accessor = LocalModelAccessor(model_path='/hpc_stor03/sjtu_home/haoxiang.jiang/models/Qwen2.5-1.5B-Instruct')
    question = QuestionV1Retrobust(
        question=example.question,
        prompt=prompt,
        num_decompositions=num_decompositions,
        model_accessor=model_accessor,
    )
    question.populate()

    gpt_answers = [
        s[-1].gpt_3_ans[:-1] if s[-1].gpt_3_ans[-1] == "." else s[-1].gpt_3_ans
        for s in question.decompsition_steps
        if len(s) > 0 and s[-1].gpt_3_ans is not None and len(s[-1].gpt_3_ans) > 0
    ]
    question_with_answer = QuestionWithAnswer(
        question=question, answers=None, gpt_answers=gpt_answers
    )
    question_with_answer_dict = dataclasses.asdict(question_with_answer)
    results = {}
    if len(question_with_answer_dict["gpt_answers"]) > 0:
        results["acc@1"] = evaluator.evaluate(
            question_with_answer_dict["gpt_answers"][0], example.gold_answer
        )
        results["acc@3"] = max(
            [
                evaluator.evaluate(ans, example.gold_answer)
                for ans in question_with_answer_dict["gpt_answers"]
            ]
        )
        majority_prediction = Counter(
            [y.lower() for y in question_with_answer_dict["gpt_answers"]]
        ).most_common(n=1)[0][0]
        majority_prediction_at_three = Counter(
            [y.lower() for y in question_with_answer_dict["gpt_answers"][:3]]
        ).most_common(n=1)[0][0]
        results["acc@majority"] = evaluator.evaluate(
            majority_prediction, example.gold_answer
        )
        results["acc@majority_3"] = evaluator.evaluate(
            majority_prediction_at_three, example.gold_answer
        )

    else:
        results["acc@1"] = False
        results["acc@3"] = False
        results["acc@majority"] = False
        results["acc@majority_3"] = False
    dataset_metadata_fields = Config().get("dataset.metadata_fields")
    results["metadata"] = {
        **{k: v for k, v in example.metadata.items() if k in dataset_metadata_fields},
        **{k: v for k, v in dataclasses.asdict(example).items() if k != "metadata"},
    }
    logger.info(results)
    question_with_answer_dict.update(results)

    filename = get_string_hash(example.question)
    with open(f"{decomposition_cache_dir}/{filename}.json", "w") as json_file:
        json.dump(question_with_answer_dict, json_file)
    return question_with_answer_dict


def _run_decompositions(
    examples: List[Dict],
    cache_dir: str,
    output_dir: str,
    experiment_unique_name: str,
    dataset: DatasetReader,
    prompt: str,
    evaluator: Evaluator,
    num_decompositions: int,
) -> List[Dict]:
    if cache_dir is None:
        decomposition_cache_dir = (
            f"{output_dir}/{experiment_unique_name}/decompositions"
        )
        os.makedirs(decomposition_cache_dir, exist_ok=True)
        logger.info(f"Saving decompositions in: {decomposition_cache_dir}")
        questions_with_decompositions = [
            _populate_decompositions(
                example,
                prompt,
                dataset,
                evaluator,
                decomposition_cache_dir,
                num_decompositions,
            )
            for example in tqdm(examples)
        ]
    else:
        logger.info(f"Reading decompositions from: {cache_dir}")
        cached_files = [f for f in os.listdir(cache_dir)]
        cached_decompositions = {}
        for filename in cached_files:
            with open(f"{cache_dir}/{filename}", "r") as f:
                data = json.load(f)
                cached_decompositions[filename] = data
        questions_with_decompositions = cached_decompositions.values()
    return questions_with_decompositions


def run_experiment(config_path: str):
    """ """
    # read config
    Config().load(config_path)

    # start experiment
    experiment_name = Config().get("experiment_name")
    logger.info(f"Starting experiment: {experiment_name}")
    output_dir = Config().get("output_dir")

    # datetime
    timestamp = datetime.now().timestamp()
    date_time = datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime("%d_%m_%Y_%H_%M_%S")
    experiment_unique_name = f"{experiment_name}_{str_date_time}"

    # read dataset
    dataset_name = Config().get("dataset.name")
    logger.info(f"Reading dataset: {dataset_name}.")
    dataset = DatasetReadersFactory().get_instance(dataset_name)
    dataset.read()
    examples = dataset.examples

    # get evaluator
    evaluator_name = Config().get("evaluator")
    logger.info(f"Using evaluator: {evaluator_name} to report metrics")
    evaluator = EvaluatorsFactory().get_instance(evaluator_name)

    # decomposition settings
    prompt_name = Config().get("decomposition.prompt")


    # Ensure prompt is correctly loaded
    prompt = PromptFactoryDict.get(prompt_name, "")
    if not prompt:
        logger.error(f"Prompt '{prompt_name}' not found in PromptFactoryDict!")
        raise ValueError(f"Prompt '{prompt_name}' not found!")

    num_decompositions = Config().get("decomposition.num_decompositions")

    # filter examples in prompts
    examples_not_in_prompts = [
        e
        for e in dataset.examples
        if not (dataset.parse_example(e).question.lower() in prompt.lower())
    ]
    num_examples, num_examples_not_in_prompts = len(examples), len(
        examples_not_in_prompts
    )
    num_examples_in_prompt = num_examples - num_examples_not_in_prompts
    logger.info(
        f"Removing {num_examples_in_prompt}/{num_examples} examples. Left with {num_examples_not_in_prompts} examples."
    )
    examples = examples_not_in_prompts

    # retries
    num_retries = Config().get("num_retries")
    logger.info(f"Running with {num_retries} retries.")
    examples_with_answers: List[Dict] = []
    examples_to_answer: List[Dict] = examples  # todo: TW - delete

    # populate question with decomposition
    cache_dir = Config().get("decomposition.cache_dir")

    res = {}
    settings = Config().get("decomposition.settings")

    # change config based on setting
    for i, setting in enumerate(settings):
        logger.info(f"At index: {i}, with setting: {setting}")
        experiment_unique_name_with_setting = f"{experiment_unique_name}_{setting}"
        if i == 0:
            if setting == "reg":
                run_retrieval_dir = Config().get("decomposition.run_output_dir")
                main_retrieval_dir = Config().get("decomposition.main_retriever_dir")
                logger.info(
                    f"run_retrieval_dir: {run_retrieval_dir}, main_retrieval_dir: {main_retrieval_dir}"
                )
            else:
                raise ValueError("Must start with reg and index 0.")
        
        if setting == "@5":
            at_5_settings = {
                "decomposition.randomize_retrieval": False,
                "decomposition.retrieve_at_5": True,
                "decomposition.run_output_dir": "/hpc_stor03/sjtu_home/haoxiang.jiang/RAG/cots/data/outputs/hotpot",
            }
            logger.info(f"Overriding settings to @5->: {at_5_settings}")
            Config().override_dict(at_5_settings)


        setting_results = _run_decompositions(
            examples=examples_to_answer,
            cache_dir=cache_dir,
            output_dir=output_dir,
            experiment_unique_name=experiment_unique_name_with_setting,
            dataset=dataset,
            prompt=prompt,  # Pass the prompt here
            evaluator=evaluator,
            num_decompositions=num_decompositions,
        )
        res[setting] = setting_results

        # save results
        if output_dir is not None:
            output_path = f"{output_dir}/{experiment_unique_name_with_setting}_{i}.csv"
            logger.info(f"Saving output path to: {output_path}")
            _save_examples(setting_results, output_path)

        # report results
        acc_at_one_strategy = np.average(
            [
                x["acc@1"]
                if len(x["gpt_answers"] )
                and x["gpt_answers"][0].lower() in {"yes", "no"}
                else 0.5
                for x in setting_results
            ]
        )
        acc_at_one = np.average([x["acc@1"] for x in setting_results])
        logger.info(
            {
                "setting": setting,
                "num_examples": len(setting_results),
                "acc@1": acc_at_one,
                "acc_at_one_strategy": acc_at_one_strategy,
            }
        )
    for setting, setting_results in res.items():
        logger.info(
            {
                "setting": setting,
                "num_examples": len(setting_results),
                "acc@1": np.average([x["acc@1"] for x in setting_results]),
                "acc_at_one_strategy": np.average(
                    [
                        x["acc@1"]
                        if len(x["gpt_answers"] )
                        and x["gpt_answers"][0].lower() in {"yes", "no"}
                        else 0.5
                        for x in setting_results
                    ]
                ),
            }
        )
    logger.info("finished")



if __name__ == "__main__":
    """ """
    args = parse_args()
    run_experiment(args.config_path)
