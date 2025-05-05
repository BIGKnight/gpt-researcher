import json
import argparse
import pandas as pd
from . import common
from .browsecomp_eval import BrowseCompEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.deepresearch_sampler import DeepResearchSampler
import asyncio
async def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )

    args = parser.parse_args()


    sampler = DeepResearchSampler()
    grading_sampler = ChatCompletionSampler(model="o4-mini")
    num_examples = (args.examples if args.examples is not None else (5 if args.debug else None))
    eval_obj = BrowseCompEval(
        grader_model=grading_sampler,
        num_examples=10 if args.debug else num_examples,
    )
    debug_suffix = "_DEBUG" if args.debug else ""
    mergekey2resultpath = {}
            
    result = await eval_obj(sampler)
    # ^^^ how to use a sampler
    file_stem = f"BrowseComp_DeepResearch"
    report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
    print(f"Writing report to {report_filename}")
    with open(report_filename, "w") as fh:
        fh.write(common.make_report(result))
    metrics = result.metrics | {"score": result.score}
    print(metrics)
    result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
    with open(result_filename, "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print(f"Writing results to {result_filename}")
    mergekey2resultpath[f"{file_stem}"] = result_filename
    
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    asyncio.run(main())
