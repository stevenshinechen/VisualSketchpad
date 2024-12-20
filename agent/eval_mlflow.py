from main import run_agent
import asyncio
from tqdm.asyncio import tqdm
import mlflow
from enum import StrEnum
from mlflow_utils import with_mlflow_server
from config import llm_config
import os
from uuid import uuid4
import json

class IsobenchTask(StrEnum):
    CHEMISTRY = "chemistry"
    GRAPH_CONNECTIVITY = "graph_connectivity"
    GRAPH_ISOMORPHISM = "graph_isomorphism"
    GRAPH_MAXFLOW = "graph_maxflow"
    MATH_BREAKPOINT = "math_breakpoint"
    MATH_CONVEXITY = "math_convexity"
    MATH_PARITY = "math_parity"
    PHYSICS = "physics"
    PUZZLE = "puzzle"
    WINNER_ID = "winner_id"

def run_agent_on_isobench(task: IsobenchTask, task_id: int):
  return run_agent(f"../tasks/{task.value}/{task_id}", f"../outputs/{task.value}", task_type="math", task_name=task.value)

async def arun_agent_on_isobench(task: IsobenchTask, task_id: int):
  loop = asyncio.get_event_loop()
  return await loop.run_in_executor(None, run_agent_on_isobench, task, task_id)

async def run_agent_with_mlflow_on_isobench(task: IsobenchTask, task_id: int):
  all_messages, usage_summary = await arun_agent_on_isobench(task, task_id)
  answer_message = all_messages[-1]["content"][0]["text"]
  answer_message = answer_message.split("ANSWER:")[1].split("TERMINATE")[0]
  mlflow.log_text(answer_message, f"{task_id}/prediction.txt")
  mlflow.log_dict(all_messages, f"{task_id}/output.json")
  mlflow.log_dict(usage_summary, f"{task_id}/usage_summary.json")

  task_json = read_isobench_task(task, task_id)
  mlflow.log_text(task_json["label"], f"{task_id}/label.txt")
  # TODO: Make more robust correctness check
  correct = str(task_json["label"]) in answer_message
  mlflow.log_text(str(correct), f"{task_id}/correct.txt")
  return correct

def read_isobench_task(task: IsobenchTask, task_id: int):
  with open(f"../tasks/{task.value}/{task_id}/example.json") as f:
    return json.load(f)

async def run_all_on_isobench(task: IsobenchTask):
  task_ids = [int(f) for f in os.listdir(f"../tasks/{task.value}") if os.path.isdir(f"../tasks/{task.value}/{f}")]
  task_ids = task_ids[:2]
  results = await tqdm.gather(*[run_agent_with_mlflow_on_isobench(task, task_id) for task_id in task_ids])
  if results:
    mlflow.log_metric("accuracy", sum(results) / len(results))

def run_evaluation(task: IsobenchTask):
    mlflow.set_experiment(f"IsoBench: {task}")
    with mlflow.start_run(run_name=f"visual_sketchpad_{uuid4()}") as run:
        mlflow.log_param("task", task)
        llm_str = llm_config["config_list"][0]["model"]
        mlflow.log_param("llm_str", llm_str)
        mlflow.set_tag("tool", "visual_sketchpad")
        asyncio.run(run_all_on_isobench(task))

if __name__ == "__main__":
    with with_mlflow_server():
      run_evaluation(IsobenchTask.GRAPH_MAXFLOW)
    