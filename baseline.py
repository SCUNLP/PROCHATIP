import typer
import os
import sys
import random
from tqdm import tqdm
import json
from utils import set_random_seed, get_model_response
from prompt import get_critic_prompt, get_user_prompt, get_standard_prompt, get_pc_prompt, get_proactive_prompt, get_iclaif_prompt

app = typer.Typer()
log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)


def run_episode(case, rule, max_turns=8, algorithm=None, user_llm=None, system_llm=None, critic_llm=None):
    conversation = []
    conversation.append({"role": "user", "content": case["user_question"]})
    print("User_Rule: \n", rule)
    print("User_Question: ", case["user_question"])
    print("Assistant_Goal: ", case["assistant_question"])
    done = -1
    reject = 0
    reply = 0
    ask = 0

    for turn in range(max_turns):
        print("-" * 20 + f"Turn {turn + 1}" + "-" * 20)

        if algorithm == "standard":
            assistant_response = get_model_response(get_standard_prompt(conversation, case), temperature=0, model=system_llm)
        elif algorithm == "proactive":
            assistant_response = get_model_response(get_proactive_prompt(conversation, case), temperature=0, model=system_llm)
            try:
                strategy = assistant_response.split("Strategy:")[1].split("\n")[0].strip()
                print(strategy)
                if strategy == "Ask":
                    ask += 1
                assistant_response = assistant_response.split("Response:")[1].strip()
            except Exception:
                pass
        elif algorithm == "iclaif":
            assistant_response = get_model_response(get_iclaif_prompt(conversation, case), temperature=0, model=system_llm, max_new_tokens=400)
            suggestions = assistant_response.split("Suggestions:")[1].split("Response:")[0].strip()
            print("Suggestions:\n", suggestions)
            assistant_response = assistant_response.split("Response:")[1].strip()
            if assistant_response.endswith("?"):
                ask += 1

        conversation.append({"role": "assistant", "content": assistant_response})
        print("Assistant: ", assistant_response)

        messages = get_user_prompt(conversation, case, rule)
        if algorithm == "standard":
            messages.pop(1)
        user_response = get_model_response(messages, temperature=0, model=user_llm)
        conversation.append({"role": "user", "content": user_response})
        print("User: ", user_response)

        critic_response = get_model_response(get_critic_prompt(conversation, case), temperature=0, model=critic_llm, max_new_tokens=None)
        print(critic_response)
        qr = critic_response.split("QR: ")[1][:4]
        ts = critic_response.split("TS: ")[1].strip()

        if ts == "Reject":
            reject += 1
        if qr == "True":
            reply += 1
        if ts == "True":
            done = turn
            break

    pc_text = get_model_response(get_pc_prompt(conversation), temperature=0, model=critic_llm, max_new_tokens=None)
    pc = int(pc_text.split("PC: ")[1][0])
    print(pc_text)

    return {"done": done, "reject": reject, "reply": reply, "pc": pc, "ask": ask}


@app.command()
def eval(
    dataset: str = "finqa",
    algorithm: str = "standard",
    user_llm: str = "Qwen",
    system_llm: str = "Qwen",
    critic_llm: str = "Qwen",
    max_turns: int = 8,
    pr: bool = False,
    checkpoint: str = "eval_checkpoint",
    save_every: int = 1,
):
    set_random_seed(42)
    _log_path = os.path.join(log_dir, checkpoint + ".log")
    checkpoint = os.path.join("checkpoints", checkpoint + ".json")
    if not pr:
        _log_file = open(_log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = _log_file
    print(f"Using checkpoint: {checkpoint}")
    print(f"Loading dataset {dataset} ...")
    datasets = json.load(open(f"data/{dataset}/{dataset}_eval.json", "r", encoding="utf-8"))
    num_samples = len(datasets)
    print(f"Number of samples: {num_samples}")

    rules = json.load(open("data/rules.json", "r", encoding="utf-8"))["eval"]

    total_turns = {"low": [], "medium": [], "high": []}
    total_rejects = {"low": [], "medium": [], "high": []}
    total_replys = {"low": [], "medium": [], "high": []}
    total_pcs = {"low": [], "medium": [], "high": []}
    total_asks = {"low": [], "medium": [], "high": []}
    successes = {"low": 0, "medium": 0, "high": 0}

    processed = set()
    checkpoint_data = []
    if os.path.exists(checkpoint):
        try:
            checkpoint_data = json.load(open(checkpoint, "r", encoding="utf-8"))
            for rec in checkpoint_data:
                idx = rec.get("i")
                if idx is None:
                    continue
                processed.add(idx)
                rel = rec["relation"]
                total_turns[rel].append(rec["turns"])
                total_rejects[rel].append(rec["reject"])
                total_replys[rel].append(rec["reply"])
                total_pcs[rel].append(rec["pc"])
                total_asks[rel].append(rec["ask"])
                if rec.get("success"):
                    successes[rel] += 1
            print(f"Loaded checkpoint with {len(processed)} processed samples, will resume from next unprocessed.")
        except Exception as e:
            print("Failed to load checkpoint:", e)
            checkpoint_data = []
            processed = set()

    for i in tqdm(range(num_samples), desc="Evaluating..."):
        if i in processed:
            continue
        case = datasets[i]
        print("\n" + "=" * 20 + f"Evaluation Episode {i + 1}" + "=" * 20)
        res = run_episode(
            case,
            random.choice(rules),
            max_turns=max_turns,
            algorithm=algorithm,
            user_llm=user_llm,
            system_llm=system_llm,
            critic_llm=critic_llm,
        )
        done = res["done"]
        reject = res["reject"]
        reply = res["reply"]
        pc = res["pc"]
        ask = res["ask"]

        turns = done + 1 if done != -1 else max_turns
        rel = case["relation"]
        total_turns[rel].append(turns)
        total_rejects[rel].append(reject)
        total_replys[rel].append(reply)
        total_pcs[rel].append(pc)
        total_asks[rel].append(ask)
        success_flag = done != -1
        if success_flag:
            successes[rel] += 1

        rec = {"i": i, "relation": rel, "turns": turns, "reject": reject, "reply": reply, "pc": pc, "success": success_flag, "ask": ask}
        checkpoint_data.append(rec)
        processed.add(i)
        if not pr and ((len(checkpoint_data) % save_every) == 0 or len(processed) == num_samples):
            tmp_path = checkpoint + ".tmp"
            try:
                with open(tmp_path, "w", encoding="utf-8") as fw:
                    json.dump(checkpoint_data, fw, ensure_ascii=False, indent=2)
                os.replace(tmp_path, checkpoint)
            except Exception as e:
                print("Failed to save checkpoint:", e)

    print(f"[EVAL] Algorithm: Standard Evaluation on {dataset} dataset")
    for level in ["low", "medium", "high"]:
        if len(total_turns[level]) == 0:
            continue
        avg_turns = float(sum(total_turns[level]) / len(total_turns[level]))
        success_rate = float(successes[level] / len(total_turns[level]))
        reject_rate = float(sum(total_rejects[level]) / sum(total_asks[level])) if sum(total_asks[level]) > 0 else 0.0
        reply_rate = float(sum(total_replys[level]) / sum(total_turns[level]))
        avg_pc = float(sum(total_pcs[level]) / len(total_pcs[level]))
        
        print(f"[EVAL-{level}] TSR={success_rate * 100:.2f}%, AvgT={avg_turns:.2f}, RPR={reject_rate * 100:.2f}%, QRR={reply_rate * 100:.2f}%, PC={avg_pc:.2f}")

    all_turns = sum(total_turns["low"] + total_turns["medium"] + total_turns["high"])
    all_asks = sum(total_asks["low"] + total_asks["medium"] + total_asks["high"])
    overall_avg_turns = float(sum(total_turns["low"] + total_turns["medium"] + total_turns["high"]) / num_samples)
    overall_success_rate = float((successes["low"] + successes["medium"] + successes["high"]) / num_samples)
    overall_reject_rate = float((sum(total_rejects["low"]) + sum(total_rejects["medium"]) + sum(total_rejects["high"])) / all_asks) if all_asks > 0 else 0.0
    overall_reply_rate = float((sum(total_replys["low"]) + sum(total_replys["medium"]) + sum(total_replys["high"])) / all_turns)
    overall_avg_pc = float((sum(total_pcs["low"]) + sum(total_pcs["medium"]) + sum(total_pcs["high"])) / num_samples)

    print(f"[EVAL] TSR={overall_success_rate * 100:.2f}%, AvgT={overall_avg_turns:.2f}, RPR={overall_reject_rate * 100:.2f}%, QRR={overall_reply_rate * 100:.2f}%, PC={overall_avg_pc:.2f}")


if __name__ == "__main__":
    app()
