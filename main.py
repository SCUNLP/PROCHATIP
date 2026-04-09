import typer
import torch
import os
import json
import random
from tqdm import tqdm
from torch.optim import AdamW
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import (
    get_model_response,
    set_random_seed,
    con_to_text,
)
from prompt import get_system_prompt, get_user_prompt, get_critic_prompt, get_pc_prompt
import sys
import time

app = typer.Typer()
log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)

def run_episode(policy_model, tokenizer, device, case, rule, max_turns=8, train=True, user_llm=None, system_llm=None, critic_llm=None):
    conversation = []
    log_probs = []
    rewards = []
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
        text_messages = con_to_text(conversation,case['assistant_question'])
        # text_messages = con_to_text(conversation)

        inputs = tokenizer(text_messages, return_tensors="pt").to(device)
        outputs = policy_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

        if train:
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)
            print(m.probs)
        else:
            action = torch.argmax(probs, dim=-1).squeeze()

        print(f"{'Ask' if action.item() else 'Answer'}")
        if action.item() == 1:
            ask += 1

        if train:
            expert_response = get_model_response(get_system_prompt(conversation, case, action.item()), temperature=0.6, llm_api=system_llm)
        else:
            expert_response = get_model_response(get_system_prompt(conversation, case, action.item()), llm_api=system_llm)
        conversation.append({"role": "assistant", "content": expert_response})
        print("Assistant: ", expert_response)

        if train:
            user_response = get_model_response(get_user_prompt(conversation, case, rule), temperature=0.6, llm_api=user_llm)
        else:
            user_response = get_model_response(get_user_prompt(conversation, case, rule), llm_api=user_llm)
        conversation.append({"role": "user", "content": user_response})
        print("User: ", user_response)

        critic_response = get_model_response(get_critic_prompt(conversation, case), llm_api=critic_llm, max_new_tokens=2000)
        print(critic_response)
        qr = critic_response.split("QR: ")[1][:4]
        ts = critic_response.split("TS: ")[1].strip()

        if ts == "Reject":
            reject += 1
        if qr == "True":
            reply += 1

        if action.item() == 1:
            if ts == "True":
                reward = 2.0
            elif ts == "Reject":
                reward = -1.0
            else:
                reward = -0.2
        else:
            if rewards and rewards[-1] == -1.0:
                reward = 1.5
            elif rewards and rewards[-1] == -0.2:
                reward = 0.3
            elif rewards and rewards[-1] == 0.0:
                reward = -2.0
            else:
                reward = 0.0
        if train:
            print(f"Reward: {reward}\n")
            rewards.append(reward)

        if reward == 2.0 and train:
            done = turn
            break
        elif ts == "True" and not train:
            done = turn
            break
    if train:
        pc = 0
    else:
        pc_text = get_model_response(get_pc_prompt(conversation), llm_api=critic_llm, max_new_tokens=2000)
        pc = int(pc_text.split("PC: ")[1][0])
        print(pc_text)

    return {"rewards": rewards, "log_probs": log_probs, "done": done, "reject": reject, "reply": reply, "pc": pc, "ask_count": ask}


@app.command()
def train(
    sft_model_path: str = "../model/Qwen3-0.6B",
    dataset: str = "finqa",
    output_dir: str = "model/rl_output",
    user_llm: str = "Qwen",
    system_llm: str = "Qwen",
    critic_llm: str = "Qwen",
    gpu: int = 7,
    episodes: int = 500,
    lr: float = 1e-5,
    gamma: float = 0.99,
    max_turns: int = 8,
    pr: bool = False,
):
    set_random_seed(42)
    device = torch.device(f"cuda:{gpu}" if not gpu == -1 else "cpu")
    output_dir = os.path.join(output_dir, dataset)
    if not pr:
        _log_path = os.path.join(log_dir, f"rltraining_{time.strftime('%Y%m%d_%H%M%S')}.log")
        _log_file = open(_log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = _log_file

    print(f"Loading policy model from {sft_model_path}...")
    policy_model = AutoModelForSequenceClassification.from_pretrained(sft_model_path, num_labels=2, trust_remote_code=True, device_map=f"cuda:{gpu}" if not gpu == -1 else "cpu")
    policy_model.train()
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True, device_map=f"cuda:{gpu}" if not gpu == -1 else "cpu")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    optimizer = AdamW(policy_model.parameters(), lr=lr)

    train_dataset = json.load(open(f"data/{dataset}/{dataset}_train.json", "r", encoding="utf-8"))
    valid_dataset = json.load(open(f"data/{dataset}/{dataset}_valid.json", "r", encoding="utf-8"))
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}")

    rules = json.load(open("data/rules.json", "r", encoding="utf-8"))["train"]

    for episode in tqdm(range(episodes), desc="Training..."):
        print("\n" + "=" * 20 + f"Episodes {episode + 1}" + "=" * 20)
        case = train_dataset[episode % len(train_dataset)]

        res = run_episode(
            policy_model,
            tokenizer,
            device,
            case,
            random.choice(rules),
            max_turns=max_turns,
            train=True,
            user_llm=user_llm,
            system_llm=system_llm,
            critic_llm=critic_llm,
        )
        rewards = res["rewards"]
        log_probs = torch.stack(res["log_probs"]) if res["log_probs"] else torch.tensor([]).to(device)
        done = res["done"]

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns).to(device)

        if len(returns) > 1:
            returns = (returns) / (returns.std() + 1e-8)

        policy_loss = [-lp * R for lp, R in zip(log_probs, returns)]

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() if policy_loss else torch.tensor(0.0, requires_grad=True).to(device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"Episode {episode}: loss={loss.item() if hasattr(loss, 'item') else 0.0:.4f}, total_reward={sum(rewards)}, turns={done + 1 if done != -1 else max_turns}, rejects={res['reject']}, replies={res['reply']}, pc={res['pc']}, ask_count={res['ask']}")
        if (episode + 1) % (episodes // 5) == 0:
            valid(
                policy_model,
                tokenizer,
                device,
                valid_dataset[:10],
                max_turns=max_turns,
                user_llm=user_llm,
                system_llm=system_llm,
                critic_llm=critic_llm,
            )
            save_model(policy_model, tokenizer, output_dir + f"/{episode + 1}")


def valid(policy_model, tokenizer, device, valid_dataset, max_turns=8, user_llm=None, system_llm=None, critic_llm=None):
    policy_model.eval()
    total_turns = []
    total_rejects = []
    total_replys = []
    total_pcs = []
    total_asks = []
    successes = 0
    num_samples = len(valid_dataset)
    rules = []
    with open("data/rules.json", "r", encoding="utf-8") as f:
        rules = json.load(f)["train"]

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating..."):
            case = valid_dataset[i]
            print("\n" + "=" * 20 + f"Evaluation Episode {i + 1}" + "=" * 20)
            res = run_episode(
                policy_model,
                tokenizer,
                device,
                case,
                random.choice(rules),
                max_turns=max_turns,
                train=False,
                user_llm=user_llm,
                system_llm=system_llm,
                critic_llm=critic_llm,
            )
            done = res["done"]
            reject = res["reject"]
            reply = res["reply"]
            pc = res["pc"]
            ask = res["ask"]
        
            total_rejects.append(reject)
            total_replys.append(reply)
            total_pcs.append(pc)
            total_turns.append(done + 1 if done != -1 else max_turns)
            total_asks.append(ask)
            if done != -1:
                successes += 1

    avg_turns = float(sum(total_turns) / num_samples)
    success_rate = float(successes / num_samples)
    reject_rate = float(sum(total_rejects) / sum(total_asks))
    reply_rate = float(sum(total_replys) / sum(total_turns))
    avg_pc = float(sum(total_pcs) / num_samples)

    print(f"[EVAL] avg_turns={avg_turns:.4f}, success_rate={success_rate:.4f}, reject_rate={reject_rate:.4f}, reply_rate={reply_rate:.4f}, avg_pc={avg_pc:.4f}")
    return success_rate, avg_turns, reject_rate


def save_model(policy_model, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


@app.command()
def eval(
    cs_model_path: str = "model/sft_rl_output/finqa/100",
    dataset: str = "finqa",
    user_llm: str = "Qwen",
    system_llm: str = "Qwen",
    critic_llm: str = "Qwen",
    gpu: int = 7,
    max_turns: int = 8,
    pr: bool = False,
    checkpoint: str = "eval_checkpoint",
    save_every: int = 1,
):
    set_random_seed(42)
    _log_path = os.path.join(log_dir, checkpoint + ".log")
    checkpoint = os.path.join("checkpoints", checkpoint + ".json")
    device = torch.device(f"cuda:{gpu}" if not gpu == -1 else "cpu")
    if not pr:
        _log_file = open(_log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = _log_file

    print(f"Loading policy model from {cs_model_path}...")
    policy_model = AutoModelForSequenceClassification.from_pretrained(cs_model_path, num_labels=2, trust_remote_code=True, device_map=f"cuda:{gpu}" if not gpu == -1 else "cpu")
    policy_model.eval()
    try:
        policy_model.to(device)
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(cs_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading evaluation dataset for {dataset}...")
    datasets = json.load(open(f"data/{dataset}/{dataset}_eval.json", "r", encoding="utf-8"))
    num_samples = len(datasets)
    print(f"Number of evaluation samples: {num_samples}")

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

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating..."):
            if i in processed:
                continue
            case = datasets[i]
            print("\n" + "=" * 20 + f"Evaluation Episode {i + 1}" + "=" * 20)
            res = run_episode(
                policy_model,
                tokenizer,
                device,
                case,
                random.choice(rules),
                max_turns=max_turns,
                train=False,
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

    for level in ["low", "medium", "high"]:
        if len(total_turns[level]) == 0:
            continue
        avg_turns = float(sum(total_turns[level]) / len(total_turns[level]))
        success_rate = float(successes[level] / len(total_turns[level]))
        reject_rate = float(sum(total_rejects[level]) / sum(total_asks[level]))
        reply_rate = float(sum(total_replys[level]) / sum(total_turns[level]))
        avg_pc = float(sum(total_pcs[level]) / len(total_pcs[level]))

        print(f"[EVAL-{level}] TSR={success_rate * 100:.2f}%, AvgT={avg_turns:.2f}, RPR={reject_rate * 100:.2f}%, QRR={reply_rate * 100:.2f}%, PC={avg_pc:.2f}")
        
    all_turns = sum(total_turns["low"] + total_turns["medium"] + total_turns["high"])
    all_asks = sum(total_asks["low"] + total_asks["medium"] + total_asks["high"])
    overall_avg_turns = float(sum(total_turns["low"] + total_turns["medium"] + total_turns["high"]) / num_samples) if num_samples > 0 else 0.0
    overall_success_rate = float((successes["low"] + successes["medium"] + successes["high"]) / num_samples) if num_samples > 0 else 0.0
    overall_reject_rate = float((sum(total_rejects["low"]) + sum(total_rejects["medium"]) + sum(total_rejects["high"])) / all_asks) if all_asks > 0 else 0.0
    overall_reply_rate = float((sum(total_replys["low"]) + sum(total_replys["medium"]) + sum(total_replys["high"])) / all_turns) if all_turns > 0 else 0.0
    overall_avg_pc = float((sum(total_pcs["low"]) + sum(total_pcs["medium"]) + sum(total_pcs["high"])) / num_samples) if num_samples > 0 else 0.0

    print(f"[EVAL] TSR={overall_success_rate * 100:.2f}%, AvgT={overall_avg_turns:.2f}, RPR={overall_reject_rate * 100:.2f}%, QRR={overall_reply_rate * 100:.2f}%, PC={overall_avg_pc:.2f}")


if __name__ == "__main__":
    app()
