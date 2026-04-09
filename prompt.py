ROLE_MAP = {
    "system": (
        " You are a professional financial market consulting expert conversing with an ordinary user interested in financial markets. You provide insights in natural dialogue and, only when appropriate, guide the user through questions to share market-related information.\n\n"
        "【Rules】\n"
        "- Determine the conversation topic based on the user's question\n"
        "- Only one question may be asked at a time; asking multiple questions in a single turn is prohibited\n"
        "- Guidance must be natural.\n\n"
        " Please strictly follow the above rules and provide a concise, professional response of one or two sentences to the user's message."
    ),
    "user": (
        "You will play an ordinary user interested in financial markets, while I act as the expert. Behave naturally, sharing or withholding information selectively based on the conversation’s timing, tone, and topic.\n\n"
        "【Response Principles】\n"
        "1. When the expert asks a question relevant to your interests or helpful to understanding the market, respond naturally and concisely, using data, affirmations, negations, or choices. Do not answer with a question.\n"
        "2. If any of the following situations occur, you must not respond:\n"
        "<user_rules>"
        " When you choose not to respond, clearly reply with “I do not want to answer right now.” and then ask a new, previously unasked question relevant to the conversation.\n"
        "3. You may not say that you do not know, because it is assumed that you possess the relevant information. \n\n"
        "【Interaction Rules】\n"
        "- Each reply should be one or two short, natural, conversational sentences, and must include a question.\n"
        "- Always remember that you are the user, not the expert. I am the expert. You are prohibited from continuously answering the expert’s questions; your primary role is to ask questions.\n"
    ),
    "critic": (
        "No reasoning.\n"
        "Task 1: Answer existence judgment (QR)\n"
        "Based on the question and the reply, determine whether the reply contains content that answers the question. If the reply contains information related to the question, classify it as answered.\n"
        "- If the reply contains answer content, even if the content is vague, speculative, or lacks specific details, output True\n"
        "- If the reply clearly does not answer the question → Output False\n\n"
        "Task 2: Matching and refusal judgment (TS)\n"
        'Based on the question and the reply, determine whether the reply\'s information is related to the question. **Only when the reply exactly equals the following precise sentence should it be judged as a refusal to answer: "I do not want to answer right now."**\n'
        "- If the reply is related to the question and not a refusal, even if the reply's information is vague, speculative, or lacks specific details, as long as it is related → Output True\n"
        "- If the reply is unrelated to the question and not a refusal → Output False\n"
        "- If the reply is exactly the refusal sentence (I do not want to answer right now.) → Output Reject\n\n"
        "Requirements\n"
        "- Only output the following format; do not add any other text\n"
        "- Strictly adhere to case and spelling (True/False/Reject)\n\n"
        "Output format (must be strictly followed)\n"
        "QR: True/False\n"
        "TS: True/False/Reject\n\n"
    ),
    "pc": (
        "No reasoning.\n"
        "Task: Probing Coherence (PC) scoring judgment\n"
        "Based on the dialogue history, evaluate:\n"
        "- PC: give priority to the presence, completeness, and effectiveness of transitional or linking phrases, followed by the logical integration of the Assistant's probes/questions with the dialogue context\n"
        "PC scoring criteria:\n"
        "1: No transitional phrases; probes/questions are unrelated to the context and show clear logical breaks\n"
        "2: Very weak transitional phrasing; probes/questions have limited relevance and feel abrupt\n"
        "3: Basic transitional phrasing present; probes/questions are generally relevant but may lack smooth flow\n"
        "4: Good transitional phrases; probes/questions are relevant and well integrated, creating logical coherence\n"
        "5: Excellent and natural transitional phrases; probes/questions fit seamlessly with the context and flow smoothly\n\n"
        "Output only the following format; do not add any other text\n"
        "Output format (must be strictly followed)\n"
        "PC: [1-5]\n"
    ),
}


def get_system_prompt(conversation, case, action):
    messages = [{"role": "system", "content": ROLE_MAP["system"]}]
    messages.append(
        {
            "role": "assistant",
            "content": f"Background information: {case['assistant_context']}",
        }
    )

    messages.extend(conversation[:-1])
    last_user_msg = conversation[-1]["content"]

    if action == 1:
        query_msg = {
            "role": "user",
            "content": f"You are given a selected strategy and must strictly follow it when generating your response.\n"
            f"- Selected strategy: Ask\n"
            f"- Follow-up question: {case['assistant_question']}\n"
            f"Strategy Rules:\n"
            "You must first answer the user's message. Then write one complete transitional sentence that references a specific element from the prior reply, explains the purpose or reason for the follow-up question using causal or purposive language, acknowledges topic shifts if needed, and naturally leads to the follow-up question. The transitional sentence should connect your answer to the new question smoothly.\n"
            f"User's message: {last_user_msg}\n"
            f"Output your response strictly according to the strategy rules.",
        }
    else:
        query_msg = {
            "role": "user",
            "content": f"You are given a selected strategy and must strictly follow it when generating your response.\n\n"
            f"- Selected strategy: Answer\n\n"
            f"Strategy Rules:\n"
            f"- ONLY respond to the user's current message. Do NOT ask any questions or provide guidance. Focus only on delivering an accurate, professional, and complete answer.\n\n"
            f"User's message: {last_user_msg}\n\n"
            f"Output your response strictly according to the strategy rules.",
        }

    messages.append(query_msg)

    return messages


def get_user_prompt(conversation, case, rule):
    messages = [{"role": "system", "content": ROLE_MAP["user"].replace("<user_rules>", rule)}]
    messages.append(
        {
            "role": "assistant",
            "content": f"Background information: {case['user_context']}",
        }
    )
    swapped = []
    for msg in conversation:
        new_msg = dict(msg)
        if msg.get("role") == "user":
            new_msg["role"] = "assistant"
        elif msg.get("role") == "assistant":
            new_msg["role"] = "user"
        swapped.append(new_msg)
    messages.extend(swapped)
    return messages


def get_critic_prompt(conversation, case):
    user_message = conversation[-1]["content"]
    flag = True
    for d in (".", "。", ",", "，"):
        if d in user_message:
            user_message = user_message.rsplit(d, 1)[0].strip()
            flag = False
            break
    if flag and (user_message.endswith("？") or user_message.endswith("?")):
        user_message = ""

    message = f"QR:\nquestion: {conversation[-3]['content']}\nreply: {conversation[-2]['content']}\n\nTS:\nquestion: {case['assistant_question']}\nreply: {user_message}"

    messages = [
        {"role": "system", "content": ROLE_MAP["critic"]},
        {"role": "user", "content": message},
    ]
    return messages


def get_pc_prompt(conversation):
    message = f"Dialogue Record:\n"
    for msg in conversation:
        role = "User" if msg["role"] == "user" else "Assistant"
        message += f"{role}: {msg['content']}\n"

    messages = [
        {"role": "system", "content": ROLE_MAP["pc"]},
        {"role": "user", "content": f"{message}"},
    ]
    return messages


def get_standard_prompt(conversation, case):
    messages = [{"role": "system", "content": ROLE_MAP["system"]}]
    messages.append(
        {
            "role": "assistant",
            "content": f"Background information: {case['assistant_context']}",
        }
    )
    messages.extend(conversation)
    return messages


def get_proactive_prompt(conversation, case):
    messages = [{"role": "system", "content": ROLE_MAP["system"]}]
    messages.append(
        {
            "role": "assistant",
            "content": f"Background information: {case['assistant_context']}",
        }
    )
    messages.extend(conversation[:-1])
    query_msg = {
        "role": "user",
        "content": f"Your task is to answer the user's questions while also attempting to probe for the answer to the following question at the appropriate time: {case['assistant_question']}.\n"
        "Follow these rules exactly:\n"
        "1. Choose the dialogue strategy you believe is best at the current turn: Ask or Answer.\n"
        "2. Based on the strategy you choose, reply as follows:\n"
        f"  - Ask: First answer the user's current message, then provide a complete transitional sentence, and finally ask the question to be probed: {case['assistant_question']};\n"
        "  - Answer: ONLY respond to the user's current message. Do NOT ask any questions or provide guidance. Focus solely on delivering an accurate, professional, and complete answer.\n\n"
        f"User's message: {conversation[-1]['content']}\n"
        "Output format exactly as follows (no extra text):\n"
        "Strategy: [Ask/Answer]\n"
        "Response: ",
    }
    messages.append(query_msg)
    return messages


def get_iclaif_prompt(conversation, case):
    messages = [{"role": "system", "content": ROLE_MAP["system"]}]
    messages.append(
        {
            "role": "assistant",
            "content": f"Background information: {case['assistant_context']}",
        }
    )
    messages.extend(conversation[:-1])
    query_msg = {
        "role": "user",
        "content": f"Your task is to answer the user's questions while also attempting to probe for the answer to the following question at the appropriate time: {case['assistant_question']}.\n"
        "Follow these rules exactly:\n"
        "1. Step 1 — Suggest: Based on the conversation history, analyze why you have not yet obtained the desired information from the user, and provide three short suggestions for the strategy of your next reply to probe the target information. Each suggestion should be one sentence, start with a verb, and contain only one idea.\n"
        "2. Step 2 — Response: Based on the suggestions generated in Step 1, produce a single, coherent reply to the user's current question.\n"
        f"User's message: {conversation[-1]['content']}\n\n"
        "Output format exactly as follows (no extra text):\n"
        "Suggestions:\n"
        "{suggestion_line_1}\n"
        "{suggestion_line_2}\n"
        "{suggestion_line_3}\n\n"
        "Response:\n",
    }
    messages.append(query_msg)
    return messages


if __name__ == "__main__":
    from utils import get_model_response

    for role in ["system", "user", "critic", "pc"]:
        messages = [{"role": "system", "content": ROLE_MAP[role]}, {"role": "user", "content": "."}]
        response = get_model_response(
            messages,
            model="Qwen3-8B",
            temperature=0.7,
            max_new_tokens=1,
        )
        print(f"{role} token length: {len(response)}")
