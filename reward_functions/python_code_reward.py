"""
Python代码RL训练的Reward函数
集成DeepSeek API进行答案正确性判断
"""

import re
import os
from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://10.250.2.24:8090/run_code")

_client = None


def get_llm_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY)
    return _client


def get_llm_output(prompt: str, model: str = "deepseek-chat") -> str:
    client = get_llm_client()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=messages,
        stream=False,
    )
    output = completion.choices[0].message.content
    return output


def extract_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def is_valid_sequence(content: str) -> tuple:
    tags_to_check = ["think", "code", "observation", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    split_pattern = r"(</?(?:think|code|observation|answer)>)"
    parts = re.split(split_pattern, content)

    state = "start"

    for i, part in enumerate(parts):
        if not part.strip():
            continue

        if re.match(r"</?(?:think|code|observation|answer)>", part):
            if part == "lessons" and state in ["start", "observation"]:
                state = "in_think"
            elif part == "learnt" and state == "in_think":
                state = "after_think"
            elif part == "<code>" and state == "after_think":
                state = "in_code"
            elif part == "</code>" and state == "in_code":
                state = "after_code"
            elif part == "<observation>" and state == "after_code":
                state = "in_observation"
            elif part == "</observation>" and state == "in_observation":
                state = "observation"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            if state in ["in_think", "in_code", "in_observation", "in_answer"]:
                pass
            elif state in ["start", "after_think", "after_code", "observation"]:
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def answer_reward(user_message: str, answer: str) -> float:
    prompt = """## 任务目标
根据执行过程判断是否成功解决问题

## 任务要求
- 认真审视执行过程，做出正确的判断
- 只输出是或否，不要输出多余内容

## 问题
{}

## 执行过程
{}""".format(user_message, answer)

    try:
        result = get_llm_output(prompt)
        result = result.strip().lower()
        if "是" in result or "yes" in result:
            return 1.0
        else:
            return -1.0
    except Exception as e:
        print(f"LLM API调用失败: {e}")
        return 0.0


def exec_code(code: str) -> tuple:
    import requests

    headers = {"Content-Type": "application/json"}
    data = {"code": code, "language": "python"}

    try:
        response = requests.post(SANDBOX_URL, json=data, headers=headers, timeout=30)
        stdout = response.json()["run_result"]["stdout"]
        stderr = response.json()["run_result"]["stderr"]
        return stdout[:1000], stderr[:1000]
    except Exception as e:
        print(f"Sandbox执行失败: {e}")
        return "", str(e)


def extract_code(text: str) -> list:
    code_block_pattern = re.compile(r"<code>(.*?)</code>", re.DOTALL)
    code_blocks = code_block_pattern.findall(text)
    return code_blocks if code_blocks else []


def code_result(solution_str: str) -> tuple:
    code_blocks = extract_code(solution_str)
    if not code_blocks:
        return "", ""
    code = code_blocks[-1]
    stdout, stderr = exec_code(code)
    return stdout, stderr


def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float:
    is_valid, _ = is_valid_sequence(solution_str)

    if is_valid:
        score = 0.5
        stdout, stderr = code_result(solution_str)

        if "error" in stderr.lower() or "traceback" in stderr.lower():
            score -= 0.5
        else:
            score += 0.5
            if extra_info and "user_message" in extra_info:
                user_message = extra_info["user_message"]
                score += answer_reward(user_message, solution_str)

        print("+++++++++++++++++++++++++++++")
        print(f"Score: {score}")
        print(solution_str[:500])
        print("+++++++++++++++++++++++++++++")
        return score
    else:
        format_score = 0.0

        if solution_str.startswith("eless"):
            format_score += 0.1
        if solution_str.endswith("</answer>"):
            format_score += 0.1

        if "Done<answer>" in solution_str.replace("\n", ""):
            format_score += 0.1

        if "eless" in solution_str and "Done" in solution_str:
            format_score += 0.02

        if "<code>" in solution_str and "</code>" in solution_str:
            format_score += 0.02

        if "<answer>" in solution_str and "</answer>" in solution_str:
            format_score += 0.02

        return format_score
