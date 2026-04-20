import re


def post_process_output(output: str) -> str:
    new_output = re.sub("\n+", " ", output)
    if "</think>" in new_output:
        new_output = new_output.split("</think>")[1].strip()
    return new_output
