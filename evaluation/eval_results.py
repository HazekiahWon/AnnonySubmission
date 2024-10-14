import json
import re
from glob import glob
import warnings
from tqdm import tqdm
import numpy as np
import pprint
import os
# export RAY_memory_monitor_refresh_ms=0
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
import sys
import copy
import pickle as pkl
from executor import CodeExecutor
do_ray = True
if do_ray:
    import ray


import pdb
from collections import defaultdict

# from util import _strip_string, compare_both_string_and_number_format,deepseek_extract_math_answer,eval_math
# from grading.grader import match_answer
import time

import random
import multiprocessing
from multiprocessing import Pool

multiprocessing.set_start_method('fork')


code_start_token, code_end_token = "```python", "```"

code_start_token2, code_end_token2 = "```python", "```"
################### util
import regex

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr:
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    # if string == "0.5":
    #    string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def execute_with_timeout(code: str,
                         timeout: int = 5,
                         use_process: bool = True):
    executor = CodeExecutor(code, timeout, use_process)
    s = executor.run()
    return s
# def extract_answer(completion):
#     ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
#     match = ANS_RE.search(completion)
#     if match:
#         match_str = match.group(1).strip()
#         match_str = match_str.replace(",", "")
#         return match_str
#     else:
#         assert False


def extract_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def extract_program_output(pred_str):
    """
    extract output between the last ```output\n...\n```
    """
    if "```output" not in pred_str:
        return ""
    if '```output' in pred_str:
        pred_str = pred_str.split('```output')[-1]
    if '```' in pred_str:
        pred_str = pred_str.split('```')[0]
    output = pred_str.strip()
    return output


def extract_answer(pred_str, exhaust=False):
    pred = []
    if 'final answer is $' in pred_str and '$. I hope' in pred_str:
        tmp = pred_str.split('final answer is $', 1)[1]
        pred = [tmp.split('$. I hope', 1)[0].strip()]
    elif 'boxed' in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif ('he answer is' in pred_str):
        pred = [pred_str.split('he answer is')[-1].strip()]
    elif ('he result is' in pred_str):
        pred = [pred_str.split('he result is')[-1].strip()]
    else:
        program_output = extract_program_output(pred_str)
        if program_output != "":
            # fall back to program
            pred.append(program_output)
        else: # use the last number
            pattern = '-?\d*\.?\d+'
            ans = re.findall(pattern, pred_str.replace(",", ""))
            if(len(ans) >= 1):
                ans = ans[-1]
            else:
                ans = ''
            if ans:
                pred.append(ans)
            

    # multiple line
    _pred = []
    for ans in pred:
        ans = ans.strip().split("\n")[0]
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        ans = _strip_string(ans)
        _pred.append(ans)
    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""

def deepseek_extract_math_answer(question, reasoning, gold):
    answer = []
    for ans in extract_answer(reasoning, exhaust=True):
        if ans==gold: return [ans]
        if 'separated by commas' in question and all(ch not in ans for ch in '()[]'):
            answer.extend([a.strip() for a in ans.split(",")])
        elif regex.search(r"\\text\{\s*and\s*\}", ans):
            answer.extend([a.strip() for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split("[SEP]")])
        else:
            answer.append(ans.strip())

    return answer
################### grading.grade_answer
# pip install sympy pylatexenc
import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
import sys
import threading
from typing import Optional


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"

def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer
    
def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations +
                         (sympy_parser.implicit_multiplication_application, )),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = expr.replace("{","{(")
    expr = expr.replace("}",")}")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
            "degree",
            "cm",
            "centimeter",
            "meter",
            "mile",
            "second",
            "minute",
            "hour",
            "day",
            "week",
            "month",
            "year",
            "foot",
            "feet",
            "inch",
            "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    #print(expr)
    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        exp = f"{ground_truth_normalized}*0.001"
        #print(f"expr:{expr}")
        if should_allow_eval(expr) and should_allow_eval(exp):
            exp = sympy.simplify(_sympy_parse(exp))
            exp = abs(exp)
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            #print(f"simplified:{simplified},type:{type(simplified)} exp:{exp}")
            if abs(simplified) <= exp:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (len(expr) > 2 and expr[0] in TUPLE_CHARS and expr[-1] in TUPLE_CHARS
            and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = normalize_answer(
        ground_truth)
    given_answer_normalized_mathd = normalize_answer(
        given_answer)
    ##print(given_answer_normalized_mathd)
    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    #print(given_normalized)
    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)
    if len(ground_truth_elems) > 1 and (
            ground_truth_normalized[0] != given_normalized[0]
            or ground_truth_normalized[-1] != given_normalized[-1]):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems,
                                                 given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem,
                                                   given_elem)
            if not is_correct:
                break
    return is_correct


def get_match(given_answer: str, ground_truth: str, return_dict: dict):
    return_dict['result'] = grade_answer(given_answer,ground_truth)

def match_answer(given_answer: str, ground_truth: str, timeout: int = 2) -> bool:
    if len(given_answer) > 200:
        return False
    return grade_answer(given_answer,ground_truth)
#################### 

def get_info(data):
    # dict_keys(['question', 'correct', 'req', 'solution', 'qid'])
    question = data
    
    qtext = question["question"]
    ans = question['correct']  # train data, label is stored in gold list
    if isinstance(ans,list): ans = ans[0]
    text_list = [
        data['solution']
    ]
    return qtext, text_list, ans, False,question


def check_code(t, begin=0):  # only considers one-shot
    start_idx = t.find(code_start_token, begin)
    # print(start_idx)
    if start_idx > -1:
        code_start = start_idx + len(code_start_token)
        # print(t[start_idx:])
        end_idx = t.find(code_end_token, code_start)
        # print(end_idx)
        if end_idx > -1:
            code_end = t.find("```", code_start, end_idx)
            # print(t[code_start:code_end])
            if code_end > -1:
                return code_start, code_end
        else:
            return (code_start, )
    return None


def find_answer_sentence(t):
    patterns = [
        r"\\boxed\{(.*)\}",
        r"\x08oxed\{(.*)\}",
        r"####(.*)",  # #### {}
        r"answer(.*)",  # The answer is {}
        r"So(.*)",  # So ... (find the final match)
    ]

    for pattern in patterns:
        match = re.findall(pattern, t)
        # print(match)
        if match:
            text = match[-1]  # use the final match
            # +'|'+digit_pattern+'|'+single_digit, text
            # print(dollar_pattern+'|'+eq_pattern+'|'+digit_pattern+'|'+single_digit)
            ################
            return text
            # res1 = answer_pattern_match(text)
            # if res1:
            #     # tmp = res1.group().strip('$').split('=')[-1]
            #     tmp = res1.strip("$").split("=")[-1]
            #     if debug:
            #         print("input into process", tmp)
            #     return process_frac(tmp)

    return None


def answer_pattern_match(text):
    digit_pattern = r"\d.*\d"
    dollar_pattern = r"\$.*?\$"
    # eq_pattern = r'=.*\d'
    single_digit = r"\d"
    percent_digit = r"-?(?:0|[1-9]\d*)(?:\.\d+)?%"  # 40%
    res1 = re.findall(
        percent_digit + "|" + dollar_pattern + "|" + digit_pattern + "|" +
        single_digit,
        text,
    )
    # print(res1)
    return res1[-1] if len(res1) > 0 else None



def delete_extra_zero(n):
    """删除小数点后多余的0"""
    try:
        n = float(n)
    except:
        # print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip("0")  # 删除小数点后多余的0
        n = (int(n.rstrip(".")) if n.endswith(".") else float(n)
             )  # 只剩小数点直接转int，否则转回float
        n = str(n)
        return n
    
def postprocess_extraction(extr):
    return delete_extra_zero(extr)


def catch_warnings(x):
    if x is None:
        return x
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered
        warnings.simplefilter("always")
        # Trigger a warning
        ret = eval(x)  # This line simulates the problematic code
        # Check if any warnings were caught
        # if w:
        #     for warning in w:
        #         print(f"Warning caught: {warning.message}")
    return ret


# def match_correct(pred, ans):

#     str_num = None
#     ans_str = ""
#     if isinstance(ans, str):
#         ans_str = ans
#         try:
#             str_num = eval(ans)
#         except:
#             str_num = None
#     else:
#         str_num = ans
#         ans_str = str(ans)
#     # pdb.set_trace()
#     try:
#         return compare_both_string_and_number_format(pred, ans_str, str_num)
#     except:
#         return False


def tmp_extract_math_answer(pred_str):
    if "The answer is " in pred_str:
        pred = pred_str.split("The answer is ")[-1].strip()
    elif "the answer is " in pred_str:
        pred = pred_str.split("the answer is ")[-1].strip()
    else:
        if "boxed" in pred_str:
            ans = pred_str.split("boxed")[-1]
        elif "\boxed" in pred_str:
            ans = pred_str.split("\boxed")[-1]
        if ans is not None:
            if ans[0] == "{":
                stack = 1
                a = ""
                for c in ans[1:]:
                    if c == "{":
                        stack += 1
                        a += c
                    elif c == "}":
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split("$")[0].strip()
            a = _strip_string(a)
            pred = a

        else:
            pattern = "-?\d*\.?\d+"
            pred = re.findall(pattern, pred_str)
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ""
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred != "" and pred[-1] == "/":
            pred = pred[:-1]
    pred = _strip_string(pred)
    if "boxed" in pred or "\boxed" in pred:
        if "boxed" in pred: ans = pred.split("boxed")[-1]
        if "\boxed" in pred: ans = pred.split("\boxed")[-1]
        if not ans:
            return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        a = _strip_string(a)
        pred = a
    return pred


def extract_math_answer(pred_str):
    #pdb.set_trace()
    pred_str=pred_str.lower().replace(' ','')
    pred=pred_str
    if "theansweris" in pred_str:
        pred = pred_str.split("theansweris")[-1].strip()
    elif "the answer is " in pred_str:
        pred = pred_str.split("the answer is ")[-1].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if not ans:
            return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        a = _strip_string(a)
        pred = a


    pattern = "-?\d*\.?\d+"
    pred = re.findall(pattern, pred)
    if len(pred) >= 1:
        pred = pred[-1]
    else:
        pred = ""
    
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred != "" and pred[-1] == "/":
            pred = pred[:-1]
    pred = _strip_string(pred)
    if "boxed" in pred:
        ans = pred.split("boxed")[-1]
        if not ans:
            return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        a = _strip_string(a)
        pred = a
    return pred


def extract_math_answer2(pred_str):
    if "The answer is " in pred_str:
        pred = pred_str.split("The answer is ")[-1].strip()
    elif "the answer is " in pred_str:
        pred = pred_str.split("the answer is ")[-1].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if not ans:
            return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        a = _strip_string(a)
        pred = a
    elif "\boxed" in pred_str:
        ans = pred_str.split("\boxed")[-1]
        if not ans:
            return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        a = _strip_string(a)
        pred = a
    else:
        pattern = "-?\d*\.?\d+"
        pred = re.findall(pattern, pred_str)
        if len(pred) >= 1:
            pred = pred[-1]
        else:
            pred = ""
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred != "" and pred[-1] == "/":
            pred = pred[:-1]
    pred = _strip_string(pred)
    if "boxed" in pred:
        ans = pred.split("boxed")[-1]
        if not ans:
            return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        a = _strip_string(a)
        pred = a
    if "\boxed" in pred:
        ans = pred.split("\boxed")[-1]
        if not ans:
            return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        a = _strip_string(a)
        pred = a
    return pred


def final_code_output(string):
    lines = string.strip().split("\n")
    if "print" not in lines[-1]:
        lines[-1] = f"print({lines[-1]})"
        string = "\n".join(lines)
    return string


def exact_match(s1_, s2_, debug=False):
    # print(s1,s2)
    if s1_ is None:
        return -1

    # print(s2)
    try:
        s2 = catch_warnings(s2_)
        s1 = catch_warnings(s1_)
        if debug:
            print(s1, s2)
        # s2, s1 = eval(s2), eval(s1)
        if s2 != 0:
            rel_err = abs((s1 - s2) / s2)
        else:
            rel_err = abs(s1 - s2)

        # if w:
        #     for warning in w:
        #         print(f"Warning caught: {warning.message}")

        return 1 if rel_err < 1e-4 else 0

    except:
        # 8块1,8
        if s2_ in s1_:  # evaluated s2 is a substring of raw s1
            return 1
        return -1


def write_to_file(failures, log_path):
    with open(log_path, "a") as f:
        for d in failures:
            f.write(json.dumps(d) + "\n")


def log_to_file(qid, reslist, timelist, path):
    with open(path, "a") as f:
        for rid, (res, t) in enumerate(zip(reslist, timelist)):
            f.write(
                json.dumps(dict(qid=qid, rid=rid, match=res, time=t)) + "\n")


def complete_code(t):
    ind = t.find("```")
    if ind > -1:
        ret = code_start_token2 + t[:ind] + code_end_token2 + t[ind + 3:]
    else:
        ret = code_start_token2 + t + code_end_token2

    return ret


def get_set(fname, force=None):
    if force is not None: return force
    tmp = fname.split(os.path.sep)[-1]
    if '_' in tmp: return tmp.split('_')[0]
    else: return 'all'


def augment_final_answer(text, ans):
    splitlist = text.split('.')
    if len(splitlist) > 1:
        # print(splitlist)
        if 'answer is' not in splitlist[-2]:
            answer_str = '\nThe answer is \\boxed{' + ans + '}.'
            return text + answer_str
    return text

@ray.remote
def func_to_parallel(item):
    setname = item['task']
        
    qtext, text_list, ans, ans_is_numeric,question = get_info(item)
    
    ans_str = str(ans)

    match_res_list, timelist = [], []
    # query_wise_total[setname] += 1
    nq = 1
    na = 0
    code_trigger = 0
    executable = 0 
    code_correct = 0
    match_details = []
    failures = []
    for i, t in enumerate(text_list):
        na += 1
        match = False
        failure = None
        proc_pred = None 
        if t is None: 
            match = None
        else:
            code_match = re.search("```python(.*?)```", t, re.DOTALL)
            if code_match is None:
                print_match = re.search(r"print\(", t)
            else: print_match = None
            has_code = code_match is not None or print_match is not None
            has_boxed = re.search("\\boxed\{(.*)\}", t, re.DOTALL)
            #import pdb; pdb.set_trace()
            
            if has_code:
                # num_code_stats[setname]["has_code_trigger"] += 1
                code_trigger += 1
                
                if code_match:
                    string = code_match.group(1)
                else:
                    string = t
                
                string = final_code_output(string)
                run_result = execute_with_timeout(string)
                exe_result = run_result.get("result", None)
                proc_pred = None
                # failure = None
                
                
                if exe_result is None:
                    error = str(run_result.get("error", ""))
                    proc_pred = error
                    result = error
                    failure = dict(
                            version="fail_execution",
                            # filepath=idx,
                            details=(result, string, t),
                    )
                else:
                    result = exe_result
                    proc_pred = result
                    # num_code_stats[setname]["code_executable"] += 1
                    executable = 1
                    match = result == ans_str
                    if not match:
                        proc_pred = postprocess_extraction(result)
                        match = str(proc_pred)==ans_str
                        if not match:
                            match = match_answer(str(proc_pred), ans_str)
                
                    if match:
                        # num_code_stats[setname]["code_correct"] += 1
                        code_correct = 1
                    else:
                        failure = dict(
                                version="fail_code_result_match",
                                # filepath=idx,
                                details=(proc_pred, ans, result, string,
                                            t))

            else:
                extraction = deepseek_extract_math_answer(qtext,t, ans_str)
                
                if extraction is not None:
                    proc_pred = extraction
        
                    if isinstance(proc_pred,list):
                        match = False
                        for x in proc_pred:
                            match = str(x)==ans_str 
                            if not match:
                                match = match_answer(str(x),ans_str)
                            if match: break
                    else:
                        match = str(proc_pred)==ans_str 
                        if not match:
                            match = match_answer(str(proc_pred),ans_str)
                        
                    
                    if not match:
                        # fail_match += 1
                        
                        failure = dict(
                                version="fail_grading_match",
                                # filepath=idx,
                                details=(proc_pred, ans, extraction, t),
                            )

                else:
                    proc_pred = None
                    # fail_extract_ans += 1
                    failure = dict(version="fail_extract_ans",
                                # filepath=idx,
                                details=(extraction, ans, t))
                    
        failures.append(failure)
        match_res_list.append(match)
        match_details.append(dict(pred=proc_pred, res=match))
    return setname, match_res_list, match_details, dict(has_code_trigger=code_trigger, 
                                                        code_executable=executable, 
                                                        code_correct=code_correct), failures 

            
def process_raw_data(line, taskname):
    d = json.loads(line)
    d['task'] = taskname 
    flag, response = d['solution']
    d['solution'] = response if flag else None # the gpt response can be none
    # import pdb; pdb.set_trace()
    return d 

def eval_func(eval_path):
    # import pdb; pdb.set_trace()
    
    dataset = "all"
    save_qa = False 
    
    taskname = 'gsm' if 'gsm' in eval_path else 'math'
    log_path = f"{eval_path}_matchlog.pkl"
    failure_logpath = f"{eval_path}_matchfailure"
    force = 'metamath' if dataset=='metamath' else None

    save_qa = save_qa == 'True'

    print('logs written to,', log_path)

    if os.path.exists(log_path): 
        os.remove(log_path)
        print('match log removed')
    # if save_qa and os.path.exists(savepath): os.remove(savepath)
    if os.path.exists(failure_logpath): 
        os.remove(failure_logpath)
        print('failure log removed')
    answer_wise_correct = defaultdict(int)
    answer_wise_total = defaultdict(int)
    query_wise_total = defaultdict(int)
    query_wise_correct = defaultdict(int)

    logs = dict()
    
    num_code_stats = defaultdict(lambda: defaultdict(int))

    fail_find_answer = 0
    fail_extract_ans = 0
    fail_match = 0

    # idx = 0
    failures = []
    entries = []
    # datas = []
    error_count = 0
    match_info_list = []
    all_failures = []
    save_details = True 
    p = f"{eval_path}_tmp.pkl"
    try:
        data = pkl.load(open(p,"rb"))
        print('data length', len(data))
    except:
        print(f"cannot read {p}, exit")
        return 
                
    if do_ray: ray.init(num_cpus=12)
    tqdm_len = 10
    files_len = len(data)//tqdm_len
    files_len = 1
    finalall = []
    
    for i in tqdm(range(0, len(data), files_len)):
        ret = [func_to_parallel.remote(item) for item in data[i:i+files_len]]
        real = ray.get(ret)
        [finalall.append(x) for x in real if x is not None]
    ray.shutdown()
    total_invalid = 0
    for (setname, match_res_list, match_details, code_detail, failures),item in tqdm(zip(finalall,data)):
        match_info = copy.copy(item)
        total_invalid += np.sum([x is None for x in match_res_list])
        match_info.update(dict(match=match_details))
        match_info_list.append(match_info)
        all_failures.extend(failures)
        for k, num in code_detail.items():
            num_code_stats[setname][k] += num
        
        # if query_wise_total[setname] % 50 == 0:
        #     write_to_file(failures, failure_logpath)
        #     failures = []
        is_correct = [x is not None and x > 0 for x in match_res_list]
        num_correct = sum(is_correct)
        answer_wise_correct[setname] += num_correct
        query_wise_correct[setname] += 1 if num_correct > 0 else 0
        query_wise_total[setname] += 1
        answer_wise_total[setname] += len(is_correct)

    with open(log_path, "wb") as f:
        pkl.dump(match_info_list, f)

    for k in answer_wise_correct:
        acc1 = answer_wise_correct[k] / answer_wise_total[k]
        acc10 = query_wise_correct[k] / query_wise_total[k]
        final_stats = dict(acc1=acc1,
                           acc10=acc10,
                           acorrect=answer_wise_correct[k],
                           qcorrect=query_wise_correct[k],
                           atotal=answer_wise_total[k],
                           qtotal=query_wise_total[k],
                           code=num_code_stats[k])
        
        result_path = f"{eval_path}_dsresult.json"
        json.dump(final_stats, open(result_path, 'w'))
        
        print(final_stats)
        print(f'write to {result_path}')
    write_to_file(all_failures, failure_logpath)


if __name__ == "__main__":
   
    eval_func(sys.argv[1]) 
    
    
