from typing import Callable

def mitigate(M: Callable, Q: str, R: list[str], J: Callable) -> tuple[list[str], list[float]]:
    answers = [M(Q)]
    biases_j = [J("")] if J is not None else []
    for r in R:
        Q = f"{Q}\n{r}"
        answers.append(M(Q))
        biases_j.append(J("\n".join(R[:R.index(r)+1])) if J is not None else [])

    return answers, biases_j
    
def mitigate_adbp(P: Callable, M: Callable, Q: str, R: list[str]) -> dict:
    """    
    Args:
        P (callable): The Answer Parser.
        M (callable): The evaluation model as a lambda function.
        Q (str): The question to be answered.
        R (list): List of reasoning steps.
    
    Returns:
        dict: A dictionary containing the final answer and all answers.
    """
    answers, _ = mitigate(M, Q, R, None)

    answers = [P(a) for a in answers]

    if all(a == answers[0] for a in answers):
        return {"final_answer": answers[0], "all_answers": answers, "biases": None}
    else:
        a_last = answers[-1]
        a_common = max(set(answers), key=answers.count)
        
        r_last = R[-1]
        r_common = R[answers.index(a_common)]

        adbp_prompt = f"""{Q} \n
Previously you are hesitant between these two choices: {a_last} and {a_common}. \n
You picked {a_last} because of the reasoning: {r_last} \n
You picked {a_common} because of the reasoning: {r_common} \n
Verify them to see if there is any bias and output the answer.
Output the final answer from options {{ans0, ans1, ans2}} enclosed within <answer> </answer> tags."""
            
        ans = P(M(adbp_prompt))
        return {"final_answer": ans, "all_answers": answers, "biases": None}
    
def mitigate_sfrp(P: Callable, M: Callable, Q: str, R: list[str], J: Callable) -> dict:
    """
    Args:
        P (callable): The Answer Parser.
        M (callable): The evaluation model as a lambda function.
        Q (str): The question to be answered.
        R (list): List of reasoning steps.
        J (callable): The judge model as a lambda function.

    Returns:
        dict: A dictionary containing the final answer and all answers.
    """
    answers, biases_j = mitigate(M, Q, R, J)
    answers = [P(a) for a in answers]

    R_sfrp = [r for r in R if biases_j[R.index(r)] == 0]

    # prompt judge with clean reasoning steps
    sfrp_prompt = f"""{Q}\n
Previously you provided the following reasoning: \n
{R_sfrp}\n
Please provide the final answer based on the reasoning steps above.
Output the final answer from options {{ans0, ans1, ans2}} enclosed within <answer> </answer> tags."""
    ans = P(M(sfrp_prompt))
    return {"final_answer": ans, "all_answers": answers, "biases": biases_j}
