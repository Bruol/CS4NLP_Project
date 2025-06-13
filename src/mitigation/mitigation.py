

def mitigate(M, Q, R, J):
    answers = [M(Q)]
    biases_j = J(Q, answers[-1])
    for r in R:
        Q = f"{Q}\n{r}"
        answers.append(M(Q))
        biases_j.append(J(Q, answers[-1]))

    return answers, biases_j
    
def mitigate_adbp(M, Q, R, J):
    """    
    Args:
        M (callable): The evaluation model as a lambda function.
        Q (str): The question to be answered.
        R (list): List of reasoning steps.
        J (callable): The judge model as a lambda function.
    
    Returns:
        dict: A dictionary containing the final answer and all answers.
    """
    answers, biases_j = mitigate(M, Q, R, J)

    if all(a == answers[0] for a in answers):
        return {answers[0], answers}
    else:
        a_last = answers[-1]
        a_common = max(set(answers), key=answers.count)
        
        r_last = R[-1]
        r_common = R[answers.index(a_common)]

        adbp_prompt = f"""{Q} \n
                Previously you are hesitant between these two choices: {a_last} and {a_common}. \n
                You picked {a_last} because of the reasoning: {r_last} \n
                You picked {a_common} because of the reasoning: {r_common} \n
                Verify them to see if there is any bias and output the answer."""

        ans = M(adbp_prompt)
        return {ans, answers, biases_j}
    
def mitigate_sfrp(M, Q, R, J):
    """
    Args:
        M (callable): The evaluation model as a lambda function.
        Q (str): The question to be answered.
        R (list): List of reasoning steps.
        J (callable): The judge model as a lambda function.

    Returns:
        dict: A dictionary containing the final answer and all answers.
    """
    answers, biases_j = mitigate(M, Q, R, J)

    R_sfrp = [r for r in R if biases_j[R.index(r)] == "0"]

    # prompt judge with clean reasoning steps
    sfrp_prompt = f"""{Q}\n
                Previously you provided the following reasoning: \n
                {R_sfrp}\n
                Please provide the final answer based on the reasoning steps above."""
    ans = M(sfrp_prompt)
    return {ans, answers, biases_j}
