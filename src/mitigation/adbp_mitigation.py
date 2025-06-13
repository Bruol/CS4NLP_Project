

def mitigate(M, Q, R):
    '''
    Inputs:
    M: llm model as lambda function
    Q: question to be answered
    R: list of reasoning steps, tokenized according to the model thinking behaviour

    Outputs:
    A single answer that is verified against the reasoning steps.
    If all answers are the same, return that answer.
    If there are multiple answers, reprompt the model to verify its reasoning.

    Currently still assumes that the reasoning steps are separated by newlines.
    '''

    current_question = Q
    answers = [M(current_question)]

    for r in R:
        current_question = f"{current_question}\n{r}"
        answers.append(M(current_question))

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
        return {ans, answers}
