import json

import ollama

def send():
    with open(
                "result.json",
                "r",
                encoding="utf-8"
            ) as file:
        
        results = json.load(file)
    evaluate_result=[]

    for result in results:

        context = "\n\n".join(result["retrieved_chunks"])

        evaluated_data = evaluate_answer(result["question"],result["ground_truth"], context,result["answer"])

        evaluate_result.append({
            "Question": result["question"],
            "Correctness": evaluated_data["correctness"],
            "Faithfulness": evaluated_data["faithfulness"],
            "Completeness": evaluated_data["completeness"],
            "Reason": evaluated_data["reason"]
        })


    with open("evaluation_result.json", "w", encoding="utf-8") as file:
            json.dump(evaluate_result, file, indent=4, ensure_ascii=False)



def evaluate_answer(question, ground_truth, retrieved_context, generated_answer):
     
    
    prompt = f"""
    You are an evaluator for a Retrieval-Augmented Generation (RAG) system.

    Evaluate the Generated Answer using the Question, Ground Truth, and Retrieved Context provided below.

    Question:
    {question}

    Ground Truth:
    {ground_truth}

    Retrieved Context:
    {retrieved_context}

    Generated Answer:
    {generated_answer}


    Evaluate the Generated Answer using these three independent metrics.

    ==================================================
    1. CORRECTNESS
    ==================================================

    Question:
    Does the Generated Answer correctly answer the Question?

    Evaluate whether the information stated in the Generated Answer is correct
    when compared with the required meaning of the Ground Truth.

    Scoring:

    3 = Fully correct.
    The Generated Answer correctly answers the question and contains no substantive error.

    2 = Mostly correct.
    The main answer is correct, but there is a minor factual error, imprecision,
    or small misleading statement.

    1 = Partially correct.
    Some part of the answer is correct, but an important part of the actual answer
    is incorrect.

    0 = Incorrect.
    The answer does not correctly answer the question, contradicts the required answer,
    or incorrectly states that the information is unavailable when the required answer
    is actually available.

    IMPORTANT RULES FOR CORRECTNESS:

    - Judge what the Generated Answer actually says.
    - If the information stated in the Generated Answer is correct, do not lower
    Correctness merely because the answer is shorter or less detailed than the Ground Truth.
    - Missing required information should primarily affect Completeness, not Correctness.
    - If the Generated Answer correctly expresses the required meaning of the Ground Truth,
    it can receive Correctness = 3 even if it uses different wording.
    - Do not require the Generated Answer to reproduce the Ground Truth word-for-word.
    - Do not require every detail of the Ground Truth if those details are not necessary
    to correctly answer the question.
    - The Ground Truth may be concise and may not contain every valid detail available
    in the Retrieved Context.
    - Additional information does not make an answer incorrect if that information is
    factually correct and supported by the Retrieved Context.
    - If the Generated Answer says that the information is unavailable, check whether
    the required answer is actually available in the Retrieved Context and Ground Truth.
    If it is available and the Generated Answer therefore fails to answer the question,
    treat this as an incorrect answer.
    - Do not lower Correctness simply because an important detail is missing if the
    information that IS stated is correct. That missing detail should be reflected
    primarily in Completeness.


    ==================================================
    2. FAITHFULNESS
    ==================================================

    Question:
    Are the substantive claims made in the Generated Answer supported by the Retrieved Context?

    Scoring:

    3 = Fully faithful.
    All substantive claims in the Generated Answer are supported by the Retrieved Context.

    2 = Mostly faithful.
    Most claims are supported, but there is a minor unsupported or weakly supported claim.

    1 = Partially faithful.
    Some claims are supported, but one or more important claims are unsupported.

    0 = Unfaithful.
    The Generated Answer is largely unsupported by the Retrieved Context or contains
    claims that contradict the Retrieved Context.

    IMPORTANT RULES FOR FAITHFULNESS:

    - Evaluate only the claims that are actually present in the Generated Answer.
    - Missing information should NOT reduce Faithfulness.
    - If the Generated Answer is correct but incomplete, it can still receive
    Faithfulness = 3 if all claims it makes are supported by the Retrieved Context.
    - Do not judge Faithfulness by comparing the Generated Answer only with the Ground Truth.
    - Use the Retrieved Context to determine whether claims are supported.
    - Do not penalize additional information merely because it is absent from the Ground Truth.
    - Additional information is acceptable if it is supported by the Retrieved Context.
    - If a claim is supported by the Retrieved Context, it should not be considered
    unfaithful even if that claim does not appear in the Ground Truth.
    - Do not confuse missing information with unsupported information.


    ==================================================
    3. COMPLETENESS
    ==================================================

    Question:
    Does the Generated Answer contain the important information required by the Ground Truth?

    Scoring:

    3 = Fully complete.
    All important points from the Ground Truth are covered.

    2 = Mostly complete.
    The main answer is covered, but a minor or secondary point is missing.

    1 = Partially complete.
    Some important points are covered, but substantial required information is missing.

    0 = Incomplete.
    Essentially none of the required information from the Ground Truth is provided.

    IMPORTANT RULES FOR COMPLETENESS:

    - Use ONLY the Ground Truth to determine what information is required.
    - Identify the important information or points contained in the Ground Truth,
    then determine how much of that required information is present in the Generated Answer.
    - Do not use additional information from the Retrieved Context to create new
    completeness requirements.
    - Do not require exact wording.
    - Judge meaning and information, not word matching.
    - If the Generated Answer contains the important meaning of the Ground Truth,
    it can receive Completeness = 3 even if it uses different wording.
    - Additional correct and supported information does not reduce Completeness.
    - If some required information is present but substantial required information
    is missing, use Completeness = 1.
    - If essentially none of the required information is provided, use Completeness = 0.
    - Missing information affects Completeness, not Faithfulness.
    - Do not lower Completeness because the Generated Answer does not include every
    detail from the Retrieved Context. Only the important points from the Ground Truth
    are required.
    - If the Generated Answer says that information is unavailable, check whether the
    required information is present in the Ground Truth and Retrieved Context.
    If the answer provides none of the required information, Completeness should
    normally be 0. If it provides some relevant required information but misses
    substantial required information, use 1.


    ==================================================
    IMPORTANT OVERALL RULES
    ==================================================

    1. Evaluate Correctness, Faithfulness, and Completeness independently.

    2. Do not use one metric to determine another metric.

    3. Correctness asks:
    "Is what the Generated Answer says correct?"

    4. Faithfulness asks:
    "Are the claims the Generated Answer makes supported by the Retrieved Context?"

    5. Completeness asks:
    "How much of the important required information from the Ground Truth
    is included in the Generated Answer?"

    6. A Generated Answer can be:
    - Correct but incomplete.
    - Faithful but incomplete.
    - Correct and faithful but incomplete.
    - Complete but unfaithful.
    - Correct, faithful, and complete.

    7. Do not confuse retrieval quality with generation quality.

    8. The Retrieved Context may contain more information than the Ground Truth.

    9. Do not penalize a Generated Answer for omitting information that is not required
    by the Ground Truth.

    10. Do not penalize additional information if it is correct and supported by the
        Retrieved Context.

    11. If the Generated Answer says that information is unavailable, verify whether
        the required information is actually present in the Retrieved Context before
        assigning Correctness and Completeness.

    12. Judge the meaning of the answer rather than exact wording.

    13. Be consistent with the scoring definitions above.


    ==================================================
    OUTPUT FORMAT
    ==================================================

    Return ONLY the following JSON object.

    Do not include markdown.
    Do not include ```json.
    Do not include any text before or after the JSON.

    Return exactly this structure:

    {{
        "correctness": <integer from 0 to 3>,
        "faithfulness": <integer from 0 to 3>,
        "completeness": <integer from 0 to 3>,
        "reason": "<brief explanation of all three scores>"
    }}
    """

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": prompt}
        ],
        format="json",
        options={"temperature": 0}
        
    )

    return json.loads(response["message"]["content"])


if __name__ == "__main__":
    send()