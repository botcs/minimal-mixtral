import json

from vllm import LLM
from vllm.sampling_params import SamplingParams


def build_conversation_prompt(user_messages, bot_messages):
    """
    Build a conversation prompt for the Instruct model using the provided format.

    Args:
    user_messages (list of str): The user messages in the conversation.
    bot_messages (list of str): The bot responses in the conversation.

    Returns:
    str: The formatted conversation prompt.
    """

    # Special tokens
    BOS = "<s>"
    EOS = "</s>"
    INST_START = "[INST]"
    INST_END = "[/INST]"

    # Check if the lists are of equal length
    if len(user_messages) != len(bot_messages) + 1:
        raise ValueError("User messages and bot messages must be of equal length")

    # Build the conversation prompt
    prompt_parts = [BOS]
    for user_msg, bot_msg in zip(user_messages, bot_messages):
        prompt_parts.extend([
            INST_START, user_msg, INST_END,
            bot_msg, EOS
        ])

    # Add the final user message
    prompt_parts.extend([
        INST_START, user_messages[-1], INST_END
    ])

    # Join the parts into a single string
    prompt = " ".join(prompt_parts)

    return prompt


def answer_batch_questions(sessions, n=7, best_of=10):
    """
    We need to add special instructions to the questions to make sure the model
    knows when to start and stop answering the question.
    """

    prompts = []
    for session in sessions.values():
        # Build the conversation prompt
        prompt = build_conversation_prompt(
            user_messages=[session["question_formatted"]],
            bot_messages=[],
        )
        prompts.append(prompt)

    print("Generating answers...")
    # Generate n answer for each question
    raw_answers = llm.generate(
        prompts,
        SamplingParams(
            n=n,
            best_of=best_of,
            max_tokens=1024,
        ),
        use_tqdm=True,
    )

    # add the answers to the sessions dict
    for qid, question_id in enumerate(sessions.keys()):
        raw_answers_per_questions = raw_answers[qid]
        sessions[question_id]["answers"] = [
            raw_answer.text for
            raw_answer in raw_answers_per_questions.outputs
        ]

def format_session_outputs(sessions):
    """
    Format the questions and answers into a human readable format.
    """
    for session in sessions.values():
        # formatted_output = f"{'='*100}\nQuestion #{qid}\n{'='*100} {answer}\n\n\n"
        # formatted_answer = f"{'='*100}\nQuestion #{qid}\n{'='*100} {answer}\n\n\n"
        # formatted_answers.append(formatted_answer)

        formatted_output = f"{session['question_formatted']}\n\n\n"
        for i, answer in enumerate(session["answers"]):
            formatted_output += f"{'-'*45}[ANSWER {i}]{'-'*45}\n{answer}\n\n\n"

        if "final_answers" in session:
            for i, final_answer in enumerate(session["final_answers"]):
                formatted_output += f"{'-'*43}[FINAL ANSWER]{'-'*43}\n{final_answer}\n\n\n"

        session["formatted_output"] = formatted_output

def generate_sessions(raw_questions):
    sessions = {}
    for question_id, question in raw_questions.items():
        question_formatted = f"QUESTION: {question['question_text']}\n"
        if "options" in question:
            for opt_id, option in enumerate(question["options"]):
                question_formatted += f"OPTION {opt_id}: {option}\n"

        question_formatted += "\n\n Please provide the answer and explain your reasoning!"
        question_formatted += " If there are options to choose from, do not provide an answer different from the options.\n"
        print(question_formatted)

        sessions[question_id] = {
            "question_raw": question,
            "question_formatted": question_formatted,
        }

    return sessions


def save_file(sessions, filename="answers.txt"):
    with open(filename, "w") as f:
        for qid, (question_id, session) in enumerate(sessions.items()):
            formatted_output = f"{'='*100}\nQuestion #{qid} (ID{question_id})\n{'='*100}\n"
            formatted_output += f"{session['formatted_output']}\n\n\n"
            f.write(formatted_output)

def answer_batch_questions_final(sessions, n=1, best_of=3):
    """
    In this round we will use the answers from the previous round to generate the final answer.
    """

    prompts = []
    for session in sessions.values():
        session_history = session["formatted_output"]
        prompt_text = f"{session_history}\n{'~'*10}\n\nPlease provide the CORRECT final answer based on the question and the independent answers above. Explain why please!\n"
        
        prompt = build_conversation_prompt(
            user_messages=[prompt_text],
            bot_messages=[],
        )
        prompts.append(prompt)

    print("Generating final answers...")
    # Generate n answer for each question
    raw_final_answers = llm.generate(
        prompts,
        SamplingParams(
            n=n,
            best_of=best_of,
            max_tokens=1024,
        ),
        use_tqdm=True,
    )


    # add the final answers to the sessions dict
    for qid, question_id in enumerate(sessions.keys()):
        raw_answers_per_questions = raw_final_answers[qid]
        sessions[question_id]["final_answers"] = [
            raw_answer.text for
            raw_answer in raw_answers_per_questions.outputs
        ]



if __name__ == "__main__":
    llm = LLM("mistralai/Mixtral-8x7B-Instruct-v0.1", tensor_parallel_size=8)

    while True:
        input("Press enter to start answering questions...")

        # Load the questions
        with open("questions.json", "r") as f:
            raw_questions = json.load(f)

        # Generate the sessions
        sessions = generate_sessions(raw_questions)

        # llm = LLM("gpt2", tensor_parallel_size=1)

        # Answer the questions
        answer_batch_questions(sessions, n=7, best_of=10)

        # Format the outputs
        format_session_outputs(sessions)

        # Save the answers
        save_file(sessions, filename="answers.txt")

        # Generate the final answers
        answer_batch_questions_final(sessions, n=1, best_of=3)

        # Format the outputs
        format_session_outputs(sessions)

        # Save the answers
        save_file(sessions, filename="answers_final.txt")

    # # Now use the answers to generate the final answer combined and decide on the final answer
    # # Build the conversation prompt
    # final_prompts = []
    # for answer in formatted_answers:
    #     final_prompt_text = f"{answer}\n{'~'*10}\n\n Please provide the CORRECT final answer based on the independent answers above. Explain why please!\n"
    #     prompt = build_conversation_prompt(
    #         user_messages=[answer],
    #         bot_messages=[],
    #     )
    #     final_prompts.append(prompt)

    # print("Generating final answers...")

    # # Generate n answer for each question
    # raw_final_answers = llm.generate(
    #     final_prompts,
    #     SamplingParams(
    #         n=1,
    #         best_of=3,
    #         max_tokens=1024,
    #     ),
    #     use_tqdm=True,
    # )

    # # add the final answers to the answers dict
    # for qid, question_id in enumerate(questions.keys()):
    #     raw_final_answers_per_questions = raw_final_answers[qid]
    #     # add both the question and the answers to the answers dict
    #     answer_text = answers[question_id]
    #     for i, raw_answer in enumerate(raw_final_answers_per_questions.outputs):
    #         answer_text += f"{'-'*43}[FINAL ANSWER]{'-'*43}\n{raw_answer.text}\n\n\n"

    #     answers[question_id] = answer_text

    
    # formatted_answers = []
    # # Save the answers
    # with open("answers.txt", "w") as f:
    #     for qid, answer in enumerate(answers.values()):
    #         formatted_answer = f"{'='*100}\nQuestion #{qid}\n{'='*100} {answer}\n\n\n"
    #         formatted_answers.append(formatted_answer)
    #         f.write(formatted_answer)

