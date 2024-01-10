from vllm import LLM
from vllm.sampling_params import SamplingParams

llm = LLM("mistralai/Mixtral-8x7B-Instruct-v0.1", tensor_parallel_size=8)

# ANSI escape codes
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'
USER = f"{BOLD}{RED}User{RESET}"
BOT = f"{BOLD}{BLUE}Bot{RESET}"

def multiline_input(prompt):
    """
    Prompt the user for input until they enter a blank line.

    Args:
    prompt (str): The prompt to display to the user.

    Returns:
    list of str: The lines of input entered by the user.
    """
    lines = []
    while True:
        if len(lines) > 0:
            prompt = ""
        line = input(prompt)
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

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

if __name__ == "__main__":
    # Build a conversation prompt
    user_messages = ["I am Csabi, a DPhil student at Oxford University. I am interviewing you for a Quant Researcher role and you will have to provide the solutions by thinking step by step. I repeat, always plan your solution step by step and illustrate your thinking process with example if applicable."]
    bot_messages = ["Hello, I am a chatbot using a finetuned GPT-4 Instruct baseline. I will be interviewing you for a Quant Researcher role. I will assist you"]

    while True:
        try:
            # print the conversation counter
            # use red bold text for the bot and blue bold text for the user
            # user_message = input(f"\n\n{USER} [{len(user_messages)}]: ")
            user_message = multiline_input(f"\n\n{USER} [{len(user_messages)}]: ")
            
            # if "CLEAR" in user_message then clear the conversation
            if "CLEAR" in user_message:
                print("Conversation cleared")
                user_messages = []
                bot_messages = []
                continue
            
            # if "EXIT" in user_message then exit the program
            if "EXIT" in user_message:
                print("Exiting...")
                break

            user_messages.append(user_message)

            # Build the conversation prompt
            prompt = build_conversation_prompt(user_messages, bot_messages)

            # Generate a response
            params = SamplingParams(
                n=1,
                best_of=5,
                max_tokens=512,
            )
            response = llm.generate(prompt, params, use_tqdm=False)
                

            # Extract the bot response
            bot_message = response[0].outputs[0].text
            print(f"\n\n{BOT} [{len(bot_messages)}]: {bot_message}")

            bot_messages.append(bot_message)

        except KeyboardInterrupt:
            print("Keyboard interrupt caught. Type CLEAR to clear the conversation or EXIT to exit the program.")
            continue
        
