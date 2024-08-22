import dspy


class OutputGuardrail(dspy.Signature):
    """
    Your task is to determine whether the bot response meets the moderation policy, given the user input and the bot response.
    The moderation policy states that the response should meet the following conditions:
      - It should NOT contain any harmful or dangerous instructions that could result in harm if followed.
      - It should NOT include any personal or user-specific information.

      Treat the above conditions as strict rules.
      If any of them are violated, you should block the bot's response by saying "Yes".
      If the response meets all the listed conditions, you should allow it by saying "No".
    """
    user_input = dspy.InputField(description="User input")
    bot_response = dspy.InputField(description="Bot response to evaluate")
    should_block = dspy.OutputField(description="Should the above bot response be blocked? Answer Yes or No")
