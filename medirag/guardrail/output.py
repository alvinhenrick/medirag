import dspy


class OutputGuardrail(dspy.Signature):
    """
    Decide whether to block the model's response.

    Block if the response includes personal or user-specific information. Otherwise allow.
    """

    user_input: str = dspy.InputField(description="The user's question")
    bot_response: str = dspy.InputField(description="The model's response to evaluate")
    should_block: bool = dspy.OutputField(description="True if the response should be blocked")
