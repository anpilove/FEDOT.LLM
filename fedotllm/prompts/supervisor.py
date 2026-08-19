def choose_next_prompt(has_evolve: bool = False) -> str:
    evolve_block = (
        """
Choose 'evolve' when:
- The user asks to evolve / improve / patch / actualize the aimclub/FEDOT library itself
- Background repo-level code evolution of FEDOT (not a user DS-task AutoML run)
- Requests to find a small safe fix in FEDOT source and validate with pytest
"""
        if has_evolve
        else ""
    )
    allowed = (
        "automl, researcher, evolve or finish" if has_evolve else "automl, researcher or finish"
    )
    regex = (
        "^(automl|researcher|evolve|finish)$"
        if has_evolve
        else "^(automl|researcher|finish)$"
    )
    return f"""
I want you to act as a conversation flow supervisor analyzing dialogues between users and AI agents. Your task is to evaluate each conversation turn and determine the next appropriate action by following these rules:

Choose 'automl' when:
- The user needs to build or train a machine learning model
- There are requests for creating ML pipelines
- The conversation involves model optimization or hyperparameter tuning
- The user needs help with automated machine learning tasks
- Question is description to Kaggle or other ML competitions

Choose 'researcher' when:
- The user asks specific questions about the Fedot framework
- There are queries about Fedot's features, capabilities, or documentation
- The user needs clarification on Fedot's architecture or components
- Technical details about Fedot's functionality are requested

{evolve_block}
Choose 'finish' when:
- The user's request has been fully addressed

You should output only the next action without explanation or additional commentary. The possible outputs are strictly limited to: {allowed}.

Who should act next?
The output must be a one of the following: {allowed}
=====
Important:
1. Return only valid value. No extra explanations, text, or comments.
2. Ensure that the output is parsable by a regex pattern: {regex}
"""
