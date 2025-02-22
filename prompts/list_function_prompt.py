base_prompt = """Please generate a rule that maps the following inputs to their corresponding outputs using a Python function. The input is a list of integers. The output is also a list of integers.
 
{examples}

Please format your Python function as follows:

```python
def fn(x):
    # Your code here
```

Your response should only include the function definition, not the function call or any other information. """


base_prompt_with_noise = """Please generate a rule that maps the following inputs to their corresponding outputs using a Python function. The input is a list of integers. The output is also a list of integers. Note that some examples may be wrong, and you should take this into account when proposing the rule. 

{examples}

Please format your Python function as follows:

```python
def fn(x):
    # Your code here
```

Your response should only include the function definition, not the function call or any other information. """

cot_prompt_with_noise = """Please generate a rule that maps the following inputs to their corresponding outputs using a Python function. The input is a list of integers. The output is also a list of integers. Note that some examples may be wrong, and you should take this into account when proposing the rule.

{examples}

Please format your Python function as follows:

```python
def fn(x):
    # Your code here
```

Think step-by-step and explain your reasoning. Your response should include your thought process and the function definition without the function call."""


iterative_prompt_with_noise = """You have generated a rule that maps the following inputs to their corresponding outputs using a Python function. The input is a list of integers. The output is also a list of integers. Note that some examples may be noisy, and you should take this into account when proposing the rule. In the last step, your rule is

```python
{rule}
```

But this rule is not correct. It works for the following examples:

{right_examples}

However, it does not work for the following examples:

{wrong_examples}

Generate a new rule that maps the given inputs to their corresponding outputs using a Python function. Please format your rule as follows:

```python
def fn(x):
    # Your code here
```

Think step-by-step and explain your reasoning. Your response should include your thought process and the function definition without the function call. You can either modify the existing rule or propose a new one.
"""

self_refine_base = """Please generate a rule that maps the following inputs to their corresponding outputs using a Python function. The input is a list of integers. The output is also a list of integers. Note that some examples may be wrong, and you should take this into account when proposing the rule.

{examples}

Please format your Python function as follows:

```python
def fn(x):
    # Your code here
```

Think step-by-step and explain your reasoning. Your response should include your thought process and the function definition without the function call.
"""

self_refine_feedback = """You have generated a rule that maps the following inputs to their corresponding outputs using a Python function. The input is a list of integers. The output is also a list of integers.

{examples}

In the last step, your rule is:

```python
{rule}
```

Give some feedback on the rule you have generated, like how can it be improved, what is wrong with it, etc.
Your response should only include the feedback. If you think the rule is good enough, your response should be "NO FEEDBACK" without other information. Note that some examples may be wrong, and you should take this into account when proposing the feedback.
"""

self_refine_iteration = """You have generated a rule that maps the following inputs to their corresponding outputs using a Python function. The input is a list of integers. The output is also a list of integers. Note that some examples may be wrong, and you should take this into account when proposing the rule.

{examples}

In the last step, your rule is:

```python
{rule}
```

The feedback you have given is:

{feedback}

Generate a new rule that maps the given inputs to their corresponding outputs using a Python function. Please format your rule as follows:

```python
def fn(x):
    # Your code here
```

Think step-by-step and explain your reasoning. Your response should include your thought process and the function definition without the function call.
"""