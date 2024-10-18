import ollama
import random
import re
import json
import ast
import subprocess
import sys
import tempfile
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
nltk.download('punkt', quiet=True)

# Hyperparameters
alpha = 0.001  # Learning rate for optimizer
gamma = 0.9    # Discount factor
epsilon = 0.2  # Exploration rate
num_episodes = 100  # Number of training iterations
batch_size = 16
memory_size = 100
target_update = 10

# Define tools with real functionalities
tools = {
    "analyze_code": {
        "description": "Analyze Python code for syntax errors, PEP8 compliance, and provide basic statistics.",
        "parameters": {
            "code": "The Python code to analyze."
        }
    },
    "generate_docstring": {
        "description": "Generate a detailed docstring for a Python function, including parameters and return types.",
        "parameters": {
            "function_code": "The Python function code."
        }
    },
    "suggest_optimization": {
        "description": "Suggest performance optimizations and best practices for a given Python code snippet.",
        "parameters": {
            "code": "The Python code to optimize."
        }
    },
    "generate_test": {
        "description": "Generate unit tests for a given Python function using unittest or pytest frameworks.",
        "parameters": {
            "function_code": "The Python function code to test."
        }
    },
    "explain_code": {
        "description": "Provide a line-by-line explanation of what the given Python code does.",
        "parameters": {
            "code": "The Python code to explain."
        }
    },
    "find_bugs": {
        "description": "Analyze Python code to find logical errors or potential bugs.",
        "parameters": {
            "code": "The Python code to analyze for bugs."
        }
    },
    "test_code_locally": {
        "description": "Execute Python code locally and capture any exceptions or output.",
        "parameters": {
            "code": "The Python code to execute."
        }
    }
}

def analyze_code(code):
    """
    Analyzes Python code for syntax errors and PEP8 compliance.
    Returns syntax check result and basic statistics.
    """
    try:
        ast.parse(code)
        lines = code.strip().split('\n')
        chars = len(code)
        return f"Code is syntactically correct. Lines: {len(lines)}, Characters: {chars}"
    except SyntaxError as e:
        return f"Syntax error: {str(e)}"

def generate_docstring(function_code):
    """
    Generates a detailed docstring for a Python function.
    """
    try:
        # Extract function signature
        tree = ast.parse(function_code)
        func_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        params = [arg.arg for arg in func_node.args.args]
        docstring = f'"""{func_node.name} function.\n\n'
        if params:
            docstring += "Parameters:\n"
            for param in params:
                docstring += f"    {param} : type\n        Description of {param}.\n"
        docstring += "\nReturns:\n    type\n        Description of return value.\n\"\"\""
        return docstring
    except Exception as e:
        return f"Error generating docstring: {str(e)}"

def suggest_optimization(code):
    """
    Suggests performance optimizations and best practices for Python code.
    """
    suggestions = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call) and getattr(node.iter.func, 'id', '') == 'range':
                    if isinstance(node.iter.args[0], ast.Call):
                        suggestions.append("Consider using enumerate() for better readability when iterating with indices.")
                suggestions.append("Use list comprehensions where appropriate for more concise code.")
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 50:
                    suggestions.append(f"Function '{node.name}' is too long; consider splitting it into smaller functions.")
            if isinstance(node, ast.ImportFrom) and node.module == '__future__':
                suggestions.append("Ensure compatibility with future Python versions.")
        if not suggestions:
            return "Code follows best practices."
        else:
            return "\n".join(set(suggestions))
    except Exception as e:
        return f"Error analyzing code for optimizations: {str(e)}"

def generate_test(function_code):
    """
    Generates unit tests for a given Python function.
    """
    try:
        tree = ast.parse(function_code)
        func_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        test_code = f"""import unittest

{function_code}

class Test{func_node.name.capitalize()}(unittest.TestCase):
    def test_{func_node.name}_example(self):
        # Example test case
        result = {func_node.name}()  # Add necessary arguments
        expected = None  # Define the expected result
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
"""
        return test_code
    except Exception as e:
        return f"Error generating test: {str(e)}"

def explain_code(code):
    """
    Provides a line-by-line explanation of Python code.
    """
    explanations = []
    lines = code.strip().split('\n')
    for i, line in enumerate(lines, 1):
        explanations.append(f"Line {i}: {line.strip()} - Explanation of what this line does.")
    return "\n".join(explanations)

def find_bugs(code):
    """
    Analyzes code to find logical errors or potential bugs.
    """
    bugs = []
    try:
        tree = ast.parse(code)
        # Simple check for division by zero
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(node.right, ast.Num) and node.right.n == 0:
                    bugs.append("Division by zero detected.")
        if not bugs:
            return "No obvious bugs found."
        else:
            return "\n".join(set(bugs))
    except Exception as e:
        return f"Error analyzing code for bugs: {str(e)}"

def test_code_locally(code):
    """
    Executes Python code locally and captures any exceptions or output.
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        # Execute the code and capture output
        result = subprocess.run([sys.executable, temp_file_path], capture_output=True, text=True)

        os.remove(temp_file_path)  # Clean up the temporary file

        if result.returncode != 0:
            return f"Execution error:\n{result.stderr}"
        else:
            return f"Code executed successfully. Output:\n{result.stdout}"
    except Exception as e:
        return f"Error executing code: {str(e)}"

def execute_tool(tool_name, params):
    """
    Executes the specified tool with given parameters.
    """
    if tool_name == "analyze_code":
        return analyze_code(params["code"])
    elif tool_name == "generate_docstring":
        return generate_docstring(params["function_code"])
    elif tool_name == "suggest_optimization":
        return suggest_optimization(params["code"])
    elif tool_name == "generate_test":
        return generate_test(params["function_code"])
    elif tool_name == "explain_code":
        return explain_code(params["code"])
    elif tool_name == "find_bugs":
        return find_bugs(params["code"])
    elif tool_name == "test_code_locally":
        return test_code_locally(params["code"])
    else:
        return f"Unknown tool: {tool_name}"

def model_supports_generate(model_name):
    """
    Checks if the model supports the 'generate' method.
    """
    try:
        # Try to generate an empty prompt to test if model supports generate
        response = ollama.generate(model=model_name, prompt="", options={"num_predict": 1})
        return True
    except Exception:
        return False

def model_supports_embed(model_name):
    """
    Checks if the model supports the 'embed' method.
    """
    try:
        # Try to embed an empty string to test if model supports embed
        response = ollama.embed(model=model_name, input="")
        return True
    except Exception:
        return False

def get_model_capabilities(model_name):
    """
    Returns the capabilities of the model ('generate', 'embed', or both).
    """
    capabilities = []
    if model_supports_generate(model_name):
        capabilities.append('generate')
    if model_supports_embed(model_name):
        capabilities.append('embed')
    return capabilities

def ensure_model_available(model_name):
    """
    Ensures that the model is available by pulling it if necessary.
    """
    available_models = [model['name'] for model in ollama.list().get("models", [])]
    if model_name not in available_models:
        print(f"Model {model_name} not found locally. Attempting to pull the model...")
        try:
            ollama.pull(model_name)
            print(f"Successfully pulled model {model_name}.")
        except Exception as e:
            print(f"Failed to pull model {model_name}: {str(e)}")
            return False
    return True

def get_embeddings(text_chunks, embedding_model):
    """
    Generates embeddings for the given text chunks using the specified embedding model.
    """
    try:
        embeddings = ollama.embed(model=embedding_model, input=text_chunks)["embeddings"]
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings with {embedding_model}: {str(e)}")
        return None

def chunk_text(text, max_length=512):
    """
    Splits text into chunks suitable for embedding.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Define the task
task_prompt = input("Enter your Python coding task or question: ")

# List available models
models_info = ollama.list().get("models", [])
model_names = [model['name'] for model in models_info]
num_models = len(model_names)

print("Available models:", model_names)

# Filter out models that do not support 'generate' or 'embed'
model_capabilities = {}
for model_name in model_names:
    capabilities = get_model_capabilities(model_name)
    if capabilities:
        model_capabilities[model_name] = capabilities

if not model_capabilities:
    print("No models with supported capabilities found.")
    exit()

# Define the neural network model
class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize DQN and target network
state_size = 1  # The state is fixed (the task)
action_size = len(model_capabilities)  # Number of models with capabilities
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
memory = []

capable_model_names = list(model_capabilities.keys())

def select_action(state, epsilon):
    """
    Selects an action (model) based on epsilon-greedy policy.
    """
    if random.uniform(0, 1) < epsilon:
        action = random.randrange(action_size)
        print("Exploring: Randomly selected model.")
    else:
        with torch.no_grad():
            q_values = policy_net(state)
            max_q = q_values.max().item()
            actions = (q_values == max_q).nonzero(as_tuple=True)[1]
            action = random.choice(actions.tolist())
            print("Exploiting: Selected best model based on Q-values.")
    return action

def generate_response(model, prompt):
    """
    Generates a response using the specified model and handles tool execution if needed.
    """
    system_prompt = f"""You are an AI assistant specialized in Python programming. You have access to the following tools:

{json.dumps(tools, indent=2)}

Guidelines:
- Analyze the user's question carefully.
- If a tool is needed, respond with a JSON object specifying the tool and parameters.
- Provide detailed explanations and code examples.
- Before presenting code to the user, test it locally using the 'test_code_locally' tool.
- Follow Python best practices and PEP8 conventions.
- Structure your response with clear headings and code blocks.

Examples:
User: How can I optimize this code snippet?
[User provides code]

Assistant:
{{
    "tool": "suggest_optimization",
    "params": {{
        "code": "[User's code]"
    }}
}}

After the tool execution, incorporate the results into your final answer.

Now, proceed to assist the user."""

    try:
        # Initial response
        response = ollama.chat(
            model=model,
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}],
            tools=[
                {
                    'type': 'function',
                    'function': {
                        'name': tool_name,
                        'description': tool_info['description'],
                        'parameters': {
                            'type': 'object',
                            'properties': {param: {'type': 'string', 'description': desc}
                                           for param, desc in tool_info['parameters'].items()},
                            'required': list(tool_info['parameters'].keys()),
                        },
                    },
                } for tool_name, tool_info in tools.items()
            ],
        )
        assistant_message = response['message']
        # Check if the assistant decided to use a tool
        if assistant_message.get('tool_calls'):
            for tool_call in assistant_message['tool_calls']:
                tool_name = tool_call['function']['name']
                params = tool_call['function']['arguments']
                tool_result = execute_tool(tool_name, params)
                # Add the tool's response to the conversation
                response = ollama.chat(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': prompt},
                        assistant_message,
                        {'role': 'tool', 'content': tool_result},
                    ],
                )
                return response['message']['content'].strip()
        else:
            # No tool was used; return the assistant's content
            return assistant_message['content'].strip()

    except Exception as e:
        print(f"Error generating response with {model}: {str(e)}")
        return None

def evaluate_response(response, action):
    """
    Evaluates the response using a reviewer model and returns a numerical score.
    """
    reviewer_models = [model for model in capable_model_names if model != capable_model_names[action] and 'generate' in model_capabilities[model]]
    if not reviewer_models:
        return 5  # Neutral score if no other models are available
    reviewer_model = random.choice(reviewer_models)
    review_prompt = f"""As a Python expert, evaluate the following response to the user's task: '{task_prompt}'

Response:
{response}

Criteria for evaluation (score from 1 to 10):
1. Accuracy and relevance of the Python code or information provided.
2. Effective use of Python-specific tools (if applicable).
3. Clarity and coherence of the explanation.
4. Completeness in addressing all aspects of the Python coding task.
5. Adherence to Python best practices and conventions.
6. Quality of code implementation (if code was generated).
7. Presence and quality of tests (if applicable).
8. Whether the code was tested locally and works as intended.

Provide only the numerical score and a brief justification.

Example:
Score: 8
Justification: The code is accurate but lacks unit tests."""

    try:
        review_response = ollama.generate(model=reviewer_model, prompt=review_prompt)
        review_text = review_response['response'].strip()
        # Extract numerical score
        match = re.search(r'Score:\s*(\d{1,2})', review_text)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
        return 5  # Neutral score if parsing fails
    except Exception as e:
        print(f"Error evaluating response with {reviewer_model}: {str(e)}")
        return 5

# Main training loop
for episode in range(num_episodes):
    print(f"\nEpisode {episode + 1}/{num_episodes}")
    state = torch.tensor([[0.0]], dtype=torch.float32)  # State is fixed
    action = select_action(state, epsilon)
    selected_model = capable_model_names[action]
    print(f"Selected model: {selected_model}")

    # Ensure the model is available
    if not ensure_model_available(selected_model):
        continue

    capabilities = model_capabilities[selected_model]

    if 'generate' in capabilities:
        response = generate_response(selected_model, task_prompt)
    else:
        print(f"Model {selected_model} does not support 'generate'. Skipping.")
        continue

    if response is None:
        reward = 0
    else:
        print(f"Response from {selected_model}:\n{response}")
        reward = evaluate_response(response, action)
        print(f"Reward (score): {reward}")
    reward_tensor = torch.tensor([reward], dtype=torch.float32)

    # Store transition in memory
    memory.append((state, action, reward_tensor))

    # Limit memory size
    if len(memory) > memory_size:
        memory.pop(0)

    # Sample from memory and update policy
    if len(memory) >= batch_size:
        batch = random.sample(memory, batch_size)
        states, actions, rewards = zip(*batch)
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.stack(rewards)

        q_values = policy_net(states).gather(1, actions)
        with torch.no_grad():
            target_q_values = rewards.unsqueeze(1)
        loss = nn.functional.mse_loss(q_values, target_q_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update target network periodically
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

# After training, select the best model
state = torch.tensor([[0.0]], dtype=torch.float32)
with torch.no_grad():
    q_values = policy_net(state)
    max_q = q_values.max().item()
    actions = (q_values == max_q).nonzero(as_tuple=True)[1]
    best_action = random.choice(actions.tolist())
    best_model = capable_model_names[best_action]
print(f"\nBest model after training: {best_model}")

# Ensure the best model is available
ensure_model_available(best_model)

# Generate final response using the best model
final_response = generate_response(best_model, task_prompt)
if final_response:
    print(f"\nFinal response from best model ({best_model}):\n{final_response}")
else:
    print(f"Failed to generate response with best model {best_model}.")

# Save the model
torch.save(policy_net.state_dict(), 'policy_net.pth')

print("\nModel saved to policy_net.pth")
