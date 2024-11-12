import os
import time
import json

import openai
from openai import OpenAI

from config import config

os.environ["OPENAI_API_KEY"] = config.credentials.openai_key
client = OpenAI()

# Constants
MODEL = "gpt-4o"
TEMPERATURE = 0.8
# MAX_TOKENS = 150
SLEEP_INTERVAL = 1  # To avoid hitting rate limits

# Define categories with instructions for generating prompts
CATEGORIES = {
    "prosocial": "Generate a list of prompts that encourage prosocial, empathetic, or altruistic responses.",
    "self_centered": "Generate a list of prompts that encourage self-centered or self-interest-based responses.",
    "neutral": "Generate a list of prompts that encourage neutral or balanced responses without strong prosocial or self-centered bias."
}

# Define JSON schema for response format
def get_json_schema(category):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{category}_template_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "description": "The behavior category for the generated prompts",
                        "type": "string"
                    },
                    "instruction": {
                        "description": "The instruction provided to generate the prompts",
                        "type": "string"
                    },
                    "generated_templates": {
                        "description": "A list of generated prompts based on the input instruction",
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["category", "instruction", "generated_templates"],
                "additionalProperties": False
            }
        }
    }

# Generate prompts for a single category
def generate_prompts_for_category(category, instruction):
    print(f"Generating prompts for category '{category}'...")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a chatbot that generates prompts for a behavioral study. Please generate a list of prompts based on the following instruction."
                },
                {
                    "role": "user",
                    "content": instruction
                }
            ],
            response_format=get_json_schema(category),
            temperature=TEMPERATURE,
        )

        # Extract and parse the structured response
        response_content = response.choices[0].message.content
        return json.loads(response_content).get("generated_templates", [])

    except Exception as e:
        print(f"Error generating prompts for category '{category}': {e}")
        raise e
        # return []

# Generate the full dataset across all categories
def generate_template_dataset():
    template_dataset = {"templates": []}

    for category, instruction in CATEGORIES.items():
        generated_templates = generate_prompts_for_category(category, instruction)
        template_dataset["templates"].append({
            "category": category,
            "instruction": instruction,
            "generated_templates": generated_templates
        })
        time.sleep(SLEEP_INTERVAL)

    return template_dataset

def save_dataset_to_json(dataset, filename="generated_prompts_dataset.json"):
    path = os.path.join(config.paths.data, filename)
    try:
        with open(path, "w") as f:
            json.dump(dataset, f, indent=4)
        print(f"Dataset saved to '{filename}'")
    except IOError as e:
        print(f"Error saving dataset to file: {e}")

def main():
    dataset = generate_template_dataset()
    save_dataset_to_json(dataset)

if __name__ == "__main__":
    main()
