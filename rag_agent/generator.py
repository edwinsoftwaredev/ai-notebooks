import json

import dask.dataframe as ddf
import faiss
import kagglehub  # pyright: ignore
import pandas as pd
import torch
from kaggle_secrets import UserSecretsClient  # pyright: ignore
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from rag_agent import agent_tools
from rag_agent.enums import DATASET_PATH, IDX_PATH

MODEL_PATH = kagglehub.model_download("google/gemma-4/transformers/gemma-4-12b-it")


def parse_tool(tool, tool_map):
    tool_map[tool.__name__] = tool
    return tool


def get_pipeline():
    user_secrets = UserSecretsClient()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # must be gpu compatible
        bnb_4bit_quant_type="nf4",
    )

    with open("/kaggle/working/ai-notebooks/rag_agent/tokenizer_config.json") as f:
        config = json.load(f)

    response_template = config["response_template"]
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.response_template = response_template
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    embedder = AutoModel.from_pretrained("facebook/contriever")
    embedder.to("cuda:0")
    embedder.eval()

    index = faiss.read_index(f"{IDX_PATH}/retreiver.index")
    index_metadata = pd.read_parquet(f"{IDX_PATH}/metadata.parquet")

    df = ddf.read_parquet(
        f"{DATASET_PATH}/nq-dataset", index="id", calculate_divisions=True
    )

    tools = []
    tool_map = {}

    tools.append(
        parse_tool(
            agent_tools.wikipedia_articles_retriever_func(
                tokenizer, embedder, index, index_metadata, df
            ),
            tool_map,
        )
    )

    system_msg = """You are an AI agent.
    Accuracy and factual correctness are critical to your work.
    Therefore you must always adhere to the following guidelines.

    GUIDELINES
        ## GENERAL BEHAVIOR
        - Once the available information is sufficient to complete the task, proceed with giving the answer
          and stop generating.
        - Ask the user for missing details only if they are critical to completing the task
          and cannot be inferred or defaulted, and do not use any tool.

        ## TOOLS USAGE
        - Do not call a tool if the available information is sufficient to complete the task without it.
        - Before calling a tool, determine how it should be used or the justification of its usage.
        - Always call the appropriate tools.
        - Do not generate tool results.
        - The controller will execute the tool and provide its result in a subsequent interaction.

        ## FACTUAL CORRECTNESS AND ACCURACY
        - Prioritize backing up your knowledge with evidence.
        - Base your answer on the available information.
        - Use the agent traces to verify that the final answer is consistent with
          the actions taken and information obtained.
        - Do not make unfounded claims.

        ## INFORMATION RETRIEVAL
        - Queries must always be relevant to the original user's query.
        - Queries generated from previous retrieval results must remain relevant to the original user's query.
        - Do not generate queries if the available information is sufficient to complete the task.

        ## UNCERTAINTY & AMBIGUITY
        - Ask for clarification when the task or the inputs are ambiguous.
        - State uncertainty in your response when the available information is insufficient to provide a grounded answer.

    """

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_msg,
                }
            ],
        },
    ]

    def pipe_closure(user_query: str):
        with torch.inference_mode():
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": user_query}]}
            )

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
                tools=tools,
            )
            inputs = processor(text=text, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            # Generate output
            outputs = model.generate(**inputs, max_new_tokens=1024)  # pyright: ignore
            response = processor.decode(
                outputs[0][input_len:], skip_special_tokens=False
            )

            # Parse thinking
            response = processor.parse_response(response, prefix=inputs["input_ids"])

            messages.append(response)

            while "tool_calls" in response:
                tools_out = []

                for tool in response["tool_calls"]:
                    if tool["function"]["name"] not in tool_map:
                        continue

                    tool, args = (
                        tool_map[tool["function"]["name"]],
                        tool["function"]["arguments"],
                    )
                    tools_out.append(tool(**args))

                messages.append({"role": "tool", "content": str(tools_out)})

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                    tools=tools,
                )
                inputs = processor(text=text, return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[-1]

                # Generate output
                outputs = model.generate(**inputs, max_new_tokens=1024)  # pyright: ignore
                response = processor.decode(
                    outputs[0][input_len:], skip_special_tokens=False
                )

                # Parse thinking
                response = processor.parse_response(
                    response, prefix=inputs["input_ids"]
                )
                messages.append(response)

        return messages[-1]

    return pipe_closure
