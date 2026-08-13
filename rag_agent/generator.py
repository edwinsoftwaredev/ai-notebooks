import inspect

import dask.dataframe as ddf
import faiss
import huggingface_hub
import pandas as pd
import torch
from kaggle_secrets import UserSecretsClient  # pyright: ignore
from transformers import (
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from rag_agent import agent_tools
from rag_agent.enums import DATASET_PATH, IDX_PATH

response_template = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<start_of_turn>model\n",
    "fields": {
        "thinking": {
            "open": "<agent_trace>",
            "close": "</agent_trace>",
            "content": "text",
            "repeats": True,
            "join": "\n",
        },
        "tool_calls": {
            "open": "<tool_call>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": "{content}"},
        },
        "content": {
            "close": "<end_of_turn>",
            "content": "text",
        },
    },
}


def parse_tool(tool):
    return f"""<tool>
        <name>{tool.__name__}</name>
        <description>{inspect.getdoc(tool)}</description>
        <signature>{inspect.signature(tool)}</signature>
    </tool>\n"""


def get_pipeline():
    user_secrets = UserSecretsClient()
    hf_gemma_token = user_secrets.get_secret("hf_gemma_3_4b_token")
    huggingface_hub.login(hf_gemma_token, skip_if_logged_in=True)
    model_id = "google/gemma-3-4b-it"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # must be gpu compatible
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    gemma_tokenizer = AutoTokenizer.from_pretrained(model_id)
    gemma_tokenizer.response_template = response_template
    processor.tokenizer.response_template = response_template

    pipe = pipeline(
        "image-text-to-text",
        model=model,
        processor=processor,
        tokenizer=gemma_tokenizer,
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

    tools.append(
        parse_tool(
            agent_tools.wikipedia_articles_retriever_func(
                tokenizer, embedder, index, index_metadata, df
            )
        )
    )

    tools = "".join(tools)

    tool_hint = '{"name":"function_name","arguments":{"argument":"value"}}'

    system_msg = f"""You are an AI agent.
    Accuracy and factual correctness are critical to your work.
    Therefore you must always adhere to the following guidelines.

    GUIDELINES
        ## GENERAL BEHAVIOR
        - When talking to yourself, generate agent traces.
          An agent trace must be enclosed in <agent_trace> tags and should be
          one or two sentences long, describing your reasoning.
        - Your very first output must be an <agent_trace> block.
        - Use the agent traces to explicitly drive your actions.
        - Do not use agent traces as a source of truth or as evidence in your final answer.
        - Once the available information is sufficient to complete the task, proceed with giving the answer
          and stop generating.
        - Ask the user for missing details only if they are critical to completing the task
          and cannot be inferred or defaulted, and do not use any tool.

        ## TOOLS USAGE
        - Do not call a tool if the available information is sufficient to complete the task without it.
        - Always call the appropriate tools.
        - Before calling a tool, refer to the agent traces to determine how it should be used or the justification of its usage.
        - A tool call must be enclosed in <tool_call> tags and use following the format:
          {tool_hint}
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

    AVAILABLE TOOLS
        {tools}

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
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": user_query}]}
        )

        input_ids = gemma_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )["input_ids"].to(model.device)

        outputs = model.generate(  # pyright: ignore
            input_ids, max_new_tokens=1024
        )[0, input_ids.shape[1] :]

        out_text = gemma_tokenizer.decode(outputs)

        print(out_text)
        print("--------------------------")

        result = pipe(text=messages, max_new_tokens=1024)  # pyright: ignore

        # print(out_text)
        # message = gemma_tokenizer.parse_response(out_text, prefix=input_ids[0])
        # print(message)

        messages.append(result[0]["generated_text"][-1])

        return messages

    return pipe_closure
