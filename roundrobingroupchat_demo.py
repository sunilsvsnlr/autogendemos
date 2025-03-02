import asyncio
import json
import os
import pandas as pd
import logging
from datetime import datetime, timedelta

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent, UserProxyAgent
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, ChatMessage, AgentEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool
from typing import AsyncGenerator, Sequence

async def main() -> None:
    metadata_file = "C:/projects/genai/metadata/metadata.json"
    merged_data_file = "C:/projects/genai/merged/merged_data.json"  # Merged data file for analysis

    model_client = OpenAIChatCompletionClient(
        model="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/qwen2.5-coder-3b-instruct-q4_0.gguf",
        seed=42,
        api_key="lm-studio",
        base_url="http://127.0.0.1:1234/v1",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
        },
    )

    metadata = json.load(open(metadata_file))
    systemPrompt = f"""
                        Generate a Python script to analyze project data based on the following metadata:
                        {json.dumps(metadata, indent=2)}

                        The script should:
                        - Load data from the JSON file: "{merged_data_file}".
                        - Always answer the question.
                        - Based on the question if it relates to date, ensure proper date handling (convert date columns to datetime before calculations).
                        - Strictly use columns specified in meta data only for processing.
                        - Don't use any depreciated libraries.
                        - Use pandas for data analysis.
                        - Return a structured result.

                        Ensure the script is properly formatted and outputs relevant insights.
                        If execution fails, refine the code and try again
                        After execution without any errors, TERMINATE.
                        """
    
    assistant = AssistantAgent(
        name="assistant",
        system_message=systemPrompt,
        model_client=model_client,
    )

    code_executor = CodeExecutorAgent(
        name="code_executor",
        code_executor=LocalCommandLineCodeExecutor(work_dir="coding"),
    )

    # The termination condition is a combination of text termination and max message termination, either of which will cause the chat to terminate.
    termination = TextMentionTermination("TERMINATE")

    # The group chat will alternate between the assistant and the code executor.
    group_chat = RoundRobinGroupChat([assistant, code_executor], termination_condition=termination)

    # `run_stream` returns an async generator to stream the intermediate messages.
    stream = group_chat.run_stream(task="what is max project cost till now?")
    # `Console` is a simple UI to display the stream.
    await Console(stream)

asyncio.run(main())
