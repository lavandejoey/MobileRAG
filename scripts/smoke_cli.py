# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.13.0
"""

import argparse
import json

import requests


def _process_sse_chunk(decoded_chunk: str, full_answer: list, evidence_list: list):
    for line in decoded_chunk.splitlines():
        if line.startswith("data:"):
            json_str = line[len("data:") :].strip()
            if json_str:
                try:
                    data = json.loads(json_str)
                    if "answer" in data:
                        answer_part = data.get("answer", "")
                        print(answer_part, end="", flush=True)
                        full_answer.append(answer_part)
                    elif "evidence" in data:
                        evidence_list.append(data["evidence"])
                except json.JSONDecodeError:
                    pass  # Ignore invalid JSON


def chat_cli(query: str, session_id: str):
    url = "http://localhost:8000/api/chat"
    payload = {"query": query, "session_id": session_id}
    headers = {"Content-Type": "application/json"}

    print(f"Sending query: '{query}' to session '{session_id}'")
    try:
        with requests.post(url, json=payload, headers=headers, stream=True) as response:
            response.raise_for_status()  # Raise an exception for HTTP errors
            print("Assistant: ", end="")
            full_answer = []
            evidence_list = []
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    decoded_chunk = chunk.decode("utf-8")
                    _process_sse_chunk(decoded_chunk, full_answer, evidence_list)

            print("\n")  # Newline after stream ends

            if evidence_list:
                print("\n--- Evidence ---")
                for i, evidence in enumerate(evidence_list, 1):
                    print(f"{i}. {evidence}")
                print("----------------\n")

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test CLI for MobileRAG API.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Send a chat message.")
    chat_parser.add_argument("--q", type=str, required=True, help="Query string.")
    chat_parser.add_argument("--session", type=str, default="default_session", help="Session ID.")

    args = parser.parse_args()

    if args.command == "chat":
        chat_cli(args.q, args.session)
    else:
        parser.print_help()
