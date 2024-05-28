import argparse
import asyncio
import json
import logging
import os
import queue
import shlex
import sys
import threading
from argparse import Namespace
from datetime import datetime
from typing import List
from urllib.parse import urljoin

import aiohttp
import matplotlib.pyplot as plt
import requests
from rich import print
from rich.console import Console
from tqdm import tqdm

from config import Endpoint, FuzzerConfig, Request, from_json


def make_curl(method, url, headers=None, **request_kwargs):
    """
    Generate a curl command based on the request parameters.

    Parameters:
    - method: HTTP method (e.g., 'GET', 'POST')
    - url: Request URL
    - headers: Dictionary of HTTP headers
    - request_kwargs: Additional request parameters (e.g., json, data for the body)

    Returns:
    - A string that represents the curl command.
    """
    if headers is None:
        headers = {}
    curl_cmd = f"curl -X {method} {shlex.quote(url)}"

    if "json" in request_kwargs:
        data_str = json.dumps(request_kwargs["json"])
        curl_cmd += f" -d {shlex.quote(data_str)}"
    elif "data" in request_kwargs:
        data_str = request_kwargs["data"]
        if isinstance(data_str, dict):
            data_str = "&".join(f"{k}={v}" for k, v in data_str.items())
        # curl_cmd += f" -d {shlex.quote(data_str)}"  change this to use ? and & for query params
        curl_cmd += f"?{data_str}"
    for header, value in headers.items():
        curl_cmd += f" -H {shlex.quote(f'{header}: {value}')}"

    return curl_cmd


def log_writer(log_folder):
    while True:
        log_data = log_queue.get()
        if log_data is None:
            break
        endpoint, msg, filename = log_data
        log_file_path = os.path.join(log_folder, f"{endpoint.method}_{endpoint.path}")
        with open(os.path.join(log_file_path, filename), "a") as file:
            file.write(json.dumps(msg, indent=1) + "\n" + "------" * 20 + "\n")


def test_connection(fuzzer_config: FuzzerConfig):
    """Test the connection to the server."""
    logging.info("Testing connection to the server...")
    try:
        requests.get(fuzzer_config.base_url, headers=fuzzer_config.custom_header)
    except requests.exceptions.RequestException as e:
        terminal_logger.exception(e)
        terminal_logger.error("\n\n\n\nServer not found, exiting...")
        os._exit(1)


def test_endpoint(path: str):
    """Test a single endpoint."""
    logging.info(f"Testing endpoint: {path}")
    response = requests.get(path)
    if response.status_code == 404:
        logging.error("Endpoint not found")
        raise ValueError("Endpoint not found")
    logging.info("Endpoint tested successfully")


# history = {}
async def fetch(
    session: aiohttp.ClientSession,
    path: str,
    req: Request,
    semaphore: asyncio.Semaphore,
    progress_bar: tqdm,
    endpoint: Endpoint,
):
    req.headers["Content-Type"] = (
        "application/json"
        if endpoint.argument_type == "json" or endpoint.argument_type == "template"
        else "application/x-www-form-urlencoded"
    )
    request_kwargs = (
        {"json": req.data}
        if endpoint.argument_type == "json" or endpoint.argument_type == "template"
        else {"params": req.data}
    )
    async with semaphore:
        async with session.request(
            method=endpoint.method, url=path, headers=req.headers, **request_kwargs
        ) as response:
            curl = make_curl(
                endpoint.method, path, req.headers, data=req.data, **request_kwargs
            )
            req.actual_code = response.status
            req.text_response = await response.text()
            req.curl = curl
            req.datetime = str(datetime.now())
            progress_bar.update(1)


async def run(path: str, reqs: List[Request], endpoint: Endpoint):
    max_concurrent_requests = 10
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        progress_bar = tqdm(total=len(reqs))
        logging.info(f"Testing {len(reqs)} requests")
        tasks = [
            fetch(session, path, req, semaphore, progress_bar, endpoint) for req in reqs
        ]
        await asyncio.gather(*tasks)
        logging.info("All requests tested")
        progress_bar.close()


def find_causative_failure(requests: List[Request]):
    causative_failures = []
    for i, request in enumerate(requests):
        if request.is_error and i > 1:
            all_previous_successful = all(
                req.actual_code in req.expected_codes for req in requests[:i]
            )
            all_subsequent_failed = all(
                req.is_error for req in requests[i + 1 :]
            )
            if all_subsequent_failed and all_previous_successful:
                causative_failures.append(request)
                break  # ou continuar, dependendo se está procurando pela primeira instância ou todas as instâncias
    return causative_failures


def handle_endpoints(fuzzer_config: FuzzerConfig, log_folder: str):
    test_connection(fuzzer_config)

    for endpoint in fuzzer_config.endpoints:
        os.makedirs(
            os.path.join(log_folder, f"{endpoint.method}_{endpoint.path}"),
            exist_ok=True,
        )
        logging.info(f"Testing endpoint: {endpoint.path}")
        path = urljoin(fuzzer_config.base_url, endpoint.path)
        print(path)
        endpoint.pretty_print()
        try:
            test_endpoint(path)
        except ValueError:
            terminal_logger.error(f"Endpoint {endpoint.path} not found, continuing...")
            continue
        logging.info("Generating inputs...")
        request_data = endpoint.generate_inputs()
        if request_data is None:
            logging.error("No inputs generated")
            continue
        logging.info("Inputs generated successfully")
        asyncio.run(run(path, request_data, endpoint))
        request_data = sorted(request_data, key=lambda x: x.datetime)
        errors = 0
        for req in request_data:
            if req.actual_code not in req.expected_codes:
                errors += 1
                req.is_error = True
                error = {
                    "status_code": req.actual_code,
                    "expected_code": req.expected_codes,
                    "data": req.data,
                    "headers": req.headers,
                    "method": endpoint.method,
                    "path": path,
                    "curl": req.curl,
                    "response": req.text_response,
                }
                if req.is_valid:
                    log_queue.put((endpoint, error, "valid_error.log"))
                else:
                    log_queue.put((endpoint, error, "invalid_error.log"))
            else:
                if args.verbose:
                    log_queue.put((endpoint, {"req": req.__dict__}, "ok.log"))
        plt.plot(
            [req.actual_code for req in request_data if req.actual_code is not None]
        )
        plt.figure(figsize=(10, 5))
        plt.plot([req.expected_codes for req in request_data if req.expected_codes is not None])
        plt.ylabel("Status Code")
        plt.xticks(rotation=50)
        plt.title(f"Endpoint: {endpoint.method} {endpoint.path}")
        plt.tight_layout()
        plt.style.use("bmh")
        if args.show_graphs:
            plt.show()
        os.makedirs(os.path.join(log_folder, "graphs"), exist_ok=True)
        plt.savefig(
            os.path.join(log_folder, "graphs", f"{endpoint.method}_{endpoint.path}.png")
        )
        plt.close()

        causative_failures = find_causative_failure(request_data)
        if causative_failures:
            print(f"[bold red]Causative Failure: {causative_failures[0]}")

        print(f"[bold green]Total errors: {errors}")
        print(f"[bold green]Total inputs: {len(request_data)}")
        logging.info(f"Total errors: {errors}")
        logging.info(
            f"Endpoint log location {os.path.join(log_folder, f'{endpoint.method}_{endpoint.path}')}"
        )


def clear_screen():
    if os.name == "posix":
        os.system("clear")
    else:
        os.system("cls")


def print_ui(console: Console, endpoint: str):
    # Clear the console
    console.clear()
    columns, rows = os.get_terminal_size()
    start_text = "REST API Fuzzer".center(columns, " ")
    print(f"[bold red]{start_text}")
    print("[bold green]" + columns * "\u2500")
    console.print(f"Current endpoint: {endpoint}")


def get_args() -> Namespace:
    parser = argparse.ArgumentParser(description="REST API Fuzzer")
    parser.add_argument(
        "-c", "--config", help="Configuration file", required=True, type=str
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output directory (default: current working directory)",
        default=os.getcwd(),
        type=str,
    )
    parser.add_argument(
        "-v", "--verbose", help="Activate verbose mode", action="store_true"
    )
    parser.add_argument("-g", "--show-graphs", help="Show graphs", action="store_true")
    return parser.parse_args()


def handle_file(args) -> FuzzerConfig:
    try:
        logging.info("Abrindo o arquivo de configuração...")
        with open(args.config, "r") as file:
            json_data = json.load(file)
        logging.info("Arquivo aberto com sucesso")
    except FileNotFoundError as e:
        logging.exception(f"File error: {e}")
        sys.exit(1)

    try:
        logging.info("Validando configuração...")
        fuzzer_config = from_json(json_data)
        logging.info("Arquivo validado com sucesso")
    except ValueError as e:
        logging.exception(f"Error loading config: {e}")
        sys.exit(1)
    return fuzzer_config


def main():
    # clear_screen()
    logging_thread = threading.Thread(target=log_writer, args=(log_folder,))
    logging_thread.start()
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        filename=os.path.join(args.output, fuzzer_log_file),
        format="%(asctime)s - %(levelname)s - %(message)s",
        style="%",
    )
    print("logs at " + os.path.join(args.output, fuzzer_log_file))
    fuzzer_config = handle_file(args)
    logging.log(logging.INFO, fuzzer_config)
    handle_endpoints(fuzzer_config, log_folder)
    log_queue.put(None)
    logging_thread.join()


if __name__ == "__main__":
    log_queue = queue.Queue()
    try:
        args = get_args()
        log_folder = os.path.join(args.output, f"fuzzer{os.getpid()}")
        terminal_logger = logging.getLogger("terminal_logger")
        terminal_logger.setLevel(logging.ERROR)
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        terminal_logger.addHandler(stream_handler)
        fuzzer_log_file = os.path.join(log_folder, "fuzzer.log")
        main()
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting...")
        log_queue.put(None)
        sys.exit(1)
