import copy
import json
import logging
import string
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, LiteralString, Optional, Tuple, Union

from data_generator import (
    ALPHANUMERIC_ALPHABET,
    DEFAULT_ALPHABET,
    MALFORMED_DATA,
    FloatGenerator,
    Input,
    IntGenerator,
    StringGenerator,
)
from rich import print

# TODO: make better logging


class Request:
    def __init__(
        self,
        data,
        expected_codes,
        is_valid,
        actual_code=None,
        method=None,
        url=None,
        headers=None,
    ):
        self.data = data
        self.expected_codes: List[int] = expected_codes
        self.method = method
        self.url = url
        self.headers: Dict = headers if headers else {}
        self.actual_code: int | None = actual_code
        self.text_response: str | None = None
        self.curl: str | None = None
        self.datetime: str
        self.is_valid: bool | None = is_valid
        self.is_error = False

    def __str__(self):
        return f"Request(data={self.data}, expected_code={self.expected_codes}, actual_code={self.actual_code}, method={self.method}, url={self.url}, headers={self.headers})"

    def __repr__(self):
        return f"Request(data={self.data}, expected_code={self.expected_codes}, actual_code={self.actual_code}, method={self.method}, url={self.url}, headers={self.headers})"


@dataclass
class Argument:
    name: str
    type: str
    max_acceptable_size: Optional[int] = None
    min_acceptable_size: Optional[int] = None
    precision: Optional[int] = None
    alphabet: Union[str, LiteralString, None] = None


def _divide_template_inputs(inputs: List[Input]):
    divided_inputs = []
    for element in zip(*inputs):
        divided_inputs.append(element)
    return divided_inputs


def _copy_from_template(template: Dict[str, Any]) -> Dict[str, Any]:
    del_keys = []
    for key, _ in template.items():
        if key.endswith("-FUZZABLE"):
            del_keys.append(key)
    for key in del_keys:
        del template[key]
    return template


@dataclass
class Endpoint:
    path: str
    method: str
    argument_type: str
    expected_response_for_valid: List[str]
    expected_response_for_invalid: List[str]
    n_inputs: int
    arguments: Optional[List[Argument]] = field(default_factory=list)
    template_filename: Optional[str] = None
    custom_header: Optional[Dict[str, str]] = None

    def pretty_print(self):
        print("[deep_sky_blue4 bold]Path: [/deep_sky_blue4 bold]", self.path)
        print("[deep_sky_blue4 bold]Method: [/deep_sky_blue4 bold]", self.method)
        print(
            "[deep_sky_blue4 bold]Argument Type: [/deep_sky_blue4 bold]",
            self.argument_type,
        )
        print(
            "[deep_sky_blue4 bold]Expected Response for Valid: [/deep_sky_blue4 bold]",
            self.expected_response_for_valid,
        )
        print(
            "[deep_sky_blue4 bold]Expected Response for Invalid: [/deep_sky_blue4 bold]",
            self.expected_response_for_invalid,
        )
        print(
            "[deep_sky_blue4 bold]Number of Inputs: [/deep_sky_blue4 bold]",
            self.n_inputs,
        )
        if self.arguments:
            print("[deep_sky_blue4 bold]Arguments: [/deep_sky_blue4 bold]")
            for arg in self.arguments:
                print(
                    "  [deep_sky_blue4 bold]Name: " f"[red bold]{arg.name}",
                )
                print(
                    "    [deep_sky_blue4 bold]Type:",
                    f"[red bold]{arg.type}",
                )
                print(
                    "    [deep_sky_blue4 bold]Max Acceptable Size: [/deep_sky_blue4 bold]",
                    arg.max_acceptable_size,
                )
                print(
                    "    [deep_sky_blue4 bold]Min Acceptable Size: [/deep_sky_blue4 bold]",
                    arg.min_acceptable_size,
                )
                if arg.precision:
                    print(
                        "    [deep_sky_blue4 bold]Precision: [/deep_sky_blue4 bold]",
                        arg.precision,
                    )
                if arg.alphabet:
                    print(
                        "    [deep_sky_blue4 bold]Alphabet: [/deep_sky_blue4 bold]",
                        arg.alphabet,
                    )
        if self.template_filename:
            print(
                "    [deep_sky_blue4 bold]Template Filename: [/deep_sky_blue4 bold]",
                self.template_filename,
            )

    def __post_init__(self):
        self._validate_template()

    def _validate_template(self):
        if self.argument_type == "template":
            self._validate_template_endpoint()
        else:
            self._validate_regular_endpoint()

    def _validate_template_endpoint(self):
        if self.arguments:
            logging.exception(
                ValueError("Arguments must not be provided for template endpoints.")
            )
        if not self.template_filename:
            logging.exception(
                ValueError("Template filename must be provided for template endpoints.")
            )

    def _validate_regular_endpoint(self):
        # if not self.arguments:
        #     logging.exception(
        #         ValueError("Arguments must be provided for non-template endpoints.")
        #     )
        if self.template_filename:
            logging.exception(
                ValueError(
                    "Template filename must not be provided for non-template endpoints."
                )
            )

    @staticmethod
    def _open_template_file(filename: str) -> Optional[dict[str, Any]]:
        try:
            with open(filename, "r") as file:
                return json.load(file)
        except Exception as e:
            logging.exception(f"Error opening template file: {e}")
            return None

    def _get_fuzzable_template_keys(self, template_json: dict[str, Any]):
        for k, v in template_json.items():
            if k.endswith("-FUZZABLE"):
                yield k, v
            elif isinstance(v, dict):
                for val in self._get_fuzzable_template_keys(v):
                    yield k, val

    def _generate_inputs_template(self, fuzzable_dict: dict[str, Any]) -> List[Input]:
        inputs = []
        for _, value in fuzzable_dict.items():
            if isinstance(value, str):
                fuzzable_parts = value.split(" ")
                fuzzable_type = fuzzable_parts[0]
                try:
                    fuzzable_min = int(fuzzable_parts[1])
                    fuzzable_max = int(fuzzable_parts[2])
                except ValueError:
                    fuzzable_min = None
                    fuzzable_max = None
                match fuzzable_type:
                    case "STR":
                        fuzzable_alphabet = fuzzable_parts[3]
                        if fuzzable_alphabet == "/default":
                            fuzzable_alphabet = string.printable
                        if fuzzable_alphabet == "/alphanumeric":
                            fuzzable_alphabet = string.ascii_letters + string.digits
                        if fuzzable_alphabet == "/malformed":
                            fuzzable_alphabet = MALFORMED_DATA
                        else:
                            fuzzable_alphabet = fuzzable_alphabet
                        generator = StringGenerator(
                            self.expected_response_for_valid,
                            self.expected_response_for_invalid,
                            self.n_inputs,
                            fuzzable_min,
                            fuzzable_max,
                            fuzzable_alphabet,
                        )
                        inputs.append(generator.generate())
                    case "INT":
                        generator = IntGenerator(
                            self.expected_response_for_valid,
                            self.expected_response_for_invalid,
                            fuzzable_min,
                            fuzzable_max,
                            self.n_inputs,
                        )
                        inputs.append(generator.generate())
                    case "FLOAT":
                        if len(fuzzable_parts) == 4:
                            fuzzable_precision = int(fuzzable_parts[3])
                        else:
                            fuzzable_precision = 2
                        generator = FloatGenerator(
                            fuzzable_min,
                            fuzzable_max,
                            self.expected_response_for_valid,
                            self.expected_response_for_invalid,
                            self.n_inputs,
                            fuzzable_precision,
                        )
                        inputs.append(generator.generate())
                    case _:
                        raise NotImplementedError
            elif isinstance(value, dict):
                inpts = self._generate_inputs_template(value)
                inputs.extend(inpts)

            elif isinstance(value, list):
                for e in value:
                    inpts = self._generate_inputs_template(e)
                    inputs.extend(inpts)
        return inputs

    def _insert_input_template_fuzzable_data(self, target, entry):
        for key, value in target.copy().items():
            if key.endswith("-FUZZABLE"):
                if isinstance(value, str):
                    del target[key]
                    target[key.strip("-FUZZABLE")] = entry
                    return 1
                if isinstance(value, dict):
                    self._insert_input_template_fuzzable_data(value, entry)
                    for k in value.keys():
                        if k.endswith("-FUZZABLE"):
                            return 1
                    target[key.strip("-FUZZABLE")] = target[key]
                    del target[key]
                    return 1
                if isinstance(value, list):
                    for idx, item in enumerate(value):
                        if (
                            self._insert_input_template_fuzzable_data(item, entry)
                            and idx != len(value) - 1
                        ):
                            return
                    target[key.strip("-FUZZABLE")] = target[key]
                    del target[key]

    @staticmethod
    def _get_correct_codes_and_is_valid(codes: List) -> Tuple[list[int], bool]:
        valid_codes = set()
        invalid_codes = set()
        numbers = []
        for code in codes:
            numbers.append(code[1])
            if code[1] == 0:
                for c in code[0]:
                    invalid_codes.add(int(c))
            else:
                for c in code[0]:
                    valid_codes.add(int(c))

        and_result = all(
            numbers
        )  # True significa que os códigos de retorno serão os válidos
        if and_result:
            return list(valid_codes), True
        return list(invalid_codes), False

    def _create_json_inputs(
        self,
        fuzzable_dict: Dict[str, Any],
        divided_inputs: List[Tuple[Input]],
        template: Dict[str, Any],
    ):
        request_data = []
        for entry in divided_inputs:
            fuzzed_inputs = copy.deepcopy(fuzzable_dict)
            fuzzed_json = _copy_from_template(copy.deepcopy(template))
            codes_request = []
            for input in entry:
                self._insert_input_template_fuzzable_data(fuzzed_inputs, input.data)
                codes_request.append((input.expected_code, input.is_valid))
            codes, is_valid = self._get_correct_codes_and_is_valid(codes_request)
            for key, value in fuzzed_inputs.items():
                fuzzed_json[key] = value
            request_data.append(
                Request(
                    fuzzed_json,
                    codes,
                    method=self.method,
                    url=self.path,
                    headers=self.custom_header,
                    is_valid=is_valid,
                )
            )
        return request_data

    def _generate_templates(self):
        if not self.template_filename:
            logging.exception(
                ValueError("Template filename must be provided for template endpoints.")
            )
            return None
        template_json = self._open_template_file(self.template_filename)
        if not template_json:
            return None
        fuzzable_dict = {}
        for key, value in self._get_fuzzable_template_keys(template_json):
            fuzzable_dict[key] = value
        inputs = self._generate_inputs_template(fuzzable_dict)
        divided_inputs = _divide_template_inputs(inputs)
        json_inputs = self._create_json_inputs(
            fuzzable_dict, divided_inputs, template_json
        )
        return json_inputs

    def _generate_inputs_regular(self):
        if not self.arguments:
            # logging.exception(
            #     ValueError("Arguments must be provided for non-template endpoints.")
            # )
            # return None

            return [
                Request(
                    {},
                    self.expected_response_for_valid,
                    url=self.path,
                    method=self.method,
                    headers=self.custom_header,
                    is_valid=True,
                )
                for i in range(self.n_inputs)
            ]
        inputs = []
        for arg in self.arguments:
            if arg.type == "string":
                if arg.alphabet == "/default" or arg.alphabet is None:
                    arg.alphabet = DEFAULT_ALPHABET
                elif arg.alphabet == "/alphanumeric":
                    arg.alphabet = ALPHANUMERIC_ALPHABET
                elif arg.alphabet == "/malformed":
                    arg.alphabet = MALFORMED_DATA
                generator = StringGenerator(
                    self.expected_response_for_valid,
                    self.expected_response_for_invalid,
                    self.n_inputs,
                    arg.min_acceptable_size,
                    arg.max_acceptable_size,
                    arg.alphabet,
                )
                inputs.append(generator.generate())
            elif arg.type == "int":
                generator = IntGenerator(
                    self.expected_response_for_valid,
                    self.expected_response_for_invalid,
                    self.n_inputs,
                    arg.min_acceptable_size,
                    arg.max_acceptable_size,
                )
                inputs.append(generator.generate())
            elif arg.type == "float":
                if arg.precision is None:
                    arg.precision = 2
                else:
                    arg.precision = int(arg.precision)
                generator = FloatGenerator(
                    arg.min_acceptable_size,
                    arg.max_acceptable_size,
                    self.expected_response_for_valid,
                    self.expected_response_for_invalid,
                    self.n_inputs,
                    arg.precision,
                )
                inputs.append(generator.generate())
            else:
                raise NotImplementedError
        # generate requests
        request = []
        for group in zip(*inputs):
            request_data = {}
            for arg, input in zip(self.arguments, group):
                request_data[arg.name] = input.data
            codes, is_valid = (
                self._get_correct_codes_and_is_valid(  # todo remover codes
                    [(input.expected_code, input.is_valid) for input in group]
                )
            )
            request.append(
                Request(
                    request_data,
                    codes,
                    url=self.path,
                    method=self.method,
                    headers=self.custom_header,
                    is_valid=is_valid,
                )
            )
        return request

    def generate_inputs(self) -> Optional[List[Request]]:
        if self.argument_type == "template":
            return self._generate_templates()
        else:
            return self._generate_inputs_regular()


@dataclass
class FuzzerConfig:
    base_url: str
    endpoints: List[Endpoint] = field(default_factory=list)
    custom_header: Optional[Dict[str, str]] = None


def is_valid_argument(arg: Dict[str, Any]) -> bool:
    """Validates an argument dictionary."""
    required_keys = {"name", "type"}
    optional_keys = {
        "precision",
        "alphabet",
        "max_acceptable_size",
        "min_acceptable_size",
    }
    return required_keys.issubset(arg.keys()) and set(arg.keys()).issubset(
        required_keys | optional_keys
    )


def is_valid_endpoint(endpoint: Dict[str, Any]) -> bool:
    """Validates an endpoint dictionary with improved logic."""
    common_required_keys = {
        "path",
        "method",
        "argument_type",
        "expected_response_for_valid",
        "expected_response_for_invalid",
        "n_inputs",
    }
    template_required_keys = common_required_keys | {"template_filename"}
    if not common_required_keys.issubset(endpoint.keys()):
        return False
    if endpoint["argument_type"] == "template":
        if not template_required_keys.issubset(endpoint.keys()):
            return False
        if "arguments" in endpoint:
            return False
    else:
        if "arguments" not in endpoint or not all(
            isinstance(arg, dict) and is_valid_argument(arg)
            for arg in endpoint["arguments"]
        ):
            return False
    return True


def validate_json(json_data: Dict[str, Any]) -> bool:
    """Validates the entire JSON structure for the API fuzzer configuration."""
    required_keys = {"base_url", "endpoints"}
    if not required_keys.issubset(json_data.keys()):
        return False
    if not all(
        isinstance(endpoint, dict) and is_valid_endpoint(endpoint)
        for endpoint in json_data["endpoints"]
    ):
        return False
    return True


def from_json(json_data: Dict[str, Any]) -> FuzzerConfig:
    if not validate_json(json_data):
        logging.exception(
            ValueError("Invalid JSON structure for API fuzzer configuration.")
        )

    try:
        endpoints = [
            Endpoint(
                path=e["path"],
                method=e["method"],
                argument_type=e["argument_type"],
                expected_response_for_valid=e.get("expected_response_for_valid", []),
                expected_response_for_invalid=e.get(
                    "expected_response_for_invalid", []
                ),
                n_inputs=e["n_inputs"],
                arguments=(
                    [Argument(**arg) for arg in e.get("arguments", [])]
                    if e["argument_type"] != "template"
                    else None
                ),
                template_filename=e.get("template_filename"),
                custom_header=(e.get("custom_header") or {})
                | (json_data.get("custom_header") or {}),
            )
            for e in json_data["endpoints"]
        ]
        return FuzzerConfig(
            base_url=json_data["base_url"],
            endpoints=endpoints,
            custom_header=json_data.get("custom_header"),
        )
    except KeyError as e:
        logging.exception(ValueError(f"Missing key in endpoint data: {e}"))
        sys.exit(0)
