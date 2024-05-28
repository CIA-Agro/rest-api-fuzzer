import random
import string
from typing import List, LiteralString, Optional

DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MALFORMED_DATA = string.ascii_letters + string.digits + "æøåÆØÅ"
PRINTABLE_ALPHABET = string.printable
ALPHANUMERIC_ALPHABET = string.ascii_letters + string.digits
ALL_ALPHABETS = (
        DEFAULT_ALPHABET + MALFORMED_DATA + ALPHANUMERIC_ALPHABET + PRINTABLE_ALPHABET
)
NON_NUMERIC_ALPHABET = string.ascii_letters + string.punctuation + "æøåÆØÅ" + " "


class Input:
    """Represents a single input data for an endpoint."""

    def __init__(self, data, expected_code, is_valid):
        self.data = data
        self.expected_code = expected_code
        self.is_valid = is_valid

    def __repr__(self):
        return f"Input(data={self.data}, expected_code={self.expected_code}, is_valid={self.is_valid})"


class BaseDataGenerator:
    """Base class for data generation with common attributes."""

    def __init__(
            self,
            exp_code_valid,
            exp_code_invalid,
            n,
            min_size: Optional[int] = None,
            max_size: Optional[int] = None,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.exp_code_valid = exp_code_valid
        self.exp_code_invalid = exp_code_invalid
        self.n = n

    def _generate_valid_input_data(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _generate_invalid_input_data(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def generate(self) -> List[Input]:
        valid = self._generate_valid_inputs()
        invalid = self._generate_invalid_inputs()
        first_part = valid[: int(0.35 * len(valid))]
        second_part = valid[int(0.35 * len(valid)):] + invalid
        random.shuffle(second_part)
        return first_part + second_part

    def _generate_valid_inputs(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _generate_invalid_inputs(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


class StringGenerator(BaseDataGenerator):
    """Generates random strings within specified size and alphabet."""

    def __init__(
            self,
            exp_code_valid,
            exp_code_invalid,
            n,
            min_size: Optional[int] = None,
            max_size: Optional[int] = None,
            alphabet: LiteralString | str = string.ascii_letters,
    ):
        super().__init__(exp_code_valid, exp_code_invalid, n, min_size, max_size)
        self.alphabet = alphabet
        self.invalid_alphabet = "".join(set(ALL_ALPHABETS) - set(self.alphabet))

    def _generate_valid_input_data(self):
        return "".join(
            random.choice(self.alphabet)
            for _ in range(
                random.randint(int(self.min_size), int(self.max_size))
                if self.min_size is not None and self.max_size is not None
                else random.randint(0, 1000)
            )
        )

    def _generate_invalid_input_data(self) -> str:
        invalid = []
        if self.min_size is not None and self.max_size is not None:
            valid_alphabet_invalid_size = "".join(  # valid alphabet, invalid size
                random.choice(self.alphabet)
                for _ in range(
                    random.randint(int(self.max_size + 1), int(self.max_size * 2))
                )
            )
            invalid.append(valid_alphabet_invalid_size)
        invalid_alphabet_valid_size = "".join(  # invalid alphabet, valid size
            random.choice(self.invalid_alphabet)
            for _ in range(
                random.randint(int(self.min_size), int(self.max_size))
                if self.min_size is not None and self.max_size is not None
                else random.randint(0, 1000)
            )
        )
        invalid_alphabet_invalid_size = "".join(  # invalid alphabet, invalid size
            random.choice(self.invalid_alphabet)
            for _ in range(
                random.randint(int(self.max_size + 1), int(self.max_size * 2))
                if self.min_size is not None and self.max_size is not None
                else random.randint(0, 1000)
            )
        )
        invalid.append(invalid_alphabet_valid_size)
        invalid.append(invalid_alphabet_invalid_size)
        invalid_string = random.choice(invalid)
        return invalid_string

    def _generate_valid_inputs(self):
        inputs = []
        for _ in range(int(self.n / 2)):
            inputs.append(
                Input(self._generate_valid_input_data(), self.exp_code_valid, True)
            )
        return inputs

    def _generate_invalid_inputs(self) -> List[Input]:
        inputs = []
        for _ in range(int(self.n / 2)):
            inputs.append(
                Input(self._generate_invalid_input_data(), self.exp_code_invalid, False)
            )
        return inputs


class FloatGenerator(BaseDataGenerator):
    """Generates random floats within specified range and precision."""

    def __init__(
            self, min_size, max_size, exp_code_valid, exp_code_invalid, n, precision=2
    ):
        super().__init__(min_size, max_size, exp_code_valid, exp_code_invalid, n)
        self.precision = precision

    def _generate_valid(self):
        value = (
            random.uniform(self.min_size, self.max_size)
            if self.min_size is not None and self.max_size is not None
            else random.uniform(-10000000000, 10000000000)
        )
        return round(value, self.precision)

    def _generate_invalid(self):
        # get from int later
        invalid = []
        if self.min_size is not None and self.max_size is not None:
            value = random.uniform(self.max_size, self.max_size * 2)
            invalid.append(round(value, self.precision))
        invalid_string = "".join(
            random.choice(NON_NUMERIC_ALPHABET) for _ in range(random.randint(0, 2000))
        )
        invalid.append(invalid_string)
        return random.choice(invalid)

    def _generate_valid_inputs(self):
        inputs = []
        for _ in range(int(self.n / 2)):
            inputs.append(Input(self._generate_valid(), self.exp_code_valid, True))
        return inputs

    def _generate_invalid_inputs(self):
        inputs = []
        for _ in range(int(self.n / 2)):
            inputs.append(Input(self._generate_invalid(), self.exp_code_invalid, False))
        return inputs


class IntGenerator(BaseDataGenerator):
    """Generates random integers within specified range."""

    def __init__(
            self,
            exp_code_valid,
            exp_code_invalid,
            n,
            min_size: Optional[int] = None,
            max_size: Optional[int] = None,
    ):
        super().__init__(exp_code_valid, exp_code_invalid, n, min_size, max_size)

    def _generate_valid_input_data(self):
        return (
            random.randint(int(self.min_size), int(self.max_size))
            if self.min_size is not None and self.max_size is not None
            else random.randint(-10000000000, 10000000000)
        )

    def _generate_invalid_input_data(self):
        invalid = []
        if self.min_size is not None and self.max_size is not None:
            value = random.randint(int(self.max_size), int(self.max_size * 2))
            invalid.append(value)
        invalid_string = "".join(
            random.choice(NON_NUMERIC_ALPHABET) for _ in range(random.randint(0, 2000))
        )
        invalid.append(invalid_string)
        return random.choice(invalid)

    def _generate_valid_inputs(self):
        inputs = []
        for _ in range(int(self.n / 2)):
            inputs.append(
                Input(self._generate_valid_input_data(), self.exp_code_valid, True)
            )
        return inputs

    def _generate_invalid_inputs(self):
        inputs = []
        for _ in range(int(self.n / 2)):
            inputs.append(
                Input(self._generate_invalid_input_data(), self.exp_code_invalid, False)
            )
        return inputs
