# Copyright 2026 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
from json import JSONDecodeError
from tokenize import TokenError
import atheris
with atheris.instrument_imports(include=['cirq']):
    import cirq


def TestOneInput(data):
    try:
        decoded_data = data.decode('utf-8')
    except UnicodeDecodeError:
        return

    try:
        cirq.read_json(json_text=decoded_data)
    except (
        KeyError,
        AttributeError,
        AssertionError,
        TypeError,
        ValueError,
        TokenError,
        JSONDecodeError,
    ):
        # Expected errors for invalid JSON or invalid Cirq objects
        pass
    except Exception as e:
        # We might want to catch other specific valid errors if cirq raises them
        # For now, let other exceptions crash to find bugs
        raise e


def main():
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
