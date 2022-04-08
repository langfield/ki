import pprint
from abc import abstractmethod
from pathlib import Path
from dataclasses import dataclass

from beartype import beartype
from beartype.typing import Union, Type


class ExtantFile(type(Path())):
    """UNSAFE: Indicates that file *was* extant when it was resolved."""


@beartype
@dataclass
class MaybeType:
    def __post_init__(self):
        """Validate input."""
        if isinstance(self.value, self.type):
            msg = f"DANGER: '{self.type}' values should not be instantiated directly!"
            self.value = msg
            return
        if not isinstance(self.value, self.orig):
            return
        self.validate()

    @abstractmethod
    def validate(self):
        raise NotImplementedError

@beartype
@dataclass
class MaybeExtantFile(MaybeType):
    value: Union[Path, ExtantFile, str]
    type: Type = ExtantFile
    orig: Type = Path

    def validate(self):
        """Validate input."""
        # Resolve path.
        path = self.value.resolve()

        # Check that path exists and is a file.
        if not self.value.exists():
            self.value = f"File or directory not found: {path}"
        elif self.value.is_dir():
            self.value = f"Expected file, got directory: {path}"
        elif not self.value.is_file():
            self.value = f"Extant but not a file: {path}"

        # Must be an extant file.
        else:
            self.value = ExtantFile(path)



if __name__ == "__main__":
    res = MaybeExtantFile("generic error")
    print(res.value)
    res = MaybeExtantFile(Path("nonexistent_path"))
    print(res.value)
    res = MaybeExtantFile(Path("machine"))
    print(res.value)
    res = MaybeExtantFile(Path("datacls.py"))
    print(res.value)
    res = MaybeExtantFile(ExtantFile("datacls.py"))
    print(res.value)
