
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class UserId:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            msg = "user id must be positive"
            raise ValueError(msg)

@dataclass(frozen=True, slots=True)
class MovieId:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            msg = "movie id must be positive"
            raise ValueError(msg)