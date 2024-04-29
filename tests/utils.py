from typing import Union, Tuple
from numpy.typing import ArrayLike, NDArray


class QuantumStateTest:
    def __init__(self, qubits_number: int) -> None:
        raise NotImplementedError()

    def apply1(self, pos: int, gate: NDArray) -> None:
        raise NotImplementedError()

    def apply2(self, pos1: int, pos2: int, gate: NDArray) -> None:
        raise NotImplementedError()

    def dens1(self, pos: int) -> NDArray:
        raise NotImplementedError()

    def dens2(self, pos1: int, pos2: int) -> NDArray:
        raise NotImplementedError()

    def measure(self, pos: int, uniform_sample: NDArray) -> int:
        raise NotImplementedError()

    def reset(self, pos: int, uniform_sample: NDArray) -> None:
        raise NotImplementedError()

    def total_reset(self) -> None:
        raise NotImplementedError()

    @property
    def state_array(self) -> NDArray:
        raise NotImplementedError()

    @property
    def qubits_number(self) -> int:
        raise NotImplementedError()


def array2state_test(array: NDArray) -> QuantumStateTest:
    raise NotImplementedError()


def random_q1_gate() -> NDArray:
    raise NotImplementedError()


def random_q2_gate() -> NDArray:
    raise NotImplementedError()


def random_state(qubits_number: int) -> NDArray:
    raise NotImplementedError()


def random_position(qubits_number: int) -> int:
    raise NotImplementedError()


def random_two_positions(qubits_number) -> Tuple[int, int]:
    raise NotImplementedError()


def uniform_sample() -> NDArray:
    raise NotImplementedError()
