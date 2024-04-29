from typing import Tuple
from math import log2
import numpy as np
from numpy.typing import NDArray


class QuantumStateTest:
    def __init__(self, qubits_number: int) -> None:
        self._qubits_number = qubits_number
        self._state = np.zeros(2**qubits_number, dtype=np.complex128)
        self._state[0] = 1

    def apply1(self, pos: int, gate: NDArray) -> None:
        assert gate.shape == (2, 2)
        assert pos >= 0
        assert pos < self._qubits_number
        new_state = self._state.reshape(self._qubits_number * (2,))
        new_state = np.tensordot(gate, new_state, axes=((1,), (pos,)))
        new_state = new_state.transpose(
            list(range(1, pos + 1)) + [0] + list(range(pos + 1, self._qubits_number))
        )
        new_state = new_state.reshape((-1,))
        self._state = new_state

    def apply2(self, pos1: int, pos2: int, gate: NDArray) -> None:
        assert gate.shape == (2, 2, 2, 2)
        assert pos1 >= 0
        assert pos2 >= 0
        assert pos1 != pos2
        assert pos1 < self._qubits_number
        assert pos2 < self._qubits_number
        new_state = self._state.reshape(self._qubits_number * (2,))
        new_state = np.tensordot(gate, new_state, axes=((2, 3), (pos1, pos2)))
        if pos1 < pos2:
            new_state = new_state.transpose(
                list(range(2, pos1 + 2))
                + [0]
                + list(range(pos1 + 2, pos2 + 1))
                + [1]
                + list(range(pos2 + 1, self._qubits_number))
            )
        else:
            new_state = new_state.transpose(
                list(range(2, pos2 + 2))
                + [1]
                + list(range(pos2 + 2, pos1 + 1))
                + [0]
                + list(range(pos1 + 1, self._qubits_number))
            )
        new_state = new_state.reshape((-1,))
        self._state = new_state

    def dens1(self, pos: int) -> NDArray:
        assert pos >= 0
        assert pos < self._qubits_number
        state = self._state.reshape(self._qubits_number * (2,))
        state = np.transpose(
            state, (pos, *filter(lambda x: x != pos, range(self._qubits_number)))
        )
        state = state.reshape((2, -1))
        dens = np.tensordot(state, state.conj(), axes=((1), (1)))
        return dens

    def dens2(self, pos1: int, pos2: int) -> NDArray:
        assert pos1 >= 0
        assert pos2 >= 0
        assert pos1 != pos2
        assert pos1 < self._qubits_number
        assert pos2 < self._qubits_number
        state = self._state.reshape(
            (
                pos1,
                pos2,
                *filter(lambda x: x != pos1 & x != pos2, range(self._qubits_number)),
            )
        )
        state = state.reshape((2, 2, -1))
        dens = np.tensordot(state, state.conj(), axes=((2), (2)))
        return dens

    def measure(self, pos: int, uniform_sample: NDArray) -> int:
        dens = self.dens1(pos)
        p0 = dens[0].real
        result = 0 if uniform_sample < p0 else 1
        proj = np.zeros((2, 2), dtype=np.complex128)
        proj[result, result] = 1
        self.apply1(pos, proj)
        self._state /= np.linalg.norm(self._state)
        return result

    def reset(self, pos: int, uniform_sample: NDArray) -> None:
        result = self.measure(pos, uniform_sample)
        if result == 1:
            self.apply1(
                pos, np.array([0, 1, 1, 0], dtype=np.complex128).reshape((2, 2))
            )

    def total_reset(self) -> None:
        self._state = np.zeros(2**self._qubits_number, dtype=np.complex128)

    @property
    def state_array(self) -> NDArray:
        return self._state

    @property
    def qubits_number(self) -> int:
        return self._qubits_number


def array2state_test(array: NDArray) -> QuantumStateTest:
    assert len(array.shape) == 1
    size = array.shape[0]
    qubits_number = int(log2(size))
    assert size == 2**qubits_number
    state = QuantumStateTest(qubits_number)
    state._state = array
    return state


def random_q1_gate() -> NDArray:
    gate = np.random.normal((2, 2, 2))
    gate = gate[..., 0] + 1j * gate[..., 1]
    gate, _ = np.linalg.qr(gate)
    return gate


def random_q2_gate() -> NDArray:
    gate = np.random.normal((4, 4, 2))
    gate = gate[..., 0] + 1j * gate[..., 1]
    gate, _ = np.linalg.qr(gate)
    gate = gate.reshape((2, 2, 2, 2))
    return gate


def random_state(qubits_number: int) -> NDArray:
    size = 2**qubits_number
    state = np.random.normal((size, 2))
    state = state[..., 0] + 1j * state[..., 1]
    state /= np.linalg.norm(state)
    return state


def random_position(qubits_number: int) -> int:
    return int(np.random.random_integers(0, qubits_number - 1, 1)[0])


def random_two_positions(qubits_number) -> Tuple[int, int]:
    pos1 = int(np.random.random_integers(0, qubits_number - 1, 1)[0])
    pos2 = int(np.random.random_integers(0, qubits_number - 1, 1)[0])
    if pos1 == pos2:
        if pos1 == 0:
            pos1 += 1
        elif pos1 == qubits_number - 1:
            pos1 -= 1
    assert pos1 != pos2
    assert pos1 >= 0
    assert pos2 >= 0
    assert pos1 < qubits_number
    assert pos2 < qubits_number
    return pos1, pos2


def uniform_sample() -> NDArray:
    return np.random.uniform(size=1)
