#!/usr/bin/env python3

from itertools import takewhile
import numpy as np
from utils import (
    QuantumStateTest,
    array2state_test,
    uniform_sample,
    random_position,
    random_two_positions,
    random_state,
    random_q1_gate,
    random_q2_gate,
)
from qem import array2state, QuantumState


def test(
    qubits_number: int,
    gates_number: int,
    two_qubits_dens_number: int,
) -> None:
    print(
        f"Random circuit test with qubits number: {qubits_number}, gates number: {gates_number}, two-qubit density matrices number {two_qubits_dens_number}"
    )
    assert np.isclose(
        QuantumState(qubits_number).state_array,
        QuantumStateTest(qubits_number).state_array,
    ).all()
    print("\tStd state: OK")
    state = random_state(qubits_number)
    test_state = array2state_test(state)
    qem_state = array2state(state.copy())
    assert (
        test_state.qubits_number == qubits_number
    ), f"{test_state.qubits_number}, {qubits_number}"
    assert (
        qem_state.qubits_number == qubits_number
    ), f"{qem_state.qubits_number}, {qubits_number}"
    print("\tQubits number: OK")
    for num in range(gates_number):
        if num % 2 == 0:
            pos = random_position(qubits_number)
            gate = random_q1_gate()
            test_state.apply1(pos, gate)
            qem_state.apply1(pos, gate)
        else:
            pos1, pos2 = random_two_positions(qubits_number)
            gate = random_q2_gate()
            test_state.apply2(pos1, pos2, gate)
            qem_state.apply2(pos1, pos2, gate)
    assert np.isclose(test_state.state_array, qem_state.state_array).all()
    print("\tState after gates application: OK")
    if qubits_number > 10:
        qem_large_dens = qem_state.dens_large((5, 7, 3, 6, 0))
        test_large_dens = test_state.dens_large((5, 7, 3, 6, 0))
        assert np.isclose(qem_large_dens, test_large_dens).all()
        print("\tBig density matrices: OK")
    for i in range(qubits_number):
        arr_dens = test_state.dens1(i)
        qem_dens = qem_state.dens1(i)
        assert np.isclose(arr_dens, qem_dens).all(), f"{i}, {arr_dens}, {qem_dens}"
    for _ in range(two_qubits_dens_number):
        pos1, pos2 = random_two_positions(qubits_number)
        arr_dens = test_state.dens2(pos1, pos2)
        qem_dens = qem_state.dens2(pos1, pos2)
        assert np.isclose(
            arr_dens, qem_dens
        ).all(), f"{pos1}, {pos2}, {arr_dens}, {qem_dens}"
    assert np.isclose(test_state.state_array, qem_state.state_array).all()
    print("\tDensity matrices: OK")
    qubits_traverser = range(qubits_number)
    # half measure, half reset
    for i in takewhile(lambda x: x < qubits_number // 2, qubits_traverser):
        sample = uniform_sample()
        test_result = test_state.measure(i, sample)
        qem_result = qem_state.measure(i, sample)
        assert test_result == qem_result, f"{test_result}, {qem_result}"
    assert np.isclose(
        np.linalg.norm(qem_state.state_array), 1.0
    ), f"{np.linalg.norm(qem_state.state_array)}"
    assert np.isclose(test_state.state_array, qem_state.state_array).all()
    print("\tMeasurements: OK")
    for i in qubits_traverser:
        sample = uniform_sample()
        test_state.reset(i, sample)
        qem_state.reset(i, sample)
    assert np.isclose(
        np.linalg.norm(qem_state.state_array), 1.0
    ), f"{np.linalg.norm(qem_state.state_array)}"
    assert np.isclose(test_state.state_array, qem_state.state_array).all()
    print("\tReset: OK")


def main():
    test(2, 10, 10)
    test(5, 40, 40)
    test(12, 100, 100)


if __name__ == "__main__":
    main()
