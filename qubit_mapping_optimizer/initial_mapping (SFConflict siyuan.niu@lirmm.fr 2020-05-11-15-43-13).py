# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (02/2020)
# Contributor: Adrien Suau (<adrien.suau@cerfacs.fr>
#                           <adrien.suau@lirmm.fr>)
#
# This software is governed by the CeCILL-B license under French law and
# abiding  by the  rules of  distribution of free software. You can use,
# modify  and/or  redistribute  the  software  under  the  terms  of the
# CeCILL-B license as circulated by CEA, CNRS and INRIA at the following
# URL "http://www.cecill.info".
#
# As a counterpart to the access to  the source code and rights to copy,
# modify and  redistribute granted  by the  license, users  are provided
# only with a limited warranty and  the software's author, the holder of
# the economic rights,  and the  successive licensors  have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using, modifying and/or  developing or reproducing  the
# software by the user in light of its specific status of free software,
# that  may mean  that it  is complicated  to manipulate,  and that also
# therefore  means that  it is reserved for  developers and  experienced
# professionals having in-depth  computer knowledge. Users are therefore
# encouraged  to load and  test  the software's  suitability as  regards
# their  requirements  in  conditions  enabling  the  security  of their
# systems  and/or  data to be  ensured and,  more generally,  to use and
# operate it in the same conditions as regards security.
#
# The fact that you  are presently reading this  means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================
import random
import typing as ty
import math
from copy import copy

import numpy
from qiskit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit

from qubit_mapping_optimizer._circuit_manipulation import add_qubits_to_quantum_circuit
from qubit_mapping_optimizer.hardware.IBMQHardwareArchitecture import (
    IBMQHardwareArchitecture,
)
from qubit_mapping_optimizer.optimisation.simulated_annealing import simulated_annealing


def get_random_mapping(quantum_circuit: QuantumCircuit) -> ty.Dict[Qubit, int]:
    random_sampling = numpy.random.permutation(len(quantum_circuit.qubits))
    return {qubit: random_sampling[i] for i, qubit in enumerate(quantum_circuit.qubits)}


def _is_fixed_point(swap_numbers: ty.List[int]) -> bool:
    if len(swap_numbers) < 2:
        return False
    return swap_numbers[-1] == swap_numbers[-2]


def _argmin(l: ty.Iterable) -> int:
    return min(((v, i) for i, v in enumerate(l)), key=lambda tup: tup[0])[1]


def _count_swaps(circuit: QuantumCircuit) -> int:
    return circuit.count_ops().get("swap", 0)


def _count_cnots(circuit: QuantumCircuit) -> int:
    ops = circuit.count_ops()
    return 3 * ops.get("swap", 0) + ops.get("cx", 0) + 4 * ops.get("bridge", 0)


def initial_mapping_from_iterative_forward_backward(
    quantum_circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    mapping_algorithm: ty.Callable[
        [QuantumCircuit, IBMQHardwareArchitecture, ty.Dict[Qubit, int]],
        ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]],
    ],
    initial_mapping: ty.Optional[ty.Dict[Qubit, int]] = None,
    circuit_cost: ty.Callable[[QuantumCircuit], int] = _count_cnots,
    maximum_mapping_procedure_calls: int = 20,
) -> ty.Tuple[ty.Dict[Qubit, int], float, int]:
    """Implementation of the initial_mapping method used by SABRE.

    :param quantum_circuit: the quantum circuit we want to find an initial mapping for
    :param hardware: the target hardware specifications
    :param mapping_algorithm: the algorithm used to map a quantum circuit to the
        given hardware with the given initial mapping.
    :param circuit_cost: a function computing the cost of a circuit. By default,
        the cost is the number of SWAP gates.
    :param initial_mapping: starting point of the algorithm. Default to a random guess.
    :param maximum_mapping_procedure_calls: the maximum number of calls to the
        mapping procedure. If the algorithm converged before, a lower number
        of evaluations will be performed.
    :return: the initial mapping, the cost of this mapping and the number of
        calls to the provided mapping procedure performed.
    """
    if maximum_mapping_procedure_calls < 2:
        raise RuntimeError(
            "You should do at least 1 iteration (2 calls to the mapping procedure)!"
        )
    # First make sure that the quantum circuit has the same number of quantum bits as
    # the hardware.
    quantum_circuit = add_qubits_to_quantum_circuit(quantum_circuit, hardware)
    reversed_quantum_circuit = quantum_circuit.inverse()
    # Generate a random initial mapping
    if initial_mapping is None:
        initial_mapping = get_random_mapping(quantum_circuit)

    # And improve this initial mapping according to an iterated method inspired from
    # SABRE.
    costs = list()
    mappings: ty.List[ty.Dict[Qubit, int]] = [initial_mapping]
    # We apply the forward-backward approach
    forward_mapping = initial_mapping
    for i in range(maximum_mapping_procedure_calls // 2):
        # Performing the forward step
        forward_circuit, reversed_mapping = mapping_algorithm(
            quantum_circuit, hardware, forward_mapping
        )
        # And the backward step
        _, forward_mapping = mapping_algorithm(
            reversed_quantum_circuit, hardware, reversed_mapping
        )
        # Adding the cost of the previous mapping to the list
        costs.append(circuit_cost(forward_circuit))
        # Adding the current mapping to the list in case we will do more iterations.
        mappings.append(forward_mapping)
        # If there is a repetition or we have a cost of 0, we can stop here.
        if costs[-1] == 0 or _is_fixed_point(costs):
            break

    # We may have not used all the calls to the mapping procedure we were allowed to do.
    # If this is the case, use one more to evaluate the last mapping added.
    current_calls_to_mapping_procedure = 2 * (i + 1)
    if current_calls_to_mapping_procedure < maximum_mapping_procedure_calls:
        forward_circuit, _ = mapping_algorithm(quantum_circuit, hardware, mappings[-1])
        costs.append(circuit_cost(forward_circuit))
    # If we finished the allowed number of iterations, return the best result.
    best_mapping_index = _argmin(costs)
    return (
        mappings[best_mapping_index],
        costs[best_mapping_index],
        current_calls_to_mapping_procedure + 1,
    )


def initial_mapping_from_sabre(
    quantum_circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    mapping_algorithm: ty.Callable[
        [QuantumCircuit, IBMQHardwareArchitecture, ty.Dict[Qubit, int]],
        ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]],
    ],
    circuit_cost: ty.Callable[[QuantumCircuit], int] = _count_cnots,
    initial_mapping: ty.Optional[ty.Dict[Qubit, int]] = None,
) -> ty.Tuple[ty.Dict[Qubit, int], float]:
    mapping, cost, _ = initial_mapping_from_iterative_forward_backward(
        quantum_circuit,
        hardware,
        mapping_algorithm,
        initial_mapping,
        circuit_cost,
        # 3 allowed calls in order to let the procedure evaluate the cost of the
        # last mapping and potentially return it.
        maximum_mapping_procedure_calls=3,
    )
    return mapping, cost


def get_neighbour_random(mapping: ty.Dict[Qubit, int]) -> ty.Dict[Qubit, int]:
    inverse_mapping = {v: k for k, v in mapping.items()}
    a, b = random.choices(list(inverse_mapping.keys()), k=2)
    inverse_mapping[a], inverse_mapping[b] = inverse_mapping[b], inverse_mapping[a]
    return {k: v for v, k in inverse_mapping.items()}


NeighbourMappingAlgorithmType = ty.Callable[
    [ty.Dict[Qubit, int], IBMQHardwareArchitecture], ty.Dict[Qubit, int]
]


def _random_execution_policy(
    p1: float, p2: float, algorithms: ty.List[NeighbourMappingAlgorithmType],
) -> NeighbourMappingAlgorithmType:
    p = random.random()
    if p < p1:
        return algorithms[0]
    elif p < p1 + p2:
        return algorithms[1]
    else:
        return algorithms[2]


def _random_shuffle(
    mapping: ty.Dict[Qubit, int], _: IBMQHardwareArchitecture
) -> ty.Dict[Qubit, int]:
    values = list(mapping.values())
    random.shuffle(values)
    new_mapping = dict()
    for i, qubit in enumerate(mapping):
        new_mapping[qubit] = values[i]
    return new_mapping


def _random_expand(
    mapping: ty.Dict[Qubit, int], hardware: IBMQHardwareArchitecture
) -> ty.Dict[Qubit, int]:
    qubit_number = hardware.qubit_number
    if len(mapping) == qubit_number:
        return _random_shuffle(mapping, hardware)
    not_used_qubits = list(set(range(qubit_number)) - set(mapping.values()))
    new_qubit = random.choice(not_used_qubits)
    new_mapping = copy(mapping)
    new_mapping[random.choice(list(new_mapping.keys()))] = new_qubit
    return new_mapping


def _random_reset(
    mapping: ty.Dict[Qubit, int], hardware: IBMQHardwareArchitecture
) -> ty.Dict[Qubit, int]:
    qubits = list(mapping.keys())
    values = random.sample(list(range(hardware.qubit_number)), len(qubits))
    new_mapping = dict()
    for q, v in zip(qubits, values):
        new_mapping[q] = v
    return new_mapping


def get_neighbour_improved(
    mapping: ty.Dict[Qubit, int],
    hardware: IBMQHardwareArchitecture,
    policy: ty.Callable[
        [
            ty.Dict[Qubit, int],
            IBMQHardwareArchitecture,
            ty.List[NeighbourMappingAlgorithmType],
        ],
        NeighbourMappingAlgorithmType,
    ],
    algorithms: ty.List[NeighbourMappingAlgorithmType],
) -> ty.Dict[Qubit, int]:
    algorithm = policy(mapping, hardware, algorithms)
    return algorithm(mapping, hardware)


def get_initial_mapping_from_annealing(
    cost_function: ty.Callable[
        [ty.Dict[Qubit, int], QuantumCircuit, IBMQHardwareArchitecture], float
    ],
    quantum_circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    initial_mapping: ty.Optional[ty.Dict[Qubit, int]] = None,
    get_neighbour_func: ty.Callable[
        [ty.Dict[Qubit, int]], ty.Dict[Qubit, int]
    ] = get_neighbour_random,
    max_steps: int = 1000,
    temp_begin: float = 10.0,
    cost_threshold: float = 1e-6,
    schedule_func: ty.Callable[[float], float] = lambda x: x * 0.99,
) -> ty.Tuple[ty.Dict[Qubit, int], float, int]:
    # Generate a random initial mapping
    if initial_mapping is None:
        initial_mapping = get_random_mapping(quantum_circuit)

    mapping, cost, iteration_number = simulated_annealing(
        initial_mapping,
        lambda mapping: cost_function(mapping, quantum_circuit, hardware),
        get_neighbour_func,
        temp_begin,
        max_steps,
        schedule_func,
        cost_threshold,
    )
    return mapping, cost, iteration_number


def get_best_mapping_random(
    circuit: QuantumCircuit,
    cost_function: ty.Callable[
        [ty.Dict[Qubit, int], QuantumCircuit, IBMQHardwareArchitecture], float
    ],
    hardware: IBMQHardwareArchitecture,
    maximum_allowed_evaluations: int,
) -> ty.Dict[Qubit, int]:
    best_mapping = get_random_mapping(circuit)
    best_cost = cost_function(best_mapping, circuit, hardware)
    for _ in range(maximum_allowed_evaluations - 1):
        mapping = get_random_mapping(circuit)
        cost = cost_function(mapping, circuit, hardware)
        if cost < best_cost:
            best_mapping = mapping
            best_cost = cost
    return best_mapping


def get_best_mapping_sabre(
    circuit: QuantumCircuit,
    mapping_algorithm: ty.Callable[
        [QuantumCircuit, IBMQHardwareArchitecture, ty.Dict[Qubit, int]],
        ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]],
    ],
    hardware: IBMQHardwareArchitecture,
    maximum_allowed_evaluations: int,
) -> ty.Dict[Qubit, int]:
    if maximum_allowed_evaluations < 3:
        print("Not enough allowed evaluations!")
        exit(1)
    best_mapping, best_cost = initial_mapping_from_sabre(
        circuit, hardware, mapping_algorithm
    )
    for i in range(maximum_allowed_evaluations // 3):
        mapping, cost = initial_mapping_from_sabre(circuit, hardware, mapping_algorithm)
        if cost < best_cost:
            best_mapping, best_cost = mapping, cost
    return best_mapping


def get_best_mapping_from_annealing(
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    cost_function: ty.Callable[
        [ty.Dict[Qubit, int], QuantumCircuit, IBMQHardwareArchitecture], float
    ],
    maximum_allowed_evaluations: int,
):
    temp_begin = 10.0
    alpha = math.exp(
        (-6 * math.log(10) - math.log(temp_begin)) / maximum_allowed_evaluations
    )
    mapping, *_ = get_initial_mapping_from_annealing(
        cost_function,
        circuit,
        hardware,
        max_steps=maximum_allowed_evaluations,
        temp_begin=temp_begin,
        schedule_func=lambda x: x ** alpha,
    )
    return mapping


def get_best_mapping_from_iterative_forward_backward(
    circuit: QuantumCircuit,
    hardware: IBMQHardwareArchitecture,
    mapping_algorithm: ty.Callable[
        [QuantumCircuit, IBMQHardwareArchitecture, ty.Dict[Qubit, int]],
        ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]],
    ],
    maximum_allowed_evaluations: int,
):
    (
        best_mapping,
        best_cost,
        call_number,
    ) = initial_mapping_from_iterative_forward_backward(
        circuit,
        hardware,
        mapping_algorithm,
        maximum_mapping_procedure_calls=maximum_allowed_evaluations,
    )
    while maximum_allowed_evaluations - call_number >= 2:
        mapping, cost, i = initial_mapping_from_iterative_forward_backward(
            circuit,
            hardware,
            mapping_algorithm,
            maximum_mapping_procedure_calls=maximum_allowed_evaluations - call_number,
        )
        call_number += i
        if cost < best_cost:
            best_mapping, best_cost = mapping, cost
    return best_mapping
