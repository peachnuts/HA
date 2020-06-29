# ======================================================================
# Copyright TOTAL / CERFACS / LIRMM (02/2020)
# Contributor: Adrien Suau (<adrien.suau@cerfacs.fr>
#                           <adrien.suau@lirmm.fr>)
#              Siyuan Niu  (<siyuan.niu@lirmm.fr>)
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

import typing as ty

import numpy
from qiskit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.dag_to_circuit import dag_to_circuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit, DAGNode

from hamap.distance_matrix import get_distance_matrix_mixed
from hamap.gates import TwoQubitGate
from hamap.hardware.IBMQHardwareArchitecture import IBMQHardwareArchitecture
from hamap.heuristics import sabre_heuristic
from hamap.layer import QuantumLayer, update_layer
from hamap.mapping_to_str import mapping_to_str
from hamap.swap import get_all_swap_bridge_candidates


def _create_empty_dagcircuit_from_existing(dagcircuit: DAGCircuit) -> DAGCircuit:
    result = DAGCircuit()
    for creg in dagcircuit.cregs.values():
        result.add_creg(creg)
    for qreg in dagcircuit.qregs.values():
        result.add_qreg(qreg)
    return result


def ha_mapping(
    quantum_circuit: QuantumCircuit,
    initial_mapping: ty.Dict[Qubit, int],
    hardware: IBMQHardwareArchitecture,
    swap_cost_heuristic: ty.Callable[
        [
            IBMQHardwareArchitecture,  # Hardware information
            QuantumLayer,  # Current front layer
            ty.List[DAGNode],  # Topologically sorted list of nodes
            int,  # Index of the first non-processed gate.
            ty.Dict[Qubit, int],  # The mapping before applying the tested SWAP/Bridge
            numpy.ndarray,  # The distance matrix between each qubits
            TwoQubitGate,  # The SWAP/Bridge we want to rank
        ],
        float,
    ] = sabre_heuristic,
    get_candidates: ty.Callable[
        [QuantumLayer, IBMQHardwareArchitecture, ty.Dict[Qubit, int], ty.Set[str],],
        ty.List[TwoQubitGate],
    ] = get_all_swap_bridge_candidates,
    get_distance_matrix: ty.Callable[
        [IBMQHardwareArchitecture], numpy.ndarray
    ] = lambda hardware: get_distance_matrix_mixed(hardware, 0.5, 0, 0.5),
) -> ty.Tuple[QuantumCircuit, ty.Dict[Qubit, int]]:
    """Map the given quantum circuit to the hardware topology provided.

    :param quantum_circuit: the quantum circuit to map.
    :param initial_mapping: the initial mapping used to start the iterative mapping
        algorithm.
    :param hardware: hardware data such as connectivity, gate time, gate errors, ...
    :param swap_cost_heuristic: the heuristic cost function that will estimate the cost
        of a given SWAP/Bridge according to the current state of the circuit.
    :param get_candidates: a function that takes as input the current front
        layer and the hardware description and that returns a list of tuples
        representing the SWAP/Bridge that should be considered by the heuristic.
    :param get_distance_matrix: a function that takes as first (and only) parameter the
        hardware representation and that outputs a numpy array containing the cost of
        performing a SWAP between each pair of qubits.
    :return: The final circuit along with the mapping obtained at the end of the
        iterative procedure.
    """
    # Creating the internal data structures that will be used in this function.
    dag_circuit = circuit_to_dag(quantum_circuit)
    distance_matrix = get_distance_matrix(hardware)
    resulting_dag_quantum_circuit = _create_empty_dagcircuit_from_existing(dag_circuit)
    current_mapping = initial_mapping
    explored_mappings: ty.Set[str] = set()
    # Sorting all the quantum operations in topological order once for all.
    # May require significant memory on large circuits...
    topological_nodes: ty.List[DAGNode] = list(dag_circuit.topological_op_nodes())
    current_node_index = 0
    # Creating the initial front layer.
    front_layer = QuantumLayer()
    current_node_index = update_layer(
        front_layer, topological_nodes, current_node_index
    )

    # Start of the iterative algorithm
    while not front_layer.is_empty():
        execute_gate_list = QuantumLayer()
        for op in front_layer.ops:
            if hardware.can_natively_execute_operation(op, current_mapping):
                execute_gate_list.add_operation(op)
                # Delaying the remove operation because we do not want to remove from
                # a container we are iterating on.
                # front_layer.remove_operation(op)
        if not execute_gate_list.is_empty():
            front_layer.remove_operations_from_layer(execute_gate_list)
            execute_gate_list.apply_back_to_dag_circuit(
                resulting_dag_quantum_circuit, initial_mapping, current_mapping
            )
            # Empty the explored mappings because at least one gate has been executed.
            explored_mappings.clear()
        else:
            # We cannot execute any gate, that means that we should insert at least
            # one SWAP/Bridge to make some gates executable.
            # First list all the SWAPs/Bridges that may help us make some gates
            # executable.
            swap_candidates = get_candidates(
                front_layer, hardware, current_mapping, explored_mappings
            )
            # Then rank the SWAPs/Bridge and take the best one.
            best_swap_qubits = None
            best_cost = float("inf")
            for potential_swap in swap_candidates:
                cost = swap_cost_heuristic(
                    hardware,
                    front_layer,
                    topological_nodes,
                    current_node_index,
                    current_mapping,
                    distance_matrix,
                    potential_swap,
                )
                if cost < best_cost:
                    best_cost = cost
                    best_swap_qubits = potential_swap
            # We now have our best SWAP/Bridge, let's perform it!
            current_mapping = best_swap_qubits.update_mapping(current_mapping)
            explored_mappings.add(mapping_to_str(current_mapping))
            best_swap_qubits.apply(resulting_dag_quantum_circuit, front_layer)
        # Anyway, update the current front_layer
        current_node_index = update_layer(
            front_layer, topological_nodes, current_node_index
        )

    # We are done here, we just need to return the results
    # resulting_dag_quantum_circuit.draw(scale=1, filename="qcirc.dot")
    resulting_circuit = dag_to_circuit(resulting_dag_quantum_circuit)
    return resulting_circuit, current_mapping
