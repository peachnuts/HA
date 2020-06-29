# HA algorithm

This repository contains the implementation of our Hardware-Aware mapping algorithm.

## Installation

The repository is a Python package. You can install by cloning with `git` and then using Python's package manager `pip`:

``` shell
git clone https://github.com/peachnuts/HA.git
python -m pip install HA
```

## How to use?

The main function that should be the entry point for any user is [`qubit_mapping_optimizer.mapping.iterative_mapping_algorithm`](https://github.com/peachnuts/HA/blob/master/src/qubit_mapping_optimizer/mapping.py#L69).
This function takes as parameter:

1. An instance of `qiskit.QuantumCircuit` representing the circuit to map.
2. An initial mapping, given as a dictionnary that maps the instances of `qiskit.Qubit` contained in the quantum circuit given as first parameter to the physical qubit identifier, i.e. an integer representing a physical qubit on the hardware we map the circuit to.
3. An instance of `qubit_mapping_optimizer.hardware.IBMQHardwareArchitecture.IBMQHardwareArchitecture` that wraps Qiskit's API to retrieve calibration data and hardware information.
4. A function `swap_cost_heuristic` that takes as parameters
   1. An instance of `qubit_mapping_optimizer.hardware.IBMQHardwareArchitecture.IBMQHardwareArchitecture`.
   2. An instance of `qubit_mapping_optimizer.layer.QuantumLayer` representing the current "first layer".
   3. A list of `qiskit.dagcircuit.dagcircuit.DAGNode` that contains the nodes of the `DAGCircuit` (i.e. quantum gates) that are not mapped yet, sorted in a topological order (i.e. the first node is guaranteed to to have all its predecessors already mapped).
   4. The index of the current "first" gate, i.e. the first gate of the `DAGNode` list that have not already been mapped.
   5. The current mapping as a dictionnary that maps instances of `qiskit.Qubit` to hardware qubits indices.
   6. A `numpy` array that stores the distance between each pair of hardware qubits.
   7. An instance of `qubit_mapping_optimizer.gates.TwoQubitGate` (or a derived class such as `SwapTwoQubitGate` or `BridgeTwoQubitGate`that represents a potential 2-qubit gate for which we want to compute the cost.
   
   and returns the cost of the `TwoQubitGate` given.

## Notes on the implementation

The implementation presented here uses a slightly different method to chose between inserting a `SWAP` or a `Bridge` gate.
The algorithm described in the scientific paper first computes the best `SWAP` and then determine if it is worth changing the `SWAP` into a `Bridge` gate.
The implementation in this repository evaluates `Bridge` gates along `SWAP` ones, and pick the best gate according to the internal metric.
A switch or a new method will be added to use the exact algorithm explained in the paper in a few days.



