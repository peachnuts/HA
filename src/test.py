from qiskit import QuantumCircuit, IBMQ, execute, Aer
from hamap import (
    ha_mapping,
    ha_mapping_paper_compliant,
    IBMQHardwareArchitecture,
)
from IBMQSubmitter import IBMQSubmitter

circuit = QuantumCircuit.from_qasm_file('/home/siyuan/Thesis/PycharmProjects/HelloWorld/HA/alu-v0_27.qasm')
hardware = IBMQHardwareArchitecture("ibmq_vigo")
#initial_mapping = {qubit: i for i,qubit in enumerate(circuit.qubits)}
initial_mapping = {}
mapping = [2, 4, 3, 1, 0]
for i,qubit in enumerate(circuit.qubits):
    initial_mapping[qubit] = mapping[i]

mapped_circuit, final_mapping = ha_mapping(
    circuit, initial_mapping, hardware
)
mapped_circuit.measure_active()

print("Loading account...")
IBMQ.load_account()
provider = IBMQ.get_provider(
    hub="ibm-q-france", group="univ-montpellier", project="default"
)

backend = provider.get_backend("ibmq_vigo")
print(f"Running on {backend.name()}.")
submitter = IBMQSubmitter(backend, tags=["test"])

submitter.add_circuit(
    mapped_circuit, [value for value in initial_mapping.values()], backend
)

print(f"Submitting {len(submitter)} circuits...")
submitter.submit()
print("Done! Saving...")
