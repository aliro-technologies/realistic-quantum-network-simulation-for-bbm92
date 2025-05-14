# Copyright © 2025 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.

from typing import Tuple, Union
import simpy
import random
import numpy as np

import aqnsim
from aqnsim.quantum_simulator import quantum_operations as ops
from aqnsim.entity.process_wrapper import process
from aqnsim import SECOND

EXPECTED_STATE_DENSITY = aqnsim.BELL_STATES_DENSITY["phi_plus"]


class EntGenProtocol(aqnsim.MidBSMEntanglementProtocol):
    """
    Extends MidBSMEntanglement Protocol, but does not loop when run; terminates after
    entanglement is established.

    :param sim_context: The simulation context used for the simulation.
    :param cport_name: The name of the port through which the node will receive BSM measurement results.
    :param qmem_port_name: The name of the QMemory's port on which Qubits will be emitted.
    :param comm_qpos: The communication-qubit's integer position in QMemory - its Qubit will be emitted.
    :param mem_qpos: The memory-qubit's integer position in QMemory - its Qubit will remain behind.
    :param correction: Whether or not the node applies a local correction, depending on the BSM result.
    :param clock: As an alternative to direct triggering, this optional parameter can
        either be provided as a period, a DelayModel, or a user-specified Clock, which will determine
        when the node entangles and emits automatically.
    :param name: The name of this component. Defaults to class name.
    :param logging: Whether or not simlogger and eventlogger statements are logged.
    :param log_fidelity: Whether or not the final entanglement fidelity is logged or not.
    :param qmemory_name: The name of the associated qmemory.
    """

    def __init__(
        self,
        sim_context: aqnsim.SimulationContext,
        cport_name: str,
        qmem_port_name: str = aqnsim.QMEMORY_DEFAULT_PORT_NAME,
        comm_qpos: int = 0,
        mem_qpos: int = 1,
        correction: bool = False,
        clock: Union[int, float, aqnsim.DelayModel, aqnsim.Clock] = None,
        name: str = None,
        logging: bool = True,
        log_fidelity: bool = True,
        qmemory_name: str = None,
    ):
        """Constructor method"""
        super().__init__(
            sim_context=sim_context,
            cport_name=cport_name,
            qmem_port_name=qmem_port_name,
            comm_qpos=comm_qpos,
            mem_qpos=mem_qpos,
            correction=correction,
            clock=clock,
            name=name,
            logging=logging,
            log_fidelity=log_fidelity,
            qmemory_name=qmemory_name
        )

    @process
    def run(self):
        """Main run method of the protocol.

        The run method of the EntangleEmitProtocol is also running, so here we just
        need to wait for the BSM result and apply any correction.
        """
        self.trigger()

        # Wait to know that entangle-emit operation is complete
        yield self.await_signal(self.eep, aqnsim.UniversalSignals.COMPLETED.name)

        # Wait for receipt of BSM result
        msg, _ = yield self.await_signal(self, self.receive_meas_signal_name)

        # Process this message, apply any necessary state correction, and signal completion
        yield self.process_message_and_correction(msg)


def setup_network(
    sim_context: aqnsim.SimulationContext,
    elementary_link_loss_in_db: float,
    elementary_link_quantum_delay: float,
    fiber1_length: float,
    fiber2_length: float,
    depolarizing_prob: float,
) -> aqnsim.Network:
    """
    Sets up the network for the simulation.
    
    :param sim_context: The simulation context for the simulation.
    :param elementary_link_loss_in_db: The link loss in dB.
    :param elementary_link_quantum_delay: The delay for the elementary link, in seconds.
    :param fiber1_length: The length of one fiber, in meters.
    :param fiber2_length: The length of the second fiber, in meters.
    :param depolarizing_prob: The probability of depolarizing for each qubit.
    """
    # Instantiate nodes and network that contains them
    alice = aqnsim.Node(sim_context, name="Alice")
    bob = aqnsim.Node(sim_context, name="Bob")
    network = aqnsim.Network(sim_context)#, nodes=[alice, bob])
    network.add_node(alice)
    network.add_node(bob)

    # Set how much time each gate takes by constructing a dictionary
    op_delays = {
        ops.H: 0 * SECOND,
        ops.X: 0 * SECOND,
        ops.Z: 0 * SECOND,
        ops.CNOT: 0 * SECOND,
    }
    # Set how much time a measurement takes
    meas_delay = 0 * SECOND

    # Instantiate qmemories and add to nodes
    memory1 = aqnsim.QMemory(sim_context, n=2, meas_delay=meas_delay, name="QMemory-Alice")
    memory1.set_op_delays(op_delays=op_delays)
    alice.add_qmemory(memory1)

    memory2 = aqnsim.QMemory(sim_context, n=2, meas_delay=meas_delay, name="QMemory-Bob")
    memory2.set_op_delays(op_delays=op_delays)
    bob.add_qmemory(memory2)

    # Create a new port on each node, that here we will use as a common
    # portal for both qubit output and classical input
    alice.add_port("port")
    bob.add_port("port")

    # Forward output from qmemory to the node's port as output
    memory1.ports["qport"].forward_output_to_output(alice.ports["port"])
    memory2.ports["qport"].forward_output_to_output(bob.ports["port"])

    # Implement qubit noise model
    qubit_noise_model = aqnsim.DepolarNoiseModel(qs=sim_context.qs, p=depolarizing_prob)

    # Use a middleBSM link to connect the nodes
    # Instantiate this link and add it to the network, connecting nodes
    bsm_link = aqnsim.MiddleBSMDualFiberLink(
        sim_context,
        length1=fiber1_length,
        length2=fiber2_length,
        attenuation_coeff=elementary_link_loss_in_db,
        loss=True,
        bsm_delay=elementary_link_quantum_delay,
        bsm_noise=qubit_noise_model,
        apply_msg_dependent_delay=False,
        refractive_index=1,
        name=f"bsm_link",
    )

    network.add_link(
        bsm_link,
        node1_name=alice.name,
        node2_name=bob.name,
        port1_name="port",
        port2_name="port")
    return network


def setup_protocols(
    sim_context: aqnsim.SimulationContext,
    alice: aqnsim.Node,
    bob: aqnsim.Node,
) -> Tuple[aqnsim.Protocol, aqnsim.Protocol]:
    """
    Sets up the protocols for each node.

    :param sim_context: The simulation context to use.
    :param alice: The first node to assign the protocol to
    :param bob: The second node to assign the protocol to
    :param name: The name of the protocol
    """
    # Indicate that we want Alice to do a correction
    alice_protocol = EntGenProtocol(
        sim_context, cport_name="port", correction=True, log_fidelity=True, qmemory_name="QMemory-Alice"
    )
    bob_protocol = EntGenProtocol(
        sim_context, cport_name="port", correction=False, log_fidelity=True, qmemory_name="QMemory-Bob"
    )
    alice.add_protocol(alice_protocol)
    bob.add_protocol(bob_protocol)

    return alice_protocol, bob_protocol


def check_example(alice_protocol: aqnsim.Protocol, bob_protocol: aqnsim.Protocol):
    """
    Check that Alice and Bob established the same number of end-to-end entanglements and that the qubits
    are within the same space.

    :param alice_protocol: Protocol attached to Node Alice
    :param bob_protocol: Protocol attached to Node Bob
    """
    if hasattr(alice_protocol, "bsm_fidelity"):
        alice_fidelity = alice_protocol.bsm_fidelity
    else:
        alice_fidelity = None
    if hasattr(bob_protocol, "bsm_fidelity"):
        bob_fidelity = bob_protocol.bsm_fidelity
    else:
        bob_fidelity = None

    if alice_fidelity is not None and bob_fidelity is not None:
        # No photon loss for either Alice or Bob
        output_state = alice_protocol.parent_component.qmemory.positions[1].qubit.state.state
        state_fidelity = aqnsim.compute_fidelity(output_state, EXPECTED_STATE_DENSITY)
    else:
        state_fidelity = None

    entanglement_generation_time = bob_protocol.env.now
    return state_fidelity, entanglement_generation_time


def run_elementary_link_simulation(
    random_seed: int,
    num_shots: int,
    elementary_link_loss_in_db: float,
    elementary_link_quantum_delay: float,
    fiber1_length: float = 1,
    fiber2_length: float = 0,
    depolarizing_prob: Union[int, float] = 0,
):
    """
    Run the simulation.

    :param random_seed: The seed to seed the simulation.
    :param num_shots: The number of shots to run the simulation.
    :param elementary_link_loss_in_db: The link loss in dB.
    :param elementary_link_quantum_delay: The delay for the elementary link, in seconds.
    :param fiber1_length: The length of one fiber, in meters.
    :param fiber2_length: The length of the second fiber, in meters.
    :param depolarizing_prob: The probability of depolarizing for each qubit.
    """

    random.seed(random_seed)
    np.random.seed(random_seed)

    state_fidelities = []
    entanglement_generation_times = []
    state_fidelity = None

    for run in range(num_shots):
        sim_context = aqnsim.SimulationContext(log_to_file=False, logging_level=0, defer_measurements=False)

        # Instantiate environment and QuantumSimulator
        while state_fidelity is None:
            # Configure logger
            aqnsim.simlogger.configure(env=sim_context.env)
            aqnsim.eventlogger.configure(env=sim_context.env)

            # Setup network and protocols, and run sim until the given time
            network = setup_network(
                sim_context,
                elementary_link_loss_in_db,
                elementary_link_quantum_delay,
                fiber1_length,
                fiber2_length,
                depolarizing_prob
            )
            [alice, bob] = network.nodes
            alice_protocol, bob_protocol = setup_protocols(sim_context, alice, bob)

            sim_context.env.run()
            state_fidelity, entanglement_generation_time = check_example(
                alice_protocol, bob_protocol
            )

        state_fidelities.append(state_fidelity)
        entanglement_generation_times.append(entanglement_generation_time)
        state_fidelity = None
    print(
        f"Average entanglement generation time: {np.mean(entanglement_generation_times)}"
    )
    expected_success_probability = 1 - (1 - 10 ** (-elementary_link_loss_in_db / 10))
    expected_time_to_entanglement = (
        1 / expected_success_probability * elementary_link_quantum_delay
    )
    print(
        f"Expected average entanglement generation time: {expected_time_to_entanglement}"
    )
    return state_fidelities, entanglement_generation_times


if __name__ == "__main__":
    random_seed = 1
    num_shots = 50
    elementary_link_loss_in_db = 10
    elementary_link_quantum_delay = 1

    # To also simulate the minimum classical communication times needed (L0/c * 1/2) for a centrally
    # located BSM station, the BSM station is instead placed at an end node.
    fiber1_length = 0
    fiber2_length = 1

    run_elementary_link_simulation(
        random_seed=random_seed,
        num_shots=num_shots,
        elementary_link_loss_in_db=elementary_link_loss_in_db,
        elementary_link_quantum_delay=elementary_link_quantum_delay,
        fiber1_length=fiber1_length,
        fiber2_length=fiber2_length,
        depolarizing_prob=0,
    )
