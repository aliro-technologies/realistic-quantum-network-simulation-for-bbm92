# Copyright © 2025 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.
from typing import List, Union, Dict
import random
import numpy as np

import aqnsim
from aqnsim.quantum_simulator import quantum_operations as ops
from aqnsim import SECOND

START_SYNC_ACTION = "START SYNC"
END_TO_END_EG_COMPLETE_ACTION = "END TO END EG COMPLETE"
EXPECTED_STATE_DENSITY = aqnsim.BELL_STATES_DENSITY["phi_plus"]

"""
An example of an idealized repeater chain.

When a repeater is informed that elementary entanglement has been achieved on both
of its links it will then initiate a BSM operation, which acts to perform entanglement
swapping. The repeater will send the measurement result along a chain towards Alice,
who will collect these messages from all repeaters before, based on the measurement
results, will perform a final correction to complete the establishment of long-distance
entanglement between Alice and Bob.

The node protocols are synchronized using a sync_start message from Alice giving the
timestamp when all protocols should be started (and accounting for the max end-to-end distance).
After receiving the sync_start message, each protocol will send a sync_start signal to itself
to start the run loop processing so that the entanglement generation starts synchronously.
After the end-to-end entanglement is established, Alice sends an end_sync message to the end node
(Bob) who then resets the local qubit states.
"""


class AliceProtocol(aqnsim.NodeProtocol):
    """
    Protocol that we will attach to Alice.
    In addition to performing a local correction to establish elementary link entanglement,
    Alice will also be responsible for waiting for classical messages from repeaters to
    perform a final correction, completing the distribution of long-distance entanglement.

    :param sim_context: The SimulationContext object which will run the simulation.
    :param N: Number of repeaters in the chain.
    :param elementary_link_quantum_delay: The channel delay time, in seconds, for qubits.
    :param elementary_link_classical_delay: The channel delay time, in seconds, for classical messages.
    :param qpos: The positions on Alice's qmemory that are involved. The first entry in
        qpos corresponds to the qubit position to be emitted, and the second entry
        to the qubit that will remain behind and become entangled with Bob.
    :param name: Name of the protocol.
    :param qmemory_name: The name of the memory used in this protocol.
    """

    def __init__(
        self,
        sim_context: aqnsim.SimulationContext,
        N: int,
        elementary_link_quantum_delay: Union[float, int],
        elementary_link_classical_delay: Union[float, int],
        qpos: List[int] = None,
        name: str = "AliceProtocol",
        qmemory_name: str = None,
    ):
        super().__init__(sim_context=sim_context, name=name)

        qpos = [0, 1] if qpos is None else qpos
        self.qs = sim_context.qs
        self.N = N
        self.qpos = qpos
        self.qmemory_name = qmemory_name

        self.elementary_link_classical_delay = elementary_link_classical_delay
        self.elementary_link_quantum_delay = elementary_link_quantum_delay

    def initialize(self, parent_component=None):
        super().initialize(parent_component)
        self.qmem = self.parent_component.subcomponents[self.qmemory_name]
        self.parent_component.ports["cport"].add_rx_input_handler(self.cport_handler)

        # Unlike a repeater, an end node just needs one instance of an
        # elementary EntGen protocol.
        # Here Alice will also be responsible for two correction:
        # 1) The ent_gen protocol is responsible for doing the link-layer correction
        # 2) Also responsible for the final correction to complete the repeater swapping process
        entgen_proto = aqnsim.MidBSMEntanglementProtocol(
            self.sim_context,
            cport_name="BSMport",
            comm_qpos=self.qpos[0],
            mem_qpos=self.qpos[1],
            correction=True,
            log_fidelity=True,
            qmemory_name=self.qmemory_name,
        )
        self.add_subprotocol(entgen_proto)
        self.entgen_proto = entgen_proto

        # Track correction information gathered from repeater messages
        self.corr_messages = []
        self.correction_completed_signal_name = "CORRECTION COMPLETE"
        self.add_signal(self.correction_completed_signal_name)

        # signal to synchronize the starting of all protocols
        self.start_sync_signal_name = "START SYNC"
        self.add_signal(self.start_sync_signal_name)

        # Collect end-to-end entangled qubits
        self.entangled_qubits = []

    def cport_handler(self, msg: aqnsim.CMessage):
        """
        Handler for correction messages that are distributed by the repeaters.  Collects all expected messages
        before performing a correction on the local qubit.

        :param msg: A message containing the measurement results for a repeater's swap.
        """
        corr_message = msg.content
        # Note: should add header to make sure msg is of correct form
        self.corr_messages.append(corr_message)

        if len(self.corr_messages) == self.N:
            # Compute final correction (mod 2 of totals)
            z_corr = sum(corr[0] for corr in self.corr_messages) % 2
            x_corr = sum(corr[1] for corr in self.corr_messages) % 2
            self.corr_messages = []

            # Apply final correction
            aqnsim.eventlogger.log_generic_node_event(
                self.parent_component.name, "applying final correction"
            )
            self.apply_correction(x_corr, z_corr, qpos=self.qpos[1])

    @aqnsim.process
    def apply_correction(self, x_corr: int, z_corr: int, qpos: int):
        """
        Apply X and Z corrections as needed to the specified qubit in memory.  Send a signal upon completion of
        correction.

        :param x_corr: Indicates that we should perform an X-correction.  0 means no correction, 1 means correction.
        :param z_corr: Indicates that we should perform an Z-correction.  0 means no correction, 1 means correction.
        :param qpos: The position in memory to apply the correction to
        """
        if x_corr:
            aqnsim.eventlogger.log_apply_operation(
                self.parent_component.name, ops.X.name, [qpos]
            )
            yield self.qmem.operate(ops.X, qpos=qpos)
        if z_corr:
            aqnsim.eventlogger.log_apply_operation(
                self.parent_component.name, ops.Z.name, [qpos]
            )
            yield self.qmem.operate(ops.Z, qpos=qpos)
        aqnsim.eventlogger.log_generic_node_event(
            self.parent_component.name, "finished applying final correction"
        )

        self.send_signal(self.correction_completed_signal_name)
        aqnsim.eventlogger.log_tx_signal(
            self.parent_component.name, self.name, self.correction_completed_signal_name
        )

    def send_start_sync_msg_to_network(self):
        """Send message to all other nodes on the network which tells the time to start running the protocols"""
        port_name = "cport"
        start_time = self.now() + (self.N + 1) * self.elementary_link_classical_delay
        message = aqnsim.CMessage(
            sender=self.parent_component.name,
            action=START_SYNC_ACTION,
            status=aqnsim.StatusMessages.SUCCESS,
            content=start_time,
        )
        self.parent_component.send(msg=message, port_name=port_name)

    def send_end2end_entanglement_complete_msg(self):
        """Send message to end node (i.e. to Bob via the repeaters) that end to end entanglement has been completed"""
        port_name = "cport"
        message = aqnsim.CMessage(
            sender=self.parent_component.name,
            action=END_TO_END_EG_COMPLETE_ACTION,
            status=aqnsim.StatusMessages.SUCCESS,
        )
        self.parent_component.send(msg=message, port_name=port_name)

    def send_start_sync_signal(self):
        """Send signal to self to start this protocol"""
        self.send_signal(self.start_sync_signal_name)
        aqnsim.eventlogger.log_tx_signal(
            self.parent_component.name, self.name, self.start_sync_signal_name
        )

    def run(self):
        """Main run method for the protocol"""
        # send a message to start the overall end-to-end protocol. The message instructs the protocol instances
        # on all nodes to start at time t = t_now + (N+1)*CHANNEL_DELAY to make sure all nodes have received
        # the message first. After the wait, send a signal to self allowing the protocol to run once.
        self.send_start_sync_msg_to_network()
        yield self.wait((self.N + 1) * self.elementary_link_classical_delay)
        self.send_start_sync_signal()

        while True:
            # wait for global start sync signal which synchronizes start of all protocols
            yield self.await_signal(self, self.start_sync_signal_name)
            aqnsim.eventlogger.log_rx_signal(
                self.parent_component.name, self.name, self.start_sync_signal_name
            )

            elementary_ent_generated = False
            while not elementary_ent_generated:
                # Trigger the EEG subprotocol, thereby starting the entanglement generation attempt
                self.entgen_proto.trigger()

                # Wait for confirmation of elementary link entanglement from subprotocol
                yield self.await_signal(
                    self.entgen_proto, aqnsim.UniversalSignals.COMPLETED.name
                )

                # We apply the POVM directly in this case on a qubit and vacuum qubit (if the other
                # qubit got lost) so here look at the state as a proxy for determining which
                # detectors clicked.
                if self.entgen_proto.bsm_fidelity is not None:
                    elementary_ent_generated = True
            # Now Alice needs to wait for N messages from all N repeater in the chain
            # Given all messages, Alice will know what correction to apply
            aqnsim.simlogger.info(
                f"{self.parent_component.name} waiting for {self.N} message(s)"
            )
            yield self.await_signal(self, self.correction_completed_signal_name)
            aqnsim.eventlogger.log_rx_signal(
                self.parent_component.name,
                self.name,
                self.correction_completed_signal_name,
            )

            # Display remaining state
            qubit = self.qmem.positions[self.qpos[1]].peek()
            state = qubit.state.state
            aqnsim.eventlogger.log_generic_node_event(
                self.parent_component.name, "finished."
            )
            aqnsim.simlogger.info(
                f"Remaining qubit, of ID {qubit.qubit_id}, shares state "
                f"space with qubits {qubit.state.qubit_ids}, and with state \n{state}\n"
            )

            self.entangled_qubits.append(qubit)

            # Entanglement distribution complete. Let subprotocols know
            self.send_signal(aqnsim.UniversalSignals.COMPLETED.name)

            # send message to end node (Bob) that we have end-to-end entanglement
            self.send_end2end_entanglement_complete_msg()


class BobProtocol(aqnsim.NodeProtocol):
    """
    Protocol for Bob.

    :param sim_context: A SimulationContext object.
    :param qpos: The positions on qmemory that are involved. The first entry in
        qpos corresponds to the qubit position to be emitted, and the second entry
        to the qubit that will remain behind and, eventually, become entangled with Alice.
    :param name: Name of the protocol
    :param qmemory_name: The name of the memory used in this protocol.
    """

    def __init__(
        self,
        sim_context: aqnsim.SimulationContext,
        qpos: List[int] = None,
        name: str = "BobProtocol",
        qmemory_name: str = None,
    ):
        super().__init__(sim_context=sim_context, name=name)

        qpos = [0, 1] if qpos is None else qpos
        self.qs = sim_context.qs
        self.qpos = qpos
        self.wait_time = 0
        self.qmemory_name = qmemory_name

    def initialize(self, parent_component=None):
        super().initialize(parent_component)
        self.qmem = self.parent_component.subcomponents[self.qmemory_name]
        self.parent_component.ports["cport"].add_rx_input_handler(self.cport_handler)

        # signal to synchronize the start of all protocols
        self.start_sync_signal_name = "START SYNC"
        self.add_signal(self.start_sync_signal_name)

        # signal to synchronize the completion of protocols (end-to-end entanglement has been established)
        self.end_sync_signal_name = "END SYNC"
        self.add_signal(self.end_sync_signal_name)

        # Bob will need one instance of an elementary entanglement generation protocol
        entgen_proto = aqnsim.MidBSMEntanglementProtocol(
            self.sim_context,
            cport_name="BSMport",
            comm_qpos=self.qpos[0],
            mem_qpos=self.qpos[1],
            qmemory_name=self.qmemory_name,
            correction=False,
            log_fidelity=True,
        )
        self.add_subprotocol(entgen_proto)
        self.entgen_proto = entgen_proto

        # Collect end-to-end entangled qubits
        self.entangled_qubits = []

    def cport_handler(self, msg: aqnsim.CMessage):
        """
        Handler for received the "start_sync" and "end sync" message, which indicate when all
        protocls should start and when end-to-end entanglement has been completed

        :param msg: A message giving starting time or indication that end-to-end entanglement has completed
        """
        if msg.action == START_SYNC_ACTION:
            self.wait_time = msg.content - self.now()
            aqnsim.simlogger.info(
                f"BobProtocol: now:{self.now()}, start timestamp:{msg.content}, wait:{self.wait_time}"
            )
            self.send_signal(self.start_sync_signal_name)
            aqnsim.eventlogger.log_tx_signal(
                self.parent_component.name, self.name, self.start_sync_signal_name
            )
        elif msg.action == END_TO_END_EG_COMPLETE_ACTION:
            self.send_signal(self.end_sync_signal_name)
            aqnsim.eventlogger.log_tx_signal(
                self.parent_component.name, self.name, self.end_sync_signal_name
            )
            # If this message is received, the entanglement generation is complete.
            self.entanglement_generation_time = self.env.now
        else:
            raise ValueError(f"Unexpected message: {msg}")

    def run(self):
        """Main run method for the protocol."""
        while True:
            # wait for global start sync signal which synchronizes start of all protocols
            yield self.await_signal(self, self.start_sync_signal_name)
            aqnsim.eventlogger.log_rx_signal(
                self.parent_component.name, self.name, self.start_sync_signal_name
            )

            if self.wait_time > 0:
                yield self.wait(self.wait_time)

            elementary_ent_generated = False
            while not elementary_ent_generated:
                # Trigger the EEG subprotocol, thereby starting the entanglement generation attempt
                self.entgen_proto.trigger()

                # Wait for confirmation of elementary link entanglement from subprotocol
                yield self.await_signal(
                    self.entgen_proto, aqnsim.UniversalSignals.COMPLETED.name
                )

                # We apply the POVM directly in this case on a qubit and vacuum qubit (if the other
                # qubit got lost) so here look at the state as a proxy for determining which
                # detectors clicked.
                if self.entgen_proto.bsm_fidelity is not None:
                    elementary_ent_generated = True

            qubit = self.qmem.positions[self.qpos[1]].peek()
            aqnsim.eventlogger.log_generic_node_event(
                self.parent_component.name, "finished."
            )
            aqnsim.simlogger.info(f"Remaining qubit has ID {qubit.qubit_id}")
            self.entangled_qubits.append(qubit)

            # wait for end-to-end entanglement to be established
            yield self.await_signal(self, self.end_sync_signal_name)
            aqnsim.eventlogger.log_rx_signal(
                self.parent_component.name, self.name, self.end_sync_signal_name
            )

            # Entanglement distribution complete. Let subprotocols know
            self.send_signal(aqnsim.UniversalSignals.COMPLETED.name)


class RepeaterProtocol(aqnsim.NodeProtocol):
    """
    The repeater's job is to swap the entanglement generated across two elementary links
    in order to establish longer-range entanglement.

    The RepeaterProtocol (RP) will have two instances of the elementary entanglement generation
    protocol as subprotocols, one for each side of the repeater,

    The RP will wait until both subprotocols have succeeded in establishing entanglement, and
    will then proceed to do the entanglement swap via a Bell-state measurement (BSM), before
    sending the measurement results in a message to Alice.

    :param sim_context: A SimulationContext object to run the simulation in.
    :param port1_name: One of the node ports that connects to the entangling link ("left side")
    :param port2_name: One of the node ports that connects to the entangling link ("right side")
    :param qpos1: The positions on qmemory that are involved with entanglement generation via
        the port with name 'port1_name' ("left side" of repeater).
    :param qpos2: The positions on qmemory that are involved with entanglement generation via
        the port with name 'port2_name' ("right side" of repeater).
        For both qpos1 and qpos2, the first entry corresponds to the qubit position to be emitted,
        and the second entry to the qubit that will remain behind
    :param name: Name of the protocol
    :param qmemory_name: The name of the qmemory used in this protocol.
    """

    def __init__(
        self,
        sim_context: aqnsim.SimulationContext,
        port1_name: str = "BSMport1",
        port2_name: str = "BSMport2",
        qpos1: List[int] = None,
        qpos2: List[int] = None,
        name: str = None,
        qmemory_name: str = None,
    ):
        super().__init__(sim_context=sim_context, name=name)

        qpos1 = [0, 1] if qpos1 is None else qpos1
        qpos2 = [3, 2] if qpos2 is None else qpos2

        self.port1_name = port1_name
        self.port2_name = port2_name
        self.qs = sim_context.qs
        self.qpos1 = qpos1
        self.qpos2 = qpos2
        self.wait_time = 0
        self.qmemory_name = qmemory_name

    def initialize(self, parent_component=None):
        super().initialize(parent_component)
        self.qmem = self.parent_component.qmemory
        self.parent_component.ports["cport1"].add_rx_input_handler(self.cport1_handler)
        self.parent_component.ports["cport2"].add_rx_input_handler(self.cport2_handler)

        # signal to synchronize the start of all protocols
        self.start_sync_signal_name = "START SYNC"
        self.add_signal(self.start_sync_signal_name)

        # Repeater will need two instances of EntGen protocol for either side
        # Have the right side of each link (entgen_proto2) perform a correction
        entgen_proto1 = aqnsim.MidBSMEntanglementProtocol(
            self.sim_context,
            cport_name=self.port1_name,
            qmem_port_name="qport1",
            comm_qpos=self.qpos1[0],
            mem_qpos=self.qpos1[1],
            qmemory_name=self.qmemory_name,
            correction=False,
            log_fidelity=True,
        )
        entgen_proto2 = aqnsim.MidBSMEntanglementProtocol(
            self.sim_context,
            cport_name=self.port2_name,
            qmem_port_name="qport2",
            comm_qpos=self.qpos2[0],
            mem_qpos=self.qpos2[1],
            qmemory_name=self.qmemory_name,
            correction=True,
            log_fidelity=True,
        )
        self.add_subprotocol(entgen_proto1)
        self.add_subprotocol(entgen_proto2)
        self.entgen_proto1 = entgen_proto1
        self.entgen_proto2 = entgen_proto2

    def cport1_handler(self, msg: aqnsim.CMessage):
        """
        Handler for received the "start_sync" message, which indicates when all
        protocols should start.

        :param msg: A message giving starting time
        """
        self.parent_component.send(msg=msg, port_name="cport2")
        self.parent_component.send(msg=msg, port_name="cport")

        if msg.action == START_SYNC_ACTION:
            self.wait_time = msg.content - self.now()
            aqnsim.simlogger.info(
                f"RepeaterProtocol: now:{self.now()}, start timestamp:{msg.content}, wait:{self.wait_time}"
            )
            self.send_signal(self.start_sync_signal_name)
            aqnsim.eventlogger.log_tx_signal(
                self.parent_component.name, self.name, self.start_sync_signal_name
            )

    def cport2_handler(self, msg: aqnsim.CMessage):
        """
        Handler for the cport2 port.

        :param msg: A message giving starting time
        """
        self.parent_component.send(msg=msg, port_name="cport1")

    def run(self):
        """Main run method for the protocol"""
        while True:
            # wait for global start sync signal which synchronizes start of all protocols
            yield self.await_signal(self, self.start_sync_signal_name)
            aqnsim.eventlogger.log_rx_signal(
                self.parent_component.name, self.name, self.start_sync_signal_name
            )

            if self.wait_time > 0:
                yield self.wait(self.wait_time)

            elementary_ent1_generated = False
            elementary_ent2_generated = False
            while True:
                if not elementary_ent1_generated and not elementary_ent2_generated:
                    # Trigger the EEG subprotocols, thereby starting the entanglement
                    # generation attempt on either side
                    self.entgen_proto1.trigger()
                    self.entgen_proto2.trigger()

                    # Entanglement generation subprotocols are now doing their thing.
                    # Wait for entanglement success signals from both before performing swap
                    protocol_signals = [
                        (self.entgen_proto1, aqnsim.UniversalSignals.COMPLETED.name),
                        (self.entgen_proto2, aqnsim.UniversalSignals.COMPLETED.name),
                    ]
                    yield self.await_all_signals(protocol_signals)

                    # We apply the POVM directly in this case on a qubit and vacuum qubit (if the other
                    # qubit got lost) so here look at the state as a proxy for determining which
                    # detectors clicked.
                    if self.entgen_proto1.bsm_fidelity is not None:
                        elementary_ent1_generated = True
                    if self.entgen_proto2.bsm_fidelity is not None:
                        elementary_ent2_generated = True
                elif not elementary_ent1_generated:
                    self.entgen_proto1.trigger()

                    # Entanglement generation subprotocols are now doing their thing.
                    # Wait for entanglement success signals from both before performing swap
                    protocol_signals = [
                        (self.entgen_proto1, aqnsim.UniversalSignals.COMPLETED.name),
                    ]
                    yield self.await_all_signals(protocol_signals)

                    # We apply the POVM directly in this case on a qubit and vacuum qubit (if the other
                    # qubit got lost) so here look at the state as a proxy for determining which
                    # detectors clicked.
                    if self.entgen_proto1.bsm_fidelity is not None:
                        elementary_ent1_generated = True
                elif not elementary_ent2_generated:
                    self.entgen_proto2.trigger()

                    # Entanglement generation subprotocols are now doing their thing.
                    # Wait for entanglement success signals from both before performing swap
                    protocol_signals = [
                        (self.entgen_proto2, aqnsim.UniversalSignals.COMPLETED.name),
                    ]
                    yield self.await_all_signals(protocol_signals)

                    # We apply the POVM directly in this case on a qubit and vacuum qubit (if the other
                    # qubit got lost) so here look at the state as a proxy for determining which
                    # detectors clicked.
                    if self.entgen_proto2.bsm_fidelity is not None:
                        elementary_ent2_generated = True
                else:
                    break

            aqnsim.eventlogger.log_rx_signal(
                self.parent_component.name, self.name, "entanglement success"
            )

            # Now perform entanglement swap (same as Bell-state measurement)
            # The qubits to be operated on will be position 1 of qpos1 and qpos2,
            # since position 0 is the one being emitted
            meas_results = yield self.swap(qpos=[self.qpos1[1], self.qpos2[1]])

            # qubits have been measured and can now be reset. Inform subprotocols
            self.send_signal(aqnsim.UniversalSignals.COMPLETED.name)

            # Lastly, send result to Alice, so she can apply correction
            # Here we have equipped our repeater with dedicated classical ports for this
            aqnsim.eventlogger.log_generic_node_event(
                self.parent_component.name, "swap complete. Sending result to Alice"
            )
            port_name = "cport1"
            message = aqnsim.CMessage(
                sender=self.parent_component.name,
                action="SWAP_COMPLETE",
                status=aqnsim.StatusMessages.SUCCESS,
                content=meas_results,
            )
            self.parent_component.send(msg=message, port_name=port_name)

    @aqnsim.process
    def swap(self, qpos: List[int]) -> int:
        """
        Apply a bell state measurement circuit to a list of qubits.

        :param qpos: Qubit indices to use in the circuit
        """
        aqnsim.eventlogger.log_generic_node_event(
            self.parent_component.name, f"beginning swap with positions {qpos}"
        )
        bsm_circuit = aqnsim.BellStateMeasurementCircuit(qpos=qpos)
        circuit_results = yield self.qmem.run_circuit(circuit=bsm_circuit)
        meas_results = [circuit_results[qp] for qp in qpos]
        for qp in qpos:
            aqnsim.eventlogger.log_measurement_result(
                self.parent_component.name, qp, circuit_results[qp]
            )

        return meas_results


class EndNode(aqnsim.Node):
    """
    An 'end node' base class, i.e. what Alice and Bob will be in our scenario.

    Equip end node with a "BSMport" port, which we will connect with a BSMlink during
    network construction. Also add a default qmemory that connects to this port.

    :param sim_context: A SimulationContext object
    :param n: The number of qubit positions in node's qmemory
    :param op_delays: A dictionary specifying how much time operations take on the qmem
    :param meas_delay: How much time a measurement takes on the qmem
    :param name: The name of the node
    """

    def __init__(
        self,
        sim_context: aqnsim.SimulationContext,
        n: int,
        op_delays: Dict[str, Dict[int, Union[int, float, aqnsim.DelayModel]]] = None,
        meas_delay: Dict[str, Union[int, float, aqnsim.DelayModel]] = None,
        name: str = None,
    ):
        super().__init__(sim_context=sim_context, name=name)

        self.qs = sim_context.qs
        self.n = n

        self.add_port("BSMport")

        qmemory = aqnsim.QMemory(
            sim_context, n, meas_delay=meas_delay, name=f"QMemory-{name}"
        )
        self.add_qmemory(qmemory)
        qmemory.set_op_delays(op_delays=op_delays)
        qmemory.ports["qport"].forward_output_to_output(self.ports["BSMport"])


class Repeater(aqnsim.Node):
    """
    A repeater node base class.

    Equip this with two 'BSMports', which we will connect to BSMlinks on either side
    during network construction. Also add a default qmemory (needs at least 4 qubits)
    that will connect with these ports.

    :param sim_context: A SimulationContext object
    :param n: The number of qubit positions in node's qmemory
    :param qpos1: A list of indices indicating the positions for the flying qubits.
    :param qpos2: A list of indices indicating the positions for the stationary qubits.
    :param op_delays: A dictionary specifying how much time operations take on the qmem
    :param meas_delay: How much time a measurement takes on the qmem
    :param name: The name of the node
    """

    def __init__(
        self,
        sim_context: aqnsim.SimulationContext,
        n: int = 4,
        qpos1: List[int] = None,
        qpos2: List[int] = None,
        op_delays: Dict[str, Dict[int, Union[int, float, aqnsim.DelayModel]]] = None,
        meas_delay: Dict[str, Union[int, float, aqnsim.DelayModel]] = None,
        name=None,
    ):
        super().__init__(sim_context=sim_context, name=name)

        qpos1 = [0, 1] if qpos1 is None else qpos1
        qpos2 = [3, 2] if qpos2 is None else qpos2

        self.qs = sim_context.qs
        self.n = n

        # Add two ports, one for each side, for outputing qubits and receiving
        # measurements from BSMLinks
        self.add_ports(["BSMport1", "BSMport2"])

        # Also add dedicated ports for forwarding corrections along the chain
        # Here we will have corrections sent left (port 1) towards Alice
        self.add_ports(["cport1", "cport2"])

        # QMemory needs enough qubits to perform entangling-emit operations on
        # either side, meaning at least 4.
        # QMemory will also need two ports, which we will forward to the node ports
        qmemory = aqnsim.QMemory(
            sim_context, n, meas_delay=meas_delay, name=f"QMemory-{name}"
        )
        self.add_qmemory(qmemory)
        qmemory.set_op_delays(op_delays=op_delays)
        qmemory.add_ports(["qport1", "qport2"])
        qmemory.ports["qport1"].forward_output_to_output(self.ports["BSMport1"])
        qmemory.ports["qport2"].forward_output_to_output(self.ports["BSMport2"])

        # For convenience: Prepackage repeater with a repeater protocol & forward message to it
        rep_proto = RepeaterProtocol(
            sim_context, qpos1=qpos1, qpos2=qpos2, qmemory_name=f"QMemory-{name}"
        )
        self.add_protocol(rep_proto)


def setup_network(
    sim_context: aqnsim.SimulationContext,
    N: int,
    elementary_link_loss_in_db: Union[int, float, aqnsim.QNoiseModel],
    elementary_link_quantum_delay: Union[int, float],
    elementary_link_classical_delay: Union[int, float],
    H_delay: Union[int, float],
    X_delay: Union[int, float],
    Z_delay: Union[int, float],
    CNOT_delay: Union[int, float],
    meas_delay: Union[int, float],
    bsm_delay: Union[int, float],
    depolarizing_prob: Union[int, float],
):
    """
    :param env: The simpy Environment to run the simulation with.
    :param qs: The QuantumSimulator object which will run the simulation.
    :param N: The number of repeaters in the simulation.
    :param elementary_link_loss_in_db: The total loss, in dB, experienced over one
        elementary link in the repeater.
    :param elementary_link_quantum_delay: The channel delay time, in seconds, for qubits.
    :param elementary_link_classical_delay: The channel delay time, in seconds, for classical messages.
    :param H_delay: The delay (in seconds) on the Hadamard gate.
    :param X_delay: The delay (in seconds) on the X gate.
    :param Z_delay: The delay (in seconds) on the Z gate.
    :param CNOT_delay: The delay (in seconds) on the CNOT gate.
    :param meas_delay: The delay (in seconds) on measurement operations.
    :param bsm_delay: The delay (in seconds) on the Bell state measurement.
    :param depolarizing_prob: The probability of depolarizing noise applied to each qubit.
    """

    # Each node will have a QMemory.
    # Set how much time each gate takes by constructing a dictionary
    op_delays = {
        ops.H: H_delay,
        ops.X: X_delay,
        ops.Z: Z_delay,
        ops.I: 0 * SECOND,
        ops.CNOT: CNOT_delay,
    }

    # Now to set up the network. Start by initializing Network and attaching Alice, an end node
    network = aqnsim.Network(sim_context=sim_context)
    alice = EndNode(
        sim_context, n=2, op_delays=op_delays, meas_delay=meas_delay, name="Alice"
    )
    network.add_node(alice)

    # Implement qubit noise model
    qubit_noise_model = aqnsim.DepolarNoiseModel(qs=sim_context.qs, p=depolarizing_prob)

    # Now go each repeater and add them to the network, as well as BSMlinks to connect them
    repeater0 = None
    for i in range(N):
        # Right now, repeater comes pre-packaged with repeater protocol
        repeater1 = Repeater(
            sim_context, op_delays=op_delays, meas_delay=meas_delay, name=f"Repeater{i}"
        )
        network.add_node(repeater1)

        # To also simulate the minimum classical communication times needed (L0/c * 1/2) for a centrally
        # located BSM station, the BSM station is instead placed at an end node.
        bsm_link = aqnsim.MiddleBSMDualFiberLink(
            sim_context,
            length1=0,
            length2=1,
            attenuation_coeff=elementary_link_loss_in_db,
            loss=True,
            bsm_delay=elementary_link_quantum_delay,
            bsm_noise=qubit_noise_model,
            apply_msg_dependent_delay=False,
            refractive_index=1,
            name=f"bsm_link{i}",
        )

        c_link = aqnsim.ClassicalLink(
            sim_context, elementary_link_classical_delay, name=f"c_link{i}"
        )
        if i == 0:
            network.add_link(
                bsm_link,
                node1_name=alice.name,
                node2_name=repeater1.name,
                port1_name="BSMport",
                port2_name="BSMport1",
            )
            network.add_link(
                c_link,
                node1_name=alice.name,
                node2_name=repeater1.name,
                port1_name="cport",
                port2_name="cport1",
            )
        else:
            network.add_link(
                bsm_link,
                node1_name=repeater0.name,
                node2_name=repeater1.name,
                port1_name="BSMport2",
                port2_name="BSMport1",
            )
            network.add_link(
                c_link,
                node1_name=repeater0.name,
                node2_name=repeater1.name,
                port1_name="cport2",
                port2_name="cport1",
            )
        repeater0 = repeater1

    # Lastly, add Bob
    bob = EndNode(
        sim_context, n=2, op_delays=op_delays, meas_delay=meas_delay, name="Bob"
    )
    network.add_node(bob)
    bsm_link = aqnsim.MiddleBSMDualFiberLink(
        sim_context,
        length1=0,
        length2=1,
        attenuation_coeff=elementary_link_loss_in_db,
        loss=True,
        bsm_delay=elementary_link_quantum_delay,
        bsm_noise=qubit_noise_model,
        apply_msg_dependent_delay=False,
        refractive_index=1,
        name=f"bsm_link{i}",
    )

    c_link = aqnsim.ClassicalLink(
        sim_context, elementary_link_classical_delay, name=f"c_link{i + 1}"
    )
    network.add_link(
        bsm_link, repeater0.name, bob.name, port1_name="BSMport2", port2_name="BSMport"
    )
    network.add_link(
        c_link, repeater0.name, bob.name, port1_name="cport2", port2_name="cport"
    )

    # Equip Alice and Bob with respective protocols
    alice_protocol = AliceProtocol(
        sim_context,
        N=N,
        elementary_link_classical_delay=elementary_link_classical_delay,
        elementary_link_quantum_delay=elementary_link_quantum_delay,
        qmemory_name="QMemory-Alice",
    )
    bob_protocol = BobProtocol(sim_context, qmemory_name="QMemory-Bob")

    alice.add_protocol(alice_protocol)
    bob.add_protocol(bob_protocol)

    return network, alice_protocol, bob_protocol


def check_example(alice_protocol: AliceProtocol, bob_protocol: BobProtocol):
    """
    Check that Alice and Bob established the same number of end-to-end entanglements and that the qubits
    are within the same space.

    :param alice_protocol: Protocol attached to Node Alice
    :param bob_protocol: Protocol attached to Node Bob
    """
    alice_qubits = alice_protocol.entangled_qubits
    bob_qubits = bob_protocol.entangled_qubits

    try:
        # Check that Alice and Bob established the same number of end-to-end entanglements
        assert len(alice_qubits) == len(bob_qubits), (
            "Alice and Bob did not get the same number of end-to-end "
            f"entanglements, {len(alice_qubits)} and {len(bob_qubits)}"
        )
        assert len(alice_qubits) > 0, "No end-to-end entanglements were delivered"

        for a_qubit, b_qubit in zip(alice_qubits, bob_qubits):
            # Check that the qubits are within the same space
            assert a_qubit.state.qubit_ids == b_qubit.state.qubit_ids
            assert len(a_qubit.state.qubit_ids) == 2
            assert np.allclose(a_qubit.state.state, b_qubit.state.state)
        output_state = a_qubit.state.state
        state_fidelity = aqnsim.compute_fidelity(output_state, EXPECTED_STATE_DENSITY)
    except:
        state_fidelity = None

    entanglement_generation_time = bob_protocol.env.now
    return state_fidelity, entanglement_generation_time


def run_repeater_chain(
    random_seed: int,
    num_shots: int,
    num_repeaters: int,
    elementary_link_loss_in_db: Union[int, float, aqnsim.QNoiseModel],
    elementary_link_quantum_delay: Union[int, float, aqnsim.DelayModel],
    elementary_link_classical_delay: Union[int, float, aqnsim.DelayModel],
    H_delay: Union[int, float],
    X_delay: Union[int, float],
    Z_delay: Union[int, float],
    CNOT_delay: Union[int, float],
    meas_delay: Union[int, float],
    bsm_delay: Union[int, float],
    depolarizing_prob: Union[int, float],
):
    """
    Main run method.

    :param random_seed: Random seed for the simulation.
    :param num_shots: The number of simulation runs to perform.
    :param num_repeaters: Number of repeaters.
    :param elementary_link_loss_in_db: The total loss, in dB, experienced over one elementary
        link in the repeater.
    :param elementary_link_quantum_delay: The channel delay time, in seconds, for qubits.
    :param elementary_link_classical_delay: The channel delay time, in seconds, for classical messages.
    :param H_delay: The delay (in seconds) on the Hadamard gate.
    :param X_delay: The delay (in seconds) on the X gate.
    :param Z_delay: The delay (in seconds) on the Z gate.
    :param CNOT_delay: The delay (in seconds) on the CNOT gate.
    :param meas_delay: The delay (in seconds) on measurement operations.
    :param bsm_delay: The delay (in seconds) on the Bell state measurement.
    :param depolarizing_prob: The probability of depolarizing for each qubit.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    state_fidelities = []
    entanglement_generation_times = []
    state_fidelity = None

    for _ in range(num_shots):
        # The simulation context
        sim_context = aqnsim.SimulationContext(
            log_to_file=False, logging_level=0, defer_measurements=False
        )

        # Instantiate environment and QuantumSimulator
        while state_fidelity is None:
            # Configure logger to turn off logging
            aqnsim.simlogger.configure(env=sim_context.env)
            aqnsim.eventlogger.configure(env=sim_context.env)
            aqnsim.simlogger.set_level(0)
            aqnsim.eventlogger.set_level(0)

            # Setup network and protocols, and run sim until the given time
            _, alice_protocol, bob_protocol = setup_network(
                sim_context,
                num_repeaters,
                elementary_link_loss_in_db,
                elementary_link_quantum_delay,
                elementary_link_classical_delay,
                H_delay,
                X_delay,
                Z_delay,
                CNOT_delay,
                meas_delay,
                bsm_delay,
                depolarizing_prob,
            )
            sim_context.env.run()
            state_fidelity, entanglement_generation_time = check_example(
                alice_protocol, bob_protocol
            )

        state_fidelities.append(state_fidelity)
        entanglement_generation_times.append(entanglement_generation_time)
        state_fidelity = None

    return state_fidelities, entanglement_generation_times
