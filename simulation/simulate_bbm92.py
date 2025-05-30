# Copyright © 2025 Aliro Technologies, Inc. All Rights Reserved.
# ALIRO QUANTUM is a registered trademark of Aliro Technologies, Inc.

# This software, including its source code and accompanying documentation
# (collectively, "Software"), is confidential and proprietary to Aliro Technologies, Inc. and is
# protected by intellectual property laws and treaties. Unauthorized reproduction, use,
# distribution, or disclosure of the Software or any part thereof, in any form, is strictly
# prohibited.
"""
This example simulates BBM92 secure key distribution.
An entangled photon source transmits entangled photons to two receiving nodes, Alice, and Bob.
The BBM92 QKD (quantum key distribution) protocol is used to establish a shared key.

When the example is run, two entangled photons are routed from the entangled photon source Phoebe to the
two nodes Alice and Bob. The two nodes are also connected through classical
channels, so they can communicate their basis results. When the example finishes running,
the two sifted keys will be logged and the QBER and sifted key rates will also be logged.
"""

from enum import Enum, auto
import time
import math

import numpy as np
from aqnsim.run_simulations import SimulationContext
from aqnsim import random_utilities

import aqnsim
from aqnsim import (
    SPEED_OF_LIGHT,
)
from get_secure_key_rate_error import get_secure_key_rate_error


BASIS_CHOICE_KEY = "basis_choice"  # key indexing choice of basis in classical messages
MEASUREMENTS_COMPLETE_SIGNAL_NAME = "measurements_complete"
CLASSICAL_MSG_RECEIVED_SIGNAL_NAME = "classical_message_received"


# possible basis choices used by the receiving nodes
class BasisChoices(Enum):
    HV_BASIS = 0  # Z-basis, Horizontal/Vertical
    DA_BASIS = auto()  # X-basis, Diagonal/Anti-diagonal


STATE_FORMALISM = aqnsim.StateFormalisms.DENSITY_MATRIX

DETECTOR_NAME_LOOKUP = {
    "Alice_detector_0_0": [0, BasisChoices.HV_BASIS, 0],
    "Alice_detector_0_1": [0, BasisChoices.HV_BASIS, 1],
    "Bob_detector_0_1": [1, BasisChoices.HV_BASIS, 1],
    "Bob_detector_0_0": [1, BasisChoices.HV_BASIS, 0],
    "Alice_detector_1_0": [0, BasisChoices.DA_BASIS, 0],
    "Bob_detector_1_0": [1, BasisChoices.DA_BASIS, 0],
    "Bob_detector_1_1": [1, BasisChoices.DA_BASIS, 1],
    "Alice_detector_1_1": [0, BasisChoices.DA_BASIS, 1],
}


"""
SendProtocol: used by Phoebe (sending node) to send entangled photon pairs with one photon sent to
each receiving node
"""


class SendProtocol(aqnsim.NodeProtocol):
    """
    Node protocol for sending entangled photon pairs.

    :param sim_context: The simulation context to use.
    :param source_pair_rate: The average pair generation rate, in counts per second (stochastic).
    :param num_shots: The number of shots to simulate.
    :param minimum_time_resolution: The minimum time resolution.
    :param detector jitter: RMS jitter of the detector.
    :param detector_dead_time: The detector dead time, in seconds.
    :param dark_count_rates: The dark count rate (in cps) per detector.
    :param name: The name of the protocol.
    :param qmemory_name: The name of the memory used in this protocol.
    """

    def __init__(
        self,
        sim_context: SimulationContext,
        source_pair_rate: float,
        num_shots: int,
        minimum_time_resolution: float,
        detector_jitter: float,
        detector_dead_time: float,
        dark_count_rates: dict,
        name: str = None,
        qmemory_name: str = None,
    ):
        super().__init__(sim_context=sim_context, name=name)
        self.qs = sim_context.qs
        self.source_pair_rate = source_pair_rate
        self.num_shots = num_shots
        self.minimum_time_resolution = minimum_time_resolution
        self.dead_time = detector_dead_time
        self.detection_jitter = detector_jitter
        self.source_delay_model = aqnsim.ExponentialDelayModel(lam=source_pair_rate)
        self.dark_count_delay_model = {
            i: aqnsim.ExponentialDelayModel(lam=dark_count_rates[i])
            for i in range(0, 8)
        }

        # For deferred measurement functionality, measurements local to Alice or Bob must be tracked here
        # and communicated at the end to Alice and Bob.
        self.alice_measurements = {}
        self.bob_measurements = {}
        self.alice_basis_choices = {}
        self.bob_basis_choices = {}

    def _send_measurements_complete_message(self):
        """
        Send a message to the receiving nodes indicating that all measurements have been completed
        """
        msg = aqnsim.CMessage(
            sender=self.name,
            action="MEASUREMENTS_COMPLETE",
            status=aqnsim.StatusMessages.SUCCESS,
            content={
                "alice_measurements": self.alice_measurements,
                "bob_measurements": self.bob_measurements,
                "alice_basis_choices": self.alice_basis_choices,
                "bob_basis_choices": self.bob_basis_choices,
            },
        )
        self.parent_component.ports["classical_channel_port"].rx_output(msg)

    @aqnsim.process
    def run(self):
        """
        Main run method of the SendProtocol--sends out multiple entangled photon
        pairs to the receiving nodes.
        """
        # In order to use deferred measurements, which speeds up the simulations and allows
        # many shots, triggering the source is necessary.
        # The stochastic time between photons is simulated after measurements are completed
        # by repeatedly drawing from a Poissonian distribution.
        self.parent_component.subcomponents["entangled_photon_source"].trigger()
        yield self.wait(1 / self.source_pair_rate)

        if len(self.qs.deferred_measurements) > 0:
            detection_data = self.qs.apply_deferred_measurements(
                num_shots=self.num_shots
            )
            times = detection_data["times"]
            detector_names = detection_data["detector_names"]

            # Create a dict of detection data from the source
            source_detection_times = {}
            prev_temporal_offset = 0
            for _, detector_outcomes in enumerate(detection_data["detector_outcomes"]):
                temporal_offset = (
                    self.source_delay_model.get_delay() + prev_temporal_offset
                )
                for detector_index, result in enumerate(detector_outcomes):
                    if result == 1:
                        # Sqrt(2) comes from the convolution of two Gaussians for two detector resolutions
                        jitter = random_utilities.normal(
                            loc=0, scale=self.detection_jitter / np.sqrt(2)
                        )
                        timestamp = temporal_offset + jitter + times[detector_index]
                        source_detection_times[timestamp] = detector_index
                prev_temporal_offset = temporal_offset

            # Create a dict of detection data from dark counts
            dark_count_detection_times = {}
            for detector_index in range(0, 8):
                prev_temporal_offset = 0
                temporal_offset = 0
                total_time = 1 / self.source_pair_rate * self.num_shots
                while temporal_offset < total_time:
                    temporal_offset = (
                        self.dark_count_delay_model[detector_index].get_delay()
                        + prev_temporal_offset
                    )
                    # Sqrt(2) comes from the convolution of two Gaussians for two detector resolutions
                    jitter = random_utilities.normal(
                        loc=0, scale=self.detection_jitter / np.sqrt(2)
                    )
                    timestamp = temporal_offset + jitter
                    dark_count_detection_times[timestamp] = detector_index
                    prev_temporal_offset = temporal_offset

            # Create a sorted list of all timestamps
            all_timestamps = sorted(
                list(source_detection_times.keys())
                + list(dark_count_detection_times.keys())
            )

            # Track the last detection to manually implement dead times,
            # as we are using deferred measurements.
            last_detections = {n: -self.dead_time for n in range(8)}
            test_counter = 0
            for timestamp in all_timestamps:
                source_detector = source_detection_times.get(timestamp)
                dark_count_detector = dark_count_detection_times.get(timestamp)
                detector_index = (
                    source_detector
                    if source_detector is not None
                    else dark_count_detector
                )

                detector_name = detector_names[detector_index]
                [is_bob, basis, detector] = DETECTOR_NAME_LOOKUP[detector_name]

                if timestamp - last_detections[detector_index] > self.dead_time:
                    test_counter += 1
                    if not is_bob:
                        self.alice_measurements[timestamp] = detector
                        self.alice_basis_choices[timestamp] = basis
                    else:
                        self.bob_measurements[timestamp] = detector
                        self.bob_basis_choices[timestamp] = basis
                    last_detections[detector_index] = timestamp
        # Send message to nodes to indicate measurement is complete
        self._send_measurements_complete_message()


"""
ReceiveProtocol: used by the nodes receiving entangled photons and establishing a shared
secret key.
"""


class ReceiveProtocol(aqnsim.NodeProtocol):
    """
    Node protocol used by Alice, and Bob. The nodes receive one photon
    from each entangled pair, measure the photons,
    and use the measurements to generate a shared secret key.

    :param sim_context: The simulation context to use.
    :param detector_jitter: The jitter of the detector, in seconds.
    :param name: The name of the protocol.
    :param qmemory_name: The name of the memory used in this protocol.
    """

    def __init__(
        self,
        sim_context: SimulationContext,
        detector_jitter: float,
        minimum_time_resolution: float,
        name: str = None,
        qmemory_name: str = None,
    ):
        super().__init__(sim_context=sim_context, name=name)

        # track the quantum simulator instance
        self.qs = sim_context.qs

        self.minimum_time_resolution = minimum_time_resolution

        # create signals
        self.add_signal(MEASUREMENTS_COMPLETE_SIGNAL_NAME)
        self.add_signal(CLASSICAL_MSG_RECEIVED_SIGNAL_NAME)

        # initialize tracking variables
        self.basis_choices = {}
        self.measurements = {}
        self.sifted_key = []
        self.classical_msg = None

        # Create a GaussianDelayModel to model detector jitter
        self.jitter_model = aqnsim.GaussianDelayModel(mean=0, std=detector_jitter)

    def initialize(self, parent_component=None):
        super().initialize(parent_component)

        # set up classical channel input handler
        self.parent_component.ports["classical_channel_port"].add_rx_input_handler(
            handler=self.input_handler_classical
        )

    def input_handler_classical(self, msg: aqnsim.CMessage):
        """Handler for messages communicated over the classical channel"""
        if not isinstance(msg, aqnsim.CMessage):
            aqnsim.simlogger.warning(
                f"classical channel input handler at {self.parent_component.name} received message with unexpected data type"
            )
        classical_msg_action = msg.action
        if classical_msg_action == "MEASUREMENTS_COMPLETE":
            # Stop processing new photodetector counts
            node_name = self.parent_component.name
            self.parent_component.subcomponents[f"{node_name}_detector_0_0"].ports[
                "cout0"
            ].rx_output_handlers = []
            self.parent_component.subcomponents[f"{node_name}_detector_0_1"].ports[
                "cout0"
            ].rx_output_handlers = []
            self.parent_component.subcomponents[f"{node_name}_detector_1_0"].ports[
                "cout0"
            ].rx_output_handlers = []
            self.parent_component.subcomponents[f"{node_name}_detector_1_1"].ports[
                "cout0"
            ].rx_output_handlers = []

            # Save basises and measurements
            if self.parent_component.name == "Bob":
                self.basis_choices = msg.content["bob_basis_choices"]
                self.measurements = msg.content["bob_measurements"]
            else:
                self.basis_choices = msg.content["alice_basis_choices"]
                self.measurements = msg.content["alice_measurements"]

            # send a signal that all measurements have been completed
            self.send_signal(MEASUREMENTS_COMPLETE_SIGNAL_NAME)
        elif classical_msg_action == "TRANSMIT_BASIS":
            self.classical_msg = msg.content["basis_choices"]
            self.send_signal(CLASSICAL_MSG_RECEIVED_SIGNAL_NAME)
        else:
            raise ValueError(
                f"Unexpected message action {classical_msg_action} received."
            )

    def _process_classical_message(self, node_name):
        """Method for processing classical messages from the peer node"""
        # process classical message related to basis choices to obtain a raw sifted key
        if BASIS_CHOICE_KEY in self.classical_msg:
            measurements = self.measurements
            basis_choices = self.basis_choices

            common_basis_choice_times = []
            remote_basis_choices = self.classical_msg[BASIS_CHOICE_KEY]

            # Create two pointers to loop through and find coincidences
            p1, p2 = 0, 0
            local_timestamps = sorted(list(basis_choices.keys()))
            remote_timestamps = sorted(list(remote_basis_choices.keys()))
            last_local_coinc_time = -self.minimum_time_resolution
            last_remote_coinc_time = -self.minimum_time_resolution

            while p1 < len(basis_choices) and p2 < len(remote_basis_choices):
                # If the basis choices match and the timestamps are close enough, add to the common_basis_choice_times list
                local_time = local_timestamps[p1]
                remote_time = remote_timestamps[p2]
                timestamp_difference = local_time - remote_time

                # Divide resolution by 2 because we take abs of the timestamp_difference
                if abs(timestamp_difference) < self.minimum_time_resolution / 2:
                    if (
                        basis_choices[local_time]
                        == remote_basis_choices[remote_timestamps[p2]]
                    ):
                        last_coinc_diff_local = local_time - last_local_coinc_time
                        last_coinc_diff_remote = remote_time - last_remote_coinc_time

                        # Don't collect multiple coincidences within the same coincidence window
                        if (
                            abs(last_coinc_diff_local) > self.minimum_time_resolution
                            and abs(last_coinc_diff_remote)
                            > self.minimum_time_resolution
                        ):
                            common_basis_choice_times.append(local_time)
                            last_local_coinc_time = local_time
                            last_remote_coinc_time = remote_time
                    p1 += 1
                    p2 += 1
                elif timestamp_difference < 0:
                    p1 += 1
                else:
                    p2 += 1

            self.sifted_key = [
                measurements[common_basis_choice_times[k]]
                for k in range(len(common_basis_choice_times))
            ]

    def _send_basis_choices(self, node_name):
        """
        Send our list of basis choices to our peer node
        """
        basis_choices = {BASIS_CHOICE_KEY: self.basis_choices}

        msg = aqnsim.CMessage(
            sender=self.name,
            action="TRANSMIT_BASIS",
            status=aqnsim.StatusMessages.SUCCESS,
            content={"basis_choices": basis_choices},
        )
        self.parent_component.ports["classical_channel_port"].rx_output(msg)

    @aqnsim.process
    def run(self):
        """Main run method of the protocol"""
        yield self.await_signal(self, MEASUREMENTS_COMPLETE_SIGNAL_NAME)
        self._send_basis_choices(self.parent_component.name)
        yield self.await_signal(self, CLASSICAL_MSG_RECEIVED_SIGNAL_NAME)
        self._process_classical_message(self.parent_component.name)


"""
Set up the network nodes, links, optical components, and protocols. The internal structure
of the nodes is as follows:

Send Node (Phoebe)
    EntangledPolarizationSource

Receiving Node (Alice, Bob)
    Beamsplitter (BS)
    BS Outport 0:
        PolarizingBeamSplitter (PBS)
            Detector 0
            Detector 1
    BS Outport 1
        Half-Waveplate (HWP)
        PolarizingBeamSplitter (PBS)
            Detector 0
            Detector 1

"""


def setup_network(
    sim_context: SimulationContext,
    channel_length: float,
    channel_delay: float,
    link_loss_in_db_a: float,
    link_loss_in_db_b: float,
    source_pair_rate: float,
    source_visibility: float,
    source_wavelength: float,
    source_bandwidth_fwhm_wavelength: float,
    dark_count_rates: dict,
    detector_dead_time: float,
    detector_jitter: float,
    detector_freq_width: float,
    detector_maximum_efficiency: float,
    minimum_time_resolution: float,
    num_shots: int,
):
    """
    Set up the network for the BBM92 protocol.

    :param sim_context: The simulation context used for the example.
    :param channel_length: Channel length, in meters.
    :param channel_delay: Latency for quantum and classical links.
    :param link_loss_in_db: Loss per link, in dB.
    :param source_pair_rate: The source brightness immediately out of the source,
        in counts per second. Calculated as CAR = 1 / (B * t_cc) where CAR is
        coincidence to accidental ratio, B is brightness, and t_cc is collection time.
    :param source_visibility: The source visibility.
    :param source_wavelength: The source wavelength.
    :param source_bandwidth_fwhm_wavelength: The source FWHM bandwidth, in wavelengths.
    :param dark_count_rates: The source dark count rates per detector, in counts per second.
    :param detector_jitter: The detector jitter, in seconds.
    :param detector_dead_time: The detector dead time, in seconds.
    :param detector_freq_width: The detector frequency width, in Hz.
    :param detector_maximum_efficiency: The detector max efficiency (unitless fraction).
    :param minimum_time_resolution: The detector minimum time resolution, in seconds.
    :param num_shots: The number of shots to simulate per analytic sampling.
    """
    # Instantiate nodes and network that contains them
    alice = aqnsim.Node(
        sim_context=sim_context, name="Alice"
    )  # receives one side of entangled photon pair
    bob = aqnsim.Node(
        sim_context=sim_context, name="Bob"
    )  # receives other side of entangled photon pair
    phoebe = aqnsim.Node(
        sim_context=sim_context, name="Phoebe"
    )  # produces entangled photon pairs
    network = aqnsim.Network(sim_context=sim_context, nodes=[alice, bob, phoebe])

    _setup_source_node(
        sim_context=sim_context,
        node=phoebe,
        source_visibility=source_visibility,
        source_wavelength=source_wavelength,
        source_bandwidth_fwhm_wavelength=source_bandwidth_fwhm_wavelength,
    )

    # set up receiving nodes--Alice and Bob each receive one side of an entangled photon pair
    receiving_nodes = [alice, bob]
    for node in receiving_nodes:
        _setup_receiving_node(
            sim_context=sim_context,
            node=node,
            source_wavelength=source_wavelength,
            detector_maximum_efficiency=detector_maximum_efficiency,
            detector_freq_width=detector_freq_width,
        )

    # create fiber links for distributing entangled photons
    fiber_link_alice = aqnsim.FiberLink(
        sim_context=sim_context,
        length=channel_length,
        insertion_losses={"port1": link_loss_in_db_a},
    )
    fiber_link_bob = aqnsim.FiberLink(
        sim_context=sim_context,
        length=channel_length,
        insertion_losses={"port1": link_loss_in_db_b},
    )

    # create classical links for coordination with source
    classical_link_phoebe_alice = aqnsim.ClassicalLink(
        sim_context=sim_context, delay=channel_delay, name="classical_link_phoebe_alice"
    )
    classical_link_phoebe_bob = aqnsim.ClassicalLink(
        sim_context=sim_context, delay=channel_delay, name="classical_link_phoebe_bob"
    )
    classical_link_bob_alice = aqnsim.ClassicalLink(
        sim_context=sim_context, delay=channel_delay, name="classical_link_bob_alice"
    )

    # connect network links
    network.add_link(
        link=fiber_link_alice,
        node1_name="Phoebe",
        node2_name="Alice",
        port1_name="entangled_photon_output_0_port",
        port2_name="entangled_photon_input_port",
    )
    network.add_link(
        link=fiber_link_bob,
        node1_name="Phoebe",
        node2_name="Bob",
        port1_name="entangled_photon_output_1_port",
        port2_name="entangled_photon_input_port",
    )
    network.add_link(
        link=classical_link_bob_alice,
        node1_name="Alice",
        node2_name="Bob",
        port1_name="classical_channel_port",
        port2_name="classical_channel_port",
    )
    network.add_link(
        link=classical_link_phoebe_alice,
        node1_name="Alice",
        node2_name="Phoebe",
        port1_name="classical_channel_port",
        port2_name="classical_channel_port",
    )
    network.add_link(
        link=classical_link_phoebe_bob,
        node1_name="Bob",
        node2_name="Phoebe",
        port1_name="classical_channel_port",
        port2_name="classical_channel_port",
    )

    # attach protocols to the nodes
    phoebe_protocol = SendProtocol(
        sim_context=sim_context,
        source_pair_rate=source_pair_rate,
        num_shots=num_shots,
        minimum_time_resolution=minimum_time_resolution,
        detector_jitter=detector_jitter,
        detector_dead_time=detector_dead_time,
        dark_count_rates=dark_count_rates,
    )
    phoebe.add_protocol(phoebe_protocol)
    alice_protocol = ReceiveProtocol(
        sim_context=sim_context,
        detector_jitter=detector_jitter,
        minimum_time_resolution=minimum_time_resolution,
    )
    alice.add_protocol(alice_protocol)
    bob_protocol = ReceiveProtocol(
        sim_context=sim_context,
        detector_jitter=detector_jitter,
        minimum_time_resolution=minimum_time_resolution,
    )
    bob.add_protocol(bob_protocol)

    return network


def _setup_source_node(
    sim_context: aqnsim.SimulationContext,
    node: aqnsim.Node,
    source_visibility: float,
    source_wavelength: float,
    source_bandwidth_fwhm_wavelength: float,
):
    """
    Set up a node that emits Werner states encoded in polarization.

    :param sim_context: The simulation context to use.
    :param node: The node that will have the entanglement source added to it.
    :param source_visibility: The source visibility.
    :param source_wavelength: The source wavelength.
    :param source_bandwidth_fwhm_wavelength: The source FWHM bandwidth, in wavelengths.
    """
    # set up entangled photon source node (Phoebe)
    node.add_port(port_name="entangled_photon_output_0_port")
    node.add_port(port_name="entangled_photon_output_1_port")
    node.add_port(port_name="classical_channel_port")

    # Different mechanisms can lower visibility; using `get_noisy_werner` mixes the Bell state with the maximally
    # mixed state to be conservative.
    source_est_qber = (1 - source_visibility) / 2  # Estimated QBER
    if STATE_FORMALISM == aqnsim.StateFormalisms.DENSITY_MATRIX:
        # Fidelity is 1 - (qber_x + qber_y + qber_z)/2 for a Bell state
        state_distribution = [
            (
                1,
                aqnsim.get_noisy_werner(
                    state="phi_plus", fidelity=1 - source_est_qber * 3 / 2
                ),
            )
        ]
    else:
        sim_context.simlogger.warning(
            "Werner state not implemented for state vector formalism."
        )
        state_distribution = [(1, aqnsim.BELL_STATES["phi_plus"])]

    state_model = aqnsim.StateModel(
        state_distribution=state_distribution, formalism=STATE_FORMALISM
    )

    source_bandwidth_st_dev = source_bandwidth_fwhm_wavelength / (
        2 * np.sqrt(2 * np.log(2))
    )
    source_freq = SPEED_OF_LIGHT / source_wavelength
    source_freq_width = (
        SPEED_OF_LIGHT * source_bandwidth_st_dev / (source_wavelength**2)
    )

    entangled_photon_source = aqnsim.EntangledPolarizationSource(
        sim_context=sim_context,
        state_model=state_model,
        name="entangled_photon_source",
        mode_shape=aqnsim.GaussianModeShape(
            frequency=source_freq, frequency_width=source_freq_width
        ),
        clock=None,
    )
    entangled_photon_source.ports["qout0"].forward_output_to_output(
        node.ports["entangled_photon_output_0_port"]
    )
    entangled_photon_source.ports["qout1"].forward_output_to_output(
        node.ports["entangled_photon_output_1_port"]
    )
    node.add_subcomponent(entangled_photon_source)


def _setup_receiving_node(
    sim_context: aqnsim.SimulationContext,
    node: aqnsim.Node,
    source_wavelength: float,
    detector_maximum_efficiency: float,
    detector_freq_width: float,
):
    """
    Set up receiving node.

    :param sim_context: The simulation context to run the simulation.
    :param node: The Node which will be the recieving node.
    :param source_wavelength: The source wavelength.
    :param detector_maximum_efficiency: The detector max efficiency (unitless fraction).
    :param detector_freq_width: The detector frequency width, in Hz.
    """
    # add port for receiving photons and another for classical communication
    node.add_port(port_name="entangled_photon_input_port")
    node.add_port(port_name="classical_channel_port")

    # create a beamsplitter to process input photons, the output port of the processed photon
    # effectively chooses the measurement basis randomly by choosing which PBS subnetwork to
    # use, one of which includes a half-wave plate in the path before the PBS
    bs = aqnsim.BeamSplitter(sim_context=sim_context, name="bs")
    node.add_subcomponent(bs)
    node.ports["entangled_photon_input_port"].forward_input_to_input(
        bs.ports[aqnsim.BS_LEFT_INPUT_PORT_NAME]
    )

    # create a polarizing beamsplitter and two detectors for each output port
    # of the input beamsplitter
    bs_outport_names = [
        aqnsim.BS_LEFT_OUTPUT_PORT_NAME,
        aqnsim.BS_RIGHT_OUTPUT_PORT_NAME,
    ]
    pbs_outport_names = [aqnsim.OUTPUT_PORT_0_NAME, aqnsim.OUTPUT_PORT_1_NAME]

    for n in range(2):
        # add polarizing beamsplitter (PBS)
        pbs = aqnsim.PolarizingBeamSplitter(sim_context=sim_context, name=f"pbs{n}")
        node.add_subcomponent(pbs)

        # add a half-wave plate on one of the output ports of the input beamsplitter
        # note: the choice of angle = pi/8, which corresponds to theta = pi/4 on the
        # Bloch sphere, and phase phi = pi for the HWP will transform |H> to |+>
        # and |V> to |->, i.e. a Z to X basis transformation
        if n == 1:
            hwp = aqnsim.HalfWavePlate(
                sim_context=sim_context, angle=np.pi / 8, name="hwp"
            )
            bs.ports[bs_outport_names[n]].forward_output_to_input(
                hwp.ports[aqnsim.INPUT_PORT_0_NAME]
            )
            hwp.ports[aqnsim.OUTPUT_PORT_0_NAME].forward_output_to_input(
                pbs.ports[aqnsim.INPUT_PORT_0_NAME]
            )
            node.add_subcomponent(hwp)
        else:
            bs.ports[bs_outport_names[n]].forward_output_to_input(
                pbs.ports[aqnsim.INPUT_PORT_0_NAME]
            )

        source_freq = SPEED_OF_LIGHT / source_wavelength
        detector_efficiency = {
            aqnsim.MAX_EFFICIENCY_KEY: detector_maximum_efficiency,
            aqnsim.FREQUENCY_PARAM_KEY: source_freq,
            aqnsim.FREQUENCY_WIDTH_PARAM_KEY: detector_freq_width,
        }  # Efficiency of detector
        # add photon detectors
        for k in range(2):
            detector = aqnsim.PhotonDetector(
                sim_context=sim_context,
                number_resolving=False,
                error_on_fail=False,
                delete_measured_qubit=True,
                efficiency_params=detector_efficiency,
                name=f"{node.name}_detector_{n}_{k}",
            )

            node.add_subcomponent(detector)
            pbs.ports[pbs_outport_names[k]].forward_output_to_input(
                detector.ports[aqnsim.PD_INPUT_PORT_NAME]
            )


"""
Run the BBM92 example
"""


def compute_QBER(node0_key: list, node1_key: list):
    """
    Estimate QBER between two nodes receiving entangled photons.

    :param node0_key: the first node's sifted key; a list of 0 and 1
    :param node1_key: the second node's sifted key; a list of 0 and 1
    """
    N = len(node0_key)
    if N == 0:
        qber = 1
    else:
        qber = 1 - (sum(1 if node0_key[i] == node1_key[i] else 0 for i in range(N)) / N)
    return qber


def run_key_gen(
    channel_length: float,
    channel_delay: float,
    link_loss_in_db_a: float,
    link_loss_in_db_b: float,
    source_pair_rate: float,
    source_visibility: float,
    source_wavelength: float,
    source_bandwidth_fwhm_wavelength: float,
    dark_count_rates: float,
    detector_jitter: float,
    detector_dead_time: float,
    detector_freq_width: float,
    detector_maximum_efficiency: float,
    minimum_time_resolution: float,
    num_shots: int,
    random_seed: int,
):
    """
    Runs the simulation to generate keys and simulate QBER.

    :param channel_length: Channel length, in meters.
    :param channel_delay: Latency for quantum and classical links.
    :param link_loss_in_db_a: Loss per link, in dB, for Alice.
    :param link_loss_in_db_b: Loss per link, in dB, for Bob.
    :param source_pair_rate: The source brightness immediately out of the source,
        in counts per second. Calculated as CAR = 1 / (B * t_cc) where CAR is
        coincidence to accidental ratio, B is brightness, and t_cc is collection time.
    :param source_visibility: The source visibility.
    :param source_wavelength: The source wavelength.
    :param source_bandwidth_fwhm_wavelength: The source FWHM bandwidth, in wavelengths.
    :param dark_count_rates: The source dark count rate, in counts per second, per detector.
    :param detector_jitter: The detector jitter, in seconds.
    :param detector_dead_time: The detector dead time, in seconds.
    :param detector_freq_width: The detector frequency width, in Hz.
    :param detector_maximum_efficiency: The detector max efficiency (unitless fraction).
    :param minimum_time_resolution: The detector minimum time resolution, in seconds.
    :param num_shots: The number of shots to simulate per analytic sampling.
    :param random_seed: The integer to seed the random number generator with.
    """
    np.random.seed(random_seed)

    start_time = time.time()

    # The simulation context
    sim_context = aqnsim.SimulationContext(
        log_to_file=False, logging_level=0, defer_measurements=True
    )
    sim_context.qs = aqnsim.QuantumSimulator(
        state_formalism=STATE_FORMALISM, defer_measurements=True
    )

    aqnsim.simlogger.info("creating simulation environment and quantum simulator")

    # setup network and protocols, and run sim until the given time
    aqnsim.simlogger.info("setting up network topology")
    network = setup_network(
        sim_context=sim_context,
        channel_length=channel_length,
        channel_delay=channel_delay,
        link_loss_in_db_a=link_loss_in_db_a,
        link_loss_in_db_b=link_loss_in_db_b,
        source_pair_rate=source_pair_rate,
        source_visibility=source_visibility,
        source_wavelength=source_wavelength,
        source_bandwidth_fwhm_wavelength=source_bandwidth_fwhm_wavelength,
        dark_count_rates=dark_count_rates,
        detector_jitter=detector_jitter,
        detector_dead_time=detector_dead_time,
        detector_freq_width=detector_freq_width,
        detector_maximum_efficiency=detector_maximum_efficiency,
        minimum_time_resolution=minimum_time_resolution,
        num_shots=num_shots,
    )

    total_secure_key_bits = 0
    sim_time = 1 / source_pair_rate
    # run the simulation in batches in order to calculate QBER and key rate
    sim_context.env.run(until=sim_time * 5)

    # post-process the results
    receiving_node0 = network.subcomponents["Alice"]
    receiving_node1 = network.subcomponents["Bob"]
    receiving_node0_protocol = receiving_node0.protocols["ReceiveProtocol"]
    receiving_node1_protocol = receiving_node1.protocols["ReceiveProtocol"]

    # Compute QBER
    qber = compute_QBER(
        receiving_node0_protocol.sifted_key, receiving_node1_protocol.sifted_key
    )

    # Estimate sifted key rate
    raw_sifted_key_size = len(receiving_node0_protocol.sifted_key)
    binary_entropy = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)

    if math.isnan(binary_entropy):
        secure_key_size = 0
    else:
        secure_key_size = max(raw_sifted_key_size * (1 - 2.1 * binary_entropy), 0)
    total_secure_key_bits += secure_key_size

    # find key rate
    raw_key_rate = raw_sifted_key_size * source_pair_rate / num_shots

    # Find secure key rate
    secure_key_rate = max(raw_key_rate * (1 - 2.1 * binary_entropy), 0)

    # Find errors:
    raw_key_rate_error = np.sqrt(raw_sifted_key_size) * source_pair_rate / num_shots
    # Find error as standard deviation of a binomial distribution divided by the square root of the number of measurements
    qber_error = np.sqrt(qber * (1 - qber)) / np.sqrt(raw_sifted_key_size)

    # Find secure key rate error
    secure_key_rate_error = get_secure_key_rate_error(
        raw_key_rate, raw_key_rate_error, qber, qber_error
    )

    end_time = time.time()

    print(f"Time elapsed: {end_time - start_time}")
    print(f"Raw key size: {raw_sifted_key_size}")
    print(f"Raw key rate: {raw_key_rate}")
    print(f"Key length: {len(receiving_node0_protocol.sifted_key)}")
    print(f"QBER of keys: {qber}")

    # End loop and return key rate, QBER
    return (
        secure_key_rate,
        secure_key_rate_error,
        raw_key_rate,
        raw_key_rate_error,
        qber,
        qber_error,
    )
