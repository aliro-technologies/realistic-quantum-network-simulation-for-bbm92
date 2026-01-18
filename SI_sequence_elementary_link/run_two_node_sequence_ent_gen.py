"""
Code adapted from: https://github.com/sequence-toolbox/SeQUeNCe/blob/master/example/demo_for_beginners/two_node_eg.ipynb
"""

import json
import uuid
from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np

from sequence.kernel.timeline import Timeline
from sequence.topology.node import QuantumRouter
from sequence.components.optical_channel import ClassicalChannel, QuantumChannel
from sequence.resource_management.rule_manager import Rule
from sequence.constants import SPEED_OF_LIGHT

from node import (
    BSMNode,
)  # Modified version of BSMNode with modified SingleAtomBSM for two photons, defined in bsm.py
from basic_ent_gen_protocol import SimpleEntGenA

SPEED_OF_LIGHT_M_PER_S = SPEED_OF_LIGHT * 1e12
VERBOSE = False


# our rule condition requires RAW (unentangled) memories
def eg_rule_condition(memory_info, manager, args):
    if memory_info.state == "RAW":
        return [memory_info]
    else:
        return []


# define action to be taken when we meet our condition on router 1
def eg_rule_action1(memories_info, args):
    # define requirement of protocols on other node
    def eg_req_func(protocols, args):
        for protocol in protocols:
            if isinstance(protocol, SimpleEntGenA):
                return protocol

    # create entanglement generation protocol with proper parameters
    memories = [info.memory for info in memories_info]
    memory = memories[0]
    protocol = SimpleEntGenA.create(None, "EGA." + memory.name, "m1", "r2", memory)
    protocol.primary = True

    # return values for a rule are: the protocol created, the destination node,
    # the destination codition, and arguments for the condition.
    return [protocol, ["r2"], [eg_req_func], [None]]


# define action to be taken when we meet our condition on router 2
def eg_rule_action2(memories_info, args):
    memories = [info.memory for info in memories_info]
    memory = memories[0]
    protocol = SimpleEntGenA.create(None, "EGA." + memory.name, "m1", "r1", memory)
    return [protocol, [None], [None], [None]]


def sequence_ent_gen(sim_time, cc_delay, qc_atten, qc_dist, num_attempts):
    """
    :param sim_time: The duration of simulation time (ms).
    :param cc_delay: the delay on classical channels (ms).
    :param qc_atten: The attenuation on quantum channels (db/m).
    :param qc_dist: The distance of quantum channels (m).
    :param num_attempts: Number of shots to run.
    """

    PS_PER_MS = 1e9

    # convert units for cc delay (to ps) and qc distance (to m)
    cc_delay_in_ps = cc_delay * PS_PER_MS

    # construct the simulation timeline; the constructor argument is the simulation time (in ps)
    tl = Timeline(sim_time * PS_PER_MS)

    # first, construct the quantum routers
    # (with arguments for the node name, timeline, and number of quantum memories)
    r1 = QuantumRouter("r1", tl, num_attempts)
    r2 = QuantumRouter("r2", tl, num_attempts)
    # next, construct the BSM nodes
    # (with arguments for the node name, timeline, and the names of connected routers)
    m1 = BSMNode("m1", tl, ["r1", "r2"])

    r1.set_seed(1)
    r2.set_seed(2)
    m1.set_seed(3)

    # Set time resolution to 1 picosecond, detector efficiency to 1, and count rate (sets dead time)
    m1.components["m1.BSM"].update_detectors_params("efficiency", 1)
    m1.components["m1.BSM"].update_detectors_params("time_resolution", 1)
    m1.components["m1.BSM"].update_detectors_params("count_rate", 1000000000000000000)

    for node in [r1, r2]:
        # get memory array object from node (note: method returns list, so access first (should be only 1))
        memory_array = node.get_components_by_type("MemoryArray")[0]
        # we update the coherence time (measured in seconds) here
        memory_array.update_memory_params("coherence_time", 1000000000000000)
        # and similarly update the fidelity of entanglement for the memories
        memory_array.update_memory_params("raw_fidelity", 1.0)

    # create all-to-all classical connections
    nodes = [r1, r2, m1]
    for node1 in nodes:
        for node2 in nodes:
            if node1 == node2:
                continue
            # construct a classical communication channel
            # (with arguments for the channel name, timeline, length (in m), and delay (in ps))
            cc = ClassicalChannel(
                "_".join(["cc", node1.name, node2.name]),
                tl,
                qc_dist / 2,
                delay=cc_delay_in_ps,
            )
            cc.set_ends(node1, node2.name)

    # create linear quantum network between routers and middle node
    # for this, we create quantum channels
    # (with arguments for the channel name, timeline, attenuation (in dB/m), and distance (in m))
    qc1 = QuantumChannel("qc_r1_m1", tl, qc_atten, qc_dist / 2)
    qc1.set_ends(r1, m1.name)
    qc2 = QuantumChannel("qc_r2_m1", tl, qc_atten, qc_dist / 2)
    qc2.set_ends(r2, m1.name)

    # initialize our simulation kernel and instantiate the written rules
    tl.init()
    rule1 = Rule(10, eg_rule_action1, eg_rule_condition, None, None)
    r1.resource_manager.load(rule1)
    rule2 = Rule(10, eg_rule_action2, eg_rule_condition, None, None)
    r2.resource_manager.load(rule2)
    tl.run()

    # display our collected metrics
    data = []
    for info in r1.resource_manager.memory_manager:
        if info.entangle_time > 0:
            data.append(info.entangle_time / 1e12)  # to seconds
    data.sort()

    if VERBOSE:
        plt.plot(data, range(1, len(data) + 1), marker="o")
        plt.xlabel("Simulation Time (S)")
        plt.ylabel("Aggregated Number of Entangled Memory")
        plt.show()

        # display collected metric for memory fidelities on r1
        # in this case, a bar chart of memory fidelity at each index
        r1_fidelity_data = []
        for info in r1.resource_manager.memory_manager:
            r1_fidelity_data.append(info.fidelity)

        print("Fidelities:")
        print(sum(r1_fidelity_data) / len(r1_fidelity_data))

        plt.bar(range(len(r1_fidelity_data)), r1_fidelity_data)
        plt.ylim(0.5, 1)
        plt.title("r1")
        plt.ylabel("Fidelity")
        plt.xlabel("Memory Number")
        plt.show()

        # display collected metric for memory fidelities on r2
        # in this case, a bar chart of memory fidelity at each index
        r2_fidelity_data = []
        for info in r2.resource_manager.memory_manager:
            r2_fidelity_data.append(info.fidelity)

        plt.bar(range(len(r2_fidelity_data)), r2_fidelity_data)
        plt.ylim(0.5, 1)
        plt.title("r2")
        plt.ylabel("Fidelity")
        plt.xlabel("Memory Number")
        plt.show()

    minimum_time_between_attempts = (
        qc_dist / 2 / SPEED_OF_LIGHT_M_PER_S
    ) + cc_delay_in_ps * 1e-12
    time_between_attempts = np.array(data)

    entangle_time_in_seconds = np.mean(time_between_attempts)

    link_loss_in_db = qc_atten * qc_dist  # on the whole link (both arms)
    link_efficiency = pow(10, -link_loss_in_db / 10)

    # Include maximum efficiency of linear optical BSM measurement as 0.5
    BSM_efficiency = 0.5
    P0 = link_efficiency * BSM_efficiency

    print(f"Fraction of successful attempts: {len(data)/num_attempts}")
    print(
        f"Simulation avg time (S): {entangle_time_in_seconds} +/- {np.std(time_between_attempts)/np.sqrt(len(time_between_attempts))}"
    )

    print(f"Expected time (S): {minimum_time_between_attempts/P0}")

    print(
        f"Ratio of expected to sim time: {(minimum_time_between_attempts/P0)/entangle_time_in_seconds}"
    )

    normalized_entangle_time = time_between_attempts / (
        qc_dist / 2 / SPEED_OF_LIGHT_M_PER_S
    )

    if VERBOSE:
        plt.plot(
            normalized_entangle_time,
            range(1, len(normalized_entangle_time) + 1),
            marker="o",
        )
        plt.xlabel("Normalized time to entanglement (attempts)")
        plt.ylabel("Aggregated Number of Entangled Memory")
        plt.show()

    return list(normalized_entangle_time)


def main():
    num_attempts = 1000
    qm_delay_per_link_in_s = 50 * 1e-3
    distance_light_travels_in_one_link = SPEED_OF_LIGHT_M_PER_S * qm_delay_per_link_in_s
    cc_delay_in_ms = 0

    link_losses_in_db = [0, 1, 2, 3, 4, 5, 6, 7]

    ent_gen_times = []

    for link_loss_in_db in link_losses_in_db:
        loss_per_meter = link_loss_in_db / distance_light_travels_in_one_link

        ent_gen_set = sequence_ent_gen(
            sim_time=10000,
            cc_delay=cc_delay_in_ms,
            qc_atten=loss_per_meter,
            qc_dist=distance_light_travels_in_one_link,
            num_attempts=num_attempts,
        )
        ent_gen_times += [ent_gen_set]

    # Save compiled results
    simulation_results = {}
    simulation_results["link_losses_in_db"] = link_losses_in_db
    simulation_results["ent_gen_times"] = ent_gen_times
    simulation_results["num_shots"] = num_attempts
    simulation_results["BSM_efficiency"] = 0.5
    simulation_results["num_repeaters"] = 0
    # Set elementary link quantum delay to 1 because we normalize our results by the elementary quantum link delay
    simulation_results["elementary_link_quantum_delay"] = 1

    # Get filename based on a uuid.
    uuid_string = str(uuid.uuid4())
    file_name_string = datetime.utcnow().strftime("%H_%M_%S") + "_" + uuid_string

    with open(
        "SI_sequence_elementary_link/"
        + file_name_string
        + "_sequence_SI_sim_results_elementary_link"
        + ".json",
        "w",
    ) as file:
        file.write(json.dumps(simulation_results, indent=2))


if __name__ == "__main__":
    main()
