"""
Copyright © 2025, UChicago Argonne, LLC

All Rights Reserved

Software Name: SeQUeNCe: Simulator of QUantum Network Communication

By: Argonne National Laboratory, Illinois Institute of Technology, and 
Encore Consulting Services, Inc

OPEN SOURCE LICENSE

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors 
may be used to endorse or promote products derived from this software without 
specific prior written permission.


*****************************************************************************
DISCLAIMER

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE.
*****************************************************************************


Code adapted from: https://github.com/sequence-toolbox/SeQUeNCe/blob/master/sequence/topology/node.py
"""

from sequence.entanglement_management.generation import EntanglementGenerationB
from sequence.topology.node import Node
from bsm import SingleAtomBSM


class BSMNode(Node):
    """Bell state measurement node.

    This node provides bell state measurement and the EntanglementGenerationB protocol for entanglement generation.
    Creates a SingleAtomBSM object within local components.

    Attributes:
        name (str): label for node instance.
        timeline (Timeline): timeline for simulation.
        eg (EntanglementGenerationB): entanglement generation protocol instance.
    """

    def __init__(
        self,
        name: str,
        timeline: "Timeline",
        other_nodes: list[str],
        seed=None,
        component_templates=None,
    ) -> None:
        """Constructor for BSM node.

        Args:
            name (str): name of node.
            timeline (Timeline): simulation timeline.
            other_nodes (list[str]): 2-member list of node names for adjacent quantum routers.
        """

        super().__init__(name, timeline, seed)
        if not component_templates:
            component_templates = {}

        self.encoding_type = component_templates.get("encoding_type", "single_atom")

        # create BSM object with optional args
        bsm_name = name + ".BSM"
        if self.encoding_type == "single_atom":
            bsm_args = component_templates.get("SingleAtomBSM", {})
            bsm = SingleAtomBSM(bsm_name, timeline, **bsm_args)
        else:
            raise ValueError(f"Encoding type {self.encoding_type} not supported")

        self.add_component(bsm)
        self.set_first_component(bsm_name)

        self.eg = EntanglementGenerationB.create(self, f"{name}_eg", other_nodes)
        bsm.attach(self.eg)

    def receive_message(self, src: str, msg: "Message") -> None:
        """
        Signal to protocol that we've received a message
        """
        for protocol in self.protocols:
            if (
                protocol.protocol_type == msg.protocol_type
                or type(protocol) == msg.protocol_type
            ):
                if protocol.received_message(src, msg):
                    return

        # if we reach here, we didn't successfully receive the message in any protocol
        print(src, msg)
        raise Exception("Unknown protocol")

    def eg_add_others(self, other):
        """Method to add other protocols to entanglement generation protocol.

        Local entanglement generation protocol stores name of other protocol for communication.
        NOTE: entanglement generation protocol should be first protocol in protocol list.

        Args:
            other (EntanglementProtocol): other entanglement protocol instance.
        """

        self.protocols[0].others.append(other.name)
