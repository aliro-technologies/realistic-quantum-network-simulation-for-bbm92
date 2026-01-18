"""
Code adapted from: https://github.com/sequence-toolbox/SeQUeNCe/blob/master/sequence/components/bsm.py
"""

from typing import Any
from sequence.components.circuit import Circuit
from sequence.components.detector import Detector
from sequence.utils import log
from sequence.components.bsm import BSM, _set_pure_state


class SingleAtomBSM(BSM):
    """Class modeling a single atom BSM device.

    Measures incoming photons and manages entanglement of associated memories.

    Attributes:
        name (str): label for BSM instance.
        timeline (Timeline): timeline for simulation.
        phase_error (float): phase error applied to measurement.
        detectors (list[Detector]): list of attached photon detection devices.
        resolution (int): maximum time resolution achievable with attached detectors.
    """

    _meas_circuit = Circuit(1)
    _meas_circuit.measure(0)

    def __init__(self, name, timeline, phase_error=0, detectors=None):
        """Constructor for the single atom BSM class.

        Args:
            name (str): name of the beamsplitter instance.
            timeline (Timeline): simulation timeline.
            phase_error (float): phase error applied to polarization qubits (unused) (default 0).
            detectors (list[dict]): list of parameters for attached detectors,
                in dictionary format (must be of length 2) (default None).
        """

        if detectors is None:
            detectors = [{}, {}]
        super().__init__(name, timeline, phase_error, detectors)
        self.encoding = "single_atom"
        assert len(self.detectors) == 2

    def get(self, photon, **kwargs):
        """See base class.

        This method adds additional side effects not present in the base class.

        Side Effects:
            May call get method of one or more attached detector(s).
            May alter the quantum state of photon and any stored photons, as well as their corresponding memories.
        """

        super().get(photon)
        log.logger.debug(self.name + " received photon")

        if len(self.photons) == 2:
            qm = self.timeline.quantum_manager
            p0, p1 = self.photons
            key0, key1 = p0.quantum_state, p1.quantum_state
            keys = [key0, key1]
            state0, state1 = qm.get(key0), qm.get(key1)
            meas0, meas1 = (
                qm.run_circuit(
                    self._meas_circuit, [key], self.get_generator().random()
                )[key]
                for key in keys
            )

            log.logger.debug(self.name + f" measured photons as {meas0}, {meas1}")

            if meas0 ^ meas1:  # meas0, meas1 = 1, 0 or 0, 1
                detector_num = self.get_generator().choice(
                    [0, 1]
                )  # randomly select a detector number
                if len(state0.keys) == 1:
                    log.logger.info(self.name + " passed stage 1")
                    if detector_num == 0:
                        _set_pure_state(keys, BSM._psi_minus, qm)
                    else:
                        _set_pure_state(keys, BSM._psi_plus, qm)
                else:
                    raise NotImplementedError("Unknown state")

                # Changed the line below from the original source code to apply loss from both arms
                if (
                    self.get_generator().random() > p0.loss
                    and self.get_generator().random() > p1.loss
                ):
                    log.logger.info(f"Triggering detector {detector_num}")
                    # middle BSM node notify two end nodes via EntanglementGenerationB.bsm_update()
                    self.detectors[detector_num].get()
                else:
                    log.logger.info(f"{self.name} lost photon p{meas1}")

            else:  # meas0, meas1 = 1, 1 or 0, 0
                if meas0 and self.get_generator().random() > p0.loss:
                    detector_num = self.get_generator().choice([0, 1])
                    self.detectors[detector_num].get()
                else:
                    log.logger.info(f"{self.name} lost photon p0")

                if meas1 and self.get_generator().random() > p1.loss:
                    detector_num = self.get_generator().choice([0, 1])
                    self.detectors[detector_num].get()
                else:
                    log.logger.info(f"{self.name} lost photon p1")

    def trigger(self, detector: Detector, info: dict[str, Any]):
        """See base class.

        This method adds additional side effects not present in the base class.

        Side Effects:
            May send a further message to any attached entities.
        """

        detector_num = self.detectors.index(detector)
        time = info["time"]

        res = detector_num
        info = {"entity": "BSM", "info_type": "BSM_res", "res": res, "time": time}
        self.notify(info)
