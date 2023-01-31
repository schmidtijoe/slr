import logging
import json
import numpy as np
from simple_parsing import ArgumentParser, helpers, choice
from dataclasses import dataclass
from pathlib import Path
from slr import utils


@dataclass
class GlobalSpecs(helpers.Serializable):
    gammaHz: float = 42577478.518  # [Hz/T]
    gammaPi: float = 2 * np.pi * gammaHz   # [rad/T]
    b0: float = 6.980963     # [T]
    maxGrad_in_mT: float = 40.0  # [mT/m]
    maxSlew: float = 200     # [T/m/s]
    eps: float = np.finfo(float).eps

    def __post_init__(self):
        self.maxGrad: float = 1e-3 * self.maxGrad_in_mT


@dataclass
class PulseSpecs(helpers.Serializable):
    sliceThickness_in_mm: float = 0.5     # [mm]
    pulseNumSamples: int = 512           # number of pulse samples
    pulseSampleCompression: int = 5     # us compressed to 1 sample: numSamples * compression = Duration_in_us
    pulseDuration_in_us: float = pulseNumSamples * pulseSampleCompression   # [us]
    phase: str = choice("linear", "minimum", default="linear")
    angle: int = 90     # ° flip angle of pulse
    pulseType: str = choice("excitation", "smalltip", "refocusing", "inversion", "saturation", default="excitation")
    ripple_1: float = 0.01
    ripple_2: float = 0.005

    def __post_init__(self):
        self.sliceThickness: float = 1e-3 * self.sliceThickness_in_mm
        self.pulseDuration: float = 1e-6 * self.pulseDuration_in_us
        self.fa: float = self.angle / 180 * np.pi


@dataclass
class SlrConfiguration:
    globals: GlobalSpecs = GlobalSpecs()
    pulse: PulseSpecs = PulseSpecs()

    def save(self, j_path):
        j_dict = {
            "globals": self.globals.to_dict(),
            "pulse": self.pulse.to_dict()
        }
        j_path = Path(j_path).absolute()
        utils.create_folder_ifn_exist(j_path)
        with open(j_path, "w") as j_file:
            j_file.write(json.dumps(j_dict, indent=2))

    @classmethod
    def load(cls, j_path: str) -> dataclass:
        slr_conf = cls()
        j_path = Path(j_path).absolute()
        if j_path.exists() and j_path.is_file():
            with open(j_path, "r") as j_file:
                j_dict = json.load(j_file)
            slr_conf.globals = GlobalSpecs.from_dict(j_dict['globals'])
            slr_conf.pulse = PulseSpecs.from_dict(j_dict['pulse'])
        else:
            raise ValueError("Could not find JSON configuration file")
        return slr_conf

    @classmethod
    def from_cmd_line_args(cls, args: ArgumentParser.parse_args) -> dataclass:
        # load any configuration file
        if "ConfigFile" in vars(args).keys():
            model_conf = cls.load(args.config.ConfigFile)
        else:
            model_conf = cls()
        # overwrite parameters explicitly given by cmd line
        for key, value in vars(args).items():
            model_conf.__setattr__(key, value)
        return model_conf


def create_command_line_parser() -> (ArgumentParser, ArgumentParser.parse_args):
    parser = ArgumentParser(prog="slr_pulse_tool")
    parser.add_arguments(GlobalSpecs, dest="globals")
    parser.add_arguments(PulseSpecs, dest="pulse")

    args = parser.parse_args()
    return parser, args
