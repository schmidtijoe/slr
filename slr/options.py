import logging
import json
import numpy as np
from simple_parsing import ArgumentParser, helpers, choice, field
from dataclasses import dataclass
from pathlib import Path
from slr import utils

logModule = logging.getLogger(__name__)


@dataclass
class FileConfig(helpers.Serializable):
    configFile: str = field(default="", alias=["-c"])
    outputPulseFile: str = field(default="", alias=["-o"])
    visualize: bool = field(default=True, alias=["-v"])

    def __post_init__(self):
        self.o_p = Path(self.outputPulseFile).absolute()
        if self.o_p.is_file():
            self.o_p = self.o_p.parent


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
    pulseNumSamples: int = 850           # number of pulse samples
    pulseDuration_in_us: float = 1700   # [us]
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
    f_config: FileConfig = FileConfig()
    globals: GlobalSpecs = GlobalSpecs()
    pulse: PulseSpecs = PulseSpecs()

    def save(self, j_path):
        j_dict = {
            "config": self.f_config.to_dict(),
            "globals": self.globals.to_dict(),
            "pulse": self.pulse.to_dict()
        }
        j_path = Path(j_path).absolute()
        if j_path.is_file():
            j_path = j_path.parent
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
            slr_conf.f_config = FileConfig.from_dict(j_dict['config'])
            slr_conf.globals = GlobalSpecs.from_dict(j_dict['globals'])
            slr_conf.pulse = PulseSpecs.from_dict(j_dict['pulse'])
        else:
            raise ValueError("Could not find JSON configuration file")
        return slr_conf

    @classmethod
    def from_cmd_line_args(cls, args: ArgumentParser.parse_args) -> dataclass:
        slr_config = cls(f_config=args.config, globals=args.globals, pulse=args.pulse)
        if args.config.configFile:
            slr_config = cls.load(args.config.configFile)
        # ToDo overwrite parameters explicitly given by cmd line
        return slr_config


def create_command_line_parser() -> (ArgumentParser, ArgumentParser.parse_args):
    parser = ArgumentParser(prog="slr_pulse_tool")
    parser.add_arguments(FileConfig, dest="config")
    parser.add_arguments(GlobalSpecs, dest="globals")
    parser.add_arguments(PulseSpecs, dest="pulse")

    args = parser.parse_args()
    return parser, args
