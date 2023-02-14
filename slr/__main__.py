from slr import core, options
import logging
import pathlib as plb


def slr_algorithm(slr_config: options.SlrConfiguration = options.SlrConfiguration()):
    # create slr obtject
    slr_pulse = core.SLR(slr_config)
    # build
    slr_pulse.build()
    # save
    slr_pulse.save_rf()
    plot_name = plb.Path(slr_config.f_config.outputPulseFile).absolute().with_name(
        f"{plb.Path(slr_config.f_config.outputPulseFile).stem}_plot").with_suffix(".png")
    slr_pulse.plot(plot_save=plot_name.__str__())


def main():
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s -- %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)

    parser, args = options.create_command_line_parser()
    slr_config = options.SlrConfiguration.from_cmd_line_args(args)
    try:
        slr_algorithm(slr_config)
    except Exception as e:
        logging.error(e)
        parser.print_usage()


if __name__ == '__main__':
    main()
