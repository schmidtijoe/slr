from slr import core, options
import logging


def slr_algorithm(slr_config: options.SlrConfiguration = options.SlrConfiguration()):
    # create slr obtject
    slr_pulse = core.SLR(slr_config=slr_config)
    # build
    slr_pulse.build()
    # save
    slr_pulse.save_rf()


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
