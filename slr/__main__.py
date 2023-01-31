from slr import slr, options
import logging


def main():
    # set up logging
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)

    parser, args = options.create_command_line_parser()
    slr_config = options.SlrConfiguration.from_cmd_line_args(args)
    try:
        _ = slr.build(slr_config)
    except Exception as e:
        logging.error(e)
        parser.print_usage()


if __name__ == '__main__':
    main()
