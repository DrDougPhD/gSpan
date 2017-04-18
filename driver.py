#!/usr/bin/env python
# -*- coding: utf-8 -*-
import getopt
from datetime import datetime
import sys
import os
import logging

from gspan import gSpan

logger = logging.getLogger(__name__)


default_args = {
    '-n':float('inf'),
    '-s':5000,
    '-l':2,
    '-u':float('inf'),
    '-d':0,
    '-v':0,
    'database_file_name':'graphdata/graph.data'
}


def main(args):
    gs = gSpan(database_file_name=args['database_file_name'],
               min_support=args['-s'],
               min_num_vertices=args['-l'],
               max_num_vertices=args['-u'],
               max_ngraphs = args['-n'],
               is_undirected=(args['-d'] == 0),
               verbose=(args['-v'] == 1),
               visualize=(args['-p'] == 1),
               where=(args['-w'] == 1)
    )

    gs.run()
    gs.time_stats()


def setup_logger(debugging_mode):
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    # todo: place them in a log directory, or add the time to the log's
    # filename, or append to pre-existing log
    log_file = os.path.join('/tmp', 'gspan.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()

    if debugging_mode:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    line_numbers_and_function_name = logging.Formatter(
        "%(levelname)s [%(filename)s:%(lineno)s - %(funcName)20s() ]"
        "%(message)s")
    fh.setFormatter(line_numbers_and_function_name)
    ch.setFormatter(line_numbers_and_function_name)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def usage():
    print('\nUsage: python main.py [-s min_support] [-n num_graph]'
          ' [-l min_num_vertices] [-u max_num_vertices] [-d] [-v] [-h] [-p]'
          ' [-w] database_file_name')
    print('''\nOptions:\n
    -s, min support, default 5000\n
    -n, only read the first n graphs in the given database, default inf, i.e. all graphs\n
    -l, lower bound of number of vertices of output subgraph, default 2\n
    -u, upper bound of number of vertices of output subgraph, default inf\n
    -d, run for directed graphs, default off, i.e. for undirected graphs\n
    -v, verbose output, default off\n
    -p, plot frequent subgraph, default off\n
    -w, output where one frequent subgraph appears in database, default off\n
    -h, help\n
    ''')


def parse_args(args, default_args):
    optlist, args = getopt.getopt(args, 'n:s:l:u:dvpw')
    opt_dict = {k:v for k, v in optlist}
    for k in ['-d', '-v', '-p', '-w']:
        opt_dict[k] = 0 if k not in opt_dict else 1
    try:
        for k in default_args:
            opt_dict[k] = default_args[k]\
                if k not in opt_dict else int(opt_dict[k])

        opt_dict['database_file_name'] = default_args['database_file_name']\
            if len(args) == 0 else args[0]

        return opt_dict

    except Exception:
        usage()
        exit()

if __name__ == '__main__':
    try:
        start_time = datetime.now()

        if len(sys.argv) == 1 or '-h' in sys.argv:
            usage()
            exit()

        if '-t' in sys.argv:
            opt_dict = parse_args('-n 2 -s 2 -l 2 -u 3 -v'.split(),
                                  default_args)
        else:
            opt_dict = parse_args(sys.argv[1:], default_args)

        if not os.path.exists(opt_dict['database_file_name']):
            print('{} does not exist.'.format(opt_dict['database_file_name']))
            exit()

        setup_logger(debugging_mode=(opt_dict['-v'] == 1))
        logger.debug('Command-line arguments:')
        for arg, value in opt_dict.items():
            logger.debug('\t{argument_key}:\t{value}'.format(argument_key=arg,
                                                             value=value))

        logger.debug(start_time)

        main(opt_dict)

        finish_time = datetime.now()
        logger.debug(finish_time)
        logger.debug('Execution time: {time}'.format(
            time=(finish_time - start_time)
        ))
        logger.debug("#" * 20 + " END EXECUTION " + "#" * 20)

        sys.exit(0)

    except KeyboardInterrupt as e:  # Ctrl-C
        raise e

    except SystemExit as e:  # sys.exit()
        raise e

    except Exception as e:
        logger.exception("Something happened and I don't know what to do D:")
        sys.exit(1)
