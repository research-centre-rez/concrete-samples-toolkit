import logging

"""
This file contains various pretty print methods for different data types
"""
logger = logging.getLogger(__name__)


def log_argparse(args) -> None:
    '''
    Logger will pretty print what arguments were passed into a main function. 
    Args: 
        args: args that argparser parsed
    Returns:
        None
    '''
    logger.info('Running with the following parameters:')
    for arg in vars(args):
        logger.info('  %s: %s', arg, getattr(args, arg))


def pprint_dict(dict_in:dict, desc:str) -> None:
    '''
    Pretty prints a dictionary via logger.info(). If the dictionary contains a dictionary, the function is recursively called. Main use case is for printing out config dictionaries that are used throughout the project.

    Args:
        dict_in (dict): Dictionary to be printed
        desc (str): Description of the dictionary, if you don't want any description then pass in an empty string.
    Returns:
        None
    '''
    if desc:
        logger.info(desc)

    for key, val in dict_in.items():
        if isinstance(val, dict):
            pprint_dict(val, key.upper())
        else:
            logger.info(' %s: %s', key, val)
