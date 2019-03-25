"""

"""
import inspect
import pickle
import os

from aiMachineFactor.utils.logger import system_log


def cache2pickle(path, *args, **kwargs):
    """将args中的值，存储为path下同名的pickle

    :param path:
    :param args:
    :param kwargs:
    :return:
    """

    tobe_write = {}
    for item in args:
        for var_name, var_value in inspect.currentframe().f_back.f_locals.items():
            try:
                if var_value is item and not (var_name is item):
                    tobe_write[var_name] = var_value
            except TypeError:
                continue
    for k, v in tobe_write.items():
        try:
            v.to_pickle(path+k)
        except AttributeError:
            try:
                with open(path+k, 'w') as f:
                    pickle.dump(v, f)
            except Exception as e:
                system_log.warn('pickle失败{}{}'.format(k, str(e)))
                os.remove(path+k)



