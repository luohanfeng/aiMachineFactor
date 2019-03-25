""""log日志

"""
from logbook import Logger, StderrHandler


# 系统运行的日志
system_log = Logger('system_log')


# 数据库读取操作的日志
dbreader_log = Logger('dbreader_log')

# 机器学习日志
machine_log = Logger('machine_log')


def init_logger():
    StderrHandler(bubble=True).push_application()


init_logger()

