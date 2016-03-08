from logf.printf import printf


def cur_mem_usage():
    try:
        import psutil
        import os
        p = psutil.Process(os.getpid())
        printf('current memory usage: {}', p.memory_info().rss, type='WARN')
    except Exception:
        printf('failed to get memory stat', type='WARN')
