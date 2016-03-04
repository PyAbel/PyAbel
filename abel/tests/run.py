# -*- coding: utf-8 -*-
import sys
import os

_base_dir, _ = os.path.split(__file__)

def run(coverage=False):
    """
    This runs the complete set of PyAbel tests.
    """
    import nose
    argv=['', '-s', '--where={}'.format(_base_dir), '--verbosity=2']
    if coverage:
        argv += ['--with-coverage', '--cover-package=abel']
    result = nose.run(argv=argv)
    status = int(not result)
    return status

def run_cli(coverage=False):
    """
    This also runs the complete set of PyAbel tests.
    But uses sys.exit(status) instead of simply returning the status.
    """
    status = run(coverage=coverage)
    print('Exit status: {}'.format(status))
    sys.exit(status)



if __name__ == '__main__':
    run_cli()




