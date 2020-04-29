import numpy as np
import abel

import os
import sys
import platform
import cpuinfo


def get_system():
    cpu = cpuinfo.get_cpu_info().get('brand').split()[2]
    plat = platform.system() + '_' + platform.release()
    return cpu + '_' + plat


def bench(method, repeats, min_time, max_time, benchmark_dir, append=True):
    """run abel.benchmark and save timings to '{benchmark_dir}/{method}.dat'.

    """
    # check whether PyAbel version has '{method}' and basis calculation
    try:
        res = abel.benchmark.AbelTiming(n=21, select=method, verbose=False)
        try:
            res.bs[method]
            has_basis = True
        except KeyError:
            has_basis = False
    except ValueError:
        # no such PyAbel method
        return

    # benchmark timing file
    fn = os.path.join(benchmark_dir, f'{method}.dat')

    try:
        with open(fn, 'r') as fp:
            previous = fp.readlines()
        lastn = int(previous[-1].split()[0])
    except IOError:
        lastn = 0
        append = False

    ns = list(sizes[sizes > lastn])
    if len(ns) == 0:
        print(f'(Nothing to do for {method})\n')
        return

    if lastn > 0:
        print('Previous results:')
        for line in previous:
            print(line.strip('\n'))

    res = abel.benchmark.AbelTiming(n=ns, select=method, repeat=repeats,
                                    t_min=min_time, t_max=max_time,
                                    verbose=True)

    # do not write anything if there are no useful results
    if np.isnan(res.iabel[method][0]):
        print('No new results\n')
        return

    print('New results:')
    header = f'# {method:15s} {"iabel(ms)":20s}'
    if has_basis:
        header += 'basis(ms)'
    if method[-5:] == 'basex':  # special case
        if method[0] == 'b':
            varnone = 'var'
            varlab = 'var.reg.'
        else:
            varnone = 'None'
            varlab = '(None)'
        methodvar = f'{method:s}({varnone:s})'
        fnvar = os.path.join(benchmark_dir, f'{methodvar}.dat')
        fpvar = open(fnvar, 'a' if append else 'w')
        headervar = f'# {methodvar:15s} {"iabel(ms)":20s}'
        if not append:
            fpvar.write(headervar+'\n')
        print(f'# {method:15s} {"iabel(ms)":20s}{varlab+"(ms)":20s}basis(ms)')
    else:
        print(header)

    with open(fn, 'a' if append else 'w') as fp:
        if not append:
            fp.write(header+'\n')

        # write timings to file: n  iabel(ms) [basis(ms) if exists]
        for i, n in enumerate(ns):
            if np.isnan(res.iabel[method][i]):
                break

            print(f'{n:5d}', end='')
            fp.write(f'{n:5d}')

            t_iabel = f'{res.iabel[method][i]:20.5f}'
            print(t_iabel, end='')
            fp.write(t_iabel)

            if method[-5:] == 'basex':  # special case
                if method[0] == 'b':
                    varlab = method+'(var.reg.)'
                else:
                    varlab = method+'(None)'
                t_var = f'{res.iabel[varlab][i]:20.5f}'
                print(t_var, end='')
                fpvar.write(f'{n:5d}')
                fpvar.write(t_var+'\n')

            if method in res.bs:
                t_bs = f'{res.bs[method][i]:20.5f}'
            else:
                t_bs = ''
            print(t_bs)
            fp.write(t_bs+'\n')

    if method[-5:] == 'basex':
        fpvar.close()

    # spacing between methods
    print(flush=True)


# main ----------------

# The following should be used for Intel CPUs, since they work very slowly
# with subnormal floating-point numbers (see PyAbel issue #246):
try:
    import daz
    daz.set_ftz()
    daz.set_daz()
except ModuleNotFoundError:
    print('\nWarning! No daz module. Intel CPUs can show poor performance.\n')

system = get_system()
benchmark_dir = 'benchmarks_' + system
print('PyAbel method timings in the directory:\n\n'
      f'    {benchmark_dir}/method.dat\n')

append = True
try:
    os.mkdir(benchmark_dir)
except OSError:
    print('    directory exists, new timings will be appended to existing'
          ' method files', end='\n\n')

# subsequent runs will append to the benchmark files
nmax = input('nmax to set maximum image size = 2**nmax + 1 [15]: ')
if len(nmax) > 0:
    nmax = int(nmax)
else:
    nmax = 15
sizes = 2**np.arange(2, nmax+1) + 1
print('\n    image sizes:\n   ', sizes, '\n')

repeats = input('min. iterations [3]: ')
if len(repeats) > 0:
    repeats = int(repeats)
else:
    repeats = 3
print(f'\n    minimum {repeats} iterations\n')

min_time = input('min. execution time per test (seconds) [10]: ')
if len(min_time) > 0:
    min_time = float(min_time)
else:
    min_time = 10  # seconds
print(f'\n    minimum {min_time} seconds per test\n')

max_time = input('max. execution time per method (minutes) [120]: ')
if len(max_time) > 0:
    max_time = int(max_time)
else:
    max_time = 120  # minutes
print(f'\n    maximum {max_time} minutes per method\n')
max_time *= 60  # seconds


methods = [
  'two_point',
  'three_point',
  'onion_peeling',
  'hansenlaw',
  'basex',
  'rbasex',
  'direct_C',
  'onion_bordas',
  'linbasex',
  'direct_Python'
]

method = input('benchmark single method (name) [blank = all]: ')
if len(method) > 0:
    methods = [method]
print('\n    methods:\n   ', methods, '\n')


for method in methods:
    bench(method, repeats, min_time, max_time, benchmark_dir, append)
