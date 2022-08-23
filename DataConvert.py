import h5py
import numpy
import scipy.io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", help="h5 file or mat file")
args = parser.parse_args()

data = {}
if args.file.endswith('.h5'):
    with h5py.File(args.file) as fd:
        for i in fd.keys():
            data[i] = fd[i][...]
    scipy.io.savemat(args.file[:-3] + '.mat', data)
elif args.file.endswith('.mat'):
    data = scipy.io.loadmat(args.file)
    with h5py.File(args.file[:-4] + '.h5', 'w') as fd:
        for i in data.keys():
            if i not in ['__globals__',  '__header__', '__version__']:
                fd[i] = numpy.squeeze(data[i])
else:
    raise ValueError('filename must ends with .h5 or .mat')