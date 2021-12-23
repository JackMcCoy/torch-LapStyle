from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
	name='PyTorch EMD',
	version='0.0',
	author='Marc Eder',
	author_email='meder@cs.unc.edu',
	description='A PyTorch module for the earth mover\'s distance loss',
	ext_package='_emd_ext',
	ext_modules=[
		CUDAExtension(
			name='_emd',
			sources=[
				'src/emd.cpp',
				'src/cuda/emd.cu',
			],
			include_dirs=['include'],
		),
	],
	packages=[
		'emd',
	],
	package_dir={
		'emd' : 'emd_layer'
	},
	cmdclass={'build_ext': BuildExtension},
)