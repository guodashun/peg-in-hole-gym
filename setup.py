import os
import setuptools
from pathlib import Path

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'peg_in_hole', 'envs', 'assets')
data_files = []
for root, dirs, files in os.walk(directory):
    for file in files:
        data_files.append(os.path.join(root, file))

setuptools.setup(
    name='peg-in-hole-gym',
    version='0.0.1',
    description='An gym env for simulating flexible tube grasp.',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(include='envs'),
    packages_data={'model_files': data_files},
    install_requires=['gym', 'pybullet', 'numpy'],
    url='https://github.com/guodashun/peg-in-hole-gym',
    author='luckky',
    author_email='luckky@gmail.com',
    license='MIT',
)