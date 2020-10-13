import setuptools
from pathlib import Path

setuptools.setup(
    name='peg_in_hole',
    version='0.0.1',
    description='An gym env for simulating flexible tube grasp.',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(include=''),
    install_requires=['gym', 'pybullet'],
    url='https://github.com/guodashun/peg_in_hole_rl',
    author='luckky',
    author_email='luckky@gmail.com',
    license='MIT',
)