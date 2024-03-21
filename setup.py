from setuptools import setup
import versioneer

setup(
    packages=["cassetta"],
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
