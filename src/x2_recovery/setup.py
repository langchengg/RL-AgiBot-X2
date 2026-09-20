from setuptools import find_packages, setup

setup(
    name="x2_recovery",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/x2_recovery"]),
        ("share/x2_recovery", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    maintainer="Lang Cheng",
    maintainer_email="96649762+langchengg@users.noreply.github.com",
    description="Runtime diagnostics for the AgiBot X2 recovery project.",
    license="UNLICENSED",
    entry_points={"console_scripts": ["runtime_check = x2_recovery.diagnostics:main"]},
)
