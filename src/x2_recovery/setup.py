from setuptools import find_packages, setup

setup(
    name="x2_recovery",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/x2_recovery"]),
        ("share/x2_recovery", ["package.xml"]),
        ("share/x2_recovery/launch", ["launch/recovery.launch.py"]),
    ],
    install_requires=["setuptools"],
    maintainer="Lang Cheng",
    maintainer_email="96649762+langchengg@users.noreply.github.com",
    description="Native MuJoCo recovery environment and diagnostics for AgiBot X2.",
    license="UNLICENSED",
    entry_points={"console_scripts": [
        "runtime_check = x2_recovery.diagnostics:main",
        "recovery_node = x2_recovery.recovery_node:main",
        "telemetry_node = x2_recovery.telemetry_node:main",
    ]},
)
