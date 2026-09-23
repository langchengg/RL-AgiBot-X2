# Reference provenance

The X2 reference in the selected `resolved_config.json` was produced through
retargeting, X2-specific dynamic search and subsequent stance/ankle refinement.
The discovery input was a HumanUP motion linked from the official
[simulation README](https://github.com/RunpeiDong/HumanUP/blob/7516e0f27e6f4d1e7365cf64ea577a78247bd8cb/simulation/README.md).
The inspected repository revision is `7516e0f27e6f4d1e7365cf64ea577a78247bd8cb`.

HumanUP's repository license is Apache-2.0, copyright 2025 Xialin He, Runpei Dong,
Zixuan Chen and Saurabh Gupta. Its original license text is retained in
[HumanUP-LICENSE.txt](HumanUP-LICENSE.txt) for attribution. No HumanUP implementation,
pretrained policy, framework or original motion file is included here. A separate
license for the externally hosted raw motion was not established; the repository
license is not represented as a separate grant for that asset. The raw asset remains
a local research input. The included numerical X2 reference, generated simulator
measurements and trained residual checkpoint preserve the actual selected controller.

The AgiBot X2 model is obtained separately from the official repository at revision
`60c5de582c523cd188f563819e62d34cfdc3d2d0`, under MulanPSL-2.0, as documented in the
project README. Model assets and their license are not replaced or vendored by this
publication. Other research references and the inspected versions are recorded in
the successful evaluation's `analysis/research.json`.
