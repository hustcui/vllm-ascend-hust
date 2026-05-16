# vLLM Hust Ascend Release and Install Guide

This page is the fork-specific source of truth for how `vllm-ascend-hust`
should be named, versioned, installed, and mapped to upstream components.

## Artifact Names

Different distribution channels intentionally use different names:

| Artifact type | Name | Notes |
|---|---|---|
| Git repository | `intellistream/vllm-ascend-hust` | Fork source repository |
| PyPI distribution | `vllm-ascend-hust` | Package name used by `pip install` |
| Python module namespace | `vllm_ascend` | Kept for compatibility with the upstream hardware plugin interface |
| vLLM platform plugin entry point | `ascend` | Registered under `vllm.platform_plugins` |
| Official Ascend container image | `quay.io/ascend/vllm-ascend:<tag>` | Upstream official container image name remains unchanged |

## Current Recommended Matrix

The current documentation set is aligned to the following recommended versions:

- `vllm` PyPI version: `|pip_vllm_version|`
- `vllm-ascend-hust` PyPI version: `|pip_vllm_ascend_version|`
- `vllm` source branch/tag: `|vllm_version|`
- `vllm-ascend-hust` source branch/tag: `|vllm_ascend_version|`
- CANN image tag: `|cann_image_tag|`

For historical compatibility details, see
[Versioning Policy](community/versioning_policy.md).

## Which Install Path To Use

Use the install path that matches how you work:

| Scenario | Recommended path |
|---|---|
| Local multi-repo development workspace | Run quickstart from `vllm-hust-dev-hub`, or install the sibling checkout with `bash scripts/install_local_ascend_plugin.sh` |
| Clean Python environment without local source checkout | Install the published fork package from PyPI |
| Source debugging or release validation | Clone `intellistream/vllm-ascend-hust` and install it from source |
| Container-based validation on Ascend hosts | Use the official `quay.io/ascend/vllm-ascend:<tag>` image |

## PyPI Installation

When you want the published fork package rather than a local editable checkout:

```{code-block} bash
   :substitutions:

pip install vllm==|pip_vllm_version|
pip install vllm-ascend-hust==|pip_vllm_ascend_version|
```

The installed Python module is still imported as `vllm_ascend`.

## Source Installation

When you need to build or debug from source:

```{code-block} bash
   :substitutions:

git clone --depth 1 --branch |vllm_ascend_version| https://github.com/intellistream/vllm-ascend-hust.git
cd vllm-ascend-hust
git submodule update --init --recursive
pip install -v -e .
```

## Local Workspace Development

Inside the `vllm-hust` multi-root workspace, the preferred path is the local
editable checkout:

```bash
cd /path/to/vllm-ascend-hust
bash scripts/install_local_ascend_plugin.sh
```

If `vllm-hust-dev-hub/scripts/quickstart.sh` does not find a sibling
`vllm-ascend-hust` checkout, it falls back to:

```bash
hust-ascend-manager runtime repair --install-plugin --plugin-package vllm-ascend-hust
```

That fallback installs the fork PyPI distribution into the selected conda
environment and verifies the `ascend` entry point.

## Release Expectations

When a new fork release is prepared, keep these channels aligned:

1. Update the source branch or tag used by the docs variables when the fork moves to a new compatible upstream baseline.
2. Publish the corresponding `vllm-ascend-hust` version to PyPI.
3. Keep installation examples using `vllm-ascend-hust` for package installs and `vllm_ascend` only for Python imports.
4. Keep references to `quay.io/ascend/vllm-ascend:<tag>` only where the upstream official image name is the actual artifact users must pull.

## Sanity Check

After installation, verify both the package and the plugin entry point:

```bash
python - <<'PY'
from importlib.metadata import entry_points, version

print(version('vllm-ascend-hust'))
print(any(ep.name == 'ascend' for ep in entry_points(group='vllm.platform_plugins')))
PY
```
