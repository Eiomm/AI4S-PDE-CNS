from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass
from typing import Callable


LATEST_PHYSICSNEMO_MIN_PYTHON = (3, 11)
IMPORT_NAME = "physicsnemo"
PACKAGE_NAME = "nvidia-physicsnemo"


@dataclass(frozen=True)
class PhysicsNeMoStatus:
    installed: bool
    usable: bool
    import_name: str
    package_name: str
    current_python: str
    latest_package_python_supported: bool
    version: str | None
    reason: str
    recommendation: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "installed": self.installed,
            "usable": self.usable,
            "import_name": self.import_name,
            "package_name": self.package_name,
            "current_python": self.current_python,
            "latest_package_python_supported": self.latest_package_python_supported,
            "version": self.version,
            "reason": self.reason,
            "recommendation": self.recommendation,
        }


def _format_python_version(version: tuple[int, int, int] | tuple[int, int]) -> str:
    return ".".join(str(part) for part in version)


def probe_physicsnemo(
    *,
    python_version: tuple[int, int, int] | tuple[int, int] | None = None,
    find_spec: Callable[[str], object | None] = importlib.util.find_spec,
    version_lookup: Callable[[str], str | None] | None = None,
) -> PhysicsNeMoStatus:
    current_version = python_version or sys.version_info[:3]
    version_text = _format_python_version(current_version)
    latest_supported = tuple(current_version[:2]) >= LATEST_PHYSICSNEMO_MIN_PYTHON
    spec = find_spec(IMPORT_NAME)

    if version_lookup is None:
        version_lookup = _installed_version

    installed_version = version_lookup(PACKAGE_NAME) if spec is not None else None
    if spec is None:
        if latest_supported:
            reason = f"{IMPORT_NAME} is not installed."
            recommendation = f"Install {PACKAGE_NAME} in an isolated environment before enabling PhysicsNeMo baselines."
        else:
            reason = (
                f"{IMPORT_NAME} is not installed; latest {PACKAGE_NAME} requires Python >=3.11, "
                f"but the active environment is Python {version_text}."
            )
            recommendation = (
                "Keep Hwpytorch stable for the current baseline. Use an isolated Python 3.11 environment "
                f"for latest {PACKAGE_NAME}, or pin nvidia-physicsnemo==1.3.* if testing inside Hwpytorch."
            )
        return PhysicsNeMoStatus(
            installed=False,
            usable=False,
            import_name=IMPORT_NAME,
            package_name=PACKAGE_NAME,
            current_python=version_text,
            latest_package_python_supported=latest_supported,
            version=None,
            reason=reason,
            recommendation=recommendation,
        )

    reason = f"{IMPORT_NAME} is importable."
    if not latest_supported:
        reason += f" Active Python {version_text} cannot use latest {PACKAGE_NAME}, so this is expected to be an older compatible install."
    return PhysicsNeMoStatus(
        installed=True,
        usable=True,
        import_name=IMPORT_NAME,
        package_name=PACKAGE_NAME,
        current_python=version_text,
        latest_package_python_supported=latest_supported,
        version=installed_version,
        reason=reason,
        recommendation="Enable PhysicsNeMo candidates only through optional Baseline Zoo experiments and keep existing FNO fallback active.",
    )


def physicsnemo_status() -> PhysicsNeMoStatus:
    return probe_physicsnemo()


def _installed_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
