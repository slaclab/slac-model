import slac_db.device as device


_CAVITY_CS_PREFIXES = {"KLYS", "ACCL", "LCAV"}

_BEAM_PATH_BEAMCODE = {
    "CU_HXR": 1,
    "CU_SXR": 2,
}

_MAGNET_DES_ATTRS = {
    "BEND": ["BDES"],
    "BTRM": ["BDES"],
    "QUAD": ["BDES"],
    "SOLE": ["BDES"],
    "XCOR": ["BDES"],
    "YCOR": ["BDES"],
}


def _klys_pvs(mad_name, beamcode):
    """Return (amplitude_pv, phase_pv, beamcode_pv) for a cavity device.

    Looks up the beamcode-specific DES accessors on the device,
    falling back to the plain amplitude/phase_desired entries if the
    beamcode-specific keys are absent. The beamcode PV comes from the
    'beamcode{N}' accessor entry. Any return slot may be None.

    Args:
        mad_name (str): MAD device name.
        beamcode (int): Klystron beamcode (e.g. 1 for CU_HXR).

    Returns:
        tuple[str | None, str | None, str | None]:
            (amplitude_pv, phase_pv, beamcode_pv)
    """
    acc = device.get_all_accessors(mad_name) or {}
    ampl_pv = (
        acc.get(f"amplitude_des_beamcode{beamcode}")
        or acc.get("amplitude_des")
        or acc.get("amplitude")
    )
    phase_pv = (
        acc.get(f"phase_desired_beamcode{beamcode}")
        or acc.get("phase_desired")
    )
    beamcode_pv = acc.get(f"beamcode{beamcode}")
    return ampl_pv, phase_pv, beamcode_pv


def get_model_pvs(devices, beampath=None):
    """Return a flat list of DES PVs for a list of MAD devices. The list
    can then be used to instantiate a LIVE model.

    Cavity-shaped entries (cs_name prefix in KLYS/ACCL/LCAV) contribute
    their resolved amplitude, phase, and beamcode PVs when present.
    Magnet entries contribute cs_name:BDES per device_type.

    Args:
        devices (list[str]): MAD device names (e.g. the return of
            `get_devices(beampath=..., device_type=...)`).
        beampath (str) optional: Beam path name used to resolve the
            klystron beamcode (CU_HXR -> 1, CU_SXR -> 2). Required if
            any device in the list is cavity-shaped.

    Returns:
        list[str]: PV names in input order; cavity entries contribute
            up to three PVs (amplitude, phase, beamcode); magnet
            entries contribute one (BDES).

    Raises:
        ValueError: If a cavity-shaped device is encountered without
            a beampath beamcode mapping, or a device's `device_type`
            is neither cavity-shaped nor in `_MAGNET_DES_ATTRS`.
    """
    beamcode = (
        _BEAM_PATH_BEAMCODE.get(beampath.upper()) if beampath else None
    )
    pvs = []
    for mad_name in devices:
        cs_name = device.get_attribute(mad_name, "cs_name")
        device_type = device.get_attribute(mad_name, "device_type")
        cs_prefix = cs_name.split(":")[0] if cs_name else ""
        if cs_prefix in _CAVITY_CS_PREFIXES:
            if beamcode is None:
                raise ValueError(
                    f"beampath {beampath!r} is required for beamcode "
                    f"resolution on cavity-shaped device {mad_name!r}. "
                    f"Supported: {list(_BEAM_PATH_BEAMCODE)}."
                )
            for pv in _klys_pvs(mad_name, beamcode):
                if pv:
                    pvs.append(pv)
        elif device_type in _MAGNET_DES_ATTRS:
            for suffix in _MAGNET_DES_ATTRS[device_type]:
                pvs.append(f"{cs_name}:{suffix}")
        else:
            raise ValueError(
                f"Unsupported device type {device_type!r} for device "
                f"{mad_name!r} (cs_name={cs_name!r}). Supported: "
                f"{', '.join(sorted(_MAGNET_DES_ATTRS))}."
            )
    return pvs
