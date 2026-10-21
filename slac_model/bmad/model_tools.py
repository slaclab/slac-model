import re
from typing import Optional

import slac_db.device as dev

# Example: dev.get_attribute("QDMP2", "cs_name") returns the control system name for a MAD element

_MODELED_DEVICE_TYPES = ['BEND', 'BTRM', 'QUAD', 'XCOR', 'YCOR', 'LCAV']

# Lazily-built reverse map: cs_name -> device_name (MAD element)
_cs_to_mad = {}


_DES_ATTRS = {
    'BEND': ['BDES'],
    'BTRM': ['BDES'],
    'QUAD': ['BDES'],
    'SOLN': ['BDES'],
    'XCOR': ['BDES'],
    'YCOR': ['BDES'],
    'LCAV': ['ADES', 'PDES'],
}

# Device types that are intentionally excluded from PV generation.
_EXCLUDED_DEVICE_TYPES = {'TCAV'}

# Mapping from beam path name to klystron beamcode.
_BEAM_PATH_BEAMCODE = {
    'CU_HXR': 1,
    'CU_SXR': 2,
}


def _klys_pvs(mad_name, beamcode):
    """Return (amplitude_pv, phase_pv) for a KLYS/ACCL device.

    Looks up the beamcode-specific DES accessors on the device, falling
    back to the plain ``amplitude`` / ``phase_desired`` entries when the
    beamcode-specific keys are absent.  Either return slot may be
    ``None`` if the DB has no matching accessor.
    """
    acc = dev.get_all_accessors(mad_name) or {}
    ampl_pv = (acc.get(f'amplitude_des_beamcode{beamcode}')
               or acc.get('amplitude'))
    phase_pv = (acc.get(f'phase_desired_beamcode{beamcode}')
                or acc.get('phase_desired'))
    return ampl_pv, phase_pv


def get_modeled_devices(device_types=None, beampath=None, regions=None,
                        first=None, last=None):
    """Return modeled MAD elements keyed by MAD name, with DB attributes.

    Queries the SLAC device database for MAD model elements matching
    ``device_types`` in the areas implied by ``regions`` or ``beampath``,
    sorts by longitudinal position (``sum_l_meters``), and returns a dict
    keyed by MAD element name.  Each value carries ``cs_name`` and
    ``device_type``; for klystron-shaped devices (``cs_name`` prefix
    ``KLYS`` or ``ACCL``) it also carries ``amplitude_pv`` and
    ``phase_pv`` resolved against the beamcode implied by ``beampath``.
    Elements whose ``cs_name`` prefix is in ``_EXCLUDED_DEVICE_TYPES``
    (e.g. TCAV) are omitted.

    Parameters
    ----------
    device_types : list of str, optional
        Device type codes to include. Defaults to ``_MODELED_DEVICE_TYPES``.
    beampath : str, optional
        Beam path name (e.g. ``'CU_HXR'``) used both to look up areas
        when ``regions`` is not supplied *and* to resolve the klystron
        beamcode (``CU_HXR`` → 1, ``CU_SXR`` → 2).  Required if any
        klystron-shaped device is in the selection.
    regions : list of str, optional
        Area names to search; overrides ``beampath`` for area lookup.
    first, last : str, optional
        MAD element names bounding the returned slice (inclusive).  If
        ``first`` or ``last`` is not found in the sorted list, or if
        ``first`` appears after ``last``, a ``ValueError`` is raised.

    Returns
    -------
    dict[str, dict]
        Mapping ``mad_name -> {'cs_name', 'device_type',
        'amplitude_pv', 'phase_pv'}``, insertion-ordered by
        ``sum_l_meters``.  ``amplitude_pv`` and ``phase_pv`` are
        ``None`` for non-klystron devices.
    """
    if device_types is None:
        device_types = _MODELED_DEVICE_TYPES
    beamcode = (
        _BEAM_PATH_BEAMCODE.get(beampath.upper()) if beampath else None
    )
    areas = regions if regions else dev.get_areas(beampath=beampath)
    mad_elements = []
    for area in areas:
        for dtype in device_types:
            result = dev.get_devices(area=area, device_type=dtype)
            if result:
                mad_elements.extend(result)
    sorted_elements = sorted(
        mad_elements,
        key=lambda e: dev.get_all_meta(e).get('sum_l_meters', 0),
    )

    if first is not None:
        try:
            first_idx = sorted_elements.index(first)
        except ValueError:
            raise ValueError(
                f"first={first!r} not found in modeled elements for "
                f"beampath={beampath!r}"
            )
    else:
        first_idx = 0

    if last is not None:
        try:
            last_idx = sorted_elements.index(last)
        except ValueError:
            raise ValueError(
                f"last={last!r} not found in modeled elements for "
                f"beampath={beampath!r}"
            )
    else:
        last_idx = len(sorted_elements) - 1

    if first_idx > last_idx:
        raise ValueError(
            f"first={first!r} appears after last={last!r} in modeled elements "
            f"for beampath={beampath!r}"
        )

    sliced = sorted_elements[first_idx : last_idx + 1]

    result = {}
    for e in sliced:
        cs_name = dev.get_attribute(e, 'cs_name')
        cs_prefix = cs_name.split(':')[0] if cs_name else ''
        if cs_prefix in _EXCLUDED_DEVICE_TYPES:
            continue
        entry = {
            'cs_name': cs_name,
            'device_type': dev.get_attribute(e, 'device_type'),
            'amplitude_pv': None,
            'phase_pv': None,
        }
        if cs_prefix in ('KLYS', 'ACCL'):
            if beamcode is None:
                raise ValueError(
                    f"beampath {beampath!r} is required / not supported for "
                    f"beamcode resolution on klystron-shaped device {e!r}. "
                    f"Supported: {list(_BEAM_PATH_BEAMCODE)}."
                )
            entry['amplitude_pv'], entry['phase_pv'] = _klys_pvs(e, beamcode)
        result[e] = entry
    return result


def get_model_pvs(devices):
    """Return DES PV names for the modeled devices in *devices*.

    *devices* is the dict returned by :func:`get_modeled_devices`, mapping
    MAD element name → ``{'cs_name', 'device_type', 'amplitude_pv',
    'phase_pv'}``.  For klystron-shaped devices the beamcode-resolved
    ``amplitude_pv`` / ``phase_pv`` entries are used directly; for other
    device types the per-type DES attribute suffixes in ``_DES_ATTRS``
    are appended to ``cs_name`` (LCAV contributes two entries).

    Devices whose ``cs_name`` prefix is in ``_EXCLUDED_DEVICE_TYPES``
    (e.g. TCAV) are silently skipped — defensive guard; the discovery
    function already filters them out.

    Parameters
    ----------
    devices : dict[str, dict]
        MAD-name-keyed dict as produced by :func:`get_modeled_devices`.

    Returns
    -------
    list of str
        Full PV names with DES attribute appended.

    Raises
    ------
    ValueError
        If a non-klystron device's ``device_type`` is not in
        ``_DES_ATTRS`` and not in ``_EXCLUDED_DEVICE_TYPES``.
    """
    pvs = []
    for mad_name, attrs in devices.items():
        cs_name = attrs['cs_name']
        device_type = attrs['device_type']
        cs_prefix = cs_name.split(':')[0] if cs_name else ''
        if cs_prefix in _EXCLUDED_DEVICE_TYPES:
            continue
        if cs_prefix in ('KLYS', 'ACCL'):
            if attrs['amplitude_pv']:
                pvs.append(attrs['amplitude_pv'])
            if attrs['phase_pv']:
                pvs.append(attrs['phase_pv'])
        elif device_type in _DES_ATTRS:
            for attr in _DES_ATTRS[device_type]:
                pvs.append(f"{cs_name}:{attr}")
        else:
            raise ValueError(
                f"Unsupported device type {device_type!r} for device {mad_name!r} "
                f"(cs_name={cs_name!r}). "
                f"Supported types: {', '.join(sorted(_DES_ATTRS))}."
            )
    return pvs


def _build_cs_to_mad():
    """Populate the module-level _cs_to_mad reverse-lookup cache.

    Issues a single bulk SELECT ordered by device_name instead of calling
    get_attribute() once per device, reducing build time from ~11 s to
    ~0.03 s.  Ordering by device_name replicates the alphabetical iteration
    of get_devices() so that the last alphabetical device_name wins when
    multiple devices share the same cs_name.
    """
    import sqlalchemy
    global _cs_to_mad
    if not dev._meta:
        dev.get_devices()
    with dev._meta.session() as s:
        rows = s.execute(
            sqlalchemy.select(dev._meta.t.devices).order_by(
                dev._meta.t.devices.c.device_name
            )
        ).fetchall()
    for row in rows:
        cs = row._mapping['cs_name']
        dn = row._mapping['device_name']
        if cs:
            _cs_to_mad[cs] = dn  # last alphabetical device_name wins on duplicates


def model_name_convert(names, type='control'):
    """Convert between MAD element names and EPICS control-system PV names.

    By default (``type='control'``) the function accepts Bmad/MAD model
    element names and returns the corresponding EPICS control-system device
    names (``cs_name``).  Pass ``type='bmad'`` to go the other direction:
    supply EPICS ``cs_name`` strings and receive MAD element names back.

    Bmad split-element suffixes (``#N``, e.g. ``QE01#1``) are stripped
    before lookup so that split slices resolve to the same DB entry as the
    unsplit element name.

    If a name cannot be found in the device database the original supplied
    string is returned unchanged.

    The reverse-lookup map (``cs_name`` → ``device_name``) is built once on
    the first call with ``type='bmad'`` and cached for the lifetime of the
    process.

    Parameters
    ----------
    names : list of str
        Names to convert.  For ``type='control'``: MAD element names
        (e.g. ``['CQ01B', 'QE01#1']``).  For ``type='bmad'``: EPICS
        control-system names (e.g. ``['QUAD:GUNB:212:1']``).
    type : {'control', 'bmad'}, optional
        Conversion direction.  ``'control'`` (default) converts MAD names to
        EPICS PVs; ``'bmad'`` converts EPICS PVs to MAD element names.

    Returns
    -------
    list of str
        Converted names in the same order as the input.  Any name not found
        in the database is returned as-is.
    """
    _SPLIT_RE = re.compile(r'#\d+$')

    if type == 'control':
        result = []
        for name in names:
            bare = _SPLIT_RE.sub('', name)
            try:
                cs = dev.get_attribute(bare, 'cs_name')
                result.append(cs if cs is not None else name)
            except Exception:
                result.append(name)
        return result

    elif type == 'bmad':
        if not _cs_to_mad:
            _build_cs_to_mad()
        return [_cs_to_mad.get(name, name) for name in names]

    else:
        raise ValueError(f"type must be 'control' or 'bmad', got {type!r}")
