import re
from typing import Optional

import slac_db.device as dev

# Example: dev.get_attribute("QDMP2", "cs_name") returns the control system name for a MAD element

_MODELED_DEVICE_TYPES = ['BEND', 'BTRM', 'QUAD', 'XCOR', 'YCOR', 'LCAV']

# Lazily-built reverse map: cs_name -> device_name (MAD element)
_cs_to_mad = {}

def get_modeled_devices(device_types=_MODELED_DEVICE_TYPES, regions=None, beam_path=None):
    """Get control system (cs_name) device names for modeled elements.

    Queries the SLAC device database for MAD model elements matching the
    given device types, optionally filtered by region or beam path. Elements
    are sorted by longitudinal position (sum_l_meters) and returned as their
    control system names (e.g. 'QUAD:IN20:361').

    Parameters
    ----------
    device_types : list of str
        Device type codes to include. Defaults to _MODELED_DEVICE_TYPES
        ('BEND', 'BTRM', 'QUAD', 'XCOR', 'YCOR').
    regions : list of str, optional
        Area names to search. If None, all areas for beam_path are used.
    beam_path : str, optional
        Beam path name (e.g. 'sc_sxr', 'cu_hxr') used to look up areas
        when regions is not provided.

    Returns
    -------
    list of str
        Control system device names sorted by longitudinal position.
    """
    areas = regions if regions else dev.get_areas(beampath=beam_path)
    mad_elements = []
    for area in areas:
        for dtype in device_types:
            result = dev.get_devices(area=area, device_type=dtype)
            if result:
                mad_elements.extend(result)
    mad_elements = sorted(mad_elements, key=lambda e: dev.get_all_meta(e).get('sum_l_meters', 0))
    return [dev.get_attribute(e, 'cs_name') for e in mad_elements]


def get_modeled_elements(device_types=None, beampath=None, first=None, last=None):
    """Get MAD element names for modeled devices in a beampath.

    Uses slac_db.device.get_beampath to retrieve devices, sorted by
    longitudinal position (sum_l_meters).

    Parameters
    ----------
    device_types : list of str, optional
        Device type codes to include. Defaults to ['LCAV', 'QUAD'].
    beampath : str, optional
        Beam path name (e.g. 'cu_hxr', 'sc_sxr').
    first : str, optional
        MAD element name marking the start of the returned slice
        (inclusive). If None, the slice starts at the first element along
        the beamline.
    last : str, optional
        MAD element name marking the end of the returned slice
        (inclusive). If None, the slice ends at the last element along
        the beamline.

    Returns
    -------
    list of str
        MAD element device_name strings sorted by longitudinal position,
        optionally sliced to the ``[first, last]`` range (inclusive).

    Raises
    ------
    ValueError
        If ``first`` or ``last`` is provided but not found in the sorted
        element list, or if ``first`` appears after ``last`` in that list.
    """
    if device_types is None:
        device_types = ['LCAV', 'QUAD']
    mad_elements = []
    for dtype in device_types:
        mad_elements.extend(dev.get_beampath(beampath=beampath, device_type=dtype))
    sorted_elements = sorted(mad_elements, key=lambda e: dev.get_all_meta(e).get('sum_l_meters', 0))

    if first is None and last is None:
        return sorted_elements

    if first is None:
        first_idx = 0
    else:
        try:
            first_idx = sorted_elements.index(first)
        except ValueError:
            raise ValueError(
                f"first={first!r} not found in modeled elements for beampath={beampath!r}"
            )

    if last is None:
        last_idx = len(sorted_elements) - 1
    else:
        try:
            last_idx = sorted_elements.index(last)
        except ValueError:
            raise ValueError(
                f"last={last!r} not found in modeled elements for beampath={beampath!r}"
            )

    if first_idx > last_idx:
        raise ValueError(
            f"first={first!r} appears after last={last!r} in modeled elements "
            f"for beampath={beampath!r}"
        )

    return sorted_elements[first_idx : last_idx + 1]


_DES_ATTRS = {
    'QUAD': ['BDES'],
    'LCAV': ['ADES', 'PDES'],
}

# Device types that are intentionally excluded from PV generation.
_EXCLUDED_DEVICE_TYPES = {'TCAV'}

# Mapping from beam path name to klystron beamcode.
_BEAM_PATH_BEAMCODE = {
    'CU_HXR': 1,
    'CU_SXR': 2,
}


def get_model_pvs(devices, beam_path=None):
    """Return DES PV names for a list of MAD element device names.

    For each MAD element in *devices*, looks up its ``device_type`` and
    ``cs_name`` from the SLAC device database, then appends the appropriate
    desired-value attribute suffix(es) based on the DB ``device_type``.
    Devices with two setpoints (LCAV) contribute two entries to the returned
    list.

    Devices whose ``device_type`` is in ``_EXCLUDED_DEVICE_TYPES`` (e.g.
    TCAV) are silently skipped.

    For devices whose ``cs_name`` starts with ``KLYS:`` or ``ACCL:``
    (klystrons), if *beam_path* is provided the beamcode-specific DES PV
    names are resolved via ``slac_db.device.get_all_accessors``:

    * amplitude: ``amplitude_des_beamcode{N}`` if present, else
      ``amplitude``
    * phase: ``phase_desired_beamcode{N}`` if present, else
      ``phase_desired``

    When *beam_path* is ``None`` all devices fall back to the fixed
    ``_DES_ATTRS`` suffix table.

    Parameters
    ----------
    devices : list of str
        MAD element device_name strings, e.g.
        ``['QE01', 'BUN1B', 'K21_5']``.
    beam_path : str, optional
        Beam path name used to resolve the klystron beamcode for KLYS and
        ACCL devices.  Supported values: ``'CU_HXR'`` (beamcode 1),
        ``'CU_SXR'`` (beamcode 2).  Case-insensitive.  If ``None``,
        klystron devices fall back to the ``_DES_ATTRS['LCAV']`` suffixes.

    Returns
    -------
    list of str
        Full PV names with DES attribute appended, e.g.
        ``['QUAD:IN20:361:BDES', 'ACCL:GUNB:455:ADES', 'ACCL:GUNB:455:PDES',
           'KLYS:LI21:51:ENLD', 'KLYS:LI21:51:PDES']``.

    Raises
    ------
    ValueError
        If a device's DB ``device_type`` is not in ``_DES_ATTRS`` and not in
        ``_EXCLUDED_DEVICE_TYPES``, or if *beam_path* is provided but not
        recognised for beamcode resolution.
    """
    pvs = []
    for device in devices:
        device_type = dev.get_attribute(device, 'device_type')
        cs_name = dev.get_attribute(device, 'cs_name')
        if cs_name.split(':')[0] in _EXCLUDED_DEVICE_TYPES:
            continue
        cs_prefix = cs_name.split(':')[0] if cs_name else ''
        if cs_prefix in ('KLYS', 'ACCL') and beam_path is not None:
            beamcode = _BEAM_PATH_BEAMCODE.get(beam_path.upper())
            if beamcode is None:
                raise ValueError(
                    f"beam_path {beam_path!r} not supported for beamcode resolution. "
                    f"Supported: {list(_BEAM_PATH_BEAMCODE)}."
                )
            acc = dev.get_all_accessors(device)
            ampl_pv = (acc.get(f'amplitude_des_beamcode{beamcode}')
                       or acc.get('amplitude'))
            phase_pv = (acc.get(f'phase_desired_beamcode{beamcode}')
                        or acc.get('phase_desired'))
            if ampl_pv:
                pvs.append(ampl_pv)
            if phase_pv:
                pvs.append(phase_pv)
        elif device_type in _DES_ATTRS:
            for attr in _DES_ATTRS[device_type]:
                pvs.append(f"{cs_name}:{attr}")
        else:
            raise ValueError(
                f"Unsupported device type {device_type!r} for device {device!r} "
                f"(cs_name={cs_name!r}). "
                f"Supported types: {', '.join(sorted(_DES_ATTRS))}."
            )
    return pvs


def get_pv_data(pvs, date_time=None, batch_n=0):
    """Fetch EPICS values for a list of PV names.

    By default reads live values from the control system.  If *date_time* is
    provided, archived values at that point in time are fetched from the LCLS
    Archiver Appliance via ``meme.archive.get_data_at_time`` instead.

    Parameters
    ----------
    pvs : list of str
        Full EPICS PV names, e.g. ``['QUAD:IN20:361:BDES', 'LCAV:IN20:2:ADES']``.
    date_time : str or datetime, optional
        If provided, fetch archived values at this point in time instead of
        reading live EPICS values.  Accepts the same formats as
        ``meme.archive.get_data_at_time``: a string like
        ``'2024-01-15T10:30:00'`` or ``'1 hour ago'``, or a Python datetime
        object.
    batch_n : int, optional
        Only used when *date_time* is provided (archive mode).  If greater
        than 0, the PV list is split into chunks of this size and each chunk
        is fetched in a separate ``get_data_at_time`` call.  Each chunk is
        retried up to 3 times on failure (exception raised or any PV missing
        from the response).  A message is printed to stdout after each failed
        attempt.  If all 3 attempts for a chunk fail, a ``ValueError`` is
        raised listing the affected PVs and chunk index.  When ``batch_n``
        is 0 (the default) or *date_time* is ``None``, all PVs are fetched
        in a single call with no retry logic.

    Returns
    -------
    dict
        Mapping of PV name → value, e.g.
        ``{'QUAD:IN20:361:BDES': 3.5, 'LCAV:IN20:2:ADES': 65.0}``.

    Raises
    ------
    ValueError
        If any PV could not be read in live mode (timed out or does not
        exist), or if a batch chunk still has failing PVs after 3 retries
        in archive mode.
    meme.archive.ArchiveRetrievalError
        If the archiver request fails in archive mode (no-batch path).
    """
    if date_time is not None:
        from meme.archive import get_data_at_time

        if batch_n <= 0:
            return get_data_at_time(pvs, at=date_time, timeout=20)

        # Split into chunks and fetch with retry logic.
        chunks = [pvs[i:i + batch_n] for i in range(0, len(pvs), batch_n)]
        n_chunks = len(chunks)
        result = {}
        for chunk_idx, chunk in enumerate(chunks):
            max_attempts = 3
            last_error = None
            last_failed = None
            for attempt in range(1, max_attempts + 1):
                try:
                    chunk_result = get_data_at_time(chunk, at=date_time, timeout=20)
                    failed = [pv for pv in chunk if chunk_result.get(pv) is None]
                    if failed:
                        last_failed = failed
                        raise ValueError(
                            f"Missing {len(failed)} PV(s) in archive response: {failed}"
                        )
                    result.update(chunk_result)
                    last_error = None
                    last_failed = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < max_attempts:
                        print(
                            f"Batch {chunk_idx + 1}/{n_chunks}: attempt {attempt}/{max_attempts} "
                            f"failed ({exc}), retrying..."
                        )
            if last_error is not None:
                failed_pvs = last_failed if last_failed is not None else chunk
                raise ValueError(
                    f"Batch {chunk_idx + 1}/{n_chunks}: all {max_attempts} attempts failed "
                    f"for {len(failed_pvs)} PV(s): {failed_pvs}"
                ) from last_error
        return result

    import epics
    values = epics.caget_many(pvs)
    failed = [pv for pv, val in zip(pvs, values) if val is None]
    if failed:
        raise ValueError(
            f"Could not read {len(failed)} PV(s): {failed}"
        )
    return dict(zip(pvs, values))


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


def get_output(tao):
    """
    Returns dictionary of modeled parameters, including element name,
    twiss and Rmats
    """
    outkeys = ['ele.name', 'ele.ix_ele', 'ele.ix_branch', 'ele.a.beta', 'ele.a.alpha', 'ele.a.eta', 'ele.a.etap', 'ele.a.gamma', 'ele.a.phi', 'ele.b.beta', 'ele.b.alpha', 'ele.b.eta', 'ele.b.etap', 'ele.b.gamma', 'ele.b.phi', 'ele.x.eta', 'ele.x.etap', 'ele.y.eta', 'ele.y.etap', 'ele.s', 'ele.l', 'ele.e_tot', 'ele.p0c', 'ele.mat6', 'ele.vec0']
    return {k: tao.lat_list("*", k) for k in outkeys}
