import re
from typing import Optional

import slac_db.device as dev

# Example: dev.get_attribute("QDMP2", "cs_name") returns the control system name for a MAD element

_MODELED_DEVICE_TYPES = ['BEND', 'BTRM', 'QUAD', 'XCOR', 'YCOR', 'LCAV']

# Lazily-built reverse map: cs_name -> device_name (MAD element)
_cs_to_mad = {}

# Lazily-built reverse map: full accessor PV -> device_name (MAD element).
# Populated from dev.get_all_accessors() for every device in the database.
# Used to resolve ACCL:*/KLY_PDES and ACCL:*/KLY_ADES PVs that are stored as
# beamcode-specific accessor values rather than as standalone cs_name entries.
_accessor_to_mad = {}

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


def _build_accessor_to_mad():
    """Populate the module-level _accessor_to_mad reverse-lookup cache.

    Issues a single bulk SELECT against the accessors table instead of
    calling get_all_accessors() once per device, reducing build time from
    ~22 s to ~0.05 s.  This covers beamcode-specific ACCL:* PVs (e.g.
    ``ACCL:LI24:300:KLY_PDES:SETDATA_1``) that are stored as accessor
    values rather than as standalone cs_names in the devices table.
    """
    import sqlalchemy
    global _accessor_to_mad
    # Ensure the DB is initialised (_meta is set by the first get_devices call).
    if not dev._meta:
        dev.get_devices()
    with dev._meta.session() as s:
        rows = s.execute(
            sqlalchemy.select(dev._meta.t.accessors)
        ).fetchall()
    for row in rows:
        cs = row._mapping['cs_address']
        dn = row._mapping['device_name']
        if cs and cs not in _accessor_to_mad:
            _accessor_to_mad[cs] = dn


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


# ---------------------------------------------------------------------------
# Attribute alias normalisation: map read-only / alternate suffixes to the
# canonical writable name so the dispatch table in pv_to_tao_cmd stays compact.
# ---------------------------------------------------------------------------
_ATTR_ALIASES = {
    # Magnets
    "BDES":     "BCTRL",
    "BACT":     "BCTRL",
    # Cavity amplitude
    "AACTMEAN": "ADES",
    "AREQ":     "ADES",
    # Cavity phase
    "PACTMEAN": "PDES",
    "PREQ":     "PDES",
    "PACT":     "PDES",
    # Beamcode-specific klystron PV compound attributes (ACCL:*/KLY_*:SETDATA_N)
    # These are handled explicitly in _pv_to_tao_cmd_single before the alias
    # table is consulted, but are listed here for documentation purposes.
    "KLY_PDES": "PDES",
    "KLY_ADES": "ADES",
}


def pv_to_tao_cmd(pv_dict: dict, tao=None) -> list:
    """Convert a dictionary of EPICS PV / value pairs to a list of Tao set commands.

    Looks up the Bmad element name for each PV's base device via the SLAC
    device database, applies the appropriate unit conversion, and returns a
    list of Tao command strings ready to pass to ``tao.cmd()``.

    PVs that cannot be converted are skipped with a ``UserWarning``; they do
    not abort processing of the remaining PVs.

    Beamcode-specific klystron PVs of the form
    ``ACCL:<sector>:<station>:KLY_PDES:SETDATA_<N>`` (phase [deg]) and
    ``ACCL:<sector>:<station>:KLY_ADES:SETDATA_<N>`` (amplitude [MV]) are
    handled specially: the MAD element is resolved via the accessor reverse-
    lookup map, and the conversions ``PDES → PHI0`` (÷ 360) and
    ``ADES → VOLTAGE`` (× 1e6) are applied as for LCAV devices.

    Unit conversion reference:
        slac-bmad-tools/references/bmad_unit_conversions.md

    Parameters
    ----------
    pv_dict : dict
        Mapping of full EPICS PV name → value in control-system units, e.g.
        ``{'QUAD:IN20:361:BCTRL': 3.5, 'LCAV:IN20:2:ADES': 65.0}``.
        The last colon-separated token of each key is taken as the attribute
        suffix; everything before it is the cs_name used for the DB lookup.
        For ``ACCL:*/KLY_PDES/KLY_ADES`` PVs the first three colon-parts form
        the device base and the fourth part is the compound attribute name.
    tao : pytao.Tao, optional
        Live Tao instance.  Required only for QUAD (needs element length L)
        and BEND / BTRM (needs G and P0C).  May be omitted for all other
        device types.

    Returns
    -------
    list of str
        Tao command strings for all successfully converted PVs, e.g.
        ``['set ele QE01 B1_GRADIENT = -0.3456', 'set ele L0A VOLTAGE = 65000000.0']``.
        PVs that raised an error during conversion are omitted and a
        ``UserWarning`` is issued for each.

    Examples
    --------
    >>> cmds = pv_to_tao_cmd({'QUAD:IN20:361:BCTRL': 3.5}, tao)
    >>> cmds
    ['set ele QE01 B1_GRADIENT = -0.12727...']

    >>> cmds = pv_to_tao_cmd({'XCOR:IN20:221:BCTRL': 0.5, 'LCAV:IN20:2:ADES': 65.0})
    >>> cmds
    ['set ele XC01B BL_KICK = -0.05', 'set ele L0A VOLTAGE = 65000000.0']
    """
    import warnings

    cmds = []
    for pv, value in pv_dict.items():
        try:
            cmd = _pv_to_tao_cmd_single(pv, value, tao)
            cmds.append(cmd)
        except Exception as exc:
            warnings.warn(f"Skipping PV {pv!r}: {exc}")
    return cmds


def _pv_to_tao_cmd_single(pv: str, value: float, tao=None) -> str:
    """Convert a single EPICS PV / value pair to a Tao set command.

    Internal helper used by :func:`pv_to_tao_cmd`.  Raises ``ValueError`` on
    any conversion error.

    Beamcode-specific cavity PVs of the form ``ACCL:<sec>:<sta>:<attr_token>``
    (4-part) or ``ACCL:<sec>:<sta>:KLY_PDES:SETDATA_<N>`` (5-part) are
    resolved via the accessor reverse-lookup map (``_accessor_to_mad``).
    The attribute kind is inferred from the token at position 3:

    * contains ``PDES`` or ``PACT`` → ``PHI0`` [turns]  = value / 360
    * contains ``ADES`` or ``AACT`` → ``VOLTAGE`` [V]   = value × 1e6
    """
    # ------------------------------------------------------------------
    # 1. Parse PV into base device (cs_name) and attribute suffix.
    # ------------------------------------------------------------------
    parts = pv.split(":")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse PV {pv!r}: expected at least one ':' separator")

    base_device = ":".join(parts[:-1])
    attr = parts[-1]
    device_type = parts[0]   # e.g. "QUAD", "XCOR", "KLYS", "LCAV", "BEND"

    # ------------------------------------------------------------------
    # 1b. Special handling for ACCL:* beamcode-specific PVs.
    #
    # Beamcode-specific cavity phase/amplitude PVs are stored as accessor
    # values on DB entries rather than as standalone cs_names.  They come
    # in two structural flavours:
    #
    #   5-part:  ACCL:<sec>:<sta>:KLY_PDES:SETDATA_<N>  (K24 klystrons)
    #            ACCL:<sec>:<sta>:KLY_ADES:SETDATA_<N>
    #
    #   4-part:  ACCL:<sec>:<sta>:<prefix>_PDES_DS<N>   (L0A/L0B/L1S …)
    #            ACCL:<sec>:<sta>:<prefix>_ADES_DS<N>
    #            ACCL:<sec>:<sta>:<prefix>_PACT_DS<N>
    #            ACCL:<sec>:<sta>:<prefix>_AACT_DS<N>
    #
    # The attribute kind is inferred from the suffix token:
    #   contains 'PDES' or 'PACT'  → phase  [deg] → PHI0   [turns] = /360
    #   contains 'ADES' or 'AACT'  → amplitude [MV] → VOLTAGE [V]  = *1e6
    #
    # The MAD element is resolved via the full-accessor reverse-lookup map.
    # ------------------------------------------------------------------
    if device_type == "ACCL":
        global _accessor_to_mad
        if not _accessor_to_mad:
            _build_accessor_to_mad()
        element_name = _accessor_to_mad.get(pv)
        if element_name is None:
            raise ValueError(
                f"PV base {base_device!r} not found in the device database. "
                "Check that the cs_name exists and the DB cache is populated."
            )
        # Determine phase vs. amplitude from the compound attribute token(s).
        # For 5-part PVs parts[3] is 'KLY_PDES'/'KLY_ADES'; for 4-part PVs
        # parts[3] is the full token like 'L0A_ADES_DS0'.
        attr_token = parts[3].upper()  # e.g. 'KLY_PDES', 'L0A_ADES_DS0'
        if "PDES" in attr_token or "PACT" in attr_token:
            bmad_val = value / 360.0
            return f"set ele {element_name} PHI0 = {bmad_val}"
        elif "ADES" in attr_token or "AACT" in attr_token:
            bmad_val = value * 1e6
            return f"set ele {element_name} VOLTAGE = {bmad_val}"
        else:
            raise ValueError(
                f"Cannot determine phase/amplitude for ACCL PV {pv!r}: "
                f"attribute token {attr_token!r} does not contain PDES, PACT, ADES, or AACT."
            )

    # ------------------------------------------------------------------
    # 2. Look up the Bmad / MAD element name via the device database.
    # ------------------------------------------------------------------
    element_name = model_name_convert([base_device], type='bmad')[0]
    if element_name == base_device:
        raise ValueError(
            f"PV base {base_device!r} not found in the device database. "
            "Check that the cs_name exists and the DB cache is populated."
        )

    # ------------------------------------------------------------------
    # 3. Normalise attribute aliases to canonical names.
    # ------------------------------------------------------------------
    attr_canonical = _ATTR_ALIASES.get(attr, attr)

    # ------------------------------------------------------------------
    # 4. Dispatch by device type and attribute; apply unit conversion.
    # ------------------------------------------------------------------

    # --- Quadrupole ---------------------------------------------------
    # EPICS BCTRL [kG] = -B1_GRADIENT [T/m] * L [m] * 10
    # → B1_GRADIENT = -value / (L * 10)
    if device_type == "QUAD":
        if attr_canonical == "BCTRL":
            if tao is None:
                raise ValueError(
                    f"tao instance required to convert QUAD:{attr} "
                    "(element length L is needed for the kG → T/m conversion)"
                )
            ele_attr = tao.ele_gen_attribs(element_name)
            L = ele_attr["L"]
            if L == 0:
                import warnings
                warnings.warn(
                    f"QUAD element {element_name!r} has zero length; "
                    "B1_GRADIENT conversion skipped."
                )
                return f"! set ele {element_name} B1_GRADIENT = {-value} / ({L} * 10)"
            bmad_val = -value / (L * 10.0)
            return f"set ele {element_name} B1_GRADIENT = {bmad_val}"
        raise ValueError(f"Unsupported attribute {attr!r} for device type 'QUAD' in PV {pv!r}")

    # --- Solenoid -----------------------------------------------------
    # EPICS BCTRL [kG] = BS_FIELD [T] * 10
    # → BS_FIELD = value / 10
    if device_type == "SOLN":
        if attr_canonical == "BCTRL":
            bmad_val = value / 10.0
            return f"set ele {element_name} BS_FIELD = {bmad_val}"
        raise ValueError(f"Unsupported attribute {attr!r} for device type 'SOLN' in PV {pv!r}")

    # --- Horizontal / Vertical corrector (kicker) ---------------------
    # EPICS BCTRL [kG] = BL_KICK [T·m] * -10
    # → BL_KICK = -value / 10
    if device_type in ("XCOR", "YCOR"):
        if attr_canonical == "BCTRL":
            bmad_val = -value / 10.0
            return f"set ele {element_name} BL_KICK = {bmad_val}"
        raise ValueError(
            f"Unsupported attribute {attr!r} for device type {device_type!r} in PV {pv!r}"
        )

    # --- Bending magnet -----------------------------------------------
    # EPICS BCTRL [GeV/c] = P0C [eV/c] * (1 + DG/G) * 1e-9
    # → DG = G * (value * 1e9 - P0C) / P0C
    if device_type in ("BEND", "BTRM"):
        if attr_canonical == "BCTRL":
            if tao is None:
                raise ValueError(
                    f"tao instance required to convert BEND:{attr} "
                    "(G and P0C are needed for the GeV/c → DG conversion)"
                )
            ele_attr = tao.ele_gen_attribs(element_name)
            g = ele_attr["G"]
            p0c = ele_attr["P0C"]   # eV/c
            if g == 0:
                raise ValueError(
                    f"Cannot convert {pv!r}: element {element_name!r} has G=0 "
                    "(straight element — no bending-strength conversion is defined)"
                )
            dp = (value * 1e9 - p0c) / p0c
            bmad_val = dp * g
            return f"set ele {element_name} DG = {bmad_val}"
        raise ValueError(
            f"Unsupported attribute {attr!r} for device type {device_type!r} in PV {pv!r}"
        )

    # --- Klystron -----------------------------------------------------
    # ENLD [MeV] → ENLD_MEV  (pass-through, units already match)
    # PDES [deg] → PHASE_DEG  (pass-through)
    # BEAMCODE*_STAT: EPICS "0"=active(IN_USE=True), "1"=inactive(IN_USE=False)
    if device_type == "KLYS":
        if attr_canonical == "ENLD":
            return f"set ele {element_name} ENLD_MEV = {value}"
        if attr_canonical == "PDES":
            return f"set ele {element_name} PHASE_DEG = {value}"
        if attr in ("BEAMCODE1_STAT", "BEAMCODE2_STAT"):
            # Inverted logic: EPICS 0 → klystron active (IN_USE = True)
            in_use = (int(value) == 0)
            return f"set ele {element_name} IN_USE = {in_use}"
        raise ValueError(f"Unsupported attribute {attr!r} for device type 'KLYS' in PV {pv!r}")

    # --- Linac cavity (SC) --------------------------------------------
    # ADES [MV] → VOLTAGE [V]:  VOLTAGE = value * 1e6
    # PDES [deg] → PHI0 [turns]: PHI0 = value / 360
    if device_type == "LCAV":
        if attr_canonical == "ADES":
            bmad_val = value * 1e6
            return f"set ele {element_name} VOLTAGE = {bmad_val}"
        if attr_canonical == "PDES":
            bmad_val = value / 360.0
            return f"set ele {element_name} PHI0 = {bmad_val}"
        raise ValueError(f"Unsupported attribute {attr!r} for device type 'LCAV' in PV {pv!r}")

    # --- Catch-all ----------------------------------------------------
    raise ValueError(
        f"Unsupported device type {device_type!r} in PV {pv!r}. "
        f"Supported types: QUAD, SOLN, XCOR, YCOR, BEND, BTRM, KLYS, LCAV."
    )
