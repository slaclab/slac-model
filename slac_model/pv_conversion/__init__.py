"""
Per-backend PV<->element accessor functions used to convert SLAC 
machine PV values to simulation parameters.

Each backend provides set/get methods that allow reading and writing of element 
PV-equivalent values from each simulation backend.

For example, to write the BCTRL value of QE04 (in kG) to the element in a Bmad / PyTao object:

```python
from slac_model.pv_conversion.bmad import set_quadrupole_bctrl

set_quadrupole_bctrl(tao_simulator, "QE04", 1.23)
```
to do the equivelent in impact:

```python
from slac_model.pv_conversion.impact import set_quadrupole_bctrl

set_quadrupole_bctrl(impact_simulator, "QE04", 1.23)
```

"""
