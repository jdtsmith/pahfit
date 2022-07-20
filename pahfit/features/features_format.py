"""pahfit.features.feature_forrmat - Special handling for bounded (val,
    min, max) values, which are represented as 3 element masked arrays
    in pahfit.
"""
import numpy.ma as ma
from astropy.table import MaskedColumn
from astropy.table.pprint import TableFormatter
from pahfit.util import bounded_is_fixed, bounded_is_missing

def fmt_func(fmt):
    def _fmt(v):
        if ma.is_masked(v[0]):
            return "  <n/a>  "
        if ma.is_masked(v[1]):
            return f"{v[0]:{fmt}} (Fixed)"
        return f"{v[0]:{fmt}} ({v[1]:{fmt}}, {v[2]:{fmt}})"

    return _fmt

class BoundedMaskedColumn(MaskedColumn):
    """Masked column which groups rows into a single item for formatting.
    To be set as Table's `MaskedColumn'.
    """

    _omit_shape = False
    
    @property
    def shape(self):
        sh = super().shape
        return sh[0:-1] if self._omit_shape and len(sh) > 1 else sh

    def is_missing(self):
        """Return a boolean array indicating which elements of the
        column are missing (i.e. their value is masked).
        """
        return bounded_is_missing(self)
        
    def is_fixed(self):
        """Return a boolean array indicating which elements of the
        column are fixed (i.e. they have both bounds masked).
        """
        return bounded_is_fixed(self)
    

class BoundedParTableFormatter(TableFormatter):
    """Format bounded parameters.  Bounded parameters are 3-field
    arrays, corresponding to 'value', 'min', and 'max'.  To be set as
    Table's `TableFormatter'.
    """
    
    def _pformat_table(self, table, *args, **kwargs):
        bpcols = []
        try:
            colsh = [(col, col.shape) for col in table.columns.values()]
            BoundedMaskedColumn._omit_shape = True
            for col, sh in colsh:
                if len(sh) == 2 and sh[1] == 3:
                    bpcols.append((col, col.info.format))
                    col.info.format = fmt_func(col.info.format or "g")
            return super()._pformat_table(table, *args, **kwargs)
        finally:
            BoundedMaskedColumn._omit_shape = False
            for col, fmt in bpcols:
                col.info.format = fmt
