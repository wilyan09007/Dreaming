"""Allow-list unpickler for the downloaded RouterBench pickle. Exactly the 11 globals the file references
(enumerated with an inert-stub unpickler that imports and runs nothing); everything else raises."""
import pickle
ALLOW = {("builtins", "slice"), ("numpy", "dtype"), ("numpy", "ndarray"), ("numpy.core.multiarray", "_reconstruct"),
         ("numpy.core.numeric", "_frombuffer"), ("pandas._libs.internals", "_unpickle_block"), ("pandas.core.frame", "DataFrame"),
         ("pandas.core.indexes.base", "Index"), ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.range", "RangeIndex"),
         ("pandas.core.internals.managers", "BlockManager")}
class RU(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) in ALLOW:
            return super().find_class(mod, name)
        raise pickle.UnpicklingError(f"blocked {mod}.{name}")
def load(path):
    with open(path, "rb") as f:
        return RU(f).load()

if __name__ == "__main__":  # self-check: the exec-capable callable that passed the old substring filter is now blocked
    import io
    class Evil:
        def __reduce__(self):
            import numpy.testing._private.utils as u
            return (u.runstring, ("print('pwned')", {}))
    try:
        RU(io.BytesIO(pickle.dumps(Evil()))).load(); raise SystemExit("FAIL: not blocked")
    except pickle.UnpicklingError as e:
        print("blocked as expected:", e)
