"""Single-file potential bundles and explicitly activated AL sessions."""
from types import SimpleNamespace
from ._imrmlp import _ModelBundle
from .atoms import Atoms
from .utils import _coerce_atoms
from .potential import Potential


class ModelBundle:
    @classmethod
    def load(cls, filename):
        obj = cls.__new__(cls)
        obj._impl = _ModelBundle.load(str(filename))
        obj.filename = str(filename)
        return obj

    @classmethod
    def from_settings(cls, settings):
        obj = cls.__new__(cls)
        obj._impl = _ModelBundle.from_settings({str(k): str(v) for k, v in settings.items()})
        obj.filename = ""
        return obj

    inspect = staticmethod(lambda filename: _ModelBundle.inspect(str(filename)))
    validate = staticmethod(lambda filename: _ModelBundle.validate(str(filename)))

    @property
    def potential(self):
        obj = Potential.__new__(Potential)
        obj._inner = self._impl.potential
        obj._wrapper = SimpleNamespace(pottype=self._impl.pottype, potfile=self.filename)
        obj._bundle = self
        return obj

    def save(self, filename, profile="preserve"):
        self._impl.save(str(filename), profile)

    def attach_al(self, session):
        self._impl.attach_al(session._impl)

    def attach_selector(self, selector):
        self._impl.attach_selector(selector._impl)

    def create_al(self, **overrides):
        """Restore grading, candidate collection, and selection explicitly."""
        return ALSession(self, self._impl.create_al({k: str(v) for k, v in overrides.items()}))


class ALSession:
    def __init__(self, bundle, impl):
        self.bundle = bundle
        self._impl = impl

    def evaluate(self, atoms):
        return self._impl.evaluate(_coerce_atoms(atoms)._impl)

    def check(self, atoms):
        return self._impl.check(_coerce_atoms(atoms)._impl)

    @property
    def pool(self):
        return [Atoms(_impl=a) for a in self._impl.pool]

    @property
    def should_abort(self):
        return self._impl.should_abort

    @property
    def selector(self):
        from .al import ActiveSelector
        inner = self._impl.selector
        if inner is None:
            return None
        obj = ActiveSelector.__new__(ActiveSelector)
        obj._impl = inner
        obj._session = self
        return obj

    def save(self, filename):
        self._impl.save(str(filename), self.bundle._impl.potential)
