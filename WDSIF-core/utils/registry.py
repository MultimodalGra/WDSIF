# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501


class Registry:

    def __init__(self, name):

        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert name not in self._obj_map, (
            f"An object named '{name}' was already registered "
            f"in '{self._name}' registry!"
        )
        self._obj_map[name] = obj

    def register(self, obj=None):

        if obj is None:

            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry!"
            )
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


DATASET_REGISTRY = Registry("dataset")
ARCH_REGISTRY = Registry("arch")
MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
METRIC_REGISTRY = Registry("metric")
