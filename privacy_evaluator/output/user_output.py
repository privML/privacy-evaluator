import json
import numpy as np


class UserOutput:
    def to_json(self, filter: np.ndarray = None) -> str:
        """
        output function for JSON
        :param filter: if needed this filters the output for the given keys
        """
        ret = {}
        if filter is not None:
            for key in filter:
                ret[key] = self._convert_to_list_if_needed(self.__dict__.get(key))
        else:
            for key, value in self.__dict__.items():
                ret[key] = self._convert_to_list_if_needed(value)
        return json.dumps(ret)

    def to_dict(self, filter: np.ndarray = None) -> dict:
        """
        output function for dicts
        :param filter: if needed this filters the output for the given keys
        """
        if filter is not None:
            ret = {}
            for key in filter:
                ret[key] = self.__dict__.get(key)
            return ret
        return self.__dict__

    def __str__(self) -> str:
        """
        Overwrite the String method so the output looks nicer
        """
        return self.to_json()

    def _convert_to_list_if_needed(self, obj):
        """
        Use internally to convert ndarray to list in order to turn it to json
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
