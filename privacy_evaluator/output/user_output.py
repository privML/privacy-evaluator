import json
import numpy as np


class UserOutput:
    @staticmethod
    def _to_json(obj, filter: np.ndarray = None) -> str:
        """Serialize given object to JSON.

        :param obj: Object to serialize.
        :param filter: If needed this filters the output for the given keys
        """
        ret = {}
        if filter is not None:
            for key in filter:
                ret[key] = UserOutput._convert_to_list_if_needed(obj.__dict__.get(key))
        else:
            for key, value in obj.__dict__.items():
                ret[key] = UserOutput._convert_to_list_if_needed(value)
        return json.dumps(ret)


    @staticmethod
    def _convert_to_list_if_needed(obj):
        """
        Use internally to convert ndarray to list in order to turn it to json
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


    def to_json(self, filter: np.ndarray = None) -> str:
        """
        output function for JSON
        :param filter: if needed this filters the output for the given keys
        """
        UserOutput._to_json(self, filter=filter)


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

