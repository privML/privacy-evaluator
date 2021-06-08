import json
import numpy as np


class UserOutput:
    def _to_json(self, filter: np.ndarray = None) -> str:
        """
        output function for JSON
        :param filter: if needed this filters the output for the given keys
        """
        if filter is not None:
            return json.dumps([self.__dict__.get(key) for key in filter])
        return json.dumps(self.__dict__)

    def _to_dict(self, filter: np.ndarray = None) -> dict:
        """
        output function for dicts
        :param filter: if needed this filters the output for the given keys
        """
        if filter is not None:
            return [self.__dict__.get(key) for key in filter]
        return self.__dict__

    def __str__(self) -> str:
        """
        Overwrite the String method so the output looks nicer
        """
        return self._to_json()
