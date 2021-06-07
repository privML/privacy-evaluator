import json
import numpy as np


class UserOutput:
    def _to_json(self, filter: np.ndarray = None) -> str:
        if filter is not None:
            return json.dumps([self.__dict__.get(key) for key in filter])
        return json.dumps(self.__dict__)

    def _to_dict(self, filter: np.ndarray = None) -> dict:
        if filter is not None:
            return [self.__dict__.get(key) for key in filter]
        return self.__dict__

    def __str__(self) -> str:
        return self._to_json()
