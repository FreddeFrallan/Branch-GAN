import copy, json


class Config:

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_file(self, path):
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.to_dict(), fp)

    @classmethod
    def from_json_file(cls, json_file):
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
