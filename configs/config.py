import yaml

#YAML中获取配置
def get_config(name):
    stream = open(name, 'r')
    config_dict = yaml.safe_load(stream)
    return Config(config_dict)

#将配置信息从字典形式转换为对象形式
class Config:
    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)
        # 遍历配置字典，将键值对作为属性设置到配置对象中。
        for key, val in in_dict.items():
            # 如果值是列表或元组，则将其转换为配置对象的列表
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Config(x) if isinstance(x, dict) else x for x in val])
            else:
                # 否则直接设置属性值
                setattr(self, key, Config(val) if isinstance(val, dict) else val)
