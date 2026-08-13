from dataclasses import dataclass

# @dataclass(...):为类自动生成常见方法，如__init__, __repr__, __eq___等
# slots=True:为类创建一个固定的属性集合，节省内存，防止用户随意添加属性，普通的Python对象使用__dict__来存储属性
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    # ignore_eos = False:生成过程中不忽略EOS(End Of Sequence)标记，遇到eos就停止生成，如果为True则反之
    ignore_eos: bool = False

    # @dataclass中的特殊方法，执行Python解释器自动生成__init__时，在最后执行self.__post_init__()，进行参数检查、
    # 初始化后处理、设置一些依赖其他字段的值等。assert断言温度必须大于1e-10，否则抛出异常，禁止贪心取样
    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
