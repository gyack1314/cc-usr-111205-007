# 叫应到达流的自激强度与窗外残尾

这是独立的防汛演练资料Python项目。`data/`保存虚构人数、观测、时序或方案，`response_data.io`提供严格数值读取与来源字节SHA-256，`response_data.domain`给出已有原始摘要，`response_data.cli`输出JSON。字段、量纲与分析口径见[资料约定](docs/data-contract.md)。这些数据用于离线复盘，不连接真实人员、传感器或指挥系统。

Python 3.11及以上即可直接使用，无第三方Python依赖。在项目根目录执行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q response_data tests
python -m response_data.cli inspect data/normal.json
```

`inspect`显示原始资料概况，不覆盖源文件。`invalid-`前缀资料为不合法输入，其余变体使用相同字段保留不同业务边界。数值null须按字段语义区别于0，不在读取时自动填补。

Dockerfile固定基础镜像，可用`docker build -t response-data .`及`docker run --rm response-data`读取资料；`docker run --rm response-data python -m unittest discover -s tests -v`执行已有检查。运行所需资料全部在本地，无需登录或实时来源。
