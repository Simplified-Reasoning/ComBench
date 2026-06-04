把这题的构造目标限制为“最小可行规模”的显式示例（即 n=9，注意不能把 n=9 这个信息泄露到 instruction 中），并保持要求尽量简单、严格、可验证

instruction 强调：
1) 只需要输出一个构造示例；
2) 构造格式是 Python 二维列表（list of lists）；
3) 元素只能是字符串 "I"、"M"、"O"。

ref_construction 提供一个可直接通过校验的 9x9 示例:
IIIMMMOOO
MMMOOOIII
OOOIIIMMM
IIIMMMOOO
MMMOOOIII
OOOIIIMMM
IIIMMMOOO
MMMOOOIII
OOOIIIMMM