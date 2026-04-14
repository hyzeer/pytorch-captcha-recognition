# -*- coding: UTF-8 -*-
import numpy as np
import captcha_setting


def encode(text):
    # 动态获取长度，确保永远是 4 * 58 = 232
    char_set_len = captcha_setting.ALL_CHAR_SET_LEN
    max_captcha = captcha_setting.MAX_CAPTCHA
    vector = np.zeros(char_set_len * max_captcha, dtype=float)

    for i, c in enumerate(text):
        try:
            # 直接去配置文件的列表中找字符对应的下标
            # 这样 'a' 对应的下标就是它在 ALL_CHAR_SET 里的位置，不会越界
            idx = i * char_set_len + captcha_setting.ALL_CHAR_SET.index(c)
            vector[idx] = 1.0
        except ValueError:
            # 如果图片文件名里包含了你没定义的字符，会报错提醒
            raise ValueError(f"错误：字符 '{c}' 不在 captcha_setting.ALL_CHAR_SET 定义的列表中！")
    return vector


def decode(vec):
    char_set_len = captcha_setting.ALL_CHAR_SET_LEN
    # 将一维向量转回二维矩阵 (4, 58)
    vec = vec.reshape(captcha_setting.MAX_CAPTCHA, char_set_len)
    text = []
    for row in vec:
        # 找到每一行概率最大的索引
        idx = np.argmax(row)
        text.append(captcha_setting.ALL_CHAR_SET[idx])
    return "".join(text)


if __name__ == '__main__':
    # 测试一下
    # 注意：确保 "BK7H" 里的每个字母都在你的 captcha_setting.ALL_CHAR_SET 里
    test_str = "BK7H"
    try:
        e = encode(test_str)
        print(f"编码成功，向量长度: {len(e)}")
        print(f"解码结果: {decode(e)}")
    except Exception as err:
        print(err)