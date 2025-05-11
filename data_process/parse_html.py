"""
@Time: 2024/9/15 20:15
@Author: xujinlingbj
@File: parse_html.py
"""
from bs4 import BeautifulSoup

from data_process.utils import contains_chinese, contains_uppercase_abcd


def parse_html_content(file_path=None, html_content=None):
    assert file_path or html_content
    if file_path:
        # 读取 HTML 文件内容
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
    # 创建BeautifulSoup对象
    soup = BeautifulSoup(html_content, 'lxml')

    def is_inside_table(element):
        if element.name == 'table':
            return element
        for parent in element.parents:
            if parent.name == 'table':
                return parent
        return None

    def find_parent_tr(element):
        if element.name in ['tr', 'table']:
            return element
        for parent in element.parents:
            if parent.name in ['tr', 'table']:
                return parent
        return None

    def is_inside_paragraph(element):
        if element.name in ['p']:
            return True
        for parent in element.parents:
            if parent.name in ['p']:
                return True
        return False

    def is_inside_span(element):
        if element.name == 'span':
            return True
        for parent in element.parents:
            if parent.name == 'span':
                return True
        return False

    def table_has_style(element):
        style_value = element.get('style')
        return style_value

    # 用于存储解析后的内容
    parsed_content = []
    pre_content = ''
    first_question = False
    # 遍历每一个元素，保留文字和图片的位置
    for element in soup.body.descendants:
        xx = element.parent.name
        if element.name == 'img':
            # 如果是图片标签，添加图片的相关信息到解析内容
            img_url = element.get('src')
            if img_url == pre_content:
                continue
            parsed_content.append({'type': 'image', 'src': img_url, 'name': 'img'})
            pre_content = img_url
        elif element.string and element.string.strip():
            if element.string[0] == '1':
                first_question = True
            # if element.string.strip() in saved_content:
            #     continue
            if element.string.strip() == pre_content:
                continue
            pre_content = element.string.strip()
            tabel_element = is_inside_table(element)
            is_style = 0
            is_tabel = False
            index = 0
            if tabel_element:
                is_tabel = True
                # parent = element.find_parent(['table', 'thead', 'tbody'])
                if tabel_element != element:
                    # 获取父元素下所有的tr子元素
                    tr_children = [child for child in tabel_element.find_all('tr')]
                    # 计算当前tr的索引

                    parent = find_parent_tr(element)

                    index = tr_children.index(parent)

                style_value = table_has_style(tabel_element)
                if style_value is None:
                    is_style = 0
                else:
                    is_style = len(style_value)

            is_paragraph = is_inside_paragraph(element)
            is_span = is_inside_span(element)


            # 如果是文字节点，添加文字到解析内容
            parsed_content.append({'type': 'text', 'content': element.string.strip(),
                                   'is_paragraph': is_paragraph, 'is_tabel': is_tabel,
                                   'is_span': is_span, 'is_style': is_style,
                                   'index': index, 'first_question': first_question})
    # print(file_path)
    # print(html_content)
    # for x in parsed_content:
    #     print(x)

    # 问题 'is_paragraph': False, 'is_tabel': False, 'is_span': False
    paragraph_data = ""
    question_data = []
    paragraph_image_list = []
    question_image_list = []
    temp_question_image_list = []
    pre_tag = 'p'
    add_option = False
    temp_question = ''
    for i in range(0, len(parsed_content)):
        if parsed_content[i]['type'] == 'image':
            image_name = parsed_content[i]['src'].split('com/')[1].replace('/', '_').split('.')[0]

            if pre_tag == 'p':
                paragraph_data += '<image>\n'
                paragraph_image_list.append(image_name)
            else:
                temp_question += '<image>\n'
                temp_question_image_list.append(image_name)
        # 题目中表格
        elif parsed_content[i]['is_style'] > 50 and parsed_content[i]['is_tabel']:
            paragraph_data += parsed_content[i]['content']

            if i + 1 < len(parsed_content) and 'is_style' in parsed_content[i+1] and parsed_content[i+1]['is_style'] > 50 and parsed_content[i]['index'] == parsed_content[i+1]['index']:
                paragraph_data += '\t'
            else:
                paragraph_data += '\n'
        elif not parsed_content[i]['first_question']:
            paragraph_data += parsed_content[i]['content']
            if contains_chinese(parsed_content[i]['content']):
                paragraph_data += '\n'
            pre_tag = 'p'
        # 是问题
        elif parsed_content[i]['is_paragraph'] is False and parsed_content[i]['is_tabel'] is False:
            if add_option:
                question_data.append(temp_question)
                question_image_list.append(temp_question_image_list)
                temp_question_image_list = []
                add_option = False
                temp_question = parsed_content[i]['content']
            else:
                temp_question += parsed_content[i]['content']
            if contains_chinese(parsed_content[i]['content']):
                temp_question += '\n'
            pre_tag = 'q'
        elif (parsed_content[i]['is_paragraph']) and parsed_content[i]['is_tabel'] is False:
            paragraph_data += parsed_content[i]['content']
            if contains_chinese(parsed_content[i]['content']):
                paragraph_data += '\n'
            pre_tag = 'p'
        else:
            pre_tag = 'q'
            if parsed_content[i]['is_tabel']:
                add_option = True
            temp_question += parsed_content[i]['content']
            if contains_chinese(parsed_content[i]['content']):
                temp_question += '\n'

    if len(temp_question) > 0:
        question_data.append(temp_question)
        question_image_list.append(temp_question_image_list)
    # print(question_data)
    # print(paragraph_data)
    if len(question_data) > 1 and len(paragraph_data) == 0:
        print('*'*30)
        print(file_path)
        print(html_content)
        for x in parsed_content:
            print(x)
        return [], [], []
    res_text = []
    for idd, query in enumerate(question_data):
        # query = remove_leading_numbering(query)
        if not contains_uppercase_abcd(query):
            # print(f'filter question:{query}')
            continue
        query = query.replace('\n', '')
        now_data = paragraph_data + "\n" + query + "\n你的答案是："
        image_count = now_data.count("<image>")
        question_image_num = len(question_image_list[idd]) + len(paragraph_image_list)
        # if question_image_num != image_count:
        #     print(html_content)
        #     for x in parsed_content:
        #         print(x)
        #     sys.exit(0)

        res_text.append(now_data)
        # print('*'*30)
        # print(now_data)
    if len(question_data) == 0 and len(paragraph_data) > 0:
        now_data = paragraph_data + "\n" + "你的答案是："
        res_text.append(now_data)
    # print(paragraph_image_list)
    # print(question_image_list)
    # sys.exit(0)

    return res_text, paragraph_image_list, question_image_list

