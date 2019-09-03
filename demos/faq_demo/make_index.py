# encoding: utf-8

import json
import logging
import logging.config


logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


def make_index(faq_file, json_file):
    logger = logging.getLogger('make_index')

    is_header = True
    auto_id = False
    field_names = []
    field_num = 0
    idx = 0
    with open(faq_file, 'r', encoding='utf8') as f_faq:
        with open(json_file, 'w', encoding='utf8') as f_json:
            for line in f_faq:
                arr = line.strip().split('\t')
                if is_header:
                    is_header = False
                    field_num = len(arr)
                    if 'question' not in arr or 'answer' not in arr:
                        logger.critical(
                            'NO question or answer in head line, make index FAILURE !')
                        raise TypeError('NO question or answer !')
                    else:
                        field_names = arr
                    if 'id' not in arr:
                        auto_id = True
                else:
                    if len(arr) != field_num:
                        logger.error(
                            'INDEX ERROR : dim NOT match !')
                    else:
                        idx += 1
                        data = dict([field_names[i], arr[i]]
                                    for i in range(field_num))
                        if auto_id:
                            data['id'] = idx
                        f_json.write(json.dumps(data, ensure_ascii=False))
                        f_json.write('\n')
    logger.info('make index success !')


if __name__ == '__main__':
    faq_f = 'QA'
    json_f = 'faq.index'
    make_index(faq_f, json_f)
