# encoding: utf-8

import logging
import logging.config
# from src.tfidf_transformer import TfidfTransformer
from src.sentence_embedding.sentence_emb import UsableEncoder
from src.sentence_embedding.model import Encoder
from src.annoy_search import AnnoySearch
import json
from src.utils import Cutter
from src.config import FaqConfig, SkipConfig

logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


# def init_embedding_model(json_file):
#     logger = logging.getLogger('init_embedding_model')
#     tfidf_transformer = TfidfTransformer(max_features=256)
#     tfidf_transformer.train(json_file)
#     tfidf_transformer.save_model('tfidftransformer.pkl')
#     logger.info('init embedding model SUCCESS !')
#     return tfidf_transformer


def init_embedding_model(skip_config: SkipConfig):
    logger = logging.getLogger('init_embedding_model')
    usable_encoder = UsableEncoder(skip_config.model_file,
                                   skip_config.dict_file)

    logger.info('init embedding model SUCCESS !')
    return usable_encoder

def make_new_index(model, json_file, new_json_file):
    logger = logging.getLogger('make_new_index')

    cutter = Cutter()

    with open(json_file, 'r', encoding='utf-8') as f_in:
        with open(new_json_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                data = json.loads(line, encoding='utf-8')
                question = data['question']
                # data['question_vec'] = model.predict(question)
                data['question_tokens_zi'] = cutter.cut_zi_and_remove_punc(question)
                data['question_tokens_jieba'] = cutter.cut_and_remove_punc(question)
                question_clean = ' '.join(data['question_tokens_jieba'])
                data['question_vec'] = model.encode(question_clean.strip())[0].tolist()
                f_out.write(json.dumps(data, ensure_ascii=False))
                f_out.write('\n')

    logger.info('make new index success !')


def init_annoy_search_from_json(json_file, annoy_file, vec_dim=100, metric='angular', n_trees=100):
    logger = logging.getLogger('init_annoy_search')
    ann_s = AnnoySearch(vec_dim=vec_dim, metric=metric)
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_json = json.loads(line, encoding='utf-8')
            try:
                ann_s.add_item(line_json['id'], line_json['question_vec'])
            except IndexError:
                logger.error(
                    'dim of vector DOES NOT match with annoy, id ' + str(line_json['id']),
                    exc_info=False)
    ann_s.build_annoy(n_trees=n_trees)
    ann_s.save_annoy(annoy_file)
    logger.info('init annoy search SUCCESS !')


# def make_annoy(json_file, annoy_file, n_trees=100):
#     logger = logging.getLogger('make_annoy')
#
#     model = init_embedding_model(json_file)
#     vec_dim = model.get_feature_dims()
#
#     new_json_file = 'faq_vec.index'
#     make_new_index(model, json_file, new_json_file)
#
#     init_annoy_search_from_json(new_json_file, annoy_file, vec_dim=vec_dim, n_trees=n_trees)
#
#     logger.info('make annoy success !')

def make_annoy(faqconfig: FaqConfig, json_file, annoy_file, n_trees=100):
    logger = logging.getLogger('make_annoy')

    model = init_embedding_model(faqconfig.skip_embedding)
    vec_dim = Encoder.thought_size # skip embedding生成的vector的维度

    new_json_file = 'faq_vec.index'
    make_new_index(model, json_file, new_json_file)

    init_annoy_search_from_json(new_json_file, annoy_file, vec_dim=vec_dim, n_trees=n_trees)

    logger.info('make annoy success !')


if __name__ == '__main__':
    cf = FaqConfig('faq.config')
    cf.load()
    make_annoy(cf, 'faq.index', 'test.annoy', n_trees=10)
