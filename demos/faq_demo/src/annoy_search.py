# encoding: utf-8

from src.config import AnnoyConfig
from src.utils import singleton
import logging.config
import logging
from annoy import AnnoyIndex
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


logging.config.fileConfig(fname='log.config', disable_existing_loggers=False)


# 使用annoy进行相似(最近邻)搜索
@ singleton
class AnnoySearch:

    def __init__(self, vec_dim=100, metric='angular'):
        self.vec_dim = vec_dim  # 要index的向量维度
        self.metric = metric    # 度量可以是"angular"，"euclidean"，"manhattan"，"hamming"，或"dot"
        self.annoy_instance = AnnoyIndex(self.vec_dim, self.metric)
        self.logger = logging.getLogger('AnnoySearch')

    def save_annoy(self, annoy_file, prefault=False):
        self.annoy_instance.save(annoy_file, prefault=prefault)
        self.logger.info('save annoy SUCCESS !')

    def unload_annoy(self):
        self.annoy_instance.unload()

    def load_annoy(self, annoy_file, prefault=False):
        try:
            self.annoy_instance.unload()
            self.annoy_instance.load(annoy_file, prefault=prefault)
            self.logger.info('load annoy SUCCESS !')
        except FileNotFoundError:
            self.logger.error(
                'annoy file DOES NOT EXIST , load annoy FAILURE !',
                exc_info=True)

    # 创建annoy索引
    def build_annoy(self, n_trees):
        self.annoy_instance.build(n_trees)

    # 查询最近邻，通过index
    def get_nns_by_item(
            self,
            index,
            nn_num,
            search_k=-1,
            include_distances=False):
        return self.annoy_instance.get_nns_by_item(
            index, nn_num, search_k, include_distances)

    # 查询最近邻，通过向量
    def get_nns_by_vector(
            self,
            vec,
            nn_num,
            search_k=-1,
            include_distances=False):
        return self.annoy_instance.get_nns_by_vector(
            vec, nn_num, search_k, include_distances)

    def get_n_items(self):
        return self.annoy_instance.get_n_items()

    def get_n_trees(self):
        return self.annoy_instance.get_n_trees()

    def get_vec_dim(self):
        return self.vec_dim

    # 添加item
    def add_item(self, index, vec):
        self.annoy_instance.add_item(index, vec)

    def get_item_vector(self, index):
        return self.annoy_instance.get_item_vector(index)


def init_annoy_search(ann_config: AnnoyConfig):
    logger = logging.getLogger('init_annoy_search')
    ann_s = AnnoySearch(vec_dim=ann_config.vec_dim)
    try:
        ann_s.load_annoy(ann_config.annoy_file)
        logger.info('init annoy search SUCCESS !')
    except IndexError:
        logger.error(
            'ERROR : vector length DOES NOT match dim of vector of loaded annoy !',
            exc_info=True)


def search_by_vector(
        vector,
        vec_dim=100,
        top_n=15,
        search_k=-1,
        include_distance=True):
    logger = logging.getLogger('search_by_vector')
    ann_s = AnnoySearch(vec_dim=vec_dim)
    sea_res = None
    try:
        sea_res = ann_s.get_nns_by_vector(
            vec=vector,
            nn_num=top_n,
            search_k=search_k,
            include_distances=include_distance)
        logger.debug('annoy search by vector: ' + str(sea_res))
    except IndexError:
        logger.error(
            'ERROR : vector length DOES NOT match dim of vector of loaded annoy ! vector dim is: ' +
            str(len(vector)) +
            ', annoy dim is: ' +
            str(vec_dim),
            exc_info=True)

    return sea_res


if __name__ == '__main__':

    from src.tfidf_transformer import TfidfTransformer
    from src.elastic_search import search_by_ids
    from src.config import init_faq_config, FaqConfig

    faq_config = init_faq_config('faq.config')
    tt = TfidfTransformer()
    tt.load_model('tfidftransformer.pkl')

    v = tt.predict('who are you？')
    print(v)
    print(tt.get_feature_dims())
    init_annoy_search(faq_config.annoy_search)

    res = search_by_vector(
        v,
        vec_dim=FaqConfig('faq.config').annoy_search.vec_dim,
        include_distance=True)
    print(res)
    rsp = search_by_ids(res[0])
    print(rsp)
