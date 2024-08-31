import collections
import json
import logging
import os
import pickle
import shutil
from base64 import b64decode

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


MODEL_MAP = {
    # vocab.txt/special_tokens_map.json/tokenizer_config.json are common to the default model.
    "uie-base": {
        "resource_file_urls": {
            "model_state.pdparams":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.0/model_state.pdparams",
            "model_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
            "vocab.txt":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json"
        }
    },
    "uie-medium": {
        "resource_file_urls": {
            "model_state.pdparams":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium_v1.0/model_state.pdparams",
            "model_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium/model_config.json",
            "vocab.txt":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-mini": {
        "resource_file_urls": {
            "model_state.pdparams":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini_v1.0/model_state.pdparams",
            "model_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini/model_config.json",
            "vocab.txt":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-micro": {
        "resource_file_urls": {
            "model_state.pdparams":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro_v1.0/model_state.pdparams",
            "model_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro/model_config.json",
            "vocab.txt":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-nano": {
        "resource_file_urls": {
            "model_state.pdparams":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano_v1.0/model_state.pdparams",
            "model_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano/model_config.json",
            "vocab.txt":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-medical-base": {
        "resource_file_urls": {
            "model_state.pdparams":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medical_base_v0.1/model_state.pdparams",
            "model_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
            "vocab.txt":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-base-en": {
        "resource_file_urls": {
            "model_state.pdparams":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en_v1.1/model_state.pdparams",
            "model_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/model_config.json",
            "vocab.txt":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/vocab.txt",
            "special_tokens_map.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/special_tokens_map.json",
            "tokenizer_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/tokenizer_config.json",
        }
    },
    # Rename to `uie-medium` and the name of `uie-tiny` will be deprecated in the future.
    "uie-tiny": {
        "resource_file_urls": {
            "model_state.pdparams":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny_v0.1/model_state.pdparams",
            "model_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/model_config.json",
            "vocab.txt":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/vocab.txt",
            "special_tokens_map.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/special_tokens_map.json",
            "tokenizer_config.json":
                "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/tokenizer_config.json"
        }
    },
    "ernie-3.0-base-zh": {
        "resource_file_urls": {
            "model_state.pdparams":
                "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh.pdparams",
            "model_config.json":
                "base64:ew0KICAiYXR0ZW50aW9uX3Byb2JzX2Ryb3BvdXRfcHJvYiI6IDAuMSwNCiAgImhpZGRlbl9hY3QiOiAiZ2VsdSIsDQogICJoaWRkZW5fZHJvcG91dF9wcm9iIjogMC4xLA0KICAiaGlkZGVuX3NpemUiOiA3NjgsDQogICJpbml0aWFsaXplcl9yYW5nZSI6IDAuMDIsDQogICJtYXhfcG9zaXRpb25fZW1iZWRkaW5ncyI6IDIwNDgsDQogICJudW1fYXR0ZW50aW9uX2hlYWRzIjogMTIsDQogICJudW1faGlkZGVuX2xheWVycyI6IDEyLA0KICAidGFza190eXBlX3ZvY2FiX3NpemUiOiAzLA0KICAidHlwZV92b2NhYl9zaXplIjogNCwNCiAgInVzZV90YXNrX2lkIjogdHJ1ZSwNCiAgInZvY2FiX3NpemUiOiA0MDAwMCwNCiAgImluaXRfY2xhc3MiOiAiRXJuaWVNb2RlbCINCn0=",
            "vocab.txt":
                "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh_vocab.txt",
            "special_tokens_map.json":
                "base64:eyJ1bmtfdG9rZW4iOiAiW1VOS10iLCAic2VwX3Rva2VuIjogIltTRVBdIiwgInBhZF90b2tlbiI6ICJbUEFEXSIsICJjbHNfdG9rZW4iOiAiW0NMU10iLCAibWFza190b2tlbiI6ICJbTUFTS10ifQ==",
            "tokenizer_config.json":
                "base64:eyJkb19sb3dlcl9jYXNlIjogdHJ1ZSwgInVua190b2tlbiI6ICJbVU5LXSIsICJzZXBfdG9rZW4iOiAiW1NFUF0iLCAicGFkX3Rva2VuIjogIltQQURdIiwgImNsc190b2tlbiI6ICJbQ0xTXSIsICJtYXNrX3Rva2VuIjogIltNQVNLXSIsICJ0b2tlbml6ZXJfY2xhc3MiOiAiRXJuaWVUb2tlbml6ZXIifQ=="
        }
    }
}


def build_params_map(model_prefix='encoder', attention_num=12):
    """ build params map from paddle-paddle's ERNIE to transformer's BERT """
    weight_map = collections.OrderedDict({
        f'{model_prefix}.embeddings.word_embeddings.weight': "encoder.embeddings.word_embeddings.weight",
        f'{model_prefix}.embeddings.position_embeddings.weight': "encoder.embeddings.position_embeddings.weight",
        f'{model_prefix}.embeddings.token_type_embeddings.weight': "encoder.embeddings.token_type_embeddings.weight",
        f'{model_prefix}.embeddings.task_type_embeddings.weight': "encoder.embeddings.task_type_embeddings.weight",
        f'{model_prefix}.embeddings.layer_norm.weight': 'encoder.embeddings.LayerNorm.gamma',
        f'{model_prefix}.embeddings.layer_norm.bias': 'encoder.embeddings.LayerNorm.beta',
    })

    # add attention layers
    for i in range(attention_num):
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.q_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.q_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.k_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.k_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.v_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.v_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.out_proj.weight'] = f'encoder.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.out_proj.bias'] = f'encoder.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm1.weight'] = f'encoder.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm1.bias'] = f'encoder.encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear1.weight'] = f'encoder.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear1.bias'] = f'encoder.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear2.weight'] = f'encoder.encoder.layer.{i}.output.dense.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear2.bias'] = f'encoder.encoder.layer.{i}.output.dense.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm2.weight'] = f'encoder.encoder.layer.{i}.output.LayerNorm.gamma'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm2.bias'] = f'encoder.encoder.layer.{i}.output.LayerNorm.beta'

    # add pooler
    weight_map.update(
        {
            f'{model_prefix}.pooler.dense.weight': 'encoder.pooler.dense.weight',
            f'{model_prefix}.pooler.dense.bias': 'encoder.pooler.dense.bias',
            'linear_start.weight': 'linear_start.weight',
            'linear_start.bias': 'linear_start.bias',
            'linear_end.weight': 'linear_end.weight',
            'linear_end.bias': 'linear_end.bias',
        }
    )
    return weight_map


def extract_and_convert(input_dir, output_dir, verbose=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if verbose:
        logger.info('=' * 20 + 'save config file' + '=' * 20)

    config = json.load(
        open(os.path.join(input_dir, 'model_config.json'), 'rt', encoding='utf-8')
    )

    if 'init_args' in config:
        config = config['init_args'][0]
    config["architectures"] = ["UIE"]
    config['layer_norm_eps'] = 1e-12
    del config['init_class']

    if 'sent_type_vocab_size' in config:
        config['type_vocab_size'] = config['sent_type_vocab_size']
    config['intermediate_size'] = 4 * config['hidden_size']
    json.dump(
        config,
        open(os.path.join(output_dir, 'config.json'), 'wt', encoding='utf-8'),
        indent=4
    )
    if verbose:
        logger.info('=' * 20 + 'save vocab file' + '=' * 20)

    shutil.copy(os.path.join(input_dir, 'vocab.txt'), os.path.join(output_dir, 'vocab.txt'))

    special_tokens_map = json.load(
        open(os.path.join(input_dir, 'special_tokens_map.json'),
        'rt',
        encoding='utf-8')
    )
    json.dump(
        special_tokens_map,
        open(os.path.join(output_dir, 'special_tokens_map.json'), 'wt', encoding='utf-8'),
        indent=4
    )

    tokenizer_config = json.load(
        open(os.path.join(input_dir, 'tokenizer_config.json'), 'rt', encoding='utf-8')
    )

    if tokenizer_config['tokenizer_class'] == 'ErnieTokenizer':
        tokenizer_config['tokenizer_class'] = "BertTokenizer"
    json.dump(
        tokenizer_config,
        open(os.path.join(output_dir, 'tokenizer_config.json'), 'wt', encoding='utf-8'),
        indent=4
    )
    spm_file = os.path.join(input_dir, 'sentencepiece.bpe.model')

    if os.path.exists(spm_file):
        shutil.copy(spm_file, os.path.join(
            output_dir, 'sentencepiece.bpe.model'))

    if verbose:
        logger.info('=' * 20 + 'extract weights' + '=' * 20)

    state_dict = collections.OrderedDict()
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    weight_map.update(build_params_map('ernie', attention_num=config['num_hidden_layers']))

    paddle_paddle_params = pickle.load(open(os.path.join(input_dir, 'model_state.pdparams'), 'rb'))
    del paddle_paddle_params['StructuredToParameterName@@']

    for weight_name, weight_value in paddle_paddle_params.items():
        transposed = ''
        if 'weight' in weight_name and ('.encoder' in weight_name or 'pooler' in weight_name or 'linear' in weight_name):
            weight_value = weight_value.transpose()
            transposed = '.T'
        # Fix: embedding error
        if 'word_embeddings.weight' in weight_name:
            weight_value[0, :] = 0
        if weight_name not in weight_map:
            if verbose:
                logger.info(f"{'='*20} [SKIP] {weight_name} {'='*20}")
            continue
        state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        if verbose:
            logger.info(
                f"{weight_name}{transposed} -> {weight_map[weight_name]} {weight_value.shape}")
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


def check_model(input_model):
    if os.path.exists(input_model):
        return
    if input_model not in MODEL_MAP:
        raise ValueError('input_model not exists!')

    resource_file_urls = MODEL_MAP[input_model]['resource_file_urls']
    logger.info("Downloading resource files...")

    for key, val in resource_file_urls.items():
        file_path = os.path.join(input_model, key)
        if not os.path.exists(file_path):
            if val.startswith('base64:'):
                base64data = b64decode(val.replace('base64:', '').encode('utf-8'))
                with open(file_path, 'wb') as f:
                    f.write(base64data)
            else:
                download_path = get_path_from_url(val, input_model)
                if download_path != file_path:
                    shutil.move(download_path, file_path)


def get_path_from_url(url, root_dir, md5sum=None, check_exist=True):
    """Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.
    Args:
        url (str): download url
        root_dir (str): root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        md5sum (str): md5 sum of download package

    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """
    import os.path as osp
    import hashlib
    import requests
    import tarfile
    import zipfile

    def _map_path(url, root_dir):
        # parse path after download under root_dir
        fname = osp.split(url)[-1]
        fpath = fname
        return osp.join(root_dir, fpath)

    def _download(url, path, md5sum=None):
        """
        Download from url, save to path.
        url (str): download url
        path (str): download to given path
        """
        os.makedirs(path, exist_ok=True)

        fname = osp.split(url)[-1]
        fullname = osp.join(path, fname)
        retry_cnt = 0

        while not (osp.exists(fullname) and _md5check(fullname, md5sum)):
            if retry_cnt < 5:
                retry_cnt += 1
            else:
                raise RuntimeError("Download from {} failed. " "Retry limit reached".format(url))

            logger.info("Downloading {} from {}".format(fname, url))

            req = requests.get(url, stream=True)
            if req.status_code != 200:
                raise RuntimeError("Downloading from {} failed with code " "{}!".format(url, req.status_code))

            # For protecting download interupted, download to
            # tmp_fullname firstly, move tmp_fullname to fullname
            # after download finished
            tmp_fullname = fullname + "_tmp"
            total_size = req.headers.get("content-length")
            with open(tmp_fullname, "wb") as f:
                if total_size:
                    with tqdm(total=int(total_size), unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                        for chunk in req.iter_content(chunk_size=1024):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in req.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            shutil.move(tmp_fullname, fullname)

        return fullname

    def _md5check(fullname, md5sum=None):
        if md5sum is None:
            return True

        logger.info("File {} md5 checking...".format(fullname))
        md5 = hashlib.md5()
        with open(fullname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        calc_md5sum = md5.hexdigest()

        if calc_md5sum != md5sum:
            logger.info("File {} md5 check failed, {}(calc) != " "{}(base)".format(fullname, calc_md5sum, md5sum))
            return False
        return True

    def _decompress(fname):
        """
        Decompress for zip and tar file
        """
        logger.info("Decompressing {}...".format(fname))

        # For protecting decompressing interupted,
        # decompress to fpath_tmp directory firstly, if decompress
        # successed, move decompress files to fpath and delete
        # fpath_tmp and remove download compress file.

        if tarfile.is_tarfile(fname):
            uncompressed_path = _uncompress_file_tar(fname)
        elif zipfile.is_zipfile(fname):
            uncompressed_path = _uncompress_file_zip(fname)
        else:
            raise TypeError("Unsupport compress file type {}".format(fname))

        return uncompressed_path

    def _uncompress_file_zip(filepath):
        files = zipfile.ZipFile(filepath, "r")
        file_list = files.namelist()

        file_dir = os.path.dirname(filepath)

        if _is_a_single_file(file_list):
            rootpath = file_list[0]
            uncompressed_path = os.path.join(file_dir, rootpath)

            for item in file_list:
                files.extract(item, file_dir)

        elif _is_a_single_dir(file_list):
            rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)

            for item in file_list:
                files.extract(item, file_dir)

        else:
            rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            if not os.path.exists(uncompressed_path):
                os.makedirs(uncompressed_path)
            for item in file_list:
                files.extract(item, os.path.join(file_dir, rootpath))

        files.close()

        return uncompressed_path

    def _uncompress_file_tar(filepath, mode="r:*"):
        files = tarfile.open(filepath, mode)
        file_list = files.getnames()
        file_dir = os.path.dirname(filepath)

        if _is_a_single_file(file_list):
            rootpath = file_list[0]
            uncompressed_path = os.path.join(file_dir, rootpath)
            files.extractall(file_dir, files.getmembers())
        elif _is_a_single_dir(file_list):
            rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            files.extractall(file_dir, files.getmembers())
        else:
            rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
            uncompressed_path = os.path.join(file_dir, rootpath)
            if not os.path.exists(uncompressed_path):
                os.makedirs(uncompressed_path)

            files.extractall(os.path.join(file_dir, rootpath), files.getmembers())

        files.close()

        return uncompressed_path

    def _is_a_single_file(file_list):
        if len(file_list) == 1 and file_list[0].find(os.sep) < -1:
            return True
        return False

    def _is_a_single_dir(file_list):
        new_file_list = []
        for file_path in file_list:
            if "/" in file_path:
                file_path = file_path.replace("/", os.sep)
            elif "\\" in file_path:
                file_path = file_path.replace("\\", os.sep)
            new_file_list.append(file_path)

        file_name = new_file_list[0].split(os.sep)[0]
        for i in range(1, len(new_file_list)):
            if file_name != new_file_list[i].split(os.sep)[0]:
                return False
        return True

    # parse path after download to decompress under root_dir
    fullpath = _map_path(url, root_dir)

    if osp.exists(fullpath) and check_exist and _md5check(fullpath, md5sum):
        logger.info("Found {}".format(fullpath))
    else:
        fullpath = _download(url, root_dir, md5sum)

    if tarfile.is_tarfile(fullpath) or zipfile.is_zipfile(fullpath):
        fullpath = _decompress(fullpath)

    # model tokenizer config, [file-lock]
    return fullpath


def convert_uie_checkpoint(input_model_path, output_model_math):
    check_model(input_model_path)
    extract_and_convert(input_model_path, output_model_math, verbose=True)


if __name__ == "__main__":
    convert_uie_checkpoint("uie-base", "uie_base_pytorch")
