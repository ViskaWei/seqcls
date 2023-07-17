import os


DATA_PATH = os.path.join(os.environ["DATA"], "emotion")

LOGDIR = "/datascope/subaru/user/swei20/nlp/logs/"
OUTDIR = "/datascope/subaru/user/swei20/nlp/ckpt/"

def get_ds(tokenizer):
    import datasets
    ds = datasets.load_dataset(DATA_PATH)
    ds = ds.rename_column('label', 'labels')
    ds = ds.map(lambda x: tokenizer(x['text'], max_length=512, padding='max_length', return_tensors='pt', truncation=True), batched=True)
    ds.set_format('torch')
    return ds


TOKEN = {
    'bert0': 'bert-base-cased',
    "xlmr0": "xlm-roberta-base",
    "xlmr1": "xlm-roberta-large",
    # "xlmr2":  

}

def get_tokenizer(name):
    from transformers import AutoTokenizer
    if name not in TOKEN:
        raise ValueError("Unknown tokenizer name: {}".format(name))
    return AutoTokenizer.from_pretrained(TOKEN[name])

# from transformers import XLMRobertaXLConfig, XLMRobertaXLModel

# # Initializing a XLM_ROBERTA_XL bert-base-uncased style configuration
# configuration = XLMRobertaXLConfig()

# # Initializing a model (with random weights) from the bert-base-uncased style configuration
# model = XLMRobertaXLModel(configuration)

# # Accessing the model configuration
# configuration = model.config