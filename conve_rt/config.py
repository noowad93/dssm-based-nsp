from typing import NamedTuple


class TrainConfig(NamedTuple):
    #: epoch 도는 횟수
    epoch: int = 10
    #: 훈련 시의 batch size
    train_batch_size: int = 200
    #: validate 때 조금 더 빠르게 validate하기 위해서 훈련 시의 batch size보다 큰 값을 지정
    eval_batch_size: int = 512
    #: input 길이의 최대값.
    max_seq_len: int = 50
    #: learning rate
    learning_rate: float = 0.001
    #: bert fine tuning 레이어의 dropout 확률
    dropout_prob: float = 0.1
    # gpu 개수
    n_gpu: int = 4

    # word embedding 사이즈
    word_embed_size = 100
    # hidden unit 차원
    hidden_size = 256

    # glove
    glove_file_path: str = "/nas/home/noowad/convert/data/glove.6B.100d.txt"
    #: train data 파일 경로
    train_file_path: str = "/nas/home/noowad/convert/data/train.csv"
    #: eval data 파일 경로
    eval_file_path: str = "/nas/home/noowad/convert/data/valid.csv"
    #: vocab 파일이 저장된 경로
    vocab_file_path: str = "/nas/home/noowad/convert/data/vocabulary_30000.txt"
    #: 모델이 저장될 경로
    save_model_file_prefix: str = "./checkpoints/model"

    #: logging을 할 step 수
    train_log_interval: int = 100
    #: validation을 할 step 수
    val_log_interval: int = 1000
    # save할 step 수
    save_interval: int = 10000


class InferenceConfig(NamedTuple):
    #: 추론 시의 batch size
    infer_batch_size: int = 256
    #: bert fine tuning 레이어의 dropout 확률
    dropout_prob: float = 0.0
    # gpu 개수
    n_gpu_use: int = 4

    # max_len
    max_len = 24
    # word embedding 사이즈
    word_embed_size = 256
    # transformer 헤드 개수
    head_num = 4
    # transformer hidden unit 차원
    hidden_num = 256
    # transformer layer 개수
    layer_num = 2

    #: vocab 파일이 저장된 경로
    vocab_file_path: str = "/nas/models/pretrained_bert/vocab/mecab-50000.vocab"
    #: pretraining model 경로
    checkpoint_file_path: str = "/nas/home/noowad/convert/checkpoints/lm_model_processed_step_step_340000.pth"
