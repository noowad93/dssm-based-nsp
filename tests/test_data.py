import pytest
from pnlp.text import Vocab

from conve_rt.data import load_dataset


@pytest.fixture
def file_path_for_test(tmpdir) -> str:
    file_path = tmpdir.join("train.csv")
    file_path.write(
        "\n".join(
            [
                ",".join(["Context", "Utterance", "Label"]),
                ",".join(
                    [
                        "split xfonts* out of xfree86* . one upload for the rest of the life and that 's it __eou__ __eot__ split the sourc packag you mean ? __eou__ __eot__ yes . same binari packag . __eou__ __eot__ i would prefer to avoid it at this stage . this be someth that have go into xsf svn , i assum ? __eou__ __eot__",
                        "basic each xfree86 upload will not forc user to upgrad 100mb of font for noth __eou__ no someth i do in my spare time . __eou__",
                        "1",
                    ]
                ),
            ]
        )
    )
    return str(file_path)


@pytest.fixture
def token_vocab_for_test(tmpdir) -> Vocab:
    vocab_path_for_test = tmpdir.join("test_vocab.txt")
    vocab_path_for_test.write(
        "\n".join(
            [
                "<PAD>",
                "<UNK>",
                "__eou__",
                "__eot__",
                "i",
                "the",
                ",",
                "be",
                "?",
                "to",
                "it",
                "a",
                "you",
                "and",
                "do",
                "that",
                "have",
            ]
        )
    )
    return Vocab(str(vocab_path_for_test), "<UNK>")


def test_load_dataset(file_path_for_test):
    assert load_dataset(file_path_for_test) == [
        [
            "split xfonts* out of xfree86* . one upload for the rest of the life and that 's it __eou__ __eot__ split the sourc packag you mean ? __eou__ __eot__ yes . same binari packag . __eou__ __eot__ i would prefer to avoid it at this stage . this be someth that have go into xsf svn , i assum ? __eou__ __eot__",
            "basic each xfree86 upload will not forc user to upgrad 100mb of font for noth __eou__ no someth i do in my spare time . __eou__",
            "1",
        ]
    ]


# def test_len_of_dataset(file_path_for_test, token_vocab_for_test):

#     lm_dataset = DSSMTrainDataset(file_path_for_test, 24, token_vocab_for_test, NLPPipeline([MecabTokenizer()]))
#     assert len(lm_dataset) == 2


# def test_dataset_output_value(file_path_for_test, token_vocab_for_test):
#     lm_dataset = DSSMTrainDataset(file_path_for_test, 24, token_vocab_for_test, NLPPipeline([MecabTokenizer()]))

#     example_instance = lm_dataset.training_instances[0]
#     assert torch.all(
#         torch.eq(
#             example_instance[0], torch.tensor([6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#         )
#     )
#     assert torch.all(
#         torch.eq(
#             example_instance[1], torch.tensor([7, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#         )
#     )
