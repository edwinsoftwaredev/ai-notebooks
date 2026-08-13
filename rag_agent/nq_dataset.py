import dask.dataframe as ddf
from huggingface_hub import snapshot_download


def _download_dataset():
    import shutil

    from rag_agent import nq_schema

    def update_document(d):
        if "html" in d:
            del d["html"]

        if "tokens" in d:
            if "end_byte" in d["tokens"]:
                del d["tokens"]["end_byte"]

            if "start_byte" in d["tokens"]:
                del d["tokens"]["start_byte"]

        return d

    def update_annotations(d):
        if "id" in d:
            del d["id"]

        if "long_answer" in d:
            for dd in d["long_answer"]:
                if "candidate_index" in dd:
                    del dd["candidate_index"]

                if "start_byte" in dd:
                    del dd["start_byte"]

                if "end_byte" in dd:
                    del dd["end_byte"]

        if "short_answers" in d:
            del d["short_answers"]

        if "yes_no_answer" in d:
            del d["yes_no_answer"]

        return d

    def update_question(d):
        if "text" in d:
            del d["text"]

        return d

    def update_partitions(df):
        df = df.drop(columns=["long_answer_candidates"])
        df["question"] = df["question"].map(update_question)
        df["document"] = df["document"].map(update_document)
        df["annotations"] = df["annotations"].map(update_annotations)
        return df

    n_parquets = int(287 * 0.5)
    repo_id = "google-research-datasets/natural_questions"
    for i in range(0, n_parquets, 25):
        files = [
            f"default/train-{(i + j):05d}-of-00287.parquet"
            for j in range(25)
            if (i + j) <= n_parquets
        ]

        parquets = snapshot_download(
            repo_id=repo_id,
            allow_patterns=files,
            repo_type="dataset",
        )

        df = ddf.read_parquet(f"{parquets}/**/*.parquet")
        df = df.map_partitions(update_partitions)
        df.to_parquet(
            "/kaggle/working/pre_nq_dataset",
            engine="pyarrow",
            schema=nq_schema.schema,
            append=True,
            write_index=False,
        )

        shutil.rmtree(
            "/kaggle/working/hf_cache/hub/datasets--google-research-datasets--natural_questions",
            ignore_errors=True,
        )

    df = ddf.read_parquet("/kaggle/working/pre_nq_dataset")
    indexed_df = df.set_index("id", sorted=False)
    indexed_df.to_parquet(
        "/kaggle/working/nq_dataset",
        engine="pyarrow",
        schema=nq_schema.schema,
        write_index=True,
    )

    shutil.rmtree("/kaggle/working/pre_nq_dataset", ignore_errors=True)
