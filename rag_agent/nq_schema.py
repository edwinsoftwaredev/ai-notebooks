import pyarrow as pa

schema = pa.schema(
    [
        ("id", pa.large_string()),
        (
            "document",
            pa.struct(
                [
                    ("title", pa.string()),
                    (
                        "tokens",
                        pa.struct(
                            [
                                ("is_html", pa.list_(pa.bool_())),
                                ("token", pa.list_(pa.string())),
                            ]
                        ),
                    ),
                    ("url", pa.string()),
                ]
            ),
        ),
        (
            "question",
            pa.struct(
                [
                    ("tokens", pa.list_(pa.string())),
                ]
            ),
        ),
        (
            "annotations",
            pa.struct(
                [
                    (
                        "long_answer",
                        pa.list_(
                            pa.struct(
                                [
                                    ("end_token", pa.int64()),
                                    ("start_token", pa.int64()),
                                ]
                            )
                        ),
                    ),
                ]
            ),
        ),
    ]
)
